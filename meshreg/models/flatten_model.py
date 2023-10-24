import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np

import open_clip
import copy

from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt, compute_bert_embedding_for_taxonomy, compute_berts_for_strs, embedding_lookup
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.vae_block import VAE
from meshreg.models.pmodel import MotionNet


from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_rotation_6d,axis_angle_to_matrix

class FlattenNet(MotionNet):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_nlayers_dec,
                        transformer_activation,

                        ntokens_per_clip,
                        spacing,
                        lambda_clustering=None,
                        lambda_hand=None,
                        lambda_action=None,
                        pose_loss='l1',
                        code_loss='l1',):

        super().__init__(transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_dim_feedforward=transformer_dim_feedforward,
                        transformer_nlayers_enc=transformer_nlayers_enc,
                        transformer_nlayers_dec=transformer_nlayers_dec,
                        transformer_activation=transformer_activation,
                        ntokens_per_clip=ntokens_per_clip,
                        spacing=spacing,
                        lambda_clustering=lambda_clustering,
                        lambda_hand=lambda_hand,
                        pose_loss=pose_loss,
                        code_loss=code_loss)

        
        self.model_bert, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="./assets/")
        self.bert_to_latent= nn.Linear(512,transformer_d_model)

        self.phase_pe=PositionalEncoding(d_model=transformer_d_model)
 
        self.lambda_action=lambda_action
        self.lambda_kl=lambda_clustering
        self.lambda_hand=lambda_hand 
        self.temperature=0.07

        loss_str2func_=loss_str2func()
        self.code_loss=loss_str2func_[code_loss]

        
    def get_gt_inputs_feature(self, batch_flatten,is_train,verbose=False):
        batch0=super().get_gt_inputs_feature(batch_flatten,is_train,verbose=False)
        
        batch0["batch_seq_since_action_start"]=batch_flatten["frame_since_action_start"].view(-1,self.ntokens_op)
        batch_obsv_obj_name=[batch_flatten["obj_name"][idx] for idx in range(0,len(batch_flatten["obj_name"]),self.ntokens_op)]

        clip_otokens_obsv=compute_berts_for_strs(self.model_bert,batch_obsv_obj_name,verbose)
        batch0["batch_seq_obj_feature_obsv"]=torch.unsqueeze(clip_otokens_obsv,1).repeat(1,self.ntokens_obsv,1)

        batch0['batch_action_idx_obsv']=torch.zeros(len(batch0['batch_action_name_obsv']),dtype=torch.int64)
        for vid, vname in enumerate(batch0['batch_action_name_obsv']):
            batch0['batch_action_idx_obsv'][vid]=self.action_name2idx[vname]
        return batch0
        

    def compute_bert_embedding_for_taxonomy(self, datasets, is_action,verbose=False):
        name_to_idx,tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action, verbose=verbose)
        if is_action:
            self.action_name2idx=copy.deepcopy(name_to_idx)
            self.action_embedding=tokens.detach().clone()
        else:
            self.object_name2idx=copy.deepcopy(name_to_idx)
            self.object_embedding=tokens.detach().clone()

    def forward(self, batch_flatten, is_train, to_reparameterize=False, gt_action_for_dec=False, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results, losses = {}, {}
        batch0=self.get_gt_inputs_feature(batch_flatten,is_train,verbose=False)

        #lets start
        batch_seq_hand_comp_obsv = batch0["batch_seq_hand_comp_obsv"] 
        batch_seq_obj_feature_obsv=batch0["batch_seq_obj_feature_obsv"]
        batch_size=batch_seq_hand_comp_obsv.shape[0]
        
        #First make sure consistent valid_frame and valid_features
        batch_seq_valid_frame=batch0["valid_frame"].cuda().reshape(batch_size,self.ntokens_op)
        
        if verbose:
            print("batch_seq_valid_features",batch0["batch_seq_valid_features"].shape)#[bs,len_op,153]
            
        #The original version compute every frames loss
        #batch0["batch_seq_valid_features"]=torch.mul(batch0["batch_seq_valid_features"],torch.unsqueeze(batch_seq_valid_frame,-1))
        if verbose:
            print("batch_seq_valid_features",batch0["batch_seq_valid_features"].shape,torch.unsqueeze(batch_seq_valid_frame,-1).shape)#[bs,len_op,153],[bs,len_op,1]

        #Pass encoder
        batch_seq_obsv_mask=~(batch_seq_valid_frame.bool()[:,:self.ntokens_obsv])
        batch_seq_phase_embed=self.phase_pe.forward_idx(batch0["batch_seq_since_action_start"].cuda())

        batch_seq_enc_out=self.feed_encoder(batch_seq_hand_comp_obsv,batch_seq_obj_feature_obsv,batch_seq_phase_embed[:,:self.ntokens_obsv].contiguous(),batch_seq_obsv_mask,verbose=verbose)
        batch_mid_mu_enc_out, batch_mid_logvar_enc_out,batch_seq_hand_feature_enc_out, batch_seq_hand_comp_enc_out=batch_seq_enc_out
        
        #obsv hands
        batch_seq_valid_frame_obsv=batch_seq_valid_frame[:,:self.ntokens_obsv]
        batch_seq_hand_size_left=torch.mul(torch.flatten(batch0['hand_size_left'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_left_size=torch.sum(batch_seq_hand_size_left,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)

        batch_seq_hand_size_right=torch.mul(torch.flatten(batch0['hand_size_right'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_right_size=torch.sum(batch_seq_hand_size_right,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)
        
        trans_info_obsv={}
        total_loss_hand,results_hand_obsv,losses_hand=self.compute_hand_loss(batch_seq_comp_gt=batch0["batch_seq_hand_comp_gt"][:,:self.ntokens_obsv], 
                                                batch_seq_comp_out=batch_seq_hand_comp_enc_out,
                                                compute_local2base=False,#True
                                                batch_seq_local2base_gt=None,
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"][:,:self.ntokens_obsv], 
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info_obsv,
                                                normalize_size_from_comp=False,verbose=verbose)
                                                

        total_loss+=total_loss_hand
        for k,v in losses_hand.items():
            losses[k+"_observ"]=v
            
        if self.lambda_clustering is not None and self.lambda_clustering>1e-10:
            losses["kld_loss"]=self.compute_kl_loss(mu1=batch_mid_mu_enc_out[:,0],logvar1=batch_mid_logvar_enc_out[:,0],verbose=verbose)
            total_loss+=self.lambda_clustering*losses["kld_loss"]
        
        if is_train:
            batch_seq_dec_mem=self.reparameterize(mu=batch_mid_mu_enc_out[:,0:1],logvar=batch_mid_logvar_enc_out[:,0:1])        
        elif to_reparameterize:
            batch_seq_dec_mem=self.reparameterize(mu=batch_mid_mu_enc_out[:,0:1],logvar=batch_mid_logvar_enc_out[:,0:1],factor=10.)#+=torch.randn_like(batch_clip_adec_mem)
        else:
            batch_seq_dec_mem=batch_mid_mu_enc_out[:,0:1]
        results["batch_seq_dec_mem"]=batch_seq_dec_mem.detach().clone()

        #Lets sync with action
        batch_aname_obsv_gt=batch0["batch_action_name_obsv"]
        results["batch_action_name_obsv"]=batch_aname_obsv_gt        
        batch_atokens_obsv_gt=compute_berts_for_strs(self.model_bert,batch_aname_obsv_gt,verbose)
        

        if verbose:
            print("****A-Enc GT Action: batch_aname_obsv_gt",len(batch_aname_obsv_gt),batch_aname_obsv_gt[-10:])
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape,torch.abs(batch_atokens_obsv_gt.norm(dim=-1)-1.).max())#[bs,512]

        batch_atokens_obsv_gt=self.bert_to_latent(batch_atokens_obsv_gt.detach().clone())
        action_embedding=torch.transpose(self.bert_to_latent(torch.transpose(self.action_embedding,0,1).detach()),0,1)
        if verbose:
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape)#[bs,512]
            print("self.action_embedding/action_embedding",self.action_embedding.shape,action_embedding.shape)#[512,nembeddings],[512,nembeddings]

        #action loss
        batch_enc_action_obsv_out=torch.squeeze(batch_seq_dec_mem,1)
        losses["action_dist_loss"]=self.code_loss(batch_enc_action_obsv_out,batch_atokens_obsv_gt)
        
        #Contrastive with Cosine Distance
        results["batch_action_idx_obsv_gt"]=batch0["batch_action_idx_obsv"].cuda()
        batch_action_similarity_obsv,results["batch_action_idx_obsv_out"]=embedding_lookup(query=batch_enc_action_obsv_out,embedding=action_embedding, verbose=verbose)#
        batch_action_similarity_obsv=batch_action_similarity_obsv/self.temperature
        results["batch_action_prob_distrib"]=nn.functional.softmax(batch_action_similarity_obsv,dim=1)
        losses["action_contrast_loss"] = torch_f.cross_entropy(batch_action_similarity_obsv,results["batch_action_idx_obsv_gt"],reduction='mean')
        
        if self.lambda_action is not None and self.lambda_action>1e-6:
            total_loss+=self.lambda_action*(losses["action_dist_loss"]+losses["action_contrast_loss"])


        
        batch_seq_dec_mem_mask=torch.zeros_like(batch_seq_obsv_mask[:,:1]).bool()
        rand_to_concat_hand_enc_out=torch.rand(1).cuda()
        if not is_train or rand_to_concat_hand_enc_out<0.5:
            batch_seq_dec_mem=torch.cat((batch_seq_dec_mem,batch_seq_hand_feature_enc_out),dim=1)            
            batch_seq_dec_mem_mask=torch.cat((batch_seq_dec_mem_mask,batch_seq_obsv_mask),dim=1)
        
        
        #Pass decoder and predict hands
        batch_seq_tgt_mask=~(batch_seq_valid_frame.bool()[:,self.ntokens_obsv:]) 
        results["batch_seq_valid_frame_pred"]=batch_seq_valid_frame[:,self.ntokens_obsv:]
        batch_seq_tgt_mask[:,0]=False        
        batch_seq_hand_feature_dec_out, batch_seq_hand_comp_dec_out=self.feed_decoder(batch_seq_dec_mem,
                                                        batch_seq_dec_mem_mask,
                                                        batch_seq_dec_query=batch_seq_phase_embed[:,self.ntokens_obsv:].contiguous(),
                                                        batch_seq_dec_tgt_key_padding_mask=batch_seq_tgt_mask,
                                                        verbose=verbose)
        if verbose:
            print('batch_seq_hand_feature/comp_dec_out',batch_seq_hand_feature_dec_out.shape,batch_seq_hand_comp_dec_out.shape) #[bs,len_p,512][bs,len_p,144]

        #hand loss
        trans_info_pred={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info_pred[k]=batch0[k]          

        batch_seq_pred_valid_features=torch.mul(batch0["batch_seq_valid_features"],torch.unsqueeze(batch_seq_valid_frame,-1))[:,self.ntokens_obsv:]
        total_loss_hand,results_hand,losses_hand=self.compute_hand_loss(batch_seq_comp_gt=batch0["batch_seq_hand_comp_gt"][:,self.ntokens_obsv:], 
                                                batch_seq_comp_out=batch_seq_hand_comp_dec_out,
                                                compute_local2base=True,
                                                batch_seq_local2base_gt=batch0['flatten_local2base_gt'].view(batch_size,self.ntokens_op,-1)[:,self.ntokens_obsv:].clone(), 
                                                batch_seq_valid_features=batch_seq_pred_valid_features,
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info_pred,
                                                normalize_size_from_comp=not is_train, verbose=verbose)
        total_loss+=total_loss_hand
        for k,v in losses_hand.items():
            losses[k+"_predict"]=v
            
        for k in ["base","local","cam"]:
            joints3d_out=results_hand[f"batch_seq_joints3d_in_{k}_out"]/self.hand_scaling_factor
            results[f"batch_seq_joints3d_in_{k}_pred_out"]=joints3d_out
            
            joints3d_gt=batch0[f"flatten_joints3d_in_{k}_gt"].view(batch_size,self.ntokens_op,self.num_joints,3)/self.hand_scaling_factor
            results[f"batch_seq_joints3d_in_{k}_gt"]=joints3d_gt
            results[f"batch_seq_joints3d_in_{k}_obsv_gt"]=joints3d_gt[:,:self.ntokens_obsv]
            results[f"batch_seq_joints3d_in_{k}_pred_gt"]=joints3d_gt[:,self.ntokens_obsv:]
            if verbose:
                print(k,"- joints out-GT#",torch.abs(joints3d_out-joints3d_gt[:,self.ntokens_obsv:,]).max()) 
       
        if verbose:
            for k,v in losses.items():
                print(k,v)
            
            for k,v in results.items():
                try:
                    print(k,v.shape) 
                except:
                    print(k,v)

            for k in ["cam","local","base"]:
                for kk in ["pred","obsv"]:
                    if kk=="obsv":
                        continue
                    print(k,kk,torch.abs(results[f"batch_seq_joints3d_in_{k}_{kk}_gt"]-results[f"batch_seq_joints3d_in_{k}_{kk}_out"]).max())

            from meshreg.netscripts.utils import sample_vis_l2r,sample_vis_ncam_cam
            '''
            from meshreg.datasets import ass101utils
            flatten_R_l2r=rotation_6d_to_matrix(torch.flatten(batch0["batch_seq_hand_comp_gt"][:,:,126+18:126+18+6],0,1))
            flatten_t_l2r=torch.flatten(batch0["batch_seq_hand_comp_gt"][:,:,126+18+6:126+18+9],0,1).view(-1,1,3)
            flatten_j3d_left=torch.flatten(results["batch_seq_joints3d_in_local_gt"],0,1)[:,:21]
            #flatten_j3d_left[:,:,1]=-flatten_j3d_left[:,:,1]
            
            print(flatten_R_l2r.shape,flatten_t_l2r.shape,flatten_j3d_left.shape)
            flatten_j3d_left_in_right=torch.bmm(flatten_j3d_left,flatten_R_l2r.double())+flatten_t_l2r.double()            
            flatten_j3d_in_right=torch.cat([flatten_j3d_left_in_right,torch.flatten(results["batch_seq_joints3d_in_local_gt"],0,1)[:,21:]],dim=1)
            
            R_local2cam,t_local2cam=get_inverse_Rt(batch_flatten["R_cam2local_right"],batch_flatten["t_cam2local_right"])
            flatten_j3d_in_right=torch.bmm(flatten_j3d_in_right,R_local2cam.cuda())+t_local2cam.cuda()
            batch_seq_j3d_in_right=flatten_j3d_in_right.view(batch_size,self.ntokens_op,42,3)

            print(torch.abs(batch_seq_j3d_in_right[:,:,21:]-results["batch_seq_joints3d_in_cam_gt"][:,:,21:]).max())
            print(torch.abs(batch_seq_j3d_in_right-results["batch_seq_joints3d_in_cam_gt"]).max())
            '''

            links=[(0, 2, 3, 4),(0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]
            tag="h2o_cam4"
            cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
            for sample_id in range(0,7): 
                '''
                sample_vis_ncam_cam(batch_seq_cam_joints2d=torch.cat([batch_flatten["joints2d_left"],batch_flatten["joints2d_right"]],dim=1).view(batch_size,self.ntokens_op,42,2),
                    batch_seq_ncam_joints2d=torch.cat([batch_flatten["aug_joints2d_left"],batch_flatten["aug_joints2d_right"]],dim=1).view(batch_size,self.ntokens_op,42,2),
                    batch_seq_cam_joints3d=torch.cat([batch_flatten["cam_joints3d_left"],batch_flatten["cam_joints3d_right"]],dim=1).view(batch_size,self.ntokens_op,42,3), 
                    batch_seq_ncam_joints3d=torch.cat([batch_flatten["ncam_joints3d_left"],batch_flatten["ncam_joints3d_right"]],dim=1).view(batch_size,self.ntokens_op,42,3),
                    joint_links=links,  
                    flatten_imgs=batch_flatten["image_vis"],
                    sample_id=sample_id,
                    prefix_cache_img=f"./vis_v3/imgs_{tag}/", path_video="./vis_v3/{:s}_{:02d}.avi".format(tag,sample_id))#{tag_out}
                '''
                print(torch.sum(batch0["valid_frame"].cuda().view(batch_size,self.ntokens_op),dim=-1))
                sample_vis_l2r(batch_seq_gt_cam=results["batch_seq_joints3d_in_cam_gt"], 
                                batch_seq_gt_local=results["batch_seq_joints3d_in_local_gt"],##batch_seq_j3d_in_right,
                                joint_links=links,  
                                flatten_imgs=batch_flatten["image_vis"],
                                sample_id=sample_id,
                                cam_info=cam_info,
                                prefix_cache_img=f"./vis_v3/imgs_{tag}/", path_video="./vis_v3/{:s}_{:02d}.avi".format(tag,sample_id))#{tag_out}
            exit(0)
        return total_loss,results,losses

    

    
    def feed_encoder(self,batch_seq_enc_in_comp,batch_seq_enc_obj_feature,batch_seq_enc_phase_embed, batch_seq_enc_mask_tokens, batch_seq_enc_in_obj=None, batch_seq_enc_in_img=None, verbose=False):
        if verbose:
            print("****Start P-Enc****")
        batch_seq_enc_hand=torch.cat([self.pose3d_to_trsfm_in(batch_seq_enc_in_comp[:,:,:self.num_joints*3]), \
                                self.globalRt_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3:self.num_joints*3+18]), \
                                self.localL2R_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3+18:self.num_joints*3+27])],dim=2)
        
        #concat hand-object
        batch_seq_enc_in_tokens=torch.cat([torch.unsqueeze(batch_seq_enc_hand,dim=2),torch.unsqueeze(batch_seq_enc_obj_feature,dim=2)],dim=2)
        batch_seq_enc_phase_embed=torch.unsqueeze(batch_seq_enc_phase_embed,dim=2)
        batch_seq_enc_in_tokens+=batch_seq_enc_phase_embed
        batch_seq_enc_mask_tokens_ho=torch.cat([torch.unsqueeze(batch_seq_enc_mask_tokens,dim=2),torch.unsqueeze(batch_seq_enc_mask_tokens,dim=2)],dim=2)
        if verbose:
            print("batch_seq_enc_in_tokens",batch_seq_enc_in_tokens.shape)#[bs,len_o,2,256]
            print("batch_seq_phase_embed",batch_seq_enc_phase_embed.shape)#[bs,len_o,1,256]
            print("batch_seq_enc_mask_tokens_ho",batch_seq_enc_mask_tokens_ho.shape)#[bs,len_o,2]
            
        batch_seq_enc_in_tokens=torch.flatten(batch_seq_enc_in_tokens,1,2)
        batch_seq_enc_mask_tokens=torch.flatten(batch_seq_enc_mask_tokens_ho,1,2)
             
        batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_tokens=super().feed_encoder_to_super(batch_seq_enc_in_tokens,batch_seq_enc_mask_tokens,verbose)
        
        batch_seq_enc_out_hand_tokens=batch_seq_enc_out_tokens[:,0::2]
        batch_seq_enc_out_comp=self.token_out_to_pose(batch_seq_enc_out_hand_tokens)
        if verbose:
            print("batch_seq_enc_out_hand_tokens/batch_seq_enc_out_comp",batch_seq_enc_out_hand_tokens.shape,batch_seq_enc_out_comp.shape)#[bs,len_p,512],[bs,len_p,144]
            print("****End P-Enc****")

        return batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_hand_tokens,batch_seq_enc_out_comp
    