import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np
import copy


from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt, embedding_lookup, align_a2b
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.amodel import ContextNet
import open_clip




class FIDNet(ContextNet):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_activation,
                        ntokens_per_video,
                        spacing,
                        code_loss):

        super().__init__(transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_dim_feedforward=transformer_dim_feedforward,
                        transformer_nlayers_enc=transformer_nlayers_enc,
                        transformer_nlayers_dec=1,
                        transformer_activation=transformer_activation,
                        code_loss=code_loss,)
                        
        self.ntokens_per_video=ntokens_per_video
        self.spacing=spacing

        #self.mean_mano_palm_joints={'left': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT'))),
        #        'right': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT')))}
        
        self.num_joints=42
        self.placeholder_joints=torch.nn.Parameter(torch.randn(1,42,3))

        
        self.phase_pe=PositionalEncoding(d_model=transformer_d_model)

        self.dim_hand_feature= 3*self.num_joints+9*3
        self.hand_scaling_factor=10
        

        self.pose3d_to_trsfm_in=nn.Linear(self.num_joints*3,transformer_d_model//2)
        self.globalRt_to_trsfm_in=nn.Linear(9*2,transformer_d_model//4)
        self.localL2R_to_trsfm_in=nn.Linear(9,transformer_d_model//4)

        self.temperature=0.07
        
    
    def get_gt_inputs_feature(self,batch,num_prefix_frames_to_remove,verbose=False):
        return_batch={}
        len_seq=self.ntokens_per_video+num_prefix_frames_to_remove
        
        #from motion batch
        return_batch['batch_action_name']=batch['action_name'][num_prefix_frames_to_remove::len_seq]#######
        
        for key in ["valid_frame"]:
            return_batch["batch_seq_"+key]= batch[key].view(-1,len_seq)
        batch_size=return_batch['batch_seq_valid_frame'].shape[0]
            
        
        batch_flatten_hand={}
        for key in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                    'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:
            batch_flatten_hand[key]=batch[key]
            
        return_batch['batch_action_idx']=torch.zeros(batch_size,dtype=torch.int64)
        for vid, vname in enumerate(return_batch['batch_action_name']):
            return_batch['batch_action_idx'][vid]=self.action_name2idx[vname]
                
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=len_seq,
                                        spacing=self.spacing,
                                        base_frame_id=0,
                                        factor_scaling=self.hand_scaling_factor,
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=verbose)

        to_return_gt=flatten_comps["gt"][:,:self.dim_hand_feature]
        return_batch["batch_seq_hand_comp"]=to_return_gt.view((batch_size,len_seq)+to_return_gt.shape[1:])

        if verbose:
            print("****Check return_batch")
            for k,v in return_batch.items():
                try:
                    print(k,v.shape)
                except:
                    print(k,len(v))
            print("End get_gt_inputs_features")
            exit(0)
        return return_batch


    
    def get_est_inputs_feature(self, batch, verbose=False):
        return_batch={"batch_action_name":batch["batch_action_name_obsv"]}
        batch_size, len_seq=batch['batch_seq_valid_frame'].shape[:2]
        
        palm_joints=[0,5,9,13,17] 
        root_idx=0
        
        batch_flatten_hand={}
        for tag in ["left","right"]:
            batch_flatten_hand["cam_joints3d_"+tag]=torch.flatten(batch["batch_seq_cam_joints3d_"+tag],0,1)
            batch_flatten_hand["local_joints3d_"+tag]=torch.flatten(batch["batch_seq_local_joints3d_"+tag],0,1)
            
            batch_flatten_hand["R_cam2local_"+tag],batch_flatten_hand["t_cam2local_"+tag]=align_a2b(batch_flatten_hand["cam_joints3d_"+tag],batch_flatten_hand["local_joints3d_"+tag],root_idx=root_idx)

            batch_flatten_hand["valid_joints_"+tag]=torch.ones_like(batch_flatten_hand["cam_joints3d_"+tag])
            batch_flatten_hand["valid_joints_"+tag][:,1]=0

            
            batch_flatten_hand["hand_size_"+tag]=torch.mean(torch.norm(batch_flatten_hand["cam_joints3d_"+tag][:,palm_joints[1:]]-batch_flatten_hand["cam_joints3d_"+tag][:, root_idx:root_idx+1],p=2,dim=-1),dim=1)

                
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=len_seq,
                                        spacing=self.spacing,
                                        base_frame_id=0,
                                        factor_scaling=self.hand_scaling_factor,
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=verbose)

        to_return_est=flatten_comps["gt"][:,:self.dim_hand_feature]
        return_batch["batch_seq_hand_comp"]=to_return_est.view((batch_size,len_seq)+to_return_est.shape[1:])

        if verbose:
            print("****Check return_batch")
            for k,v in return_batch.items():
                try:
                    print(k,v.shape)
                except:
                    print(k,len(v))
            print("End get_est_inputs_features")
            exit(0)
        return return_batch



    def forward(self, batch, num_prefix_frames_to_remove,  batch_is_gt, compute_loss, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results,losses = {},{}
        
        #reformat batch
        if batch_is_gt:
            batch0=self.get_gt_inputs_feature(batch, num_prefix_frames_to_remove, verbose=verbose) 
            batch.update(batch0)
        else:
            batch0=self.get_est_inputs_feature(batch, verbose=verbose)
            batch.update(batch0)
        
        #lets start
        batch_seq_hand_comp_obsv = batch["batch_seq_hand_comp"]
        batch_size=batch_seq_hand_comp_obsv.shape[0]

        batch_seq_hand_obsv_mask=~(batch["batch_seq_valid_frame"].cuda().bool())
        #batch_seq_phase_pe=self.phase_pe(batch["clip_since_action_start"].cuda())

        if verbose:
            print(batch_seq_hand_comp_obsv[0,0,self.num_joints*3:])
            print("batch_seq_hand_comp_obsv,batch_seq_hand_obsv_mask",batch_seq_hand_comp_obsv.shape,batch_seq_hand_obsv_mask.shape)#[bs,len_seq,len_comp],[bs,len_seq]
            #print("batch[clip_since_action_start]",batch["clip_since_action_start"].shape)#[bs,len_seq]


        #A-Enc output for action classification
        if num_prefix_frames_to_remove>0:
            batch_seq_hand_comp_obsv=batch_seq_hand_comp_obsv[:,num_prefix_frames_to_remove:]
            batch_seq_hand_obsv_mask=batch_seq_hand_obsv_mask[:,num_prefix_frames_to_remove:]
            #batch_seq_phase_pe=batch_seq_phase_pe[:,num_prefix_frames_to_remove:]

        batch_aenc_action_obsv_out=self.feed_encoder(batch_seq_enc_in_comp=batch_seq_hand_comp_obsv,
                                                    batch_seq_enc_mask_tokens=batch_seq_hand_obsv_mask,
                                                    verbose=verbose)


        #FC(CLIP)
        batch_aname_obsv_gt=batch["batch_action_name"]
        results["batch_action_name_obsv"]=batch_aname_obsv_gt


        if not compute_loss:
            results["batch_enc_out_global_feature"]=batch_aenc_action_obsv_out
            return total_loss,results,losses

        batch_atokens_obsv_gt=open_clip.tokenizer.tokenize(batch_aname_obsv_gt).cuda()
        with torch.no_grad():            
            batch_atokens_obsv_gt=self.model_bert.encode_text(batch_atokens_obsv_gt).float()
        batch_atokens_obsv_gt/=batch_atokens_obsv_gt.norm(dim=-1,keepdim=True)
        if verbose:
            print("***GT Action: batch_aname_obsv_gt",len(batch_aname_obsv_gt),batch_aname_obsv_gt[-10:])
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape,torch.abs(batch_atokens_obsv_gt.norm(dim=-1)-1.).max())#[bs,512]

        batch_atokens_obsv_gt=self.bert_to_latent(batch_atokens_obsv_gt.detach().clone())
        action_embedding=torch.transpose(self.bert_to_latent(torch.transpose(self.action_embedding,0,1).detach()),0,1)
        #action_embedding=self.action_embedding
        if verbose:
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape)#[bs,512]
            print("self.action_embedding/action_embedding",self.action_embedding.shape,action_embedding.shape)#[512,nembeddings],[512,nembeddings]

        #Contrastive with Cosine Distance        
        results["batch_action_idx_obsv_gt"]=batch["batch_action_idx"].cuda()
        batch_action_similarity_obsv,results["batch_action_idx_obsv_out"]=embedding_lookup(query=batch_aenc_action_obsv_out,embedding=action_embedding, verbose=verbose)#
        batch_action_similarity_obsv=batch_action_similarity_obsv/self.temperature
        results["batch_action_prob_distrib"]=nn.functional.softmax(batch_action_similarity_obsv,dim=1)
        losses["action_contrast_loss"] = torch_f.cross_entropy(batch_action_similarity_obsv,results["batch_action_idx_obsv_gt"],reduction='mean')
        losses["action_dist_loss"]=self.code_loss(batch_aenc_action_obsv_out,batch_atokens_obsv_gt)

        total_loss+=losses["action_contrast_loss"]+losses["action_dist_loss"]
        if verbose:
            print("batch_aenc_action_obsv_out/batch_atokens_obsv_gt",batch_aenc_action_obsv_out.shape,batch_atokens_obsv_gt.shape)#[bs,512]
            print("batch_action_idx_obsv_gt",results["batch_action_idx_obsv_gt"].shape)#[bs]
            print("batch_action_similarity_obsv/batch_action_prob_distrib",batch_action_similarity_obsv.shape,results["batch_action_prob_distrib"].shape)#[bs,tax_size]x2
            prob_norm=torch.sum(results["batch_action_prob_distrib"], axis=-1)
            print("check sum", prob_norm.shape, torch.abs(prob_norm-1).max())#[bs]

            print('if gt input, check action_idx gt vs out',torch.abs(results["batch_action_idx_obsv_gt"]-results["batch_action_idx_obsv_out"]).max())
            print("==================================")
            exit(0)
            
        return total_loss,results,losses

    def feed_encoder(self,batch_seq_enc_in_comp, batch_seq_enc_mask_tokens, verbose): 
        batch_seq_enc_in_tokens=torch.cat([self.pose3d_to_trsfm_in(batch_seq_enc_in_comp[:,:,:self.num_joints*3]), \
                                self.globalRt_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3:self.num_joints*3+18]), \
                                self.localL2R_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3+18:self.num_joints*3+27])],dim=2)
                                
        if verbose:
            print("****Start Enc****")
            print("batch_seq_enc_in_tokens/batch_seq_phase_pe",batch_seq_enc_in_tokens.shape)#,batch_seq_phase_pe.shape)#[bs,len_seq,512]x2
        #batch_seq_enc_in_tokens+=batch_seq_phase_pe
        
        batch_size=batch_seq_enc_in_tokens.shape[0]        
        batch_enc_in_mu=repeat(self.mid_token_mu_enc_in,'() n d -> b n d',b=batch_size)
        batch_enc_mask_global=batch_seq_enc_mask_tokens[:,0:1].detach().clone()
        assert not batch_enc_mask_global.any()
        
        
        batch_seq_enc_in=torch.cat((batch_enc_in_mu,batch_seq_enc_in_tokens),dim=1)
        batch_seq_enc_pe=self.transformer_pe(batch_seq_enc_in)
        batch_seq_enc_mask=torch.cat((batch_enc_mask_global,batch_seq_enc_mask_tokens),dim=1)
        
        if verbose:
            print('batch_enc_in_mu/batch_enc_mask_global',batch_enc_in_mu.shape,batch_enc_mask_global.shape)#[bs,1,512],[bs,1]
            print('batch_seq_enc_in/batch_seq_enc_pe/batch_seq_enc_mask',batch_seq_enc_in.shape,batch_seq_enc_pe.shape,batch_seq_enc_mask.shape)#[bs,1+len_tokens,512]x2,[bs,1+len_tokens]
            print(batch_seq_enc_mask[:2])
        
        
        batch_seq_enc_out, _ =self.encoder(src=batch_seq_enc_in, src_pos=batch_seq_enc_pe, key_padding_mask=batch_seq_enc_mask,verbose=False)      
        
        batch_enc_out_mu =batch_seq_enc_out[:,0]
        
        if verbose:
            print("batch_seq_enc_out",batch_seq_enc_out.shape)#[bs,1+len_tokens,512]

        return batch_enc_out_mu
    