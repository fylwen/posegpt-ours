import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np
import copy


from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding


from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt
from meshreg.models.utils import compute_bert_embedding_for_taxonomy,compute_berts_for_strs, embedding_lookup
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.vae_block import VAE

import open_clip

class ContextNet(VAE):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_nlayers_dec,
                        transformer_activation,

                        model_bert=None,
                        lambda_action=None,
                        lambda_mid_sync=None,
                        lambda_hand=None,
                        lambda_kl=None,
                        noise_std=1.,
                        ntokens_per_clip=15,
                        ntokens_per_video=16,
                        
                        pose_loss='l1',
                        code_loss='l1',
                        online_midpe=False,
                        allow_grad_to_pblock=False,):

        super().__init__(transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_dim_feedforward=transformer_dim_feedforward,
                        transformer_nlayers_enc=transformer_nlayers_enc,
                        transformer_nlayers_dec=transformer_nlayers_dec,
                        transformer_activation=transformer_activation,
                        num_mids=1)
                        
        self.ntokens_per_clip=ntokens_per_clip
        self.ntokens_obsv=ntokens_per_video
        self.ntokens_pred=ntokens_per_video

        if model_bert is None:
            self.model_bert, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="./assets/")
        else:
            self.model_bert=model_bert
        self.model_bert.eval()

        self.bert_to_latent= nn.Linear(512,transformer_d_model)

        #todo update with model_pblock
        self.model_pblock=None
        self.clip_base_frame_id=self.ntokens_per_clip-1

        self.phase_pe=PositionalEncoding(d_model=transformer_d_model)
 
        self.lambda_action=lambda_action
        self.lambda_kl=lambda_kl

        self.lambda_mid_sync=lambda_mid_sync
        self.lambda_hand=lambda_hand

        loss_str2func_=loss_str2func()
        self.pose_loss=loss_str2func_[pose_loss]
        self.code_loss=loss_str2func_[code_loss]
        self.temperature=0.07

        self.online_midpe=online_midpe
        self.noise_factor_midpe=noise_std
        self.allow_grad_to_pblock=allow_grad_to_pblock


    def assign_pblock(self,pblock):
        self.model_pblock=pblock
        self.model_pblock.eval()

        self.placeholder_joints=self.model_pblock.placeholder_joints
        self.hand_scaling_factor=self.model_pblock.hand_scaling_factor
        self.num_joints=self.model_pblock.num_joints
        self.mean_mano_palm_joints=self.model_pblock.mean_mano_palm_joints

        if not self.allow_grad_to_pblock:
            for param in self.model_pblock.parameters():
                param.requires_grad = False
        
    def compute_bert_embedding_for_taxonomy(self, datasets, is_action,verbose=False):
        name_to_idx,tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action, verbose=verbose)
        if is_action:
            self.action_name2idx=copy.deepcopy(name_to_idx)
            self.action_embedding=tokens.detach().clone()
        else:
            self.object_name2idx=copy.deepcopy(name_to_idx)
            self.object_embedding=tokens.detach().clone()


    def get_inputs_feature(self,batch,verbose=False):
        return_batch={}
        if verbose:
            print("ABlock-get_gt_inputs_features")

        #obsv video-wise action name        
        batch_size=len(batch['obsv_clip_action_name'])
        return_batch['obsv_batch_action_name']=[item.split("@")[0] for item in batch['obsv_clip_action_name']]
        return_batch['obsv_batch_action_idx']=torch.zeros(len(return_batch['obsv_batch_action_name']),dtype=torch.int64)
        for vid, vname in enumerate(return_batch['obsv_batch_action_name']):
            return_batch['obsv_batch_action_idx'][vid]=self.action_name2idx[vname]

        #obsv clip-wise object name
        return_batch["obsv_clip_obj_name"]=[]
        for item in batch["obsv_clip_obj_name"]:
            return_batch["obsv_clip_obj_name"]+=copy.deepcopy(item.split("@"))

        #and object feature from object name
        clip_otokens_obsv=compute_berts_for_strs(self.model_bert,return_batch["obsv_clip_obj_name"],verbose)
        return_batch["obsv_batch_clip_otokens"]=clip_otokens_obsv.view(batch_size,-1,clip_otokens_obsv.shape[-1]).detach().clone()

        for label in ["obsv","pred"]:
            #then valid
            for name in ["clip_valid_clip","clip_frame_valid_frame","clip_midpe","clip_since_action_start"]:
                if f"{label}_{name}" in batch:
                    return_batch[f"{label}_batch_{name}"]=batch[f"{label}_{name}"].detach().clone()
            for name in ["clip_frame_image_vis"]:
                if f"{label}_{name}" in batch:
                    return_batch[f"{label}_{name}"]=batch[f"{label}_{name}"].detach().clone()

            #then get hand components            
            if "obsv_clip_frame_hand_size_left" in batch:
                batch_clip_frame_shape=batch[label+'_clip_frame_cam_joints3d_left'].shape[:3]
                #hand size is based on only observation
                for k in ["hand_size_left","hand_size_right"]:
                    return_batch["obsv_frame_flatten_"+k]=torch.flatten(batch["obsv_clip_frame_"+k],0,1)

                batch_flatten_hand={}
                for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                            'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:
                    batch_flatten_hand[name]=torch.flatten(batch[f"{label}_clip_frame_{name}"],start_dim=0,end_dim=2)
                
                flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                                len_seq=self.ntokens_per_clip,
                                                spacing=self.model_pblock.spacing,
                                                base_frame_id=self.model_pblock.base_frame_id,
                                                factor_scaling=self.hand_scaling_factor,
                                                masked_placeholder=self.placeholder_joints,
                                                with_augmentation=False,
                                                compute_local2first=False, verbose=verbose)

                to_return_gt=flatten_comps["gt"][:,:self.model_pblock.dim_hand_feature]
                return_batch[label+"_batch_clip_frame_hand_comp"]=to_return_gt.view(batch_clip_frame_shape+to_return_gt.shape[1:])
                return_batch[label+"_batch_clip_frame_joints3d_in_cam_gt"]=hand_gts["flatten_joints3d_in_cam_gt"].view(batch_clip_frame_shape+hand_gts["flatten_joints3d_in_cam_gt"].shape[1:])
                return_batch[label+"_batch_clip_frame_joints3d_in_local_gt"]=hand_gts["flatten_joints3d_in_local_gt"].view(return_batch[label+"_batch_clip_frame_joints3d_in_cam_gt"].shape)

        if verbose:
            print("****Check return_batch")
            for k,v in return_batch.items():
                try:
                    print(k,v.shape)
                except:
                    print("**list**",k,len(v))
            print("End get_gt_inputs_features")
            exit(0)
        return return_batch
    
    def get_pblock_concat_inout_seq(self,batch,label,verbose=False):
        batch_flatten_hand={}
        for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                    'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:
            feature_op=batch[f"{label}_clip_frame_{name}"].detach().clone()
            batch_flatten_hand[name]=torch.flatten(feature_op,start_dim=0,end_dim=2)
            len_seq_op=feature_op.shape[1]*feature_op.shape[2]
            if verbose:
                print("len_seq_op/base_frame_id",len_seq_op,self.model_pblock.base_frame_id)#(len_a+1)*len_p,len_p-1
        if verbose:
            for k,v in batch_flatten_hand.items():
                print("batch_flatten_hand-",k,v.shape)

        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=len_seq_op, 
                                        spacing=self.model_pblock.spacing,
                                        base_frame_id=self.model_pblock.base_frame_id, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=None,#no need to use masked_placeholder
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=verbose)
        for k,v in hand_gts.items():
            batch_flatten_hand[k]=v[:,:self.model_pblock.dim_hand_feature] if k=="flatten_valid_features" else v
        batch_flatten_hand["flatten_hand_comp_gt"]=flatten_comps["gt"][:,:self.model_pblock.dim_hand_feature]
        if verbose:
            print("===============")
            for k,v in batch_flatten_hand.items():
                print("batch_flatten_hand-",k,v.shape)
        if label+"_clip_frame_image_vis" in batch:
            batch_flatten_hand["batch_seq_image_vis_"+label]=torch.flatten(batch[label+"_clip_frame_image_vis"],1,2)
        return batch_flatten_hand


    def forward(self, batch, is_train, to_reparameterize, gt_action_for_dec, batch_pdec_gt=None, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results,losses = {},{}

        already_batch0="obsv_batch_clip_otokens" in batch
        if not already_batch0:        
            #reformat batch
            batch0=self.get_inputs_feature(batch,verbose)
            batch_pdec_gt=self.get_pblock_concat_inout_seq(batch,"pred",verbose=verbose and False) if "obsv_clip_frame_hand_size_left" in batch else None
        else:
            batch0=batch
        
        batch_size=len(batch0['obsv_batch_action_name'])
        #First get A-Enc mid-level(Pe)
        if self.online_midpe:
            #first pass P-Enc to get mid-level, P-Enc input comp and mask
            batch_clip_frame_hand_comp_obsv=batch0["obsv_batch_clip_frame_hand_comp"]
            
            #[batch_size*len_a,len_p,dim_code] for P-Enc
            clip_frame_penc_in_hand_comp_obsv=torch.flatten(batch_clip_frame_hand_comp_obsv,start_dim=0,end_dim=1)
            clip_frame_penc_in_mask_obsv=~torch.flatten(batch0["obsv_batch_clip_frame_valid_frame"].cuda().bool(),start_dim=0,end_dim=1)
            
            if verbose:
                print("****A-Enc, P-Enc input ****")
                print("clip_frame_penc_in_hand_comp_obsv",clip_frame_penc_in_hand_comp_obsv.shape)#[bs*len_a,len_p,144]
                print(clip_frame_penc_in_hand_comp_obsv[0,0,self.num_joints*3:self.num_joints*3+18])
                print(torch.abs(clip_frame_penc_in_hand_comp_obsv[:,0,self.num_joints*3:self.num_joints*3+18]-clip_frame_penc_in_hand_comp_obsv[0,0,self.num_joints*3:self.num_joints*3+18]).max())
                print("clip_frame_penc_in_mask_obsv",clip_frame_penc_in_mask_obsv.shape)#[bs*len_a,len_p]

            #P-Enc output, mu for Mid(Pe)
            with torch.no_grad():
                clip_penc_mid_obsv,_,_,clip_frame_penc_out_comp_obsv=self.model_pblock.feed_encoder(clip_frame_penc_in_hand_comp_obsv,clip_frame_penc_in_mask_obsv,verbose=False)
            
            results["clip_frame_penc_out_comp_obsv"]=clip_frame_penc_out_comp_obsv

            #A-Enc input motion, resize to [batch_size, len_a, dim_code]; and stop_gradient
            batch_clip_penc_mid_obsv=clip_penc_mid_obsv[:,0].view(batch_size,-1,clip_penc_mid_obsv.shape[-1])
        
            if verbose and False:
                for bid in range(0,batch_clip_penc_mid_obsv.shape[0]):
                    for sid in range(0,batch_clip_penc_mid_obsv.shape[1]):
                        if batch0["obsv_batch_clip_valid_clip"][bid,sid]<1e-6:
                            batch_clip_penc_mid_obsv[bid,sid]=0
                print(batch_clip_penc_mid_obsv.shape)
                v1=torch.mean(torch.abs(batch0["obsv_batch_clip_midpe"].cuda()-batch_clip_penc_mid_obsv))
                if v1>5e-4:
                    print(batch_clip_penc_mid_obsv[0,0],'\n',torch.abs(batch0["obsv_batch_clip_midpe"].cuda())[0,0],'\n',batch0["obsv_batch_clip_frame_valid_frame"][0,0])
                    print("obsv",v1)
                    assert False
                print("==================================")
                print("****A-Enc in: batch_clip_penc_mid_obsv",batch_clip_penc_mid_obsv.shape)#[bs,len_a,512]
        else:
            batch_clip_penc_mid_obsv=batch0["obsv_batch_clip_midpe"].cuda()
            
        #Augmentation
        if not self.allow_grad_to_pblock:
            batch_clip_penc_mid_obsv_gt=batch_clip_penc_mid_obsv.detach().clone()
        if is_train and not already_batch0:
            batch_clip_penc_mid_obsv += torch.randn_like(batch_clip_penc_mid_obsv)*self.noise_factor_midpe

        #A-Enc input object, [batch_size*len_a, dim] for Bert
        batch_clip_otokens_obsv=batch0["obsv_batch_clip_otokens"]
        #Concat for A-Enc input
        batch_clip_aenc_in_mo=torch.cat([torch.unsqueeze(batch_clip_penc_mid_obsv,2),torch.unsqueeze(batch_clip_otokens_obsv,2)],dim=2)
        
        #add phase embedding
        batch_clip_aenc_in_phase=self.phase_pe.forward_idx(batch0["obsv_batch_clip_since_action_start"].cuda())
        batch_clip_aenc_in_mo+=batch_clip_aenc_in_phase
        if verbose:
            #print("batch0[obsv_clip_obj_name]",len(batch0["obsv_clip_obj_name"]),batch0["obsv_clip_obj_name"][-10:])
            print("****A-Enc in: batch_clip_otokens_obsv",batch_clip_otokens_obsv.shape,torch.abs(batch_clip_otokens_obsv.norm(dim=-1)-1.).max())#[bs,len_a,512]
            print("batch_clip_aenc_in_mo",batch_clip_aenc_in_mo.shape)#[bs,len_a,2,512]  
            print("batch_clip_aenc_in_phase",batch_clip_aenc_in_phase.shape)#[bs,len_a,1,512]
        
        #Reshape to [batch_size,len_a,dim_code] for A-Enc input
        batch_clip_aenc_in_mo=batch_clip_aenc_in_mo.view(batch_size,-1,self.code_dim)       
        
        batch_clip_aenc_mask_ptokens=~batch0["obsv_batch_clip_valid_clip"].cuda().bool()
        batch_clip_aenc_mask_mo=torch.cat([batch_clip_aenc_mask_ptokens,batch_clip_aenc_mask_ptokens],dim=-1)
        batch_clip_aenc_mask_mo=torch.flatten(batch_clip_aenc_mask_mo,start_dim=1)
        if verbose:
            print("batch_clip_aenc_mask_ptokens",batch_clip_aenc_mask_ptokens.shape)#[bs,len_a,1]
            print("batch_clip_aenc_in_mo",batch_clip_aenc_in_mo.shape)#[bs,len_a*2,512]
            print("batch_clip_aenc_mask_mo",batch_clip_aenc_mask_mo.shape)#[bs,len_a*2]
        
        #A-Enc output for action classification
        batch_aenc_action_mu,batch_aenc_action_logvar,batch_clip_aenc_out_mid=self.feed_encoder(batch_clip_aenc_in_mo,batch_clip_aenc_mask_mo,verbose=verbose)

        #sync A-Enc output with input?
        if False and self.lambda_mid_sync is not None and self.lambda_mid_sync>1e-10:
            assert False
            batch_clip_valid_obsv=batch0["obsv_batch_clip_valid_clip"].cuda().clone()
            total_loss_aenc, losses_aenc, results_aenc= self.compute_code_sync_loss(batch_clip_gt=batch_clip_penc_mid_obsv_gt,
                                                                                batch_clip_out=batch_clip_aenc_out_mid,
                                                                                batch_clip_valid=batch_clip_valid_obsv,verbose=verbose) 
            total_loss+=self.lambda_mid_sync*total_loss_aenc
            for k in losses_aenc.keys():
                losses[k+"_obsv"]=losses_aenc[k]


        #first compute KL-div        
        losses["kld_loss"]=self.compute_kl_loss(mu1=batch_aenc_action_mu[:,0],logvar1=batch_aenc_action_logvar[:,0],verbose=verbose)
        if self.lambda_kl is not None and self.lambda_kl>1e-10:
            total_loss+=self.lambda_kl*losses["kld_loss"]
        
        #FC(CLIP)
        batch_aname_obsv_gt=batch0["obsv_batch_action_name"]
        results["batch_action_name_obsv"]=batch_aname_obsv_gt
        batch_atokens_obsv_gt=open_clip.tokenizer.tokenize(batch_aname_obsv_gt).cuda()
        with torch.no_grad():            
            batch_atokens_obsv_gt=self.model_bert.encode_text(batch_atokens_obsv_gt).float()
        batch_atokens_obsv_gt/=batch_atokens_obsv_gt.norm(dim=-1,keepdim=True)
        if verbose:
            print("****A-Enc GT Action: batch_aname_obsv_gt",len(batch_aname_obsv_gt),batch_aname_obsv_gt[-10:])
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape,torch.abs(batch_atokens_obsv_gt.norm(dim=-1)-1.).max())#[bs,512]

        batch_atokens_obsv_gt=self.bert_to_latent(batch_atokens_obsv_gt.detach().clone())
        action_embedding=torch.transpose(self.bert_to_latent(torch.transpose(self.action_embedding,0,1).detach()),0,1)
        #action_embedding=self.action_embedding
        if verbose:
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape)#[bs,512]
            print("self.action_embedding/action_embedding",self.action_embedding.shape,action_embedding.shape)#[512,nembeddings],[512,nembeddings]


        #A-Dec
        batch_clip_adec_mem=torch.unsqueeze(batch_atokens_obsv_gt,1) if gt_action_for_dec else batch_aenc_action_mu[:,0:1]#30*batch_atokens_obsv_gt
        if is_train:
            batch_clip_adec_mem=self.reparameterize(mu=batch_clip_adec_mem,logvar=batch_aenc_action_logvar[:,0:1])
        elif to_reparameterize:
            batch_clip_adec_mem=self.reparameterize(mu=batch_clip_adec_mem,logvar=batch_aenc_action_logvar[:,0:1],factor=5.)#+=torch.randn_like(batch_clip_adec_mem)

        #action loss
        batch_aenc_action_obsv_out=torch.squeeze(batch_clip_adec_mem,1)
        losses["action_dist_loss"]=self.code_loss(batch_aenc_action_obsv_out,batch_atokens_obsv_gt)

        #Contrastive with Cosine Distance        
        results["batch_action_idx_obsv_gt"]=batch0["obsv_batch_action_idx"].cuda()
        batch_action_similarity_obsv,results["batch_action_idx_obsv_out"]=embedding_lookup(query=batch_aenc_action_obsv_out,embedding=action_embedding, verbose=verbose)#
        batch_action_similarity_obsv=batch_action_similarity_obsv/self.temperature
        results["batch_action_prob_distrib"]=nn.functional.softmax(batch_action_similarity_obsv,dim=1)
        losses["action_contrast_loss"] = torch_f.cross_entropy(batch_action_similarity_obsv,results["batch_action_idx_obsv_gt"],reduction='mean')
        
        if self.lambda_action is not None and self.lambda_action>1e-6:
            total_loss+=self.lambda_action*(losses["action_dist_loss"]+losses["action_contrast_loss"])
        if verbose:
            print("batch_aenc_action_obsv_out",batch_aenc_action_obsv_out.shape)#[bs,512]
            print("batch_action_idx_obsv_gt",results["batch_action_idx_obsv_gt"].shape)#[bs]
            print("batch_action_similarity_obsv/batch_action_prob_distrib",batch_action_similarity_obsv.shape,results["batch_action_prob_distrib"].shape)#[bs,tax_size]x2
            prob_norm=torch.sum(results["batch_action_prob_distrib"], axis=-1)
            print("check sum", prob_norm.shape, torch.abs(prob_norm-1).max())#[bs]

            print('if gt input, check action_idx gt vs out',torch.abs(results["batch_action_idx_obsv_gt"]-results["batch_action_idx_obsv_out"]).max())
            print("==================================")

        #continue cross-attn
        batch_clip_adec_mem_mask=torch.zeros_like(batch_clip_aenc_mask_ptokens[:,0:1,0])
        if verbose:
            print("****A-Dec in- batch_clip_adec_mem",batch_clip_adec_mem.shape)#[bs,1,512]
            print("batch_clip_adec_mem_mask",batch_clip_adec_mem_mask.shape)#[bs,1]

        rand_to_concat_mid_aenc_out=torch.rand(1).cuda()
        if (not is_train)or rand_to_concat_mid_aenc_out<0.5:
            batch_clip_adec_mem=torch.cat((batch_clip_adec_mem, batch_clip_aenc_out_mid),dim=1)            
            batch_clip_adec_mem_mask=torch.cat((batch_clip_adec_mem_mask, batch_clip_aenc_mask_ptokens[:,:,0]),dim=1)       

        #Get Mid(Ad) for pred
        batch_clip_adec_ptoken_mask=~batch0["pred_batch_clip_valid_clip"][:,:self.ntokens_pred,0].cuda().bool()
        batch_clip_adec_ptoken_mask[:,0]=False
        
        if verbose:
            print("batch_clip_adec_mem",batch_clip_adec_mem.shape)#[bs,1,512] or [bs,1+len_a,512]
            print("batch_clip_adec_mem_mask",batch_clip_adec_mem_mask.shape)#[bs,1] or [bs,1+len_a]            
            print("batch_clip_adec_ptoken_mask",batch_clip_adec_ptoken_mask.shape)#[bs,len_a]


        #add phase embedding
        batch_clip_adec_in_phase=torch.squeeze(self.phase_pe.forward_idx(batch0["pred_batch_clip_since_action_start"][:,:self.ntokens_pred].cuda()),2)
        batch_clip_adec_mid_pred=self.feed_decoder(batch_seq_dec_query=batch_clip_adec_in_phase,
                                batch_seq_dec_mem=batch_clip_adec_mem,
                                batch_seq_dec_mem_mask=batch_clip_adec_mem_mask,
                                batch_seq_dec_tgt_key_padding_mask=batch_clip_adec_ptoken_mask,verbose=verbose)

        if verbose:
            print("batch_clip_adec_in_phase",batch_clip_adec_in_phase.shape)#[bs,len_a,512]
            print("batch_clip_adec_mid_pred",batch_clip_adec_mid_pred.shape)#[bs,len_a,512]
            print("==================================")

        #Get Mid(Ae) from pred, here no augmentation.
        if self.online_midpe:
            batch_clip_frame_hand_comp_pred=batch0["pred_batch_clip_frame_hand_comp"][:,:self.ntokens_pred].clone()
            clip_frame_penc_in_hand_comp_pred=torch.flatten(batch_clip_frame_hand_comp_pred,start_dim=0,end_dim=1)
            clip_frame_penc_in_mask_pred=~torch.flatten(batch0["pred_batch_clip_frame_valid_frame"][:,:self.ntokens_pred].cuda().bool().clone(),start_dim=0,end_dim=1)

            if verbose:
                print("****Pred, P-Enc input clip_frame_penc_in_hand_comp_pred",clip_frame_penc_in_hand_comp_pred.shape)#[bs*len_a,len_p,144]
                print(clip_frame_penc_in_hand_comp_pred[0,0,self.num_joints*3:self.num_joints*3+18])
                print(torch.abs(clip_frame_penc_in_hand_comp_pred[:,0,self.num_joints*3:self.num_joints*3+18]-clip_frame_penc_in_hand_comp_pred[0,0,self.num_joints*3:self.num_joints*3+18]).max())
                print("clip_frame_penc_in_mask_pred",clip_frame_penc_in_mask_pred.shape)#[bs*len_a,len_p]
            with torch.no_grad():
                clip_penc_mid_pred,_,_,_=self.model_pblock.feed_encoder(clip_frame_penc_in_hand_comp_pred,clip_frame_penc_in_mask_pred,verbose=verbose)
            #reshape to [bs,len_a,dim_code] and stop gradient
            batch_clip_penc_mid_pred=clip_penc_mid_pred[:,0].view(batch_size,-1,self.code_dim).detach().clone()
        
            if verbose and False:
                #check with lmdb!
                print(batch0["pred_batch_clip_frame_valid_frame"].shape,batch0["pred_batch_clip_valid_clip"].shape)
                batch0_midpe=batch0["pred_batch_clip_midpe"].cuda()
                for bid in range(0,batch_clip_penc_mid_pred.shape[0]):
                    for sid in range(0,batch_clip_penc_mid_pred.shape[1]):
                        if batch0["pred_batch_clip_valid_clip"][bid,sid]<1e-6:
                            batch_clip_penc_mid_pred[bid,sid]=batch0_midpe[bid,sid]

                v1=torch.mean(torch.abs(batch0_midpe[:,:self.ntokens_pred].cuda()-batch_clip_penc_mid_pred))
                if v1>5e-4:
                    print(batch0_midpe[:,:self.ntokens_pred].cuda(),'\n',batch_clip_penc_mid_pred)
                    print("pred",v1)
                    assert False
                    
                print("==================================")
                print("****A-Enc in: batch_clip_penc_mid_pred",batch_clip_penc_mid_pred.shape)#[bs,len_a,512]
        else:
            batch_clip_penc_mid_pred=batch0["pred_batch_clip_midpe"][:,:self.ntokens_pred].clone().cuda()        
        
        
        #compute Mid(Pe)-Mid(Ad) sync loss
        batch_clip_valid_pred=batch0["pred_batch_clip_valid_clip"][:,:self.ntokens_pred].cuda().clone()
        if self.lambda_mid_sync is not None and self.lambda_mid_sync>1e-10:
            total_loss_adec,losses_adec,results_adec=self.compute_code_sync_loss(batch_clip_gt=batch_clip_penc_mid_pred,
                                                                            batch_clip_out=batch_clip_adec_mid_pred,
                                                                            batch_clip_valid=batch_clip_valid_pred,verbose=verbose)
            total_loss+=self.lambda_mid_sync*total_loss_adec
            losses.update(losses_adec)
            results.update(results_adec)
            
        #generate motion with A-Dec output Mid(Ad)
        if batch_pdec_gt is not None:
            #Examine hand loss and get hand output
            total_loss_motion,results_motion,losses_motion=self.generate_motion(batch0,batch_pdec_gt,batch_clip_adec_mid_pred,
                                                        normalize_size_from_comp=not is_train,verbose=verbose)
            total_loss+=total_loss_motion
            results.update(results_motion)
            losses.update(losses_motion)

        if verbose:       
            for k,v in losses.items():
                print(k,v)
            for k,v in results.items():
                try:
                    print(k,v.shape)
                except:
                    print(k,len(v))
        
        return total_loss,results,losses
    
    def feed_encoder(self,batch_seq_enc_in_tokens,batch_seq_enc_mask_tokens, verbose):      
        if verbose:
            print("****Start A-Enc****")
        batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_tokens=super().feed_encoder(batch_seq_enc_in_tokens,batch_seq_enc_mask_tokens,verbose)
        batch_seq_enc_out_mid=batch_seq_enc_out_tokens[:,0::2]########
        if verbose:
            print("batch_seq_enc_out_mid",batch_seq_enc_out_mid.shape)#[bs,len_a*2,512],[bs,len_a,512]
            print("****End A-Enc****")
            
        return batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_mid
        
    def compute_code_sync_loss(self,batch_clip_gt,batch_clip_out,batch_clip_valid,verbose=False):        
        losses,results={},{}
        total_loss= torch.Tensor([0]).cuda()
        batch_clip_loss_mid=self.code_loss(batch_clip_out,batch_clip_gt,reduction='none')
        batch_clip_loss_mid=torch.mean(batch_clip_loss_mid,dim=-1,keepdim=True)
        if verbose:
            print("batch_clip_gt/batch_clip_out",batch_clip_gt.shape,batch_clip_out.shape)#[bs,len_a,512]x2
            print("batch_clip_loss_mid/batch_clip_valid",batch_clip_loss_mid.shape,batch_clip_valid.shape)#[bs,len_a,1]x2
            print(batch_clip_valid[:4,:,0])
            print("==================================")
            
        batch_syn_loss=torch.mul(batch_clip_loss_mid,batch_clip_valid)
        cnt=torch.sum(batch_clip_valid)
        losses["mid_code_dist"]=torch.sum(batch_syn_loss)/torch.where(cnt<1,1,cnt)
        total_loss+=losses["mid_code_dist"]

        #cnt=torch.sum(batch_clip_valid,axis=0)
        #results['clip_mid_code_dist']=torch.flatten(torch.sum(batch_syn_loss,axis=0)/torch.where(cnt<1,1,cnt))
        return total_loss,losses,results


    
    def generate_motion(self,batch0, batch_pdec_gt, batch_clip_adec_mid_pred, normalize_size_from_comp,verbose=False):
        losses,results={},{}
        total_loss= torch.Tensor([0]).cuda()
        batch_size=batch_clip_adec_mid_pred.shape[0]

        #And use Mid(Ad) to generate hand pose
        #reshape and unsqueeze to [bs*len_a,1,code_dim]
        clip_frame_pdec_mem=torch.unsqueeze(torch.flatten(batch_clip_adec_mid_pred,start_dim=0,end_dim=1),1)
        clip_frame_pdec_mem_mask=torch.zeros(clip_frame_pdec_mem.shape[0],1,device=batch_clip_adec_mid_pred.device).bool()
        if verbose:
            print("batch_size",batch_size)
            print("****P-Dec in from A-Dec out, clip_frame_pdec_mem",clip_frame_pdec_mem.shape)#[bs*len_a,1,512]
            print("clip_frame_pdec_mem_mask",clip_frame_pdec_mem_mask.shape)#[bs*len_a,1]
        
        #Pass P-Dec to get hand pose
        clip_frame_pdec_hand_feature_out,clip_frame_pdec_hand_comp_out=self.model_pblock.feed_decoder(clip_frame_pdec_mem,clip_frame_pdec_mem_mask,verbose=verbose)
        if verbose:
            print("****P-Dec out,clip_frame_pdec_hand_feature_out",clip_frame_pdec_hand_feature_out.shape)#[bs*len_a,len_p,512]
            print("clip_frame_pdec_hand_comp_out",clip_frame_pdec_hand_comp_out.shape)#[bs*len_a,len_p,144]
            print("==================================")
            
        
        #hand size
        if "batch_mean_hand_size_left" in batch0:
            batch_mean_hand_left_size=batch0["batch_mean_hand_size_left"]
            batch_mean_hand_right_size=batch0["batch_mean_hand_size_right"]
        else:
            flatten_valid_frame=torch.flatten(batch0['obsv_batch_clip_frame_valid_frame'])
            batch_hand_size_left=torch.mul(torch.flatten(batch0['obsv_frame_flatten_hand_size_left']),flatten_valid_frame).view(-1,self.ntokens_obsv*self.model_pblock.ntokens_obsv)
            batch_mean_hand_left_size=torch.sum(batch_hand_size_left,dim=1,keepdim=True)/torch.sum(flatten_valid_frame.view(-1,self.ntokens_obsv*self.model_pblock.ntokens_obsv),dim=1,keepdim=True)

            batch_hand_size_right=torch.mul(torch.flatten(batch0['obsv_frame_flatten_hand_size_right']),flatten_valid_frame).view(-1,self.ntokens_obsv*self.model_pblock.ntokens_obsv)
            batch_mean_hand_right_size=torch.sum(batch_hand_size_right,dim=1,keepdim=True)/torch.sum(flatten_valid_frame.view(-1,self.ntokens_obsv*self.model_pblock.ntokens_obsv),dim=1,keepdim=True)

            if verbose:
                print("flatten_valid_frame,batch_hand_size_left,batch_hand_size_right",flatten_valid_frame.shape,batch_hand_size_left.shape,batch_hand_size_right.shape)#[bs*len_a*len_p],[bs,len_a*len_p]x2
        if verbose:
            print("batch_mean_hand_left_size, batch_mean_hand_right_size",batch_mean_hand_left_size.shape, batch_mean_hand_right_size.shape)#[bs,1]x2
            
        
        #valid frames and features
        batch_seq_valid_frames=torch.flatten(batch0["pred_batch_clip_frame_valid_frame"][:,1:].cuda().clone(),1,2)
        batch_seq_comp_out=clip_frame_pdec_hand_comp_out.view(batch_size,self.ntokens_pred*self.model_pblock.ntokens_pred,-1)
        results["batch_seq_valid_frames_pred_token0"]=batch_seq_valid_frames[:,0:self.model_pblock.ntokens_pred].detach().clone()

        d1,d2=batch_size,(self.ntokens_pred+1)*self.model_pblock.ntokens_pred
        batch_seq_comp_gt=batch_pdec_gt["flatten_hand_comp_gt"].view(d1,d2,-1)[:,self.model_pblock.ntokens_obsv:].clone()
        batch_seq_local2base_gt=batch_pdec_gt["flatten_local2base_gt"].view(d1,d2,-1)[:,self.model_pblock.ntokens_obsv:].clone()
        batch_seq_valid_features=batch_pdec_gt["flatten_valid_features"].view(d1,d2,-1)[:,self.model_pblock.ntokens_obsv:].clone()
        
        results["batch_seq_valid_frames_pred"]=batch_seq_valid_frames.detach().clone()
        batch_seq_valid_features=torch.mul(batch_seq_valid_features,torch.unsqueeze(batch_seq_valid_frames,dim=-1))
        
        if verbose:
            print("==================================")
            print("batch_seq_hand_comp_gt/batch_seq_local2base_gt",batch_seq_comp_gt.shape,batch_seq_local2base_gt.shape)#[bs*len_a,len_p,144],[bs*len_a,len_p,18]///[bs,len_a*len_p,144],[bs,len_a*len_p,18]            
            print('results[batch_seq_valid_frames_pred]/batch_seq_valid_frames',results["batch_seq_valid_frames_pred"].shape,batch_seq_valid_frames.shape)#[bs*len_a,len_p],[bs*len_a,len_p,144]///[bs,len_a*len_p],[bs,len_a*len_p,144]
            print("batch_seq_valid_features",batch_seq_valid_features.shape)#[bs*len_a,len_p,144]///[bs,len_a*len_p,144]
            print("batch_seq_comp_out",batch_seq_comp_out.shape)#[bs*len_a,len_p,144]///[bs,len_a*len_p,144]
            print("batch_mean_hand_left_size/batch_mean_hand_right_size",batch_mean_hand_left_size.shape,batch_mean_hand_right_size.shape)#[bs,1]x2
      
        trans_info_pred={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info_pred[k]=batch_pdec_gt[k]
            
        #for checking the compute_hand_loss (check only the max element of hand_comp loss), no masked_placeholder, frame_dim discards the first frame,
        #verify_gt_comp=torch.flatten(batch0["pred_batch_clip_frame_hand_comp"][:,1:],0,1)
        
        total_loss_hand,results_hand,losses_hand=self.model_pblock.compute_hand_loss(batch_seq_comp_gt=batch_seq_comp_gt,#[:,1:], 
                                                            batch_seq_comp_out=batch_seq_comp_out,#batch_seq_comp_out,#verify_gt_comp[:,1:],
                                                            compute_local2base=True,
                                                            batch_seq_local2base_gt=batch_seq_local2base_gt,#[:,1:], 
                                                            batch_seq_valid_features=batch_seq_valid_features,#[:,1:],
                                                            batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                            trans_info=trans_info_pred,
                                                            normalize_size_from_comp=normalize_size_from_comp,
                                                            verbose=verbose)
        
        for k,v in losses_hand.items():
            losses[k+"_predict"]=v
        if self.lambda_hand is not None and self.lambda_hand>1e-6:
            assert False
            total_loss+=self.lambda_hand*(losses_hand["recov_hand_loss"]+losses_hand["recov_trj_in_base_loss"])

        #aggregate output for img, here suppose self.type_motion_to_gen in ["video_wise"]:
        if "batch_seq_image_vis_pred" in batch_pdec_gt:
            results["batch_seq_image_vis_cdpred"]=batch_pdec_gt["batch_seq_image_vis_pred"]
            results["batch_seq_image_vis_obsv"]=torch.flatten(batch0["obsv_clip_frame_image_vis"],1,2)        
        results["batch_seq_valid_frames_obsv"]=torch.flatten(batch0["obsv_batch_clip_frame_valid_frame"].cuda().clone(),1,2)
        results["batch_seq_valid_frames_cdpred"]=torch.flatten(batch0["pred_batch_clip_frame_valid_frame"].cuda().clone(),1,2)
        
        
        for key in ["local","base","cam"]:
            results[f"batch_seq_joints3d_in_{key}_cdpred_gt"]=batch_pdec_gt[f"flatten_joints3d_in_{key}_gt"].view(d1,d2,self.num_joints,3)/self.hand_scaling_factor
            results[f"batch_seq_joints3d_in_{key}_pred_gt"]=results[f"batch_seq_joints3d_in_{key}_cdpred_gt"][:,self.model_pblock.ntokens_obsv:]
            results[f"batch_seq_joints3d_in_{key}_pred_out"]=results_hand[f"batch_seq_joints3d_in_{key}_out"]/self.hand_scaling_factor
            
            results[f"batch_seq_joints3d_in_{key}_pred_gt_token0"]=results[f"batch_seq_joints3d_in_{key}_pred_gt"][:,0:self.model_pblock.ntokens_pred]
            results[f"batch_seq_joints3d_in_{key}_pred_out_token0"]=results[f"batch_seq_joints3d_in_{key}_pred_out"][:,0:self.model_pblock.ntokens_pred]

            if key in ["local","cam"]:
                results[f"batch_seq_joints3d_in_{key}_obsv_gt"]=torch.flatten(batch0[f"obsv_batch_clip_frame_joints3d_in_{key}_gt"],1,2)/self.hand_scaling_factor
            if verbose:
                print(key,torch.abs(results[f"batch_seq_joints3d_in_{key}_pred_gt"]-results[f"batch_seq_joints3d_in_{key}_pred_out"]).max())
        
        if verbose:
            from meshreg.netscripts.utils import sample_vis_trj_dec
            from meshreg.datasets import ass101utils
            batch_clip_frame_imgs=torch.cat([batch0["obsv_clip_frame_image_vis"],batch0["pred_clip_frame_image_vis"]],dim=1)
            flatten_imgs=torch.flatten(batch_clip_frame_imgs,0,2)
            
            batch_clip_frame_joints3d_in_cam_gt=torch.cat([batch0["obsv_batch_clip_frame_joints3d_in_cam_gt"],batch0["pred_batch_clip_frame_joints3d_in_cam_gt"]],dim=1)/self.hand_scaling_factor
            batch_clip_frame_joints3d_in_local_gt=torch.cat([batch0["obsv_batch_clip_frame_joints3d_in_local_gt"],batch0["pred_batch_clip_frame_joints3d_in_local_gt"]],dim=1)/self.hand_scaling_factor
            
            batch_seq_joints3d_in_cam_gt=torch.flatten(batch_clip_frame_joints3d_in_cam_gt,1,2)
            batch_seq_joints3d_in_local_gt=torch.flatten(batch_clip_frame_joints3d_in_local_gt,1,2)

            batch_seq_joints3d_in_cam_out=results_hand[f"batch_seq_joints3d_in_cam_out"].view(batch_size,-1,self.num_joints,3)/self.hand_scaling_factor
            batch_seq_joints3d_in_local_out=results_hand[f"batch_seq_joints3d_in_local_out"].view(batch_size,-1,self.num_joints,3)/self.hand_scaling_factor
            
            print("batch_clip_frame_imgs",batch_clip_frame_imgs.shape)
            print("batch_clip_frame_joints3d_gt",batch_clip_frame_joints3d_in_cam_gt.shape,batch_clip_frame_joints3d_in_local_gt.shape)
            print("batch_seq_joints3d_gt",batch_seq_joints3d_in_cam_gt.shape,batch_seq_joints3d_in_local_gt.shape)
            print("batch_seq_joints3d_out",batch_seq_joints3d_in_cam_out.shape,batch_seq_joints3d_in_local_out.shape)

            batch_action=batch0["obsv_batch_action_name"]

            links=[(0, 2, 3, 4),(0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]
            tag="h2o"
            if tag!="h2o":
                cam_info=ass101utils.get_view_extrinsic_intrisic(path_calib_txt='../ass101/annotations/calib.txt')["C10118_rgb"]
            else:
                cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
                #cam_info["intr"][:2]*=480/1280.0
            for sample_id in range(0,8):
                sample_vis_trj_dec(batch_seq_gt_cam=batch_seq_joints3d_in_cam_gt, 
                            batch_seq_est_cam=batch_seq_joints3d_in_cam_out, 
                            batch_seq_gt_local=batch_seq_joints3d_in_local_gt,
                            batch_seq_est_local=batch_seq_joints3d_in_local_out,
                            batch_gt_action_name=batch_action, 
                            joint_links=links,  
                            flatten_imgs=flatten_imgs,
                            sample_id=sample_id,
                            cam_info=cam_info,
                            prefix_cache_img=f"./vis_v3/{tag}_imgs/", path_video=f"./vis_v3/{tag}"+'_{:02d}.avi'.format(sample_id))
            exit(0)
        return total_loss,results,losses
            
            
    def process_until_enc(self, batch, postprocess_hand,  batch_is_flatten=True, verbose=False):
        if batch_is_flatten:
            results = self.model_pblock.process_until_enc(batch,postprocess_hand)
        else:
            assert False
        batch_size=batch["batch_clip_valid_clip"].shape[0]
        batch_clip_penc_mid_obsv=results["batch_mid_mu_enc_out"][:,0].view(batch_size,-1,results["batch_mid_mu_enc_out"].shape[-1]).detach().clone()

        clip_otokens_obsv=compute_berts_for_strs(self.model_bert,batch["clip_obj_name"],verbose)
        batch_clip_otokens_obsv=clip_otokens_obsv.view(batch_size,-1,clip_otokens_obsv.shape[-1]).detach().clone()
        #batch_clip_itokens_obsv=torch.zeros_like(batch_clip_otokens_obsv)
        
        #Concat for A-Enc input
        batch_clip_aenc_in_mo=torch.cat([torch.unsqueeze(batch_clip_penc_mid_obsv,2),torch.unsqueeze(batch_clip_otokens_obsv,2)],dim=2)#torch.unsqueeze(batch_clip_itokens_obsv,2)],dim=2)
        
        if verbose:
            print("batch[clip_obj_name]",len(batch["clip_obj_name"]),batch["clip_obj_name"][-10:])
            print("****A-Enc in: batch_clip_otokens_obsv",batch_clip_otokens_obsv.shape,torch.abs(batch_clip_otokens_obsv.norm(dim=-1)-1.).max())#[bs,len_a,512]
            print("batch_clip_aenc_in_mo",batch_clip_aenc_in_mo.shape)#[bs,len_a,2,512]  
        #Reshape to [batch_size,len_a,dim_code] for A-Enc input
        batch_clip_aenc_in_mo=batch_clip_aenc_in_mo.view(batch_size,-1,self.code_dim)       
        
        batch_clip_aenc_mask_ptokens=~batch["batch_clip_valid_clip"].cuda().bool()
        #batch_clip_aenc_mask_itokens=torch.ones_like(batch_clip_aenc_mask_ptokens)
        batch_clip_aenc_mask_mo=torch.cat([batch_clip_aenc_mask_ptokens,batch_clip_aenc_mask_ptokens],dim=-1)#,batch_clip_aenc_mask_itokens],dim=-1)
        batch_clip_aenc_mask_mo=torch.flatten(batch_clip_aenc_mask_mo,start_dim=1)
        if verbose:
            print("batch_clip_aenc_mask_ptokens",batch_clip_aenc_mask_ptokens.shape)#[bs,len_a,1]
            print("batch_clip_aenc_in_mo",batch_clip_aenc_in_mo.shape)#[bs,len_a*3,512]
            print("batch_clip_aenc_mask_mo",batch_clip_aenc_mask_mo.shape)#[bs,len_a*3]
        
        #A-Enc output for action classification
        batch_aenc_action_mu,batch_aenc_action_logvar,batch_clip_aenc_out_mid=self.feed_encoder(batch_clip_aenc_in_mo,batch_clip_aenc_mask_mo,verbose=verbose)
        batch_aenc_action_obsv_out=batch_aenc_action_mu[:,0]
        
        action_gt_names=batch["action_name"]
        action_gt_labels=torch.zeros(len(action_gt_names),dtype=torch.int64)
        for vid, vname in enumerate(action_gt_names):
            action_gt_labels[vid]=self.action_name2idx[vname]
        results["batch_action_idx_obsv_gt"]=action_gt_labels.cuda()
        action_embedding=torch.transpose(self.bert_to_latent(torch.transpose(self.action_embedding,0,1).detach()),0,1)
        _,results["batch_action_idx_obsv_out"]=embedding_lookup(query=batch_aenc_action_obsv_out,embedding=action_embedding, verbose=verbose)#
        results.update(batch)

        print(results["batch_action_idx_obsv_gt"],results["batch_action_idx_obsv_out"])
        return results
