import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np
import copy

from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt
from meshreg.models.utils import compute_bert_embedding_for_taxonomy,compute_berts_for_strs, embedding_lookup
from meshreg.models.utils import align_torch_batch_for_cam2local,recover_3d_proj_pinhole_wo_rescale,augment_hand_pose_2_5D,compute_in_cam_goal2start
from meshreg.netscripts.utils import sample_vis_l2r,sample_vis_ncam_cam
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.amodel import ContextNet as ContextNet_

import open_clip

class ContextNet(ContextNet_):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_nlayers_dec,
                        transformer_activation,

                        lambda_action=None,
                        lambda_mid_sync=None,
                        lambda_hand=None,
                        lambda_kl=None,
                        noise_std=1.,
                        ntokens_per_clip=15,
                        ntokens_per_video=16,
                        
                        pose_loss='l1',
                        code_loss='l1',
                        online_midpe=True,
                        allow_grad_to_pblock=True,):


        super().__init__(transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_dim_feedforward=transformer_dim_feedforward,
                        transformer_nlayers_enc=transformer_nlayers_enc,
                        transformer_nlayers_dec=transformer_nlayers_dec,
                        transformer_activation=transformer_activation,

                        lambda_action=lambda_action,
                        lambda_mid_sync=lambda_mid_sync,
                        lambda_hand=lambda_hand,
                        lambda_kl=lambda_kl,
                        noise_std=noise_std,
                        ntokens_per_clip=ntokens_per_clip,
                        ntokens_per_video=ntokens_per_video,
                        
                        pose_loss=pose_loss,
                        code_loss=code_loss,
                        online_midpe=online_midpe,
                        allow_grad_to_pblock=allow_grad_to_pblock,)
        
        self.mean_mano_palm_joints={'left': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT'))),
                'right': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT')))}


        self.pose_from_resnet=False
        self.obj_from_resnet=False
        self.gt_ite0=True

    
    def assign_ibe(self,model_ibe):
        self.model_ibe=model_ibe
        self.model_ibe.eval()
    
    def assign_poenc(self,model_poenc):
        self.model_poenc=model_poenc
        self.model_poenc.eval()    




    def get_inputs_feature(self,batch,verbose=False):
        return_batch_ablock={}#get hand size, and hand comp for A-Block
            
        num_videos=len(batch['obsv_clip_action_name'])  
        #hand motion and mid-level
        for label in ["obsv","pred"]:
            prepare_for_pblock_gt=label=="pred" or self.ntokens_obsv>1
            value_vf=batch[f"{label}_clip_frame_valid_frame"].cuda()
            nclips_per_part,nframes_per_clip=value_vf.shape[1:3]

            #GT supervision
            batch_flatten_hand={}
            for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                        'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:
                    
                value=batch[f"{label}_clip_frame_{name}"].cuda()
                batch_flatten_hand[name]=torch.flatten(value,start_dim=0,end_dim=2)
            
            flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=nframes_per_clip, 
                                        spacing=self.model_pblock.spacing,
                                        base_frame_id=0, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=False)

            if label=="pred":
                shape1=(num_videos,nclips_per_part,nframes_per_clip)+flatten_comps["gt"].shape[1:]
                return_batch_ablock["pred_batch_clip_frame_hand_comp"]=flatten_comps["gt"].view(shape1)[:,:-1].contiguous()

            #rearrange for A-Block GT
            for camtag in ["cam","local"]:
                shape2=(num_videos,nclips_per_part,nframes_per_clip)+hand_gts[f"flatten_joints3d_in_{camtag}_gt"].shape[1:]
                return_batch_ablock[f"{label}_batch_clip_frame_joints3d_in_{camtag}_gt"]=hand_gts[f"flatten_joints3d_in_{camtag}_gt"].view(shape2)                
                
            #Resnet feature
            batch_flatten_hand={}
            for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                        'hand_size_left','hand_size_right']:
                name1=f"{label}_clip_frame_{name}_resnet" if self.pose_from_resnet else f"{label}_clip_frame_{name}"
                batch_flatten_hand[name]=torch.flatten(batch[name1].cuda(),start_dim=0,end_dim=2)#
                
            for name in ["valid_joints_left","valid_joints_right"]:
                batch_flatten_hand[name]=torch.flatten(batch[f"{label}_clip_frame_{name}"].cuda(),start_dim=0,end_dim=2)
            
            #resnet
            if label=="obsv":
                flatten_comps, hands_resnet = get_flatten_hand_feature(batch_flatten_hand, 
                                                len_seq=self.model_pblock.ntokens_obsv, 
                                                spacing=self.model_pblock.spacing,
                                                base_frame_id=0, 
                                                factor_scaling=self.hand_scaling_factor, 
                                                masked_placeholder=self.placeholder_joints,
                                                with_augmentation=False,
                                                compute_local2first=False, verbose=verbose and False)

                shape2=(num_videos, nclips_per_part, nframes_per_clip)+flatten_comps["gt"].shape[1:]
                batch_clip_frame_hand_comp=flatten_comps["gt"].view(shape2)
                
                return_batch_ablock["obsv_batch_clip_frame_hand_comp"]=batch_clip_frame_hand_comp
                for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
                    return_batch_ablock[f"resnet_{k}"]=hands_resnet[k].detach().clone()
                
                batch_seq_valid_frame_obsv=torch.flatten(batch["obsv_clip_frame_valid_frame"].cuda(),0,1)
                batch_flatten_hand["batch_seq_valid_frame"]=batch_seq_valid_frame_obsv
                blend_info = compute_in_cam_goal2start(batch_flatten_hand,factor_scaling=self.hand_scaling_factor,verbose=verbose)
                for k,v in blend_info.items():
                    return_batch_ablock[f"resnet_{k}"]=v.detach().clone()
            

        #Compute hand size, with Resnet hands
        flatten_valid_frame=torch.flatten(batch['obsv_clip_frame_valid_frame']).cuda()
        shape3=(num_videos,self.ntokens_obsv*self.model_pblock.ntokens_obsv)

        #Use non-aug resnet hand.
        name1="obsv_clip_frame_hand_size_left_resnet" if self.pose_from_resnet else "obsv_clip_frame_hand_size_left"
        batch_hand_size_left=torch.mul(torch.flatten(batch[name1].cuda()),flatten_valid_frame).view(shape3) 
        batch_mean_hand_left_size=torch.sum(batch_hand_size_left,dim=1,keepdim=True)/torch.sum(flatten_valid_frame.view(shape3),dim=1,keepdim=True)
        name1="obsv_clip_frame_hand_size_right_resnet" if self.pose_from_resnet else "obsv_clip_frame_hand_size_right"
        batch_hand_size_right=torch.mul(torch.flatten(batch[name1].cuda()),flatten_valid_frame).view(shape3)
        batch_mean_hand_right_size=torch.sum(batch_hand_size_right,dim=1,keepdim=True)/torch.sum(flatten_valid_frame.view(shape3),dim=1,keepdim=True)

        return_batch_ablock["batch_mean_hand_size_left"]=batch_mean_hand_left_size
        return_batch_ablock["batch_mean_hand_size_right"]=batch_mean_hand_right_size
        if verbose:
            print("shape3",shape3)
            print("obsv_clip_frame_valid_frame",batch["obsv_clip_frame_valid_frame"].shape)#[bs,len_a,len_p]
            print("obsv_clip_frame_hand_size",batch["obsv_clip_frame_hand_size_left"].shape,batch["obsv_clip_frame_hand_size_right"].shape)#[bs,len_a,len_p]
            print("batch_mean_hand_size",batch_mean_hand_left_size.shape,batch_mean_hand_right_size.shape)#[bs,1]x2

        #PO-ENC
        return_batch_poenc_for_super={}    
        #obsv video-wise action name for A-Enc GT
        label="obsv"
        list_flatten_obj_name=[]
        for item in batch[f"{label}_clip_obj_name"]:
            list_flatten_obj_name+=copy.deepcopy(item.split("@"))
        clip_otokens_obsv=compute_berts_for_strs(self.model_bert,list_flatten_obj_name,verbose)
        
        return_batch_ablock["obsv_batch_clip_otokens_gt"]=clip_otokens_obsv.view(num_videos,self.ntokens_obsv,clip_otokens_obsv.shape[-1])
        if self.obj_from_resnet:
            for name in ["object_feature","valid_frame"]:
                return_batch_poenc_for_super[name]=torch.flatten(batch[f"{label}_clip_frame_{name}"].cuda(),0,2)
                
        #Action
        return_batch_ablock['obsv_batch_action_name']=[item.split("@")[0] for item in batch['obsv_clip_action_name']]
        return_batch_ablock['obsv_batch_action_idx']=torch.zeros(len(return_batch_ablock['obsv_batch_action_name']),dtype=torch.int64)
        for vid, vname in enumerate(return_batch_ablock['obsv_batch_action_name']):
            return_batch_ablock['obsv_batch_action_idx'][vid]=self.action_name2idx[vname]

        for label in ["obsv","pred"]:
            #then valid
            for name in ["clip_valid_clip","clip_frame_valid_frame","clip_since_action_start"]:
                if f"{label}_{name}" in batch:
                    return_batch_ablock[f"{label}_batch_{name}"]=batch[f"{label}_{name}"].detach().clone()
            for name in ["clip_frame_image_vis"]:
                if f"{label}_{name}" in batch:
                    return_batch_ablock[f"{label}_{name}"]=batch[f"{label}_{name}"].detach().clone()
                 
        return return_batch_ablock,return_batch_poenc_for_super


    def compute_pose_estimation(self,batch,results,verbose=False):
        clip_frame_valid_frames=torch.flatten(batch["obsv_batch_clip_frame_valid_frame"],0,1).cuda()
        clip_frame_penc_out_comp_obsv=results["clip_frame_penc_out_comp_obsv"]#torch.flatten(batch["obsv_batch_clip_frame_hand_comp"],0,1)#
        
        batch_mean_hand_left_size=batch["batch_mean_hand_size_left"].view(-1,1)
        batch_mean_hand_right_size=batch["batch_mean_hand_size_right"].view(-1,1)
        
        if verbose:
            print("clip_frame_valid_frames",clip_frame_valid_frames.shape)#[bs*len_a,len_p]
            print("clip_frame_penc_out_comp_obsv",clip_frame_penc_out_comp_obsv.shape)#[bs*len_a,len_p,153]
            print("batch_mean_hand_left_size",batch_mean_hand_left_size.shape)#[bs,1]
            print("batch_mean_hand_right_size",batch_mean_hand_right_size.shape)#[bs,1]

        clip_frame_mean_hand_left_size=repeat(batch_mean_hand_left_size,"b ()->b n",n=self.ntokens_obsv).reshape(-1,1)
        clip_frame_mean_hand_right_size=repeat(batch_mean_hand_right_size,"b ()->b n",n=self.ntokens_obsv).reshape(-1,1)
        
        trans_info_obsv={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right',\
            'batch_seq_lambda','batch_goal_idx', 'flatten_incam_R_goal2start_left','flatten_incam_t_goal2start_left','flatten_incam_R_goal2start_right','flatten_incam_t_goal2start_right']:
            if f"resnet_{k}" in batch:
                trans_info_obsv[k]=batch[f"resnet_{k}"]
            
        return_results2=self.model_pblock.batch_seq_from_comp_to_joints(batch_seq_comp=clip_frame_penc_out_comp_obsv,
                                                batch_mean_hand_size=(clip_frame_mean_hand_left_size,clip_frame_mean_hand_right_size),
                                                trans_info=trans_info_obsv, 
                                                ensure_consistent_goal=True,
                                                normalize_size_from_comp=False,
                                                batch_seq_valid_features=None,
                                                verbose=verbose)

        return_results={}
        return_results["batch_seq_valid_frames_obsv"]=clip_frame_valid_frames

        for k in ["local","cam"]:
            return_results[f"batch_seq_joints3d_in_{k}_out_obsv"]=return_results2[f"batch_seq_joints3d_in_{k}"]/self.hand_scaling_factor
            shape1=return_results[f"batch_seq_joints3d_in_{k}_out_obsv"].shape
            return_results[f"batch_seq_joints3d_in_{k}_gt_obsv"]=batch[f"obsv_batch_clip_frame_joints3d_in_{k}_gt"].view(shape1)/self.hand_scaling_factor
            
            if verbose:
                print(k,torch.abs(return_results[f"batch_seq_joints3d_in_{k}_out_obsv"]-return_results[f"batch_seq_joints3d_in_{k}_gt_obsv"]).max())
        return return_results
        
    def get_decode_motion_info(self,batch,verbose=False):
        batch_flatten_hand,batch_flatten_hand_op={},{}
        batch_size,nclips,nframes_per_clip=batch["pred_clip_frame_valid_frame"].shape
        ntokens_obsv=torch.sum(batch["obsv_clip_valid_clip"],dim=(1,2))[0]
        if verbose:
            print("ntokens_obsv",ntokens_obsv)
            print(batch_size,nclips,nframes_per_clip)
        batch_flatten_hand["batch_last_obsv_valid_frame"]=batch["obsv_clip_frame_valid_frame"][:,ntokens_obsv-1].detach().clone()
        for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                    'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:
            name1=f"obsv_clip_frame_{name}_resnet" if self.pose_from_resnet else f"obsv_clip_frame_{name}"

            feature_olast_token=batch[name1][:,ntokens_obsv-1:ntokens_obsv].detach().clone()
            feature_op=torch.cat([feature_olast_token,batch[f"pred_clip_frame_{name}"].detach().clone()],dim=1)

            if verbose:
                print("feature_op",name,feature_op.shape)

            batch_flatten_hand_op[name]=torch.flatten(feature_op,start_dim=0,end_dim=2)
            batch_flatten_hand[name]=torch.flatten(batch[f"pred_clip_frame_{name}"],0,2)


        len_seq_op=(nclips+1)*nframes_per_clip
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand_op, 
                                        len_seq=len_seq_op, 
                                        spacing=self.model_pblock.spacing,
                                        base_frame_id=self.model_pblock.ntokens_obsv-1, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,#mask joints because need to feed to decoder
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=verbose)
                                        
        for k,v in hand_gts.items():
            #batch_clip_frame_v=v.view((batch_size,nclips+1,nframes_per_clip)+v.shape[1:])
            batch_flatten_hand[k]=v#torch.flatten(batch_clip_frame_v[:,1:].contiguous().clone(),0,2)
            if verbose:
                print("hand_gts",k,batch_clip_frame_v.shape, batch_flatten_hand[k].shape)

        batch_clip_frame_comp=flatten_comps["gt"].view((batch_size,nclips+1,nframes_per_clip)+flatten_comps["gt"].shape[1:])
        batch_flatten_hand["flatten_hand_comp_gt"]=flatten_comps["gt"][:,:self.model_pblock.dim_hand_feature]#=torch.flatten(batch_clip_frame_comp[:,1:].contiguous().clone(),2)
        batch_flatten_hand["batch_last_obsv_hand_comp"]=batch_clip_frame_comp[:,0].contiguous().clone()

        if verbose:
            print("===============")
            for k,v in batch_flatten_hand.items():
                print("batch_flatten_hand-",k,v.shape)
        if "pred_clip_frame_image_vis" in batch:
            batch_flatten_hand["batch_seq_image_vis_pred"]=torch.flatten(batch["pred_clip_frame_image_vis"],1,2)
        return batch_flatten_hand

    def forward(self, batch, is_train, to_reparameterize, gt_action_for_dec, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results,losses = {},{}
        batch0_super,batch0_poenc_for_super=self.get_inputs_feature(batch,verbose=verbose)


        #And now A-Block
        num_videos=len(batch0_super['obsv_batch_action_name'])
        if self.obj_from_resnet:
            clip_otokens=self.model_poenc.feed_encoder(batch0_poenc_for_super,is_train,verbose=verbose)
            #quantized
            obj_embedding=self.object_embedding.detach().clone()
            _, obj_idx_out=embedding_lookup(query=clip_otokens,embedding=obj_embedding, verbose=verbose)#
            clip_otokens=torch_f.embedding(obj_idx_out, obj_embedding.transpose(0, 1))
            batch0_super["obsv_batch_clip_otokens"]=clip_otokens.view(num_videos,self.ntokens_obsv,clip_otokens.shape[-1])#
        else:
            batch0_super["obsv_batch_clip_otokens"]=batch0_super["obsv_batch_clip_otokens_gt"]

        batch_pdec_gt= super().get_decode_motion_info(batch,verbose=verbose) if self.gt_ite0 else self.get_decode_motion_info(batch,verbose=verbose)
        if verbose:
            print("check consistency with super class")
            batch2_super=super().get_inputs_feature(batch,verbose=False)
            for k,v in batch0_super.items():
                print(k)
                if "name" not in k and k in batch2_super:
                    print(batch2_super[k].shape)
                    print(batch0_super[k].shape)
                    if k in ["pred_batch_clip_frame_hand_comp"]:
                        print(torch.abs(batch2_super[k][:,:-1]-batch0_super[k]).max())
                    else:
                        print(torch.abs(batch2_super[k]-batch0_super[k]).max())
        
        loss_super,results_super,losses_super=super().forward(batch0_super,is_train,
                                        to_reparameterize=to_reparameterize,gt_action_for_dec=gt_action_for_dec,
                                        batch_pdec_gt=batch_pdec_gt,verbose=verbose)
        total_loss+=loss_super
        losses.update(losses_super)
        results.update(results_super)
        if not is_train:
            results_obsv=self.compute_pose_estimation(batch0_super,results_super,verbose=False)
            for k,v in results_obsv.items():
                results["est_"+k]=v
                
        return total_loss,results,losses
    