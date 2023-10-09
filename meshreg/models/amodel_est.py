import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np
import copy

from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt
from meshreg.models.utils import compute_bert_embedding_for_taxonomy,compute_berts_for_strs, embedding_lookup
from meshreg.models.utils import align_torch_batch_for_cam2local,recover_3d_proj_pinhole_wo_rescale,augment_hand_pose_2_5D
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


        self.pose_from_resnet=True
        self.obj_from_resnet=False

    
    def assign_ibe(self,model_ibe):
        self.model_ibe=model_ibe
        self.model_ibe.eval()
    
    def assign_poenc(self,model_poenc):
        self.model_poenc=model_poenc
        self.model_poenc.eval()    

    def augment_hand_pose_from_2_5D(self,batch_clip_frame_resnet25d,batch_clip_frame_cam_intr,verbose):
        return_batch={}
        if verbose:
            print("batch_clip_frame_resnet25d",batch_clip_frame_resnet25d.shape)
            print("batch_clip_frame_cam_intr",batch_clip_frame_cam_intr.shape)
        
        flatten_resnet25d=torch.flatten(batch_clip_frame_resnet25d.clone().detach(),0,2)
        #flatten_resnet25d=augment_hand_pose_2_5D(flatten_resnet25d)##############################
        return_batch["resnet25d_aug"]=flatten_resnet25d

        flatten_cam_intr=torch.flatten(batch_clip_frame_cam_intr,0,2)
        flatten_cam_joints3d=recover_3d_proj_pinhole_wo_rescale(flatten_cam_intr,flatten_resnet25d[:,:,:2],flatten_resnet25d[:,:,2:3],verbose=verbose)
        return_batch["cam_joints3d_left"]=flatten_cam_joints3d[:,:21]
        return_batch["cam_joints3d_right"]=flatten_cam_joints3d[:,21:]

        flatten_cam_joints={tag:return_batch[f"cam_joints3d_{tag}"].double() for tag in ["left","right"]}
        flatten_mano_palm_joints={tag:self.mean_mano_palm_joints[tag].detach().clone() for tag in ["left","right"]}
        align_results=align_torch_batch_for_cam2local(flatten_cam_joints, flatten_mano_palm_joints,palm_joints=[0,5,9,13,17])
        return_batch.update(align_results)

        return return_batch


    def get_inputs_feature_pblock(self,batch,is_train,return_batch_ablock=None,verbose=False):
        return_batch_pblock={}
        if return_batch_ablock is None:
            return_batch_ablock={}#get hand size, and hand comp for A-Block
        if verbose:
            for key, value in batch.items():
                try:
                    print(key,value.shape)
                except:
                    print(key,len(value))
        
        #obsv video-wise action name for A-Enc GT
        num_videos=len(batch['obsv_clip_action_name'])  
        for label in ["obsv","pred"]:
            #Process for P-Block
            name="valid_frame"
            value=batch[f"{label}_clip_frame_{name}"].cuda()
            nclips_per_part,nframes_per_clip=value.shape[1:3]
            return_batch_pblock[f"{label}_batch_seq_{name}"]=torch.flatten(torch.cat([value[:,:-1].detach().clone(),value[:,1:].detach().clone()],dim=2),start_dim=0,end_dim=1)
            if verbose:
                print("num_videos,nclips_per_part,nframes_per_clip",num_videos,nclips_per_part,nframes_per_clip)
                print("**value**",name,value.shape)

            '''
            for name in ['image_vis']:                        
                value=batch[f"{label}_clip_frame_{name}"].cuda()
                feature_enc,feature_dec=value[:,:-1].detach().clone(),value[:,1:].detach().clone()
                feature_enc_dec=torch.cat([feature_enc,feature_dec],dim=2)
                return_batch_pblock[f"{label}_{name}"]=torch.flatten(feature_enc_dec,start_dim=0,end_dim=1)
            '''

            #GT supervision
            batch_flatten_hand={}
            for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                        'hand_size_left','hand_size_right','valid_joints_left','valid_joints_right']:            
                value=batch[f"{label}_clip_frame_{name}"].cuda()
                if verbose:
                    print("**value**",name, value.shape)

                feature_enc=value[:,:-1].detach().clone()
                feature_dec=value[:,1:].detach().clone()
                feature_enc_dec=torch.cat([feature_enc,feature_dec],dim=2)
                batch_flatten_hand[name]=torch.flatten(feature_enc_dec,start_dim=0,end_dim=2)
            
            if verbose:
                print("GT batch_flatten_hand")
                for k,v in batch_flatten_hand.items():
                    print(k,v.shape)
                print("========")

            flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=self.model_pblock.ntokens_op, 
                                        spacing=self.model_pblock.spacing,
                                        base_frame_id=self.model_pblock.base_frame_id, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=False)
            
            return_batch_pblock[f"{label}_batch_seq_hand_comp_gt"]=flatten_comps["gt"].view((-1,self.model_pblock.ntokens_op)+flatten_comps["gt"].shape[1:])
            return_batch_pblock[f"{label}_batch_seq_valid_features"]=hand_gts["flatten_valid_features"].view((-1,self.model_pblock.ntokens_op)+hand_gts["flatten_valid_features"].shape[1:])

            for k,v in hand_gts.items():
                return_batch_pblock[f"{label}_{k}"]=v.view((-1,self.model_pblock.ntokens_op)+v.shape[1:])

            if label=="pred":
                shape1=(num_videos,nclips_per_part-1,nframes_per_clip)+return_batch_pblock["pred_batch_seq_hand_comp_gt"].shape[2:]
                return_batch_ablock["pred_batch_clip_frame_hand_comp"]=return_batch_pblock["pred_batch_seq_hand_comp_gt"][:,:self.model_pblock.ntokens_obsv].view(shape1)

            #rearrange for A-Block GT
            for camtag in ["cam","local"]:
                shape1=(num_videos,nclips_per_part-1,self.model_pblock.ntokens_op)
                batch_clip_frame_joints3d=hand_gts[f"flatten_joints3d_in_{camtag}_gt"].view(shape1+hand_gts[f"flatten_joints3d_in_{camtag}_gt"].shape[1:])
                if verbose:
                    print("batch_clip_frame_joints3d",batch_clip_frame_joints3d.shape)
                batch_clip_frame_joints3d=torch.cat([batch_clip_frame_joints3d[:,:,:self.model_pblock.ntokens_obsv],batch_clip_frame_joints3d[:,-1:,self.model_pblock.ntokens_obsv:]],dim=1)
                if verbose:
                    print("concat",batch_clip_frame_joints3d.shape)
                shape2=(num_videos,nclips_per_part,nframes_per_clip)
                return_batch_ablock[f"{label}_batch_clip_frame_joints3d_in_{camtag}_gt"]=batch_clip_frame_joints3d.view(shape2+batch_clip_frame_joints3d.shape[3:])
                
            #Resnet feature
            if is_train and self.pose_from_resnet:
                batch_flatten_hand=self.augment_hand_pose_from_2_5D(batch[f"{label}_clip_frame_joints25d_resnet"].cuda(),
                                batch[f"{label}_clip_frame_ncam_intr"].cuda(),verbose=verbose)#
            else:
                batch_flatten_hand={}
                for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                            'hand_size_left','hand_size_right']:
                    name1=f"{label}_clip_frame_{name}_resnet" if self.pose_from_resnet else f"{label}_clip_frame_{name}"
                    batch_flatten_hand[name]=torch.flatten(batch[name1],start_dim=0,end_dim=2)#
                
            for name in ["valid_joints_left","valid_joints_right"]:
                batch_flatten_hand[name]=torch.flatten(batch[f"{label}_clip_frame_{name}"],start_dim=0,end_dim=2)
            
            if verbose:
                print("Resnet batch_flatten_hand")
                for k,v in batch_flatten_hand.items():
                    print(k,v.shape)
                print("========")
                for key in batch_flatten_hand0.keys():
                    print(key,torch.abs(batch_flatten_hand0[key]-batch_flatten_hand[key].cuda()).max())
                '''
                links=[(0, 2, 3, 4),(0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]
                tag="h2o_cam4"
                cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
                batch_seq_joints3d_cam=torch.cat([batch_flatten_hand["cam_joints3d_left"],batch_flatten_hand["cam_joints3d_right"]],dim=1).view(num_videos,-1,42,3)
                flatten_image_vis=torch.flatten(batch[f"{label}_clip_frame_image_vis"],start_dim=0,end_dim=2)
                for sample_id in range(0,7): 
                    sample_vis_l2r(batch_seq_gt_cam=batch_seq_joints3d_cam, 
                                    batch_seq_gt_local=batch_seq_joints3d_cam,
                                    joint_links=links,  
                                    flatten_imgs=flatten_image_vis,
                                    sample_id=sample_id,
                                    cam_info=cam_info,
                                    prefix_cache_img=f"./vis_v3/imgs_{tag}/", path_video="./vis_v3/{:s}_{:02d}.avi".format(tag,sample_id))
                exit(0)
                '''
                
            flatten_comps, hands_resnet = get_flatten_hand_feature(batch_flatten_hand, 
                                        len_seq=self.model_pblock.ntokens_obsv, 
                                        spacing=self.model_pblock.spacing,
                                        base_frame_id=0, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False, verbose=verbose and False)

            shape2=(num_videos, nclips_per_part, nframes_per_clip)
            batch_clip_frame_hand_comp=flatten_comps["gt"].view(shape2+flatten_comps["gt"].shape[1:])
            return_batch_pblock[f"{label}_batch_seq_hand_comp_obsv"]=torch.flatten(batch_clip_frame_hand_comp[:,:-1],start_dim=0,end_dim=1)
            if label=="obsv":
                return_batch_ablock["obsv_batch_clip_frame_hand_comp"]=batch_clip_frame_hand_comp
                for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
                    return_batch_ablock[f"resnet_{k}"]=hands_resnet[k].detach().clone()

            if verbose:
                print("check with gt rather than resnet")
                print("hand comp",torch.abs(return_batch_pblock[f"{label}_batch_seq_hand_comp_obsv"].view(num_videos,nclips_per_part-1,nframes_per_clip,153)-return_batch_pblock[f"{label}_batch_seq_hand_comp_gt"].view(num_videos,nclips_per_part-1,self.model_pblock.ntokens_op,153)[:,:,:nframes_per_clip]).max())

                if label=="obsv":
                    print(torch.abs(return_batch_ablock[f"{label}_batch_clip_frame_hand_comp"][:,1:,1:]-return_batch_pblock[f"{label}_batch_seq_hand_comp_gt"].view(num_videos,nclips_per_part-1,self.model_pblock.ntokens_op,153)[:,:,self.model_pblock.ntokens_obsv+1:]).max())
                else:
                    print(torch.abs(return_batch_ablock[f"{label}_batch_clip_frame_hand_comp"]-return_batch_pblock[f"{label}_batch_seq_hand_comp_gt"].view(num_videos,nclips_per_part-1,self.model_pblock.ntokens_op,153)[:,:,:self.model_pblock.ntokens_obsv]).max())

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

        #Aggregate batch0 for P-Block
        return_batch_pblock1={}
        for key in ["batch_seq_valid_frame","batch_seq_valid_features","batch_seq_hand_comp_gt","batch_seq_hand_comp_obsv",
            "flatten_firstclip_R_base2cam_left","flatten_firstclip_t_base2cam_left", "flatten_firstclip_R_base2cam_right","flatten_firstclip_t_base2cam_right",
            "flatten_R_local2base_left","flatten_t_local2base_left","flatten_R_local2base_right","flatten_t_local2base_right",
            "flatten_joints3d_in_cam_gt","flatten_joints3d_in_local_gt","flatten_joints3d_in_base_gt","flatten_local2base_gt"]:#,"image_vis",]:
            return_batch_pblock1[key]=torch.cat([return_batch_pblock["obsv_"+key],return_batch_pblock["pred_"+key]],dim=0)
        
        return_batch_pblock1["batch_hand_size_left"]=repeat(batch_mean_hand_left_size.view(-1,1,1),
                        'b () () -> b n m',n=self.ntokens_obsv+self.ntokens_pred-1,m=self.model_pblock.ntokens_op).reshape(-1,self.model_pblock.ntokens_op)
        return_batch_pblock1["batch_hand_size_right"]=repeat(batch_mean_hand_right_size.view(-1,1,1),
                        'b () () -> b n m',n=self.ntokens_obsv+self.ntokens_pred-1,m=self.model_pblock.ntokens_op).reshape(-1,self.model_pblock.ntokens_op)
        

        #Get only consecutive frames
        pblock_valid_clips_idx=torch.where(torch.sum(return_batch_pblock1["batch_seq_valid_frame"],dim=1)>self.model_pblock.ntokens_obsv)[0]
        if verbose:
            print("pblock_valid_clips_idx",pblock_valid_clips_idx)
        for k,v in return_batch_pblock1.items():
            return_batch_pblock1[k]=return_batch_pblock1[k][pblock_valid_clips_idx].contiguous()
            if verbose:
                print(k,return_batch_pblock1[k].shape)
            if "flatten_" in k or k=="image_vis":
                return_batch_pblock1[k]=torch.flatten(return_batch_pblock1[k],0,1)
                
        return_batch_pblock1["valid_frame"]=torch.flatten(return_batch_pblock1["batch_seq_valid_frame"])
        return_batch_pblock1["hand_size_left"]=torch.flatten(return_batch_pblock1["batch_hand_size_left"])
        return_batch_pblock1["hand_size_right"]=torch.flatten(return_batch_pblock1["batch_hand_size_right"])
        return_batch_pblock1.pop("batch_seq_valid_frame")
        return_batch_pblock1.pop("batch_hand_size_left")
        return_batch_pblock1.pop("batch_hand_size_right")

        if verbose:
            print("return_batch_pblock1")
            for k,v in return_batch_pblock1.items():
                print(k,v.shape)                
        return return_batch_pblock1, return_batch_ablock
        

    def get_inputs_feature_poenc(self,batch,return_batch_ablock=None,verbose=False):
        return_batch_poenc,return_batch_poenc_for_super={},{}
        if return_batch_ablock is None:
            return_batch_ablock={}
        if verbose:
            for key, value in batch.items():
                try:
                    print(key,value.shape)
                except:
                    print(key,len(value))
    
        #obsv video-wise action name for A-Enc GT
        num_videos=len(batch['obsv_clip_action_name'])
        for label in ["obsv","pred"]:
            return_batch_poenc[f"{label}_obj_name"]=[]
            for item in batch[f"{label}_clip_obj_name"]:
                return_batch_poenc[f"{label}_obj_name"]+=copy.deepcopy(item.split("@"))
            clip_otokens_obsv=compute_berts_for_strs(self.model_bert,return_batch_poenc[f"{label}_obj_name"],verbose)
            return_batch_poenc[f"{label}_otokens_gt"]=clip_otokens_obsv.view(-1,clip_otokens_obsv.shape[-1]).detach().clone()
            if label=="obsv":
                return_batch_ablock["obsv_batch_clip_otokens_gt"]=clip_otokens_obsv.view(num_videos,self.ntokens_obsv,clip_otokens_obsv.shape[-1])

            for name in ["object_feature","valid_frame"]:
                return_batch_poenc[f"{label}_batch_seq_{name}"]=torch.flatten(batch[f"{label}_clip_frame_{name}"].cuda(),0,1)
                if label=="obsv":
                    return_batch_poenc_for_super[name]=torch.flatten(batch[f"{label}_clip_frame_{name}"].cuda(),0,2)
            return_batch_poenc[f"{label}_valid_clip"]=torch.flatten(batch[f"{label}_clip_valid_clip"].cuda())
            if verbose:
                print("check consistency")
                print((torch.where(torch.sum(return_batch_poenc[f"{label}_batch_seq_valid_frame"],dim=1)>0,1,0)==return_batch_poenc[f"{label}_valid_clip"]).all())
               
        #Aggregate batch0 for PO-Enc
        return_batch_poenc1={}
        for key in ["batch_seq_valid_frame","batch_seq_object_feature","otokens_gt","valid_clip"]:
            return_batch_poenc1[key]=torch.cat([return_batch_poenc["obsv_"+key],return_batch_poenc["pred_"+key]],dim=0)
        for key in ["obj_name"]:
            return_batch_poenc1[key]=return_batch_poenc["obsv_"+key]+return_batch_poenc["pred_"+key]
        
        poenc_valid_clips_idx=torch.where(return_batch_poenc1["valid_clip"]>0)[0]
        for key,v in return_batch_poenc1.items():
            if key in ["obj_name"]:
                return_batch_poenc1[key]=[return_batch_poenc1[key][i] for i in poenc_valid_clips_idx]
            else:
                return_batch_poenc1[key]=return_batch_poenc1[key][poenc_valid_clips_idx].contiguous()
        for key in ["valid_frame","object_feature"]:
            return_batch_poenc1[key]=torch.flatten(return_batch_poenc1[f"batch_seq_{key}"],0,1)
            return_batch_poenc1.pop(f"batch_seq_{key}")

        #And finally, get obj_idx_gt
        return_batch_poenc1["obj_idx_gt"]=torch.zeros(len(return_batch_poenc1["obj_name"]),dtype=torch.int64).cuda()
        for vid, vname in enumerate(return_batch_poenc1["obj_name"]):
            return_batch_poenc1['obj_idx_gt'][vid]=self.object_name2idx[vname] if vname!="NIL" else 0
            assert vname!="NIL"
        return_batch_poenc1["object_embedding"]=self.object_embedding

        if verbose:
            print("return_batch_poenc1")
            for k,v in return_batch_poenc1.items():
                try:
                    print(k,v.shape)
                except:
                    print(k,len(v))
        return return_batch_poenc1, return_batch_poenc_for_super, return_batch_ablock


    def get_inputs_feature_super(self,batch,return_batch_ablock=None,verbose=False):        
        #obsv video-wise action name        
        num_videos=len(batch['obsv_clip_action_name'])
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

        if verbose:
            print("return_batch_ablock")
            for k,v in return_batch_ablock.items():
                try:
                    print(k,v.shape)
                except:
                    print("**list**",k,len(v))                    
        return return_batch_ablock


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
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info_obsv[k]=batch[f"resnet_{k}"]
            
        return_results2=self.model_pblock.batch_seq_from_comp_to_joints(batch_seq_comp=clip_frame_penc_out_comp_obsv,
                                                batch_mean_hand_size=(clip_frame_mean_hand_left_size,clip_frame_mean_hand_right_size),
                                                trans_info=trans_info_obsv, 
                                                normalize_size_from_comp=False,
                                                batch_seq_valid_features=None, verbose=verbose)

        return_results={}
        return_results["batch_seq_valid_frames_obsv"]=clip_frame_valid_frames

        for k in ["local","cam"]:
            return_results[f"batch_seq_joints3d_in_{k}_out_obsv"]=return_results2[f"batch_seq_joints3d_in_{k}"]/self.hand_scaling_factor
            shape1=return_results[f"batch_seq_joints3d_in_{k}_out_obsv"].shape
            return_results[f"batch_seq_joints3d_in_{k}_gt_obsv"]=batch[f"obsv_batch_clip_frame_joints3d_in_{k}_gt"].view(shape1)/self.hand_scaling_factor
            
            if verbose:
                print(k,torch.abs(return_results[f"batch_seq_joints3d_in_{k}_out_obsv"]-return_results[f"batch_seq_joints3d_in_{k}_gt_obsv"]).max())
        return return_results
        

    def forward(self, batch, is_train, to_reparameterize, gt_action_for_dec, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results,losses = {},{}
        batch0_pblock,batch0_super=self.get_inputs_feature_pblock(batch,is_train=is_train,verbose=verbose)
        batch0_poenc,batch0_poenc_for_super,batch0_super=self.get_inputs_feature_poenc(batch,batch0_super,verbose=verbose)
        batch0_super=self.get_inputs_feature_super(batch,batch0_super,verbose=verbose)
        
        #first valid clip of obsv/pred clips of A, pass to train P
        if batch0_pblock["valid_frame"].shape[0]>0:
            loss_pblock,results_pblock,losses_pblock=self.model_pblock(batch0_pblock,is_train,verbose=verbose)
            total_loss+=loss_pblock
            for k,v in losses_pblock.items():
                losses[k+"_pblock"]=v
            results.update(results_pblock)

        #then valid clip of obsv/pred clips of A, pass to train PO
        _, results_poenc, losses_poenc=self.model_poenc(batch0_poenc,is_train,verbose=verbose)
        if self.lambda_action is not None and self.lambda_action>1e-6:
            total_loss+=self.lambda_action*(losses_poenc["obj_dist_loss"]+losses_poenc["obj_contrast_loss"])
        losses.update(losses_poenc)
        results.update(results_poenc)

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

        batch_pdec_gt=None if is_train else self.get_pblock_concat_inout_seq(batch,"pred",verbose=verbose)
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
    