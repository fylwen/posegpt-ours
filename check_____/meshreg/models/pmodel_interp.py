import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np

from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import  loss_str2func,get_flatten_hand_feature, get_inverse_Rt, from_comp_to_joints
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.pmodel import MotionNet as MotionNet_


class MotionNet(MotionNet_):
    def __init__(self,  transformer_d_model,
                    transformer_nhead,
                    transformer_dim_feedforward,
                    transformer_nlayers_enc,
                    transformer_nlayers_dec,
                    transformer_activation,
                    ntokens_per_clip,
                    spacing,
                    num_iterations=1,
                    lambda_clustering=None,
                    lambda_hand=None,
                    pose_loss='l1',
                    code_loss='l1',):

        super().__init__(transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_nlayers_dec,
                        transformer_activation,
                        ntokens_per_clip,
                        spacing,
                        lambda_clustering,
                        lambda_hand,
                        pose_loss,
                        code_loss,)
        self.num_segs=3
        self.num_iterations=1
        self.ntokens_op=self.ntokens_obsv+self.ntokens_pred*3
        self.dev_id=15#self.ntokens_obsv//2
        self.base_frame_id=self.ntokens_obsv-1
    

    def get_gt_inputs_feature(self,batch_flatten,verbose=False):           
        if not "batch_action_name_obsv" in batch_flatten.keys():
            batch_flatten["batch_action_name_obsv"]=batch_flatten["action_name"][self.ntokens_obsv-1::self.ntokens_op]
        
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten, 
                                        len_seq=self.ntokens_op, 
                                        base_frame_id=self.base_frame_id,
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False,
                                        verbose=verbose)
                                        
        batch_seq_hand_comp_gt = flatten_comps["gt"].view(-1,self.ntokens_op, self.dim_hand_feature)
        batch_seq_valid_features=hand_gts["flatten_valid_features"].view(-1,self.ntokens_op,self.dim_hand_feature)
        if not "batch_seq_hand_comp_gt" in batch_flatten.keys():
            batch_flatten["batch_seq_hand_comp_gt"]=batch_seq_hand_comp_gt
            batch_flatten["batch_seq_valid_features"]=batch_seq_valid_features
            
        return hand_gts


        
    def forward(self, batch_flatten, to_reparameterize, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results = {}
        losses = {} 
        hand_gts=self.get_gt_inputs_feature(batch_flatten, verbose=verbose)        
        results.update(hand_gts)

        #lets start
        batch_size=batch_flatten["batch_seq_hand_comp_gt"].shape[0]
        batch_seq_valid_frames=batch_flatten["valid_frame"].bool().cuda().reshape((-1,self.ntokens_op))

        batch_seq_hand_comp_obsv=[]
        batch_seq_obsv_mask=[]

        for cid in range(0,self.num_segs):
            batch_seq_hand_comp_obsv.append(torch.unsqueeze(batch_flatten["batch_seq_hand_comp_gt"][:,cid*self.dev_id:cid*self.dev_id+self.ntokens_obsv].clone(),1))
            batch_seq_obsv_mask.append(torch.unsqueeze(batch_seq_valid_frames[:,cid*self.dev_id:cid*self.dev_id+self.ntokens_obsv].clone(),1))
        
        batch_seq_hand_comp_obsv=torch.cat(batch_seq_hand_comp_obsv,dim=1)
        batch_seq_obsv_mask=~torch.cat(batch_seq_obsv_mask,dim=1)
        if verbose:
            print("batch_seq_hand_comp_obsv",batch_seq_hand_comp_obsv.shape,batch_seq_obsv_mask.shape)#[bs,3, len_o,144],[bs,3,len_o]
        
        batch_seq_hand_comp_obsv=torch.flatten(batch_seq_hand_comp_obsv,0,1)
        batch_seq_obsv_mask=torch.flatten(batch_seq_obsv_mask,0,1)        
        
        batch_mid_mu_enc_out, _,_, batch_seq_hand_comp_enc_out = self.feed_encoder(batch_seq_hand_comp_obsv,batch_seq_obsv_mask, verbose=verbose)
        batch_seq_hand_comp_enc_out_obsv = batch_seq_hand_comp_enc_out.view(batch_size,self.num_segs,self.ntokens_obsv, self.dim_hand_feature)[:,0]
        if verbose:
            print("batch_seq_hand_comp_enc_out_obsv",batch_seq_hand_comp_enc_out_obsv.shape)#[bs,len_o,144]


        #First process GT
        batch_mean_hand_left_size=torch.mean(batch_flatten['hand_size_left'].view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv],dim=1,keepdim=True)
        batch_mean_hand_right_size=torch.mean(batch_flatten['hand_size_right'].view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv],dim=1,keepdim=True)

        for key in ["local","cam"]:
            batch_seq_gt=results[f"flatten_joints3d_in_{key}_gt"].view(batch_size,self.ntokens_op,self.num_joints,3)/self.hand_scaling_factor 
            batch_seq_gt=batch_seq_gt[:,:self.ntokens_obsv+self.ntokens_pred+2*self.dev_id]
            results[f"batch_seq_joints3d_in_{key}_gt"]=batch_seq_gt
            results[f"batch_seq_joints3d_in_{key}_obsv_gt"]=batch_seq_gt[:,:self.ntokens_obsv]
            results[f"batch_seq_joints3d_in_{key}_pred_gt"]=batch_seq_gt[:,self.ntokens_obsv:]
        
        batch_flatten["image"]=batch_flatten["image"].view((batch_size,self.ntokens_op)+batch_flatten["image"].shape[-3:])[:,:self.ntokens_obsv+self.ntokens_pred+2*self.dev_id].clone()
        batch_flatten["image"]=torch.flatten(batch_flatten["image"],0,1)
        
        #pass the pose->mid(Pe)->decoder
        batch_seq_dec_mem=batch_mid_mu_enc_out[:,0:1]
        batch_seq_dec_mem_mask=torch.zeros_like(batch_seq_obsv_mask[:,:1]).bool()    
        
        if verbose:
            print("batch_seq_dec_mem/batch_seq_dec_mem_mask",batch_seq_dec_mem.shape,batch_seq_dec_mem_mask.shape)#[bs*3,1,512],[bs*3,1]
        _, batch_seq_hand_comp_dec_out=self.feed_decoder(batch_seq_dec_mem,batch_seq_dec_mem_mask,verbose)
        batch_seq_hand_comp_dec_out=batch_seq_hand_comp_dec_out.view(batch_size,3,self.ntokens_pred,-1)

        #batch_seq_hand_comp_dec_out1=torch.cat([batch_seq_hand_comp_dec_out[:,0,:self.dev_id].clone(),
        #                                        batch_seq_hand_comp_dec_out[:,1].clone(),
        #                                        batch_seq_hand_comp_dec_out[:,2,-self.dev_id:].clone()],dim=1)
        

        #pass the interpolated mid(Pe)->decoder
        batch_seq_dec_mem2=batch_mid_mu_enc_out[:,0:1].view(batch_size,self.num_segs,1,self.code_dim)
        
        if verbose:
            print("batch_seq_dec_mem2",batch_seq_dec_mem2.shape)#[bs,3,1,512]
        #Warning！！！！
        lambda_1=0.5#self.dev_id/float(self.ntokens_obsv)
        lambda_0=1-lambda_1
        batch_seq_dec_mem2=lambda_0*batch_seq_dec_mem2[:,0]+lambda_1*batch_seq_dec_mem2[:,2]
        batch_seq_dec_mem_mask2=torch.zeros(batch_size,1,dtype=batch_seq_dec_mem_mask.dtype,device=batch_seq_dec_mem_mask.device)
        if verbose:
            print("batch_seq_dec_mem2/batch_seq_dec_mem_mask2",batch_seq_dec_mem2.shape,batch_seq_dec_mem_mask2.shape)#[bs,1,512],[bs,1]
        _, batch_seq_hand_comp_dec_out2=self.feed_decoder(batch_seq_dec_mem2,batch_seq_dec_mem_mask2,verbose)


        batch_seq_hand_comp_dec_out2=torch.cat([batch_seq_hand_comp_dec_out[:,0,:self.dev_id].clone(),
                                            batch_seq_hand_comp_dec_out2.clone(),
                                            batch_seq_hand_comp_dec_out[:,2,-self.dev_id:].clone()],dim=1)        
        if verbose:
            print('batch_seq_hand_comp_dec_out_1/_2',batch_seq_hand_comp_dec_out1.shape,batch_seq_hand_comp_dec_out2.shape)
            #[bs,len_p+2*dev,144]x2
        
        trans_info={}
        nbase_idx=self.ntokens_obsv-1
        for hand_tag in ["left","right"]:
            R_cam2base=batch_flatten[f"R_cam2local_{hand_tag}"].view(batch_size,-1,3,3)[:,nbase_idx]
            t_cam2base=self.hand_scaling_factor*batch_flatten[f"t_cam2local_{hand_tag}"].view(batch_size,-1,1,3)[:,nbase_idx]
            R_base2cam, t_base2cam=get_inverse_Rt(R_cam2base,t_cam2base)
            trans_info[f'flatten_firstclip_R_base2cam_{hand_tag}']=R_base2cam
            trans_info[f'flatten_firstclip_t_base2cam_{hand_tag}']=t_cbase2cam

            
        results_hand=self.batch_seq_from_comp_to_joints(batch_seq_comp=batch_seq_hand_comp_dec_out2,
                                            batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                            trans_info=trans_info,
                                            normalize_size_from_comp=False,verbose=verbose)
                                            
        for key in ["local","cam"]:
            batch_seq_out=results_hand[f"batch_seq_joints3d_in_{key}"]/self.hand_scaling_factor
            results[f"batch_seq_joints3d_in_{key}_pred_out"]=batch_seq_out#[:,self.ntokens_obsv:]
            results[f"batch_seq_joints3d_in_{key}_out"]=batch_seq_out 
            #print(key,results[f"batch_seq_joints3d_in_{key}_pred_out"]-results[f"batch_seq_joints3d_in_{key}_pred_gt"])
      
        if verbose:
            for k,v in results.items():
                print(k,v.shape) 
            exit(0)
        return total_loss,results,losses
