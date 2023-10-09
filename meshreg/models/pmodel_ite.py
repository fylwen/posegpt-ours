import torch
import torch.nn.functional as torch_f
from torch import nn
import numpy as np
torch.set_printoptions(threshold=np.inf)

from einops import repeat

from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import  loss_str2func,get_flatten_hand_feature, compose_Rt_a2b
from meshreg.models.utils import from_comp_to_joints, load_mano_mean_pose
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
      

        self.num_iterations=num_iterations
        self.ntokens_op=self.ntokens_obsv+self.ntokens_pred*self.num_iterations

        self.output_obsv= True and self.num_iterations==1
        self.base_frame_id=0 if self.output_obsv else self.ntokens_obsv-1

        
        self.use_resnet_input=False
        self.gt_ite0=True and self.num_iterations>1



    def update_with_resnet_obsv(self,batch_flatten,verbose=False):   
        return_batch={}
        for name in ["hand_size_left","hand_size_right"]:
            return_batch[name]=batch_flatten[name+"_resnet"]

        batch_flatten_resnet={}
        for name in ['cam_joints3d_left','cam_joints3d_right','R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right',\
                    'hand_size_left','hand_size_right']:                
            batch_seq_resnet_feature=batch_flatten[name+"_resnet"].view((-1,self.ntokens_op)+batch_flatten[name+"_resnet"].shape[1:])[:,:self.ntokens_obsv]
            batch_flatten_resnet[name]=torch.flatten(batch_seq_resnet_feature,0,1)
            
        for name in ["valid_joints_left","valid_joints_right"]:
            batch_flatten_resnet[name]=torch.flatten(batch_flatten[name].view((-1,self.ntokens_op)+batch_flatten[name].shape[1:])[:,:self.ntokens_obsv],0,1)
        
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten_resnet, 
                                        len_seq=self.ntokens_obsv, 
                                        spacing=self.spacing,
                                        base_frame_id=self.base_frame_id,
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=False,
                                        compute_local2first=False,#True,
                                        verbose=verbose)
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            return_batch[k]=hand_gts[k]
                 
        flatten_hand_comp=flatten_comps["gt"][:,:self.dim_hand_feature]
        batch_seq_hand_comp = flatten_hand_comp.view(-1,self.ntokens_obsv, self.dim_hand_feature)

        return_batch["batch_seq_hand_comp_obsv"] = batch_seq_hand_comp[:,:self.ntokens_obsv].clone()
        return return_batch
        
    def forward(self, batch_flatten, to_reparameterize, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results,losses={},{} 

        batch0=self.get_gt_inputs_feature(batch_flatten,is_train=False, verbose=verbose)

        if self.use_resnet_input:
            batch0_resnet=self.update_with_resnet_obsv(batch_flatten,verbose)
            batch0.update(batch0_resnet)

        results["batch_action_name_obsv"]=batch_flatten["action_name"][::self.ntokens_op]
        
        #lets start
        batch_size=batch0["batch_seq_hand_comp_obsv"].shape[0]
        trans_info={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info[k]=batch0[k]
        
        batch_seq_valid_frame=batch0["valid_frame"].cuda().view(batch_size,self.ntokens_op)
        batch0["batch_seq_valid_features"]=torch.mul(batch0["batch_seq_valid_features"],torch.unsqueeze(batch_seq_valid_frame,-1))
        results["batch_seq_valid_frame_out"]=batch_seq_valid_frame

        #compute hand size from observation
        batch_seq_valid_frame_obsv=batch_seq_valid_frame[:,:self.ntokens_obsv]
        batch_seq_hand_size_left=torch.mul(torch.flatten(batch0['hand_size_left'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_left_size=torch.sum(batch_seq_hand_size_left,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)

        batch_seq_hand_size_right=torch.mul(torch.flatten(batch0['hand_size_right'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_right_size=torch.sum(batch_seq_hand_size_right,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)

        if verbose:
            print("batch_seq_valid_frame_obsv",batch_seq_valid_frame_obsv.shape)#[bs,len_o]
            print("batch_seq_hand_size_left/batch_mean_hand_left_size",batch_seq_hand_size_left.shape,batch_mean_hand_left_size.shape)#[bs,len_o],[bs,1]
            print("batch_seq_hand_size_right/batch_mean_hand_right_size",batch_seq_hand_size_right.shape,batch_mean_hand_right_size.shape)#[bs,len_o],[bs,1])

        for ite_id in range(0,self.num_iterations):
            batch_seq_hand_comp_obsv = batch0["batch_seq_hand_comp_obsv"] if ite_id==0 else batch_seq_hand_comp_dec_out
            
            #set first frame velocity as identical mat.
            batch_seq_hand_comp_obsv[:,0,self.num_joints*3:self.num_joints*3+18]=batch0["batch_seq_hand_comp_obsv"][:,0,self.num_joints*3:self.num_joints*3+18].detach().clone()
            batch_seq_hand_comp_obsv=batch_seq_hand_comp_obsv.detach().clone()
            
            ##here simply set every token is visible
            batch_seq_obsv_mask=torch.zeros_like(batch_seq_valid_frame.bool())[:,:self.ntokens_obsv]
            if verbose:
                print(f"Ite #{ite_id}")
                
            batch_mid_mu_enc_out, batch_mid_logvar_enc_out,batch_seq_hand_feature_enc_out, batch_seq_hand_comp_enc_out = self.feed_encoder(batch_seq_hand_comp_obsv,batch_seq_obsv_mask, verbose=verbose)
            
            noise_factor=1.
            if to_reparameterize:
                batch_seq_dec_mem=self.reparameterize(mu=batch_mid_mu_enc_out[:,0:1],logvar=batch_mid_logvar_enc_out[:,0:1],factor=5.)
                
                #batch_seq_dec_mem=batch_mid_mu_enc_out[:,0:1]
                #noise_batch_seq_dec_mem=torch.randn_like(batch_seq_dec_mem)*noise_factor
                #batch_seq_dec_mem=batch_seq_dec_mem+noise_batch_seq_dec_mem
            else:
                batch_seq_dec_mem=batch_mid_mu_enc_out[:,0:1]
                
            if ite_id==0:
                results["batch_seq_dec_mem"]=batch_mid_mu_enc_out[:,0:1].detach().clone()           
            
            #batch_seq_dec_mem=np.load("./0069.npz")["batch_seq_dec_mem"]
            #batch_seq_dec_mem=torch.from_numpy(batch_seq_dec_mem).cuda()
            #batch_seq_dec_mem=batch_seq_dec_mem[6:7].repeat(8,1,1)
            
            batch_seq_dec_mem_mask=torch.zeros_like(batch_seq_obsv_mask[:,:1]).bool() 
            if True:
                batch_seq_dec_mem=torch.cat((batch_seq_dec_mem,batch_seq_hand_feature_enc_out),dim=1)            
                batch_seq_dec_mem_mask=torch.cat((batch_seq_dec_mem_mask,batch_seq_obsv_mask),dim=1)
            
            #print(batch_seq_dec_mem.shape)
            batch_seq_hand_feature_dec_out, batch_seq_hand_comp_dec_out=self.feed_decoder(batch_seq_dec_mem,batch_seq_dec_mem_mask,verbose)
            if verbose:
                print('batch_seq_hand_feature_dec_out/batch_seq_hand_comp_dec_out',batch_seq_hand_feature_dec_out.shape,batch_seq_hand_comp_dec_out.shape)
                #[bs,len_p,512][bs,len_p,144]
                
            ################   
            #if check gt, rmb to mask valid_features in batch_seq_from_comp_to_joints func
            #batch_seq_hand_comp_enc_out=batch0["batch_seq_hand_comp_obsv"]
            if self.gt_ite0 and ite_id==0:
                results_hand0,_=self.batch_seq_from_comp_to_joints(batch_seq_comp_out= batch_seq_hand_comp_dec_out,
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"][:,:self.ntokens_obsv],
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info, 
                                                nbase_idx=self.ntokens_pred-1, verbose=verbose)#                
                batch_seq_hand_comp_dec_out=results_hand0["batch_seq_comp_local_normalized"]

                results_hand,trans_info=self.batch_seq_from_comp_to_joints(batch_seq_comp_out=batch0["batch_seq_hand_comp_gt"][:,ite_id*self.ntokens_pred+self.ntokens_obsv:(ite_id+1)*self.ntokens_pred+self.ntokens_obsv],
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"][:,:self.ntokens_obsv],
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info, 
                                                nbase_idx=self.ntokens_pred-1, verbose=verbose)#
            else:
                results_hand,trans_info=self.batch_seq_from_comp_to_joints(batch_seq_comp_out=torch.cat((batch_seq_hand_comp_enc_out,batch_seq_hand_comp_dec_out),dim=1) if self.output_obsv else batch_seq_hand_comp_dec_out,
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"] if self.output_obsv else batch0["batch_seq_valid_features"][:,:self.ntokens_obsv],
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info, 
                                                nbase_idx=self.ntokens_pred-1, verbose=verbose)#                
                batch_seq_hand_comp_dec_out=results_hand["batch_seq_comp_local_normalized"]
                                                
            for key in ["local","cam"]:
                batch_seq_out=results_hand[f"batch_seq_joints3d_in_{key}"]/self.hand_scaling_factor
                batch_seq_pred_out=batch_seq_out[:,self.ntokens_obsv:] if self.output_obsv else batch_seq_out
                if ite_id==0:
                    batch_seq_gt=batch0[f"flatten_joints3d_in_{key}_gt"].view(batch_size,self.ntokens_op,self.num_joints,3)/self.hand_scaling_factor
                    results[f"batch_seq_joints3d_in_{key}_gt"]=batch_seq_gt 
                    results[f"batch_seq_joints3d_in_{key}_out"]=batch_seq_out
                    results[f"batch_seq_joints3d_in_{key}_obsv_gt"]=batch_seq_gt[:,:self.ntokens_obsv]
                    results[f"batch_seq_joints3d_in_{key}_pred_gt"]=batch_seq_gt[:,self.ntokens_obsv:]
                    results[f"batch_seq_joints3d_in_{key}_pred_out"]=batch_seq_pred_out
                else:
                    results[f"batch_seq_joints3d_in_{key}_pred_out"]=torch.cat([results[f"batch_seq_joints3d_in_{key}_pred_out"],batch_seq_pred_out],dim=1)
                    results[f"batch_seq_joints3d_in_{key}_out"]=torch.cat([results[f"batch_seq_joints3d_in_{key}_out"],batch_seq_pred_out],dim=1)

                #print(key,torch.abs(results[f"batch_seq_joints3d_in_{key}_gt"]-results[f"batch_seq_joints3d_in_{key}_out"]).max())
                #print(key,torch.abs(results[f"batch_seq_joints3d_in_{key}_pred_gt"][:,ite_id*15:ite_id*15+15]-results[f"batch_seq_joints3d_in_{key}_pred_out"][:,ite_id*15:ite_id*15+15]).max())
        if self.gt_ite0:
            for key in ["local","cam"]:
                results[f"batch_seq_joints3d_in_{key}_pred_gt"]=results[f"batch_seq_joints3d_in_{key}_pred_gt"][:,self.ntokens_pred:]
                results[f"batch_seq_joints3d_in_{key}_pred_out"]=results[f"batch_seq_joints3d_in_{key}_pred_out"][:,self.ntokens_pred:]
                results[f"batch_seq_joints3d_in_{key}_out"]=results[f"batch_seq_joints3d_in_{key}_out"][:,self.ntokens_pred:]
        if verbose:
            for k,v in results.items():
                print(k,v.shape) 
        return total_loss,results,losses
        
    def batch_seq_from_comp_to_joints(self, batch_seq_comp_out, batch_mean_hand_size, batch_seq_valid_features, trans_info, nbase_idx, verbose=False):   
        output_results=super().batch_seq_from_comp_to_joints(batch_seq_comp=batch_seq_comp_out,
                                                        batch_mean_hand_size=batch_mean_hand_size,
                                                        trans_info=trans_info,
                                                        normalize_size_from_comp=True,
                                                        batch_seq_valid_features=batch_seq_valid_features,
                                                        verbose=verbose,)

        batch_seq_trans_info=output_results['batch_seq_trans_info']        
        return_trans_info={}     
        for hand in ["left","right"]:
            R_nbase2cbase=batch_seq_trans_info[f"batch_seq_R_local2base_{hand}"][:,nbase_idx].clone()
            t_nbase2cbase=batch_seq_trans_info[f"batch_seq_t_local2base_{hand}"][:,nbase_idx].clone()
            R_cbase2cam=batch_seq_trans_info[f"batch_seq_R_base2cam_{hand}"][:,nbase_idx].clone()
            t_cbase2cam=batch_seq_trans_info[f"batch_seq_t_base2cam_{hand}"][:,nbase_idx].clone()

            R_nbase2cam, t_nbase2cam=compose_Rt_a2b(batch_R_c2a=R_nbase2cbase,batch_t_c2a=t_nbase2cbase,
                batch_R_c2b=R_cbase2cam,batch_t_c2b=t_cbase2cam,is_c2a=False)
            
            return_trans_info[f"batch_nextclip_R_base2cam_{hand}"]=R_nbase2cam
            return_trans_info[f"batch_nextclip_t_base2cam_{hand}"]=t_nbase2cam
        
        return output_results,return_trans_info
    
    