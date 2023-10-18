import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np


from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.utils import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.vae_block import VAE


from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_rotation_6d,axis_angle_to_matrix

class MotionNet(VAE):
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
                        pose_loss='l1',
                        code_loss='l1',):

        
        self.num_mids=1
        super().__init__(transformer_d_model=transformer_d_model,
                        transformer_nhead=transformer_nhead,
                        transformer_dim_feedforward=transformer_dim_feedforward,
                        transformer_nlayers_enc=transformer_nlayers_enc,
                        transformer_nlayers_dec=transformer_nlayers_dec,
                        transformer_activation=transformer_activation,
                        num_mids=self.num_mids)

        self.ntokens_obsv=ntokens_per_clip
        self.ntokens_pred=ntokens_per_clip
        self.ntokens_op=self.ntokens_obsv+self.ntokens_pred
        self.spacing=spacing
 
        self.base_frame_id=self.ntokens_obsv-1
        self.num_iterations=1
        
        self.mean_mano_palm_joints={'left': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT'))),
                'right': torch.from_numpy(load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT')))}
        
        self.num_joints=42
        self.placeholder_joints=torch.nn.Parameter(torch.randn(1,42,3))
        self.dim_hand_feature= 3*self.num_joints+9*3
        self.hand_scaling_factor=10
        
        self.pose3d_to_trsfm_in=nn.Linear(self.num_joints*3,transformer_d_model//2)
        self.globalRt_to_trsfm_in=nn.Linear(9*2,transformer_d_model//4)
        self.localL2R_to_trsfm_in=nn.Linear(9,transformer_d_model//4)
        
        self.token_out_to_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], 
                                out_dim=self.dim_hand_feature,
                                act_hidden='leakyrelu',act_final='none')  

        self.lambda_clustering=lambda_clustering
        self.lambda_hand=lambda_hand
        loss_str2func_=loss_str2func()
        self.pose_loss=loss_str2func_[pose_loss]

    def get_gt_inputs_feature(self,batch_flatten,is_train,verbose=False):   
        return_batch={}
        for key in ["valid_frame","hand_size_left","hand_size_right"]:
            return_batch[key]=batch_flatten[key]
        if not "batch_action_name_obsv" in batch_flatten.keys():
            return_batch["batch_action_name_obsv"]=batch_flatten["action_name"][self.ntokens_obsv-1::self.ntokens_op]
        else:
            return_batch["batch_action_name_obsv"]=batch_flatten["batch_action_name_obsv"]
            
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten, 
                                        len_seq=self.ntokens_op, 
                                        spacing=self.spacing,
                                        base_frame_id=self.base_frame_id,
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.placeholder_joints,
                                        with_augmentation=is_train,
                                        compute_local2first=False,#True,
                                        verbose=verbose)
                 
        flatten_hand_comp_gt=flatten_comps["gt"][:,:self.dim_hand_feature]
        if verbose and False:
            flatten_comps2, hand_gts2 = get_flatten_hand_feature(batch_flatten, 
                                        len_seq=self.ntokens_op, 
                                        spacing=self.spacing,
                                        base_frame_id=0, 
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=None,
                                        with_augmentation=False,
                                        compute_local2first=True,
                                        verbose=True)     
            flatten_hand_comp_gt2=flatten_comps2["gt"]
            print('check consistency of flatten_hand_comp',torch.abs(flatten_hand_comp_gt2-flatten_hand_comp_gt).max())
            print('check consistency of local2base and local2first',(hand_gts2["flatten_local2base_gt"]-hand_gts2["flatten_local2first_gt"]).max())
            for hand_tag in ["left","right"]:
                print('check consistency of first2cam and base2cam with base frame at 0',hand_tag,torch.abs(hand_gts2[f'flatten_firstclip_R_base2cam_{hand_tag}']-hand_gts2[f'flatten_firstclip_R_first2cam_{hand_tag}']).max())
                print('check consistency of first2cam and base2cam with base frame at 0',hand_tag,torch.abs(hand_gts2[f'flatten_firstclip_t_base2cam_{hand_tag}']-hand_gts2[f'flatten_firstclip_t_first2cam_{hand_tag}']).max())
        
        batch_seq_hand_comp_gt = flatten_hand_comp_gt.view(-1,self.ntokens_op, self.dim_hand_feature)
        if "aug" in flatten_comps:
            batch_seq_hand_comp_aug=flatten_comps["aug"][:,:self.dim_hand_feature].view(-1,self.ntokens_op,self.dim_hand_feature)

        batch_seq_valid_features=hand_gts["flatten_valid_features"][:,:self.dim_hand_feature].view(-1,self.ntokens_op,self.dim_hand_feature)
        return_batch["batch_seq_hand_comp_obsv"] = batch_seq_hand_comp_aug[:,:self.ntokens_obsv].clone() if "aug" in flatten_comps else batch_seq_hand_comp_gt[:,:self.ntokens_obsv].clone()
        return_batch["batch_seq_hand_comp_gt"]=batch_seq_hand_comp_gt
        return_batch["batch_seq_valid_features"]=batch_seq_valid_features
        return_batch.update(hand_gts)
            
        return return_batch
        
    def forward(self, batch_flatten, is_train, verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        results, losses = {}, {}
        if "batch_seq_hand_comp_obsv" not in batch_flatten:
            batch0=self.get_gt_inputs_feature(batch_flatten,is_train,verbose=verbose)
        else:
            batch0=batch_flatten

        #lets start
        batch_seq_hand_comp_obsv = batch0["batch_seq_hand_comp_obsv"] 
        batch_size=batch_seq_hand_comp_obsv.shape[0]
        
        #First make sure consistent valid_frame and valid_features
        batch_seq_valid_frame=batch0["valid_frame"].cuda().reshape(batch_size,self.ntokens_op)
        if verbose:
            print("batch_seq_valid_features",batch0["batch_seq_valid_features"].shape)#[bs,len_op,153]
        batch0["batch_seq_valid_features"]=torch.mul(batch0["batch_seq_valid_features"],torch.unsqueeze(batch_seq_valid_frame,-1))
        if verbose:
            print("batch_seq_valid_features,unsqueeze(batch_seq_valid_frame)",batch0["batch_seq_valid_features"].shape,torch.unsqueeze(batch_seq_valid_frame,-1).shape)#[bs,len_op,153],[bs,len_op,1]
            print(torch.abs(torch.where(torch.sum(batch0["batch_seq_valid_features"],dim=-1)>0.,1,0)-batch_seq_valid_frame).max())

        #Pass encoder
        batch_seq_obsv_mask=~(batch_seq_valid_frame.bool()[:,:self.ntokens_obsv])
        batch_mid_mu_enc_out, batch_mid_logvar_enc_out,batch_seq_hand_feature_enc_out, batch_seq_hand_comp_enc_out=self.feed_encoder(batch_seq_hand_comp_obsv,batch_seq_obsv_mask,verbose=verbose)
        
        #obsv hands
        batch_seq_valid_frame_obsv=batch_seq_valid_frame[:,:self.ntokens_obsv]
        batch_seq_hand_size_left=torch.mul(torch.flatten(batch0['hand_size_left'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_left_size=torch.sum(batch_seq_hand_size_left,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)

        batch_seq_hand_size_right=torch.mul(torch.flatten(batch0['hand_size_right'].cuda().view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]),torch.flatten(batch_seq_valid_frame_obsv)).view(-1,self.ntokens_obsv)
        batch_mean_hand_right_size=torch.sum(batch_seq_hand_size_right,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame_obsv,dim=1,keepdim=True)
        
        trans_info_obsv={}
        #for k1 in ['R','t']:
        #    for k2 in ['left','right']:
        #        trans_info_obsv[f'flatten_firstclip_{k1}_base2cam_{k2}']=results[f'flatten_firstclip_{k1}_first2cam_{k2}']
        total_loss_hand,results_hand_obsv,losses_hand=self.compute_hand_loss(batch_seq_comp_gt=batch0["batch_seq_hand_comp_gt"][:,:self.ntokens_obsv], 
                                                batch_seq_comp_out=batch_seq_hand_comp_enc_out,
                                                compute_local2base=False,#True
                                                batch_seq_local2base_gt=None,#results['flatten_local2first_gt'].view(batch_size,self.ntokens_op,-1)[:,:self.ntokens_obsv].clone(), 
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"][:,:self.ntokens_obsv], 
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info_obsv,
                                                normalize_size_from_comp=False,verbose=verbose)
        #for k in ["local","cam"]:            
        #    joints3d_out=results_hand_obsv[f"batch_seq_joints3d_in_{k}_out"]/self.hand_scaling_factor            
        #    results[f"batch_seq_joints3d_in_{k}_obsv_out"]=joints3d_out
        total_loss+=total_loss_hand
        for k,v in losses_hand.items():
            losses[k+"_observ"]=v
            
        if self.lambda_clustering is not None and self.lambda_clustering>1e-10:
            loss_key=[""]
            for item_id in range(len(loss_key)):
                losses[loss_key[item_id]+"kld_loss"]=self.compute_kl_loss(mu1=batch_mid_mu_enc_out[:,item_id],logvar1=batch_mid_logvar_enc_out[:,item_id],verbose=verbose)
                total_loss+=self.lambda_clustering*losses[loss_key[item_id]+"kld_loss"]
        
        if is_train:
            batch_seq_dec_mem=self.reparameterize(mu=batch_mid_mu_enc_out[:,0:1],logvar=batch_mid_logvar_enc_out[:,0:1])
        else:
            batch_seq_dec_mem=batch_mid_mu_enc_out[:,0:1]
        results["batch_seq_dec_mem"]=batch_seq_dec_mem.detach().clone()
        
        batch_seq_dec_mem_mask=torch.zeros_like(batch_seq_obsv_mask[:,:1]).bool()
        rand_to_concat_hand_enc_out=torch.rand(1).cuda()
        if is_train and rand_to_concat_hand_enc_out<0.5:
            batch_seq_dec_mem=torch.cat((batch_seq_dec_mem,batch_seq_hand_feature_enc_out),dim=1)            
            batch_seq_dec_mem_mask=torch.cat((batch_seq_dec_mem_mask,batch_seq_obsv_mask),dim=1)
            
        #Pass decoder and predict hands
        batch_seq_hand_feature_dec_out, batch_seq_hand_comp_dec_out=self.feed_decoder(batch_seq_dec_mem,batch_seq_dec_mem_mask,verbose)
        if verbose:
            print('batch_seq_hand_feature/comp_dec_out',batch_seq_hand_feature_dec_out.shape,batch_seq_hand_comp_dec_out.shape) #[bs,len_p,512][bs,len_p,144]

        #hand loss
        trans_info_pred={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info_pred[k]=batch0[k]            
        total_loss_hand,results_hand,losses_hand=self.compute_hand_loss(batch_seq_comp_gt=batch0["batch_seq_hand_comp_gt"][:,self.ntokens_obsv:], 
                                                batch_seq_comp_out=batch_seq_hand_comp_dec_out,
                                                compute_local2base=True,
                                                batch_seq_local2base_gt=batch0['flatten_local2base_gt'].view(batch_size,self.ntokens_op,-1)[:,self.ntokens_obsv:].clone(), 
                                                batch_seq_valid_features=batch0["batch_seq_valid_features"][:,self.ntokens_obsv:], 
                                                batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                trans_info=trans_info_pred,
                                                normalize_size_from_comp=False, verbose=verbose)
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

        if  verbose:
            for k,v in losses.items():
                print(k,v)
            for k,v in results.items():
                print(k,v.shape) 
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
                print(torch.sum(batch0["batch_seq_valid_features"],dim=-1)+batch0["valid_frame"].cuda().view(batch_size,self.ntokens_op))
                sample_vis_l2r(batch_seq_gt_cam=results["batch_seq_joints3d_in_cam_gt"], 
                                batch_seq_gt_local=results["batch_seq_joints3d_in_local_gt"],##batch_seq_j3d_in_right,
                                joint_links=links,  
                                flatten_imgs=batch_flatten["image_vis"],
                                sample_id=sample_id,
                                cam_info=cam_info,
                                prefix_cache_img=f"./vis_v3/imgs_{tag}/", path_video="./vis_v3/{:s}_{:02d}.avi".format(tag,sample_id))#{tag_out}
            exit(0)
        return total_loss,results,losses

    
    def batch_seq_from_comp_to_joints(self, batch_seq_comp,batch_mean_hand_size,trans_info, normalize_size_from_comp,batch_seq_valid_features=None, verbose=False):
        results={}
        batch_size,len_seq=batch_seq_comp.shape[0],batch_seq_comp.shape[1]
        hand_left_size=repeat(batch_mean_hand_size[0],'b ()-> b n',n=len_seq)
        hand_right_size=repeat(batch_mean_hand_size[1],'b ()-> b n',n=len_seq)
        flatten_mean_hand_size=[torch.flatten(hand_left_size),torch.flatten(hand_right_size)]

        if normalize_size_from_comp:            
            flatten_local_left=batch_seq_comp[:,:,0:21*3].clone().view(-1,21,3)
            flatten_local_right=batch_seq_comp[:,:,21*3:42*3].clone().view(-1,21,3)
            if verbose:
                print("flatten_local_left,flatten_local_right",flatten_local_left.shape,flatten_local_right.shape)#[bs*len_seq,21,3]x2

            #again normalize output w.r.t. hand size
            palm_joints=[0,5,9,13,17]
            palm3d_left = flatten_local_left[:,palm_joints]
            palm3d_right = flatten_local_right[:,palm_joints]
            left_size=torch.mean(torch.linalg.norm(palm3d_left[:,1:]-palm3d_left[:,0:1],ord=2,dim=2,keepdim=False),dim=1,keepdim=True)
            right_size=torch.mean(torch.linalg.norm(palm3d_right[:,1:]-palm3d_right[:,0:1],ord=2,dim=2,keepdim=False),dim=1,keepdim=True)

            if verbose:
                print("linalg",torch.linalg.norm(palm3d_left[:,1:]-palm3d_left[:,0:1],ord=2,dim=2,keepdim=False).shape)#[bs*len_seq,4]
                print("left/right",left_size.shape,right_size.shape)#,left_size,right_size)#[bs*len_seq,1]
                print("batch_seq_valid_features",batch_seq_valid_features.shape)#[bs,len_seq,len_comp]
                print(batch_seq_valid_features[0,0])
                
            flatten_local_left/=left_size.view(-1,1,1)
            flatten_valid_features_left=batch_seq_valid_features[:,:,0:21*3].clone().view(-1,21,3)
            flatten_local_left=torch.where(flatten_valid_features_left>0.,flatten_local_left,self.placeholder_joints[:,:21])
            
            flatten_local_right/=right_size.view(-1,1,1)
            flatten_valid_features_right=batch_seq_valid_features[:,:,21*3:42*3].clone().view(-1,21,3)
            flatten_local_right=torch.where(flatten_valid_features_right>0.,flatten_local_right,self.placeholder_joints[:,21:])           
            
            batch_seq_comp2=torch.cat([flatten_local_left.view(batch_size,len_seq,63), 
                                            flatten_local_right.view(batch_size,len_seq,63), 
                                            batch_seq_comp[:,:,126:]],dim=2)
            results["batch_seq_comp_local_normalized"]=batch_seq_comp2
        else:
            batch_seq_comp2=batch_seq_comp

        flatten_out=from_comp_to_joints(batch_seq_comp2, flatten_mean_hand_size, factor_scaling=self.hand_scaling_factor,trans_info=trans_info)
        for key in ["base","cam","local"]:        
            results[f"batch_seq_joints3d_in_{key}"]=flatten_out[f"joints_in_{key}"].view(batch_size,len_seq,self.num_joints,3)
        results["batch_seq_local2base"]=flatten_out["local2base"].view(batch_size,len_seq,flatten_out["local2base"].shape[-1])
        results["batch_seq_trans_info"]=flatten_out["batch_seq_trans_info"]
        return results


    def compute_hand_loss(self, batch_seq_comp_gt, batch_seq_comp_out, batch_seq_local2base_gt, batch_seq_valid_features, compute_local2base,
                 batch_mean_hand_size, trans_info,normalize_size_from_comp, verbose=False):
        losses={}
        results={}
        total_loss= torch.Tensor([0]).cuda()
        if verbose:
            print("batch_seq_comp_gt/batch_seq_comp_out,batch_seq_valid_features",batch_seq_comp_gt.shape,batch_seq_comp_out.shape,batch_seq_valid_features.shape)
            #[bs,len_seq,144]x3
            if batch_seq_local2base_gt is not None:
                print("batch_seq_local2base_gt",batch_seq_local2base_gt.shape)#[bs,len_seq,18]
        
        if self.lambda_hand is not None and self.lambda_hand>1e-6:           
            recov_hand_loss=self.pose_loss(batch_seq_comp_gt,batch_seq_comp_out,reduction='none')
            if verbose:
                print('recov_hand_loss',torch.abs(recov_hand_loss).max(),recov_hand_loss.shape)#[bs,len_seq,144]
            recov_hand_loss=torch.mul(recov_hand_loss,batch_seq_valid_features)
            cnt=torch.sum(batch_seq_valid_features)
            losses["recov_hand_loss"]=torch.sum(recov_hand_loss)/torch.where(cnt<1.,1.,cnt)#torch.mean(recov_hand_loss) 
            total_loss+=self.lambda_hand*losses["recov_hand_loss"]
        if not compute_local2base:
            return total_loss,results,losses

        #trjectory only for pred
        output_results=self.batch_seq_from_comp_to_joints(batch_seq_comp=batch_seq_comp_out,
                                                        batch_mean_hand_size=batch_mean_hand_size,
                                                        trans_info=trans_info,
                                                        normalize_size_from_comp=normalize_size_from_comp, 
                                                        batch_seq_valid_features=batch_seq_valid_features,
                                                        verbose=verbose)
        for k in output_results.keys():
            if "joints" in k:
                results[k+"_out"]=output_results[k]
            if "batch_seq_comp_local_normalized" in k:
                results[k]=output_results[k]
        
        #global trj in base frame        
        if self.lambda_hand is not None and self.lambda_hand>1e-6:
            batch_seq_local2base_out=output_results["batch_seq_local2base"]
            recov_trj_loss=self.pose_loss(batch_seq_local2base_gt,batch_seq_local2base_out,reduction='none') 

            batch_seq_trj_valid=batch_seq_valid_features[:,:,self.num_joints*3:self.num_joints*3+18]
            recov_trj_loss=torch.mul(batch_seq_trj_valid,recov_trj_loss)
            cnt=torch.sum(batch_seq_trj_valid)
            losses["recov_trj_in_base_loss"]=torch.sum(recov_trj_loss)/torch.where(cnt<1.,1.,cnt)
            #print("mean recov_trj_loss",torch.mean(recov_trj_loss))
            if verbose:
                print("recov_trj_in_base_loss/batch_seq_trj_valid",recov_trj_loss.shape,batch_seq_trj_valid.shape)#[bs,len_p,18]x2
            total_loss+=self.lambda_hand*losses["recov_trj_in_base_loss"]
        return total_loss,results,losses
    

    
    def feed_encoder(self,batch_seq_enc_in_comp,batch_seq_enc_mask_tokens, batch_seq_enc_in_obj=None, batch_seq_enc_in_img=None, verbose=False):       
        if verbose:
            print("****Start P-Enc****")
        batch_seq_enc_in_tokens=torch.cat([self.pose3d_to_trsfm_in(batch_seq_enc_in_comp[:,:,:self.num_joints*3]), \
                                self.globalRt_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3:self.num_joints*3+18]), \
                                self.localL2R_to_trsfm_in(batch_seq_enc_in_comp[:,:,self.num_joints*3+18:self.num_joints*3+27])],dim=2)
             
        batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_tokens=super().feed_encoder(batch_seq_enc_in_tokens,batch_seq_enc_mask_tokens,verbose)
        batch_seq_enc_out_hand_tokens=batch_seq_enc_out_tokens
        batch_seq_enc_out_comp=self.token_out_to_pose(batch_seq_enc_out_hand_tokens)
        if verbose:
            print("batch_seq_enc_out_hand_tokens/batch_seq_enc_out_comp",batch_seq_enc_out_hand_tokens.shape,batch_seq_enc_out_comp.shape)#[bs,len_p,512],[bs,len_p,144]
            print("****End P-Enc****")
        return batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_hand_tokens,batch_seq_enc_out_comp
    


    def feed_decoder(self,batch_seq_dec_mem,batch_seq_dec_mem_mask,verbose):
        if verbose:
            print("****Start P-Dec****")
        batch_seq_dec_out_tokens=super().feed_decoder(batch_seq_dec_query=None,
                                                    batch_seq_dec_mem=batch_seq_dec_mem,
                                                    batch_seq_dec_mem_mask=batch_seq_dec_mem_mask,
                                                    batch_seq_dec_tgt_key_padding_mask=None,verbose=verbose)
        
        batch_seq_dec_out_comp=self.token_out_to_pose(batch_seq_dec_out_tokens)
        if verbose:
            print("batch_seq_dec_out_comp",batch_seq_dec_out_comp.shape)#[bs,len_p,144]
            print("****End: P-Dec Feed Decoder") 
        return batch_seq_dec_out_tokens,batch_seq_dec_out_comp

    
    def process_until_enc(self, batch_flatten, postprocess_hand, verbose=False):
        batch0=self.get_gt_inputs_feature(batch_flatten,is_train=False,verbose=False)
        
        #lets start
        batch_seq_hand_comp_obsv = batch0["batch_seq_hand_comp_obsv"] 
        batch_size=batch_seq_hand_comp_obsv.shape[0]

        batch_seq_obsv_mask=batch_flatten["valid_frame"].cuda().bool()
        batch_seq_obsv_mask=batch_seq_obsv_mask.reshape(batch_size,self.ntokens_op)[:,:self.ntokens_obsv]
        batch_seq_obsv_mask=(~batch_seq_obsv_mask)
        
        #Pass encoder
        batch_mid_mu_enc_out, batch_mid_logvar_enc_out,_, batch_seq_hand_comp_enc_out=self.feed_encoder(batch_seq_hand_comp_obsv,batch_seq_obsv_mask,verbose=verbose)        
        return_results={"batch_mid_mu_enc_out":batch_mid_mu_enc_out,"batch_mid_logvar_enc_out":batch_mid_logvar_enc_out}
                        
        if postprocess_hand:
            batch_mean_hand_left_size=torch.mean(batch_flatten['hand_size_left'].view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv],dim=1,keepdim=True)
            batch_mean_hand_right_size=torch.mean(batch_flatten['hand_size_right'].view(batch_size,self.ntokens_op)[:,:self.ntokens_obsv],dim=1,keepdim=True)

            trans_info_obsv={}
            for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
                trans_info_obsv[k]=batch0[k]
                
            return_results2=self.batch_seq_from_comp_to_joints(batch_seq_comp=batch_seq_hand_comp_enc_out,
                                                    batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                    trans_info=trans_info_obsv, 
                                                    normalize_size_from_comp=False,
                                                    batch_seq_valid_features=None, verbose=False)

            for k in ["local","base","cam"]:
                return_results[f"batch_seq_joints3d_in_{k}_out"]=return_results2[f"batch_seq_joints3d_in_{k}"]/self.hand_scaling_factor
        return return_results

