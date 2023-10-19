import torch
import torch.nn.functional as torch_f

from einops import repeat

from meshreg.models.transformer import Transformer_Encoder, PositionalEncoding
from meshreg.models.actionbranch import ActionClassificationBranch
from meshreg.models.utils import To25DBranch,compute_hand_loss,loss_str2func,augment_hand_pose_2_5D
from meshreg.models.utils import compute_bert_embedding_for_taxonomy, compute_berts_for_strs, embedding_lookup
from meshreg.models.mlp import MultiLayerPerceptron
from meshreg.models.resnet_ho import ResNet_, BasePerceptionBlock, ImageBlock
import cv2
from libyana.visutils.viz2d import visualize_joints_2d

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as Axes
import numpy as np
import copy
import open_clip


class TemporalNet(BasePerceptionBlock):
    def __init__(self,  transformer_d_model,
                        transformer_dropout,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_num_encoder_layers_action,
                        transformer_num_encoder_layers_pose,
                        transformer_normalize_before=True,

                        lambda_action=None,
                        lambda_hand_2d=None,
                        lambda_hand_z=None,
                        ntokens_pose=1,
                        ntokens_action=1,
                        
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        code_loss="l2"):

        super().__init__(lambda_hand_2d=lambda_hand_2d,
                        lambda_hand_z=lambda_hand_z,
                        lambda_obj=lambda_action,
                        trans_factor=trans_factor,
                        scale_factor=scale_factor,
                        pose_loss=pose_loss,
                        code_loss=code_loss)

        
        self.ntokens_pose= ntokens_pose
        self.ntokens_action=ntokens_action

        
        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model) 

        
        #Feature to Action        
        self.hand_pose3d_to_pin=torch.nn.Linear(self.num_joints*3,transformer_d_model)
        self.olabel_to_pin=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.concat_to_pin=torch.nn.Linear(transformer_d_model*3,transformer_d_model)

        self.transformer_pose=Transformer_Encoder(d_model=transformer_d_model, 
                                nhead=transformer_nhead, 
                                num_encoder_layers=transformer_num_encoder_layers_pose,
                                dim_feedforward=transformer_dim_feedforward,
                                dropout=0.0, 
                                activation="relu", 
                                normalize_before=transformer_normalize_before)
                                    
       
        #Hand 2.5D branch        
        self.pout_to_hand_pose=MultiLayerPerceptron(base_neurons=[transformer_d_model, transformer_d_model,transformer_d_model], out_dim=self.num_joints*3,
                                act_hidden='leakyrelu',act_final='none')        

        #Object classification
        self.pout_to_olabel_embed=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.object_name2idx=None
        
        #Feature to Action        
        self.hand_pose2d_to_ain=torch.nn.Linear(self.num_joints*2,transformer_d_model)
        self.olabel_to_ain=torch.nn.Linear(transformer_d_model,transformer_d_model)
        self.concat_to_ain=torch.nn.Linear(transformer_d_model*3,transformer_d_model)

        self.action_token=torch.nn.Parameter(torch.randn(1,1,transformer_d_model))
        self.transformer_action=Transformer_Encoder(d_model=transformer_d_model, 
                            nhead=transformer_nhead, 
                            num_encoder_layers=transformer_num_encoder_layers_action,
                            dim_feedforward=transformer_dim_feedforward,
                            dropout=0.0,
                            activation="relu", 
                            normalize_before=transformer_normalize_before) 
        
        self.action_bert_to_latent= torch.nn.Linear(512,transformer_d_model)

        if self.lambda_obj is not None and self.lambda_obj>1e-6:
            self.model_bert, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="./assets/")
            self.model_bert.eval()

        self.img_h, self.img_w=270,540
        

    def assign_ibe(self,model_ibe):
        self.model_ibe=model_ibe
        self.model_ibe.eval()

    def compute_bert_embedding_for_taxonomy(self, datasets, verbose=False):
        name_to_idx,tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action=False, verbose=verbose)
        self.object_name2idx=copy.deepcopy(name_to_idx)
        self.object_embedding=tokens.detach().clone()
        
        action_name_to_idx, action_tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action=True, verbose=verbose)
        self.action_name2idx=copy.deepcopy(action_name_to_idx)
        self.action_embedding=action_tokens.detach().clone()
        
    def preprocess_input_feature(self, batch_flatten, is_train, verbose=False):
        flatten_image_feature=batch_flatten["image_feature"].cuda()
        flatten_object_feature=batch_flatten["object_feature"].cuda()

        flatten_resnet25d=batch_flatten["joints25d_resnet"].cuda()
        flatten_ncam_intr=batch_flatten["ncam_intr"].cuda()
        
        #augmentaion
        if is_train:
            flatten_object_feature+=torch.randn_like(flatten_object_feature)*0.01
            #flatten_resnet25d=augment_hand_pose_2_5D(flatten_resnet25d)

        #reproject to latent space
        flatten_hpose_feature=self.postprocess_hand_pose.reproject_to_latent(flatten_camintr=flatten_ncam_intr,
                                                                            flatten_pose25d=flatten_resnet25d,
                                                                            height=self.img_h,width=self.img_w)
        if verbose:
            recovered_pose=self.postprocess_hand_pose(flatten_camintr=flatten_ncam_intr,flatten_scaletrans=flatten_hpose_feature,height=self.img_h,width=self.img_w)

            print("check consistency of pose-feature-pose, 2D/z",torch.abs(recovered_pose["rep2d"]-flatten_resnet25d[:,:,:2]).max(),\
            torch.abs(recovered_pose["rep_absz"]-flatten_resnet25d[:,:,2:]).max())
        
        return {"image_feature":flatten_image_feature,
            "obj_feature":flatten_object_feature,
            "hpose_feature":torch.flatten(flatten_hpose_feature,1,2),
            "flatten_resnet25d_aug":flatten_resnet25d,}
        



    def forward(self, batch_flatten, is_train,  verbose=False): 
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}

        batch0=self.preprocess_input_feature(batch_flatten, is_train, verbose=verbose)
        
        #_, results_ibe, _ = self.model_ibe(batch_flatten, verbose=verbose)

        #flatten_pin_image_feature=results_ibe["image_feature"].detach()
        #flatten_pin_hpose_feature=results_ibe["hpose_feature"].detach()
        #flatten_pin_olabel_feature=results_ibe["obj_feature"].detach()

        flatten_pin_image_feature=batch0["image_feature"]
        flatten_pin_hpose_feature=batch0["hpose_feature"]
        flatten_pin_olabel_feature=batch0["obj_feature"]
        
        if verbose:
            print("flatten_pin_image/hpose/olabel_feature",flatten_pin_image_feature.shape,flatten_pin_hpose_feature.shape,flatten_pin_olabel_feature.shape)#[-1,512],[-1,126],[-1,512]
        #Block P input
        flatten_pin_hpose_feature=self.hand_pose3d_to_pin(flatten_pin_hpose_feature)
        flatten_pin_olabel_feature=self.olabel_to_pin(flatten_pin_olabel_feature)
        flatten_in_feature=torch.cat((flatten_pin_image_feature,flatten_pin_hpose_feature,flatten_pin_olabel_feature),dim=-1)
        if verbose:
            print("flatten_pin_hpose/olabel_feature, flatten_in_feature",flatten_pin_hpose_feature.shape,flatten_pin_olabel_feature.shape,flatten_in_feature.shape)#[-1,512]x2,[-1,1536]

        flatten_in_feature=self.concat_to_pin(flatten_in_feature)
        if verbose:
            print("flatten_in_feature",flatten_in_feature.shape)#[-1,512]
        
        #Block P
        batch_seq_pin_feature=flatten_in_feature.contiguous().view(-1,self.ntokens_pose,flatten_in_feature.shape[-1])
        batch_seq_pin_pe=self.transformer_pe(batch_seq_pin_feature)
         
        batch_seq_pweights=batch_flatten['valid_frame'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()

        batch_seq_pout_feature,_=self.transformer_pose(src=batch_seq_pin_feature, src_pos=batch_seq_pin_pe,
                                        key_padding_mask=batch_seq_pmasks, verbose=False)
 
        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
        if verbose:
            print("flatten_pout_feature",flatten_pout_feature.shape)#[-1,512]
            
        
        #hand pose
        flatten_pout_hpose=self.pout_to_hand_pose(flatten_pout_feature)
        flatten_pout_hpose=flatten_pout_hpose.view(-1,self.num_joints,3)

        flatten_pout_hpose_25d_3d=self.postprocess_hand_pose(flatten_camintr=batch_flatten["ncam_intr"].cuda(), 
                                                    flatten_scaletrans=flatten_pout_hpose,
                                                    height=self.img_h,#batch_flatten["image"].shape[2],
                                                    width=self.img_w,#batch_flatten["image"].shape[3],
                                                    verbose=False)

        weights_hand_loss=batch_flatten['valid_frame'].cuda().float()
        hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,
                                                    flatten_hpose_25d_3d=flatten_pout_hpose_25d_3d,
                                                    weights=weights_hand_loss,
                                                    total_loss=total_loss,verbose=verbose)        
        results.update(hand_results)
        losses.update(hand_losses)

        #Object label
        flatten_olabel_feature=self.pout_to_olabel_embed(flatten_pout_feature)        
        weights_olabel_loss=batch_flatten['valid_frame'].cuda().float()

        olabel_results,total_loss,olabel_losses=self.predict_object(flatten_sample=batch_flatten,
                                                                    flatten_features=flatten_olabel_feature,
                                                                    weights=weights_olabel_loss,
                                                                    total_loss=total_loss,verbose=verbose)
        results.update(olabel_results)
        losses.update(olabel_losses)

    
        #Block A input
        flatten_pout_hpose2d=torch.flatten(flatten_pout_hpose[:,:,:2],1,2)
        flatten_ain_feature_hpose=self.hand_pose2d_to_ain(flatten_pout_hpose2d)
        flatten_ain_feature_olabel=self.olabel_to_ain(flatten_olabel_feature)
        
        flatten_ain_feature=torch.cat((flatten_pout_feature,flatten_ain_feature_hpose,flatten_ain_feature_olabel),dim=-1)
        if verbose:
            print("flatten_pout_hpose2d/flatten_ain_feature_hpose/flatten_ain_feature_olabel,flatten_ain_feature",flatten_pout_hpose2d.shape,flatten_ain_feature_hpose.shape,flatten_ain_feature_olabel.shape,flatten_ain_feature.shape)#[-1,84],[-1,512],[-1,512],[-1,512*3]
        
        flatten_ain_feature=self.concat_to_ain(flatten_ain_feature)
        batch_seq_ain_feature=flatten_ain_feature.contiguous().view(-1,self.ntokens_action,flatten_ain_feature.shape[-1])
        if verbose:
            print("batch_seq_ain_feature",batch_seq_ain_feature.shape)#[bs,ntokens,512]
        
        #Concat trainable token
        batch_aglobal_tokens = repeat(self.action_token,'() n d -> b n d',b=batch_seq_ain_feature.shape[0])
        batch_seq_ain_feature=torch.cat((batch_aglobal_tokens,batch_seq_ain_feature),dim=1)
        batch_seq_ain_pe=self.transformer_pe(batch_seq_ain_feature)
 
        batch_seq_weights_action=batch_flatten['valid_frame'].cuda().float().view(-1,self.ntokens_action)
        batch_seq_amasks_frames=(1-batch_seq_weights_action).bool()
        batch_seq_amasks_global=torch.zeros_like(batch_seq_amasks_frames[:,:1]).bool() 
        batch_seq_amasks=torch.cat((batch_seq_amasks_global,batch_seq_amasks_frames),dim=1)       

        if verbose:
            print("batch_seq_ain_feature",batch_seq_ain_feature.shape)#[bs,ntokens+1,512]
            print("batch_seq_amasks",batch_seq_amasks.shape,batch_seq_amasks)#[bs,ntokens+1]
         
        batch_seq_aout_feature,_=self.transformer_action(src=batch_seq_ain_feature, src_pos=batch_seq_ain_pe,
                                                key_padding_mask=batch_seq_amasks, verbose=False)
        
        #Action
        batch_out_action_feature=torch.flatten(batch_seq_aout_feature[:,0],1,-1)     
        
        action_results, total_loss, action_losses=self.predict_action(flatten_sample=batch_flatten,
                                                            batch_aenc_action_obsv_out=batch_out_action_feature, 
                                                            total_loss=total_loss,verbose=verbose)
        
        results.update(action_results)
        losses.update(action_losses)

        if verbose:
            img_vis=batch_flatten["image_vis"].detach().cpu().numpy()
            #oricam_joints3d=torch.cat([batch_flatten["cam_joints3d_left"],batch_flatten["cam_joints3d_right"]],dim=1).detach().cpu().numpy()
            #oricam_intr=batch_flatten["cam_intr"].detach().cpu().numpy()
            oricam_joints2d=torch.cat([batch_flatten["aug_joints2d_left"],batch_flatten["aug_joints2d_right"]],dim=1).detach().cpu().numpy()
            
            ncam_joints3d=torch.cat([batch_flatten["ncam_joints3d_left"],batch_flatten["ncam_joints3d_right"]],dim=1).detach().cpu().numpy()
            ncam_intr=batch_flatten["ncam_intr"].detach().cpu().numpy()

            
            aug_joints25d=batch0["flatten_resnet25d_aug"].detach().cpu().numpy()


            for i in list(range(0,32))+list(range(64,90)):#img_vis.shape[0]):
                to_vis=cv2.cvtColor(img_vis[i],cv2.COLOR_RGB2BGR)
                #oricam_proj=(oricam_intr[i]@oricam_joints3d[i].T).T
                #oricam_proj=(oricam_proj/oricam_proj[:,2:3])[:,0:2]
                ncam_proj=(ncam_intr[i]@ncam_joints3d[i].T).T
                ncam_proj=(ncam_proj/ncam_proj[:,2:3])[:,0:2]

                oricam_proj=aug_joints25d[i]

                #print(np.fabs(oricam_proj-ncam_proj).max(),np.fabs(oricam_proj-oricam_joints2d[i]).max())
                
                fig = plt.figure(figsize=(3,2))
                axi=plt.subplot2grid((1,1),(0,0),colspan=1)
                axi.axis("off")
                

                axi.imshow(to_vis) 
                joint_links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]

                gt_c=(0,1.,0)
                visualize_joints_2d(axi, oricam_proj[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
                visualize_joints_2d(axi, oricam_proj[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
                
                gt_c=(1.,0.,0)
                visualize_joints_2d(axi, ncam_proj[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
                visualize_joints_2d(axi, ncam_proj[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)

                
                gt_c=(0.,0.,1)
                visualize_joints_2d(axi, oricam_joints2d[i,:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
                visualize_joints_2d(axi, oricam_joints2d[i,21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)

                fig.savefig(f"{i}.png", dpi=200)
                plt.close(fig)
            exit(0)

        return total_loss, results, losses
    
    


    def predict_action(self,flatten_sample,batch_aenc_action_obsv_out,total_loss=None,verbose=False):
        results,losses={},{}
        batch_aname_obsv_gt=flatten_sample["action_name"][::self.ntokens_action]
        results["batch_action_name_obsv"]=batch_aname_obsv_gt
        batch_atokens_obsv_gt=open_clip.tokenizer.tokenize(batch_aname_obsv_gt).cuda()
        with torch.no_grad():            
            batch_atokens_obsv_gt=self.model_bert.encode_text(batch_atokens_obsv_gt).float()
        batch_atokens_obsv_gt/=batch_atokens_obsv_gt.norm(dim=-1,keepdim=True)
        if verbose:
            print("****A-Enc GT Action: batch_aname_obsv_gt",len(batch_aname_obsv_gt),batch_aname_obsv_gt)
            print("batch_atokens_obsv_gt",batch_atokens_obsv_gt.shape,torch.abs(batch_atokens_obsv_gt.norm(dim=-1)-1.).max())#[bs,512]

        batch_atokens_obsv_gt=self.action_bert_to_latent(batch_atokens_obsv_gt.detach().clone())
        action_embedding=torch.transpose(self.action_bert_to_latent(torch.transpose(self.action_embedding,0,1).detach()),0,1)
        
        if verbose:
            print("batch_atokens_obsv_gt/batch_aenc_action_obsv_out",batch_atokens_obsv_gt.shape,batch_aenc_action_obsv_out.shape)#[bs,512],[bs,512]
            print("self.action_embedding/action_embedding",self.action_embedding.shape,action_embedding.shape)#[512,nembeddings],[512,nembeddings]


        #action loss
        losses["action_dist_loss"]=self.code_loss(batch_aenc_action_obsv_out,batch_atokens_obsv_gt)

        #Contrastive with Cosine Distance     
        results["batch_action_idx_obsv_gt"]=torch.zeros(len(batch_aname_obsv_gt),dtype=torch.int64).cuda()
        for vid, vname in enumerate(batch_aname_obsv_gt):
            results["batch_action_idx_obsv_gt"][vid]=self.action_name2idx[vname]

        batch_action_similarity_obsv,results["batch_action_idx_obsv_out"]=embedding_lookup(query=batch_aenc_action_obsv_out,embedding=action_embedding, verbose=verbose)#############
        batch_action_similarity_obsv=batch_action_similarity_obsv/self.temperature
        results["batch_action_prob_distrib"]=torch.nn.functional.softmax(batch_action_similarity_obsv,dim=1)
        losses["action_contrast_loss"] = torch_f.cross_entropy(batch_action_similarity_obsv,results["batch_action_idx_obsv_gt"],reduction='mean')
        
        if self.lambda_obj is not None and self.lambda_obj>1e-6:
            total_loss+=self.lambda_obj*(losses["action_dist_loss"]+losses["action_contrast_loss"])
        if verbose:
            print("batch_action_idx_obsv_gt",results["batch_action_idx_obsv_gt"].shape)#[bs]
            print("batch_action_similarity_obsv/batch_action_prob_distrib",batch_action_similarity_obsv.shape,results["batch_action_prob_distrib"].shape)#[bs,tax_size]x2
            prob_norm=torch.sum(results["batch_action_prob_distrib"], axis=-1)
            print("check sum", prob_norm.shape, torch.abs(prob_norm-1).max())#[bs]

            print('if gt input, check action_idx gt vs out',torch.abs(results["batch_action_idx_obsv_gt"]-results["batch_action_idx_obsv_out"]).max())
            print("==================================")
        
        return  results,total_loss, losses