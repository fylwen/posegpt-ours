import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat

from meshreg.models import resnet
from meshreg.models.transformer import Transformer_Encoder, PositionalEncoding
from meshreg.models.actionbranch import ActionClassificationBranch
from meshreg.models.utils import To25DBranch,loss_str2func, compute_hand_loss
from meshreg.models.utils import compute_bert_embedding_for_taxonomy,compute_berts_for_strs, embedding_lookup
from meshreg.models.mlp import MultiLayerPerceptron
from libyana.visutils.viz2d import visualize_joints_2d


import open_clip
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as Axes
import numpy as np

class ResNet_(torch.nn.Module):
    def __init__(self,resnet_version=18):
        super().__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            self.base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            self.base_net = resnet.resnet50(pretrained=True)
        else:
            self.base_net=None

    def forward(self, image):
        features, res_layer5 = self.base_net(image)
        return features, res_layer5

class BasePerceptionBlock(torch.nn.Module):
    def __init__(self,  lambda_hand_2d=None,
                        lambda_hand_z=None,
                        lambda_obj=None,
        
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        code_loss="l2"):

        super().__init__()
        
        self.pose_loss=loss_str2func()[pose_loss]
        self.code_loss=loss_str2func()[code_loss]
        self.temperature=0.07
        
        self.lambda_hand_z=lambda_hand_z
        self.lambda_hand_2d=lambda_hand_2d        
        self.lambda_obj=lambda_obj

        self.num_joints=42
        
        #Hand 2.5D branch        
        self.scale_factor = scale_factor 
        self.trans_factor = trans_factor        
        self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)


        self.object_name2idx=None
        self.action_name2idx=None
    
    def recover_hand(self, flatten_sample, flatten_hpose_25d_3d, weights, total_loss,verbose=False):
        hand_results, hand_losses={},{}
        
        '''
        hand_results["gt_joints3d"]=torch.cat([flatten_sample["cam_joints3d_left"],flatten_sample["cam_joints3d_right"]],dim=1).float().cuda()#in meter
        hand_results["pred_joints3d"]=torch.cat([flatten_sample["cam_joints3d_left_resnet"],flatten_sample["cam_joints3d_right_resnet"]],dim=1).float().cuda()#in meter
        assert torch.abs(hand_results["pred_joints3d"]-flatten_hpose_25d_3d["rep3d"].detach().clone()).max()<1e-6
        '''
        
        hand_results["valid_frames"]=weights
        
        hand_results["pred_joints3d"]=flatten_hpose_25d_3d["rep3d"].detach().clone()
        hand_results["cam_joints3d_left_out"],hand_results["cam_joints3d_right_out"]=hand_results["pred_joints3d"][:,:21].float(),hand_results["pred_joints3d"][:,21:].float()
        
        hand_results["pred_joints2d"]=flatten_hpose_25d_3d["rep2d"]
        hand_results["pred_jointsz"]=flatten_hpose_25d_3d["rep_absz"]
        hand_results["pred_joints25d"]=torch.cat([hand_results["pred_joints2d"],hand_results["pred_jointsz"]],dim=-1)
        
        hpose_loss=0.
        if self.lambda_hand_2d is not None and self.lambda_hand_2d>1e-6:
            joints3d_gt = torch.cat([flatten_sample["ncam_joints3d_left"],flatten_sample["ncam_joints3d_right"]],dim=1).float().cuda()#in meter
            joints2d_gt = torch.cat([flatten_sample["aug_joints2d_left"],flatten_sample["aug_joints2d_right"]],dim=1).float().cuda()
            hand_results["gt_joints3d"]=joints3d_gt   
            hand_results["gt_joints2d"]=joints2d_gt
            hand_results["gt_joints25d"]=torch.cat([joints2d_gt,joints3d_gt[:,:,2:3]],dim=-1)

            #assert torch.abs(hand_results["gt_joints25d"]-flatten_sample["joints25d"].cuda()).max()<1e-6
            #assert torch.abs(hand_results["pred_joints25d"]-flatten_sample["joints25d_resnet"].cuda()).max()<1e-6
            #assert torch.abs(hand_results["gt_joints3d"]-torch.cat([flatten_sample["cam_joints3d_left"],flatten_sample["cam_joints3d_right"]],dim=1).cuda()).max()<1e-6
            #assert torch.abs(hand_results["pred_joints3d"]-torch.cat([flatten_sample["cam_joints3d_left_resnet"],flatten_sample["cam_joints3d_right_resnet"]],dim=1).cuda()).max()<1e-6


            hand_losses=compute_hand_loss(est2d=flatten_hpose_25d_3d["rep2d"],
                                        gt2d=joints2d_gt,
                                        estz=flatten_hpose_25d_3d["rep_absz"],
                                        gtz=joints3d_gt[:,:,2:3],
                                        est3d=flatten_hpose_25d_3d["rep3d"],
                                        gt3d= joints3d_gt,
                                        weights=weights,
                                        pose_loss=self.pose_loss,
                                        verbose=verbose)

            total_loss+=hand_losses["recov_joints2d"]*self.lambda_hand_2d + hand_losses["recov_joints_absz"]*self.lambda_hand_z    
        
        return hand_results, total_loss, hand_losses

    def predict_object(self,flatten_sample,flatten_features, weights, total_loss,verbose=False):
        results,losses={},{}
        sum_weights=torch.where(torch.sum(weights)>0,torch.sum(weights),torch.Tensor([1]).cuda())[0]
        results["valid_obj"]=weights

        #dist loss
        #assert torch.abs(flatten_features-flatten_sample["object_feature"].cuda()).max()<1e-6        
        
        if self.lambda_obj is not None and self.lambda_obj>1e-6:            
            #tokenize for gt obj_name
            oname_gt=flatten_sample["obj_name"]
            otokens_gt=open_clip.tokenizer.tokenize(oname_gt).cuda()
            with torch.no_grad():
                otokens_gt=self.model_bert.encode_text(otokens_gt).float()
            otokens_gt/=otokens_gt.norm(dim=-1,keepdim=True)
            
            if verbose:
                print("****GT Object: oname_gt",len(oname_gt))#[bs]
                print("sum_weights",sum_weights)
                print("otokens_gt",otokens_gt.shape,torch.abs(otokens_gt.norm(dim=-1)-1.).max())#[bs,512]

            obj_dist_loss=torch.mean(self.code_loss(otokens_gt,flatten_features,reduction="none"),dim=-1,keepdim=False)
            if verbose:
                print("obj_dist_loss/weights",obj_dist_loss.shape,weights.shape)#[bs],[bs]
            obj_dist_loss=torch.mul(obj_dist_loss,weights)
            losses["obj_dist_loss"]=torch.sum(obj_dist_loss)/sum_weights

            #contrastive loss
            object_embedding=self.object_embedding.detach().clone()
            if verbose:
                print("object_embedding",object_embedding.shape)#[512,size]
            
            results['obj_idx_gt']=torch.zeros(otokens_gt.shape[0],dtype=torch.int64).cuda()
            for vid, vname in enumerate(flatten_sample["obj_name"]):
                results['obj_idx_gt'][vid]=self.object_name2idx[vname] if vname!="NIL" else 0
                #assert (weights[vid] and vname!="NIL") or (weights[vid]<1 and vname=="NIL") 
                #assert vname!="NIL"

            obj_similarity,results["obj_idx_out"]=embedding_lookup(query=flatten_features,embedding=object_embedding, verbose=verbose)#
            obj_similarity=obj_similarity/self.temperature
            results["obj_prob_distrib"]=nn.functional.softmax(obj_similarity,dim=-1)
            
            obj_contrast_loss = torch_f.cross_entropy(obj_similarity,results["obj_idx_gt"],reduction='none')
            if verbose:
                print("results[obj_prob_distrib]",results["obj_prob_distrib"].shape)#[bs,nsize]
                prob_norm=torch.sum(results["obj_prob_distrib"], axis=-1)
                print("check sum", prob_norm.shape, torch.abs(prob_norm-1).max())#[bs]
                print("obj_contrast_loss/weights",obj_contrast_loss.shape,weights.shape)#[bs],[bs]
            
            obj_contrast_loss=torch.mul(obj_contrast_loss,weights)
            losses["obj_contrast_loss"]=torch.sum(obj_contrast_loss)/sum_weights
            
            total_loss+=self.lambda_obj*(losses["obj_dist_loss"]+losses["obj_contrast_loss"])
        return results,total_loss,losses

 

class ImageBlock(BasePerceptionBlock):
    def __init__(self,  lambda_hand_2d=None,
                        lambda_hand_z=None,
                        lambda_obj=None,
        
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        code_loss="l2"):

        super().__init__(lambda_hand_2d=lambda_hand_2d,
                        lambda_hand_z=lambda_hand_z,
                        lambda_obj=lambda_obj,
                        trans_factor=trans_factor,
                        scale_factor=scale_factor,
                        pose_loss=pose_loss,
                        code_loss=code_loss)
        
        
        #Image Feature
        self.backbone = ResNet_(resnet_version=18)
        d_model=512
        
       
        self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[d_model, d_model,d_model], out_dim=self.num_joints*3,
                                act_hidden='leakyrelu',act_final='none') 
        
        #Object classification
        self.image_to_obj_feature=torch.nn.Linear(d_model,d_model)
        
        if self.lambda_obj is not None:
            self.model_bert, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="./assets/")
            self.model_bert.eval()
    
    def compute_bert_embedding_for_taxonomy(self, datasets, verbose=False):
        name_to_idx,tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action=False, verbose=verbose)

        self.object_name2idx=copy.deepcopy(name_to_idx)
        self.object_embedding=tokens.detach().clone()

    def forward(self, batch_flatten,is_train,  verbose=False): 
        flatten_images=batch_flatten["image"].cuda()
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses, results = {}, {}

        #resnet for by-frame        
        flatten_out_feature, _ =self.backbone(flatten_images) 
        results["image_feature"]=flatten_out_feature
        #assert torch.abs(results["image_feature"]-batch_flatten["image_feature"].cuda()).max()<1e-6
        
        if verbose:
            print("flatten_out_feature",flatten_out_feature.shape)#[bs,512]
            print("batch_flatten[image]",batch_flatten["image"].shape)#torch.Size([bs, 3, 270, 480])
        
        #hand pose
        flatten_hpose=self.image_to_hand_pose(flatten_out_feature)
        results["hpose_feature"]=flatten_hpose
        flatten_hpose=flatten_hpose.view(-1,self.num_joints,3)
        
        flatten_hpose_25d_3d=self.postprocess_hand_pose(flatten_camintr=batch_flatten["ncam_intr"].cuda(), 
                                                    flatten_scaletrans=flatten_hpose,
                                                    height=batch_flatten["image"].shape[2],
                                                    width=batch_flatten["image"].shape[3],
                                                    verbose=False)
                                                    
        weights_hand_loss=torch.mul(batch_flatten["valid_frame"].float().cuda(),batch_flatten["has_camera_and_image"].float().cuda())
        hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,
                                                            flatten_hpose_25d_3d=flatten_hpose_25d_3d,
                                                            weights=weights_hand_loss,
                                                            total_loss=total_loss,verbose=verbose)        

        results.update(hand_results)
        losses.update(hand_losses)

        #Object label
        flatten_olabel_feature=self.image_to_obj_feature(flatten_out_feature)
        results["obj_feature"]=flatten_olabel_feature
        
        if self.object_name2idx is not None:
            weights_olabel_loss=torch.mul(batch_flatten["valid_frame"].float().cuda(),batch_flatten["has_obj_and_image"].float().cuda())
            olabel_results,total_loss,olabel_losses=self.predict_object(flatten_sample=batch_flatten,
                                                            flatten_features=flatten_olabel_feature,
                                                            weights=weights_olabel_loss,
                                                            total_loss=total_loss,verbose=verbose)
            results.update(olabel_results)
            losses.update(olabel_losses)

        if verbose:
            img_vis=batch_flatten["image_vis"].detach().cpu().numpy()
            oricam_joints3d=torch.cat([batch_flatten["cam_joints3d_left"],batch_flatten["cam_joints3d_right"]],dim=1).detach().cpu().numpy()
            oricam_intr=batch_flatten["cam_intr"].detach().cpu().numpy()
            #oricam_joints2d=torch.cat([batch_flatten["aug_joints2d_left"],batch_flatten["aug_joints2d_right"]],dim=1).detach().cpu().numpy()
            oricam_joints2d=results["pred_joints2d"].detach().cpu().numpy()


            ncam_joints3d=torch.cat([batch_flatten["ncam_joints3d_left"],batch_flatten["ncam_joints3d_right"]],dim=1).detach().cpu().numpy()
            ncam_intr=batch_flatten["ncam_intr"].detach().cpu().numpy()


            for i in range(0,64):#img_vis.shape[0]):
                to_vis=cv2.cvtColor(img_vis[i],cv2.COLOR_RGB2BGR)
                oricam_proj=(oricam_intr[i]@oricam_joints3d[i].T).T
                oricam_proj=(oricam_proj/oricam_proj[:,2:3])[:,0:2]
                ncam_proj=(ncam_intr[i]@ncam_joints3d[i].T).T
                ncam_proj=(ncam_proj/ncam_proj[:,2:3])[:,0:2]

                print(np.fabs(oricam_proj-ncam_proj).max(),np.fabs(oricam_proj-oricam_joints2d[i]).max())
                
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
    







class POEnc(torch.nn.Module):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_activation,
                        ntokens_per_clip,
                        lambda_obj,
                        code_loss):
        super().__init__()
        
        self.code_loss=loss_str2func()[code_loss]
        self.temperature=0.07
        self.object_name2idx=None

        self.ntokens_per_clip=ntokens_per_clip
        self.lambda_obj=lambda_obj
        if self.lambda_obj is not None:
            self.model_bert, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir="./assets/")
            self.model_bert.eval()
        
        
        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model)
        self.mid_token_mu_enc_in=torch.nn.Parameter(torch.randn(1,1,transformer_d_model))


        self.encoder=Transformer_Encoder(d_model=transformer_d_model, 
                                        nhead=transformer_nhead, 
                                        num_encoder_layers=transformer_nlayers_enc,
                                        dim_feedforward=transformer_dim_feedforward,
                                        dropout=0.0,#transformer_dropout,
                                        activation=transformer_activation, 
                                        normalize_before=True)
        self.noise_std=0.01


    def compute_bert_embedding_for_taxonomy(self, datasets, verbose=False):
        name_to_idx,tokens=compute_bert_embedding_for_taxonomy(self.model_bert, datasets, is_action=False, verbose=verbose)

        self.object_name2idx=copy.deepcopy(name_to_idx)
        self.object_embedding=tokens.detach().clone()


    def forward(self, batch_flatten, is_train,  verbose=False):
        total_loss = torch.Tensor([0]).cuda()
        losses, results = {}, {}

        batch_enc_out_mu=self.feed_encoder(batch_flatten,is_train=is_train,verbose=verbose)
        batch_size=batch_enc_out_mu.shape[0]       
        
        #Compute loss       
        #tokenize for gt obj_name
        if "otokens_gt" not in batch_flatten:
            oname_gt=[batch_flatten["obj_name"][i] for i in range(0,len(batch_flatten["obj_name"]),self.ntokens_per_clip)]
            otokens_gt=open_clip.tokenizer.tokenize(oname_gt).cuda()
            with torch.no_grad():
                otokens_gt=self.model_bert.encode_text(otokens_gt).float()
            otokens_gt/=otokens_gt.norm(dim=-1,keepdim=True)
        
            if verbose:
                print("****GT Object: oname_gt",len(oname_gt))#[bs]
                print("otokens_gt",otokens_gt.shape,torch.abs(otokens_gt.norm(dim=-1)-1.).max())#[bs,512]
                print("batch_seq_enc_out",batch_seq_enc_out.shape)
                print("batch_enc_out_mu",batch_enc_out_mu.shape)#[bs,512]
            
            
            #contrastive loss
            object_embedding=self.object_embedding.detach().clone() 
            if verbose:
                print("object_embedding",object_embedding.shape)#[512,size]
            
            results['obj_idx_gt']=torch.zeros(batch_size,dtype=torch.int64).cuda()
            for vid, vname in enumerate(oname_gt):
                results['obj_idx_gt'][vid]=self.object_name2idx[vname] if vname!="NIL" else 0
                assert vname!="NIL"
        else:
            otokens_gt=batch_flatten["otokens_gt"]
            object_embedding=batch_flatten["object_embedding"].detach().clone()
            results['obj_idx_gt']=batch_flatten["obj_idx_gt"]

        losses["obj_dist_loss"]=torch.mean(self.code_loss(otokens_gt,batch_enc_out_mu))

        obj_similarity,results["obj_idx_out"]=embedding_lookup(query=batch_enc_out_mu,embedding=object_embedding, verbose=verbose)#
        obj_similarity=obj_similarity/self.temperature
        results["obj_prob_distrib"]=nn.functional.softmax(obj_similarity,dim=-1)
        
        losses["obj_contrast_loss"] = torch_f.cross_entropy(obj_similarity,results["obj_idx_gt"],reduction='mean')
        if verbose:
            print("results[obj_prob_distrib]",results["obj_prob_distrib"].shape)#[bs,nsize]
            prob_norm=torch.sum(results["obj_prob_distrib"], axis=-1)
            print("check sum", prob_norm.shape, torch.abs(prob_norm-1).max())#[bs]
            print(losses)
            
        if self.lambda_obj is not None and self.lambda_obj>1e-6:     
            total_loss+=self.lambda_obj*(losses["obj_dist_loss"]+losses["obj_contrast_loss"])
        return total_loss, results, losses

    def feed_encoder(self,batch_flatten,is_train,verbose):
        batch_seq_enc_in_tokens=batch_flatten["object_feature"].cuda().view((-1,self.ntokens_per_clip,)+batch_flatten["object_feature"].shape[1:])        
        batch_size=batch_seq_enc_in_tokens.shape[0]
        batch_seq_enc_in_mask_tokens=~(batch_flatten["valid_frame"].cuda().view(batch_size,self.ntokens_per_clip).bool())
        
        #augmentation
        if is_train:
            batch_seq_enc_in_tokens+=torch.randn_like(batch_seq_enc_in_tokens)*self.noise_std
        #Pass to encoder
        batch_enc_in_mu=repeat(self.mid_token_mu_enc_in,'() n d -> b n d',b=batch_size)
        batch_enc_mask_global=torch.zeros_like(batch_seq_enc_in_mask_tokens[:,0:1].detach().clone())

        batch_seq_enc_in=torch.cat((batch_enc_in_mu,batch_seq_enc_in_tokens),dim=1)
        batch_seq_enc_pe=self.transformer_pe(batch_seq_enc_in)
        batch_seq_enc_mask=torch.cat((batch_enc_mask_global,batch_seq_enc_in_mask_tokens),dim=1)

        if verbose:
            print('batch_enc_in_mu/batch_enc_mask_global', batch_enc_in_mu.shape,batch_enc_mask_global.shape)#[bs,1,512],[bs,1]
            print('batch_seq_enc_in/batch_seq_enc_pe/batch_seq_enc_mask', batch_seq_enc_in.shape,batch_seq_enc_pe.shape,batch_seq_enc_mask.shape)#[bs,len_tokens+1,512]x2,[bs,len_tokens+1]
            print(batch_seq_enc_mask[:2])
        
        batch_seq_enc_out, _ =self.encoder(src=batch_seq_enc_in, src_pos=batch_seq_enc_pe, key_padding_mask=batch_seq_enc_mask,verbose=False)      
        batch_enc_out_mu =batch_seq_enc_out[:,0]
        return batch_enc_out_mu
        
