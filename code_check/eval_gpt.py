# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from argparse import ArgumentParser
from itertools import product
import os
import sys
import time

import numpy as np
import roma
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_encode import Trainer
from models.transformer_vqvae import CausalVQVAE, OnlineVQVAE, TransformerVQVAE
from models.poseGPT import poseGPT
from dataset.mocap import MocapDataset, worker_init_fn
from evaluate import classification_evaluation
from utils.ae_utils import get_parameters, get_user, red
from utils.data import get_data_loaders
from utils.body_model import get_trans
from utils.checkpointing import get_last_checkpoint
from utils.constants import mm
from utils.param_count import print_parameters_count #benchmark, 
from utils.stats import AverageMeter, class_accuracy
from utils.variable_length import repeat_last_valid
from utils.visu import visu_sample_gt_rec
from utils.sampling import compute_fid
from utils.amp_helpers import NativeScalerWithGradNormCount as NativeScaler

from meshreg.datasets import collate
from meshreg.netscripts import reloadmodel,get_dataset
from meshreg.netscripts.utils import sample_vis_trj_dec
from meshreg.models.utils_tra import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt, compute_berts_for_strs############utils_tra
from torch.utils.data._utils.collate import default_collate

from meshreg.netscripts import position_evaluator as evaluate
from meshreg.netscripts.position_evaluator import MyMEPE, feed_mymepe_evaluators_hands, MyVAE, feed_myvae_evaluator_hands

from distutils.dir_util import copy_tree
from einops import repeat
import shutil

from meshreg.models.fid_net import FIDNet
import pickle

print('*********Sucessfully import*************')
extend_queries = []
def collate_fn(seq, extend_queries=extend_queries):
    return collate.seq_extend_flatten_collate(seq,extend_queries)#seq_extend_collate(seq, extend_queries)



class GTrainer(Trainer):
    """ Trainer for the generation step (training of the transformer that predicts latent variables) """
    def __init__(self, *, best_val=None, class_conditional=True, gen_eos=False, seqlen_conditional=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.best_val = 1e5 if best_val is None else best_val
        self.class_conditional = class_conditional
        self.seqlen_conditional = seqlen_conditional
        self.gen_eos = gen_eos

        self.base_frame_id=16-1
        self.hand_scaling_factor=10
        self.pose_loss=F.l1_loss

    def get_gt_inputs_feature(self,batch_flatten,verbose=False):
        return_batch={}
        for key in ["valid_frame","hand_size_left","hand_size_right"]:
            return_batch[key]=batch_flatten[key].cuda()

            
        return_batch["batch_action_name_obsv"]=[batch_flatten["action_name"][i] for i in range(0,len(batch_flatten["action_name"]),self.seq_len)]
        
        #return_batch["batch_action_name_obsv"]=["open milk" for i in range(0,len(batch_flatten["action_name"]),self.seq_len)]
        return_batch["batch_action_name_embed"]=compute_berts_for_strs(self.model.model_bert, return_batch["batch_action_name_obsv"], verbose=verbose)

        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten, 
                                        len_seq=self.seq_len, 
                                        spacing=1,
                                        base_frame_id=self.base_frame_id,
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.model.vqvae.placeholder_joints,
                                        with_augmentation=False,#is_train,
                                        compute_local2first=False,#True,
                                        verbose=verbose)
                 
        flatten_hand_comp_gt=flatten_comps["gt"]
        dim_hand_feature=flatten_hand_comp_gt.shape[-1]
        
        batch_seq_hand_comp_gt = flatten_hand_comp_gt.view(-1,self.seq_len, dim_hand_feature)
        batch_seq_valid_features=hand_gts["flatten_valid_features"].view(-1,self.seq_len,dim_hand_feature)
        return_batch["batch_seq_len"]=(return_batch["valid_frame"].cuda().view(-1,self.seq_len)).sum(1)

        batch_seq_valid_frame=return_batch["valid_frame"].view(-1,self.seq_len,1)
        batch_seq_valid_features=torch.mul(batch_seq_valid_features,batch_seq_valid_frame)
        
        return_batch["batch_seq_hand_comp_gt"]=batch_seq_hand_comp_gt
        return_batch["batch_seq_valid_features"]=batch_seq_valid_features
        return_batch.update(hand_gts)

        if verbose:
            for k,v in return_batch.items():
                try:
                    print(k,v.shape)
                except:
                    print(k,len(v))
        
        return return_batch


    def eval_ours(self,data, tag_out,model_fid=None, verbose=False):
        self.model.eval()
        self.model.vqvae.eval()
        evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
        evaluators_pred_local = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

        evaluators_vae = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}
        evaluators_vae_local = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}    


        joint_links=[(0, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),]
        
        valid_joints=[0]+list(range(2,21))
        
        save_dict_fid={"batch_action_name_obsv":[],"batch_enc_out_global_feature":[]}
        with torch.no_grad():
            for batch_idx,batch_flatten in enumerate(tqdm(data)):    
                batch0=self.get_gt_inputs_feature(batch_flatten)
                
                batch_rs_seq_in_cam_pred_out=[]
                batch_rs_seq_in_local_pred_out=[]
                for rs_id in range(21):
                    x=batch0["batch_seq_hand_comp_gt"]
                    valid=batch0["valid_frame"].view(-1,self.seq_len).cuda()

                    if verbose:
                        print("x/valid",x.shape,valid.shape)#[bs,seq_len,feature_dim],[bs,seq_len]
                
                    _, zidx, zvalid = self.model.vqvae.forward_latents(x, valid, return_indices=True, return_mask=True)

                    actions_emb = torch.unsqueeze(self.model.action_to_embedding(batch0["batch_action_name_embed"]),1)
                    #actions_to_embeddings(actions, factor) if self.class_conditional else None
                    seqlens= batch0["batch_seq_len"]
                    seqlens_emb = self.model.seqlens_to_embeddings(seqlens) if self.seqlen_conditional else None

                    len_obsv=self.base_frame_id+1
                    batch_seq_comp_out,batch_seq_valid_out=self.model.sample_poses(zidx,x=x, valid=valid, 
                                        actions_emb=actions_emb,
                                        seqlens_emb=seqlens_emb,
                                        temperature=None,
                                        top_k=20,
                                        cond_steps=len_obsv//2,
                                        return_index_sample=False,
                                        return_zidx=False)
                    
                    assert batch_seq_valid_out.all()

                    trans_info_pred={}
                    for k in ['flatten_firstclip_R_base2cam_left',  'flatten_firstclip_t_base2cam_left', 'flatten_firstclip_R_base2cam_right', 'flatten_firstclip_t_base2cam_right']:
                        trans_info_pred[k]=batch0[k]
                    
                    
                    batch_mean_hand_size_left=torch.mean(batch0['hand_size_left'].view(-1,self.seq_len)[:,:len_obsv],dim=1,keepdim=True)
                    batch_mean_hand_size_right=torch.mean(batch0['hand_size_right'].view(-1,self.seq_len)[:,:len_obsv],dim=1,keepdim=True)

                    
                    results_hand=self.batch_seq_from_comp_to_joints(batch_seq_comp_out[:,len_obsv:],#
                                                    batch_mean_hand_size=(batch_mean_hand_size_left,batch_mean_hand_size_right),
                                                    trans_info=trans_info_pred)

                    results={}
                    for k in ["base","local","cam"]:
                        joints3d_out=results_hand[f"batch_seq_joints3d_in_{k}"]/self.hand_scaling_factor
                        results[f"batch_seq_joints3d_in_{k}_pred_out"]=joints3d_out
                        
                        joints3d_gt=batch0[f"flatten_joints3d_in_{k}_gt"].view(-1,self.seq_len,42,3)/self.hand_scaling_factor
                        results[f"batch_seq_joints3d_in_{k}_gt"]=joints3d_gt
                        results[f"batch_seq_joints3d_in_{k}_obsv_gt"]=joints3d_gt[:,:len_obsv]
                        results[f"batch_seq_joints3d_in_{k}_pred_gt"]=joints3d_gt[:,len_obsv:]

                    batch_rs_seq_in_cam_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_cam_pred_out"],dim=1))
                    batch_rs_seq_in_local_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_local_pred_out"],dim=1))

                    if rs_id==0:
                        batch_seq_weights=valid.view(-1,self.seq_len).float()[:,len_obsv:]
                        feed_mymepe_evaluators_hands(evaluators_pred,results["batch_seq_joints3d_in_cam_pred_out"], 
                                            results["batch_seq_joints3d_in_cam_pred_gt"], 
                                            batch_seq_weights,valid_joints=valid_joints)


                        feed_mymepe_evaluators_hands(evaluators_pred_local,results["batch_seq_joints3d_in_local_pred_out"], 
                                            results["batch_seq_joints3d_in_local_pred_gt"], 
                                        batch_seq_weights,valid_joints=valid_joints)#[1:])

                        batch_seq_in_cam_pred_gt=results["batch_seq_joints3d_in_cam_pred_gt"]
                        batch_seq_in_local_pred_gt=results["batch_seq_joints3d_in_local_pred_gt"]
                                        
                                        
                    if model_fid is not None:
                        num_frames_to_pad=results["batch_seq_joints3d_in_cam_gt"].shape[1]-results["batch_seq_joints3d_in_cam_pred_out"].shape[1]
                        batch_seq_joints3d_in_cam_for_fid=torch.cat([results["batch_seq_joints3d_in_cam_gt"][:,:num_frames_to_pad],results["batch_seq_joints3d_in_cam_pred_out"]],dim=1)
                        batch_seq_joints3d_in_local_for_fid=torch.cat([results["batch_seq_joints3d_in_local_gt"][:,:num_frames_to_pad],results["batch_seq_joints3d_in_local_pred_out"]],dim=1)

                        batch_to_fid={"batch_seq_cam_joints3d_left":batch_seq_joints3d_in_cam_for_fid[:,:,:21],
                                    "batch_seq_cam_joints3d_right":batch_seq_joints3d_in_cam_for_fid[:,:,21:],
                                    "batch_seq_local_joints3d_left":batch_seq_joints3d_in_local_for_fid[:,:,:21],
                                    "batch_seq_local_joints3d_right":batch_seq_joints3d_in_local_for_fid[:,:,21:],
                                    "batch_seq_valid_frame":valid.view(-1,self.seq_len),#results["batch_seq_valid_frame_out"],
                                    "batch_action_name_obsv":batch0["batch_action_name_obsv"]}
                        #batch_to_fid.update(batch)                       
                        with torch.no_grad():
                            _, results_fid, _ =model_fid(batch_to_fid, num_prefix_frames_to_remove=0,batch_is_gt=False,compute_loss=False,verbose=verbose)
                        for k in ["batch_action_name_obsv"]:
                            save_dict_fid[k]+=results_fid[k]
                        for k in ["batch_enc_out_global_feature"]:
                            save_dict_fid[k].append(results_fid[k])

                    
                    
                    if "image_vis" in batch_flatten:
                        for sample_id in range(results["batch_seq_joints3d_in_cam_gt"].shape[0]):
                            cam_info={"intr":batch_flatten["cam_intr"][0].cpu().numpy(),"extr":np.eye(4)}

                            rs_id=0
                            sample_vis_trj_dec(batch_seq_gt_cam=results["batch_seq_joints3d_in_cam_gt"], 
                                        batch_seq_est_cam=results["batch_seq_joints3d_in_cam_pred_out"], 
                                        batch_seq_gt_local=results["batch_seq_joints3d_in_local_gt"],#results["batch_seq_joints3d_in_local_gt"],
                                        batch_seq_est_local=results["batch_seq_joints3d_in_local_pred_out"], #results["batch_seq_joints3d_in_local_out"],
                                        batch_gt_action_name=batch0["batch_action_name_obsv"], 
                                        joint_links=joint_links,  
                                        flatten_imgs=batch_flatten["image_vis"],
                                        sample_id=sample_id,
                                        cam_info=cam_info,
                                        prefix_cache_img=f"./{tag_out}/imgs/", path_video=f"./{tag_out}"+'/{:04d}_{:02d}_{:02d}.avi'.format(batch_idx,sample_id,rs_id))
                
                if False and len(batch_rs_seq_in_cam_pred_out)>1:
                    batch_rs_seq_in_cam_pred_out=torch.cat(batch_rs_seq_in_cam_pred_out,dim=1)
                    batch_rs_seq_in_local_pred_out=torch.cat(batch_rs_seq_in_local_pred_out,dim=1)
                    print("batch_rs_seq_in_cam_pred_out",batch_rs_seq_in_cam_pred_out.shape)
                    print("batch_rs_seq_in_local_pred_out",batch_rs_seq_in_local_pred_out.shape)
                    feed_myvae_evaluator_hands(evaluator=evaluators_vae,batch_rs_seq_joints3d_out=batch_rs_seq_in_cam_pred_out, 
                                    batch_seq_joints3d_gt=batch_seq_in_cam_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=valid_joints)
                    feed_myvae_evaluator_hands(evaluator=evaluators_vae_local,batch_rs_seq_joints3d_out=batch_rs_seq_in_local_pred_out, 
                                batch_seq_joints3d_gt=batch_seq_in_local_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=valid_joints[1:])
        
        save_dict={}
        evaluator_results= evaluate.parse_evaluators(evaluators_pred)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_epe_mean"]=eval_res["epe_mean"]
            #for pid in range(ntokens_pred):
            #    save_dict[f"pred{pid}_{eval_name}_epe_mean"]=eval_res["per_frame_epe_mean"][pid]

        evaluator_results= evaluate.parse_evaluators(evaluators_pred_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
            #for pid in range(ntokens_pred):
            #    save_dict[f"pred{pid}_{eval_name}_local_epe_mean"]=eval_res["per_frame_epe_mean"][pid]
        
        
        
        vae_results=evaluate.parse_evaluators(evaluators_vae)
        for eval_name,eval_res in vae_results.items():
            print(eval_name)
            for kk,vv in eval_res.items():
                save_dict[eval_name+"_"+kk]=vv
                print(kk,"{:.2f}".format(100*vv))
        
        vae_results_local=evaluate.parse_evaluators(evaluators_vae_local)
        for eval_name,eval_res in vae_results_local.items():
            print(eval_name+"_local")
            for kk,vv in eval_res.items():
                save_dict[eval_name+"_local_"+kk]=vv
                print(kk,"{:.2f}".format(100*vv))
        
        
        for k,v in save_dict.items():
            print(k,v)
        
        if model_fid is not None:        
            for k in ["batch_enc_out_global_feature"]:
                save_dict_fid[k]=torch.cat(save_dict_fid[k],dim=0).detach().cpu().numpy()
            with open(f"fid_{tag_out}.pkl", 'wb') as f:
                pickle.dump(save_dict_fid, f)
                        

    '''
    def fid_sample_one_batch(pose_gpt, *, x, valid, actions, device, cond_steps,
            class_conditional, seqlen_conditional, zidx_or_bs, temperature,
            top_k, data_loader, i, preparator):
        """ Sample one batch from the model to evaluate fid on it.
        Used in the inner loop of new_compute_fid"""
        _x, _valid = x, valid
        x, valid, actions = x.to(device), valid.to(device), actions.to(device)
        x, *_ = preparator(x)

        zidx = zidx_or_bs
        if zidx is None or cond_steps > 0 or i == len(data_loader) - 1:
            _, zidx = pose_gpt.vqvae.forward_latents(x, valid, return_indices=True)

        seqlens = valid.sum(1)
        (rotvec, trans), valid, idx = pose_gpt.sample_poses(zidx, actions=actions, seqlens=seqlens, x=x,
                valid=x, temperature=temperature, top_k=top_k, cond_steps=cond_steps, return_index_sample=True)
        sample_valid = _valid.to(device) if pose_gpt.sample_eos_force else valid
        rotvec, trans = [repeat_last_valid(x, sample_valid) for x in [rotvec, trans]]
        poses = torch.cat((rotvec.flatten(2), trans.flatten(2)), dim=-1)
        return poses, sample_valid, zidx

    '''
    
    def batch_seq_from_comp_to_joints(self, batch_seq_comp,batch_mean_hand_size,trans_info, verbose=False):
        results={}
        batch_size,len_seq=batch_seq_comp.shape[0],batch_seq_comp.shape[1]
        hand_left_size=repeat(batch_mean_hand_size[0],'b ()-> b n',n=len_seq)
        hand_right_size=repeat(batch_mean_hand_size[1],'b ()-> b n',n=len_seq)
        flatten_mean_hand_size=[torch.flatten(hand_left_size),torch.flatten(hand_right_size)]

        batch_seq_comp2=batch_seq_comp

        flatten_out=from_comp_to_joints(batch_seq_comp2, flatten_mean_hand_size, factor_scaling=self.hand_scaling_factor,trans_info=trans_info)
        for key in ["base","cam","local"]:        
            results[f"batch_seq_joints3d_in_{key}"]=flatten_out[f"joints_in_{key}"].view(batch_size,len_seq,42,3)
        return results


    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = super(GTrainer, GTrainer).add_trainer_specific_args(parent_parser)
        parser.add_argument("--alpha_root",
                            type=float)  # Not giving them default values to make sure we don't forget. 
        parser.add_argument("--alpha_body",
                            type=float)  # Not giving them default values to make sure we don't forget.
        parser.add_argument("--alpha_trans", type=float, default=1)
        return parser

def get_parsers_and_models(args):
    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=600)
    #parser.add_argument("--data_device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dummy_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--overfit", type=int, default=0)

    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=2)
    parser.add_argument("--class_freq", type=int, default=100)
    parser.add_argument("--visu_freq", type=int, default=200)
    parser.add_argument("--visu_to_tboard", type=int, default=int(get_user() == 'tlucas'), choices=[0,1])
    parser.add_argument("--ckpt_freq", type=int, default=5)
    parser.add_argument("--fid_freq", type=int, default=40)
    parser.add_argument("--restart_ckpt_freq", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--n_visu_to_save", type=int, default=0)
    parser.add_argument("--train_data_dir", type=str, default='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--val_data_dir", type=str, default='data/smplx/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--n_train", type=int, default=1000000)
    parser.add_argument("--n_iter_val", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default='../ckpts_panda/checkpoints/posegpt')
    parser.add_argument("--name", type=str, default='debug')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vq_model", type=str, default='OnlineVQVAE')
    parser.add_argument('--vq_ckpt', type=str, default=None)
    parser.add_argument("--model", type=str, default='generator.GPT_Transformer')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--detailed_count", type=int, default=0)
    parser.add_argument("--class_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seqlen_conditional", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gen_eos", type=int, default=0, choices=[0,1])
    parser.add_argument("--eos_force", type=int, default=1, choices=[0,1])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    parser.add_argument("--data_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_start", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_classif", type=int, default=0, choices=[0, 1])
    parser.add_argument("--concat_emb", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_fid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--head_type", type=str, default='fc_wo_bias', choices=['fc_wo_bias', 'mlp'])
    parser.add_argument("--pos_emb", type=str, default='scratch', choices=['scratch', 'sine_frozen', 'sine_ft', 'scratch_frozen'])
    parser.add_argument("--seqlen_emb", type=str, default='scratch', choices=['scratch', 'sine_frozen', 'sine_ft', 'scratch_frozen'])
    parser.add_argument("--action_emb", type=str, default='scratch', choices=['scratch', 'glove_frozen', 'glove_ft', 'scratch_frozen'])

    parser.add_argument("--use_amp", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fid_path", type=str, default=None)
    

    #dataset parameters
    parser.add_argument('--dataset_folder',default='../')
    
    # Dataset params
    parser.add_argument("--val_dataset", choices=["h2o","ass101","asshand"],  default="h2o")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--val_view_id", default=-1, type=int)
    parser.add_argument("--min_window_sec",type=float,default=2,help="min window length in sec to the end of video/trimmed action")




    script_args, _ = parser.parse_known_args(args)
    assert script_args.vq_model in ['CausalVQVAE', 'OnlineVQVAE', 'OfflineVQVAE'], "Invalid VQVAE model"
    VQModel = eval(script_args.vq_model)
    Model = eval(script_args.model)
    parser = GTrainer.add_trainer_specific_args(parser)
    parser = VQModel.add_model_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    return parser, VQModel, Model

def get_data(args, user):
    kwargs={"action_taxonomy_to_use": "fine", "max_samples":-1}#,"e4"]}

    val_dataset = get_dataset.get_dataset_motion([args.val_dataset],
                            list_splits=[args.val_split],   
                            list_view_ids=[args.val_view_id],
                            dataset_folder=args.dataset_folder,
                            use_same_action=True,
                            ntokens_per_clip=16,#args.seq_len,#args.ntokens_per_clip,
                            spacing=1.,#args.spacing,
                            nclips=args.seq_len//16,
                            min_window_sec=args.min_window_sec,#(args.ntokens_per_clip*args.spacing/30.)*2,
                            is_shifting_window=True,
                            dict_is_aug={},
                            **kwargs,)

    loader_val = get_dataset.DataLoaderX(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch_factor,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,)

    return loader_val


def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user = get_user()
    parser, VQModel, Model = get_parsers_and_models(args)

    args = parser.parse_args(args)
    args.factor = 2 if isinstance(args.n_layers, int) else 2 ** len(args.n_layers)

    loader_val = get_data(args, user)
    print(f"\nBuilding the quantization model...")
    print(args)

    in_dim = 153-12#((loader_train.dataset.pose[0].size(1) // 3) - 1) * 6 + 3  # jts in 6D repr, trans in 3d coord
    vq_model = VQModel(in_dim=in_dim, **vars(args)).to(device)


    vq_ckpt = torch.load(args.vq_ckpt)
    weights = vq_ckpt['model_state_dict']
    weights = {k.replace('log_sigmas.verts', 'log_sigmas.vert'): v for (k, v) in weights.items()}
    missing, unexpected = vq_model.load_state_dict(weights, strict=False)
    assert len(unexpected) == 0, "Unexpected keys"
    assert all(['log_sigmas' in m for m in missing]), "Missing keys: " + ','.join([m for m in missing if 'log_sigmas' not in m])

    if 'balance_stats' in vq_ckpt.keys():
        bins = vq_ckpt['balance_stats']
        vq_model.quantizer.load_state(bins)
        
    model = Model(**vars(args), vqvae=vq_model).to(device)
    print("VQ model parameter count: ")
    print_parameters_count(model.vqvae, detailed=args.detailed_count, tag='VQ - ')
    print("Transformer model parameter count: ")
    print_parameters_count(model.gpt, detailed=args.detailed_count, tag='GPT - ')

    print(f"Number of parameters: {get_parameters(model):,}")
    checkpoint = torch.load(args.pretrained_ckpt)
    ckpt_path = args.pretrained_ckpt

    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    assert len(unexpected) == 0, "Unexpected keys"
    assert all(['log_sigmas' in m for m in missing]), "Missing keys: " + ','.join([m for m in missing if 'log_sigmas' not in m])

    epoch, saved_iter = 1, 0
    bv, bc = None, None

    # Trainer
    print(f"\nSetting up the trainer...")
    trainer = GTrainer(model=model, optimizer=None, device=device,
                       args=args, epoch=epoch, start_iter=saved_iter,
                       best_val=bv, 
                       class_conditional=args.class_conditional,
                       seqlen_conditional=args.seqlen_conditional,
                       gen_eos=args.gen_eos,
                       seq_len=args.seq_len, 
                       loss_scaler=None)
    tag_out=f"vis_{args.val_dataset}_{args.val_split}_view_id{args.val_view_id}_minwindow{args.min_window_sec}_seqlen{args.seq_len}"


    model_fid=None
    if args.fid_path!="":
        model_fid=FIDNet(transformer_d_model=512,
                            transformer_nhead=8,
                            transformer_dim_feedforward=2048,
                            transformer_nlayers_enc=9,
                            transformer_activation="gelu",
                            
                            ntokens_per_video=256,
                            spacing=1, 
                            code_loss="l1")
                            
        reloadmodel.reload_model(model_fid,args.fid_path) 
        model_fid=model_fid.cuda()


    trainer.eval_ours(loader_val,tag_out, model_fid)
if __name__ == "__main__":
    main()
