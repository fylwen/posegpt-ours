# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from argparse import ArgumentParser
from functools import partial
import sys,os
import time
import warnings

import numpy as np
import roma
import smplx
import torch
import torch.nn.functional as torch_f
from einops import repeat
from tqdm import tqdm

from models.transformer_vqvae import CausalVQVAE, OnlineVQVAE, TransformerVQVAE, OfflineVQVAE
from trainer import Trainer
from utils.ae_losses import gaussian_nll, laplacian_nll
from utils.ae_utils import get_parameters, get_user, red
from utils.data import get_data_loaders
from utils.body_model import get_trans, pose_to_vertices
from utils.checkpointing import get_last_checkpoint
from utils.constants import SMPLX_DIR, MB
from utils.log_helpers import add_histogram
from utils.param_count import print_parameters_count
from utils.stats import AverageMeter
from utils.utils import (count_dim, subsamble_random_offset,
    valid_reduce as _valid_reduce)
from utils.amp_helpers import NativeScalerWithGradNormCount as NativeScaler



from meshreg.datasets import collate
from meshreg.netscripts import reloadmodel,get_dataset
from meshreg.models.utils_tra import loss_str2func,get_flatten_hand_feature, from_comp_to_joints, load_mano_mean_pose, get_inverse_Rt
from torch.utils.data._utils.collate import default_collate

from distutils.dir_util import copy_tree
import shutil


print('*********Sucessfully import*************')
extend_queries = []
def collate_fn(seq, extend_queries=extend_queries):
    return collate.seq_extend_flatten_collate(seq,extend_queries)#seq_extend_collate(seq, extend_queries)


if not sys.warnoptions:
    warnings.simplefilter("ignore")

class QTrainer(Trainer):
    """ Trainer specialized for the auto-encoder based quantization step. """

    def __init__(self, *, best_val=None, best_class=None, **kwargs):
        super().__init__(**kwargs)

        self.best_val = 1e5 if best_val is None else best_val
        self.best_class = -1e5 if best_class is None else best_class

        '''
        if hasattr(self.args, 'tprop_vert') and self.args.tprop_vert != 1.:
            """ Apply loss on vertices, for certain time steps only"""
            self.tdim = int(self.seq_len * self.args.tprop_vert)
            nb_poses = self.tdim * int(self.args.train_batch_size * self.args.prop_vert)
            self.bm_light = smplx.create(SMPLX_DIR, self.type, use_pca=False, batch_size=nb_poses).to(self.device)
            self.pose_to_vertices_light = partial(pose_to_vertices, pose_type=self.type,
                                                  alpha=self.args.alpha_trans, bm=self.bm_light,
                                                  parallel=True)
        elif self.args.alpha_vert > 0.:
            self.tdim = self.seq_len
            print("Warning: this will be slow. If you are sure, discard & proceed.")
            sys.exit(0)
        '''

        self.base_frame_id=0
        self.hand_scaling_factor=10
        self.pose_loss=torch_f.l1_loss

    def get_gt_inputs_feature(self,batch_flatten,verbose=False):
        return_batch={}
        for key in ["valid_frame","hand_size_left","hand_size_right"]:
            return_batch[key]=batch_flatten[key].cuda()
        if not "batch_action_name_obsv" in batch_flatten.keys():
            return_batch["batch_action_name_obsv"]=batch_flatten["action_name"][0::self.seq_len]
        else:
            return_batch["batch_action_name_obsv"]=batch_flatten["batch_action_name_obsv"]
            
        flatten_comps, hand_gts = get_flatten_hand_feature(batch_flatten, 
                                        len_seq=self.seq_len, 
                                        spacing=1,
                                        base_frame_id=self.base_frame_id,
                                        factor_scaling=self.hand_scaling_factor, 
                                        masked_placeholder=self.model.placeholder_joints,
                                        with_augmentation=False,#is_train,
                                        compute_local2first=False,#True,
                                        verbose=verbose)
                 
        flatten_hand_comp_gt=flatten_comps["gt"]
        dim_hand_feature=flatten_hand_comp_gt.shape[-1]
        
        batch_seq_hand_comp_gt = flatten_hand_comp_gt.view(-1,self.seq_len, dim_hand_feature)
        batch_seq_valid_features=hand_gts["flatten_valid_features"].view(-1,self.seq_len,dim_hand_feature)
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
    
    
    def batch_seq_from_comp_to_joints(self, batch_seq_comp,batch_mean_hand_size,trans_info, normalize_size_from_comp,batch_seq_valid_features=None, verbose=False):
        results={}
        batch_size,len_seq=batch_seq_comp.shape[0],batch_seq_comp.shape[1]
        hand_left_size=repeat(batch_mean_hand_size[0],'b ()-> b n',n=len_seq)
        hand_right_size=repeat(batch_mean_hand_size[1],'b ()-> b n',n=len_seq)
        flatten_mean_hand_size=[torch.flatten(hand_left_size),torch.flatten(hand_right_size)]

        if normalize_size_from_comp:            
            assert False, "Not implemented yet"
        else:
            batch_seq_comp2=batch_seq_comp

        flatten_out=from_comp_to_joints(batch_seq_comp2, flatten_mean_hand_size, factor_scaling=self.hand_scaling_factor,trans_info=trans_info)
        for key in ["base","cam","local"]:        
            results[f"batch_seq_joints3d_in_{key}"]=flatten_out[f"joints_in_{key}"].view(batch_size,len_seq,42,3)
        results["batch_seq_local2base"]=flatten_out["local2base"].view(batch_size,len_seq,flatten_out["local2base"].shape[-1])
        results["batch_seq_trans_info"]=flatten_out["batch_seq_trans_info"]
        return results


    def compute_hand_loss(self, batch_seq_comp_gt, batch_seq_comp_out, batch_seq_valid_features, compute_local2base, batch_seq_local2base_gt,
                 batch_mean_hand_size, trans_info,normalize_size_from_comp, verbose=False):
                 
        losses={}
        results={}
        total_loss=0.
        if verbose:
            print("batch_seq_comp_gt/batch_seq_comp_out,batch_seq_valid_features",batch_seq_comp_gt.shape,batch_seq_comp_out.shape,batch_seq_valid_features.shape)
         
        recov_hand_loss=self.pose_loss(batch_seq_comp_gt,batch_seq_comp_out,reduction='none')
        if verbose:
            print('recov_hand_loss',torch.abs(recov_hand_loss).max(),recov_hand_loss.shape)#[bs,len_seq,144]
        recov_hand_loss=torch.mul(recov_hand_loss,batch_seq_valid_features)
        cnt=torch.sum(batch_seq_valid_features)
        losses["recov_hand_loss"]=torch.sum(recov_hand_loss)/torch.where(cnt<1.,1.,cnt)#torch.mean(recov_hand_loss) 
        total_loss+=losses["recov_hand_loss"]

        if not compute_local2base:
            return total_loss,results,losses

        #trjectory only for pred
        output_results=self.batch_seq_from_comp_to_joints(batch_seq_comp=batch_seq_comp_out,
                                                        batch_mean_hand_size=batch_mean_hand_size,
                                                        trans_info=trans_info,
                                                        normalize_size_from_comp=normalize_size_from_comp, 
                                                        batch_seq_valid_features=batch_seq_valid_features,
                                                        verbose=verbose)
        
        batch_seq_local2base_out=output_results["batch_seq_local2base"]
        recov_trj_loss=self.pose_loss(batch_seq_local2base_gt,batch_seq_local2base_out,reduction='none')
        batch_seq_trj_valid=batch_seq_valid_features[:,:,42*3:42*3+6]#18]
        recov_trj_loss=torch.mul(batch_seq_trj_valid,recov_trj_loss)
        
        cnt=torch.sum(batch_seq_trj_valid)
        losses["recov_trj_in_base_loss"]=torch.sum(recov_trj_loss)/torch.where(cnt<1.,1.,cnt)
        total_loss+=losses["recov_trj_in_base_loss"]
        
        for k in output_results.keys():
            if "joints" in k:
                results[k+"_out"]=output_results[k]
            if "batch_seq_comp_local_normalized" in k:
                results[k]=output_results[k]
                
        return total_loss,results,losses

    def forward_one_batch(self, batch_flatten, loss_type,training=True,verbose=False):
        # Forward model
        batch0=self.get_gt_inputs_feature(batch_flatten,verbose=verbose)

        output_comp, loss_z, indices = self.model(batch0,verbose=verbose)
        
        
        #trans_hat = get_trans(trans_delta_hat, valid)

        #verts, verts_hat, verts_valid = None, None, None
        #if training:
        #    # Convert smpl sequence to sequence of vertices
        #    verts, verts_hat, verts_valid = self.params_to_vertices(rotmat_hat, rotvec, trans_gt, trans_hat, valid)
        
        # Define masked reductions
        #def valid_reduce(x, mask=None, reduction='sum'):
        #    """ Accounts for 0 padding in shorter sequences """
        #    if len(x.shape) == 1 and x.shape[0] == 1: # Just a scalar
        #        return x.sum() if reduction == 'sum' else x.mean()
        #    mask = (valid if valid.shape[1] == x.shape[1] else
        #            verts_valid) if mask is None else mask
        #    return _valid_reduce(x, mask, reduction)

        #valid_sum = partial(valid_reduce, reduction='sum')
        #valid_mean = partial(valid_reduce, reduction='mean')

        # Predictions, targets and ground truths
        #pred = {'trans': trans_hat, 'vert': verts_hat, 'body': rotmat_hat[:, :, 1:, ...],
        #        'root': rotmat_hat[:, :, 0, ...].unsqueeze(2)}
        #gt = {'trans': trans_gt, 'vert': verts, 'body': rotmat[:, :, 1:, ...],
        #      'root': rotmat[:, :, 0, ...].unsqueeze(2)}

        # Variance can be learned; not used by default
        #log_sigmas = self.model.log_sigmas if loss_type in ['gaussian', 'laplacian'] else \
        #    {k: torch.zeros((1), requires_grad=False).to(x.device) for k in pred.keys()}

        #nll_loss = gaussian_nll if loss_type in ['gaussian', 'l2'] else laplacian_nll
        #nll_verts_loss = laplacian_nll if not self.args.l2_verts else gaussian_nll
        #nll = {k: nll_loss(pred[k], gt[k], log_sigmas[k]) for k in ['trans', 'root', 'body']}
        #nll.update({'vert': nll_verts_loss(pred['vert'], gt['vert'], log_sigmas['vert'])})

        # Energy and norm seperated for logging;
        #energy_values, norm_values, nll_values = [{k: nll[k][n] for k in nll.keys()} for n in ['energy', 'norm', 'nll']]

        # Elbo computations (for logging, not optimized by sgd)
        #elbo_params, elbo_verts, valid_kl = self.compute_elbos(nll_values, valid_sum, loss_z)

        # Gather losses and multiply by the right coefficients
        #losses = ['root', 'body', 'trans'] + (['vert'] if pred['vert'] is not None else [])
        #total_loss = sum([getattr(self.args, 'alpha_' + k) * valid_mean(nll_values[k]).mean(0) for k in losses])
        batch_seq_valid_frame=batch0["valid_frame"].view(-1,self.seq_len)
        batch_seq_hand_size_left=torch.mul(batch0['hand_size_left'],batch0["valid_frame"]).view(-1,self.seq_len)
        batch_mean_hand_left_size=torch.sum(batch_seq_hand_size_left,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame,dim=1,keepdim=True)

        batch_seq_hand_size_right=torch.mul(batch0['hand_size_right'],batch0["valid_frame"]).view(-1,self.seq_len)
        batch_mean_hand_right_size=torch.sum(batch_seq_hand_size_right,dim=1,keepdim=True)/torch.sum(batch_seq_valid_frame,dim=1,keepdim=True)


        trans_info={}
        for k in ['flatten_firstclip_R_base2cam_left','flatten_firstclip_t_base2cam_left','flatten_firstclip_R_base2cam_right','flatten_firstclip_t_base2cam_right']:
            trans_info[k]=batch0[k]       

        compute_local2base=True
        total_loss,results_hand,loss_hand=self.compute_hand_loss(batch_seq_comp_gt=batch0["batch_seq_hand_comp_gt"], 
                                                            batch_seq_comp_out=output_comp, 
                                                            batch_seq_valid_features=batch0["batch_seq_valid_features"], 
                                                            compute_local2base=compute_local2base,
                                                            batch_seq_local2base_gt=batch0['flatten_tra_local2base_gt'].view(batch_seq_valid_frame.shape[0],batch_seq_valid_frame.shape[1],-1),
                                                            batch_mean_hand_size=(batch_mean_hand_left_size,batch_mean_hand_right_size),
                                                            trans_info=trans_info,
                                                            normalize_size_from_comp=False,verbose=verbose)


        results={}
        if compute_local2base:
            for k in ["base","local","cam"]:
                joints3d_out=results_hand[f"batch_seq_joints3d_in_{k}_out"]/self.hand_scaling_factor
                results[f"batch_seq_joints3d_in_{k}_pred_out"]=joints3d_out
                
                joints3d_gt=batch0[f"flatten_joints3d_in_{k}_gt"].view(joints3d_out.shape)/self.hand_scaling_factor
                results[f"batch_seq_joints3d_in_{k}_gt"]=joints3d_gt

                if verbose:
                    print(k,torch.abs(joints3d_gt-joints3d_out).max())

        total_loss += self.args.alpha_codebook * loss_z['quant_loss']
        # Putting usefull statistics together (for tensorboard)
        statistics = {#'elbo/valid_kl': valid_kl, 'elbo/params': elbo_params, 'elbo/verts': elbo_verts,
                      'total_loss': total_loss,'quant_loss': loss_z['quant_loss'] if 'quant_loss' in loss_z else 0}
        statistics.update(loss_hand)
        
        #for tag, vals in zip(['', '_energy', '_norm'], [nll_values, energy_values, norm_values]):
        #    statistics.update({'nll/' + k + tag: valid_mean(vals[k]).mean(0) for k in nll_values.keys()})

        # Return pose predictions (smpl and vertices), centroid indices and mask)
        #outputs = {'rotmat_hat': rotmat_hat, 'trans_hat': trans_hat, 'indices': indices, 'valid': valid}
        return total_loss, statistics, results

    def compute_elbos(self, nll_values, valid_sum, loss_z):
        """ Elbos are usefull for logging (put reconstruction and KL together) in principle."""
        params_dim = sum([count_dim(nll_values[k]) for k in ['root', 'body', 'trans']])
        valid_kl = valid_sum(loss_z['kl'], loss_z['kl_valid'])
        elbo_params = - (torch.stack([valid_sum(nll_values[k])
                         for k in ['root', 'body', 'trans']]).sum(0) + valid_kl).mean(0) / params_dim
        elbo_verts = torch.zeros((1), requires_grad=False).to(self.device)
        if 'vert' in nll_values and len(nll_values['vert'].shape) > 1:
            elbo_verts = - (valid_sum(nll_values['vert']) + valid_kl).mean(0) / count_dim(nll_values['vert'])
        return elbo_params, elbo_verts, valid_kl

    def params_to_vertices(self, rotmat_hat, rotvec, trans_gt, trans_hat, valid, use_fast_smpl=False):
        """ Maps smpl parameters to vertices for some time steps """
        tag = '' if not use_fast_smpl else 'fast_'
        prop_vert, freq_vert = [getattr(self.args, tag + k) for k in  ['prop_vert', 'freq_vert']]
        verts, verts_hat, verts_valid = None, None, None
        compute_vert = (self.current_iter % freq_vert == 0)
        if self.args.alpha_vert > 0 and compute_vert:
            rotvec_hat = roma.rotmat_to_rotvec(rotmat_hat)
            bs = int(self.args.train_batch_size * prop_vert)
            period = int(1 / self.args.tprop_vert)
            # Select evenly spaced time steps, with random offset at start, to compute faster.
            _rotvec, _trans, _rotvec_hat, _trans_hat, verts_valid = subsamble_random_offset(bs, period, self.tdim,
                    [rotvec, trans_gt, rotvec_hat, trans_hat, valid])
            ptv = self.pose_to_vertices if self.args.tprop_vert == 1.0 else self.pose_to_vertices_light
            verts, verts_hat = [ptv(torch.cat([r.flatten(2), t], -1))
                                for r, t in zip([_rotvec, _rotvec_hat], [_trans, _trans_hat])]
        return verts, verts_hat, verts_valid
    
    def train_n_iters(self, data, loss_type):
        """ Do a pass on the dataset; sometimes log statistics"""
        self.model.train()

        data_time, batch_time, max_mem = [AverageMeter(k, ':6.3f') for k in ['data_time', 'batch_time', 'max_mem']]
        average_meters = {'data_time': data_time, 'batch_time': batch_time}

        end = time.time()
        print(red("> Training auto-encoder..."))
        for batch_idx, batch_flatten in enumerate(tqdm(data)):
            data_time.update(time.time() - end)

            # Input preparation
            #x, valid = x.to(self.device), valid.to(self.device)
            #x, rotvec, rotmat, trans_gt, _ = prepare_input(x)
            #x_noise, rotvec, rotmat, trans_gt, _, _ = self.preparator(x)


            # TODO refactor with a context manager to avoid code duplication.
            if self.args.use_amp:
                assert False, "AMP not supported yet."
                assert self.loss_scaler is not None, "Need a loss scaler for AMP."
                with torch.cuda.amp.autocast():
                    total_loss, statistics, outputs = self.forward_one_batch(x_noise, actions, valid,
                            loss_type, trans_gt, rotmat, rotvec)
                self.optimizer.zero_grad()
                self.loss_scaler(total_loss, self.optimizer, parameters=self.model.parameters(),
                                 update_grad=True)

            else:
                total_loss, statistics, outputs = self.forward_one_batch(batch_flatten,loss_type)

                # optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Track time and memory
            batch_time.update(time.time() - end)
            max_mem.update(torch.cuda.max_memory_allocated() / (MB))
            end = time.time()

            # Update metrics
            if len(average_meters) == 2:
                average_meters.update({k: AverageMeter(k, ':6.3f') for k in statistics.keys()})
            for k in statistics.keys():
                average_meters[k].update(statistics[k].mean())
            self.current_iter += 1

        for k, v in average_meters.items():
            if 'nll' in k or 'total' in k:
                print(f"    - {k}: {v.avg:.3f}")
        
        self.log_train_statistics(average_meters)
        #self.log_compute_efficiency(batch_time, data_time, max_mem)

    def visu_rec(self, rotvec, trans_gt, outputs, valid, epoch, is_train=True):
        gt = torch.cat([rotvec.flatten(2), trans_gt], -1)
        pred = torch.cat([roma.rotmat_to_rotvec(outputs['rotmat_hat']).flatten(2), outputs['trans_hat']], -1)
        verts, verts_hat = [self.pose_to_vertices(x) for x in [gt, pred]]
        samples = None
        self.save_visu(verts_hat, verts, valid, samples,
                self.current_iter, self.args.visu_to_tboard, is_train=is_train,
                tag='auto_encoding')

    def log_train_statistics(self, average_meters):
        """ Log statistics to tensorboard and console, reset the average_meters. """
        #if not (self.current_iter % (self.args.log_freq - 1) == 0 and self.current_iter > 0):
        #    return
        for k, v in average_meters.items():
            self.writer.add_scalar(f"train/{k}", v.avg, self.current_epoch)
            v.reset()
        for k, v in self.model.log_sigmas.items():
            self.writer.add_scalar(f"train/log_sigmas_{k}", v.data.detach(), self.current_epoch)

        # Tracking centroid usage with histograms and a score
        centroid_balance_scores = []
        if hasattr(self.model, 'quantizer'):
            for k in self.model.quantizer.embeddings.keys():
                hist = self.model.quantizer.get_hist(int(k))
                if hist is not None:
                    hist = hist.cpu().numpy()
                    add_histogram(writer=self.writer, tag='train_stats/z_histograms_' + k,
                                  hist=hist, global_step=self.current_iter)
                    centroid_balance_scores.append(1 - np.abs((1 - hist*hist.shape[-1])).mean())
            self.writer.add_scalar(f"train/centroid_balance_score", np.mean(centroid_balance_scores), self.current_epoch)

    def log_compute_efficiency(self, batch_time, data_time, max_mem):
        """ Measuring computation efficiency """
        self.writer.add_scalar(f"gpu_load/batch_time", batch_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/it_per_sec", 1. / (batch_time.avg +
                               (1e-4 if self.args.debug else 1e-4)), self.current_iter)
        self.writer.add_scalar(f"gpu_load/data_time", data_time.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/max_mem", max_mem.avg, self.current_iter)
        self.writer.add_scalar(f"gpu_load/max_mem_ratio", max_mem.avg / (torch.cuda.memory_reserved() / MB), self.current_iter)
        print(f"    - batch_time: {batch_time.avg:.3f}")
        print(f"    - data_time: {data_time.avg:.3f}")
    
    def eval(self, data, *, loss_type, epoch, save_to_tboard):
        """ Run the model on validation data; no optimization"""
        #pve, pve_wo_trans, pve_diff = [AverageMeter(k, ':6.3f') for k in ['pve', 'pve_wo_trans', 'pve_diff']]
        average_meters = {}#{'pve': pve, 'pve_wo_trans': pve_wo_trans, 'pve_diff': pve_diff}

        self.model.eval()
        #nb_visu_saved, need_more_visu = 0, True
        with torch.no_grad():
            print(red("> Evaluating auto-encoder..."))
            for batch_flatten in tqdm(data):
                total_loss, statistics, outputs = self.forward_one_batch(batch_flatten,loss_type)

            
                #for x, valid, actions in tqdm(data):
                #    x, valid = x.to(self.device), valid.to(self.device)
                #    x_noise, rotvec, rotmat, trans_gt, _, _ = self.preparator(x)
                #    _, statistics, outputs = self.forward_one_batch(x_noise, actions, valid, loss_type,
                #                                                    trans_gt, rotmat, rotvec, training=False)
                #    err, err_wo_trans, verts_hat, verts = self.eval_pve(
                #        rotvec, outputs['rotmat_hat'], trans_gt, outputs['trans_hat'], valid)

                    # Logging
                #    if len(average_meters) == 3:
                #        average_meters.update({k: AverageMeter(k, ':6.3f') for k in statistics.keys()})
                
                if len(average_meters) == 0:
                    average_meters.update({k: AverageMeter(k, ':6.3f') for k in statistics.keys()})
                for k in statistics.keys():
                    average_meters[k].update(statistics[k].mean())
                #    for k, v in zip([pve, pve_wo_trans, pve_diff], [err, err_wo_trans, err - err_wo_trans]):
                #        k.update(v)

                # Save visu
                #do_visu = self.args.n_visu_to_save > 0 and nb_visu_saved < self.args.n_visu_to_save and epoch % self.args.visu_freq == 0
                #if do_visu and need_more_visu:
                #    samples = None
                #    self.save_visu(verts_hat, verts, valid, samples, self.current_iter, save_to_tboard)
                #    need_more_visu = False

            print(red(f"VAL:"))
            for k, v in average_meters.items():
                print(f"    - {k}: {v.avg:.3f}")
                self.writer.add_scalar(('val/' + k),v.avg,self.current_epoch)#if 'pve' not in k else ('pves/' + k), v.avg, self.current_iter)
                #if k == 'pve':
                #    self.writer.add_scalar('pve', v.avg, self.current_iter)
        return #pve.avg

    def fit(self, data_train, data_val, *, loss='l2'):
        """
        Train and evaluate a model using training and validation data
        """
        while self.current_epoch <= self.args.max_epochs:
            epoch = self.current_epoch
            sys.stdout.flush()

            print(f"\nEPOCH={epoch:03d}/{self.args.max_epochs} - ITER={self.current_iter}")
            # Shuffle training data and vqvae_v1 for n iters
            self.train_n_iters(data_train, loss_type=loss)

            if epoch % self.args.val_freq == 0:
                # Validate the model
                self.eval(data_val, loss_type=loss,
                                epoch=epoch,
                                save_to_tboard=self.args.visu_to_tboard)
                # Save ckpt
                #if val < self.best_val:
                #    self.checkpoint(tag='best_val', extra_dict={'pve': val})
                #    self.best_val = val

            if epoch % self.args.ckpt_freq == 0:# and epoch > 0:
                self.checkpoint(tag='ckpt_' + str(epoch), extra_dict={'best_val': self.best_val,
                    'best_class': self.best_class})
            #if epoch % self.args.restart_ckpt_freq == 0 and epoch > 0:
            #        # This one is saved more frequently but erases itself to save memory. Usefull for best-effort models. 
            #    self.checkpoint(tag='ckpt_restart', extra_dict={'best_val': self.best_val,
            #        'best_class': self.best_class})
            self.current_epoch += 1
        return None

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = super(QTrainer, QTrainer).add_trainer_specific_args(parent_parser)
        parser.add_argument("--alpha_root", type=float, default=1)
        parser.add_argument("--alpha_body", type=float, default=1)
        parser.add_argument("--alpha_trans", type=float, default=1)
        parser.add_argument("--alpha_vert", type=float, default=100)
        parser.add_argument("--alpha_fast_vert", type=float, default=0.)
        parser.add_argument("--alpha_codebook", type=float, default=0.25)
        parser.add_argument("--alpha_kl", type=float, default=1.)
        parser.add_argument("--freq_vert", type=int, default=1)
        parser.add_argument("--prop_vert", type=float, default=1.)
        parser.add_argument("--tprop_vert", type=float, default=0.1)
        parser.add_argument("--vert_string", type=str, default=None)


        return parser

def main(args=None):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=2000)
    #parser.add_argument("--data_device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dummy_data", type=int, default=0, choices=[0, 1])
    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=5000)
    parser.add_argument("--val_freq", type=int, default=2)
    parser.add_argument("--ckpt_freq", type=int, default=5)
    parser.add_argument("--restart_ckpt_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=2000)

    # Parameters for the classif evaluation
    parser.add_argument("--class_freq", type=int, default=-1)
    parser.add_argument("--fid_freq", type=int, default=20)
    parser.add_argument("--visu_freq", type=int, default=50)
    parser.add_argument("--train_visu_freq", type=int, default=50)
    parser.add_argument("--visu_to_tboard", type=int, default=int(get_user() == 'tlucas'), choices=[0,1])
    parser.add_argument("--n_visu_to_save", type=int, default=2)
    parser.add_argument("--train_data_dir", type=str,
            default='data/smplx/babel_trimmed/train_60/seqLen900_fps30_overlap0_minSeqLen16')
    parser.add_argument("--val_data_dir", type=str,
            default='data/smplx/babel_trimmed/val_60/seqLen900_fps30_overlap0_minSeqLen16')

    # -train_data_dir  --val_data_dir
    parser.add_argument("--n_train", type=int, default=1000000)
    parser.add_argument("--n_iter_val", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default='../ckpts_panda/checkpoints/posegpt')
    parser.add_argument("--name", type=str, default='debug')

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-b_train", type=int, default=64)
    parser.add_argument("--val_batch_size", "-b_val", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # parser.add_argument("--bench_w", type=int, default=0)
    # parser.add_argument("--bench_t", type=int, default=0)

    parser.add_argument("--model", type=str, default='conv.Resnet')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument("--eval_only", type=int, default=0, choices=[0, 1])

    # optim parameters
    parser.add_argument("--ab1", type=float, default=0.95, help="Adam beta 1 parameter")
    parser.add_argument("--ab2", type=float, default=0.999, help="Adam beta 2 parameter")

    # Loss choices
    parser.add_argument("--loss", type=str, default='l2', choices=['l2', 'l1', 'laplacian', 'gaussian'])
    parser.add_argument("--l2_verts", type=int, default=0, choices=[0, 1])

    # Print a list of all layers.
    parser.add_argument("--detailed_count", type=int, default=0)

    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vq_seq_len", type=int, default=64)
    parser.add_argument("--num_workers", "-j", type=int, default=16)
    parser.add_argument("--data_augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--sample_start", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dummy", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_classif", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eval_fid", type=int, default=0, choices=[0, 1])
    parser.add_argument("--class_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seqlen_conditional", type=int, default=1, choices=[0, 1])
    parser.add_argument("--classif_ckpt", type=str, default=None)
    parser.add_argument("--classif_ckpt_babel", type=str, default=None)
    parser.add_argument("--eos_force", type=int, default=1, choices=[0,1])
    parser.add_argument("--use_amp", type=int, default=0, choices=[0, 1])


    #dataset parameters
    parser.add_argument('--dataset_folder',default='../')
    
    # Dataset params
    parser.add_argument("--train_datasets", choices=["h2o", "ass101","asshand"], default=["ass101"],nargs="+")
    parser.add_argument("--train_splits", default=["train"], nargs="+")
    parser.add_argument("--val_datasets", choices=["h2o","ass101","asshand"],  default=["h2o"], nargs="+")
    parser.add_argument("--val_splits", default=["val"], nargs="+")


    script_args, _ = parser.parse_known_args(args)
    print("build model with",script_args.model)
    Model = {'CausalVQVAE': CausalVQVAE, 'OnlineVQVAE': OnlineVQVAE,
            'TransformerVQVAE': TransformerVQVAE, 'OfflineVQVAE': OfflineVQVAE}[script_args.model]

    parser = QTrainer.add_trainer_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args(args)
    
    copy_tree("./",os.path.join(args.save_dir,args.name,'code'))
    #cfile_name=os.path.realpath(__file__).split("/")[-1]
    #shutil.copyfile(cfile_name,os.path.join(exp_id,cfile_name))



    try:
        args.factor = np.prod(args.pool_kernel) # temporal downsampling
    except:
        args.factor = 1


    # Data
    print(f"\nLoading data...")
    #loader_train, loader_val = get_data_loaders(args)
    #args.type = loader_train.dataset.type

    #known_datadirs_to_classifier = {'babel': args.classif_ckpt_babel}
    #matching = [k for k in known_datadirs_to_classifier.keys() if k in loader_train.dataset.data_dir]
    #args.dataset_type = matching[0] if len(matching) else 'unknown'
    #if args.classif_ckpt is None:
    #    assert len(matching) == 1, "Unknow data dir, provide classif_ckpt manually"
    #    args.classif_ckpt = known_datadirs_to_classifier[args.dataset_type]

    #print(f"Data - N_train={len(loader_train.dataset.pose)} - N_val={len(loader_val.dataset.pose)}")
    #########Start my dataloader#############


    kwargs={"action_taxonomy_to_use": "fine", "max_samples":-1}#,"e4"]}
    train_dataset= get_dataset.get_dataset_motion(args.train_datasets,
                                list_splits=args.train_splits,          
                                list_view_ids=[-1 for i in range(len(args.train_splits))],
                                dataset_folder=args.dataset_folder,
                                use_same_action=False,
                                ntokens_per_clip=args.seq_len,
                                spacing=1,
                                nclips=1,
                                is_shifting_window=True,
                                min_window_sec=0.,#(args.ntokens_per_clip*args.spacing/30.)*2,
                                dict_is_aug={"aug_obsv_len":False},
                                **kwargs,)
        
    loader_train=get_dataset.DataLoaderX(train_dataset,
                                    batch_size=args.train_batch_size,
                                    shuffle=True,
                                    num_workers=args.prefetch_factor,
                                    pin_memory=False,#True,
                                    drop_last=True,
                                    collate_fn= collate_fn,)
                                    
    val_dataset = get_dataset.get_dataset_motion(args.val_datasets,
                            list_splits=args.val_splits,   
                            list_view_ids=[-1 for i in range(len(args.val_splits))],
                            dataset_folder=args.dataset_folder,
                            use_same_action=False,
                            ntokens_per_clip=args.seq_len,#args.ntokens_per_clip,
                            spacing=1.,#args.spacing,
                            nclips=1,
                            min_window_sec=0.,#(args.ntokens_per_clip*args.spacing/30.)*2,
                            is_shifting_window=True,
                            dict_is_aug={},
                            **kwargs,)

    loader_val = get_dataset.DataLoaderX(val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.prefetch_factor,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,)

    #########End my dataloader################
    # Model
    print(f"\nBuilding the model...")
    print(args)
    in_dim=153-12#153
    #in_dim = ((loader_train.dataset.pose[0].size(1) // 3) - 1) * 6 + 3  # jts in 6D repr, trans in 3d coord
    model = Model(in_dim=in_dim, **vars(args)).to(device)
    model.seq_len = args.vq_seq_len

    total_param = print_parameters_count(model, detailed=args.detailed_count)

    reload_epoch = True
    print(f"Number of parameters: {get_parameters(model):,}")
    checkpoint, ckpt_path = get_last_checkpoint(args.save_dir, args.name)
    if checkpoint is None and args.pretrained_ckpt is not None:
        ckpt_path, reload_epoch = args.pretrained_ckpt, True
        checkpoint = torch.load(args.pretrained_ckpt)

    if checkpoint is not None:
        weights = checkpoint['model_state_dict']
        missing, unexpected = model.load_state_dict(weights, strict=False)
        assert not (len(unexpected) or len(missing)), "Problem with loading"
        # Reload centroid counts
        if 'balance_stats' in checkpoint.keys():
            bins = checkpoint['balance_stats']
            model.quantizer.load_state(bins)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.ab1, args.ab2))
    loss_scaler = NativeScaler() if args.use_amp else None

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch, saved_iter = [checkpoint[k] for k in ['epoch', 'iter']] if reload_epoch else [1, 0]
        bv, bc = [checkpoint[k] if k in checkpoint
                else None for k in ['best_val', 'best_class']] if reload_epoch else [None, None]
        print(f"Ckpt succesfully loaded from: {ckpt_path}")
        if 'scaler' in checkpoint:
            assert loss_scaler is not None, "I have found weights for the loss_scaler, but don't have it."
            loss_scaler.load_state_dict(checkpoint['scaler'])
    else:
        epoch, saved_iter = 1, 0
        bv, bc = None, None

    # Trainer
    print(f"\nSetting up the trainer...")
    trainer = QTrainer(model=model, optimizer=optimizer, device=device,
                       args=args, epoch=epoch, start_iter=saved_iter,
                       best_val=bv, best_class=bc, #type=loader_train.dataset.type,
                       seq_len=args.seq_len, loss_scaler=loss_scaler)
    
    if args.eval_only:
        # Validate the model; will compute standard fid (without time conditioning), and classification accuracy
        val = trainer.eval(loader_val, loss_type=args.loss, epoch=epoch,
                           save_to_tboard=args.visu_to_tboard)
        print(val)
    else:
        trainer.writer.add_scalar('train/z_parameter_count', total_param, trainer.current_iter)
        trainer.fit(loader_train, loader_val, loss=args.loss)

if __name__ == "__main__":
    main()
