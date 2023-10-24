import os
import pickle
import cv2

from tqdm import tqdm
import torch

import torch.nn.functional as torch_f
import numpy as np

from libyana.evalutils.avgmeter import AverageMeters
from meshreg.models.utils import torch2numpy,solve_pnp_ransac_and_translate, solve_pnp_and_transform, compute_root_aligned_and_palmed_aligned
from meshreg.netscripts import position_evaluator as evaluate
from meshreg.netscripts.classification_evaluator import FrameClassificationEvaluator
from meshreg.netscripts.position_evaluator import MyMEPE, feed_mymepe_evaluators_hands, MyVAE, feed_myvae_evaluator_hands


def epoch_pass(
    loader,
    model,
    optimizer,
    scheduler,
    epoch,
    pose_dataset,
    split_tag,
    lr_decay_gamma=0,

    tensorboard_writer=None,
    back_to_ori_camera=False,
):
    train= (split_tag=='train')
    obj_evaluator = FrameClassificationEvaluator(model.object_name2idx)

    try:
        action_evaluator = FrameClassificationEvaluator(model.action_name2idx)
    except:
        action_evaluator=None    
    print("obj_evaluator",obj_evaluator.num_labels)
    if action_evaluator is not None:
        print("action_evaluator",action_evaluator.num_labels)
    evaluate_action=False

    avg_meters = AverageMeters()
    evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
    evaluators_pred_ra={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    
    evaluators_pred_local={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    valid_joints=pose_dataset.valid_joints#

    # Loop over dataset
    verbose=False
    model.eval() 
    for batch_idx, batch in enumerate(tqdm(loader)):
        if train:
            loss, results, losses = model(batch, is_train=True, verbose=verbose)  
        else:
            with torch.no_grad():
                loss, results, losses = model(batch, is_train=False, verbose=verbose)
       
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
        
        if not train:
            if not back_to_ori_camera:
                batch_seq_cam_joints3d_pred=torch.unsqueeze(results["pred_joints3d"],1)
                batch_seq_cam_joints3d_gt=torch.unsqueeze(results["gt_joints3d"],1)
            else:
                flatten_cam_joints3d_pred_ncam=torch2numpy(results["pred_joints3d"])
                flatten_cam_joints2d_pred=torch2numpy(results["pred_joints2d"])
                flatten_cam_joints3d_pred=np.zeros_like(flatten_cam_joints3d_pred_ncam)
                
                flatten_cam_joints3d_gt=torch2numpy(torch.cat([batch["cam_joints3d_left"],batch["cam_joints3d_right"]],dim=1))
                flatten_cam_intr=torch2numpy(batch["cam_intr"])
                
                for sample_idx in range(flatten_cam_intr.shape[0]):
                    flatten_cam_joints3d_pred[sample_idx]=solve_pnp_and_transform(flatten_cam_joints3d_pred_ncam[sample_idx],flatten_cam_joints2d_pred[sample_idx],flatten_cam_intr[sample_idx])["cam2_joints3d"]
                
                #Goto cuda, otherwise the computation of ra will be problematic.
                batch_seq_cam_joints3d_pred=torch.unsqueeze(torch.from_numpy(flatten_cam_joints3d_pred),1).cuda()
                batch_seq_cam_joints3d_gt=torch.unsqueeze(torch.from_numpy(flatten_cam_joints3d_gt),1).cuda()

            batch_seq_weights=results["valid_frames"].view(-1,1)
            feed_mymepe_evaluators_hands(evaluators_pred, batch_seq_cam_joints3d_pred, batch_seq_cam_joints3d_gt, batch_seq_weights, valid_joints=valid_joints)
            
            '''
            batch_seq_cam_joints3d_pred=batch_seq_cam_joints3d_pred.view(-1,16,42,3)
            batch_seq_ra_joints3d_pred=batch_seq_cam_joints3d_pred.clone()
            batch_seq_ra_joints3d_pred[:,:,:21]-=batch_seq_cam_joints3d_pred[:,0:1,0:1]
            batch_seq_ra_joints3d_pred[:,:,21:]-=batch_seq_cam_joints3d_pred[:,0:1,21:22]
            
            batch_seq_cam_joints3d_gt=batch_seq_cam_joints3d_gt.view(-1,16,42,3)
            batch_seq_ra_joints3d_gt=batch_seq_cam_joints3d_gt.clone()
            
            batch_seq_ra_joints3d_gt[:,:,:21]-=batch_seq_cam_joints3d_gt[:,0:1,0:1]
            batch_seq_ra_joints3d_gt[:,:,21:]-=batch_seq_cam_joints3d_gt[:,0:1,21:22]
            feed_mymepe_evaluators_hands(evaluators_pred_ra, 
                                        torch.unsqueeze(torch.flatten(batch_seq_ra_joints3d_pred,0,1)[:batch_seq_weights.shape[0]],1), 
                                        torch.unsqueeze(torch.flatten(batch_seq_ra_joints3d_gt,0,1)[:batch_seq_weights.shape[0]],1),
                                        batch_seq_weights, valid_joints=valid_joints)#[1:])
            '''

            batch_seq_ra_joints3d_pred=batch_seq_cam_joints3d_pred.clone()
            batch_seq_ra_joints3d_pred[:,:,:21]-=batch_seq_cam_joints3d_pred[:,:,0:1]
            batch_seq_ra_joints3d_pred[:,:,21:]-=batch_seq_cam_joints3d_pred[:,:,21:22]
            
            batch_seq_ra_joints3d_gt=batch_seq_cam_joints3d_gt.clone()
            batch_seq_ra_joints3d_gt[:,:,:21]-=batch_seq_cam_joints3d_gt[:,:,0:1]
            batch_seq_ra_joints3d_gt[:,:,21:]-=batch_seq_cam_joints3d_gt[:,:,21:22]
            
            assert len(valid_joints)==20
            res_flatten=compute_root_aligned_and_palmed_aligned(batch_seq_ra_joints3d_pred.view(-1,42,3), batch_seq_ra_joints3d_gt.view(-1,42,3), align_to_gt_size=True,valid_joints=valid_joints)
            batch_seq_ra_joints3d_pred=torch.cat([res_flatten["flatten_left_ra_out"],res_flatten["flatten_right_ra_out"]],dim=1).view(batch_seq_ra_joints3d_pred.shape)
            feed_mymepe_evaluators_hands(evaluators_pred_ra, batch_seq_ra_joints3d_pred, batch_seq_ra_joints3d_gt, batch_seq_weights, valid_joints=valid_joints[1:])
            
            batch_seq_pa_joints3d_pred=torch.cat([res_flatten["flatten_left_pa_out"],res_flatten["flatten_right_pa_out"]],dim=1).view(batch_seq_ra_joints3d_pred.shape)            
            feed_mymepe_evaluators_hands(evaluators_pred_local, batch_seq_pa_joints3d_pred, batch_seq_ra_joints3d_gt, batch_seq_weights, valid_joints=valid_joints)#[1:])

                
            obj_evaluator.feed(gt_labels=results["obj_idx_gt"],pred_labels=results["obj_idx_out"],weights=results["valid_obj"])
            if "batch_action_idx_obsv_out" in results:
                action_evaluator.feed(gt_labels=results["batch_action_idx_obsv_gt"],pred_labels=results["batch_action_idx_obsv_out"],weights=None)
                evaluate_action=True
                
    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
        
    ##eval
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
         
    evaluator_results={} 
    if not train: 
        obj_result=obj_evaluator.get_recall_rate()
        for k,v in obj_result.items():
            save_dict['objlabel_'+k]=v
        
        if evaluate_action:
            action_result=action_evaluator.get_recall_rate()
            for k,v in action_result.items():
                save_dict['action_'+k]=v

        evaluator_results = evaluate.parse_evaluators(evaluators_pred)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_epe_mean"]=eval_res["epe_mean"]

        evaluator_results = evaluate.parse_evaluators(evaluators_pred_ra)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_ra_epe_mean"]=eval_res["epe_mean"]
                
                
        evaluator_results = evaluate.parse_evaluators(evaluators_pred_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]


    print("Epoch",epoch)
    for k,v in save_dict.items():
        if "mean" in k:
            print(k, "{:.2f}".format(100*v))
        else:
            print(k,"{:.4f}".format(v))
            
    if not tensorboard_writer is None:
        for k,v in save_dict.items():
            if k in losses.keys() or 'epe_mean' in k  or k in ['learning_rate','total_loss','action_recall_rate_mean', 'objlabel_recall_rate_mean', 'num_activated_embeddings']:
                print(k,v)
                tensorboard_writer.add_scalar(split_tag+'/'+k, v, epoch)                
    return



def epoch_pass_poenc(loader,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        split_tag,
                        lr_decay_gamma=0,
                        tensorboard_writer=None,):
    train= (split_tag=='train')
    
    obj_evaluator = FrameClassificationEvaluator(model.object_name2idx)
    avg_meters = AverageMeters()

    print("obj_evaluator",obj_evaluator.num_labels)
    verbose=False
    # Loop over dataset
    model.eval() 
    for batch_idx, batch in enumerate(tqdm(loader)): 
        if train:
            loss, results, losses = model(batch, is_train=True, verbose=verbose)  
        else:
            with torch.no_grad():
                loss, results, losses = model(batch, is_train=False, verbose=verbose)
       
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
        
        if not train:            
            obj_evaluator.feed(gt_labels=results["obj_idx_gt"],pred_labels=results["obj_idx_out"],weights=None)
            
            
    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
     
    
    ##eval
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
         
    evaluator_results={} 
    if not train: 
        obj_result=obj_evaluator.get_recall_rate()
        for k,v in obj_result.items():
            save_dict['objlabel_'+k]=v
        
    print("Epoch",epoch)
    for k,v in save_dict.items():
        if "mean" in k:
            print(k, "{:.2f}".format(100*v))
        else:
            print(k,"{:.2f}".format(v))
    if not tensorboard_writer is None:
        for k,v in save_dict.items():
            if k in losses.keys() or 'epe_mean' in k  or k in ['learning_rate','total_loss','action_recall_rate_mean', 'objlabel_recall_rate_mean', 'num_activated_embeddings']:
                print(k,v)
                tensorboard_writer.add_scalar(split_tag+'/'+k, v, epoch)
    return
