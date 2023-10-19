import os
import pickle
import cv2

from tqdm import tqdm
import torch

import torch.nn.functional as torch_f
import numpy as np

from libyana.evalutils.avgmeter import AverageMeters
from meshreg.models.utils import torch2numpy
from meshreg.netscripts import position_evaluator as evaluate
from meshreg.netscripts.classification_evaluator import FrameClassificationEvaluator
from meshreg.netscripts.position_evaluator import MyMEPE, feed_mymepe_evaluators_hands, MyVAE, feed_myvae_evaluator_hands
from meshreg.netscripts.utils import sample_vis_trj_dec
from meshreg.netscripts.timer import Timer


def epoch_pass(
    loader,
    model,
    optimizer,
    scheduler,
    epoch,
    pose_dataset,
    split_tag,
    ntokens_per_clip,
    lr_decay_gamma=0,
    tensorboard_writer=None,
):
    train= (split_tag=='train')
    
    action_obsv_evaluator = FrameClassificationEvaluator(model.action_name2idx)
    try:
        object_obsv_evaluator=FrameClassificationEvaluator(model.object_name2idx)
    except:
        object_obsv_evaluator=None
    with_object_classification=False
    with_obsv_pose_estimation=False

    avg_meters = AverageMeters()
    evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
    evaluators_pred_local={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

    evaluators_obsv = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    evaluators_obsv_local={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    evaluators_obsv_ra={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

    verbose=False
    # Loop over dataset
    model.eval() 
    for batch_idx, batch in enumerate(tqdm(loader)):
        # Forward
        if train:
            loss, results, losses = model(batch, is_train=True, to_reparameterize=False, gt_action_for_dec=False, verbose=verbose)
        else:
            with torch.no_grad():
                loss, results, losses = model(batch, is_train=False, to_reparameterize=False, gt_action_for_dec=False, verbose=verbose)
       
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
            batch_seq_weights=results["batch_seq_valid_frames_pred_token0"]
            feed_mymepe_evaluators_hands(evaluators_pred,results["batch_seq_joints3d_in_base_pred_out_token0"], \
                results["batch_seq_joints3d_in_base_pred_gt_token0"], batch_seq_weights, valid_joints=pose_dataset.valid_joints)
            feed_mymepe_evaluators_hands(evaluators_pred_local,results["batch_seq_joints3d_in_local_pred_out_token0"], \
                results["batch_seq_joints3d_in_local_pred_gt_token0"], batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])
            
            action_obsv_evaluator.feed(gt_labels=results["batch_action_idx_obsv_gt"],pred_labels=results["batch_action_idx_obsv_out"],weights=None)

            if "obj_idx_gt" in results and object_obsv_evaluator is not None:
                object_obsv_evaluator.feed(gt_labels=torch.flatten(results["obj_idx_gt"]),
                                    pred_labels=torch.flatten(results["obj_idx_out"]),weights=None)
                with_object_classification=True
                
            if "est_batch_seq_joints3d_in_cam_out_obsv" in results:
                with_obsv_pose_estimation=True
                batch_seq_weights=results["est_batch_seq_valid_frames_obsv"]
                batch_seq_cam_joints3d_gt, batch_seq_cam_joints3d_out=results["est_batch_seq_joints3d_in_cam_gt_obsv"],results["est_batch_seq_joints3d_in_cam_out_obsv"]
                feed_mymepe_evaluators_hands(evaluators_obsv, batch_seq_cam_joints3d_out, batch_seq_cam_joints3d_gt, batch_seq_weights, valid_joints=pose_dataset.valid_joints)
                
                batch_seq_ra_joints3d_gt=batch_seq_cam_joints3d_gt.clone()
                batch_seq_ra_joints3d_gt[:,:,:21]-=batch_seq_cam_joints3d_gt[:,:,0:1]
                batch_seq_ra_joints3d_gt[:,:,21:]-=batch_seq_cam_joints3d_gt[:,:,21:22]
                batch_seq_ra_joints3d_out=batch_seq_cam_joints3d_out.clone()
                batch_seq_ra_joints3d_out[:,:,:21]-=batch_seq_cam_joints3d_out[:,:,0:1]
                batch_seq_ra_joints3d_out[:,:,21:]-=batch_seq_cam_joints3d_out[:,:,21:22]
                feed_mymepe_evaluators_hands(evaluators_obsv_ra, batch_seq_ra_joints3d_out, batch_seq_ra_joints3d_gt, batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])

                feed_mymepe_evaluators_hands(evaluators_obsv_local,results["est_batch_seq_joints3d_in_local_out_obsv"], 
                                        results["est_batch_seq_joints3d_in_local_gt_obsv"],  batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])

    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        assert False
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
     
    
    ##eval
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
         
    evaluator_results={} 
    if not train:# or epoch==1 or epoch%epoch_display_freq==0:
        action_result=action_obsv_evaluator.get_recall_rate()
        for k,v in action_result.items():
            save_dict['action_'+k]=v
            
        evaluator_results = evaluate.parse_evaluators(evaluators_pred)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_epe_mean"]=eval_res["epe_mean"]
        evaluator_results = evaluate.parse_evaluators(evaluators_pred_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
        
        if with_object_classification:
            object_result=object_obsv_evaluator.get_recall_rate()
            for k,v in object_result.items():
                save_dict['objlabel_'+k]=v

        if with_obsv_pose_estimation:
            evaluator_results = evaluate.parse_evaluators(evaluators_obsv)
            for eval_name, eval_res in evaluator_results.items():
                save_dict[f"obsv_{eval_name}_epe_mean"]=eval_res["epe_mean"]                
            evaluator_results = evaluate.parse_evaluators(evaluators_obsv_local)
            for eval_name, eval_res in evaluator_results.items():
                save_dict[f"obsv_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
            evaluator_results = evaluate.parse_evaluators(evaluators_obsv_ra)
            for eval_name, eval_res in evaluator_results.items():
                save_dict[f"obsv_{eval_name}_ra_epe_mean"]=eval_res["epe_mean"]
    
    print("Epoch",epoch)
    if not tensorboard_writer is None:
        for k,v in save_dict.items():
            if k in losses.keys() or 'epe_mean' in k  or k in ['learning_rate','total_loss','action_recall_rate_mean', 'objlabel_recall_rate_mean', 'num_activated_embeddings']:
                print(k,v)
                tensorboard_writer.add_scalar(split_tag+'/'+k, v, epoch)
    return



def epoch_pass_eval(
    loader,
    model,
    pose_dataset,
    ntokens_pred,
    gt_action_for_dec,
    tag_out='',
    model_fid=None,
):
    action_obsv_evaluator = FrameClassificationEvaluator(model.action_name2idx)
    avg_meters = AverageMeters()
    
    evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
    evaluators_pred_local = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

    evaluators_vae = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}
    evaluators_vae_local = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}
    
    evaluators_obsv = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    evaluators_obsv_local={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    evaluators_obsv_ra={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    with_obsv_pose_estimation=False
    
    save_dict_fid={"batch_action_name_obsv":[],"batch_enc_out_global_feature":[]}

    verbose=False
    # Loop over dataset
    model.eval() 
    vis_batch_idx=[26,41,89]#[9,17,12,54,57,24,49,48,0,1]
    for batch_idx, batch in enumerate(tqdm(loader)):
        batch_rs_seq_in_cam_pred_out=[]
        batch_rs_seq_in_local_pred_out=[]

        for rs_id in range(0,1):
            with torch.no_grad():
                loss, results, losses = model(batch, is_train=False, to_reparameterize=rs_id>0, gt_action_for_dec=gt_action_for_dec, verbose=verbose)

            if rs_id>0:                
                batch_rs_seq_in_cam_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_cam_pred_out"][:,:ntokens_pred],dim=1))
                batch_rs_seq_in_local_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_local_pred_out"][:,:ntokens_pred],dim=1))
            else:
                batch_seq_in_cam_pred_gt=results["batch_seq_joints3d_in_cam_pred_gt"][:,:ntokens_pred]
                batch_seq_in_local_pred_gt=results["batch_seq_joints3d_in_local_pred_gt"][:,:ntokens_pred]
                batch_seq_weights=results["batch_seq_valid_frames_pred"][:,:ntokens_pred]

                for loss_name, loss_val in losses.items():
                    if loss_val is not None:
                        avg_meters.add_loss_value(loss_name, loss_val.mean().item())
                            
                feed_mymepe_evaluators_hands(evaluators_pred,results["batch_seq_joints3d_in_cam_pred_out"][:,:ntokens_pred], \
                    results["batch_seq_joints3d_in_cam_pred_gt"][:,:ntokens_pred], batch_seq_weights, valid_joints=pose_dataset.valid_joints)  
                feed_mymepe_evaluators_hands(evaluators_pred_local,results["batch_seq_joints3d_in_local_pred_out"][:,:ntokens_pred], \
                    results["batch_seq_joints3d_in_local_pred_gt"][:,:ntokens_pred], batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])
                
                action_obsv_evaluator.feed(gt_labels=results["batch_action_idx_obsv_gt"],pred_labels=results["batch_action_idx_obsv_out"],weights=None)

                if "est_batch_seq_joints3d_in_cam_out_obsv" in results:
                    with_obsv_pose_estimation=True
                    batch_seq_weights=results["est_batch_seq_valid_frames_obsv"]
                    batch_seq_cam_joints3d_gt, batch_seq_cam_joints3d_out=results["est_batch_seq_joints3d_in_cam_gt_obsv"],results["est_batch_seq_joints3d_in_cam_out_obsv"]
                    feed_mymepe_evaluators_hands(evaluators_obsv, batch_seq_cam_joints3d_out, batch_seq_cam_joints3d_gt, batch_seq_weights, valid_joints=pose_dataset.valid_joints)
                    
                    batch_seq_ra_joints3d_gt=batch_seq_cam_joints3d_gt.clone()
                    batch_seq_ra_joints3d_gt[:,:,:21]-=batch_seq_cam_joints3d_gt[:,:,0:1]
                    batch_seq_ra_joints3d_gt[:,:,21:]-=batch_seq_cam_joints3d_gt[:,:,21:22]

                    batch_seq_ra_joints3d_out=batch_seq_cam_joints3d_out.clone()
                    batch_seq_ra_joints3d_out[:,:,:21]-=batch_seq_cam_joints3d_out[:,:,0:1]
                    batch_seq_ra_joints3d_out[:,:,21:]-=batch_seq_cam_joints3d_out[:,:,21:22]
                    feed_mymepe_evaluators_hands(evaluators_obsv_ra, batch_seq_ra_joints3d_out, batch_seq_ra_joints3d_gt, batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])

                    feed_mymepe_evaluators_hands(evaluators_obsv_local,results["est_batch_seq_joints3d_in_local_out_obsv"], 
                                            results["est_batch_seq_joints3d_in_local_gt_obsv"],  batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])
                                            
            if model_fid is not None:
                num_obsv_frames=torch.sum(results["batch_seq_valid_frames_obsv"][0])
                num_frames_to_pad=num_obsv_frames+model.model_pblock.ntokens_pred
                batch_seq_valid_frame=torch.cat([results["batch_seq_valid_frames_obsv"][:,:num_obsv_frames],
                                                results["batch_seq_valid_frames_cdpred"][:,:model.model_pblock.ntokens_pred+ntokens_pred]],dim=1)
                if verbose:
                    print("num_obsv_frames",num_obsv_frames,"num_frames_to_pad",num_frames_to_pad)
                    print("batch_seq_valid_frame",batch_seq_valid_frame.shape)
                    print(results["batch_seq_valid_frames_obsv"].shape,torch.sum(results["batch_seq_valid_frames_obsv"],dim=1))
                    print(results["batch_seq_valid_frames_cdpred"].shape,torch.sum(results["batch_seq_valid_frames_cdpred"],dim=1))                
                    
                    assert (torch.sum(results["batch_seq_valid_frames_obsv"],dim=1)==num_obsv_frames).all()
                    for i in range(batch_seq_valid_frame.shape[0]):
                        cnum_valid=torch.sum(batch_seq_valid_frame[i])
                        if cnum_valid<batch_seq_valid_frame.shape[1]:
                            assert batch_seq_valid_frame[i,cnum_valid-1]>0. and batch_seq_valid_frame[i,cnum_valid]<1e-6

                #use valid obsv only
                batch_seq_joints3d_in_cam_gt=torch.cat([results["batch_seq_joints3d_in_cam_obsv_gt"][:,:num_obsv_frames],results["batch_seq_joints3d_in_cam_cdpred_gt"]],dim=1)
                batch_seq_joints3d_in_cam_out=results["batch_seq_joints3d_in_cam_pred_out"][:,:ntokens_pred]

                batch_seq_joints3d_in_local_gt=torch.cat([results["batch_seq_joints3d_in_local_obsv_gt"][:,:num_obsv_frames],results["batch_seq_joints3d_in_local_cdpred_gt"]],dim=1)
                batch_seq_joints3d_in_local_out=results["batch_seq_joints3d_in_local_pred_out"][:,:ntokens_pred]

                batch_seq_joints3d_in_cam_for_fid=torch.cat([batch_seq_joints3d_in_cam_gt[:,:num_frames_to_pad],batch_seq_joints3d_in_cam_out],dim=1)
                batch_seq_joints3d_in_local_for_fid=torch.cat([batch_seq_joints3d_in_local_gt[:,:num_frames_to_pad],batch_seq_joints3d_in_local_out],dim=1)
                
                if verbose:
                    print(results["batch_seq_joints3d_in_cam_obsv_gt"].shape,results["batch_seq_joints3d_in_cam_cdpred_gt"].shape)
                    print(results["batch_seq_joints3d_in_local_obsv_gt"].shape,results["batch_seq_joints3d_in_local_cdpred_gt"].shape)
                    print("cam",batch_seq_joints3d_in_cam_gt.shape,batch_seq_joints3d_in_cam_out.shape)
                    print("local",batch_seq_joints3d_in_local_gt.shape,batch_seq_joints3d_in_local_out.shape)

                    print("for fid",batch_seq_joints3d_in_cam_for_fid.shape,batch_seq_joints3d_in_local_for_fid.shape)
                
                batch_to_fid={"batch_seq_cam_joints3d_left":batch_seq_joints3d_in_cam_for_fid[:,:,:21],
                            "batch_seq_cam_joints3d_right":batch_seq_joints3d_in_cam_for_fid[:,:,21:],
                            "batch_seq_local_joints3d_left":batch_seq_joints3d_in_local_for_fid[:,:,:21],
                            "batch_seq_local_joints3d_right":batch_seq_joints3d_in_local_for_fid[:,:,21:],
                            "batch_seq_valid_frame":batch_seq_valid_frame,
                            "batch_action_name_obsv":results["batch_action_name_obsv"]}
                #batch_to_fid.update(batch)
                
                with torch.no_grad():
                    _, results_fid, _ =model_fid(batch_to_fid, num_prefix_frames_to_remove=0,batch_is_gt=False,compute_loss=False,verbose=verbose)
                for k in ["batch_action_name_obsv"]:
                    save_dict_fid[k]+=results_fid[k]
                for k in ["batch_enc_out_global_feature"]:
                    save_dict_fid[k].append(results_fid[k])

            is_vis=True and "batch_seq_image_vis_cdpred" in results
            if is_vis:                
                batch_seq_joints3d_in_cam_gt=torch.cat([results["batch_seq_joints3d_in_cam_obsv_gt"],results["batch_seq_joints3d_in_cam_cdpred_gt"]],dim=1)
                batch_seq_joints3d_in_cam_out=results["batch_seq_joints3d_in_cam_pred_out"]

                batch_seq_joints3d_in_local_gt=torch.cat([results["batch_seq_joints3d_in_local_obsv_gt"],results["batch_seq_joints3d_in_local_cdpred_gt"]],dim=1)
                batch_seq_joints3d_in_local_out=results["batch_seq_joints3d_in_local_pred_out"]
                
                batch_est_action_name=[]
                for eid in torch2numpy(results["batch_action_idx_obsv_out"]):
                    batch_est_action_name.append(list(model.action_name2idx.keys())[eid])
                print(batch_est_action_name)
                print(results["batch_action_name_obsv"])
                
                flatten_imgs=torch.cat([results["batch_seq_image_vis_obsv"],results["batch_seq_image_vis_cdpred"]],dim=1)
                flatten_imgs=torch.flatten(flatten_imgs,0,1)

                print(batch_seq_joints3d_in_cam_gt.shape,batch_seq_joints3d_in_cam_out.shape)
                print(batch_seq_joints3d_in_local_gt.shape,batch_seq_joints3d_in_local_out.shape)
                print("flatten_imgs",flatten_imgs.shape)

                links=[(0, 2, 3, 4),(0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]
                cam_info={"intr":batch['obsv_clip_frame_cam_intr'][0,0,0].detach().numpy(),"extr":np.eye(4)}
                
                #cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
                for sample_id in range(0,batch_seq_joints3d_in_cam_gt.shape[0],1):
                    print(batch_idx,sample_id)
                    sample_vis_trj_dec(batch_seq_gt_cam=batch_seq_joints3d_in_cam_gt, 
                                batch_seq_est_cam=batch_seq_joints3d_in_cam_out, 
                                batch_seq_gt_local=batch_seq_joints3d_in_local_gt,
                                batch_seq_est_local=batch_seq_joints3d_in_local_out,
                                batch_gt_action_name=results["batch_action_name_obsv"],
                                batch_est_action_name=batch_est_action_name, 
                                joint_links=links,  
                                flatten_imgs=flatten_imgs,
                                sample_id=sample_id,
                                cam_info=cam_info,
                                batch_seq_valid_frames_obsv=results["batch_seq_valid_frames_obsv"],
                                batch_seq_valid_frames_pred=results["batch_seq_valid_frames_cdpred"],
                                prefix_cache_img=f"./{tag_out}/imgs/", path_video=f"./{tag_out}/"+'{:04d}_{:02d}_{:02d}.avi'.format(batch_idx,sample_id, rs_id))
        
        if torch.mean(batch_seq_weights.float())>1.0-1e-5 and len(batch_rs_seq_in_cam_pred_out)>0:
            batch_rs_seq_in_cam_pred_out=torch.cat(batch_rs_seq_in_cam_pred_out,dim=1)
            batch_rs_seq_in_local_pred_out=torch.cat(batch_rs_seq_in_local_pred_out,dim=1)
            feed_myvae_evaluator_hands(evaluator=evaluators_vae,batch_rs_seq_joints3d_out=batch_rs_seq_in_cam_pred_out, 
                            batch_seq_joints3d_gt=batch_seq_in_cam_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=pose_dataset.valid_joints)
            feed_myvae_evaluator_hands(evaluator=evaluators_vae_local,batch_rs_seq_joints3d_out=batch_rs_seq_in_local_pred_out, 
                            batch_seq_joints3d_gt=batch_seq_in_local_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=pose_dataset.valid_joints[1:])
    ##eval
    save_dict={}
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
         
    evaluator_results={} 
    action_result=action_obsv_evaluator.get_recall_rate(path_to_save=None)#"{:s}_action.npz".format(tag_out))
    for k,v in action_result.items():
        save_dict['action_'+k]=v

    if with_obsv_pose_estimation:
        evaluator_results = evaluate.parse_evaluators(evaluators_obsv)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_epe_mean"]=eval_res["epe_mean"]                
        evaluator_results = evaluate.parse_evaluators(evaluators_obsv_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
        evaluator_results = evaluate.parse_evaluators(evaluators_obsv_ra)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_ra_epe_mean"]=eval_res["epe_mean"]
    print(save_dict)
    
    evaluator_results= evaluate.parse_evaluators(evaluators_pred)
    for eval_name, eval_res in evaluator_results.items():
        save_dict[f"pred_{eval_name}_epe_mean"]=eval_res["epe_mean"]
        for pid in range(ntokens_pred):
            save_dict[f"pred{pid}_{eval_name}_epe_mean"]=eval_res["per_frame_epe_mean"][pid]
            
    evaluator_results= evaluate.parse_evaluators(evaluators_pred_local)
    for eval_name, eval_res in evaluator_results.items():
        save_dict[f"pred_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
        for pid in range(ntokens_pred):
            save_dict[f"pred{pid}_{eval_name}_local_epe_mean"]=eval_res["per_frame_epe_mean"][pid]
            
    #for pid in range(0,ntokens_pred):
    #    print('pred #{:d} L/R {:.2f} {:.2f}'.format(pid, 100*save_dict[f"pred{pid}_left_joints3d_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_epe_mean"]),\
    #            'Frame-RA L/R {:.2f} {:.2f}'.format(100*save_dict[f"pred{pid}_left_joints3d_local_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_local_epe_mean"]))

    save_dict["ntokens_pred"]=ntokens_pred
    evaluate.aggregate_and_save(f"./{tag_out}.npz",save_dict)
    
    if model_fid is not None:        
        for k in ["batch_enc_out_global_feature"]:
            save_dict_fid[k]=torch.cat(save_dict_fid[k],dim=0).detach().cpu().numpy()
        with open(f"fid_{tag_out}.pkl", 'wb') as f:
            pickle.dump(save_dict_fid, f)
    
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


    ######################    
    print("Mean w/o sampling\n {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(100*save_dict["pred_left_joints3d_epe_mean"],100*save_dict["pred_right_joints3d_epe_mean"],
                            100*save_dict["pred_left_joints3d_local_epe_mean"],100*save_dict["pred_right_joints3d_local_epe_mean"]))
    pid=ntokens_pred-1
    print("pred #{:d}\n {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(pid, 100*save_dict[f"pred{pid}_left_joints3d_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_epe_mean"],
                                100*save_dict[f"pred{pid}_left_joints3d_local_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_local_epe_mean"]))

    for kk in ["ape","fpe","apd"]:
        print(kk)
        print("{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(100*vae_results["left_joints3d"][kk],100*vae_results["right_joints3d"][kk],
                        100*vae_results_local["left_joints3d"][kk],100*vae_results_local["right_joints3d"][kk]))
    return
    
    
def epoch_pass_fid(
    loader,
    model,
    optimizer,
    scheduler,
    epoch,
    split_tag,
    lr_decay_gamma=0,
    tensorboard_writer=None,
):
    train= (split_tag=='train')
    
    avg_meters = AverageMeters()
    action_obsv_evaluator = FrameClassificationEvaluator(model.action_name2idx)
    
    verbose=False
    # Loop over dataset
    model.eval() 
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        # Forward
        if train:
            loss, results, losses = model(batch, num_prefix_frames_to_remove=0, batch_is_gt=True, compute_loss=True, verbose=verbose)  
        else:
            with torch.no_grad():
                loss, results, losses = model(batch, num_prefix_frames_to_remove=0, batch_is_gt=True, compute_loss=True, verbose=verbose)
       
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
            action_obsv_evaluator.feed(gt_labels=results["batch_action_idx_obsv_gt"],pred_labels=results["batch_action_idx_obsv_out"],weights=None)

    save_dict = {}
    if train and lr_decay_gamma and scheduler is not None:
        save_dict['learning_rate']=scheduler.get_last_lr()[0]
        scheduler.step()
        
    ##eval
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
        
    evaluator_results={} 
    if not train:# or epoch==1 or epoch%epoch_display_freq==0:
        action_result=action_obsv_evaluator.get_recall_rate()
        for k,v in action_result.items():
            save_dict['action_'+k]=v

    if not tensorboard_writer is None:
        for k,v in save_dict.items():
            if k in losses.keys() or 'epe_mean' in k  or k in ['learning_rate','total_loss','action_recall_rate_mean', 'objlabel_recall_rate_mean', 'num_activated_embeddings']:
                print(k,v)
                tensorboard_writer.add_scalar(split_tag+'/'+k, v, epoch)
    else:
        print(save_dict)
    return




def epoch_pass_fid_eval_for_gt(
    loader,
    model,
    num_prefix_frames_to_remove,
    tag_to_save):
    
    verbose=False
    model.eval() 
    dict_to_save={"batch_action_name_obsv":[],"batch_enc_out_global_feature":[]}#"imgpath_start_frame":[], "imgpath_end_frame":[],
    
    for batch_idx, batch in enumerate(tqdm(loader)):        
        with torch.no_grad():
            loss, results, losses = model(batch, num_prefix_frames_to_remove=num_prefix_frames_to_remove, batch_is_gt=True, compute_loss=False, verbose=verbose)
            for k in ["batch_enc_out_global_feature"]:
                dict_to_save[k].append(results[k])
            assert not "NIL" in results["batch_action_name_obsv"]

    #for k in ["imgpath_start_frame", "imgpath_end_frame","batch_action_name_obsv"]:
    #    print(k,len(dict_to_save[k]))
    for k in ["batch_enc_out_global_feature"]:
        dict_to_save[k]=torch.cat(dict_to_save[k],dim=0).detach().cpu().numpy()
        print(k,dict_to_save[k].shape)
    
    print("Saving to",f"fidgt_{tag_to_save}.pkl")
    with open(f"fidgt_{tag_to_save}.pkl", 'wb') as f:
        pickle.dump(dict_to_save, f)
    return
