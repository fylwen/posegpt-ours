import os
import pickle
import cv2

from tqdm import tqdm
import torch
import copy

import torch.nn.functional as torch_f
import numpy as np

from libyana.evalutils.avgmeter import AverageMeters
from libyana.evalutils.zimeval import EvalUtil


from meshreg.netscripts import position_evaluator as evaluate
from meshreg.netscripts.position_evaluator import MyMEPE, feed_mymepe_evaluators_hands, MyVAE, feed_myvae_evaluator_hands
from meshreg.netscripts.classification_evaluator import ActionVersusExpert
from meshreg.netscripts.utils import sample_vis_trj_dec,sample_vis_trj_resnet

from meshreg.netscripts.timer import Timer
from meshreg.models.utils import  torch2numpy
from meshreg.datasets import ass101utils

def epoch_pass(
    loader,
    model,
    optimizer,
    scheduler,
    epoch,
    pose_dataset,
    split_tag,
    ntokens_obsv,
    ntokens_pred,
    lr_decay_gamma=0,
    tensorboard_writer=None,
):
    train= (split_tag=='train')

    avg_meters = AverageMeters()
    evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
    evaluators_pred_local={"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

    verbose=False
    # Loop over dataset
    model.eval() 

    for batch_idx, batch in enumerate(tqdm(loader)): 
        if train:
            loss, results, losses = model(batch, is_train=True,verbose=verbose)  
        else:
            with torch.no_grad():
                loss, results, losses = model(batch, is_train=False,verbose=verbose)
                
        if train:
            if torch.isnan(loss):
                raise ValueError(f"Loss made of {losses} became nan!")
            optimizer.zero_grad()                
            loss.backward()            
            optimizer.step()
            
        for loss_name, loss_val in losses.items():
            if loss_val is not None:
                avg_meters.add_loss_value(loss_name, loss_val.mean().item())
        if not train:# or epoch==1 or epoch%epoch_display_freq==0:
            batch_seq_weights=batch["valid_frame"].view(-1, ntokens_obsv+ntokens_pred)
            feed_mymepe_evaluators_hands(evaluators_pred,results["batch_seq_joints3d_in_base_pred_out"], \
                results["batch_seq_joints3d_in_base_pred_gt"], batch_seq_weights[:,ntokens_obsv:], valid_joints=pose_dataset.valid_joints)             
            feed_mymepe_evaluators_hands(evaluators_pred_local,results["batch_seq_joints3d_in_local_pred_out"], \
                results["batch_seq_joints3d_in_local_pred_gt"], batch_seq_weights[:,ntokens_obsv:], valid_joints=pose_dataset.valid_joints[1:])

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
        evaluator_results = evaluate.parse_evaluators(evaluators_pred)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_epe_mean"]=eval_res["epe_mean"]

        evaluator_results = evaluate.parse_evaluators(evaluators_pred_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"pred_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]

    print("Epoch",epoch)
    if not tensorboard_writer is None:
        for k,v in save_dict.items():
            if k in losses.keys() or 'epe_mean' in k or k in ['learning_rate','total_loss','action_recall_rate_mean', 'objlabel_recall_rate_mean', 'num_activated_embeddings']:
                print(k,v)
                tensorboard_writer.add_scalar(split_tag+'/'+k, v, epoch)
    return#save_dict, avg_meters, evaluator_results

def epoch_pass_eval(
    loader,
    model,
    pose_dataset,
    ntokens_obsv,
    ntokens_pred,
    tag_out='',
    model_fid=None
):
    output_obsv=model.output_obsv
    if model.gt_ite0:
        ntokens_pred-=ntokens_obsv
    if output_obsv:
        ntokens_pred+=ntokens_obsv
    avg_meters = AverageMeters()
    evaluators_pred = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
    evaluators_pred_local = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}

    if output_obsv:
        evaluators_obsv = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()} 
        evaluators_obsv_ra = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
        evaluators_obsv_local = {"left_joints3d":  MyMEPE(),"right_joints3d": MyMEPE()}
        
    evaluators_vae = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}
    evaluators_vae_local = {"left_joints3d":  MyVAE(),"right_joints3d": MyVAE()}    
    
    save_dict_fid={"batch_action_name_obsv":[],"batch_enc_out_global_feature":[]}
    verbose=False
    model.eval()
    print("Lets start evaluation")
    
    if pose_dataset.name=="h2o":
        if pose_dataset.split=="val":
            to_vis_batches=[17,40,58,45,163,32]#[17,18,130,   40,58,171,64,  45,46,47,163,164,165,    93,94,111, 32,100]#h2o val
        else:
            to_vis_batches=[16,68,58,104,42]#[16,17,31,33,138,149,   51,52,68,169,61,62,78,79,   58,59,104,   9,132, 42,338] #h2o test[149,68,169,58,104]#
    else:
        to_vis_batches=[68,69]#[0,4,25,37,38,63,   1,2,13, 24,68,69,   6,92,232,12]#+[72, 205,186,31,19,207,288,16,235,315,10,182,218,242,41,56,91,21]#ass
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        batch_rs_seq_in_cam_pred_out=[]
        batch_rs_seq_in_local_pred_out=[]
        
        for rs_id in range(0,21):             
            with torch.no_grad():
                loss, results, losses = model(batch, to_reparameterize=rs_id>0,verbose=verbose)# to_reparameterize=rs_id>0
            
            if rs_id>0:
                batch_rs_seq_in_cam_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_cam_pred_out"],dim=1))
                batch_rs_seq_in_local_pred_out.append(torch.unsqueeze(results["batch_seq_joints3d_in_local_pred_out"],dim=1))
            else:
                batch_seq_in_cam_pred_gt=results["batch_seq_joints3d_in_cam_pred_gt"]
                batch_seq_in_local_pred_gt=results["batch_seq_joints3d_in_local_pred_gt"]
            
                for loss_name, loss_val in losses.items():
                    if loss_val is not None:
                        avg_meters.add_loss_value(loss_name, loss_val.mean().item())             

                batch_seq_joints3d_pred_gt = results["batch_seq_joints3d_in_cam_gt" if output_obsv else "batch_seq_joints3d_in_cam_pred_gt"]
                batch_seq_joints3d_pred_out = results["batch_seq_joints3d_in_cam_out" if output_obsv else "batch_seq_joints3d_in_cam_pred_out"]
                
                batch_seq_weights=results["batch_seq_valid_frame_out"][:,-batch_seq_joints3d_pred_out.shape[1]:]#None
                feed_mymepe_evaluators_hands(evaluators_pred,batch_seq_joints3d_pred_out, batch_seq_joints3d_pred_gt, batch_seq_weights,valid_joints=pose_dataset.valid_joints)#[:,ntokens_obsv:]

                batch_seq_joints3d_pred_gt = results["batch_seq_joints3d_in_local_gt" if output_obsv else "batch_seq_joints3d_in_local_pred_gt"]
                batch_seq_joints3d_pred_out = results["batch_seq_joints3d_in_local_out" if output_obsv else "batch_seq_joints3d_in_local_pred_out"]
                feed_mymepe_evaluators_hands(evaluators_pred_local,batch_seq_joints3d_pred_out, batch_seq_joints3d_pred_gt, batch_seq_weights, valid_joints=pose_dataset.valid_joints[1:])#[:,ntokens_obsv:],

                if output_obsv:
                    #reshape to [bs*len_obsv,1]
                    batch_seq_weights_obsv=results["batch_seq_valid_frame_out"][:,:ntokens_obsv].contiguous().view(-1,1)
                    batch_seq_joints3d_obsv_gt=torch.unsqueeze(torch.flatten(results["batch_seq_joints3d_in_cam_gt"][:,:ntokens_obsv],0,1),1)
                    batch_seq_joints3d_obsv_out=torch.unsqueeze(torch.flatten(results["batch_seq_joints3d_in_cam_out"][:,:ntokens_obsv],0,1),1)
                    feed_mymepe_evaluators_hands(evaluators_obsv,batch_seq_joints3d_obsv_out, batch_seq_joints3d_obsv_gt, batch_seq_weights_obsv,valid_joints=pose_dataset.valid_joints)

                    batch_seq_ra_joints3d_obsv_gt=batch_seq_joints3d_obsv_gt.clone()
                    batch_seq_ra_joints3d_obsv_gt[:,:,:21]-=batch_seq_joints3d_obsv_gt[:,:,0:1]
                    batch_seq_ra_joints3d_obsv_gt[:,:,21:]-=batch_seq_joints3d_obsv_gt[:,:,21:22]

                    batch_seq_ra_joints3d_obsv_out=batch_seq_joints3d_obsv_out.clone()
                    batch_seq_ra_joints3d_obsv_out[:,:,:21]-=batch_seq_joints3d_obsv_out[:,:,0:1]
                    batch_seq_ra_joints3d_obsv_out[:,:,21:]-=batch_seq_joints3d_obsv_out[:,:,21:22]
                    feed_mymepe_evaluators_hands(evaluators_obsv_ra,batch_seq_ra_joints3d_obsv_out, batch_seq_ra_joints3d_obsv_gt, batch_seq_weights_obsv,valid_joints=pose_dataset.valid_joints[1:])

                    batch_seq_joints3d_obsv_gt=torch.unsqueeze(torch.flatten(results["batch_seq_joints3d_in_local_gt"][:,:ntokens_obsv],0,1),1)
                    batch_seq_joints3d_obsv_out=torch.unsqueeze(torch.flatten(results["batch_seq_joints3d_in_local_out"][:,:ntokens_obsv],0,1),1)
                    feed_mymepe_evaluators_hands(evaluators_obsv_local,batch_seq_joints3d_obsv_out, batch_seq_joints3d_obsv_gt, batch_seq_weights_obsv,valid_joints=pose_dataset.valid_joints[1:])


            if model_fid is not None:
                num_frames_to_pad=results["batch_seq_joints3d_in_cam_gt"].shape[1]-results["batch_seq_joints3d_in_cam_pred_out"].shape[1]
                batch_seq_joints3d_in_cam_for_fid=torch.cat([results["batch_seq_joints3d_in_cam_gt"][:,:num_frames_to_pad],results["batch_seq_joints3d_in_cam_pred_out"]],dim=1)
                batch_seq_joints3d_in_local_for_fid=torch.cat([results["batch_seq_joints3d_in_local_gt"][:,:num_frames_to_pad],results["batch_seq_joints3d_in_local_pred_out"]],dim=1)

                batch_to_fid={"batch_seq_cam_joints3d_left":batch_seq_joints3d_in_cam_for_fid[:,:,:21],
                            "batch_seq_cam_joints3d_right":batch_seq_joints3d_in_cam_for_fid[:,:,21:],
                            "batch_seq_local_joints3d_left":batch_seq_joints3d_in_local_for_fid[:,:,:21],
                            "batch_seq_local_joints3d_right":batch_seq_joints3d_in_local_for_fid[:,:,21:],
                            "batch_seq_valid_frame":results["batch_seq_valid_frame_out"],
                            "batch_action_name_obsv":results["batch_action_name_obsv"]}
                #batch_to_fid.update(batch)                       
                with torch.no_grad():
                    _, results_fid, _ =model_fid(batch_to_fid, num_prefix_frames_to_remove=0,batch_is_gt=False,compute_loss=False,verbose=verbose)
                for k in ["batch_action_name_obsv"]:
                    save_dict_fid[k]+=results_fid[k]
                for k in ["batch_enc_out_global_feature"]:
                    save_dict_fid[k].append(results_fid[k])


            is_vis= ("image_vis" in batch.keys())
            if is_vis:
                #np.savez("{:04d}.npz".format(batch_idx),batch_seq_dec_mem=torch2numpy(results["batch_seq_dec_mem"]),
                #    batch_seq_joints3d_in_cam_gt=torch2numpy(results["batch_seq_joints3d_in_cam_gt"]),
                #    flatten_imgs=torch2numpy(batch["image_vis"]),batch_action_name_obsv=batch["batch_action_name_obsv"])
                #continue
                for sample_id in range(0,batch_seq_joints3d_pred_gt.shape[0],1):
                    #if not batch["batch_action_name_obsv"][sample_id].split(' ')[0] in ["pick","put","pass","position","rotate","screw","unscrew","open","close","apply"]:
                    #    continue
                    print(batch_idx,sample_id)
                    '''
                    cam_info={"intr":batch["ncam_intr"][0].cpu().numpy(),"extr":np.eye(4)}
                    #cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
                    #tag_out="6906"
                    
                    cam_info=ass101utils.get_view_extrinsic_intrisic('../ass101/annotations/calib.txt')["C10118_rgb"]
                    #cam_info["intr"][:2]*=480/1920.
                    sample_vis_trj_dec(batch_seq_gt_cam=results["batch_seq_joints3d_in_cam_gt"], 
                                batch_seq_est_cam=results["batch_seq_joints3d_in_cam_out"], 
                                batch_seq_gt_local=results["batch_seq_joints3d_in_cam_gt"],#results["batch_seq_joints3d_in_local_gt"],
                                batch_seq_est_local=results["batch_seq_joints3d_in_cam_out"], #results["batch_seq_joints3d_in_local_out"],
                                batch_gt_action_name=results["batch_action_name_obsv"], 
                                joint_links=pose_dataset.links,  
                                flatten_imgs=batch["image_vis"],
                                sample_id=sample_id,
                                cam_info=cam_info,
                                prefix_cache_img=f"./{tag_out}/imgs/", path_video=f"./{tag_out}"+'/{:04d}_{:02d}_{:02d}.avi'.format(batch_idx,sample_id,rs_id))
                    '''
                    cam_info={"intr":np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32),"extr":np.eye(4)}
                    batch_seq_joints3d_in_cam_resnet=torch.cat([batch["cam_joints3d_left_resnet"],batch["cam_joints3d_right_resnet"]],dim=1).view(results["batch_seq_joints3d_in_cam_gt"].shape)
                    sample_vis_trj_resnet(batch_seq_gt_cam=results["batch_seq_joints3d_in_cam_gt"], 
                                batch_seq_est_cam1=results["batch_seq_joints3d_in_cam_out"], 
                                batch_seq_est_cam2=batch_seq_joints3d_in_cam_resnet,
                                batch_seq_gt_local=results["batch_seq_joints3d_in_cam_gt"],#results["batch_seq_joints3d_in_local_gt"],
                                batch_seq_est_local1=results["batch_seq_joints3d_in_cam_out"], #results["batch_seq_joints3d_in_local_out"],
                                batch_seq_est_local2=batch_seq_joints3d_in_cam_resnet,
                                joint_links=pose_dataset.links,  
                                flatten_imgs=batch["image_vis"],
                                sample_id=sample_id,
                                cam_info=cam_info,
                                prefix_cache_img=f"./{tag_out}/imgs/", path_video=f"./{tag_out}"+'/{:04d}_{:02d}_{:02d}.avi'.format(batch_idx,sample_id,rs_id))
                    

        if torch.mean(batch_seq_weights.float())>1.0-1e-5 and len(batch_rs_seq_in_cam_pred_out)>0:
            batch_rs_seq_in_cam_pred_out=torch.cat(batch_rs_seq_in_cam_pred_out,dim=1)
            batch_rs_seq_in_local_pred_out=torch.cat(batch_rs_seq_in_local_pred_out,dim=1)
            feed_myvae_evaluator_hands(evaluator=evaluators_vae,batch_rs_seq_joints3d_out=batch_rs_seq_in_cam_pred_out, 
                            batch_seq_joints3d_gt=batch_seq_in_cam_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=pose_dataset.valid_joints)
            feed_myvae_evaluator_hands(evaluator=evaluators_vae_local,batch_rs_seq_joints3d_out=batch_rs_seq_in_local_pred_out, 
                            batch_seq_joints3d_gt=batch_seq_in_local_pred_gt,batch_seq_weights=batch_seq_weights,valid_joints=pose_dataset.valid_joints[1:])
    save_dict = {}
    for loss_name, avg_meter in avg_meters.average_meters.items():
        loss_val = avg_meter.avg
        save_dict[loss_name] = loss_val
        

    if output_obsv:
        evaluator_results= evaluate.parse_evaluators(evaluators_obsv)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_epe_mean"]=eval_res["epe_mean"]

        evaluator_results= evaluate.parse_evaluators(evaluators_obsv_ra)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_ra_epe_mean"]=eval_res["epe_mean"]
        evaluator_results= evaluate.parse_evaluators(evaluators_obsv_local)
        for eval_name, eval_res in evaluator_results.items():
            save_dict[f"obsv_{eval_name}_local_epe_mean"]=eval_res["epe_mean"]
                  
    for k,v in save_dict.items():
        if "mean" in k:
            print(k, "{:.2f}".format(100*v))
        else:
            print(k,"{:.4f}".format(v))
            
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


    for pid in range(0,ntokens_pred):
        print('pred #{:d} L/R {:.2f} {:.2f}'.format(pid, 100*save_dict[f"pred{pid}_left_joints3d_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_epe_mean"]),\
                'Frame-RA L/R {:.2f} {:.2f}'.format(100*save_dict[f"pred{pid}_left_joints3d_local_epe_mean"],100*save_dict[f"pred{pid}_right_joints3d_local_epe_mean"]))
    #print('pred ave. L/R {:.2f} {:.2f}'.format(100*save_dict["pred_left_joints3d_epe_mean"],100*save_dict["pred_right_joints3d_epe_mean"]),
    #         'Frame-RA L/R {:.2f} {:.2f}'.format(100*save_dict["pred_left_joints3d_local_epe_mean"],100*save_dict["pred_right_joints3d_local_epe_mean"]))

    save_dict["ntokens_obsv"]=ntokens_obsv
    save_dict["ntokens_pred"]=ntokens_pred

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

    evaluate.aggregate_and_save(f"./{tag_out}.npz",save_dict)
    if model_fid is not None:        
        for k in ["batch_enc_out_global_feature"]:
            save_dict_fid[k]=torch.cat(save_dict_fid[k],dim=0).detach().cpu().numpy()
        with open(f"fid_{tag_out}.pkl", 'wb') as f:
            pickle.dump(save_dict_fid, f)
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
    
    return 0

