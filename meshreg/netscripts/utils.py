import torch
import os
from libyana.visutils.viz2d import visualize_joints_2d

from meshreg.models.utils import torch2numpy,project_hand_3d2img
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.axes as Axes

from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA



def subplot_frame_gt_est_onehand(ctrj_gt,ctrj_est,gt_frame_id,est_frame_id,plot_id,title,is_left,joint_links, ctrj_est2=None, elev=40,azim=110,num_rows=4,num_cols=8,boundary=0.0):
    #azim+=180
    ax3d=plt.subplot(num_rows, num_cols, plot_id, projection='3d')   
    
    ctrj_gt=ctrj_gt.copy()
    ctrj_est=ctrj_est.copy()
    if ctrj_est2 is not None:
        ctrj_est2=ctrj_est2.copy()



    ax3d.view_init(elev,azim)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_title(title,fontsize=6,pad=0.)

    for l in joint_links:
        if not is_left:
            l=[x+21 for x in l]
        ax3d.plot(ctrj_gt[gt_frame_id,l,0],ctrj_gt[gt_frame_id,l,2],ctrj_gt[gt_frame_id,l,1],alpha=0.8,c=(0,1.,0),linewidth=0.75)#0.5)
    
    if est_frame_id>=0:
        for l in joint_links:
            if not is_left:
                l=[x+21 for x in l]
            ax3d.plot(ctrj_est[est_frame_id,l,0],ctrj_est[est_frame_id,l,2],ctrj_est[est_frame_id,l,1], alpha=1.0,c=(0,0,1.),linewidth=0.75)#0.5)
            ax3d.plot(ctrj_est[est_frame_id,l,0],ctrj_est[est_frame_id,l,2],ctrj_est[est_frame_id,l,1], alpha=1.0,c=(0,0,1.),linewidth=0.75)#0.5)

            if ctrj_est2 is not None:
                ax3d.plot(ctrj_est2[est_frame_id,l,0],ctrj_est2[est_frame_id,l,2],ctrj_est2[est_frame_id,l,1], alpha=1.0,c=(1.,1.,0.),linewidth=0.75)#0.5)
                ax3d.plot(ctrj_est2[est_frame_id,l,0],ctrj_est2[est_frame_id,l,2],ctrj_est2[est_frame_id,l,1], alpha=1.0,c=(1.,1.,0.),linewidth=0.75)#0.5)


    if is_left:
        ax3d.set_xlim(ctrj_gt[:,:21,0].min()-boundary,ctrj_gt[:,:21,0].max()+boundary)   
        ax3d.set_ylim(ctrj_gt[:,:21,2].min()-boundary,ctrj_gt[:,:21,2].max()+boundary)   
        ax3d.set_zlim(ctrj_gt[:,:21,1].min()-boundary,ctrj_gt[:,:21,1].max()+boundary)  
    else:
        ax3d.set_xlim(ctrj_gt[:,21:,0].min()-boundary,ctrj_gt[:,21:,0].max()+boundary)   
        ax3d.set_ylim(ctrj_gt[:,21:,2].min()-boundary,ctrj_gt[:,21:,2].max()+boundary)   
        ax3d.set_zlim(ctrj_gt[:,21:,1].min()-boundary,ctrj_gt[:,21:,1].max()+boundary)  

def sample_vis_trj_dec(batch_seq_gt_cam, batch_seq_est_cam, batch_gt_action_name, joint_links, prefix_cache_img, path_video, sample_id, cam_info,
                    batch_seq_gt_local=None, batch_seq_est_local=None, flatten_imgs=None, batch_est_action_name=None,
                    batch_seq_valid_frames_obsv=None,batch_seq_valid_frames_pred=None):
    batch_seq_gt_cam=torch2numpy(batch_seq_gt_cam)
    batch_seq_est_cam=torch2numpy(batch_seq_est_cam)
    if batch_seq_gt_local is not None:
        batch_seq_gt_local=torch2numpy(batch_seq_gt_local)
        batch_seq_est_local=torch2numpy(batch_seq_est_local)
    if batch_seq_valid_frames_obsv is not None:
        batch_seq_valid_frames_obsv=torch2numpy(batch_seq_valid_frames_obsv)
        batch_seq_valid_frames_pred=torch2numpy(batch_seq_valid_frames_pred)
    flatten_imgs=torch2numpy(flatten_imgs)


    batch_size, len_frames =batch_seq_gt_cam.shape[0],batch_seq_gt_cam.shape[1]

    ctrj_gt_cam,ctrj_est_cam= batch_seq_gt_cam[sample_id],batch_seq_est_cam[sample_id]
    if batch_seq_gt_local is not None:
        ctrj_gt_local,ctrj_est_local=batch_seq_gt_local[sample_id],batch_seq_est_local[sample_id]
    
    dir_cache=os.path.dirname(f"{prefix_cache_img}0.png")
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    dir_cache=os.path.dirname(path_video)
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    
    
    if batch_seq_valid_frames_obsv is None:
        len_cd=0
        len_pred_actual=batch_seq_est_cam.shape[1]+len_cd
        len_obsv=len_frames-len_pred_actual
        len_obsv_actual=len_obsv
        #len_cd=len_obsv
        print(len_frames,len_pred_actual,len_obsv)
    else:
        len_obsv_actual=np.sum(batch_seq_valid_frames_obsv[sample_id])
        len_pred_actual=np.sum(batch_seq_valid_frames_pred[sample_id])
        len_obsv=batch_seq_valid_frames_obsv.shape[1]
        len_cd=16

        print(len_obsv_actual,len_obsv, len_pred_actual,len_cd)
    for frame_id in range(0,len_frames): 
        if (batch_seq_valid_frames_obsv is not None) and (frame_id>=len_obsv_actual and frame_id<len_obsv) or frame_id>=len_obsv+len_pred_actual:
            continue
        fig = plt.figure(figsize=(2,4))

        num_rows, num_cols=3,2
        axes=fig.subplots(num_rows,num_cols)
        
        axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=2)
        axi.axis("off")
        
        if frame_id<len_obsv:
            title_ite_tag="Obsv"
        elif frame_id<len_obsv+len_cd:
            title_ite_tag=""
        else:
            title_ite_tag="Pred"

        title_tag=f"Frame #{frame_id}/{title_ite_tag} \nGT Observed Action {batch_gt_action_name[sample_id]}"
        
        if not (batch_est_action_name is None):
            title_tag+=f"/Est {batch_est_action_name[sample_id]}"
        
        title_tag+="\nGT projection"
        axi.set_title(title_tag,fontsize=6)

        cimg=flatten_imgs[sample_id*len_frames+frame_id][:,:,::-1].copy() 
        axi.imshow(cimg) 
        
        gt_frame_id=frame_id#+batch_seq_gt_local.shape[1]-len_frames
        cframe_gt_joints2d=project_hand_3d2img(ctrj_gt_cam[gt_frame_id]*1000,cam_info["intr"],cam_info["extr"])            
        gt_c=(0,1.,0) if title_ite_tag!="" else (1.,0.,0)
        visualize_joints_2d(axi, cframe_gt_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        visualize_joints_2d(axi, cframe_gt_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        axi=plt.subplot2grid((num_rows,num_cols),(1,0),colspan=2)
        axi.imshow(cimg) 

        axi.axis("off")
        est_frame_id=frame_id+batch_seq_est_local.shape[1]-len_frames
        err_str=""
        if est_frame_id>=0:
            err=np.linalg.norm(ctrj_est_cam[est_frame_id]-ctrj_gt_cam[gt_frame_id],axis=-1)
            vis_left=[0]+list(range(2,21))
            vis_right=[21]+list(range(23,42))
            err_str="Err L/R {:.2f}/{:.2f}mm".format(err[vis_left].mean()*1000,err[vis_right].mean()*1000)
            
        axi.set_title(f"Est. {err_str}",fontsize=6)
        if est_frame_id>=0:
            #c=(0.,1.,1) if est_frame_id%45<15 else ((1.,0,1.) if est_frame_id%45>=15 and est_frame_id%45<30 else (1.,1.,0.))
            c=(0.,1.,1) if est_frame_id%(16*2)<16 else(1.,0,1.)#########
            cframe_est_joints2d=project_hand_3d2img(ctrj_est_cam[est_frame_id]*1000,cam_info["intr"],cam_info["extr"])
            visualize_joints_2d(axi, cframe_est_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)
            visualize_joints_2d(axi, cframe_est_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)
        else:
            visualize_joints_2d(axi, cframe_gt_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
            visualize_joints_2d(axi, cframe_gt_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
            

        if batch_seq_gt_local is not None:
            subplot_frame_gt_est_onehand(ctrj_gt_local,ctrj_est_local,gt_frame_id,est_frame_id,6,'L',True,joint_links, num_rows=num_rows,num_cols=num_cols)
            subplot_frame_gt_est_onehand(ctrj_gt_local,ctrj_est_local,gt_frame_id,est_frame_id,5,'Trj Est(blue)/GT(green)-R',False,joint_links, num_rows=num_rows,num_cols=num_cols)
        

        
        #subplot_trj(ctrj_gt_cam,ctrj_est_cam,gt_frame_id,est_frame_id,8,'L',True,joint_links, num_rows=num_rows,num_cols=num_cols)
        #subplot_trj(ctrj_gt_cam,ctrj_est_cam,gt_frame_id,est_frame_id,7,'Trj Est(blue)/GT(green)-R',False,joint_links, num_rows=num_rows,num_cols=num_cols)
        
        fig.savefig(f"{prefix_cache_img}{frame_id}.png", dpi=200)
        plt.close(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(f"{prefix_cache_img}0.png")
    #print('write to',save_img_prefix+'_video.avi')
    videoWriter = cv2.VideoWriter(path_video, fourcc, 10, (cimg.shape[1],cimg.shape[0]))  ##########

    for frame_id in range(0,len_frames):
        if (batch_seq_valid_frames_obsv is not None) and (frame_id>=len_obsv_actual and frame_id<len_obsv) or frame_id>=len_obsv+len_pred_actual:
            continue
        cimg=cv2.imread(f"{prefix_cache_img}{frame_id}.png")
        videoWriter.write(cimg)
    
    videoWriter.release()



def subplot_frame_gt_twohands(ctrj_gt,gt_frame_id, plot_id, joint_links, elev=20,azim=250,num_rows=4,num_cols=8,boundary=0.0):
    #azim+=180
    ax3d=plt.subplot(num_rows, num_cols, plot_id, projection='3d')
    ctrj_gt=ctrj_gt.copy()   
    ax3d.set_xlim(ctrj_gt[:,:,0].min()-boundary,ctrj_gt[:,:,0].max()+boundary)   
    ax3d.set_ylim(ctrj_gt[:,:,2].min()-boundary,ctrj_gt[:,:,2].max()+boundary)   
    ax3d.set_zlim(ctrj_gt[:,:,1].min()-boundary,ctrj_gt[:,:,1].max()+boundary)  

    ax3d.view_init(elev,azim)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_title('',fontsize=6,pad=0.)
    for l in joint_links:
        ax3d.plot(ctrj_gt[gt_frame_id,l,0],ctrj_gt[gt_frame_id,l,2],ctrj_gt[gt_frame_id,l,1],alpha=0.8,c=(0,1.,0),linewidth=0.75)#0.5)
    
    
    for l in joint_links:
        l=[x+21 for x in l]
        ax3d.plot(ctrj_gt[gt_frame_id,l,0],ctrj_gt[gt_frame_id,l,2],ctrj_gt[gt_frame_id,l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.75)#0.5)



def sample_vis_l2r(batch_seq_gt_cam, batch_seq_gt_local, joint_links, prefix_cache_img, path_video, sample_id, cam_info,  flatten_imgs):
    batch_seq_gt_cam=torch2numpy(batch_seq_gt_cam)
    batch_seq_gt_local=torch2numpy(batch_seq_gt_local).copy()
    batch_seq_gt_local[...,1]=-batch_seq_gt_local[...,1]    
    batch_seq_gt_local[...,2]=-batch_seq_gt_local[...,2]
    flatten_imgs=torch2numpy(flatten_imgs)
    
    batch_size, len_frames =batch_seq_gt_cam.shape[0],batch_seq_gt_cam.shape[1]

    ctrj_gt_cam= batch_seq_gt_cam[sample_id]
    ctrj_gt_local=batch_seq_gt_local[sample_id]
    
    dir_cache=os.path.dirname(f"{prefix_cache_img}0.png")
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    dir_cache=os.path.dirname(path_video)
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
     
    for frame_id in range(0,len_frames): 
        fig = plt.figure(figsize=(3,3))

        num_rows, num_cols=2,1
        axes=fig.subplots(num_rows,num_cols)
        
        axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=1)
        axi.axis("off")
        
        title_tag=f"Frame #{frame_id}"         
        axi.set_title(title_tag,fontsize=6)

        cimg=flatten_imgs[sample_id*len_frames+frame_id][:,:,::-1].copy() 
        axi.imshow(cimg) 
        
        gt_frame_id=frame_id#+batch_seq_gt_local.shape[1]-len_frames
        cframe_gt_joints2d = ctrj_gt_cam[gt_frame_id] if cam_info is None else project_hand_3d2img(ctrj_gt_cam[gt_frame_id]*1000,cam_info["intr"],cam_info["extr"])            

        gt_c=(0,1.,0)
        visualize_joints_2d(axi, cframe_gt_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        visualize_joints_2d(axi, cframe_gt_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)

        subplot_frame_gt_twohands(ctrj_gt_local, gt_frame_id,2,joint_links, num_rows=num_rows,num_cols=num_cols)
        fig.savefig(f"{prefix_cache_img}{frame_id}.png", dpi=200)
        plt.close(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(f"{prefix_cache_img}0.png")
    #print('write to',save_img_prefix+'_video.avi')
    videoWriter = cv2.VideoWriter(path_video, fourcc, 5, (cimg.shape[1],cimg.shape[0]))  

    for frame_id in range(0,len_frames):
        cimg=cv2.imread(f"{prefix_cache_img}{frame_id}.png")
        videoWriter.write(cimg)
    
    videoWriter.release()


def sample_vis_ncam_cam(batch_seq_cam_joints2d, batch_seq_ncam_joints2d, batch_seq_cam_joints3d, batch_seq_ncam_joints3d, 
            joint_links, prefix_cache_img, path_video, sample_id, flatten_imgs):
    batch_seq_cam_joints2d=torch2numpy(batch_seq_cam_joints2d)
    batch_seq_cam_joints3d=torch2numpy(batch_seq_cam_joints3d)

    batch_seq_ncam_joints2d=torch2numpy(batch_seq_ncam_joints2d)
    batch_seq_ncam_joints3d=torch2numpy(batch_seq_ncam_joints3d)    
    flatten_imgs=torch2numpy(flatten_imgs)
    
    batch_size, len_frames = batch_seq_cam_joints2d.shape[0], batch_seq_cam_joints2d.shape[1]
    
    dir_cache=os.path.dirname(f"{prefix_cache_img}0.png")
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    dir_cache=os.path.dirname(path_video)
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
     
    for frame_id in range(0,len_frames): 
        fig = plt.figure(figsize=(3,3))

        num_rows, num_cols=2,1
        axes=fig.subplots(num_rows,num_cols)
        
        axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=1)
        axi.axis("off")
        
        title_tag=f"Frame #{frame_id}"
         
        axi.set_title(title_tag,fontsize=6)

        cimg=flatten_imgs[sample_id*len_frames+frame_id][:,:,::-1].copy() 
        axi.imshow(cimg) 
        
        gt_c=(0,1.,0)
        visualize_joints_2d(axi, batch_seq_cam_joints2d[sample_id,frame_id,:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        visualize_joints_2d(axi, batch_seq_cam_joints2d[sample_id,frame_id,21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)

        
        est_c=(1.,0.,0)
        visualize_joints_2d(axi, batch_seq_ncam_joints2d[sample_id,frame_id,:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[est_c]*5)
        visualize_joints_2d(axi, batch_seq_ncam_joints2d[sample_id,frame_id,21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[est_c]*5)


        ax3d=plt.subplot(2,1, 2, projection='3d')        
        ax3d.set_xlim(min(batch_seq_cam_joints3d[sample_id,frame_id,:,0].min(),batch_seq_ncam_joints3d[sample_id,frame_id,:,0].min()),
                max(batch_seq_cam_joints3d[sample_id,frame_id,:,0].max(),batch_seq_ncam_joints3d[sample_id,frame_id,:,0].max()))   
        ax3d.set_ylim(min(batch_seq_cam_joints3d[sample_id,frame_id,:,2].min(),batch_seq_ncam_joints3d[sample_id,frame_id,:,2].min()),
                max(batch_seq_cam_joints3d[sample_id,frame_id,:,2].max(),batch_seq_ncam_joints3d[sample_id,frame_id,:,2].max()))   
        ax3d.set_zlim(min(batch_seq_cam_joints3d[sample_id,frame_id,:,1].min(),batch_seq_ncam_joints3d[sample_id,frame_id,:,1].min()),
                max(batch_seq_cam_joints3d[sample_id,frame_id,:,1].max(),batch_seq_ncam_joints3d[sample_id,frame_id,:,1].max()))   
        
        for l in joint_links:
            ax3d.plot(batch_seq_cam_joints3d[sample_id,frame_id,l,0],batch_seq_cam_joints3d[sample_id,frame_id,l,2],batch_seq_cam_joints3d[sample_id,frame_id,l,1],alpha=0.8,c=gt_c,linewidth=0.75)
            ax3d.plot(batch_seq_ncam_joints3d[sample_id,frame_id,l,0],batch_seq_ncam_joints3d[sample_id,frame_id,l,2],batch_seq_ncam_joints3d[sample_id,frame_id,l,1],alpha=0.8,c=est_c,linewidth=0.75) 

        for l in joint_links:
            l=[x+21 for x in l]
            ax3d.plot(batch_seq_cam_joints3d[sample_id,frame_id,l,0],batch_seq_cam_joints3d[sample_id,frame_id,l,2],batch_seq_cam_joints3d[sample_id,frame_id,l,1],alpha=0.8,c=gt_c,linewidth=0.75)         
            ax3d.plot(batch_seq_ncam_joints3d[sample_id,frame_id,l,0],batch_seq_ncam_joints3d[sample_id,frame_id,l,2],batch_seq_ncam_joints3d[sample_id,frame_id,l,1],alpha=0.8,c=est_c,linewidth=0.75) 


        fig.savefig(f"{prefix_cache_img}{frame_id}.png", dpi=200)
        plt.close(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(f"{prefix_cache_img}0.png")
    #print('write to',save_img_prefix+'_video.avi')
    videoWriter = cv2.VideoWriter(path_video, fourcc, 5, (cimg.shape[1],cimg.shape[0]))  

    for frame_id in range(0,len_frames):
        cimg=cv2.imread(f"{prefix_cache_img}{frame_id}.png")
        videoWriter.write(cimg)
    
    videoWriter.release()



def subplot_trj(ctrj_gt,ctrj_est,gt_frame_id,est_frame_id,plot_id,title,is_left,joint_links, elev=30,azim=150,num_rows=4,num_cols=8,boundary=0.03):
    #azim+=180
    #trj for left
    ax3d=plt.subplot(num_rows, num_cols, plot_id, projection='3d')  
    len_frames=ctrj_gt.shape[0]
    if is_left:
        ax3d.set_xlim(ctrj_gt[:,:21,0].min()-boundary,ctrj_gt[:,:21,0].max()+boundary+0.1)   
        ax3d.set_ylim(ctrj_gt[:,:21,2].min()-boundary,ctrj_gt[:,:21,2].max()+boundary)   
        ax3d.set_zlim(ctrj_gt[:,:21,1].min()-boundary,ctrj_gt[:,:21,1].max()+boundary)  
    else:
        ax3d.set_xlim(ctrj_gt[:,21:,0].min()-boundary,ctrj_gt[:,21:,0].max()+boundary+0.1)   
        ax3d.set_ylim(ctrj_gt[:,21:,2].min()-boundary,ctrj_gt[:,21:,2].max()+boundary)   
        ax3d.set_zlim(ctrj_gt[:,21:,1].min()-boundary,ctrj_gt[:,21:,1].max()+boundary)  
    ax3d.view_init(elev,azim)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    cmap_est=cm.get_cmap('cool')        
    
    
    ax3d.set_title(title,fontsize=6,pad=0.)
    
    for trj_id in range(4,-1,-1):
        ref_id=est_frame_id-trj_id
        if ref_id<0:
            continue
        for l in joint_links:
            if not is_left:
                l=[x+21 for x in l]
            ax3d.plot(ctrj_est[ref_id,l,0]+0.1,ctrj_est[ref_id,l,2],ctrj_est[ref_id,l,1],c=cmap_est(ref_id/len_frames),linewidth=0.75)#0.5)
    cmap_gt=cm.get_cmap('summer')
    
    for trj_id in range(4,-1,-1):
        ref_id=gt_frame_id-trj_id
        if ref_id<0:
            continue         
        for l in joint_links:
            if not is_left:
                l=[x+21 for x in l]
            ax3d.plot(ctrj_gt[ref_id,l,0],ctrj_gt[ref_id,l,2],ctrj_gt[ref_id,l,1],alpha=1.0,c=cmap_gt(ref_id/len_frames),linewidth=0.75)#0.5)
            ax3d.set_xlabel("x",fontsize=6,labelpad=-15)
            ax3d.set_ylabel("z",fontsize=6,labelpad=-15)
            ax3d.set_zlabel("y",fontsize=6,labelpad=-15)


#below are for HTT figures.
def plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title):#,gt_colors,est_colors):
    ax3d_xmin,ax3d_xmax=trj_gt[0:33,:,0].min(),trj_gt[0:33,:,0].max()#min(trj_est[:,:,0].min(),trj_gt[:,:,0].min()),max(trj_est[:,:,0].max(),trj_gt[:,:,0].max())
    ax3d_ymax,ax3d_ymin=trj_gt[0:33,:,1].min(),trj_gt[0:33,:,1].max()#min(trj_est[:,:,1].min(),trj_gt[:,:,1].min()),max(trj_est[:,:,1].max(),trj_gt[:,:,1].max())
    ax3d_zmin,ax3d_zmax=trj_gt[0:33,:,2].min(),trj_gt[0:33,:,2].max()#min(trj_est[:,:,2].min(),trj_gt[:,:,2].min()),max(trj_est[:,:,2].max(),trj_gt[:,:,2].max())
    ax3d.set_xlim(ax3d_xmin-0.005,ax3d_xmax+0.005)            
    ax3d.set_zlim(ax3d_ymin-0.005,ax3d_ymax+0.005)            
    ax3d.set_ylim(ax3d_zmin-0.005,ax3d_zmax+0.005)
    
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    cgt,cest=trj_gt[frame_id],trj_est[frame_id]
    err=0.
    for i in range(0,21):
        cerr=0
        for ii in range(0,3):
            cerr+=(cest[i,ii]-cgt[i,ii])**2
        err+=cerr**0.5
    err=err/21*1000

    if not (title is None):
        title+='{:.2f}mm'.format(err)
        ax3d.set_title(title,fontsize=4,pad=0.)

    link= [[0, 1, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]]
    for l in link:
        ax3d.plot(trj_gt[frame_id,l,0],trj_gt[frame_id,l,2],trj_gt[frame_id,l,1],alpha=0.8,c=(0,1.,0),linewidth=0.75)#0.5)
    for l in link:
        ax3d.plot(trj_est[frame_id,l,0],trj_est[frame_id,l,2],trj_est[frame_id,l,1],alpha=1.0,c=(0,0,1.),linewidth=0.75)#0.5)



def draw_2d_3d_pose(batch_seq_gt_cam,batch_seq_est_cam,batch_seq_imgs,sample_id,frame_id,cam_intr, num_rows, num_cols, is_single_hand, title):
    ctrj_gt_cam,ctrj_est_cam=batch_seq_gt_cam[sample_id],batch_seq_est_cam[sample_id]
    ctrj_gt_cam2d,ctrj_est_cam2d=ctrj_gt_cam.copy(),ctrj_est_cam.copy()
    
    offset=0.025

    #img
    axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=1 if is_single_hand and num_rows>1 else 2) 
    #axi.set_xlim([-150,500])
    #axi.set_ylim([300,-100])
    axi.axis("off")
    if not (title is None):
        axi.set_title(title,fontsize=4,pad=0.) #'Seq MEPE: {:.2f}mm'.format(mepe*1000,batch_pred_action[sample_id], batch_gt_action[sample_id]),
    if is_single_hand:
        simg=np.zeros((400,550,3),dtype=np.uint8)+255
        simg[100:100+270,50:50+480]=batch_seq_imgs[sample_id,frame_id].copy()
    else:
        simg=batch_seq_imgs[sample_id,frame_id].copy()
    axi.imshow(simg)#images[row_idx])
    cframe_gt_joints2d=project_hand_3d2img(ctrj_gt_cam2d[frame_id],cam_intr)
    cframe_est_joints2d=project_hand_3d2imgd(ctrj_est_cam2d[frame_id],cam_intr)  #ctrj_est_cam2d[frame_id]
    if is_single_hand:
        cframe_gt_joints2d[:,0]+=50
        cframe_gt_joints2d[:,1]+=100
        cframe_est_joints2d[:,0]+=50
        cframe_est_joints2d[:,1]+=100

    visualize_joints_2d(axi, cframe_gt_joints2d[:21], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
    visualize_joints_2d(axi, cframe_est_joints2d[:21], alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,0,1.)]*5)

    try:
        visualize_joints_2d(axi, cframe_gt_joints2d[21:], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
        visualize_joints_2d(axi, cframe_est_joints2d[21:], alpha=1,linewidth=.7, scatter=False, joint_idxs=False,color=[(0,0,1.)]*5)#[(1.,0.,0.)]*5)
    except:
        x=0
    
    if is_single_hand:
        dev=3 if num_rows==1 else 2
    else:
        dev=4# if num_cols==1 else 4
    ax3d=plt.subplot(num_rows, num_cols, dev, projection='3d')
    #ax3d.axis("off")
    if is_single_hand:
        trj_est=ctrj_est_cam[:,:21]#[frame_id,:21]
        trj_gt=ctrj_gt_cam[:,:21]#[frame_id,:21]
        if num_rows==1:
            title=None
        else:
            title='Frame MEPE: '
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title)#,gt_colors,est_colors)
    else:
        trj_est=ctrj_est_cam[:,21:]
        trj_gt=ctrj_gt_cam[:,21:]

        if num_rows==1:
            title=None
        else:
            title='R: '
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title)#,gt_colors,est_colors)
    ax3d.view_init(30,210)

    if not is_single_hand:
        ax3d=plt.subplot(num_rows, num_cols, dev-1, projection='3d')
        trj_est=ctrj_est_cam[:,:21]#[frame_id,:21]
        trj_gt=ctrj_gt_cam[:,:21]#[frame_id,:21]
        if num_rows==1:
            title=None
        else:
            title='Frame EPE - L: '
        plot_cframe_est_gt(trj_est,trj_gt,frame_id,ax3d,title)#,gt_colors,est_colors)



def sample_vis_trj_enc_attn(batch_seq_gt_cam, batch_seq_est_cam, batch_seq_padding, cam_intr, len_seq_pose, dir_iout,prefix_vout, batch_seq_imgs,
                batch_pred_action, batch_gt_action, batch_seq_pred_olabel,batch_seq_gt_olabel,batch_seq_attn_pose,batch_seq_attn_action,
                sample_id=0,mepe=0.,is_single_hand=True):
    
    batch_seq_gt_cam=torch2numpy(batch_seq_gt_cam)
    batch_seq_est_cam=torch2numpy(batch_seq_est_cam)
    batch_seq_padding=torch2numpy(batch_seq_padding)
    cam_intr=torch2numpy(cam_intr)
    batch_seq_imgs=torch2numpy(batch_seq_imgs)
    batch_pred_action=torch2numpy(batch_pred_action)
    batch_gt_action=torch2numpy(batch_gt_action)
    batch_seq_pred_olabel=torch2numpy(batch_seq_pred_olabel)
    batch_seq_gt_olabel=torch2numpy(batch_seq_gt_olabel)    
    
    for i in range(0,len(batch_seq_attn_pose)):
        batch_seq_attn_pose[i]=torch2numpy(batch_seq_attn_pose[i])
    for i in range(0,len(batch_seq_attn_action)):
        batch_seq_attn_action[i]=torch2numpy(batch_seq_attn_action[i])
    
    batch_size, len_frames=batch_seq_gt_cam.shape[0:2]
    
    end_frame=len_frames
    for frame_id in range(0,len_frames):
        if batch_seq_padding[sample_id,frame_id]<1e-4:
            end_frame=frame_id
            break
            

    #gt_colors=cm.Greens(np.linspace(0,1,len_frames+20))[::-1]
    #est_colors=cm.Oranges(np.linspace(0,1,len_frames+20))[::-1]
    for frame_id in range(0,end_frame):
        
        num_cols=3 if is_single_hand else 4
        fig = plt.figure(figsize=(num_cols-1,1))
        # draw main fig
        draw_2d_3d_pose(batch_seq_gt_cam,batch_seq_est_cam,batch_seq_imgs,sample_id,frame_id,cam_intr, \
                    num_rows=1, num_cols=num_cols,is_single_hand=is_single_hand, title=None)
        fig.savefig(os.path.join(dir_iout,"mp_{:03d}.png".format(frame_id+1)), dpi=400)
        plt.close(fig)

        continue
        
        #draw pose video
        fig = plt.figure(figsize=(3,3))
        num_rows=3#2 if is_single_hand else 3
        num_cols=1 if is_single_hand else 2
        #axes=fig.subplots(num_rows,num_cols)

        if len_seq_pose==1:
            title='w/o Temporal Cue ($t$=1, $T$=128)\n'
        elif len_seq_pose==16:
            title='w/ Short-term Temporal Cue ($t$=16, $T$=128)\n'
        else:
            title='w/ Long-term Temporal Cue ($t$=128, $T$=128)\n'
        if is_single_hand:
            title+='Seq MEPE: {:.2f}mm\nFrame {:d}'.format(mepe*1000, frame_id+1)
        else:
            title+='Seq MEPE - L: {:.2f}mm / R: {:.2f}mm\nFrame {:d}'.format(mepe[0]*1000, mepe[1]*1000, frame_id+1)

        draw_2d_3d_pose(batch_seq_gt_cam,batch_seq_est_cam,batch_seq_imgs,sample_id,frame_id,cam_intr, num_rows=num_rows, num_cols=num_cols, \
                title=title,is_single_hand=is_single_hand)
        
        #fig.savefig(os.path.join(dir_iout,"vp_{:03d}.png".format(frame_id)), dpi=300)
        #dev= num_rows-1
        for i in range(1,2):
            axi=plt.subplot2grid((num_rows,1),(2,0),colspan=2)#1 if is_single_hand else 2) 
            #axi.set_position([-0.5,0.1,0.5,0.337])
            #print(axi.get_position())
            axi.axis("off")
            if len_seq_pose==1:
                continue
            axi.set_title('Attention Weights in the Final Layer of $P$'.format(i+1),fontsize=4, pad=4)
            #axes[dev,i].axis("off")
            #axes[dev,i].set_title('$P$ Layer {:d}'.format(i+1),fontsize=4, pad=0)
            if len_seq_pose==128:
                cmap=batch_seq_attn_pose[i][sample_id,0]#,:end_frame,:end_frame]        
                calpha=np.zeros(cmap.shape,cmap.dtype)+0.5
                calpha[frame_id]=1.
            else:
                cmap=np.zeros((128,128),dtype=np.float32)
                for ffid in range(0,end_frame,len_seq_pose):
                    sid,eid=ffid//len_seq_pose*len_seq_pose,(ffid//len_seq_pose+1)*len_seq_pose
                    cmap[sid:eid,sid:eid]=batch_seq_attn_pose[i][sample_id,ffid//len_seq_pose]            
                calpha=np.zeros(cmap.shape,cmap.dtype)+0.5
                calpha[frame_id,sid:eid]=1.
            im=axi.imshow(cmap[:end_frame,:end_frame],vmin=0,vmax=0.3)  #alpha=calpha[:end_frame,:end_frame],
            #axi.arrow(-2,frame_id,1,0,width=0.3,fc='r',ec=None)           
            if i==1:
                cbar=axi.figure.colorbar(im, ax=axi, pad=0.02,aspect=20, fraction=0.05, ticks=[])#0,0.1,0.2,0.3])#,orientation='horizontal')
                cbar.ax.tick_params(labelsize=4,size=0,pad=0.01)
                cbar.outline.set_visible(False)

        
        fig.savefig(os.path.join(dir_iout,"vp_{:03d}.png".format(frame_id)), dpi=300,facecolor='white',transparent=False)
        plt.close(fig)

        continue


        #draw action video
        if False and frame_id==0:
            fig = plt.figure(figsize=(8,0.8))

            #axi=plt.subplot2grid((2,16),(0,0),colspan=16) 
            #axi.imshow(batch_seq_attn_action[0][sample_id,0:1,1:end_frame+1],vmin=0,vmax=0.2) 
            #axi.axis("off")

            axi=plt.subplot2grid((1,16),(0,0),colspan=16) 
            alpha=np.zeros(batch_seq_attn_action[1][sample_id,0:1,1:end_frame+1].shape,np.float32)
            for i in range(0,71,10):
                alpha[0,i]=1.
            im=axi.imshow(batch_seq_attn_action[1][sample_id,0:1,1:end_frame+1],alpha=alpha,vmin=0,vmax=0.2) 

            cbar=axi.figure.colorbar(im, ax=axi,pad=0.5,  aspect=8, fraction=0.25, ticks=[0,0.2])
            cbar.ax.tick_params(labelsize=16,size=0,pad=0.01)
            cbar.outline.set_visible(False)
            axi.axis("off")


            fig.savefig(os.path.join(dir_iout,"_alpha.png"), dpi=300)
            #fig.savefig(f"{prefix_cache_img}_{frame_id}.png", dpi=300)
            plt.close(fig)
            for ii in range(0,end_frame):
                cv2.imwrite(os.path.join(dir_iout,"_{:04d}.png".format(ii)), batch_seq_imgs[sample_id,ii][:,:,::-1])


        fig = plt.figure(figsize=(3,2))
        #axes=fig.subplots(3,2)


        axi=plt.subplot2grid((2,1),(0,0),colspan=1) 
        axi.set_title('Ours w/ $t$={:d}, $T$=128\nEst: {:s}/GT: {:s}\nFrame {:d}'.format(len_seq_pose,batch_pred_action[sample_id],batch_gt_action[sample_id],frame_id+1),\
                    fontsize=4,pad=0.) #'Seq MEPE: {:.2f}mm'.format(mepe*1000,batch_pred_action[sample_id], batch_gt_action[sample_id]),
        axi.imshow(batch_seq_imgs[sample_id,frame_id])#images[row_idx])
        axi.axis("off")

        dev=1
        for i in range(1,2):
            axi=plt.subplot2grid((2,1),(1,0),colspan=1)
            #axes[dev+1,i].set_title('$A$ Layer {:d}'.format(i),fontsize=4, pad=0)
            #axes[dev+1,i].axis("off")

            #cmap=batch_seq_attn_action[i][sample_id,:end_frame+1,:end_frame+1]
            #im=axi.imshow(cmap,vmin=0,vmax=0.2) 

            im=axi.imshow(batch_seq_attn_action[1][sample_id,0:1,1:end_frame+1],vmin=0,vmax=0.2) 
            axi.arrow(frame_id,-2.5,0,1.,width=0.3,head_length=0.8,head_width=0.75,fc='red',ec='red')#ec='white') 
            axi.set_xlim([-1,end_frame+2])
            if i==1:
                cbar=axi.figure.colorbar(im, ax=axi, pad=0.1,aspect=25,fraction=0.05, ticks=[0,0.1,0.2],orientation='horizontal')#aspect=8, fraction=0.2, 
                cbar.ax.tick_params(labelsize=4,size=0,pad=0.01)
                cbar.outline.set_visible(False)


            axi.set_title("Attention Weights in the Final Layer of $A$\nFrom Action Token to Frames",fontsize=4, pad=4)# r'$\alpha$'
            axi.axis("off")
        
        fig.savefig(os.path.join(dir_iout,"va_{:03d}.png".format(frame_id)), dpi=300,facecolor='white',transparent=False)
        #fig.savefig(f"{prefix_cache_img}_{frame_id}.png", dpi=300)
        plt.close(fig)
        

    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(os.path.join(dir_iout,"mp_{:03d}.png".format(1)))
    videoWriter = cv2.VideoWriter(prefix_vout+'_mp.avi', fourcc, 4, (610-140,cimg.shape[0]))  

    for frame_id in range(1,end_frame+1):
        cimg=cv2.imread(os.path.join(dir_iout,"mp_{:03d}.png".format(frame_id)))
        videoWriter.write(cimg[:,140:610])
    
    videoWriter.release()
    return

    cimg=cv2.imread(os.path.join(dir_iout,"vp_{:03d}.png".format(0)))
    videoWriter = cv2.VideoWriter(prefix_vout+'_vp.avi', fourcc, 2, (cimg.shape[1],cimg.shape[0]))  

    for frame_id in range(0,end_frame):
        cimg=cv2.imread(os.path.join(dir_iout,"vp_{:03d}.png".format(frame_id)))
        videoWriter.write(cimg)
    
    videoWriter.release()



def write_images_to_video(seq_images, path_video, gt_pose=None, fps=1):
    bones_start_idx=[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    bones_end_idx=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    videoWriter = cv2.VideoWriter(path_video, fourcc, fps, (seq_images.shape[2],seq_images.shape[1]))  
    for img_id in range(0,seq_images.shape[0]):#seq_images.shape[0]-1,-1,-1):
        out_img=seq_images[img_id].astype(np.uint8).copy()
        print(out_img.shape)
        skel2d=gt_pose[img_id]
        skel2d[:,0]=np.clip(skel2d[:,0],0,480)
        skel2d[:,1]=np.clip(skel2d[:,1],0,270)
        print(skel2d.shape)
        for i in range(0,len(bones_start_idx)):
            sid,eid=bones_start_idx[i],bones_end_idx[i]

            cv2.line(out_img,(int(skel2d[sid][0]),int(skel2d[sid][1])),(int(skel2d[eid][0]),int(skel2d[eid][1])),(0,0,255),2)
            if skel2d.shape[0]>21:
                cv2.line(out_img,(int(skel2d[sid+21][0]),int(skel2d[sid+21][1])),(int(skel2d[eid+21][0]),int(skel2d[eid+21][1])),(255,0,0),2)

        videoWriter.write(out_img)
    videoWriter.release()


def compare_two_pose_sets(dict_pose1, dict_pose2, dict_mano, dict_imgs, cam_info, file_tag, joint_links1,joint_links2, prefix_cache_img,path_video):
    dir_cache=os.path.dirname(f"{prefix_cache_img}0.png")
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    dir_cache=os.path.dirname(path_video)
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)

    def get_local_pose(frame_raw):        
        pose_cam=frame_raw[:42*3].reshape((42,3))
        rt=frame_raw[42*3:42*3+12*2]
        
        palm_joints=[0,5,9,13,17]

        pose_local_left=np.dot(pose_cam[:21],rt[:9].reshape((3,3)))+rt[9:12].reshape((1,3))
        pose_local_right=np.dot(pose_cam[21:],rt[12:21].reshape((3,3)))+rt[21:24].reshape((1,3))


        left_size=np.mean(np.linalg.norm(pose_local_left[palm_joints][1:]-pose_local_left[palm_joints][0],ord=2,axis=1))
        right_size=np.mean(np.linalg.norm(pose_local_right[palm_joints][1:]-pose_local_right[palm_joints][0],ord=2,axis=1))
        pose_local_left_n=pose_local_left/left_size
        pose_local_right_n=pose_local_right/right_size

        return pose_cam,pose_local_left,pose_local_right,pose_local_left_n,pose_local_right_n,left_size,right_size
        

    def plot_local(plot_id, pose1_local_hand, pose2_local_hand, joint_links1, joint_links2, pose_mano, title):
        ax3d=plt.subplot(num_rows,num_cols,plot_id,projection='3d')
        ax3d.set_title(title,fontsize=6,pad=0.)
        ax3d.view_init(35,110)            
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        

        for l in joint_links1:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)
        for l in joint_links2:
            ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(1.,0.,0),linewidth=0.8)#0.5)


        for l in [[0,5],[0,9],[0,13],[0,17]]:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,1.),linewidth=0.8)#0.5)
        for l in [[0,5],[0,9],[0,13],[0,17]]:
            ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(1.,0.,1.),linewidth=0.8)#0.5)
        
        if pose_mano is not None:
            for l in [[0,1],[0,2],[0,3],[0,4]]:
                ax3d.plot(pose_mano[l,0],pose_mano[l,2],pose_mano[l,1],alpha=0.8,c=(0.,0.,1.),linewidth=0.8)#0.5)
        
        
    max_imgs=20*60
    for ffid, frame_id in enumerate(dict_pose1.keys()): 
        if ffid>max_imgs:
            break
        pose1,pose1_local_left,pose1_local_right,pose1_local_left_n,pose1_local_right_n,left_size1,right_size1=get_local_pose(dict_pose1[frame_id])
        pose2,pose2_local_left,pose2_local_right,pose2_local_left_n,pose2_local_right_n,left_size2,right_size2=get_local_pose(dict_pose2[frame_id])


        assert np.fabs(pose1_local_right[0]).max()<1e-8
        assert np.fabs(pose1_local_left[0]).max()<1e-8        
        assert np.fabs(pose2_local_right[0]).max()<1e-8
        assert np.fabs(pose2_local_left[0]).max()<1e-8
        

        fig = plt.figure(figsize=(5,3))

        num_rows, num_cols=2,4
        axes=fig.subplots(num_rows,num_cols)
        
        axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=2)
        axi.axis("off")
                
           
        cimg=dict_imgs[frame_id][:,:,::-1].copy() 
        axi.imshow(cimg) 
    
        pose1_2d=project_hand_3d2img(pose1*1000,cam_info["intr"],cam_info["extr"])        
        visualize_joints_2d(axi, pose1_2d[:21], alpha=1,linewidth=.6,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
        visualize_joints_2d(axi, pose1_2d[21:], alpha=1,linewidth=.6,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)


        #########################
        pose2_2d=project_hand_3d2img(pose2*1000,cam_info["intr"],cam_info["extr"])
        visualize_joints_2d(axi, pose2_2d[:21], alpha=1,linewidth=.6,scatter=False, joint_idxs=False,color=[(1.,0.,0)]*5)
        visualize_joints_2d(axi, pose2_2d[21:], alpha=1,linewidth=.6,scatter=False, joint_idxs=False,color=[(1.,0.,0)]*5)
        
        #Local but un-normalized shape

        palm_joints=[0,5,9,13,17]
        visible_joints=[0]+list(range(2,21))

        err=np.mean(np.linalg.norm(pose1[:21][visible_joints]-pose2[:21][visible_joints],ord=2,axis=1))
        plot_local(6,pose1[:21],pose2[:21],joint_links1,joint_links2,title="L-err {:.1f}mm".format(err*1000),pose_mano=None)
        err=np.mean(np.linalg.norm(pose1[21:][visible_joints]-pose2[21:][visible_joints],ord=2,axis=1))
        plot_local(5,pose1[21:],pose2[21:],joint_links1,joint_links2,title="Cam R-err {:.1f}mm".format(err*1000),pose_mano=None)
        
        
        
        mano_left=dict_mano["left"].copy()
        mano_palm_left=mano_left/np.mean(np.linalg.norm(mano_left[1:]-mano_left[0],ord=2,axis=1))
        err1_mano=np.mean(np.linalg.norm(mano_palm_left*left_size1-pose1_local_left[palm_joints],ord=2,axis=1))
        err2_mano=np.mean(np.linalg.norm(mano_palm_left*left_size2-pose2_local_left[palm_joints],ord=2,axis=1))
        err=np.mean(np.linalg.norm(pose1_local_left[visible_joints]-pose2_local_left[visible_joints],ord=2,axis=1))
        plot_local(8,pose1_local_left,pose2_local_left,joint_links1,joint_links2,title="L err{:.1f}mm\n w/ mano err\n {:.1f}/{:.1f}mm".format(err*1000,err1_mano*1000,err2_mano*1000),pose_mano=dict_mano["left"])
        
        mano_right=dict_mano["right"].copy()
        mano_palm_right=mano_right/np.mean(np.linalg.norm(mano_right[1:]-mano_right[0],ord=2,axis=1))
        err1_mano=np.mean(np.linalg.norm(mano_palm_right*right_size1-pose1_local_right[palm_joints],ord=2,axis=1))
        err2_mano=np.mean(np.linalg.norm(mano_palm_right*right_size2-pose2_local_right[palm_joints],ord=2,axis=1))        
        err=np.mean(np.linalg.norm(pose1_local_right[visible_joints]-pose2_local_right[visible_joints],ord=2,axis=1))
        plot_local(7,pose1_local_right,pose2_local_right,joint_links1,joint_links2, title="U-Local R-err {:.1f}mm\n w/ mano err\n {:.1f}/{:.1f}mm".format(err*1000,err1_mano*1000,err2_mano*1000),pose_mano=dict_mano["right"])
        
        

        #Local but normalized shape
        err=np.mean(np.linalg.norm(pose1_local_left_n[visible_joints]-pose2_local_left_n[visible_joints],ord=2,axis=1))
        plot_local(4,pose1_local_left_n,pose2_local_left_n,joint_links1,joint_links2,title="L err {:.1f}\nsize{:.1f}/{:.1f} cm".format(err,left_size1*100,left_size2*100),pose_mano=None)#dict_mano["left"])
        err=np.mean(np.linalg.norm(pose1_local_right_n[visible_joints]-pose2_local_right_n[visible_joints],ord=2,axis=1))
        plot_local(3,pose1_local_right_n,pose2_local_right_n,joint_links1,joint_links2,title="N-Local R err {:.1f}\n size {:.1f}/{:.1f} cm".format(err,right_size1*100,right_size2*100),pose_mano=None)#dict_mano["right"])


        fig.savefig(f"{prefix_cache_img}{frame_id}.png", dpi=100)
        plt.close(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(f"{prefix_cache_img}{list(dict_pose1.keys())[0]}.png")
    videoWriter = cv2.VideoWriter(path_video, fourcc, 30, (cimg.shape[1],cimg.shape[0]))  

    for ffid,frame_id in enumerate(dict_pose1.keys()):
        if ffid>max_imgs:
            break
        cimg=cv2.imread(f"{prefix_cache_img}{frame_id}.png")
        videoWriter.write(cimg)
    
    videoWriter.release()



def vis_taxonomy_embedding(model,list_pose_datasets,list_tags):
    def get_berts(pd,str_cmap):
        action_info=pd.action_info
        list_action_name=sorted(action_info.keys(),key=lambda k:action_info[k]["verb_name"])
        print(list_action_name)    

        action_berts=model.compute_berts_for_strs(list_action_name, verbose=True).detach().cpu().numpy()
        print(action_berts.shape)
        
        cmap=cm.get_cmap(str_cmap)
        vis_colors=[cmap(float(i)/action_berts.shape[0]) for i in range(action_berts.shape[0])]

        return list_action_name,action_berts,vis_colors
    
    list_names,list_berts,list_colors=[],[],[]
    list_cmaps=["nipy_spectral","Spectral"]
    for pd,cmap in zip(list_pose_datasets,list_cmaps):
        cname,cbert,ccolor=get_berts(pd,cmap)
        list_names.append(cname)
        list_berts.append(cbert)
        list_colors.append(ccolor)

    action_berts=np.concatenate(list_berts,axis=0)

    def vis(reduce_transformer,action_berts,list_colors,ttag):
        X_2d=reduce_transformer.fit_transform(action_berts)
        fig=plt.figure(figsize=(5,5),facecolor="white")

        pt=0
        markers=["o","v"]
        for sid, vis_colors in enumerate(list_colors):
            n=len(vis_colors)
            plt.scatter(X_2d[pt:pt+n,0],X_2d[pt:pt+n,1],s=10*(1+sid),marker=markers[sid],color=vis_colors)
            pt+=n
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"tax_{ttag}.png")
        plt.close()
            
    n_components=2
    reduce_transformer=TSNE(n_components=n_components,perplexity=100,random_state=0)
    vis(reduce_transformer,action_berts,list_colors,"tsne")
    
    reduce_transformer=PCA(n_components=n_components)
    vis(reduce_transformer,action_berts,list_colors,"pca")
    
    def vis_tax(pd,list_action_name,vis_colors,tag):
        cverb="NIL"
        bar_width= 10 if len(list_action_name)<100 else 1
        verb_color=np.zeros((len(list_action_name)*bar_width+10,200,3),dtype=np.uint8)+255
        for vid,vcolor in enumerate(vis_colors):
            r,g,b,a=vcolor
            vcolor=np.array([b*255,g*255,r*255],dtype=np.uint8)#.reshape((1,3))
            verb_color[5+vid*bar_width:5+(vid+1)*bar_width,170:]=vcolor
            if pd.action_info[list_action_name[vid]]["verb_name"]!=cverb:
                cverb=pd.action_info[list_action_name[vid]]["verb_name"]
                print(vid,cverb)
                cv2.putText(verb_color,cverb,(5,vid*bar_width+10), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_4)
        
        cv2.imwrite(f"tax_cbar_{tag}.png",verb_color)
    
    for pd,ctax,ctag,ccolor in zip(list_pose_datasets,list_names,list_tags,list_colors):
        print(ctag)
        vis_tax(pd,ctax,ccolor,ctag)
    







def sample_vis_frame(batch_est_cam,batch_gt_cam, batch_est_cent, batch_est_aligned, batch_gt_cent, batch_cam_intr, batch_weights,
                    batch_imgs,dir_to_save,batch_image_path,batch_est_cam0=None,batch_est_cent0=None,batch_est_aligned0=None,):
    
    batch_gt_cam=torch2numpy(batch_gt_cam)
    batch_gt_cent=torch2numpy(batch_gt_cent)
    batch_cam_intr=torch2numpy(batch_cam_intr)
    batch_imgs=torch2numpy(batch_imgs)

    batch_est_cam=torch2numpy(batch_est_cam)
    batch_est_cent=torch2numpy(batch_est_cent)
    batch_est_aligned=torch2numpy(batch_est_aligned)
    batch_weights=torch2numpy(batch_weights)
     
    batch_est_cam0=torch2numpy(batch_est_cam0)
    batch_est_cent0=torch2numpy(batch_est_cent0)
    batch_est_aligned0=torch2numpy(batch_est_aligned0)
    
    batch_size=batch_gt_cam.shape[0]
    
    for frame_id in range(0,batch_size):
        num_cols=3
        fig = plt.figure(figsize=(3,5.5))
        
        axi=plt.subplot2grid((5,1),(0,0),rowspan=2) 
        axi.axis("off")
            
        axi.imshow(batch_imgs[frame_id])#images[row_idx])
        #cframe_gt_joints2d=project_hand_3d2img(batch_gt_cam[frame_id],batch_cam_intr[frame_id])
        cframe_est_joints2d=project_hand_3d2img(batch_est_cam[frame_id],batch_cam_intr[frame_id])
        if batch_est_cam0 is not None:
            cframe_est_joints2d0=project_hand_3d2img(batch_est_cam0[frame_id],batch_cam_intr[frame_id])
        
        #visualize_joints_2d(axi, cframe_gt_joints2d[:21], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
        #visualize_joints_2d(axi, cframe_gt_joints2d[21:], alpha=0.8,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,1.,0)]*5)
        
        if batch_weights[frame_id]>0:
            visualize_joints_2d(axi, cframe_est_joints2d[:21], alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(0,0,1.)]*5)
            visualize_joints_2d(axi, cframe_est_joints2d[21:], alpha=1,linewidth=.7, scatter=False, joint_idxs=False,color=[(1.,0,1.)]*5)
        
        if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,:21]).sum()>0:
            visualize_joints_2d(axi, cframe_est_joints2d0[:21], alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[(1.,0,0.)]*5)
        if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,21:]).sum()>0:
            visualize_joints_2d(axi, cframe_est_joints2d0[21:], alpha=1,linewidth=.7, scatter=False, joint_idxs=False,color=[(1.,0,0.)]*5)
        

        links = [
            [0, 1,2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],]


        ax3d=plt.subplot(5,1,3, projection='3d')
        pose1_local_hand=batch_gt_cam[frame_id][:21]
        pose2_local_hand=batch_est_cam[frame_id][:21]
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_cam0[frame_id][:21]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)    
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(0.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,:21]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)
        
        pose1_local_hand=batch_gt_cam[frame_id][21:]
        pose2_local_hand=batch_est_cam[frame_id][21:]
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_cam0[frame_id][21:]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(1.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,21:]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)
        

        
        ax3d=plt.subplot(5,2,8, projection='3d')
        pose1_local_hand=batch_gt_cent[frame_id][:21]
        pose2_local_hand=batch_est_cent[frame_id][:21]
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_cent0[frame_id][:21]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5) 
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(0.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,:21]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)
        
        ax3d=plt.subplot(5,2,7, projection='3d')
        pose1_local_hand=batch_gt_cent[frame_id][21:]
        pose2_local_hand=batch_est_cent[frame_id][21:]        
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_cent0[frame_id][21:]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(1.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,21:]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)

        
        ax3d=plt.subplot(5,2,10, projection='3d')
        pose1_local_hand=batch_gt_cent[frame_id][:21]
        pose2_local_hand=batch_est_aligned[frame_id][:21]        
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_aligned0[frame_id][:21]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)   
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(0.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,:21]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)
        
        ax3d=plt.subplot(5,2,9, projection='3d')
        pose1_local_hand=batch_gt_cent[frame_id][21:]
        pose2_local_hand=batch_est_aligned[frame_id][21:]
        if batch_est_cam0 is not None:
            pose3_local_hand=batch_est_aligned0[frame_id][21:]
        for l in links:
            ax3d.plot(pose1_local_hand[l,0],pose1_local_hand[l,2],pose1_local_hand[l,1],alpha=0.8,c=(0,1.,0),linewidth=0.8)#0.5)
            if batch_weights[frame_id]>0:
                ax3d.plot(pose2_local_hand[l,0],pose2_local_hand[l,2],pose2_local_hand[l,1],alpha=0.8,c=(1.,0.,1.),linewidth=0.8)#0.5)
            if batch_est_cam0 is not None and np.fabs(batch_est_cam0[frame_id,21:]).sum()>0:
                ax3d.plot(pose3_local_hand[l,0],pose3_local_hand[l,2],pose3_local_hand[l,1],alpha=0.8,c=(1.,0.,0.),linewidth=0.8)#0.5)
                
        save_name2d=os.path.join(dir_to_save,batch_image_path[frame_id]+".png")
        if not os.path.exists(os.path.dirname(save_name2d)):
            os.makedirs(os.path.dirname(save_name2d))
        
        fig.savefig(save_name2d, dpi=150)
        plt.close(fig)


#utility from ACR
def save_video(path, out_name):
    print('saving to :', out_name + '.mp4')
    img_array = []
    height, width = 0, 0
    for filename in tqdm(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))):
        img = cv2.imread(path + '/' + filename)
        if height != 0:
            img = cv2.resize(img, (width, height))
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(out_name + '.mp4', 0x7634706d, 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('done')





def sample_vis_trj_resnet(batch_seq_gt_cam, batch_seq_est_cam1, batch_seq_est_cam2, joint_links, prefix_cache_img, path_video, sample_id, cam_info,
                    batch_seq_gt_local=None, batch_seq_est_local1=None, batch_seq_est_local2=None, flatten_imgs=None):
    batch_seq_gt_cam=torch2numpy(batch_seq_gt_cam)
    batch_seq_est_cam1=torch2numpy(batch_seq_est_cam1)
    batch_seq_est_cam2=torch2numpy(batch_seq_est_cam2)

    if batch_seq_gt_local is not None:
        batch_seq_gt_local=torch2numpy(batch_seq_gt_local)
        batch_seq_est_local1=torch2numpy(batch_seq_est_local1)
        batch_seq_est_local2=torch2numpy(batch_seq_est_local2)
        

    flatten_imgs=torch2numpy(flatten_imgs)
    batch_size, len_frames =batch_seq_gt_cam.shape[0],batch_seq_gt_cam.shape[1]

    ctrj_gt_cam,ctrj_est_cam1= batch_seq_gt_cam[sample_id],batch_seq_est_cam1[sample_id]
    ctrj_est_cam2= batch_seq_est_cam2[sample_id]
    if batch_seq_gt_local is not None:
        ctrj_gt_local,ctrj_est_local1=batch_seq_gt_local[sample_id],batch_seq_est_local1[sample_id]
        ctrj_est_local2=batch_seq_est_local2[sample_id]
    
    dir_cache=os.path.dirname(f"{prefix_cache_img}0.png")
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    dir_cache=os.path.dirname(path_video)
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)
    
    ctrj_err_cam1=np.linalg.norm(ctrj_gt_cam-ctrj_est_cam1,axis=-1)
    ctrj_err_cam2=np.linalg.norm(ctrj_gt_cam-ctrj_est_cam2,axis=-1)
    validL,validR=[0]+list(range(2,21)),[21]+list(range(23,42))
    ctrj_err_cam1_L, ctrj_err_cam1_R=np.mean(ctrj_err_cam1[:,validL],axis=-1), np.mean(ctrj_err_cam1[:,validR],axis=-1)
    ctrj_err_cam2_L, ctrj_err_cam2_R=np.mean(ctrj_err_cam2[:,validL],axis=-1), np.mean(ctrj_err_cam2[:,validR],axis=-1)

    #print(ctrj_err_cam1_L.shape,ctrj_err_cam1_R.shape)
    #print(ctrj_err_cam2_L.shape,ctrj_err_cam2_R.shape)
    #exit(0)
    for frame_id in range(0,len_frames): 
        
        fig = plt.figure(figsize=(2,5))

        num_rows, num_cols=4,2
        axes=fig.subplots(num_rows,num_cols)
        
        axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=2)
        axi.axis("off")
        
        if frame_id<16:
            title_ite_tag="Obsv"
        else:
            title_ite_tag="Pred"

        title_tag=f"Frame #{frame_id}/{title_ite_tag}"
        
        title_tag+="\nGT projection"
        axi.set_title(title_tag,fontsize=6)

        cimg=flatten_imgs[sample_id*len_frames+frame_id][:,:,::-1].copy() 
        axi.imshow(cimg) 
        
        gt_frame_id=frame_id
        cframe_gt_joints2d=project_hand_3d2img(ctrj_gt_cam[gt_frame_id]*1000,cam_info["intr"],cam_info["extr"])            
        gt_c=(0,1.,0)
        visualize_joints_2d(axi, cframe_gt_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        visualize_joints_2d(axi, cframe_gt_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[gt_c]*5)
        
        
        axi=plt.subplot2grid((num_rows,num_cols),(1,0),colspan=2)
        axi.imshow(cimg) 
        axi.axis("off")
        axi.set_title("P-Block L(cam) {:.2f},R(cam) {:.2f}".format(100*ctrj_err_cam1_L[frame_id],100*ctrj_err_cam1_R[frame_id]),fontsize=6)

    
        est_frame_id=frame_id
        c=(0.,1.,1) if est_frame_id%(16*2)<16 else(1.,0, 1.)#########
        cframe_est_joints2d=project_hand_3d2img(ctrj_est_cam1[est_frame_id]*1000,cam_info["intr"],cam_info["extr"])
        visualize_joints_2d(axi, cframe_est_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)
        visualize_joints_2d(axi, cframe_est_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)
        
        axi=plt.subplot2grid((num_rows,num_cols),(2,0),colspan=2)
        axi.imshow(cimg) 
        axi.axis("off")
        axi.set_title("Resnet  L(cam) {:.2f},R(cam) {:.2f}".format(100*ctrj_err_cam2_L[frame_id],100*ctrj_err_cam2_R[frame_id]),fontsize=6)
    
        c=(1.,1.,0.)
        cframe_est_joints2d=project_hand_3d2img(ctrj_est_cam2[est_frame_id]*1000,cam_info["intr"],cam_info["extr"])
        visualize_joints_2d(axi, cframe_est_joints2d[:21],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)
        visualize_joints_2d(axi, cframe_est_joints2d[21:],links=joint_links, alpha=1,linewidth=.7,scatter=False, joint_idxs=False,color=[c]*5)

        subplot_frame_gt_est_onehand(ctrj_gt_local,ctrj_est_local1,gt_frame_id,est_frame_id,8,'L',True,joint_links, num_rows=num_rows,num_cols=num_cols,ctrj_est2=ctrj_est_local2)
        subplot_frame_gt_est_onehand(ctrj_gt_local,ctrj_est_local1,gt_frame_id,est_frame_id,7,'Trj Est(blue)/GT(green)-R',False,joint_links, num_rows=num_rows,num_cols=num_cols,ctrj_est2=ctrj_est_local2)
        
        #subplot_trj(ctrj_gt_cam,ctrj_est_cam,gt_frame_id,est_frame_id,8,'L',True,joint_links, num_rows=num_rows,num_cols=num_cols)
        #subplot_trj(ctrj_gt_cam,ctrj_est_cam,gt_frame_id,est_frame_id,7,'Trj Est(blue)/GT(green)-R',False,joint_links, num_rows=num_rows,num_cols=num_cols)
        
        fig.savefig(f"{prefix_cache_img}{frame_id}.png", dpi=200)
        plt.close(fig)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')     
    cimg=cv2.imread(f"{prefix_cache_img}0.png")
    #print('write to',save_img_prefix+'_video.avi')
    videoWriter = cv2.VideoWriter(path_video, fourcc, 10, (cimg.shape[1],cimg.shape[0]))  ##########

    for frame_id in range(0,len_frames):
        cimg=cv2.imread(f"{prefix_cache_img}{frame_id}.png")
        videoWriter.write(cimg)
    
    videoWriter.release()



def overlay_segs_on_image_cv2(image,hand_pose,joint_links,color):
    for l in joint_links:
        for iid in range(0,len(l)-1):
            start,end=l[iid],l[iid+1]
            cv2.line(image, tuple(hand_pose[start].astype(np.int32)), tuple(hand_pose[end].astype(np.int32)), color, 2)
    return image



def plot_on_image_opencv(batch_seq_gt_cam, batch_seq_est_cam, joint_links, prefix_cache_img, sample_id, cam_info, flatten_imgs,
                            batch_seq_gt_local=None, batch_seq_est_local=None, batch_seq_valid_frames_obsv=None,batch_seq_valid_frames_pred=None):

    batch_seq_gt_cam=torch2numpy(batch_seq_gt_cam)
    batch_seq_est_cam=torch2numpy(batch_seq_est_cam)
    if batch_seq_gt_local is not None:
        batch_seq_gt_local=torch2numpy(batch_seq_gt_local)
        batch_seq_est_local=torch2numpy(batch_seq_est_local)
    if batch_seq_valid_frames_obsv is not None:
        batch_seq_valid_frames_obsv=torch2numpy(batch_seq_valid_frames_obsv)
        batch_seq_valid_frames_pred=torch2numpy(batch_seq_valid_frames_pred)
    flatten_imgs=torch2numpy(flatten_imgs)


    batch_size, len_frames = batch_seq_gt_cam.shape[0],batch_seq_gt_cam.shape[1]
    ctrj_gt_cam,ctrj_est_cam= batch_seq_gt_cam[sample_id],batch_seq_est_cam[sample_id]
    if batch_seq_gt_local is not None:
        ctrj_gt_local,ctrj_est_local=batch_seq_gt_local[sample_id],batch_seq_est_local[sample_id]
    
    dir_cache=os.path.dirname(os.path.join(prefix_cache_img,"{:02d}".format(sample_id),"0.png"))
    if not os.path.exists(dir_cache):
        os.makedirs(dir_cache)

    

    if batch_seq_valid_frames_obsv is None:
        len_pred_actual=batch_seq_est_cam.shape[1]
        len_obsv=len_frames-len_pred_actual
        len_obsv_actual=len_obsv
    else:
        len_pred_actual=np.sum(batch_seq_valid_frames_pred[sample_id])
        len_obsv_actual=np.sum(batch_seq_valid_frames_obsv[sample_id])
        len_obsv=batch_seq_valid_frames_obsv.shape[1]
        
    output_frame_id=0
    cimg=flatten_imgs[sample_id*len_frames+15].copy()
    color_bar=np.zeros((66*5,20,3),dtype=np.uint8)+255

    len_pred_actual=64
    
    cmap_est=cm.get_cmap('cool')        
    for frame_id in range(len_obsv,len_obsv+len_pred_actual):#(0,len_obsv_actual): 
        if (frame_id>=len_obsv_actual and frame_id<len_obsv) or frame_id>=len_obsv+len_pred_actual:
            continue
            
        #cimg=flatten_imgs[sample_id*len_frames+frame_id].copy() 
        

        
        gt_frame_id=frame_id#+batch_seq_gt_local.shape[1]-len_frames
        cframe_gt_joints2d=project_hand_3d2img(ctrj_gt_cam[gt_frame_id]*1000,cam_info["intr"],cam_info["extr"])            
        err_str=""        
        est_frame_id=frame_id+batch_seq_est_cam.shape[1]-len_frames
        if est_frame_id>=0:
            err=np.linalg.norm(ctrj_est_cam[est_frame_id]-ctrj_gt_cam[gt_frame_id],axis=-1)
            vis_left=[0]+list(range(2,21))
            vis_right=[21]+list(range(23,42))
            err_str="_L{:.2f}_R{:.2f}".format(err[vis_left].mean()*1000,err[vis_right].mean()*1000)
            err_str=err_str.replace(".","_")
        
        


        if est_frame_id>=0:
            cframe_est_joints2d=project_hand_3d2img(ctrj_est_cam[est_frame_id]*1000,cam_info["intr"],cam_info["extr"])
            color_=cmap_est(est_frame_id/66)#(255,0,0)
            color=(int(color_[2]*255),int(color_[1]*255),int(color_[0]*255))
            color_bar[est_frame_id*5:(est_frame_id+1)*5,:]=color
            if est_frame_id==0:
                pad=np.zeros_like(cimg)+255
                cimg=cv2.addWeighted(cimg, 0.5, pad, 0.5, 0)
            if est_frame_id%4==0:
                cimg=overlay_segs_on_image_cv2(cimg,cframe_est_joints2d[:21],joint_links,color)
                cimg=overlay_segs_on_image_cv2(cimg,cframe_est_joints2d[21:],joint_links,color)
            output_frame_id+=1
        else:
            color=(0,255,0)
            cimg=overlay_segs_on_image_cv2(cimg,cframe_gt_joints2d[:21],joint_links,color)
            cimg=overlay_segs_on_image_cv2(cimg,cframe_gt_joints2d[21:],joint_links,color)
            output_frame_id+=1

    cimg=cimg[40:40+150,240:240+230].copy()
    cv2.imwrite(os.path.join(prefix_cache_img,"{:02d}".format(sample_id),"{:02d}{:s}.png".format(output_frame_id,err_str)),cimg)
    #cv2.imwrite("color_bar.png",color_bar)

    exit(0)
        