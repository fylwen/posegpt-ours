import os
import pickle
import random
import tqdm
import lmdb
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFile
import cv2, math
import torch

import sys
import copy
sys.path.append('../../')

from meshreg.datasets.queries import BaseQueries


class VanillaDataset(object):
    def __init__(self):
        super().__init__()

        self.fps=30
        self.all_queries=[BaseQueries.JOINTS3D,
                          BaseQueries.ACTIONNAME,BaseQueries.OBJNAME]
        # get paired links as neighboured joints
        self.links = [
            (0, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
        
        self.palm_joints=[0,5,9,13,17] 
        self.valid_joints=[0]+list(range(2,21))#list(range(0,21))#
        
    
    
    def get_window_info(self,idx):
        idx=min(idx,len(self.window_info)-1)
        window_info = self.window_info[idx]
        window_info.update(self.video_infos[window_info["video_tag"]])
        if not self.view_tag is None:
            window_info["view_tag"]=self.view_tag

        if self.list_meta_action!=None:
            caction_meta=self.list_meta_action[window_info["meta_idx"]]
            window_info["action_name"]=caction_meta["action_name"]
            window_info["action_start_frame"]=caction_meta["start_frame"]
            window_info["action_end_frame"]=caction_meta["end_frame"]

            assert window_info["video_tag"]==caction_meta["video_tag"], "get_window_info video_tag consistency?"
        return window_info
        

    def open_seq_lmdb(self,window_info, load_pose=True, load_action=True,load_image=False):
        file_tag=window_info["video_tag"]
        #print(file_tag)
        #read pose@60fps
        txn_pose,txn_action,txn_midpe,txn_img=None,None,None,None
        if load_pose:
            subdb_pose=self.env_pose.open_db(file_tag.encode('ascii'),create=False)
            txn_pose=self.env_pose.begin(write=False,db=subdb_pose)
        
        #read action@30fps
        if load_action:
            subdb_action=self.env_action.open_db(file_tag.encode('ascii'),create=False)
            txn_action=self.env_action.begin(write=False,db=subdb_action)


        if self.env_midpe!=None:
            subdb_midpe=self.env_midpe.open_db(file_tag.encode('ascii'),create=False)
            txn_midpe=self.env_midpe.begin(write=False,db=subdb_midpe)
            assert txn_midpe is not None
            
        #read imgs@60fps
        if load_image:
            view_tag=window_info["view_tag"]
            subdb_img = self.env_img.open_db((file_tag+"_"+view_tag).encode('ascii'),create=False)
            txn_img = self.env_img.begin(db=subdb_img,write=False)
        return {"pose":txn_pose,"action":txn_action,"image":txn_img,"midpe":txn_midpe}
    

    def __get_cam_extr_intr__(self,txn,frame_id,view_id):
        if view_id is None:
            cam_extr=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.,]])
            cam_intr=np.array([[1265,0.,0.,],[0.,1265.,0.,],[0.,0.,1]])
            return {"cam_extr":cam_extr,"cam_intr":cam_intr}        
        buf=txn.get("{:06d}".format(frame_id).encode('ascii'))
        raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
        cam_raw=raw_pose[42*3+24:]
        cam_raw=cam_raw[view_id*(12+9):(view_id+1)*(12+9)]
        cam_extr,cam_intr=cam_raw[:12].reshape((3,4)),cam_raw[12:].reshape((3,3))

        cam_intr=cam_intr.copy()
        cam_intr[:2] = cam_intr[:2] * self.reduce_factor  
        return {"cam_extr":cam_extr,"cam_intr":cam_intr}
    
    def __get_joints3d_cam2local_from_lmdb__(self, txn, frame_id, view_id):
        buf=txn.get("{:06d}".format(frame_id).encode('ascii'))
        raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
        #world camera
        cam_joints3d_left=raw_pose[:21*3].reshape((21,3)).astype(np.float32)#in meter
        cam_joints3d_right=raw_pose[21*3:42*3].reshape((21,3)).astype(np.float32)#in meter
        R_cam2local_left,t_cam2local_left=raw_pose[42*3:42*3+9].reshape((3,3)),raw_pose[42*3+9:42*3+12].reshape((1,3))
        R_cam2local_right,t_cam2local_right=raw_pose[42*3+12:42*3+21].reshape((3,3)),raw_pose[42*3+21:42*3+24].reshape((1,3))
        
        valid_joints_left=np.zeros_like(cam_joints3d_left)
        for jid in self.valid_joints:
            valid_joints_left[jid]=1
        valid_joints_right=valid_joints_left.copy()

        cam_palm3d_left = cam_joints3d_left[self.palm_joints,:]
        cam_palm3d_right = cam_joints3d_right[self.palm_joints,:]
        left_size=np.mean(np.linalg.norm(cam_palm3d_left[1:]-cam_palm3d_left[0],ord=2,axis=1))
        right_size=np.mean(np.linalg.norm(cam_palm3d_right[1:]-cam_palm3d_right[0],ord=2,axis=1))

        is_world_camera=True        
        cam_extr=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.,]])
        cam_intr=np.array([[1265,0.,256.,],[0.,1265.,256.,],[0.,0.,1]])
        
        if view_id is not None:
            verbose=True
            #assert False,"for asshand need check by projecting to image plane!"
            world_joints=raw_pose[:42*3].reshape((42,3)).astype(np.float32)
            R_world2local_left,t_world2local_left = R_cam2local_left.copy(),t_cam2local_left.copy()
            R_world2local_right,t_world2local_right= R_cam2local_right.copy(),t_cam2local_right.copy()

            cam_raw=raw_pose[42*3+24:]
            cam_raw=cam_raw[view_id*(12+9):(view_id+1)*(12+9)]


            if not np.sum(np.fabs(cam_raw))<1e-6:
                is_world_camera=False                    
                cam_extr,cam_intr=cam_raw[:12].reshape((3,4)),cam_raw[12:].reshape((3,3))
                cam_intr=cam_intr.copy()
                cam_intr[:2] = cam_intr[:2] * self.reduce_factor  


                world_joints_hom = np.concatenate([world_joints, np.ones([world_joints.shape[0], 1])], 1)
                cam_joints=cam_extr.dot(world_joints_hom.transpose()).transpose()[:, :3].astype(np.float32)
                cam_joints3d_left,cam_joints3d_right=cam_joints[:21],cam_joints[21:]


                R_cam2world,t_world2cam=cam_extr[:,:3],cam_extr[:,3:4].reshape((1,3))#left multiply to right multiply
                t_cam2world=-np.matmul(t_world2cam,R_cam2world) 
                
                if verbose:
                    cam_joints2=np.dot(world_joints,R_cam2world.transpose())+t_world2cam
                    assert np.fabs(cam_joints2-cam_joints).max()<1e-5

                R_cam2local_left=np.matmul(R_cam2world,R_world2local_left)
                R_cam2local_right=np.matmul(R_cam2world,R_world2local_right)
                t_cam2local_left=np.matmul(t_cam2world,R_world2local_left)+t_world2local_left
                t_cam2local_right=np.matmul(t_cam2world,R_world2local_right)+t_world2local_right


                if verbose:
                    assert np.fabs(np.dot(world_joints[:21],R_world2local_left)+t_world2local_left-(np.dot(cam_joints3d_left,R_cam2local_left)+t_cam2local_left)).max()<1e-5
                    assert np.fabs(np.dot(world_joints[21:],R_world2local_right)+t_world2local_right-(np.dot(cam_joints3d_right,R_cam2local_right)+t_cam2local_right)).max()<1e-5
            
        dict_to_return={"cam_joints3d_left":cam_joints3d_left,"cam_joints3d_right":cam_joints3d_right,\
                "R_cam2local_left":R_cam2local_left,"t_cam2local_left":t_cam2local_left,\
                "R_cam2local_right":R_cam2local_right,"t_cam2local_right":t_cam2local_right,\
                "valid_joints_left":valid_joints_left.copy(),"valid_joints_right":valid_joints_right.copy(),\
                "hand_size_left":left_size,"hand_size_right":right_size,
                "is_world_camera":is_world_camera,"cam_extr":cam_extr,"cam_intr":cam_intr}

        return dict_to_return

    
    def get_resnet_output_from_lmdb(self, txn, frame_id, cam_intr):
        buf=txn.get("{:d}".format(frame_id).encode('ascii'))
        raw_data=np.frombuffer(buf,dtype=np.float32).reshape(-1,)
        #world camera

        dev_pt=[0, 512, 512*2, 512*2+(126+24), 512*2+(126+24)*2, 512*2+(126+24)*2+42*3, 512*2+(126+24)*2+42*3*2]
        raw_img_feature=raw_data[:dev_pt[1]]
        raw_obj_feature=raw_data[dev_pt[1]:dev_pt[2]]

        raw_pose_gt=raw_data[dev_pt[2]:dev_pt[3]]
        raw_pose_pred=raw_data[dev_pt[3]:dev_pt[4]]

        gt_25d=raw_data[dev_pt[4]:dev_pt[5]].reshape((42,3))
        pred_25d=raw_data[dev_pt[5]:dev_pt[6]].reshape((42,3))

        ncam_joints3d_left_gt=raw_pose_gt[:21*3].reshape((21,3)).astype(np.float32)#in meter
        ncam_joitns3d_right_gt=raw_pose_gt[21*3:42*3].reshape((21,3)).astype(np.float32)#in meter
        R_cam2local_left_gt,t_cam2local_left_gt=raw_pose_gt[42*3:42*3+9].reshape((3,3)),raw_pose_gt[42*3+9:42*3+12].reshape((1,3))
        R_cam2local_right_gt,t_cam2local_right_gt=raw_pose_gt[42*3+12:42*3+21].reshape((3,3)),raw_pose_gt[42*3+21:42*3+24].reshape((1,3))

        ncam_joints3d_left_pred=raw_pose_pred[:21*3].reshape((21,3)).astype(np.float32)#in meter
        ncam_joitns3d_right_pred=raw_pose_pred[21*3:42*3].reshape((21,3)).astype(np.float32)#in meter
        R_cam2local_left_pred,t_cam2local_left_pred=raw_pose_pred[42*3:42*3+9].reshape((3,3)),raw_pose_pred[42*3+9:42*3+12].reshape((1,3))
        R_cam2local_right_pred,t_cam2local_right_pred=raw_pose_pred[42*3+12:42*3+21].reshape((3,3)),raw_pose_pred[42*3+21:42*3+24].reshape((1,3))
        
        
        valid_joints_left=np.zeros_like(ncam_joints3d_left_pred)
        for jid in self.valid_joints:
            valid_joints_left[jid]=1
        valid_joints_right=valid_joints_left.copy()
        
        ncam_palm3d_left_gt = ncam_joints3d_left_gt[self.palm_joints,:]
        ncam_palm3d_right_gt = ncam_joitns3d_right_gt[self.palm_joints,:]

        left_size_gt=np.mean(np.linalg.norm(ncam_palm3d_left_gt[1:]-ncam_palm3d_left_gt[0],ord=2,axis=1))
        right_size_gt=np.mean(np.linalg.norm(ncam_palm3d_right_gt[1:]-ncam_palm3d_right_gt[0],ord=2,axis=1))

        ncam_palm3d_left_pred = ncam_joints3d_left_pred[self.palm_joints,:]
        ncam_palm3d_right_pred = ncam_joitns3d_right_pred[self.palm_joints,:]

        left_size_pred=np.mean(np.linalg.norm(ncam_palm3d_left_pred[1:]-ncam_palm3d_left_pred[0],ord=2,axis=1))
        right_size_pred=np.mean(np.linalg.norm(ncam_palm3d_right_pred[1:]-ncam_palm3d_right_pred[0],ord=2,axis=1))
        
        dict_to_return={"cam_joints3d_left":ncam_joints3d_left_gt,"cam_joints3d_right":ncam_joitns3d_right_gt, "joints25d":gt_25d,\
                "R_cam2local_left":R_cam2local_left_gt,"t_cam2local_left":t_cam2local_left_gt,\
                "R_cam2local_right":R_cam2local_right_gt,"t_cam2local_right":t_cam2local_right_gt,\
                "hand_size_left":left_size_gt,"hand_size_right":right_size_gt,\
                "valid_joints_left":valid_joints_left.copy(),"valid_joints_right":valid_joints_right.copy(),\
                "ncam_extr":np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.,]]),"ncam_intr":cam_intr,"is_world_camera":False,\
                
                "cam_joints3d_left_resnet":ncam_joints3d_left_pred,"cam_joints3d_right_resnet":ncam_joitns3d_right_pred,"joints25d_resnet":pred_25d,\
                "R_cam2local_left_resnet":R_cam2local_left_pred,"t_cam2local_left_resnet":t_cam2local_left_pred,\
                "R_cam2local_right_resnet":R_cam2local_right_pred,"t_cam2local_right_resnet":t_cam2local_right_pred,
                "hand_size_left_resnet":left_size_pred,"hand_size_right_resnet":right_size_pred,
                "image_feature":raw_img_feature,
                "object_feature":raw_obj_feature,}

        return dict_to_return
    


    def __get_image__(self,txn,frame_id):        
        try:
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))        
            raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
            img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            has_img=True
        except:
            assert False
            print(frame_id,sample_info)
            
            buf=txn.get("{:06d}".format(0).encode('ascii'))        
            raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
            img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

            has_img=False
        
        return img, has_img


    def get_image(self, txn, sample_info):
        frame_id=sample_info["frame_idx"]        
        if self.fps<=30:
            frame_id=frame_id*(30//self.fps)
        else:
            frame_id=frame_id//(self.fps//30)
        #img_path=os.path.join("../h2o/",sample_info["video_tag"],sample_info["view_tag"],"rgb480_270","{:06d}.png".format(frame_id))
        #img = Image.open(img_path).convert("RGB")
        #has_img=True
        return self.__get_image__(txn,frame_id)
        
    
    def get_midpe_from_lmdb(self, txn, assigned_action_name, sample_info):
        frame_id=sample_info["frame_idx"]
        if self.fps<=30:
            frame_id=frame_id*(30//self.fps)
        else:
            frame_id=frame_id//(self.fps//30)
        
        if assigned_action_name is None:
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))
        else:
            buf=txn.get("{:06d},{:s}".format(frame_id,assigned_action_name).encode('ascii'))

        try:
            midpe=np.frombuffer(buf,dtype=np.float32).reshape(-1,)
        except:
            print(frame_id,assigned_action_name)
            print(sample_info)
            assert False

        return midpe
    
    def get_action_idxs(self, sample_info):
        assert not isinstance(sample_info["action_name"],list)
        aname=sample_info["action_name"]
        return -1 if aname=="NIL" else  self.action_to_idx[aname]      

    def get_obj_idxs(self,sample_info):
        assert not isinstance(sample_info["object_name"],list)
        oname=sample_info["object_name"]
        return -1 if oname=="NIL" else self.object_to_idx[oname]  

    def get_verb_idxs(self,sample_info):
        assert not isinstance(sample_info["verb_name"],list)
        vname=sample_info["verb_name"]
        return -1 if vname=="NIL" else self.verb_to_idx[vname] 

    
    
def get_window_starts_untrimmed_videos(untrimmed_video_infos,ntokens, spacing, min_window_len, is_shifting_window):
    window_start_info=[]
    
    for vfile_tag,vfile_info in untrimmed_video_infos.items(): 
        video_start_idx, video_end_idx=untrimmed_video_infos[vfile_tag]["start_idx"],untrimmed_video_infos[vfile_tag]["end_idx"]
        cv_len=video_end_idx-video_start_idx
        
        for sample_idx in range(cv_len):
            if ((not is_shifting_window) or sample_idx%(ntokens*spacing)<spacing) and sample_idx+video_start_idx+min_window_len<=video_end_idx:
                window_start_info.append({'frame_idx': sample_idx+video_start_idx,'video_tag': vfile_tag})
    
    return window_start_info
    

def get_window_starts_trimmed_videos(trimmed_seg_infos, untrimmed_video_infos, trimmed_type, list_meta_action, ntokens, spacing, min_window_len, buf_len, is_shifting_window, shift_as_htt):    
    window_start_info=[]
    for k, v in trimmed_seg_infos.items():
        vfile_tag,vsegs=k,v
        if trimmed_type=="h2o":
            vsegs=v["action_segs"]
        if trimmed_type=="ass_coarse":
            vfile_tag="_".join(k.split("_")[1:])
            
            
        if vfile_tag not in untrimmed_video_infos:
            continue
        
        video_start_idx, video_end_idx=untrimmed_video_infos[vfile_tag]["start_idx"],untrimmed_video_infos[vfile_tag]["end_idx"]

        for cvseg in vsegs:
            #process action segments meta; For convenience, suppose end_frame not included in the performed action
            cvseg_meta = {"video_tag":vfile_tag,"action_name":cvseg["action_name"],
                "start_frame":max(video_start_idx,cvseg["start_frame"]-buf_len),"end_frame":min(video_end_idx,cvseg["end_frame"])}

            cvseg_len=cvseg_meta["end_frame"]-cvseg_meta["start_frame"]

            if cvseg_len<max(2*spacing,min_window_len+buf_len):
                continue
                
            #Add meta info
            meta_idx=len(list_meta_action)
            list_meta_action.append(cvseg_meta)
            
            #add samples; ensure sample_idx in the current trimmed action segment.  
            for idx_in_cvseg in range(0,cvseg_len,1):
                idx_in_video=idx_in_cvseg+cvseg_meta["start_frame"]
                
                if (not shift_as_htt and (not is_shifting_window or idx_in_cvseg%(ntokens*spacing)<spacing) and idx_in_cvseg+min_window_len<=cvseg_len) or \
                (shift_as_htt and ((not is_shifting_window and idx_in_cvseg%256<16*spacing) or (is_shifting_window and idx_in_cvseg%256<spacing))):
                #if ((not is_shifting_window and idx_in_cvseg<16) or (is_shifting_window and idx_in_cvseg==0)) and idx_in_cvseg+min_window_len<=cvseg_len:
                    window_start_info.append({'frame_idx': idx_in_video, 'video_tag':vfile_tag, 'meta_idx':meta_idx, 'action_is_coarse':"coarse" in trimmed_type})

    return window_start_info
