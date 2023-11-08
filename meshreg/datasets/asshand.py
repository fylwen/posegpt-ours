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
from meshreg.datasets import ass101utils, asshandutils
from meshreg.datasets.ass101 import Ass101
from meshreg.models.utils import project_hand_3d2img
from meshreg.datasets.queries import BaseQueries

class AssHand(Ass101):
    def __init__(
        self,
        dataset_folder,
        split,#="train",
        view_id,
        ntokens_per_seq,
        spacing,#=1,
        mode,#="enc",
        is_shifting_window,#=True,
        action_taxonomy_to_use,
        max_samples,
        buf_sec,
        min_window_sec,
        shift_as_htt,
    ):
        super().__init__(dataset_folder=dataset_folder,
                        split=split,
                        view_id=view_id,
                        ntokens_per_seq=ntokens_per_seq,
                        spacing=spacing,
                        mode=mode,
                        is_shifting_window=is_shifting_window,
                        action_taxonomy_to_use=action_taxonomy_to_use,
                        max_samples=max_samples,
                        buf_sec=buf_sec,
                        min_window_sec=min_window_sec,
                        shift_as_htt=shift_as_htt)
        
        
        
        self.name = "asshand" 
        self.root_pose_annotation=os.path.join(dataset_folder,'ass101/asshand')
        
    def load_dataset(self,verbose=False):
        #load action
        videos_coarse_action_info,videos_fine_grained_action_info=self.load_split_action_info(verbose=verbose)

        #Load pose len@fps60        
        splits_to_load=["train","val","test"]
        loaded_split_pose_meta=asshandutils.load_dataset_video_pose_infos(self.root_pose_annotation,splits_to_load)

        if self.view_id is not None and self.view_id>3:
            loaded_split_pose_meta=asshandutils.remove_frames_without_exo_imgs(loaded_split_pose_meta,self.root_pose_annotation,self.view_tag)

        #need to modify here for more segments
        annotations=asshandutils.gather_video_meta_for_untrimmed_pose_seq(task_seqs_info=videos_fine_grained_action_info,
                                                                        fps_to_load=self.fps,
                                                                        pose_meta=loaded_split_pose_meta, 
                                                                        max_samples=self.max_samples) 
                                                                        
        self.video_infos=annotations["untrimmed_video_info"] 
        
        self.env_action=lmdb.open(os.path.join('../lmdbsv2/asshand_action@30fps'),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
                                
        if BaseQueries.JOINTS3D in self.all_queries:
            assert BaseQueries.RESNET_JOINTS3D not in self.all_queries
            self.env_pose=lmdb.open(os.path.join('../lmdbsv2/asshand_posev3@60fps'),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        else:
            assert self.view_id is not None and BaseQueries.RESNET_JOINTS3D in self.all_queries
            self.env_pose=lmdb.open(os.path.join(f'../lmdbsv2/asshand_resnet@30fps/viewid_{self.view_id}'),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)

        
        if self.view_id is not None and self.view_id<4:
            self.env_img_path=os.path.join('../lmdbsv2/asshand_egoimgs@60fps')
        else:
            self.env_img_path=os.path.join('../lmdbsv2/asshand_exoimgs@60fps')
        self.env_img=lmdb.open(self.env_img_path,readonly=True,lock=False,readahead=False,meminit=False,\
                            map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        
        
        try:
            self.env_midpe=lmdb.open(os.path.join(f'../lmdbsv2/asshand_midpe_world16x1@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        except:
            self.env_midpe=None
        #compute window_info
        self.compute_window_info(videos_coarse_action_info=videos_coarse_action_info,
                                videos_fine_grained_action_info=videos_fine_grained_action_info,
                                verbose=verbose)
        
        '''
        asshandutils.visualize_per_frame_label_for_untrimmed_videos(untrimmed_video_infos=self.video_infos,
                                                                    env_pose=self.env_pose, env_action=self.env_action, env_img=self.env_img,
                                                                    hand_links=self.links,fps_to_load=self.fps, 
                                                                    view_id=self.view_id,view_tag=self.view_tag,
                                                                    dir_out_videos=f'./vis_{self.view_tag}/')
                                                                    
        exit(0)
        '''
        
    def get_joints3d_cam2local_from_lmdb(self, txn, sample_info):
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)+sample_info["mod"]

        return self.__get_joints3d_cam2local_from_lmdb__(txn,frame_id,self.view_id)

    
    def get_image(self, txn, sample_info):
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)+sample_info["mod"]

        return self.__get_image__(txn,frame_id,self.view_id)
    

    
    def get_cam_extr_intr(self, txn, sample_info):
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)+sample_info["mod"]

        return self.__get_cam_extr_intr__(txn,frame_id,self.view_id)


    
    def get_image(self,txn, sample_info):
        if "ass101" in self.env_img_path:
            return super().get_image(txn,sample_info)
            
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)+sample_info["mod"]

        try:
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))        
            raw_data=np.frombuffer(buf,dtype=np.uint8)
            img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            has_img=True
        except:
            img=np.zeros((270,357,3),dtype=np.uint8)+255
            has_img=False
        
        return img, has_img
    
    
    