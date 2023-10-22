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
from meshreg.datasets import ass101utils
from meshreg.datasets.vanilla_dataset import VanillaDataset,get_window_starts_untrimmed_videos,get_window_starts_trimmed_videos
from meshreg.models.utils import project_hand_3d2img


from meshreg.datasets.queries import BaseQueries

class Ass101(VanillaDataset):
    def __init__(
        self,
        dataset_folder,
        split,
        view_id,
        ntokens_per_seq,
        spacing,
        mode,
        is_shifting_window,
        action_taxonomy_to_use,
        max_samples,
        buf_sec,
        min_window_sec,
        shift_as_htt,
    ):
    
        super().__init__()
        self.mode = mode
        self.ntokens_per_seq=ntokens_per_seq

        self.action_taxonomy_to_use=action_taxonomy_to_use        
        assert action_taxonomy_to_use in ["coarse","fine","coarse+fine"]

        assert self.fps==30
        self.spacing=spacing*(self.fps//30)#to make it consistent with a 30fps video
        assert buf_sec==0,"if buf_sec!=0, need to change action name assignment in seqset"
        self.buf_len=int(buf_sec*self.fps)#self.fps//2 if buf_len is None else buf_len
        self.min_window_len= int(min_window_sec*self.fps)
        
        
        self.name = "ass101" 
        self.split = split
        self.root = os.path.join(dataset_folder,"ass101")
        
        self.is_shifting_window=is_shifting_window
        self.shift_as_htt=shift_as_htt

        self.max_samples=max_samples

        self.view_lists=["e1","e2","e3","e4","v1","v2","v3","v4","v5","v6","v7","v8"]
        self.view_id= None if view_id<0 else view_id
        self.view_tag="v3" if self.view_id is None else self.view_lists[self.view_id]
        self.reduce_factor = 270/480. if self.view_id is not None and self.view_id<4 else 480/1920.
        #self.cam_for_view_tag=ass101utils.get_view_extrinsic_intrisic(path_calib_txt=os.path.join(self.root,'./annotations/calib.txt'))

        
        #Load action labels
        self.coarse_action_taxonomy=ass101utils.load_action_taxonomy(path_tax=os.path.join(self.root, './annotations/coarse-annotations/actions.csv'),
                                                                path_tail_tax=os.path.join(self.root,'./annotations/coarse-annotations/tail_actions.txt'),
                                                                reverse_head_tail=False)
        self.dir_fine_grained_split_files=os.path.join(self.root,'./annotations/fine-grained-annotations/') if self.mode in ['motion','synimg'] \
                                                            else os.path.join(self.root, './annotations/anticipation-annotations/CSVs/')
        self.fine_grained_action_taxonomy=ass101utils.load_action_taxonomy(path_tax=os.path.join(self.dir_fine_grained_split_files,'actions.csv'),
                                                        path_tail_tax=os.path.join(self.root,'./annotations/fine-grained-annotations/head_actions.txt'),
                                                        reverse_head_tail=True) 
        #By examine found coarse action taxonomy and fine grained action taxonomy are with overlap!

        if self.action_taxonomy_to_use == "coarse":
            self.action_to_idx=copy.deepcopy(self.coarse_action_taxonomy["action_name2idx"])
            self.object_to_idx=copy.deepcopy(self.coarse_action_taxonomy["object_name2idx"])
            self.verb_to_idx=copy.deepcopy(self.coarse_action_taxonomy["verb_name2idx"])
            self.action_info=copy.deepcopy(self.coarse_action_taxonomy["action_info"])
        elif self.action_taxonomy_to_use =="fine":
            self.action_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["action_name2idx"])
            self.object_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["object_name2idx"])
            self.verb_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["verb_name2idx"])
            self.action_info=copy.deepcopy(self.fine_grained_action_taxonomy["action_info"])
        else:
            self.action_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["action_name2idx"])
            self.object_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["object_name2idx"])
            self.verb_to_idx=copy.deepcopy(self.fine_grained_action_taxonomy["verb_name2idx"])
            self.action_info=copy.deepcopy(self.fine_grained_action_taxonomy["action_info"])

            for coarse_name in self.coarse_action_taxonomy["action_name2idx"].keys():
                if coarse_name in self.action_to_idx:
                    continue
                self.action_to_idx[coarse_name]=len(self.action_to_idx)
                self.action_info[coarse_name]=self.coarse_action_taxonomy["action_info"][coarse_name]
            
            for coarse_name in self.coarse_action_taxonomy["verb_name2idx"].keys():
                if coarse_name in self.verb_to_idx:
                    continue
                self.verb_to_idx[coarse_name]=len(self.verb_to_idx)
            
            for coarse_name in self.coarse_action_taxonomy["object_name2idx"].keys():
                if coarse_name in self.object_to_idx:
                    continue
                self.object_to_idx[coarse_name]=len(self.object_to_idx)
                
        self.num_actions = len(self.action_to_idx)
        self.num_objects = len(self.object_to_idx)
        self.num_verbs = len(self.verb_to_idx)

        #self.vis_action_tax=self.fine_grained_action_taxonomy["action_name2idx"].keys()
        #self.list_head_actions=self.fine_grained_action_taxonomy["list_head_tax"]
        #self.list_head_actions.sort()
        
        #check order
        for i,(k,v) in enumerate(self.action_to_idx.items()):
            assert self.action_to_idx[k]==i,'action #{:d}: {:s} {:d}'.format(i,k,v)        
        for i,(k,v) in enumerate(self.object_to_idx.items()):
            assert self.object_to_idx[k]==i,'object #{:d}: {:s} {:d}'.format(i,k,v)        
        for i,(k,v) in enumerate(self.verb_to_idx.items()):
            assert self.verb_to_idx[k]==i,'verb #{:d}: {:s} {:d}'.format(i,k,v)
        
        
    def load_dataset(self,verbose=False):
        #load action
        videos_coarse_action_info,videos_fine_grained_action_info=self.load_split_action_info(verbose=verbose)

        #Load pose len@fps60        
        if not os.path.exists(os.path.join(self.root,'./poses@60fps/',f"{self.split}.txt")):
            assert False
            ass101utils.save_split_num_pose_frames_at_60fps(videos_fine_grained_action_info,os.path.join(self.root,'./poses@60fps/'),self.split)

        annotations=ass101utils.gather_video_meta_for_untrimmed_pose_seq(task_seqs_info=videos_fine_grained_action_info,
                                                                        fps_to_load=self.fps,
                                                                        path_split_num_pose_frames_at_60fps=os.path.join(self.root,'./poses@60fps/',f"{self.split}.txt"), 
                                                                        max_samples=self.max_samples)

        self.video_infos=annotations["untrimmed_video_info"] 
        
        self.env_action=lmdb.open(os.path.join('../lmdbsv2/ass101_action@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                    map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
                                    
        self.env_pose=lmdb.open(os.path.join('../lmdbsv2/ass101_pose@60fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)

        self.env_img= lmdb.open(os.path.join('../lmdbsv2/ass101_v3_imgs@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        
    
        try:
            self.env_midpe=lmdb.open(os.path.join(f'../lmdbsv2/ass101_midpe_world16x1@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                            map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        except:
            self.env_midpe=None
            
        

        #compute window_info
        self.compute_window_info(videos_coarse_action_info=videos_coarse_action_info,
                                videos_fine_grained_action_info=videos_fine_grained_action_info,
                                verbose=verbose)
        
        #ass101utils.visualize_per_frame_label_for_untrimmed_videos(untrimmed_video_infos=self.video_infos, 
        #                                                                env_pose=self.env_pose, 
        #                                                                env_action=self.env_action,
        #                                                                env_img=self.env_img, 
        #                                                                hand_links=self.links,
        #                                                                fps_to_load=self.fps,
        #                                                                dir_out_videos='./vis_v3/')
        #exit(0)        


    def load_split_action_info(self,verbose=False):
        videos_coarse_action_info=ass101utils.load_split_coarse_info(dir_split_files=os.path.join(self.root,'./annotations/coarse-annotations/coarse_splits'),
                                                            dir_coarse_labels=os.path.join(self.root,'./annotations/coarse-annotations/coarse_labels/'),
                                                            fps_to_load=self.fps,split=self.split, verbose=verbose)
                                                            
        videos_fine_grained_action_info=ass101utils.load_split_fine_grained_info(dir_split_files=self.dir_fine_grained_split_files,
                                                            view_id=None if self.view_id is None else self.view_lists[self.view_id],#note not using view_tag directly
                                                            split=self.split, fps_to_load=self.fps,
                                                            verbose=verbose,dict_action_info=self.fine_grained_action_taxonomy)


        return videos_coarse_action_info,videos_fine_grained_action_info

    def compute_window_info(self,videos_coarse_action_info,videos_fine_grained_action_info,verbose=False):        
        verbose=True
        if self.mode in ["motion"]:
            self.window_info=get_window_starts_untrimmed_videos(untrimmed_video_infos=self.video_infos,
                                                ntokens=self.ntokens_per_seq,#15, 
                                                spacing=self.spacing,#2, 
                                                min_window_len=self.min_window_len,
                                                is_shifting_window=self.is_shifting_window)
            self.list_meta_action=None
        elif self.mode in ["context"]:
            list_meta_action=[]
            window_info,window_info2=[],[]
            if "coarse" in self.action_taxonomy_to_use:
                window_info=get_window_starts_trimmed_videos(trimmed_seg_infos=videos_coarse_action_info,
                                                    untrimmed_video_infos=self.video_infos,
                                                    trimmed_type="ass_coarse",
                                                    list_meta_action=list_meta_action,
                                                    ntokens=self.ntokens_per_seq,
                                                    spacing=self.spacing,
                                                    min_window_len=self.min_window_len,
                                                    buf_len=self.buf_len,
                                                    is_shifting_window=self.is_shifting_window,
                                                    shift_as_htt=self.shift_as_htt)
                if verbose:
                    print("Coarse- check list_meta_action/window_info",len(list_meta_action),len(window_info))
                    print(list_meta_action[-1])
                    print(window_info[-1])
            if "fine" in self.action_taxonomy_to_use:                
                window_info2=get_window_starts_trimmed_videos(trimmed_seg_infos=videos_fine_grained_action_info,
                                                    untrimmed_video_infos=self.video_infos,
                                                    trimmed_type="ass_fine",
                                                    list_meta_action=list_meta_action,
                                                    ntokens=self.ntokens_per_seq,
                                                    spacing=self.spacing,
                                                    min_window_len=self.min_window_len, 
                                                    buf_len=self.buf_len,
                                                    is_shifting_window=self.is_shifting_window,
                                                    shift_as_htt=self.shift_as_htt)
                if verbose:
                    print("Fine Grained- check list_meta_action/window_info",len(list_meta_action),len(window_info2))
                    print(list_meta_action[-1])
                    print(window_info2[-1])

            self.list_meta_action=list_meta_action
            self.window_info=window_info+window_info2

            if verbose:            
                print("check list_meta_action/window_info",len(self.list_meta_action),len(self.window_info))
                print(self.list_meta_action[-1])
                print(self.window_info[-1])
        else:
            assert False
            

    def get_joints3d_cam2local_from_lmdb(self, txn, sample_info):        
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)        
        return self.__get_joints3d_cam2local_from_lmdb__(txn,frame_id,view_id=self.view_id)

    
    def get_cam_extr_intr(self, txn, sample_info):        
        #actually use the world camera
        frame_id=sample_info["frame_idx"]
        frame_id=frame_id*(60//self.fps)        
        return self.__get_cam_extr_intr__(txn,frame_id,view_id=self.view_id)

    def get_sample_info(self, window_info, frame_id, txn=None):
        # sample_infos, and read action names
        sample_info={}
        for k,v in window_info.items():
            if "action" not in k and "verb" not in k and "object" not in k:
                sample_info[k]=v
        if txn is None:
            return sample_info
        sample_info["frame_idx"]=frame_id

        #action info
        if self.fps<=30:
            frame_id=frame_id*(30//self.fps)
        else:
            frame_id=frame_id//(self.fps//30)  
            
        buf=txn.get("{:06d}".format(frame_id).encode('ascii')).decode('ascii')
            
        terms=buf.split('*')
        keys=["coarse_action_name","coarse_verb_name","coarse_object_name",\
            "fine_grained_action_name", "fine_grained_verb_name","fine_grained_object_name",\
            "anticipate_action_name", "anticipate_verb_name","anticipate_object_name"]
        
        sample_info_={}
        for i in range(0,3):
            sample_info_[keys[i]]=terms[i]
        for i in range(3,9):
            sample_info_[keys[i]]=terms[i].split('+')#[] if terms[i]=='NIL' else 
        
        if self.action_taxonomy_to_use=="coarse":            
            for key in ["action","object","verb"]:
                sample_info[f"{key}_name"]=[sample_info_[f"coarse_{key}_name"]]
        else:                
            for key in ["action","object","verb"]:
                sample_info[f"{key}_name"]=sample_info_[f"fine_grained_{key}_name"]
                if self.action_taxonomy_to_use=="coarse+fine" and sample_info_[f"coarse_{key}_name"]!="NIL":
                    if "NIL" in sample_info[f"{key}_name"]:
                        sample_info[f"{key}_name"].clear()
                    
                    sample_info[f"{key}_name"].append(sample_info_[f"coarse_{key}_name"])
        return sample_info
    
    def __len__(self):
        return len(self.window_info)