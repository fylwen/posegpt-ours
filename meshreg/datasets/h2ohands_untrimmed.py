import os
import lmdb

import numpy as np
from PIL import Image, ImageFile
import sys
sys.path.append('../../')
from meshreg.datasets import h2outils,ass101utils
from meshreg.datasets.vanilla_dataset import VanillaDataset,get_window_starts_untrimmed_videos,get_window_starts_trimmed_videos
from meshreg.models.utils import project_hand_3d2img
from meshreg.datasets.queries import BaseQueries
import cv2


class H2OHands(VanillaDataset):
    def __init__(
        self,
        dataset_folder,
        split, 
        view_id,
        ntokens_per_seq,
        spacing, 
        mode,
        is_shifting_window,
        buf_sec,
        min_window_sec,
        shift_as_htt,
    ):
        super().__init__()
        self.mode=mode
        self.ntokens_per_seq=ntokens_per_seq

        assert buf_sec==0,"if buf_sec!=0, need to change action name assignment in seqset"
        self.buf_len=int(buf_sec*self.fps)
        self.min_window_len= int(min_window_sec*self.fps)

        print("min_window_len",self.min_window_len)

        self.is_shifting_window=is_shifting_window 
        self.shift_as_htt=shift_as_htt
        self.spacing=spacing
        
        self.name = "h2o"
        self.split = split
        self.root = os.path.join(dataset_folder,"h2o")

        action_taxonomy=h2outils.load_action_taxonomy()
        self.action_to_idx=action_taxonomy["action_name2idx"]
        self.object_to_idx=action_taxonomy["object_name2idx"] 
        self.verb_to_idx=action_taxonomy["verb_name2idx"]
        self.action_info=action_taxonomy["action_info"] 


        self.num_actions = len(self.action_to_idx.keys())
        self.num_objects = len(self.object_to_idx.keys())
        self.num_verbs=len(self.verb_to_idx.keys())

        #check order
        for i,(k,v) in enumerate(self.action_to_idx.items()):
            assert self.action_to_idx[k]==i,'action #{:d}: {:s} {:d}'.format(i,k,v)        
        for i,(k,v) in enumerate(self.object_to_idx.items()):
            assert self.object_to_idx[k]==i,'object #{:d}: {:s} {:d}'.format(i,k,v)        
        for i,(k,v) in enumerate(self.verb_to_idx.items()):
            assert self.verb_to_idx[k]==i,'verb #{:d}: {:s} {:d}'.format(i,k,v)
                 

        self.view_id= None if view_id<0 else view_id#######        
        self.view_tag="cam4" if self.view_id is None else "cam{:d}".format(self.view_id)
        self.reduce_factor = 480 / 1280.0
        
        trimmed_action_segments=h2outils.load_split_video_segments_dict(self.root)  
        self.video_infos=h2outils.gather_video_meta_for_untrimmed_pose_seq(trimmed_action_segments[self.split].keys(),self.root)
        
        if self.mode in ["motion"]:
            self.window_info=get_window_starts_untrimmed_videos(untrimmed_video_infos=self.video_infos,
                                                                ntokens=self.ntokens_per_seq, #5,
                                                                spacing=self.spacing,#2,
                                                                min_window_len=self.min_window_len,#60,
                                                                is_shifting_window=is_shifting_window)
            self.list_meta_action=None
        elif self.mode in ["context"]:
            list_meta_action=[]
            self.window_info =get_window_starts_trimmed_videos(trimmed_seg_infos=trimmed_action_segments[self.split],
                                                    untrimmed_video_infos=self.video_infos,
                                                    trimmed_type="h2o",
                                                    list_meta_action=list_meta_action,
                                                    ntokens=self.ntokens_per_seq,
                                                    spacing=self.spacing,
                                                    min_window_len=self.min_window_len,
                                                    buf_len=self.buf_len,
                                                    is_shifting_window=is_shifting_window,
                                                    shift_as_htt=self.shift_as_htt)
            self.list_meta_action=list_meta_action

            print("H2O- check list_meta_action/window_info",len(self.list_meta_action),len(self.window_info))
            print("list_meta_action",self.list_meta_action[-1])
            print("window_info",self.window_info[-1])
        else:
            assert False

        self.env_action=lmdb.open(os.path.join('../lmdbsv2/h2o_action@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)



        if BaseQueries.JOINTS3D in self.all_queries:
            assert BaseQueries.RESNET_JOINTS3D not in self.all_queries
            self.env_pose=lmdb.open(os.path.join('../lmdbsv2/h2o_pose@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        else:
            assert self.view_id is not None and BaseQueries.RESNET_JOINTS3D in self.all_queries
            self.env_pose=lmdb.open(os.path.join(f'../lmdbsv2/h2o_resnet@30fps/{self.split}_viewid_{self.view_id}'),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)




        self.env_img=lmdb.open(os.path.join('../lmdbsv2/h2o_imgs@30fps',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        
        
        try:
            self.env_midpe=lmdb.open(os.path.join(f'../lmdbsv2/h2o_midpe_world16x1@30fps_2sets_pdobsv',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
        except:
            self.env_midpe=None
            
        print(len(self.window_info))

        #h2outils.visualize_per_frame_label_for_untrimmed_videos(self.video_infos, self.env_pose, self.env_action, self.env_img, self.links,dir_out_videos='./vis_v3/')
        #exit(0)
            
            
    def get_joints3d_cam2local_from_lmdb(self, txn, sample_info):
        frame_id=sample_info["frame_idx"]
        return self.__get_joints3d_cam2local_from_lmdb__(txn,frame_id,view_id=self.view_id)

    
    def get_cam_extr_intr(self, txn, sample_info):
        frame_id=sample_info["frame_idx"]
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
        buf=txn.get("{:06d}".format(frame_id).encode('ascii')).decode('ascii')
        aname,vname,oname=buf.split('*')
        sample_info["action_name"]=[aname]
        sample_info["verb_name"]=[vname]
        sample_info["object_name"]=[oname]


        return sample_info
    
    
        
    def __len__(self):
        return len(self.window_info)

