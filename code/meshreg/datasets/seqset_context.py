import random
import traceback

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


from meshreg.datasets.queries import BaseQueries
from meshreg.datasets.seqset_motion import SeqSet as SeqSet_

import sys
sys.path.append('./')
sys.path.append('../')


class SeqSet(SeqSet_):
    def __init__(self,
                list_pose_datasets,
                queries,
                ntokens_per_clip,
                spacing,
                capacity_ntokens,
                const_ntokens_obsv,
                const_ntokens_pred,
                nclips_dev,
                strict_untrimmed_video):
        super().__init__(list_pose_datasets=list_pose_datasets,
                        queries=queries,
                        ntokens_per_clip=ntokens_per_clip,
                        spacing=spacing,
                        nclips=2,
                        nclips_dev=nclips_dev,
                        aug_img=False,
                        aug_obsv_len=False)
                        
        self.capacity_ntokens=capacity_ntokens
        self.strict_untrimmed_video=strict_untrimmed_video
        
        self.random_obsv_len=const_ntokens_obsv<=0
        self.const_ntokens_obsv=const_ntokens_obsv if const_ntokens_obsv>0 else self.capacity_ntokens
        self.const_ntokens_pred=const_ntokens_pred if const_ntokens_pred>0 else self.capacity_ntokens

        self.grp_key=[['R_cam2local_left','t_cam2local_left','R_cam2local_right','t_cam2local_right','cam_joints3d_left','cam_joints3d_right','cam_intr','cam_extr'], 
                        ['hand_size_left','hand_size_right'],  ['valid_joints_left','valid_joints_right'], ['valid_frame','frame_idx'],]
        
        if BaseQueries.RESNET_JOINTS3D in self.queries:
            self.grp_key[0]+=["joints25d","ncam_intr","cam_joints3d_left_resnet","cam_joints3d_right_resnet","joints25d_resnet",
                        "R_cam2local_left_resnet","t_cam2local_left_resnet","R_cam2local_right_resnet","t_cam2local_right_resnet",
                        "image_feature","object_feature"]
            self.grp_key[1]+=["hand_size_left_resnet","hand_size_right_resnet"]
        if BaseQueries.IMAGE in self.queries:
            self.grp_key[0]+=["image","image_vis"]
        
        print(self.grp_key)

    def get_obsv_or_pred_part(self,is_obsv,pose_dataset, meta_info, window_info, fframe_idx, txn, verbose):
        batch_sample={"clip_valid_clip":np.zeros((meta_info["max_num_clips"],1),dtype=np.int32),\
                "clip_since_action_start":np.zeros((meta_info["max_num_clips"],1),dtype=np.int32),\
                "clip_action_name":["NIL" for i in range(meta_info["max_num_clips"])],\
                "clip_obj_name":["NIL" for i in range(meta_info["max_num_clips"])],}
            
        assigned_action_name= window_info["action_name"]
        if verbose:
            assert assigned_action_name != "NIL", "obsv shd be trimmed segs with action"

        fv_cid,fv_fid=0,0
        for cidx in range(0,meta_info["num_clips"]):
            #get the current clip info
            for fidx in range(0,self.ntokens_per_clip):
                is_valid_frame = fframe_idx<meta_info["end_frame"]
                if not is_valid_frame:
                    if verbose:
                        assert fidx==fv_fid+1
                        assert cidx==meta_info["num_clips"]-1, "Expect non-pad frames, unless last clip"
                        assert cidx==fv_cid
                    break
                    
                cframe_idx = fframe_idx
                fv_cid,fv_fid=cidx,fidx
                csample_info=pose_dataset.get_sample_info(window_info,cframe_idx,txn["action"])                
                frame_sample=self.get_sample(pose_dataset=pose_dataset,sample_info=csample_info, assigned_action_name=assigned_action_name, txn=txn, verbose=verbose)
                frame_sample["valid_frame"]=is_valid_frame
                frame_sample["frame_idx"]=cframe_idx
                
                if fidx==0:    
                    if BaseQueries.MIDPE in self.queries:
                        cclip_midpe=pose_dataset.get_midpe_from_lmdb(txn["midpe"],assigned_action_name=None,sample_info=csample_info)#np.zeros((512,),dtype=np.float32)
                    batch_sample["clip_since_action_start"][cidx,0]=(cframe_idx-meta_info["start_frame"])//(self.ntokens_per_clip*self.spacing)

                #initialize per-frame and per-clip info
                if cidx==0 and fidx==0:
                    for key in frame_sample.keys():
                        if key in self.grp_key[0]+self.grp_key[2]:
                            if key in ["image"]:
                                batch_sample['clip_frame_'+key]=torch.zeros((meta_info["max_num_clips"],self.ntokens_per_clip,)+frame_sample[key].shape,dtype=frame_sample[key].dtype)
                            else:
                                batch_sample['clip_frame_'+key]=np.zeros((meta_info["max_num_clips"],self.ntokens_per_clip,)+frame_sample[key].shape,dtype=frame_sample[key].dtype)
                        elif key in self.grp_key[1]:
                            batch_sample['clip_frame_'+key]=np.zeros((meta_info["max_num_clips"],self.ntokens_per_clip),dtype=frame_sample[key].dtype)
                        elif key in self.grp_key[3]:
                            batch_sample['clip_frame_'+key]=np.zeros((meta_info["max_num_clips"],self.ntokens_per_clip),dtype=np.int32)
                    if BaseQueries.MIDPE in self.queries:
                        batch_sample["clip_midpe"]=np.zeros((meta_info["max_num_clips"],cclip_midpe.shape[0]),dtype=cclip_midpe.dtype)
                        
                #update per-frame info
                for key in self.grp_key[0]+self.grp_key[1]+self.grp_key[2]+self.grp_key[3]: 
                    if key in frame_sample.keys():
                        batch_sample["clip_frame_"+key][cidx,fidx]=frame_sample[key]  

                #update per-clip info
                if fidx==0: 
                    if BaseQueries.MIDPE in self.queries:
                        batch_sample["clip_midpe"][cidx]=cclip_midpe
                        
                    for key in ['action_name','obj_name']:
                        batch_sample["clip_"+key][cidx]=frame_sample[key]
                        if verbose:
                            assert batch_sample["clip_"+key][cidx]==batch_sample["clip_"+key][0], "inconsistent obsved-clip "+key+batch_sample["clip_"+key][-1]+batch_sample["clip_"+key][0]
                            assert batch_sample["clip_action_name"][cidx]==assigned_action_name
                
                #update fframe_idx
                fframe_idx = cframe_idx+self.spacing

            #update clip-wise padding, test stage need to mask the last valid clip.
            batch_sample["clip_valid_clip"][cidx,0]= 0 if (not self.random_obsv_len) and cidx==meta_info["num_clips"]-1 and not is_obsv else 1

        for pcidx in range(fv_cid+1,meta_info["max_num_clips"]):
            batch_sample["clip_since_action_start"][pcidx,0]=(pcidx-fv_cid)+batch_sample["clip_since_action_start"][fv_cid,0]
        if verbose:
            for pcidx in range(1,meta_info["max_num_clips"]):
                assert batch_sample["clip_since_action_start"][pcidx]-batch_sample["clip_since_action_start"][pcidx-1]==1,"clip_since_action_start"

        for key in self.grp_key[0]+self.grp_key[1]:
            if "clip_frame_"+key in batch_sample and key!="image":
                batch_sample['clip_frame_'+key][fv_cid,fv_fid+1:]=batch_sample['clip_frame_'+key][fv_cid,fv_fid:fv_fid+1].copy()
                batch_sample['clip_frame_'+key][fv_cid+1:,:]=batch_sample["clip_frame_"+key][fv_cid:fv_cid+1,fv_fid:fv_fid+1].copy() 

        return batch_sample,cframe_idx

    def empty_pred(self, meta_info, batch_obsv):
        batch_pred={}
        for k,v in batch_obsv.items():
            if k in ["clip_action_name","clip_obj_name"]:
                batch_pred[k]=["NIL" for i in range(meta_info["max_num_clips"])]  
            elif k in ["clip_frame_image"]:
                batch_pred[k]=torch.zeros((meta_info["max_num_clips"],)+v.shape[1:],dtype=v.dtype)
            else:
                batch_pred[k]=np.zeros((meta_info["max_num_clips"],)+v.shape[1:],dtype=v.dtype)
                
        for key in self.grp_key[0]+self.grp_key[1]:
            if "clip_frame_"+key in batch_pred and key!="image":
                batch_pred['clip_frame_'+key][:,:]=batch_obsv["clip_frame_"+key][-2:-1,-2:-1].copy()
                    
        return batch_pred
    

    def get_safe_item(self,pose_dataset,window_info,verbose=False):
        txn=pose_dataset.open_seq_lmdb(window_info,load_pose=BaseQueries.JOINTS3D in self.queries or BaseQueries.RESNET_JOINTS3D in self.queries, 
                                        load_action=BaseQueries.ACTIONNAME in self.queries, 
                                        load_image=BaseQueries.IMAGE in self.queries)
        
        video_end_frame=window_info["end_idx"]#@dataset load fps (i.e., self.pose_dataset.fps=30)
        action_start_frame,action_end_frame=window_info["action_start_frame"],window_info["action_end_frame"]
        action_name=window_info["action_name"]
        
        cframe_idx=window_info["frame_idx"]+self.ndev_tokens*self.spacing
        if verbose:
            assert action_end_frame<=video_end_frame
            
        num_action_clips=int(np.ceil((action_end_frame-cframe_idx)/(self.spacing*self.ntokens_per_clip)))
        #obsv random num_obsv_clips
        if self.random_obsv_len:
            num_obsv_clips=random.randint(1, min(self.const_ntokens_obsv, num_action_clips))
            num_pred_clips=0 if num_action_clips-num_obsv_clips<1 else random.randint(1, min(self.const_ntokens_pred, num_action_clips-num_obsv_clips))
        elif self.const_ntokens_obsv<self.capacity_ntokens:            
            num_obsv_clips=min(self.const_ntokens_obsv, num_action_clips-1)
            num_pred_clips=0 if num_action_clips-num_obsv_clips<1 else min(self.const_ntokens_pred+1, num_action_clips-num_obsv_clips)
        else:
            num_obsv_clips=min(self.const_ntokens_obsv,num_action_clips)
            num_pred_clips=0 if num_action_clips-num_obsv_clips<1 else min(self.const_ntokens_pred+1, num_action_clips-num_obsv_clips)

        batch_sample={}
        meta_end_frame=action_end_frame if self.strict_untrimmed_video else video_end_frame
        
        meta_info={"obsv":{"start_frame":action_start_frame,"num_clips":num_obsv_clips, "end_frame":meta_end_frame,"max_num_clips":self.capacity_ntokens},
            "pred":{"start_frame":action_start_frame,"num_clips":num_pred_clips, "end_frame":meta_end_frame, "max_num_clips":self.capacity_ntokens+1},}
        
        #First fetch obsvation
        batch_sample["obsv"],cframe_idx=self.get_obsv_or_pred_part(is_obsv=True, pose_dataset=pose_dataset, 
                                                    meta_info=meta_info["obsv"], window_info=window_info, 
                                                    fframe_idx=cframe_idx, txn=txn, verbose=verbose)
        #move forward, and then fetch predion     
        if num_pred_clips==0:
            batch_sample["pred"]=self.empty_pred(meta_info["pred"],batch_sample["obsv"])
        else:       
            if verbose:
                assert cframe_idx==batch_sample["obsv"]["clip_frame_frame_idx"][num_obsv_clips-1,-1],\
                    "inconsistent end frame {:d} vs {:d}".format(cframe_idx,batch_sample["obsv"]["clip_frame_frame_idx"][num_obsv_clips-1,-1])
            batch_sample["pred"],_=self.get_obsv_or_pred_part(is_obsv=False, pose_dataset=pose_dataset, 
                                                    meta_info=meta_info["pred"], window_info=window_info, 
                                                    fframe_idx=cframe_idx+self.spacing, txn=txn, verbose=verbose)
        
        return_sample={}
        for label in ["obsv","pred"]:
            for k,v in batch_sample[label].items():
                if isinstance(v,list):
                    return_sample[label+"_"+k]="@".join(v)
                else:
                    return_sample[label+"_"+k]=v         
                 
        return return_sample



    def __getitem__(self, idx, verbose=False): 
        pid=0
        while pid<len(self.list_pose_datasets_start) and idx>self.list_pose_datasets_start[pid+1]:
            pid+=1 
        window_idx=idx-self.list_pose_datasets_start[pid]
        pose_dataset=self.list_pose_datasets[pid]
        
        try:
            window_info=pose_dataset.get_window_info(window_idx)
            return_sample=self.get_safe_item(pose_dataset,window_info,verbose)
        except Exception:
            print("window_info",window_info)
            traceback.print_exc()
            assert False
        return return_sample
