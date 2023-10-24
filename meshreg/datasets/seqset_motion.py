import random
import traceback

from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFilter
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torchvision.transforms import functional as func_transforms

from meshreg.datasets.queries import BaseQueries
from meshreg.models.utils import solve_pnp_and_transform
from libyana.transformutils import colortrans, handutils

import sys,copy
sys.path.append('./')
sys.path.append('../')
import cv2


class SeqSet(Dataset):
    def __init__(self,
            list_pose_datasets,
            queries,
            ntokens_per_clip,
            spacing,
            nclips,
            nclips_dev,
            aug_obsv_len,
            aug_img,
            const_obsv=-1,
            const_pred=-1):
        self.list_pose_datasets = list_pose_datasets
        self.list_pose_datasets_start=[0]
        for pose_dataset in self.list_pose_datasets:
            self.list_pose_datasets_start.append(pose_dataset.__len__())
        for pd_len in range(1,len(self.list_pose_datasets_start)):
            self.list_pose_datasets_start[pd_len]+=self.list_pose_datasets_start[pd_len-1]

        # Training attributes
        self.queries = queries
        self.aug_obsv_len=aug_obsv_len
        self.use_same_action=self.list_pose_datasets[0].mode=="context"
        
        self.const_obsv=const_obsv
        self.const_pred=const_pred
        self.ntokens_per_clip=ntokens_per_clip
        self.spacing=int(spacing)
        self.len_tokens=int(nclips*self.ntokens_per_clip)
        self.ndev_tokens=int(nclips_dev*self.ntokens_per_clip)

        self.ncam_K=np.array([[240.,0.0,240.],[0.0,240.0,135.],[0.0,0.0,1.0]],dtype=np.float32)
        self.ncam_res=(480,270)
        
        # Color jitter attributes
        self.aug_img=aug_img
        self.hue = 0.15
        self.contrast = 0.5
        self.brightness = 0.5
        self.saturation = 0.5
        self.blur_radius = 0.5
        self.scale_jittering = 0
        self.center_jittering = 0.1

        print(self.__len__(), self.list_pose_datasets_start)
        print("use_same_aciton: ",self.use_same_action,"len_tokens",self.len_tokens)

    def __len__(self):
        return self.list_pose_datasets_start[-1]


    def get_sample(self,pose_dataset, sample_info, assigned_action_name=None,  txn=None, color_augm=None, space_augm=None, verbose=False):
        sample = {}
        #Select action
        num_labels=len(sample_info["action_name"])      
        if verbose:
            assert num_labels>0, "num_labels should always > 0"
            keys_to_check=["action_name","verb_name","object_name"]
            for k in keys_to_check:
                if not k in sample_info:
                    continue
                assert isinstance(sample_info[k],list), "{:s} shd be a list".format(k)
                assert len(sample_info[k])==num_labels, '{:s} in sample_info shd have len {:d}'.format(k,num_labels)
                if "NIL" in sample_info[k]:
                    assert len(sample_info[k])==1, "if NIL then len 1"


        if assigned_action_name is None or assigned_action_name not in sample_info["action_name"]:
            rid=np.random.randint(num_labels)
        else:
            rid=sample_info["action_name"].index(assigned_action_name)
            try:
                assert assigned_action_name in sample_info["action_name"]
                assert sample_info["action_name"][rid]==assigned_action_name
            except:
                print("check assigned_action_name",assigned_action_name, rid)
                print("sample_info",sample_info["action_name"])
                assert False, "assigned_action not in sample_info"

        #Assign action
        if BaseQueries.ACTIONNAME in self.queries:
            sample["action_name"]=sample_info["action_name"][rid]
        if BaseQueries.OBJNAME in self.queries:
            sample["obj_name"]=sample_info["object_name"][rid]
            sample["has_valid_obj"]=sample["obj_name"]!="NIL"
        if BaseQueries.VERBNAME in self.queries:
            sample["verb_name"]=sample_info["verb_name"][rid]

        # Get 3D hand joints
        if BaseQueries.JOINTS3D in self.queries or BaseQueries.RESNET_JOINTS3D in self.queries:
            if BaseQueries.JOINTS3D in self.queries:
                pose_info = pose_dataset.get_joints3d_cam2local_from_lmdb(txn["pose"],sample_info)
            else:
                pose_info = pose_dataset.get_resnet_output_from_lmdb(txn["pose"], sample_info["frame_idx"], self.ncam_K)
            sample.update(pose_info)
            
            
        if BaseQueries.JOINTS2D in self.queries or BaseQueries.IMAGE in self.queries:
            #Here begins augmentation for img and 2D related
            center = np.array((self.ncam_res[0]/2,self.ncam_res[1]/2))
            scale = self.ncam_res[0]
            # Data augmentation
            if space_augm is not None:
                center = space_augm["center"]
                scale = space_augm["scale"]
            elif self.aug_img:
                # Randomly jitter center
                # Center is located in square of size 2*center_jitter_factor, in center of cropped image
                center_jit = Uniform(low=-1, high=1).sample((2,)).numpy()
                center_offsets = self.center_jittering * scale * center_jit
                center = center + center_offsets.astype(int)

                # Scale jittering
                scale_jit = Normal(0, 1).sample().item() + 1
                scale_jittering = self.scale_jittering * scale_jit
                scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
                scale = scale * scale_jittering

            space_augm = {"scale": scale, "center": center}
            affinetrans, post_rot_trans = handutils.get_affine_transform(center, scale, self.ncam_res, rot=0)
        
        sample["space_augm"] = space_augm
        # Get 2D hand joints and correctify w.r.t. ncam
        if BaseQueries.JOINTS2D in self.queries:
            if BaseQueries.RESNET_JOINTS3D in self.queries:
                sample["aug_joints2d_left"],sample["aug_joints2d_right"]= sample["joints25d"][:21,:2].copy(),sample["joints25d"][21:,:2].copy()
                sample["ncam_joints3d_left"], sample["ncam_joints3d_right"] = sample["cam_joints3d_left"].copy(),sample["cam_joints3d_right"].copy()
            else:
                cam_joints=np.concatenate([sample["cam_joints3d_left"],sample["cam_joints3d_right"]],axis=0)#unit in meter
                hom2d = sample["cam_intr"].dot(cam_joints.transpose()).transpose()
                joints2d = (hom2d / hom2d[:, 2:])[:, :2]
                sample["joints2d_left"],sample["joints2d_right"]=joints2d[:21],joints2d[21:]

                #augmentate joints2d
                aug_joints2d=np.array(handutils.transform_coords(joints2d,affinetrans)).astype(np.float32)
                sample["aug_joints2d_left"],sample["aug_joints2d_right"]=aug_joints2d[:21],aug_joints2d[21:]


                #rectify 3d joints w.r.t. normalized camera            
                results_cam2ncam=solve_pnp_and_transform(cam_joints,joints2d,self.ncam_K)
                sample["ncam_joints3d_left"], sample["ncam_joints3d_right"] = results_cam2ncam["cam2_joints3d"][:21],results_cam2ncam["cam2_joints3d"][21:]
                
                #augmentate ncam_K
                sample["ncam_intr"] = post_rot_trans.dot(self.ncam_K.copy()).astype(np.float32)
                sample["cam_intr"] = post_rot_trans.dot(sample["cam_intr"].copy()).astype(np.float32)
            
            if verbose:
                ncam_joints2d = results_cam2ncam["cam2_joints2d"]
                #sample["ncam_joints2d_left"], sample["ncam_joints2d_right"] = results_cam2ncam["cam2_joints2d"][:21],results_cam2ncam["cam2_joints2d"][21:]
                print("check 2d for under ncam and under cam",(ncam_joints2d-joints2d).max())            
                results_ncam2cam=solve_pnp_and_transform(results_cam2ncam["cam2_joints3d"],joints2d, sample["cam_intr"])
                print("check 3d for cam-ncam-cam and ori cam",(cam_joints-results_ncam2cam["cam2_joints3d"]).max())
        
        
        sample["color_augm"] = None
        if BaseQueries.IMAGE in self.queries:
            img, has_img = pose_dataset.get_image(txn["image"],sample_info) 
            
            if img.shape[1]<480: #asshand ego
                assert False, "check projection after augmentation"
                img_ori=img.copy()
                img=np.zeros((270,480,3),dtype=np.uint8)+255
                start_x=(480-img_ori.shape[1])//2                
                img[:,start_x:start_x+img_ori.shape[1]]=img_ori.copy()
                #modify also the intrinsics
                sample["cam_intr"][0,2]+=start_x
            sample["image_vis"]=img.copy()
            
            if self.aug_img:
                assert False, "examine video seq with augmentation"
                trans_img=Image.fromarray(img[:,:,::-1].copy())
                blur_radius = Uniform(low=0, high=1).sample().item() * self.blur_radius
                trans_img = trans_img.filter(ImageFilter.GaussianBlur(blur_radius))
                if color_augm is None:
                    bright, contrast, sat, hue = colortrans.get_color_params(brightness=self.brightness,saturation=self.saturation,hue=self.hue, contrast=self.contrast,)
                else:
                    sat = color_augm["sat"]
                    contrast = color_augm["contrast"]
                    hue = color_augm["hue"]
                    bright = color_augm["bright"]
                trans_img = colortrans.apply_jitter(trans_img, brightness=bright, saturation=sat, hue=hue, contrast=contrast)
                sample["color_augm"] = {"sat": sat, "bright": bright, "contrast": contrast, "hue": hue}

                trans_img=handutils.transform_img(trans_img,affinetrans,self.ncam_res)
                trans_img=trans_img.crop((0,0,self.ncam_res[0],self.ncam_res[1]))
                trans_img=np.array(trans_img)
                sample["image_vis"]=trans_img
            else:
                trans_img=img[:,:,::-1].copy()
            # Tensorize and normalize_img
            trans_img = func_transforms.to_tensor(trans_img).float()
            trans_img = func_transforms.normalize(trans_img, [0.5, 0.5, 0.5], [1, 1, 1]) 
            sample["image"]=trans_img

            sample["has_camera_and_image"]=has_img and not sample["is_world_camera"]
            sample["has_obj_and_image"]=has_img and sample["obj_name"]!="NIL"

        
        sample["tag"]="{:s},{:s},{:d}".format(sample_info["video_tag"],assigned_action_name if self.use_same_action else "NIL",sample_info["frame_idx"]) 
        return sample

    def __getitem__(self, idx, verbose=False):
        pid=0
        while pid<len(self.list_pose_datasets_start) and idx>self.list_pose_datasets_start[pid+1]:
            pid+=1 
        
        window_idx=idx-self.list_pose_datasets_start[pid]
        pose_dataset=self.list_pose_datasets[pid]
        
        window_info=pose_dataset.get_window_info(window_idx)
        txn=pose_dataset.open_seq_lmdb(window_info,
                            load_pose=BaseQueries.JOINTS3D in self.queries or BaseQueries.RESNET_JOINTS3D in self.queries, 
                            load_action=BaseQueries.ACTIONNAME in self.queries, 
                            load_image=BaseQueries.IMAGE in self.queries)

        end_video=window_info["action_end_frame"] if self.use_same_action else window_info["end_idx"]#@dataset load fps (i.e., self.pose_dataset.fps=30)
        assigned_action_name=window_info["action_name"] if self.use_same_action else None

        cframe_idx=window_info["frame_idx"]+self.ndev_tokens*self.spacing
        csample_info=pose_dataset.get_sample_info(window_info, cframe_idx, txn["action"])

        sample = self.get_sample(pose_dataset,csample_info,assigned_action_name=assigned_action_name,txn=txn,verbose=verbose)# 
        assert csample_info["frame_idx"]==cframe_idx, "check frame_idx"
        sample['valid_frame'] = 1
        sample["frame_since_action_start"]=max(0,cframe_idx-window_info["action_start_frame"])
        
        space_augm = sample.pop("space_augm")
        color_augm = sample.pop("color_augm")

        samples=[sample]
        if False:
            if self.aug_obsv_len:
                len_obsv=random.randint(16, min(end_video-cframe_idx,self.ntokens_per_clip))
                start_idx=16 if end_video-cframe_idx-len_obsv>16 else 1
                len_pred=random.randint(start_idx, min(max(0,end_video-cframe_idx-len_obsv)+1,self.ntokens_per_clip))
            else:
                len_obsv=min(self.const_obsv,self.ntokens_per_clip)
                len_pred=min(self.const_pred,self.ntokens_per_clip)
        else:
            len_obsv=  random.randint(2,self.ntokens_per_clip) if self.ntokens_per_clip>2 and self.aug_obsv_len and random.random()<0.25  else self.ntokens_per_clip
            len_pred=self.ntokens_per_clip
        
        for sample_idx in range(1,self.len_tokens):
            if sample_idx in range(len_obsv,self.ntokens_per_clip) or sample_idx in range(self.ntokens_per_clip+len_pred,self.ntokens_per_clip*2):
                fframe_idx=cframe_idx
            else:
                fframe_idx=min(cframe_idx+self.spacing, end_video-1)

            #fetch next frame
            fsample_info=pose_dataset.get_sample_info(window_info, fframe_idx, txn["action"])
            
            fut_valid_frame = fframe_idx-cframe_idx==self.spacing 

            if fut_valid_frame:
                sample_fut_frame = self.get_sample(pose_dataset,fsample_info,assigned_action_name=assigned_action_name, txn=txn, color_augm=color_augm, space_augm=space_augm, verbose=verbose)
                sample_fut_frame.pop("space_augm")
                sample_fut_frame.pop("color_augm")
                cframe_idx=fframe_idx
            else:
                sample_fut_frame=copy.deepcopy(samples[-1])

            assert fsample_info["frame_idx"]==fframe_idx, "check frame_idx" 
            sample_fut_frame["valid_frame"] = fut_valid_frame
            sample_fut_frame["frame_since_action_start"]=max(0,fframe_idx-window_info["action_start_frame"])
            samples.append(sample_fut_frame)
        return samples

