# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import cv2
import numpy as np
from PIL import Image
import random
import math
import json,lmdb,glob
from meshreg.datasets import ass101utils
from meshreg.models.utils import project_hand_3d2img,align_np_sample_for_cam2local,load_mano_mean_pose
from meshreg.netscripts.utils import compare_two_pose_sets

def load_skeleton(path, joint_num):
    # load joint_world info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton



def visualize_per_frame_label_for_untrimmed_videos(untrimmed_video_infos, env_pose, env_action, env_img,
                                fps_to_load,hand_links,view_id,view_tag, dir_out_videos='./vis_v3/'): 
                                
    for seq_id, file_tag in enumerate(untrimmed_video_infos.keys()):
        subdb_pose=env_pose.open_db(file_tag.encode('ascii'),create=False)#tag_pose_dataset+"_"+file_tag
        txn_pose=env_pose.begin(write=False,db=subdb_pose)
        hand_poses_cam={}

        start_idx,end_idx=untrimmed_video_infos[file_tag]["start_idx"],untrimmed_video_infos[file_tag]["end_idx"]
        mod=untrimmed_video_infos[file_tag]["mod"]
        
        print(file_tag)
        for fid in range(start_idx,end_idx):
            frame_id=fid*(60//fps_to_load)+mod
            buf=txn_pose.get("{:06d}".format(frame_id).encode('ascii'))

            
            raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
            hand_poses_world=raw_pose[:42*3].reshape((42,3)).astype(np.float32)

            cam_raw=raw_pose[42*3+24:]
            cam_raw=cam_raw[view_id*(12+9):(view_id+1)*(12+9)]
            cam_extr,cam_intr=cam_raw[:12].reshape((3,4)),cam_raw[12:].reshape((3,3))

            reduce_factor= 270/480. if view_id<4 else 480/1920.
            cam_intr=cam_intr.copy()
            cam_intr[:2] = cam_intr[:2] *reduce_factor  

            hand_poses_hom = np.concatenate([hand_poses_world, np.ones([hand_poses_world.shape[0], 1])], 1)
            hand_poses_cam[fid-start_idx]=cam_extr.dot(hand_poses_hom.transpose()).transpose()[:, :3].astype(np.float32)
            
        #read img@30fps        
        imgs=[]
        subdb_img=env_img.open_db(f"{file_tag}_{view_tag}".encode('ascii'),create=False)
        txn_img=env_img.begin(write=False,db=subdb_img)
        for fid in range(start_idx,end_idx):
            frame_id=fid*(60//fps_to_load)+mod        
            
            try:
                buf=txn_img.get("{:06d}".format(frame_id).encode('ascii'))        
                raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
                img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            except:
                print(fid,start_idx,end_idx,"//60 fps,",frame_id)
                print(file_tag,untrimmed_video_infos[file_tag])
                assert False

            imgs.append(img)

        #read action@30fps
        subdb_action=env_action.open_db(file_tag.encode('ascii'),create=False)
        txn_action=env_action.begin(write=False,db=subdb_action)
        action_infos={}
        for fid in range(start_idx,end_idx):
            if fps_to_load<30:
                frame_id=fid*(30//fps_to_load)
            else:
                frame_id=fid//(fps_to_load//30)
            buf=txn_action.get("{:06d}".format(frame_id).encode('ascii')).decode('ascii')

            terms=buf.split('*')
            keys=["coarse_action_name","coarse_verb_name","coarse_object_name",\
                "fine_grained_action_name", "fine_grained_verb_name","fine_grained_object_name",\
                "anticipate_action_name", "anticipate_verb_name","anticipate_object_name"]
            
            caction_info=[]
            caction_info.append('Coarse '+'/'.join([terms[i] for i in range(0,3)]))   

            labels=["action","verb","noun"]

            for i in range(0,3):           
                caction_info.append('fine '+labels[i]+': '+terms[i+3])
            for i in range(0,3):
                caction_info.append('atcp '+labels[i]+': '+terms[i+6])
            
            action_infos[fid-start_idx]=caction_info
            
        if not os.path.exists(dir_out_videos):
            os.makedirs(dir_out_videos)

        path_out=os.path.join(dir_out_videos,f"{file_tag}_{view_tag}.avi")
        camera_info={"intr":cam_intr,"extr":np.eye(4)}
        ass101utils.visualize_per_frame_label_for_seq(imgs,hand_poses_cam,action_infos,camera_info,fps_to_load,path_out,hand_links)

    exit(0)
    return


def check_is_subset_in_split_ass101(split):
    with open(f'../asshand/annotations/{split}/assemblyhands_{split}_joint_3d_v1-1.json') as f:
        asshand_joints = json.load(f)["annotations"]
    ass101_videos=ass101utils.load_split_video_tags(dir_split_files='../ass101/annotations/fine-grained-annotations',split=split)
    print(len(asshand_joints))
    for fid, file_tag in enumerate(asshand_joints.keys()):
        if file_tag not in ass101_videos:
            print(split,fid,file_tag)

def load_dataset_video_pose_infos(dir_annotation,splits_to_load):
    dict_info={}
    #view_id2tag=ass101utils.get_view_id2tag()
    for split in splits_to_load:
        with open(os.path.join(dir_annotation,f"{split}_annotated@fps60.json"),"r") as f:
            cinfo_world=json.load(f)
            dict_info.update(cinfo_world)
    return dict_info

def remove_frames_without_exo_imgs(loaded_split_pose_meta,dir_annotation,view_tag):
    with open(os.path.join(dir_annotation,"exo_with_img_info.json"),"r") as f:
        dict_exo_imgs=json.load(f)
    
    to_return_pose_meta={}
    for file_tag in loaded_split_pose_meta.keys():
        if view_tag in dict_exo_imgs[file_tag].keys():
            to_return_pose_meta[file_tag]=dict_exo_imgs[file_tag][view_tag]
    
    return to_return_pose_meta


def gather_video_meta_for_untrimmed_pose_seq(task_seqs_info,pose_meta,fps_to_load,max_samples=-1):    
    total_frames=0
    set_err_no_pose_video=set()
    untrimmed_video_info={}
    pose_fps=60


    for seq_id, (file_tag,_) in enumerate(task_seqs_info.items()):
        if file_tag not in pose_meta.keys(): 
            continue
                    
        print(seq_id,file_tag)
        
        assert len(pose_meta[file_tag])==1, file_tag
        start_frame60,end_frame60 = pose_meta[file_tag][0]
        start_frame,end_frame = start_frame60//(pose_fps//fps_to_load), end_frame60//(pose_fps//fps_to_load)
        

        untrimmed_video_info[file_tag]={"start_idx":start_frame,"end_idx":end_frame+1,"mod":start_frame60%(pose_fps//fps_to_load),
                        "start_idx60":start_frame60,"end_idx60":end_frame60,}
        total_frames+=end_frame-start_frame+1

        if max_samples>0 and total_frames>max_samples:
            break 

    #Note that: fine grained action has overlap, and some video has only partial frames 
    return {'untrimmed_video_info':untrimmed_video_info}



def convert_asshand_action_to_lmdb():
    verbose=True
    root='../ass101/'
    
    dir_lmdb=os.path.join('../lmdbsv2/','asshand_action@30fps')
    fps=30 
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    data_size=1024**3 
    env = lmdb.open(dir_lmdb,map_size=data_size,max_dbs=1000)

    all_split_pose_meta=load_dataset_video_pose_infos('../ass101/asshand/')
    for split in ["val","train"]:
        coarse_action_taxonomy, videos_coarse_action_info,videos_fine_grained_action_info,videos_anticipate_action_info=ass101utils.aggregate_action_info(root,split,fps,verbose=verbose)

        for seq_id, (file_tag,fine_segs) in enumerate(videos_fine_grained_action_info.items()):    
            if file_tag not in all_split_pose_meta.keys():
                continue

            print(seq_id,file_tag)
                
            cvideo_coarse_segs=[]
            for bg_act in ["disassembly","assembly"]:
                coarse_key=bg_act+'_'+file_tag
                if coarse_key in videos_coarse_action_info.keys():
                    cvideo_coarse_segs+=videos_coarse_action_info[coarse_key]

            #process with the given fps            
            start_video,end_video=all_split_pose_meta[file_tag][0]
            start_video=start_video//(60//fps)
            end_video=end_video//(60//fps)+1
            assert len(all_split_pose_meta[file_tag])==1, file_tag

            strs_to_return=ass101utils.aggregate_video_action_info(file_tag,cvideo_coarse_segs=cvideo_coarse_segs,
                                        cvideo_fine_segs=fine_segs,
                                        cvideo_anticipate_segs=videos_anticipate_action_info[file_tag],
                                        coarse_action_taxonomy=coarse_action_taxonomy,
                                        start_video=start_video,end_video=end_video)
           
            subdb=env.open_db(file_tag.encode('ascii'))#lmdb.open(dir_lmdb,map_size=data_size*10)
            txn=env.begin(db=subdb,write=True)

            for k, v in strs_to_return.items():
                key_byte=k.encode('ascii')
                data_str=v.encode('ascii')
                txn.put(key_byte,data_str)            
            txn.commit()
    env.close() 



def convert_asshand_pose_to_lmdb():
    #reorder and check reorder
    reorder_idx=[20,3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12, 19,18,17,16]  
    palm_joints=[0,5,9,13,17] 
    links=[(0, 1, 2, 3, 4),(0, 5, 6, 7, 8),(0, 9, 10, 11, 12),(0, 13, 14, 15, 16),(0, 17, 18, 19, 20),]
    print("palm_joints",palm_joints)    

    view_tags=ass101utils.get_view_id2tag()

    mean_mano_palm_joints={'left': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT')),
                'right': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT'))}
    
    #skeleton=load_skeleton('../ass101/asshand/annotations/skeleton.txt',42)
    #for rid in reorder_idx:
    #    print(rid, skeleton[rid])
    #for rid in reorder_idx:
    #    print(rid,skeleton[rid+21])

    dir_lmdb=os.path.join('../lmdbsv3/','asshand_pose@60fps')#,split)
    data_size=1024**3 
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)
    env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)

    root_pose_annotation='../ass101/asshand/annotations'
    for split in ["val","test","train"]:
        #db = COCO(os.path.join(proot_pose_annotation,split, f'assemblyhands_{split}_ego_data_v1-1.json'))
        with open(os.path.join(root_pose_annotation, split, f'assemblyhands_{split}_ego_calib_v1-1.json')) as f:
            info_cameras = json.load(f)["calibration"]
        with open(os.path.join(root_pose_annotation, split, f'assemblyhands_{split}_joint_3d_v1-1.json')) as f:
            info_joints = json.load(f)["annotations"]
            
        dict_file_valid_segs={}
        for file_name,file_infos in info_joints.items():
            info_reorder={}
            valid_frames=[]
            print(file_name)

            path_exo_cam_json=os.path.join(root_pose_annotation,"cam_para",file_name+".json")
            dict_exo_cam_paras=load_third_person_cam_para_from_json(path_exo_cam_json)


            for frame_id, frame_info in file_infos.items():
                valid_frames.append(int(frame_id))
                craw_hand=np.array(frame_info["world_coord"],dtype=np.float32)/1000.#mm to m
                chand=np.concatenate([craw_hand[21:][reorder_idx],craw_hand[:21][reorder_idx]],axis=0)
                info_reorder[frame_id]={"world_coord":chand,"joint_valid":np.array(frame_info["joint_valid"])}

            #extract segments
            valid_frames.sort()
            dict_file_valid_segs[file_name]=[]
            cstart,cend = valid_frames[0],valid_frames[0]
            for cframe in valid_frames[1:]:
                if cframe-cend==2:#fps=30
                    cend=cframe
                else:
                    dict_file_valid_segs[file_name].append((cstart,cend))
                    cstart=cframe
            dict_file_valid_segs[file_name].append((cstart,cend))
            
            #write to lmdb            
            subdb=env.open_db(file_name.encode('ascii'))
            txn=env.begin(db=subdb,write=True)
            for frame_tag, frame_info in info_reorder.items():
                #world coord
                hand_left=info_reorder[frame_tag]["world_coord"][:21]
                hand_right=info_reorder[frame_tag]["world_coord"][21:]
                
                palm={"left":hand_left[palm_joints],"right":hand_right[palm_joints]}
                key_byte = frame_tag.encode('ascii')
                data_in_world = np.concatenate((hand_left,hand_right),axis=0).astype(np.float64)#in m

                data_world2local=align_np_sample_for_cam2local(palm,mean_mano_palm_joints,verbose=int(frame_tag)%1000==0)#in meter
                #world2cam
                data_cam_info=[]
                for vid in ["e1","e2","e3","e4"]:
                    has_cam=False
                    for ctag in view_tags[vid]:
                        if ctag not in info_cameras[file_name]["extrinsics"][frame_tag]:
                            continue
                        assert not has_cam
                        has_cam=True
                        cintr=np.array(info_cameras[file_name]["intrinsics"][ctag])
                        cextr=np.array(info_cameras[file_name]["extrinsics"][frame_tag][ctag])#tra in mm

                        cextr[:,-1]*=0.001#tra from mm to m
                        ccam=np.concatenate([cextr.flatten(),cintr.flatten()]).astype(np.float64)
                        data_cam_info.append(ccam)
                    if not has_cam:
                        data_cam_info.append(np.zeros((12+9,)))

                for vid in ["v1","v2","v3","v4","v5","v6","v7","v8"]:
                    view_tag=view_tags[vid][0]
                    
                    cintr=dict_exo_cam_paras[view_tag]["intr"].copy()
                    cextr=dict_exo_cam_paras[view_tag]["extr"].copy()#tra in mm

                    cextr[:,-1]*=0.001#tra from mm to m
                    ccam=np.concatenate([cextr.flatten(),cintr.flatten()]).astype(np.float64)
                    
                    data_cam_info.append(ccam)

                data_cam_info=np.array(data_cam_info)
                assert data_cam_info.shape[0]==12 and data_cam_info.shape[1]==21
                data=np.concatenate([data_in_world.flatten(),data_world2local.flatten(),data_cam_info.flatten()])#,info_reorder[frame_tag]["joint_valid"].flatten()])
                txn.put(key_byte,data)
            txn.commit()


            visualize_per_frame_pose_for_video(camera_info=dict_exo_cam_paras,#info_cameras[file_name],#view_cameras,#
                                    file_tag=file_name, dict_frame_poses=info_reorder, hand_links=links, 
                                    dir_videos='../ass101/videos/', dir_out_videos='./vis_v3/')        
            
        path_meta=os.path.join(f"../ass101/asshand/{split}_annotated@fps60.json")
        with open(path_meta,"w") as f:
            json.dump(dict_file_valid_segs,f)

    env.close()

def convert_asshand_ego_imgs_to_lmdb():
    root='../asshand/'
    list_videos=[]
    for split in ["train","val","test"]:
        list_videos+=glob.glob(os.path.join(root,split,"*"))

    
    data_size_per_frame= np.zeros((270,480,3),dtype=np.uint8).nbytes
    data_size=data_size_per_frame*(1024**2)

    dir_lmdb=os.path.join('../lmdbsv2/','asshand_egoimgs@60fps')
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)

    view_to_tag=ass101utils.get_view_id2tag()
    view_ids=['e3','e4']
    for cvideo_dir in list_videos:
        cvideo_tag=cvideo_dir.split("/")[-1]
        for cview_id in view_ids:
            list_imgs=[]
            for cview_tag in view_to_tag[cview_id]:
                list_imgs+=glob.glob(os.path.join(cvideo_dir,cview_tag,"*.jpg"))
            
            print(cvideo_tag,cview_tag,len(list_imgs))
            list_imgs.sort()            
            subdb=env.open_db((cvideo_tag+"_"+cview_id).encode('ascii'))
            txn=env.begin(db=subdb,write=True)
            
            for cimg_path in list_imgs:
                frame=cv2.imread(cimg_path)
                ori_w,ori_h=frame.shape[1],frame.shape[0]
                
                ratio=ori_h/270.              
                cur_w,cur_h=int(ori_w//ratio),int(ori_h//ratio)                
                frame=cv2.resize(frame,(cur_w,cur_h))
                
                frame_id=cimg_path[:-4].split("/")[-1]
                
                key_byte = frame_id.encode('ascii')
                data=cv2.imencode('.jpg',frame)[1]
                txn.put(key_byte,data)       
                
            txn.commit()
    env.close()



def convert_asshand_exo_imgs_to_lmdb():
    dir_videos="../ass101/videos/"
    dict_video_infos=load_dataset_video_pose_infos("../ass101/asshand/",["train","val","test"])
    view_to_tag=ass101utils.get_view_id2tag()
    view_ids=['v1','v2','v3','v4','v6','v8']
    
    
    data_size_per_frame= np.zeros((270,480,3),dtype=np.uint8).nbytes
    data_size=data_size_per_frame*(1024**2)

    dir_lmdb=os.path.join('../lmdbsv2/','asshand_exoimgs@60fps')
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)

    for cvideo_tag in dict_video_infos:
        cstart_idx60,cend_idx60=dict_video_infos[cvideo_tag][0]
        to_add_frame_idx60=list(range(cstart_idx60,cend_idx60+1,2))
        print(cstart_idx60,cend_idx60)
        
        for cview_id in view_ids:
            cview_tag=view_to_tag[cview_id][0]
            print(cvideo_tag,cview_id,cview_tag)


            if not os.path.exists(os.path.join(dir_videos,cvideo_tag, cview_tag+".mp4")):
                print("not exist",os.path.join(dir_videos,cvideo_tag, cview_tag+".mp4"))
                continue
                
            imgs=[]
            stream=cv2.VideoCapture(os.path.join(dir_videos,cvideo_tag, f"{cview_tag}.mp4"))
            video_fps, num_frames=int(stream.get(cv2.CAP_PROP_FPS)),int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
                      
            frame_id=0
            success, frame=stream.read()
            while success:
                #load in 60 fps
                if frame_id in to_add_frame_idx60:
                    imgs.append(frame)
                frame_id+=1
                success, frame=stream.read()
            stream.release() 

            print("loaded",len(imgs),len(to_add_frame_idx60))          
           

            subdb=env.open_db((cvideo_tag+"_"+cview_id).encode('ascii'))
            txn=env.begin(db=subdb,write=True)
            
            for fid, cimg in zip(to_add_frame_idx60,imgs):
                key_byte = "{:06d}".format(fid).encode('ascii')
                data=cv2.imencode('.jpg',cimg)[1]
                txn.put(key_byte,data)       
            txn.commit()
    env.close()
    


def visualize_by_comparing_asshand_ass101():
    split="train"
    
    mean_mano_palm_joints={'left': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT')),
                'right': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT'))}

    ass101_env_action =lmdb.open(os.path.join('../lmdbs/action@30fps',split),readonly=True,lock=False,readahead=False,meminit=False, map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
    ass101_env_pose =lmdb.open(os.path.join('../lmdbs/posev2@60fps',split),readonly=True,lock=False,readahead=False,meminit=False,map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
    ass101_env_img =lmdb.open(os.path.join('../lmdbs/imgs3rdpv@30fps',split),readonly=True,lock=False,readahead=False,meminit=False,map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)

    asshand_env_action =lmdb.open(os.path.join('../lmdbs/asshand_action@30fps'),readonly=True,lock=False,readahead=False,meminit=False, map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
    asshand_env_pose =lmdb.open(os.path.join('../lmdbs/asshand_posev2@60fps'),readonly=True,lock=False,readahead=False,meminit=False,map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
    

    asshand_pose_meta=load_dataset_video_pose_infos('../ass101/asshand')
    
    path_ass101_pose_meta=f"../ass101/poses@60fps/{split}.txt"
    with open(path_ass101_pose_meta,"r") as f:
        list_file_pose=f.readlines()
    ass101_hand_meta={}
    for line in list_file_pose:
        k,v=line.strip('\n').split(' ')
        ass101_hand_meta[k]=int(v)


    for file_tag, file_info in asshand_pose_meta.items():
        if file_tag not in ass101_hand_meta:
            continue
        if file_tag != "nusar-2021_action_both_9053-c08b_9053_user_id_2021-02-08_140757":
            continue
        
        start_idx60,end_idx60=file_info[0]
        print(file_tag)

        #load asshand pose
        asshand_poses={}
        subdb=asshand_env_pose.open_db(("asshand_"+file_tag).encode('ascii'),create=False)
        txn=asshand_env_pose.begin(write=False,db=subdb)
        for frame_id in range(start_idx60,end_idx60,2):                       
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))
            raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
            asshand_poses[frame_id//2]=raw_pose.copy()
        
        #Load asshand_action
        asshand_actions={}
        subdb=asshand_env_action.open_db(("asshand_"+file_tag).encode('ascii'),create=False)
        txn=asshand_env_action.begin(write=False,db=subdb)

        for frame_id in range(start_idx60//2,end_idx60//2):            
            buf=txn.get("{:06d}".format(frame_id).encode('ascii')).decode('ascii')
            asshand_actions[frame_id]=buf
        
        #load images
        asshand_imgs={}
        subdb=ass101_env_img.open_db(("ass101_C10118_rgb_"+file_tag).encode('ascii'),create=False)
        txn=ass101_env_img.begin(write=False,db=subdb)

        for frame_id in range(start_idx60//2,end_idx60//2):            
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))        
            raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
            asshand_imgs[frame_id]=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)


        #load ass101 pose
        ass101_poses={}
        subdb=ass101_env_pose.open_db(("ass101_"+file_tag).encode('ascii'),create=False)
        txn=ass101_env_pose.begin(write=False,db=subdb)
        for frame_id in range(start_idx60,end_idx60,2):
            try:                
                buf=txn.get("{:06d}".format(frame_id).encode('ascii'))
                raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
                ass101_poses[frame_id//2]=raw_pose.copy()
            except:
                break
        
        #check ass101 action consistency
        ass101_actions={}
        subdb=ass101_env_action.open_db(("ass101_"+file_tag).encode('ascii'),create=False)
        txn=ass101_env_action.begin(write=False,db=subdb)
        
        for frame_id in range(start_idx60//2,end_idx60//2):
            buf=txn.get("{:06d}".format(frame_id).encode('ascii'))#.decode('ascii')
            if buf==None: break
            assert buf.decode('ascii')==asshand_actions[frame_id], buf+"\n"+asshand_actions[frame_id]



        view_cameras=ass101utils.get_view_extrinsic_intrisic(path_calib_txt='../ass101/annotations/calib.txt')
        joint_links_ass101=[(0, 2, 3, 4),#warning, no 1 using here
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),]
        joint_links_asshand=[(0, 1,2, 3, 4),#warning, no 1 using here
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),]
        compare_two_pose_sets(dict_pose1=ass101_poses, 
                        dict_pose2=asshand_poses, 
                        dict_mano=mean_mano_palm_joints,
                        dict_imgs=asshand_imgs, 
                        cam_info=view_cameras['C10118_rgb'], 
                        file_tag=file_tag, 
                        joint_links1=joint_links_ass101,
                        joint_links2=joint_links_asshand, 
                        prefix_cache_img='./vis_v3/imgs/',
                        path_video=f'./{file_tag}.avi')




def load_third_person_cam_para_from_json(path_video_json):
    with open(path_video_json, "r") as f:
        camera_jsons = json.load(f)

    dict_to_return={}    
    for js in camera_jsons:
        if isinstance(js, str):
            js = json.loads(js)
        js = js.get("Camera", js)

        width = js["ImageSizeX"]
        height = js["ImageSizeY"]
        model = js["DistortionModel"]
        fx = js["fx"]
        fy = js["fy"]
        cx = js["cx"]
        cy = js["cy"]

        view_tag=js["SerialNo"]
        extrinsics = np.array(js["ModelViewMatrix"]).astype(np.float64)
        intrinsics=np.eye(3,dtype=extrinsics.dtype)
        intrinsics[0,0],intrinsics[1,1]=fx,fy
        intrinsics[0,2],intrinsics[1,2]=cx,cy

        dict_to_return[view_tag]={"extr":extrinsics[:3],"intr":intrinsics}
    return dict_to_return
        

def check_has_exoimg():
    root_pose_annotation='../ass101/asshand/'
    splits_to_load=["train","val","test"]
    loaded_split_pose_meta=load_dataset_video_pose_infos(root_pose_annotation,splits_to_load)
       
    env_img_path=os.path.join('../lmdbsv2/asshand_exoimgs@60fps')
    env_img=lmdb.open(env_img_path,readonly=True,lock=False,readahead=False,meminit=False,\
                            map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)

    view_tags=["v1","v2","v3","v4","v6","v8"]
    dict_video_frames_with_imgs={}

    for file_tag, seq_info in loaded_split_pose_meta.items():
        dict_video_frames_with_imgs[file_tag]={}
        start_idx, end_idx = seq_info[0]
        print(file_tag,start_idx,end_idx)

        for view_tag in view_tags:
            try:
                subdb_img = env_img.open_db((file_tag+"_"+view_tag).encode('ascii'),create=False)
                txn_img = env_img.begin(db=subdb_img,write=False)
            except:
                print("not found",file_tag,view_tag)
                continue
            
            has_ended=False
            for frame_idx in range(start_idx,end_idx+1,2):
                
                buf=txn_img.get("{:06d}".format(frame_idx).encode('ascii'))        
                #raw_data=np.frombuffer(buf,dtype=np.uint8)
                #img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                assert not has_ended
                if buf is None:                    
                    dict_video_frames_with_imgs[file_tag][view_tag]=[(start_idx,frame_idx-2)]
                    has_ended=True
                    break

            if not has_ended:
                dict_video_frames_with_imgs[file_tag][view_tag]=[(start_idx,end_idx)]
            else:
                print(dict_video_frames_with_imgs[file_tag][view_tag],start_idx,end_idx)
    
    path_meta=os.path.join(f"../ass101/asshand/exo_with_img_info.json")
    with open(path_meta,"w") as f:
        json.dump(dict_video_frames_with_imgs,f)
    
    
def check_has_egoimg_has_camera():
    root_pose_annotation='../ass101/asshand/annotations'
    root_imgs="../asshand/"
    dict_video_frames_no_imgs, dict_video_frames_no_cams={},{}
    
    view_tags=ass101utils.get_view_id2tag()

    for split in ["val","test","train"]:
        with open(os.path.join(root_pose_annotation, split, f'assemblyhands_{split}_ego_calib_v1-1.json')) as f:
            info_cameras = json.load(f)["calibration"]
        with open(os.path.join(root_pose_annotation, split, f'assemblyhands_{split}_joint_3d_v1-1.json')) as f:
            info_joints = json.load(f)["annotations"]

            
        for file_name,file_infos in info_joints.items():  
            valid_frames=[]
            for frame_id, frame_info in file_infos.items():
                valid_frames.append(int(frame_id))
            
            dict_video_frames_no_imgs[file_name]={"e1":[],"e2":[],"e3":[],"e4":[]}
            dict_video_frames_no_cams[file_name]={"e1":[],"e2":[],"e3":[],"e4":[]}
                
            #extract segments
            valid_frames.sort()            
            for frame_id in valid_frames:
                frame_tag="{:06d}".format(frame_id)

                for vid in ["e3","e4"]:
                    has_cam=False
                    real_ctag=None
                    for ctag in view_tags[vid]:
                        if ctag not in info_cameras[file_name]["extrinsics"][frame_tag]:
                            continue
                        assert not has_cam
                        has_cam=True
                        real_ctag=ctag
                        
                    if not has_cam:
                        dict_video_frames_no_cams[file_name][vid].append(frame_id)
                        dict_video_frames_no_imgs[file_name][vid].append(frame_id)
                    elif not os.path.exists(os.path.join(root_imgs,split,file_name,real_ctag,frame_tag+".jpg")):
                        dict_video_frames_no_imgs[file_name][vid].append(frame_id)
                        
            print(file_name,valid_frames[0],valid_frames[-1])
            for vid in ["e3","e4"]:
                print(vid,"cam",len(dict_video_frames_no_cams[file_name][vid])/len(valid_frames))
                print(vid,"img",len(dict_video_frames_no_imgs[file_name][vid])/len(valid_frames))

    
    path_meta=os.path.join(f"../ass101/asshand/ego_missing_img_cam_info.json")
    with open(path_meta,"w") as f:
        json.dump({"no_imgs":dict_video_frames_no_imgs,"no_cams":dict_video_frames_no_cams},f)
    
    