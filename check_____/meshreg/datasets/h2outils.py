import glob
import os,cv2
import numpy as np

from meshreg.models.utils import project_hand_3d2img, align_np_sample_for_cam2local,load_mano_mean_pose
import meshreg.datasets.vanilla_dataset as vd_utils
import lmdb

def remove_third_person_views():
    list_dirs=[]
    for i in [0,1,2,3]: 
        list_dirs+=(glob.glob('./*/*/*/cam{:d}/'.format(i)))
    print(list_dirs)

    with open("./rm.sh","w") as f:
        f.writelines("#!/bin/bash\n")
        for cline in list_dirs:
            f.writelines("echo \" rm -rf {:s} \"\n".format(cline))
            f.writelines("rm -rf {:s} \n".format(cline))


def remove_redundant_images():
    list_dirs=[]
    for i in [4]: 
        list_dirs+=(glob.glob('./*/*/*/cam{:d}/rgb'.format(i)))
        list_dirs+=(glob.glob('./*/*/*/cam{:d}/rgb256'.format(i)))
    print(list_dirs)

    with open("./rm.sh","w") as f:
        f.writelines("#!/bin/bash\n")
        for cline in list_dirs:
            f.writelines("echo \" rm -rf {:s} \"\n".format(cline))
            f.writelines("rm -rf {:s} \n".format(cline))

def convert_to_480_270_imgs():
    list_dirs=glob.glob('./*/*/*/cam*/')
    for cdir in list_dirs:
        print(cdir)
        out_dir=os.path.join(cdir,'rgb480_270')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        list_imgs=glob.glob(os.path.join(cdir,'rgb','*.png'))
        for im_id in range(0,len(list_imgs)):
            path_cimg=os.path.join(cdir,'rgb','{:06d}.png'.format(im_id))
            print(path_cimg)
            cimg=cv2.imread(path_cimg)
            assert cimg is not None, 'cimg shd not be None'
            cimg=cv2.resize(cimg,(480,270))
            cv2.imwrite(os.path.join(cdir,'rgb480_270','{:06d}.png'.format(im_id)),cimg)



def read_text(text_path, offset=0, half=0):
    with open(text_path, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))

    if half:
        data_list = np.array(data)[offset:half].tolist() + np.array(data)[half+offset:].tolist()
        return np.array(data_list).reshape((-1, 3)).astype(np.float32)
    else:
        return np.array(data)[offset:].reshape((-1, 3)).astype(np.float32)

def get_skeleton(path_hand_pose):
    return read_text(path_hand_pose,offset=1,half=64)

def get_cam_extr(path_extr):    
    with open(path_extr, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))
    return np.array(data).reshape((4,4))[:3].astype(np.float32)

def get_cam_intr(path_intr):
    with open(path_intr, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))
    intr=np.array([data[0],0,data[2],0,data[1],data[3],0,0,1]).reshape((3,3)).astype(np.float32)
    return intr



def check_consistent_camera_intrinsic(path_camera_intr,self_camera_intr):
    with open(path_camera_intr,"r") as f:
        text=f.readlines()[0].strip('\n').split(' ')
        ccam_intr=np.array([float(text[0]),0,float(text[2]),0,float(text[1]),float(text[3]),0,0,1]).reshape((3,3))
        assert np.fabs(ccam_intr-self_camera_intr).max()<1e-5,'cam intr not aligned'



def get_action_idx_to_tag():
    string_idx_tag="0 background\n1 grab book\n2 grab espresso\n3 grab lotion\n4 grab spray\n5 grab milk\n6 grab cocoa\n"
    string_idx_tag+="7 grab chips\n8 grab cappuccino\n9 place book\n10 place espresso\n11 place lotion\n12 place spray\n13 place milk\n"
    string_idx_tag+="14 place cocoa\n15 place chips\n16 place cappuccino\n17 open lotion\n18 open milk\n19 open chips\n20 close lotion\n"
    string_idx_tag+="21 close milk\n22 close chips\n23 pour milk\n24 take out espresso\n25 take out cocoa\n26 take out chips\n27 take out cappuccino\n"
    string_idx_tag+="28 put in espresso\n29 put in cocoa\n30 put in cappuccino\n31 apply lotion\n32 apply spray\n33 read book\n34 read espresso\n"
    string_idx_tag+="35 spray spray\n36 squeeze lotion"
    string_idx_tag=string_idx_tag.split('\n')
    
    list_idx_tag=[]
    for aid,tag in enumerate(string_idx_tag[1:]):
        action_name=' '.join(tag.split(' ')[1:])
        list_idx_tag.append(action_name)
    return list_idx_tag

def get_object_tag_to_idx():
    string_idx_tag="0 background\n1 book\n2 espresso\n3 lotion\n4 spray\n5 milk\n6 cocoa\n7 chips\n8 cappuccino"
    string_idx_tag=string_idx_tag.split('\n')
    dict_tag_idx={}
    for oid,tag in enumerate(string_idx_tag[1:]):
        object_name=tag.split(' ')[1]
        dict_tag_idx[object_name]=oid
    return dict_tag_idx

def get_verb_tag_to_idx():
    string_idx_tag="0 background\n1 grab\n2 place\n3 open\n4 close\n5 pour\n6 take out\n7 put in\n8 apply\n9 read\n10 spray\n11 squeeze"
    string_idx_tag=string_idx_tag.split('\n')
    dict_tag_idx={}
    for vid,tag in enumerate(string_idx_tag[1:]):
        vname=' '.join(tag.split(' ')[1:])
        dict_tag_idx[vname]=vid
    return dict_tag_idx

def load_action_taxonomy():    
    dict_action_info={}
    list_action_name=get_action_idx_to_tag()
    dict_object_name_to_idx=get_object_tag_to_idx()
    dict_verb_name_to_idx=get_verb_tag_to_idx()
    dict_action_name_to_idx={aname:aid for aid,aname in enumerate(list_action_name)}

    for action_idx, action_name in enumerate(list_action_name):
        object_name=action_name.split(' ')[-1]
        object_idx=dict_object_name_to_idx[object_name]
        verb_name=' '.join(action_name.split(' ')[:-1])
        verb_idx=dict_verb_name_to_idx[verb_name]

        dict_action_info[action_name]={'object_idx':object_idx,'object_name':object_name, 'verb_idx':verb_idx,'verb_name':verb_name,'action_idx':action_idx,}
        #print(action_name,dict_action_info[action_name])

        aname=action_name
    return {"action_info":dict_action_info,"object_name2idx":dict_object_name_to_idx,"verb_name2idx":dict_verb_name_to_idx,"action_name2idx":dict_action_name_to_idx}



def load_split_video_segments_dict(path_dataset='./'):
    frame_segments={'train':{},'val':{}, 'test':{}}
    list_action_name=get_action_idx_to_tag()
    for split_tag in frame_segments.keys():
        with open(os.path.join(path_dataset,'./action_labels/action_{:s}.txt'.format(split_tag)),'r') as f:
            segs=f.readlines()[1:]
            for cline in segs:
                cline=cline.strip('\n').split(' ')
                if split_tag=='test':
                    seg_idx,action_idx=int(cline[0]),int(cline[-1])
                    start_frame_idx,end_frame_idx=int(cline[2]),int(cline[3])
                else:
                    seg_idx,action_idx=int(cline[0]),int(cline[2])
                    start_frame_idx,end_frame_idx=int(cline[3]),int(cline[4])
                tag_subject,tag_scene,tag_sequence=cline[1].split('/')

                assert action_idx>0#action idx starts from 1
                action_name=list_action_name[action_idx-1]
                
                #segment_idx start from 0
                cinfo={'segment_idx':seg_idx-1,'subject':tag_subject,'scene':tag_scene,'sequence':tag_sequence,
                        'action_name':action_name,'start_frame':start_frame_idx,'end_frame':end_frame_idx,}

                if not (cline[1] in frame_segments[split_tag].keys()):
                    frame_segments[split_tag][cline[1]]={"action_segs":[]}
                frame_segments[split_tag][cline[1]]["action_segs"].append(cinfo)
    return frame_segments

def gather_video_meta_for_untrimmed_pose_seq(list_videos,root_rgb):
    meta_videos={}
    for vtag in list_videos:
        list_frames=glob.glob(os.path.join(root_rgb, vtag,"cam4/rgb480_270/*.png"))
        num_frames=len(list_frames)
        meta_videos[vtag]={"start_idx":0,"end_idx":num_frames}    
    return meta_videos




def get_seq_map(frame_segments,sample_infos,ntokens_pose, ntokens_action, spacing,is_shifting_window):
    window_starts=[]
    full = []
    
    for sample_idx, sample_info in enumerate(sample_infos):
        cseg=frame_segments[sample_info['seq_idx']]
        if sample_info["frame_idx"]==cseg["start_idx"]:        
            seq_count=0
            cur_seq_len=cseg["end_idx"]-cseg["start_idx"]+1
            
                
        if (not is_shifting_window and seq_count%(ntokens_action*spacing)<ntokens_pose*spacing) or \
            (is_shifting_window and seq_count%(ntokens_action*spacing)<spacing):
                window_starts.append(sample_idx)    
        full.append(sample_idx)
        seq_count += 1
    
    return window_starts, full




def convert_h2o_untrimmed_split_to_lmdb(split):
    root='../h2o/'
    fps=30     
    data_size=1024**3 

    frame_segments_with_action=load_split_video_segments_dict(root)
    frame_segments_with_action=frame_segments_with_action[split]     
    verbose=True

    if False:        
        palm_joints=[0,5,9,13,17] 
        mean_mano_palm_joints={'left': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT')),
                    'right': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT'))}


        dir_lmdb=os.path.join('../lmdbsv2/','h2o_pose0@30fps',split)
        print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
        if not os.path.exists(dir_lmdb):
            os.makedirs(dir_lmdb)
        env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)

        for vtag, vinfo in frame_segments_with_action.items():
            cam_intrs=[]
            for view_id in range(5):
                cam_intrs.append(get_cam_intr(os.path.join(root,vtag,f"cam{view_id}/cam_intrinsics.txt")))
                
                
            list_frames=glob.glob(os.path.join(root, vtag,"cam4/rgb480_270/*.png"))
            num_frames=len(list_frames)
            print(vtag,num_frames)
            
            subdb=env.open_db(vtag.encode('ascii'))#lmdb.open(dir_lmdb,map_size=data_size*10)
            txn=env.begin(db=subdb,write=True)
            
            for fid in range(0,num_frames):
                cam2worlds=[]#cam2world
                for view_id in range(5):
                    cam2worlds.append(get_cam_extr(os.path.join(root,vtag,f"cam{view_id}","cam_pose","{:06d}.txt".format(fid))))


                
                skels_in_cam=[]
                for view_id in range(5):
                    cskel=get_skeleton(os.path.join(root,vtag,f"cam{view_id}","hand_pose","{:06d}.txt".format(fid)))
                    skels_in_cam.append(cskel.reshape(42,3).astype(np.float64))
                
                skels_in_world,cam_info_to_save=[],[]
                for view_id in range(5):
                    cskel,cc2w=skels_in_cam[view_id],cam2worlds[view_id]
                    R_c2w,t_c2w=cc2w[:,:3].transpose(),cc2w[:,3:4].reshape((1,3))
                    
                    skel_w=np.dot(cskel,R_c2w)+t_c2w
                    skels_in_world.append(skel_w)


                    t_w2c=-np.matmul(t_c2w,R_c2w.transpose()) 
                    cextr=np.concatenate([R_c2w,t_w2c.reshape(3,1)],axis=1)
                    cam_info_to_save.append(np.concatenate([cextr.flatten(),cam_intrs[view_id].flatten()]))
                    
                    if verbose:
                        assert np.fabs(skels_in_world[-1]-skels_in_world[0]).max()<1e-5
                        world_joints_hom = np.concatenate([skels_in_world[-1], np.ones([skel_w.shape[0], 1])], 1)
                        ccextr=cam_info_to_save[-1][:12].reshape(3,4)
                        cam_joints=ccextr.dot(world_joints_hom.transpose()).transpose()[:, :3]
                        assert np.fabs(cam_joints-skels_in_cam[view_id]).max()<1e-5                       

                skel_in_world=skels_in_world[-1]
                hand_left,hand_right=skel_in_world[:21],skel_in_world[21:]

                palm={"left":hand_left[palm_joints],"right":hand_right[palm_joints]}
                key_byte ="{:06d}".format(fid).encode('ascii')
                data_world2local=align_np_sample_for_cam2local(palm,mean_mano_palm_joints,verbose=fid%1000==0)#in meter    

                data=np.concatenate([skel_in_world.flatten(),data_world2local.flatten(),np.array(cam_info_to_save).flatten()])
                txn.put(key_byte,data)                
            txn.commit()
        env.close()
    
    if False:
        dir_lmdb=os.path.join('../lmdbsv2/','h2o_action@30fps',split)
        
        if not os.path.exists(dir_lmdb):
            os.makedirs(dir_lmdb)
        env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)
        
        action_info=load_action_taxonomy()["action_info"]
        #print(action_info)
        for vtag, vinfo in frame_segments_with_action.items():
            list_frames=glob.glob(os.path.join(root, vtag,"cam4/rgb480_270/*.png"))
            num_frames=len(list_frames)
            #print(vtag,num_frames)
            
            subdb=env.open_db(vtag.encode('ascii'))
            txn=env.begin(db=subdb,write=True)    
            
            cur_info=[{'action_name':'NIL', 'object_name':'NIL','verb_name':'NIL','frame_idx':i} for i in range(0,num_frames)]            
            for seg_info in vinfo["action_segs"]:
                for fid in range(seg_info["start_frame"],seg_info["end_frame"]):
                    for key in ["action_name","verb_name","object_name"]:
                        assert cur_info[fid][key]=="NIL"

                    cur_info[fid]["action_name"]=seg_info["action_name"]
                    cur_info[fid]["object_name"]=action_info[seg_info["action_name"]]["object_name"]
                    cur_info[fid]["verb_name"]=action_info[seg_info["action_name"]]["verb_name"]
                    
            for fid in range(0,num_frames):
                key_byte ="{:06d}".format(fid).encode('ascii')
                data="*".join([cur_info[fid]['action_name'],cur_info[fid]["verb_name"],cur_info[fid]['object_name']]).encode('ascii')
                txn.put(key_byte,data)                
            txn.commit()
        env.close()


    if True:
        dir_lmdb=os.path.join('../lmdbsv2/','h2o_imgs0@30fps',split)
        print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
        if not os.path.exists(dir_lmdb):
            os.makedirs(dir_lmdb)
        
        env = lmdb.open(dir_lmdb,map_size=data_size*50,max_dbs=10000)
        for view_tag in ["cam0","cam1","cam2","cam3","cam4"]:
            for vtag, vinfo in frame_segments_with_action.items():
                list_frames=glob.glob(os.path.join(root, vtag,f"{view_tag}/rgb480_270/*.png"))
                num_frames=len(list_frames)
                print(vtag,num_frames)
                
                subdb=env.open_db((vtag+"_"+view_tag).encode('ascii'))
                txn=env.begin(db=subdb,write=True)
                
                for fid in range(0,num_frames):
                    key_byte ="{:06d}".format(fid).encode('ascii')
                    cimg=cv2.imread(os.path.join(root,vtag,view_tag,"rgb480_270/{:06d}.png".format(fid)))
                    data=cv2.imencode('.jpg',cimg)[1]
                    txn.put(key_byte,data)     
                txn.commit()
        env.close()
        
        
def visualize_per_frame_label_for_untrimmed_videos(untrimmed_video_infos, env_pose, env_action, env_img, hand_links, dir_out_videos='./vis_v3/'): 
    for seq_id, file_tag in enumerate(untrimmed_video_infos.keys()):        
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        path_out=os.path.join(dir_out_videos,"_".join(file_tag.split("/"))+".avi")
        videoWriter = cv2.VideoWriter(path_out, fourcc, 30, (480,270))  


        view_id=4

        subdb_img=env_img.open_db((file_tag+f"_cam{view_id}").encode('ascii'),create=False)
        txn_img=env_img.begin(write=False,db=subdb_img)
        
        subdb_pose=env_pose.open_db(file_tag.encode('ascii'),create=False)
        txn_pose=env_pose.begin(write=False,db=subdb_pose)
        
        subdb_action=env_action.open_db(file_tag.encode('ascii'),create=False)
        txn_action=env_action.begin(write=False,db=subdb_action)
        for frame_id in range(0,untrimmed_video_infos[file_tag]["end_idx"]):
            buf=txn_img.get("{:06d}".format(frame_id).encode('ascii'))  
            raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
            img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            out_img=img.copy()
            
            buf=txn_pose.get("{:06d}".format(frame_id).encode('ascii'))
            raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
            hand_poses_world=raw_pose[:42*3].reshape((42,3)).astype(np.float32)
            
                        
            cam_raw=raw_pose[42*3+24:]
            cam_raw=cam_raw[view_id*(12+9):(view_id+1)*(12+9)]
            cam_extr,cam_intr=cam_raw[:12].reshape((3,4)),cam_raw[12:].reshape((3,3))

            reduce_factor= 480 / 1280.0
            cam_intr=cam_intr.copy()
            cam_intr[:2] = cam_intr[:2] *reduce_factor  

            hand_poses_hom = np.concatenate([hand_poses_world, np.ones([hand_poses_world.shape[0], 1])], 1)
            hand_poses_cam=cam_extr.dot(hand_poses_hom.transpose()).transpose()[:, :3].astype(np.float32)


           
            #hand_poses_cam=hand_poses_world
            chand_2d=project_hand_3d2img(hand_poses_cam*1000,cam_intr,None)
            
            for l in hand_links:
                for j in range(0,len(l)-1):
                    cv2.line(out_img,(int(chand_2d[l[j]][0]),int(chand_2d[l[j]][1])),(int(chand_2d[l[j+1]][0]),int(chand_2d[l[j+1]][1])), (0,255,0),2)
                    cv2.line(out_img,(int(chand_2d[l[j]+21][0]),int(chand_2d[l[j]+21][1])),(int(chand_2d[l[j+1]+21][0]),int(chand_2d[l[j+1]+21][1])), (0,0,255),2)


            buf=txn_action.get("{:06d}".format(frame_id).encode('ascii')).decode('ascii')
            terms=buf.split('*')
            caction_info=f"action {terms[0]};verb{terms[1]}+ obj {terms[2]}"
            cv2.putText(out_img,f'{caction_info}',(20,30),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)


            videoWriter.write(out_img)
        videoWriter.release()
        exit(0)