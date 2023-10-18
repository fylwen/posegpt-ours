import glob,copy
import os,cv2
import numpy as np
import pandas as pd
import json,lmdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from meshreg.models.utils import project_hand_3d2img, align_np_sample_for_cam2local,load_mano_mean_pose
import meshreg.datasets.vanilla_dataset as vd_utils

def plot_hand_3d(trj,frame_id,ax3d,title):
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    
    ax3d.set_title(title,fontsize=4,pad=0.)

    link=[[0, 2, 3, 4],[0, 5, 6, 7, 8],[0, 9, 10, 11, 12],[0, 13, 14, 15, 16],[0, 17, 18, 19, 20]]
    cs=[(0,1.,0),(0,0,0),(1.,0,0),(1.,1.,0),(1.,0,1.)]
    for lid, l in enumerate(link):
        ax3d.plot(trj[frame_id,l,0],trj[frame_id,l,2],trj[frame_id,l,1],alpha=0.8,c=cs[lid],linewidth=2)#0.5)


def vis_locally_aligned_hand_3d(cframe_hand,cframe_palm,cframe_mano,title,path_tosave):
    fig = plt.figure(figsize=(1,1))
    ax3d=plt.subplot(1,1,1, projection='3d')
    plot_hand_3d(cframe_hand.reshape((-1,21,3)),0,ax3d,title=title)

    ax3d.scatter(cframe_palm[:,0],cframe_palm[:,2],cframe_palm[:,1],c='b',s=5)
    ax3d.scatter(cframe_mano[:,0],cframe_mano[:,2],cframe_mano[:,1],c='c',s=5)
   
    fig.savefig(path_tosave, dpi=500)
    plt.close(fig)
    
    
def get_view_id2tag():
    return {'e1':['HMC_84346135_mono10bit','HMC_21176875_mono10bit'],
                'e2':['HMC_84347414_mono10bit','HMC_21176623_mono10bit'],
                'e3':['HMC_84355350_mono10bit','HMC_21110305_mono10bit'],
                'e4':['HMC_84358933_mono10bit','HMC_21179183_mono10bit'],
                'v1':['C10095_rgb'], 'v2':['C10115_rgb'],
                'v3':['C10118_rgb'], 'v4':['C10119_rgb'],
                'v5':['C10379_rgb'], 'v6':['C10390_rgb'],
                'v7':['C10395_rgb'], 'v8':['C10404_rgb']}
def get_view_tag2id():
    id2tag=get_view_id2tag() 
    dict_tag2id={}
    for k, vs in id2tag.items():
        for v in vs:
            dict_tag2id[v]=k
    return dict_tag2id
    
def get_view_extrinsic_intrisic(path_calib_txt='./annotations/calib.txt'):
    tag_to_camera={}
    with open(path_calib_txt,'r') as f:
        contents=f.readlines()
         
        for vid in range(0,12):
            pt=vid*9
            name=contents[pt].strip('\n')
            c_intr=[]
            for i in range(0,3):
                pt+=1
                c_intr+=[float(n) for n in contents[pt].strip('\n').split(' ')]
            
            pt+=1
            c_extr=[]
            for i in range(0,3):
                pt+=1                
                extr=contents[pt].strip('\n').split(' ')
                c_extr+=[float(extr[i]) for i in range(0,4)]

            c_intr=np.array(c_intr).reshape((3,3))
            c_intr[:2]*=480/1920.
            tag_to_camera[name]={'intr': c_intr, 'extr':np.array(c_extr).reshape((3,4))}

    return tag_to_camera

 
                
def load_action_taxonomy(path_tax='./annotations/coarse-annotations/actions.csv',
                path_tail_tax='./annotations/coarse-annotations/tail_actions.txt',
                reverse_head_tail=False):
    with open(path_tail_tax,'r') as f:
        _ori_list=f.readlines()
    
    list_tail_tax=[]
    list_head_tax=[]
    for tail_act in _ori_list:
        list_tail_tax.append(tail_act.strip('\n'))
        
    tax=pd.read_csv(path_tax)
    tax_size=tax.shape[0]
    dict_action_info={}
    dict_verb_name2idx={}
    dict_noun_name2idx={}

    for id in range(0,tax_size):
        citem=tax.iloc[id]
        dict_action_info[citem["action_cls"]]={"action_id":int(citem["action_id"]), "verb_id":int(citem["verb_id"]),"verb_name":citem["verb_cls"], "object_id":int(citem["noun_id"]),"object_name":citem["noun_cls"]}
        if not citem["action_cls"] in list_tail_tax:
            list_head_tax.append(citem["action_cls"])
        
        if not(citem["verb_cls"] in dict_verb_name2idx.keys()):
            dict_verb_name2idx[citem["verb_cls"]]=int(citem["verb_id"])
        else:
            assert dict_verb_name2idx[citem["verb_cls"]]==int(citem["verb_id"]), 'inconsistent for '+citem["verb_cls"]
        
        if not (citem["noun_cls"] in dict_noun_name2idx.keys()):
            dict_noun_name2idx[citem["noun_cls"]]=int(citem["noun_id"])
        else:
            assert dict_noun_name2idx[citem["noun_cls"]]==int(citem["noun_id"]), 'inconsistent for '+citem["noun_cls"]
    
    #sort to follow the ascending order for idx
    list_action_info=sorted(dict_action_info.items(),key=lambda item:item[1]["action_id"])
    list_verb_name2idx=sorted(dict_verb_name2idx.items(),key=lambda item:item[1])
    list_noun_name2idx=sorted(dict_noun_name2idx.items(),key=lambda item:item[1])

    dict_action_info,dict_verb_name2idx,dict_noun_name2idx,dict_action_name2idx={},{},{},{}
    for (k,v) in list_action_info:
        dict_action_info[k]=v
        dict_action_name2idx[k]=v["action_id"]
    for (k,v) in list_verb_name2idx:
        dict_verb_name2idx[k]=v
    for (k,v) in list_noun_name2idx:
        dict_noun_name2idx[k]=v

    results={'action_info':dict_action_info,#key: action_name
            'action_name2idx':dict_action_name2idx,
            'verb_name2idx':dict_verb_name2idx,
            'object_name2idx':dict_noun_name2idx,}

    if reverse_head_tail:
        results['list_head_tax'],results['list_tail_tax']=list_tail_tax,list_head_tax
    else:
        results['list_head_tax'],results['list_tail_tax']=list_head_tax,list_tail_tax
        
    return results
    
def load_split_coarse_info(dir_split_files='./annotations/coarse-annotations/coarse_splits/',
                        dir_coarse_labels='./annotations/coarse-annotations/coarse_labels/',
                        fps_to_load=60,
                        split='train', verbose=False):
    dict_video_info={}
    for bg_act in ["disassembly","assembly"]:
        path_split_file=os.path.join(dir_split_files,'{:s}_coarse_{:s}.txt'.format(split,bg_act))
        with open(path_split_file,'r') as f:
            list_videos=f.readlines()
        for line in list_videos:
            line=line.strip('\n').split('\t')
            
            #for each dis/ass video, read meta
            if split=='train':
                file_name,is_shared,toy_id,toy_name=line[0], True,line[3],line[4]
            elif split=='val':
                file_name,is_shared,toy_id,toy_name=line[0], line[2]=='shared',line[3],line[4]
            else:#test
                file_name,is_shared,toy_id,toy_name=line[0], line[2]=='shared','-','-'

            file_tag=file_name.split('.')[0]
            csegs=[]
            
            #load coarse segmentation
            if split!='test':
                with open(os.path.join(dir_coarse_labels,file_name),'r') as fv:
                    text_actions=fv.readlines()
                    
                    for info_ca in text_actions:
                        info_ca=info_ca.strip('\n').split('\t')
                        
                        for idx in [0,1]:
                            info_ca[idx]=int(info_ca[idx])*(fps_to_load//30)
                        csegs.append({'start_frame':info_ca[0],'end_frame':info_ca[1],'action_name':info_ca[2]})# 
                        
            dict_video_info[file_tag]=csegs
    # key: disassembly_nusar-2021_action_both_9033-a30_9033_user_id_2021-02-04_131528
    # value: list of segs [{'start_frame': 10655, 'end_frame': 13096, 'action_name': 'inspect toy'}]
    return dict_video_info
    
def load_split_video_tags(dir_split_files='./annotations/fine-grained-annotations',split='train'):
    split = 'validation' if split=='val' else split

    segs=pd.read_csv(os.path.join(dir_split_files,split+'.csv'))
    segs_size=segs.shape[0]
    print("hello segs_size",os.path.join(dir_split_files,split+'.csv'),segs_size)
    set_video_tags=set()
    for id in range(0,segs_size):
        citem=segs.iloc[id]  
        path_video=citem['video']
        cfile_tag,view_tag=path_video.split('/')
        set_video_tags.add(cfile_tag)
    
    return set_video_tags

def load_split_fine_grained_info(dir_split_files='./annotations/fine-grained-annotations', view_id=None, fps_to_load=60, split='train', verbose=False,dict_action_info=None):
    split = 'validation' if split=='val' else split
    segs=pd.read_csv(os.path.join(dir_split_files,split+'.csv'))

    segs_size=segs.shape[0]
    dict_video_info={}
    dict_existed_seq={}
    is_segid_counted=np.zeros((segs_size,),dtype=np.bool)

    view_id2tag=get_view_id2tag()
    if view_id is not None:
        valid_view_tags=view_id2tag[view_id]
    for id in range(0,segs_size):
        citem=segs.iloc[id] 
        if view_id is not None:
            cview_tag=citem["video"].split('/')[1].split(".")[0]
            if not cview_tag in valid_view_tags:
                continue


        if is_segid_counted[citem['id']]:
            continue
        is_segid_counted[citem['id']]=True

        path_video=citem['video']
        cfile_tag,view_tag=path_video.split('/')
        if cfile_tag not in dict_video_info.keys():
            dict_video_info[cfile_tag]=[]
            dict_existed_seq[cfile_tag]=set()
            

        #process fine-grained action item
        if split in ['train','validation']:
            cdict={'seq_id':int(citem['id']), 'video_tag':cfile_tag,
                'start_frame':int(citem['start_frame' if 'start_frame' in citem.keys() else 'start']),
                'end_frame':int(citem['end_frame' if 'end_frame' in citem.keys() else 'end']),
                'action_name':citem['action_cls'],
                'action_id':int(citem['action_id' if 'action_id' in citem.keys() else 'action']),
                'verb_name':citem['verb_cls'],
                'verb_id':int(citem['verb_id' if 'verb_id' in citem.keys() else 'verb']),
                'object_name':citem['noun_cls'],
                'object_id':int(citem['noun_id' if 'noun_id' in citem.keys() else 'noun'])}#,'is_shared': citem['is_shared']==1
            
            if verbose and dict_action_info is not None:
                ditem=dict_action_info['action_info'][citem['action_cls']]
                for kk in ['action_id','verb_id','verb_name','object_id','object_name']:
                    assert ditem[kk]==cdict[kk], '{:d} {:s} has inconsistant {:s}'.format(id,cfile_tag, kk)
                for kk in ['action','verb','object']:
                    assert dict_action_info['{:s}_name2idx'.format(kk)][cdict['{:s}_name'.format(kk)]]==cdict['{:s}_id'.format(kk)], '{:d} {:s} has inconsistant {:s}'.format(id,cfile_tag, kk)
                 
        else:#split in ['test','test_challenge']
            cdict={'seq_id':int(citem['id']), 'start_frame':int(citem['start_frame']),'end_frame':int(citem['end_frame'])}#, 'is_shared': citem['is_shared']==1}

        cdict["start_frame"]*=(fps_to_load//30)
        cdict["end_frame"]*=(fps_to_load//30)

        if (cdict['start_frame'],cdict['end_frame'],cdict['action_id']) in dict_existed_seq[cfile_tag]:
            continue
        
        
        #if cdict['start_frame']==cdict['end_frame']:
        #    continue
        dict_existed_seq[cfile_tag].add((cdict['start_frame'],cdict['end_frame'],cdict['action_id']))
        dict_video_info[cfile_tag].append(cdict)
        
    #key: nusar-2021_action_both_9033-a30_9033_user_id_2021-02-04_131528
    #value: list of segs
    return dict_video_info

def save_split_num_pose_frames_at_60fps(task_seqs_info,dir_poses,split):
    assert not os.path.exists(os.path.join(dir_poses,f"{split}.txt"))
    dict_to_save={}    
    for seq_id, (file_tag,_) in enumerate(task_seqs_info.items()):        
        print(seq_id,file_tag)
        if not os.path.exists(os.path.join(dir_poses,file_tag+'.json')):
            continue
        with open(os.path.join(dir_poses,file_tag+'.json')) as f:
            pose_data=json.load(f)
            dict_to_save[file_tag]=len(pose_data)

    
    with open(os.path.join(dir_poses,f"{split}.txt"),"w") as f:
        for k,v in dict_to_save.items():
            f.write(f"{k} {v}\n")
        f.close()

def gather_video_meta_for_untrimmed_pose_seq(task_seqs_info,fps_to_load,path_split_num_pose_frames_at_60fps,max_samples=-1):    
    total_frames=0
    set_err_no_pose_video=set()
    untrimmed_video_info={}
    pose_fps=60

    with open(path_split_num_pose_frames_at_60fps,"r") as f:
        list_file_pose=f.readlines()
    dict_file_pose_60fps={}
    for line in list_file_pose:
        k,v=line.strip('\n').split(' ')
        dict_file_pose_60fps[k]=int(v)

    for seq_id, (file_tag,_) in enumerate(task_seqs_info.items()): 
        print(seq_id,file_tag)
        if file_tag not in dict_file_pose_60fps.keys(): 
            set_err_no_pose_video.add(file_tag)
            continue
            
        len_pose_data=dict_file_pose_60fps[file_tag] 
        num_frames=len_pose_data//(60//30)*(60//30)
        num_frames=num_frames//(pose_fps//fps_to_load)
        
        untrimmed_video_info[file_tag]={'start_idx':0,'end_idx':num_frames}
        total_frames+=num_frames 

        if max_samples>0 and total_frames>max_samples:
            break 

    #Note that: fine grained action has overlap, and some video has only partial frames 
    return {'untrimmed_video_info':untrimmed_video_info}





    

def visualize_per_frame_label_for_seq(imgs,hand_poses_cam,action_infos,camera_info,fps_to_load,path_out,hand_links):
    cur_frame=imgs[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    videoWriter = cv2.VideoWriter(path_out, fourcc, 10, (cur_frame.shape[1],cur_frame.shape[0]))  
    num_frames=len(imgs)
    for img_id in range(0,num_frames):#seq_images.shape[0]-1,-1,-1):

        out_img=imgs[img_id]
        cv2.putText(out_img,'Ori timestamp: {:.2f} sec'.format(img_id/fps_to_load),(20,40),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
        
        if img_id in action_infos.keys():
            caction=action_infos[img_id]
            #cv2.putText(out_img,'coarse :'+caction[0],(50,60),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
            for pt, info in enumerate(caction):            
                cv2.putText(out_img,f'{info}',(20,60+pt*20),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
                
        if img_id in hand_poses_cam.keys():
            chand_2d=project_hand_3d2img(hand_poses_cam[img_id]*1000,camera_info["intr"],camera_info["extr"])
            #chand_2d*=480/1920. 
            for l in hand_links:
                for j in range(0,len(l)-1):
                    try:
                        cv2.line(out_img,(int(chand_2d[l[j]][0]),int(chand_2d[l[j]][1])),(int(chand_2d[l[j+1]][0]),int(chand_2d[l[j+1]][1])), (0,255,0),2)
                        cv2.line(out_img,(int(chand_2d[l[j]+21][0]),int(chand_2d[l[j]+21][1])),(int(chand_2d[l[j+1]+21][0]),int(chand_2d[l[j+1]+21][1])), (0,0,255),2)
                    except:
                        assert False
                        cv2.line(out_img,(10,10),(100,100), (255,255,255),5)
                        
        videoWriter.write(out_img)
    videoWriter.release()



def visualize_per_frame_label_for_untrimmed_videos(untrimmed_video_infos, env_pose, env_action, env_img,
                                fps_to_load,hand_links, dir_out_videos='./vis_v3/'): 
    view_tag='C10118_rgb'
    view_id = "v3"
    cam_for_view_tag=get_view_extrinsic_intrisic(path_calib_txt=os.path.join('../ass101/annotations/calib.txt'))
    if "HMC" in view_tag:
        camera_info={"extr":None}
    else:
        camera_info=cam_for_view_tag[view_tag]

    print(len(untrimmed_video_infos))


    for seq_id, file_tag in enumerate(untrimmed_video_infos.keys()):        
        subdb_pose=env_pose.open_db(file_tag.encode('ascii'),create=False)#tag_pose_dataset+"_"+file_tag
        txn_pose=env_pose.begin(write=False,db=subdb_pose)
        hand_poses_cam={}

        start_idx,end_idx=untrimmed_video_infos[file_tag]["start_idx"],untrimmed_video_infos[file_tag]["end_idx"]
        end_idx=min(end_idx,60*100)
        
        
        print(untrimmed_video_infos[file_tag])
        for fid in range(start_idx,end_idx):
            frame_id=fid*(60//fps_to_load)
            buf=txn_pose.get("{:06d}".format(frame_id).encode('ascii'))
            raw_pose=np.frombuffer(buf,dtype=np.float64).reshape(-1,)
            raw_joints3d=raw_pose[:42*3].reshape((42,3)).astype(np.float32)
            
            hand_poses_cam[fid]=raw_joints3d
            
        #read img@30fps        
        imgs=[]
        subdb_img=env_img.open_db(f"{file_tag}_{view_id}".encode('ascii'),create=False)
        txn_img=env_img.begin(write=False,db=subdb_img)
        for fid in range(start_idx,end_idx):
            if fps_to_load<30:
                frame_id=fid*(30//fps_to_load)
            else:
                frame_id=fid//(fps_to_load//30)            
            
            try:
                buf=txn_img.get("{:06d}".format(frame_id).encode('ascii'))        
                raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
                img=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
            except:
                print(fid,start_idx,end_idx)
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
            
            action_infos[fid]=caction_info
            
        if not os.path.exists(dir_out_videos):
            os.makedirs(dir_out_videos)

        path_out=os.path.join(dir_out_videos,f"{file_tag}_{view_tag}.avi")
        visualize_per_frame_label_for_seq(imgs,hand_poses_cam,action_infos,camera_info,fps_to_load,path_out,hand_links)

        exit(0)

    return


def check_video_list(path_coarse_seq_views='./annotations/coarse-annotations/coarse_seq_views.txt',dir_videos='./videos/'):
    #load txt
    with open(path_coarse_seq_views,'r') as f:
        list_videos=f.readlines()
        for pv in list_videos:
            if not ('HMC' in pv):
                continue
            pv='_'.join(pv.strip('\n').split('_')[1:])

            if not os.path.exists(os.path.join(dir_videos,pv)):
                print(pv)


def reduce_ass101_3rdp_video(dir_videos,ratio=4):
    list_videos=glob.glob(os.path.join(dir_videos,"*/HMC_*.mp4"))
    for cpath in list_videos:
        sin=cv2.VideoCapture(cpath)

        ori_w,ori_h=int(sin.get(cv2.CAP_PROP_FRAME_WIDTH)),int(sin.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if ori_w==480 or ori_h==270:
            print("skip",cpath, ori_w,ori_h)
            sin.release()
            continue
        
        print(">>>>>>hello",cpath)
        fps, num_frames=int(sin.get(cv2.CAP_PROP_FPS)),int(sin.get(cv2.CAP_PROP_FRAME_COUNT)) 
        fourcc= cv2.VideoWriter_fourcc(*"mp4v") 
        cur_w,cur_h=int(ori_w//ratio),int(ori_h//ratio)
        print(cur_w,cur_h)

        sout = cv2.VideoWriter("out.mp4", fourcc, fps, (cur_w,cur_h))  


        for i in range(0,num_frames):
            if i%(fps*60)==0:
                print(cpath,"Frame #",i)
            success, frame=sin.read()
            frame=cv2.resize(frame,(cur_w,cur_h))
            sout.write(frame)
        sin.release()
        sout.release()

        os.rename("out.mp4",cpath)


def check_consistent_train_list(path_to_ref,path_to_query,query_is_train):
    #for query set, check whether in ref set or not.

    if path_to_ref[-3:]=="csv":
        ref_segs=pd.read_csv(path_to_ref)

        ref_segs_size=ref_segs.shape[0]    
        path_ref_videos=set()
        for id in range(0,ref_segs_size):
            citem=ref_segs.iloc[id]
            path_video=citem['video'].split('/')[0]
            path_ref_videos.add(path_video)
    else:
        path_to_refs= ['../ass101/annotations/coarse-annotations/coarse_splits/train_coarse_assembly.txt',\
                    '../ass101/annotations/coarse-annotations/coarse_splits/train_coarse_disassembly.txt']
        for path_to_ref in path_to_refs:
            with open(path_to_ref,'r') as f:
                list_videos=f.readlines()
            path_ref_videos=set()
            for line in list_videos:
                line=line.strip('\n').split('\t')
                file_tag=line[0].split('.')[0]
                file_tag='_'.join(file_tag.split('_')[1:])  
                path_ref_videos.add(file_tag)

    query_segs=pd.read_csv(path_to_query)
    query_segs_size=query_segs.shape[0]
    for id in range(0,query_segs_size):
        citem=query_segs.iloc[id]
        path_video=citem['video'].split('/')[0] 
        try:
            assert (path_video in path_ref_videos) ==query_is_train
        except:
            print("#",id,citem['video'].split('/')[0])
            print(citem)
            exit(0)
 


#below are for lmdb

def aggregate_action_info(root, split, fps=30,verbose=False):
    coarse_action_taxonomy=load_action_taxonomy(path_tax=os.path.join(root, './annotations/coarse-annotations/actions.csv'),
                                                                path_tail_tax=os.path.join(root,'./annotations/coarse-annotations/tail_actions.txt'),
                                                                reverse_head_tail=False)
    fine_grained_action_taxonomy=load_action_taxonomy(path_tax=os.path.join(root, './annotations/fine-grained-annotations/actions.csv'),
                                                                path_tail_tax=os.path.join(root,'./annotations/fine-grained-annotations/head_actions.txt'),
                                                                reverse_head_tail=True)    
    anticipate_action_taxonomy=load_action_taxonomy(path_tax=os.path.join(root, './annotations/anticipation-annotations/CSVs/actions.csv'),
                                                                path_tail_tax=os.path.join(root,'./annotations/fine-grained-annotations/head_actions.txt'),
                                                                reverse_head_tail=True) 

    videos_coarse_action_info=load_split_coarse_info(dir_split_files=os.path.join(root,'./annotations/coarse-annotations/coarse_splits'),
                                                            dir_coarse_labels=os.path.join(root,'./annotations/coarse-annotations/coarse_labels/'),
                                                            fps_to_load=fps,split=split, verbose=verbose)    
    videos_fine_grained_action_info=load_split_fine_grained_info(dir_split_files=os.path.join(root,'./annotations/fine-grained-annotations'),
                                                        split=split, fps_to_load=fps,verbose=verbose,dict_action_info=fine_grained_action_taxonomy)
    videos_anticipate_action_info=load_split_fine_grained_info(dir_split_files=os.path.join(root,'./annotations/anticipation-annotations/CSVs/'),
                                                        split=split, fps_to_load=fps, verbose=verbose,dict_action_info=anticipate_action_taxonomy)

    return coarse_action_taxonomy, videos_coarse_action_info,videos_fine_grained_action_info,videos_anticipate_action_info


def aggregate_video_action_info(file_tag,cvideo_coarse_segs, cvideo_fine_segs,cvideo_anticipate_segs,coarse_action_taxonomy,start_video,end_video,verbose=True):
    cur_info=[{'coarse_action_name':'NIL', 'coarse_verb_name':'NIL','coarse_object_name':'NIL', 'frame_idx':i,\
                    'fine_grained_action_name':[], 'fine_grained_verb_name':[],'fine_grained_object_name':[], \
                    'anticipate_action_name':[], 'anticipate_verb_name':[],'anticipate_object_name':[]} for i in range(0,end_video)]            
    #Coarse        
    for c_seg in cvideo_coarse_segs:
        cact_info=coarse_action_taxonomy['action_info'][c_seg['action_name']]
        for frame_id in range(c_seg['start_frame'],c_seg['end_frame']):                    
            if frame_id>=end_video:
                break
            if verbose:
                assert cur_info[frame_id]['coarse_action_name']=='NIL', 'detected overlap coarse labels for frame {:d}//{:d}, {:d}'.format(frame_id,end_video,c_seg['start_frame'])

            cur_info[frame_id]['coarse_action_name']=c_seg['action_name']
            cur_info[frame_id]['coarse_verb_name']=cact_info['verb_name']
            cur_info[frame_id]['coarse_object_name']=cact_info['object_name']
    #Fine grained
    for fg_seg in cvideo_fine_segs:
        for frame_id in range(fg_seg['start_frame'],fg_seg['end_frame']+1):
            if frame_id>=end_video:
                break
            if fg_seg['action_name'] not in cur_info[frame_id]['fine_grained_action_name']:
                cur_info[frame_id]['fine_grained_action_name'].append(fg_seg['action_name'])
                cur_info[frame_id]['fine_grained_verb_name'].append(fg_seg['verb_name'])
                cur_info[frame_id]['fine_grained_object_name'].append(fg_seg['object_name'])
    #anticipated
    for fg_seg in cvideo_anticipate_segs:
        for frame_id in range(fg_seg['start_frame'],fg_seg['end_frame']+1):
            if frame_id>=end_video:
                break                    
            if fg_seg['action_name'] not in cur_info[frame_id]['anticipate_action_name']:
                cur_info[frame_id]['anticipate_action_name'].append(fg_seg['action_name'])
                cur_info[frame_id]['anticipate_verb_name'].append(fg_seg['verb_name'])
                cur_info[frame_id]['anticipate_object_name'].append(fg_seg['object_name'])

    strs_to_return={}
    for fid in range(start_video,end_video):
        cdata=cur_info[fid]
        assert fid==cur_info[fid]["frame_idx"]                
        data_strs=[]
        for k in ["coarse_action_name","coarse_verb_name","coarse_object_name"]:
            data_strs.append(cdata[k])
        for k in ["fine_grained_action_name", "fine_grained_verb_name","fine_grained_object_name",\
                        "anticipate_action_name", "anticipate_verb_name","anticipate_object_name"]:
            data_strs.append('+'.join(cdata[k]) if len(cdata[k])>0 else 'NIL')
        
        #print(data_strs)
        data_str=('*'.join(data_strs))
        strs_to_return["{:06d}".format(fid)]=data_str
    
    return strs_to_return



def convert_ass101_action_split_to_lmdb(split):
    verbose=True
    root="../ass101/"
    dir_lmdb=os.path.join('../lmdbsv2/','ass101_action@30fps',split)
    fps=30 
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    set_err_truncated_video=set()
    set_err_no_pose_video=set()

    
    dir_poses=os.path.join(root,'./poses@60fps/')
    pose_fps=60

    data_size=1024**3 

    env = lmdb.open(dir_lmdb,map_size=data_size,max_dbs=1000)
    
    #Warning: I didnt check the code after re-organizing it
    coarse_action_taxonomy, videos_coarse_action_info,videos_fine_grained_action_info,videos_anticipate_action_info=aggregate_action_info(root,split,fps,verbose=verbose)

    for seq_id, (file_tag,fine_segs) in enumerate(videos_fine_grained_action_info.items()):
        print(seq_id,file_tag)
 
        if not os.path.exists(os.path.join(dir_poses,file_tag+'.json')):
            set_err_no_pose_video.add(file_tag)
            continue
        with open(os.path.join(dir_poses,file_tag+'.json')) as f:
            pose_data=json.load(f)       
            num_frames=len(pose_data)//(pose_fps//fps)

        cvideo_coarse_segs=[]
        for bg_act in ["disassembly","assembly"]:
            coarse_key=bg_act+'_'+file_tag
            if coarse_key in videos_coarse_action_info.keys():
                cvideo_coarse_segs+=videos_coarse_action_info[coarse_key]

        strs_to_return=aggregate_video_action_info(file_tag,cvideo_coarse_segs=cvideo_coarse_segs,
                                    cvideo_fine_segs=fine_segs,
                                    cvideo_anticipate_segs=videos_anticipate_action_info[file_tag],
                                    coarse_action_taxonomy=coarse_action_taxonomy,  
                                    start_video=0,end_video=num_frames)


        subdb=env.open_db((file_tag).encode('ascii'))#lmdb.open(dir_lmdb,map_size=data_size*10)
        txn=env.begin(db=subdb,write=True)
 

        for k, v in strs_to_return.items():
            key_byte=k.encode('ascii')
            data_str=v.encode('ascii')
            txn.put(key_byte,data_str)            
        txn.commit()
    env.close() 

    env_r=lmdb.open(dir_lmdb,readonly=True,lock=False,readahead=False,meminit=False, map_size=1024**3,max_spare_txns=32,max_dbs=1000)
    subdb=env_r.open_db((list(videos_fine_grained_action_info.keys())[0]).encode('ascii'),create=False)
    txn=env_r.begin(write=False,db=subdb)
    test=txn.get("{:06d}".format(100).encode('ascii')).decode('ascii')
    print(test) 


def convert_ass101_pose_split_to_lmdb(split):
    root='../ass101/'
    palm_joints=[0,5,9,13,17] 
    mean_mano_palm_joints={'left': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('LEFT')),
                'right': load_mano_mean_pose('./assets/mano/MANO_{:s}.pkl'.format('RIGHT'))}
 
    set_video_tags=load_split_video_tags(dir_split_files=os.path.join(root,'./annotations/fine-grained-annotations'),
                                                                    split=split)

    data_size_per_frame= np.zeros((150,),dtype=np.float64).nbytes
    data_size=data_size_per_frame*100000000

    dir_lmdb=os.path.join('../lmdbsv2/','ass101_pose@60fps',split)
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)
    set_err_no_pose_video=set()
    for seq_id, file_tag in enumerate(set_video_tags):        
        if not os.path.exists(os.path.join(root,'poses@60fps',file_tag+'.json')):
            set_err_no_pose_video.add(file_tag)
            continue
        with open(os.path.join(root,'poses@60fps',file_tag+'.json')) as f:
            pose_data=json.load(f)       
            fps,num_frames=60,len(pose_data)
        print(file_tag,num_frames)
        
        subdb=env.open_db((file_tag).encode('ascii'))
        txn=env.begin(db=subdb,write=True)        
        for fid in range(0,num_frames):
            joint_idx_proj=[5,20,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4]
            hand_left=np.array(pose_data[fid]["landmarks"]["0"])[joint_idx_proj]/1000
            hand_right=np.array(pose_data[fid]["landmarks"]["1"])[joint_idx_proj]/1000#mm to m

            palm={"left":hand_left[palm_joints],"right":hand_right[palm_joints]}

            key_byte = "{:06d}".format(fid).encode('ascii')
            data_in_world = np.concatenate((hand_left,hand_right),axis=0).astype(np.float64)#in m
            data_world2local=align_np_sample_for_cam2local(palm,mean_mano_palm_joints,verbose=fid%1000==0)# in meter

            data=np.concatenate([data_in_world.flatten(),data_world2local.flatten()])
            txn.put(key_byte,data)
        txn.commit()
    env.close()


    
def convert_ass101_video_split_to_lmdb(split,view_tag="C10118_rgb"):
    root='../ass101/'
    dir_videos=os.path.join(root, 'videos')
    fps_to_load=30
 
    set_video_tags=load_split_video_tags(dir_split_files=os.path.join(root,'./annotations/fine-grained-annotations'),split=split)

    data_size_per_frame= np.zeros((480,270,3),dtype=np.uint8).nbytes
    data_size=data_size_per_frame*(1024**2)

    view_id=get_view_tag2id()[view_tag]

    dir_lmdb=os.path.join(f'../lmdbsv2/ass101_{view_id}_imgs@30fps',split)
    print('check exist',dir_lmdb,os.path.exists(dir_lmdb))
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*8,max_dbs=1000)
    commit_interval=100
    set_err_no_pose_video=set()

    for seq_id, file_tag in enumerate(set_video_tags):
        if not os.path.exists(os.path.join(dir_videos,file_tag, view_tag+".mp4")):
            set_err_no_pose_video.add(file_tag)
            continue
        
        imgs=[]

        stream=cv2.VideoCapture(os.path.join(dir_videos,file_tag, f"{view_tag}.mp4"))
        video_fps, num_frames=int(stream.get(cv2.CAP_PROP_FPS)),int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames=num_frames//(video_fps//fps_to_load)
        
        frame_id=0
        success, frame=stream.read()
        
        print(seq_id,file_tag,"shape",frame.shape,"num_imgs",num_frames)

        while success:
            if frame_id%(video_fps//fps_to_load)==0:
                imgs.append(frame)
            frame_id+=1
            success, frame=stream.read()
        stream.release() 

        print(seq_id,file_tag+"_"+view_id,"len(imgs)",len(imgs))

        subdb=env.open_db((file_tag+"_"+view_id).encode('ascii'))
        txn=env.begin(db=subdb,write=True)
        
        for fid, cimg in enumerate(imgs):
            key_byte = "{:06d}".format(fid).encode('ascii')
            data=cv2.imencode('.jpg',cimg)[1]
            txn.put(key_byte,data)       
        txn.commit()
    env.close()


    '''
    env =lmdb.open(dir_lmdb,readonly=True,lock=False,readahead=False,meminit=False, map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
    for seq_id, file_tag in enumerate(set_video_tags):
        if seq_id>0:
            break
        subdb = env.open_db(("ass101_"+view_tag+"_"+file_tag).encode('ascii'),create=False)
        txn=env.begin(db=subdb,write=False)
            
        buf=txn.get("{:06d}".format(0).encode('ascii'))        
        raw_data=np.frombuffer(buf,dtype=np.uint8)#reshape(-1,)
        data=cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        print(data.shape)

        cv2.imwrite('data.png',data)

    exit(0)
    '''




if __name__=='__main__':
    path_to_ref='../ass101/annotations/fine-grained-annotations/train.csv'
    for key in ["train","validation","test"]:
        path_to_query=f"../ass101/annotations/anticipation-annotations/CSVs/{key}.csv"
        print("Lets check", path_to_query)
        check_consistent_train_list(path_to_ref,path_to_query,key=="train")
        if key=="train":
            print("Lets check inverse",path_to_ref)
            check_consistent_train_list(path_to_query,path_to_ref,True)


    path_to_ref='../ass101/annotations/coarse-annotations/coarse_splits/train_coarse_assembly.txt'
    for key in ["validation","test"]:
        path_to_query=f"../ass101/annotations/anticipation-annotations/CSVs/{key}.csv"
        print("Lets check", path_to_query)
        check_consistent_train_list(path_to_ref,path_to_query,key=="train")
        #exit(0)
