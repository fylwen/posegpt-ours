import pickle

import torch
import torch.nn.functional as torch_f
import torch.nn as nn
torch.set_printoptions(precision=4,sci_mode=False)

from pytorch3d.ops import corresponding_points_alignment
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_rotation_6d,axis_angle_to_matrix

import matplotlib.pyplot as plt
from einops import repeat
import tqdm
import cv2
import open_clip
import numpy as np


from meshreg.webuser.smpl_handpca_wrapper_HAND_only import load_model as load_mano_mean

def loss_str2func():
    return {'l1': torch_f.l1_loss, 'l2':torch_f.mse_loss}

def act_str2func():
    return {'softmax': nn.Softmax(),'elu':nn.ELU(),'leakyrelu':nn.LeakyReLU(),'relu':nn.ReLU()}


def torch2numpy(input):
    if input is None:
        return None
    if torch.is_tensor(input):
        input=input.detach().cpu().numpy()
    return input


def print_dict_torch(dict_):    
    for k,v in dict_.items():
        if torch.is_tensor(v):
            print(k,v.size())
        else:
            print(k,v)



def instance_normalization(tensor,epsilon=1e-5,verbose=False):
    mean=torch.mean(tensor,dim=1,keepdim=True)
    if verbose:
        print('tensor',tensor.shape,tensor[:5,:3])
        print('mean',mean.shape,mean[:5])
    tensor=tensor-mean#torch.mean(tensor,dim=1,keepdim=True)
    rsqrt=torch.rsqrt(torch.mean(torch.square(tensor),dim=1,keepdim=True)+epsilon)
    if verbose:
        print('tensor',tensor.shape,tensor[:5,:3])
        print('rsqrt',rsqrt.shape,rsqrt[:5])
    tensor=rsqrt*tensor#torch.rsqrt(torch.mean(torch.square(tensor),dim=1,keepdim=True)+epsilon)
    if verbose:
        print('tensor',tensor.shape,tensor[:5,:3])
    return tensor



def calcualte_bone_length(joint1,joint2,verbose=False):
    joint1=joint1.flatten()
    joint2=joint2.flatten()
    len_bone=np.sqrt(np.sum(np.square(joint2-joint1)))
    if verbose:
        print('joint1/2',joint1.shape,joint2.shape)
        len_bone2=0
        for i in range(0,3):
            len_bone2+=(joint1[i]-joint2[i])**2
        len_bone2=len_bone2**0.5
        assert np.fabs(len_bone-len_bone2)<1e-4, f'caculate_bone_length error, not match,{len_bone},{len_bone2}'
    return len_bone



def load_mano_mean_pose(path_mano_pkl):
    with open(path_mano_pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    output=load_mano_mean(data,ncomps=15, flat_hand_mean=False)
    output.faces=np.array(data['f'])
    palm_joints=np.array(output.J_transformed[[0,1,4,10,7]])#root->2->3->5->4->1

    palm_joints_copy=palm_joints.copy()
    #set wrist root to be [0,0,0]
    palm_joints-=palm_joints[0]
    
    for i in range(palm_joints.shape[0]):
        assert np.fabs(palm_joints_copy[i]-palm_joints_copy[0]-palm_joints[i]).max()<1e-6
        
    return palm_joints




def check_mano_mean_pose(aligned_palm=None):    
    fig = plt.figure(figsize=(2,4))

    num_rows, num_cols=2,1
    axes=fig.subplots(num_rows,num_cols)
    
    axi=plt.subplot2grid((num_rows,num_cols),(0,0),colspan=2)
    axi.axis("off")
    
    for plot_id, hname in enumerate(["LEFT","RIGHT"]):
        path_mano_pkl='./assets/mano/MANO_{:s}.pkl'.format(hname)

        with open(path_mano_pkl, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        output=load_mano_mean(data,ncomps=15, flat_hand_mean=False)
        palm_joints_mano=np.array(output.J_transformed[[0,1,4,10,7]]).copy()##root->2->3->5->4->1 

        ax3d=plt.subplot(num_rows,num_cols,plot_id+1,projection='3d')
        ax3d.set_title(hname,fontsize=6,pad=0.)
        ax3d.view_init(110,45)            
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])

        verts=output.v_template.copy()
        ax3d.scatter(verts[:,2],verts[:,1],verts[:,0],alpha=0.2,c=(0,0.,0.),s=0.01)

        if aligned_palm is not None:
            palm_joints_vis=aligned_palm[hname].detach().cpu().numpy().reshape((-1,3))+palm_joints_mano[0]
        
        cs=[(0,0,1.),(0,1.,1),(0,1,0.),(1,1,0.)]
        for lid,l in enumerate([[0,1],[0,2],[0,3],[0,4]]):
            if aligned_palm is not None:
                ax3d.plot(palm_joints_vis[l,2],palm_joints_vis[l,1],palm_joints_vis[l,0],alpha=1.,c=cs[lid])
            ax3d.plot(palm_joints_mano[l,2],palm_joints_mano[l,1],palm_joints_mano[l,0],alpha=1.,c=(1.,0,0))
    fig.savefig(f"mano.png", dpi=200)
    plt.close(fig)



def align_a2b(a,b,root_idx):    
    rt_a2b= corresponding_points_alignment(a,b,estimate_scale=False,allow_reflection=False)

    R_a2b=rt_a2b.R
    t_a2b=torch.unsqueeze(rt_a2b.T,1)
    a_in_b =torch.bmm(a,R_a2b)+t_a2b

    if root_idx is not None:
        root_a_in_b=a_in_b[:,root_idx:root_idx+1,:].clone()
        t_a2b-=root_a_in_b
    
    return R_a2b, t_a2b

def align_np_sample_for_cam2local(palm, mean_mano_palm_joints,verbose=False):
    cam2local=[]#translation in meter

    for tag in ["left","right"]:
        palm_cam_size=np.mean(np.linalg.norm(palm[tag][1:]-palm[tag][0],ord=2,axis=1))
        palm_mano_size=np.mean(np.linalg.norm(mean_mano_palm_joints[tag][1:]-mean_mano_palm_joints[tag][0],ord=2,axis=1))
        
        palm_mano_reshaped=mean_mano_palm_joints[tag].copy()/palm_mano_size*palm_cam_size
        palm_cam2=palm[tag].copy()

        palm_cam=torch.unsqueeze(torch.from_numpy(palm_cam2),0).double()
        palm_mano=torch.unsqueeze(torch.from_numpy(palm_mano_reshaped),0).double()

        R_cam2local, t_cam2local = align_a2b(palm_cam,palm_mano,root_idx=0)
        cam2local.append(torch.cat([torch.flatten(R_cam2local),torch.flatten(t_cam2local)]))
        #vis_palm_local[tag.upper()]=torch.bmm(palm_cam,R_cam2local)+t_cam2local    
    #check_mano_mean_pose(vis_palm_local)
    return torch.cat(cam2local).detach().cpu().numpy()


def transform_by_align_a2b(flatten_cam_a,flatten_cam_b,palm_joints=[0,5,9,13,17]):
    palm_a=flatten_cam_a[:,palm_joints]
    palm_b=flatten_cam_b[:,palm_joints]

    R_a2b,t_a2b=align_a2b(palm_a,palm_b,root_idx=0)
    a_in_b=torch.bmm(flatten_cam_a,R_a2b)+t_a2b
    return a_in_b

def align_torch_batch_for_cam2local(flatten_cam_joints, flatten_mano_palm_joints,palm_joints=[0,5,9,13,17]):
    return_results={}
    for tag in ["left","right"]:
        return_results[f"cam_joints3d_{tag}"]=flatten_cam_joints[tag]
        flatten_cam_palm=flatten_cam_joints[tag][:,palm_joints]
        flatten_mano_palm=repeat(torch.unsqueeze(flatten_mano_palm_joints[tag].clone().to(flatten_cam_palm.device),0),
                                '() n1 n2-> b n1 n2',b=flatten_cam_palm.shape[0])

        flatten_palm_mano_size=torch.mean(torch.norm(flatten_mano_palm[:,1:]-flatten_mano_palm[:,0:1],p=2,dim=-1),dim=1,keepdim=True)#[bs,1]
        flatten_palm_cam_size=torch.mean(torch.norm(flatten_cam_palm[:,1:]-flatten_cam_palm[:,0:1],p=2,dim=-1),dim=1,keepdim=True)#[bs,1]
        flatten_palm_cam_size=torch.where(flatten_palm_cam_size==0.,flatten_palm_mano_size,flatten_palm_cam_size)
        
        return_results[f"hand_size_{tag}"]=torch.flatten(flatten_palm_cam_size)
        flatten_mano_palm=((flatten_palm_cam_size/flatten_palm_mano_size).view(-1,1,1))*flatten_mano_palm#[bs,5,3]
        return_results[f"R_cam2local_{tag}"],return_results[f"t_cam2local_{tag}"]=align_a2b(flatten_cam_palm.double(),flatten_mano_palm,root_idx=0)
    return return_results
    

def compute_flatten_local2base_info(R_cam2local,t_cam2local,base_frame_id,len_seq,j3d_cam,verbose):
    results={}

    #cam2base and j3d in base, unit:meter*factor_scaling
    R_cam2base=R_cam2local[base_frame_id::len_seq].clone()
    t_cam2base=t_cam2local[base_frame_id::len_seq].clone()
    R_cam2base=R_cam2base.view(-1,1,3,3).repeat(1,len_seq,1,1).view(-1,3,3)
    t_cam2base=t_cam2base.view(-1,1,1,3).repeat(1,len_seq,1,1).view(-1,1,3)
    
    results["j3d_base"]=torch.bmm(j3d_cam,R_cam2base) + t_cam2base 
    #base2cam and local2base, unit: meter*factor_scaling
    R_base2cam, t_base2cam=get_inverse_Rt(R_cam2base,t_cam2base)
    results['R_base2cam']=R_base2cam
    results['t_base2cam']=t_base2cam
    
    R_local2base, t_local2base= compose_Rt_a2b(batch_R_c2a=R_cam2local,batch_t_c2a=t_cam2local,batch_R_c2b=R_cam2base,batch_t_c2b=t_cam2base,is_c2a=True)                                                    
    results['R_local2base']=R_local2base
    results['t_local2base']=t_local2base            
    results["local2base"]= torch.cat([matrix_to_rotation_6d(R_local2base).view(-1,6),t_local2base.view(-1,3)],1)

    return results


def accumulate_flatten_local2base(flatten_Rt_vel,len_out,bs=-1,dim_rot=6):
    R_vel=rotation_6d_to_matrix(flatten_Rt_vel[:,:dim_rot]).view(bs,len_out,3,3)
    t_vel=flatten_Rt_vel[:,dim_rot:].view(bs,len_out,1,3)
    
    trj_R_local2base,trj_t_local2base=R_vel[:,0:1],t_vel[:,0:1]  
    for tt in range(1,len_out):
        R_tt,t_tt=compose_Rt_a2b(batch_R_c2a=R_vel[:,tt],batch_t_c2a=t_vel[:,tt],batch_R_c2b=trj_R_local2base[:,tt-1],batch_t_c2b=trj_t_local2base[:,tt-1],is_c2a=False)    
        trj_R_local2base=torch.cat([trj_R_local2base,torch.unsqueeze(R_tt,1)],1)
        trj_t_local2base=torch.cat([trj_t_local2base,torch.unsqueeze(t_tt,1)],1)

    flatten_R_local2base=torch.flatten(trj_R_local2base,start_dim=0,end_dim=1)
    return {"batch_seq_R_local2base":trj_R_local2base,
            "batch_seq_t_local2base":trj_t_local2base,
            "flatten_R_local2base":flatten_R_local2base,
            "flatten_t_local2base":torch.flatten(trj_t_local2base,start_dim=0,end_dim=1),
            "flatten_R_local2base_6d":matrix_to_rotation_6d(flatten_R_local2base).view(-1,6)} 
  

def augment_rotation_translation(R,t,noise_factor_angle,noise_factor_trans,verbose=False):
    t_aug=t+torch.randn_like(t)*noise_factor_trans
    noise_R_axis=torch.randn(R.shape[0],3,dtype=R.dtype,device=R.device)
    noise_R_axis=torch_f.normalize(noise_R_axis, p=2,dim=1,eps=1e-6)
    noise_R_angle=torch.randn(R.shape[0],1,dtype=R.dtype,device=R.device)*noise_factor_angle
    noise_R_axis_angle=torch.mul(noise_R_axis,noise_R_angle)
    noise_R_mat=axis_angle_to_matrix(noise_R_axis_angle)
    R_aug=torch.bmm(noise_R_mat,R)
    
    if verbose:
        print("noise_R_axis/noise_R_angle/noise_R_axis_angle",noise_R_axis.shape,noise_R_angle.shape,noise_R_axis_angle.shape)#[bs,3],[bs,1],[bs,3]
        print(torch.abs(noise_R_axis_angle.norm(p=2,dim=1,keepdim=True)-torch.abs(noise_R_angle)).max())
        print("noise_R_mat/R/R_aug",noise_R_mat.shape,R.shape,R_aug.shape)#[bs,3,3]x3
        print(torch.abs(torch.det(R_aug)-1.).max(),torch.abs(torch.det(R)-1.).max(),torch.abs(torch.det(noise_R_mat)-1.).max())
    return R_aug,t_aug

def augment_hand_pose_2_5D(flatten_resnet25d):
    flatten_resnet25d[:,:,:2]+=torch.randn_like(flatten_resnet25d[:,:,:2])*2
    flatten_resnet25d[:,:,2:3]*=1.0+torch.randn_like(flatten_resnet25d[:,:,2:3])*0.005
    flatten_resnet25d[:,:,2:3]+=torch.randn_like(flatten_resnet25d[:,:,2:3])*0.015
    flatten_resnet25d[:,1:21,2:3]+=torch.randn_like(flatten_resnet25d[:,1:21,2:3])*0.003
    flatten_resnet25d[:,22:42,2:3]+=torch.randn_like(flatten_resnet25d[:,22:42,2:3])*0.003
    return flatten_resnet25d


def get_flatten_hand_feature(batch_flatten, len_seq,spacing, base_frame_id,  factor_scaling,  masked_placeholder, with_augmentation,  compute_local2first, verbose=False):
    list_flatten_j3d_cam=[]
    list_flatten_j3d_local, list_flatten_j3d_local_normed, list_flatten_j3d_local_aug=[],[],[]
    list_flatten_j3d_base=[]
    list_flatten_valid_features=[]

    list_flatten_vel,list_flatten_vel_aug=[],[]
    list_flatten_local2base, list_flatten_local2first=[],[]
    base_frame_id= 0 if base_frame_id is None else base_frame_id

    noise_factor_local_joints=0.002/0.08#assume mean hand size is 8 cm
    noise_factor_angle_L2R=np.pi/36
    noise_factor_trans_L2R=0.05 #L2R in meter space
    noise_factor_angle=(np.pi/144)*spacing
    noise_factor_trans=(factor_scaling*0.0075)*spacing# in factor_scaling*meter space
    
    results={}
    num_frames=batch_flatten[f"cam_joints3d_left"].shape[0]
    for hand_tag in ["left","right"]:
        #j3d in cam and cam2local, unit: meter*factor_scaling
        j3d_cam  =  factor_scaling*batch_flatten[f'cam_joints3d_{hand_tag}'].double().cuda()
        t_cam2local=factor_scaling*batch_flatten[f't_cam2local_{hand_tag}'].double().cuda()
        R_cam2local=batch_flatten[f'R_cam2local_{hand_tag}'].double().cuda()
        list_flatten_j3d_cam.append(torch.flatten(j3d_cam,start_dim=1))#[bs*len,63]
 
        #j3d in local, unit: meter*factor_scaling
        j3d_local=torch.bmm(j3d_cam,R_cam2local)+t_cam2local          
        list_flatten_j3d_local.append(torch.flatten(j3d_local,start_dim=1))#[bs*len,63]

        #base-related and j3d in base, unit:meter*factor_scaling
        results_base=compute_flatten_local2base_info(R_cam2local=R_cam2local,t_cam2local=t_cam2local,base_frame_id=base_frame_id,len_seq=len_seq,j3d_cam=j3d_cam,verbose=verbose)
        list_flatten_j3d_base.append(torch.flatten(results_base["j3d_base"],start_dim=1))
        list_flatten_local2base.append(results_base["local2base"])
        for ttag in ["R","t"]:
            ttag2=ttag+"_base2cam" 
            results[f'flatten_firstclip_{ttag2}_{hand_tag}']=results_base[ttag2]
            ttag2=ttag+"_local2base"
            results[f'flatten_{ttag2}_{hand_tag}']=results_base[ttag2]
                
        if verbose:
            recov_cam=torch.bmm(results_base["j3d_base"],results_base["R_base2cam"])+results_base["t_base2cam"]
            print(hand_tag.upper(),'check cam-base-cam vs oricam',torch.abs(recov_cam-j3d_cam).max())
            print('R_base2cam/t_base2cam',results_base["R_base2cam"].shape,results_base["t_base2cam"].shape)#[bs*len,3,3],[bs*len,1,3]                
            print('R_local2base/t_local2base',results_base["R_local2base"].shape,results_base["t_local2base"].shape)
            print('Check base@local2base vs Indentity', torch.abs(results_base["R_local2base"][base_frame_id::len_seq]-torch.eye(3).cuda()).max(),torch.abs(results_base["t_local2base"][base_frame_id::len_seq]).max())

            R_local2base2cam, t_local2base2cam = compose_Rt_a2b(batch_R_c2a=results_base["R_local2base"],batch_t_c2a=results_base["t_local2base"], batch_R_c2b=results_base["R_base2cam"],batch_t_c2b=results_base["t_base2cam"],is_c2a=False)
            R_local2cam,t_local2cam=get_inverse_Rt(R_cam2local,t_cam2local)
            print('check local-base-cam vs cam-local-cam', torch.abs(R_local2cam- R_local2base2cam).max(), torch.abs(t_local2cam- t_local2base2cam).max())

            print('check root in local space',torch.abs(j3d_local[:,0,:]).max())
            print('check base@local vs base@base frames',torch.abs(j3d_local[base_frame_id::len_seq]-results_base["j3d_base"][base_frame_id::len_seq]).max())

            recov_r=rotation_6d_to_matrix(results_base["local2base"][:,:6])
            print('check local2base mat-6d-mat vs orimat',recov_r.shape,torch.abs(recov_r-results_base["R_local2base"]).max())

        #first-related
        if compute_local2first:
            results_first=compute_flatten_local2base_info(R_cam2local=R_cam2local,t_cam2local=t_cam2local,base_frame_id=0,len_seq=len_seq,j3d_cam=j3d_cam,verbose=verbose)
            list_flatten_local2first.append(results_first["local2base"])
            for ttag in ["R","t"]:
                results[f'flatten_firstclip_{ttag}_first2cam_{hand_tag}']=results_first[f"{ttag}_base2cam"]
        
        #then start to normalize, tra still has unit of meter*factor_scaling, but hand local pose is normalized by hand size
        j3d_local_normed=j3d_local/(factor_scaling*batch_flatten[f'hand_size_{hand_tag}'].view(-1,1,1).cuda())
        if with_augmentation:
            j3d_local_aug=j3d_local_normed+torch.randn_like(j3d_local_normed)*noise_factor_local_joints
        
        #add mask
        j3d_valid=batch_flatten[f'valid_joints_{hand_tag}'].cuda()
        list_flatten_valid_features.append(torch.flatten(j3d_valid,start_dim=1))#[bs*len,63]        
        if masked_placeholder is not None:
            mask_embed=masked_placeholder[:,:21] if hand_tag=="left" else masked_placeholder[:,21:]
            j3d_local_normed_=j3d_local_normed.detach().clone().type_as(masked_placeholder)
            j3d_local_normed =torch.where(j3d_valid>0.,j3d_local_normed_,mask_embed)
            if with_augmentation:
                j3d_local_aug_=j3d_local_aug.detach().clone().type_as(masked_placeholder)
                j3d_local_aug=torch.where(j3d_valid>0.,j3d_local_aug_,mask_embed)   
            if verbose:
                print("mask_embed/j3d_local_normed",mask_embed.shape,j3d_local_normed.shape)
                print("check root joints",torch.abs(j3d_local_normed[:,0]).max()) 
                print("check masked joints",torch.abs(mask_embed[:,1]-j3d_local_normed[:,1]).max())   
                print("check other joints",torch.abs(j3d_local_normed[:,2:]-j3d_local_normed_[:,2:]).max())           
        
        list_flatten_j3d_local_normed.append(torch.flatten(j3d_local_normed,start_dim=1))#[bs*len,63]
        if with_augmentation:
            list_flatten_j3d_local_aug.append(torch.flatten(j3d_local_aug,start_dim=1))#[bs*len,63]

        #vel is local(t)2local(t-1), unit: meter*factor_scaling
        R_vel, t_vel = compose_Rt_a2b(batch_R_c2a=R_cam2local[1:],batch_t_c2a=t_cam2local[1:], batch_R_c2b=R_cam2local[:-1],batch_t_c2b=t_cam2local[:-1],is_c2a=True)
        #set first frame velocity
        R_vel=torch.cat([torch.eye(3,dtype=R_vel.dtype,device=R_vel.device).view(-1,3,3),R_vel],0)
        t_vel=torch.cat([torch.zeros_like(t_vel[0:1]),t_vel],0)
        R_vel[0::len_seq]=torch.eye(3)
        t_vel[0::len_seq]=0.
        list_flatten_vel.append(torch.cat([matrix_to_rotation_6d(R_vel).view(-1,6),t_vel.view(-1,3)],1))

        if with_augmentation:
            R_vel_aug,t_vel_aug=augment_rotation_translation(R=R_vel,t=t_vel,noise_factor_angle=noise_factor_angle,noise_factor_trans=noise_factor_trans,verbose=verbose)
            list_flatten_vel_aug.append(torch.cat([matrix_to_rotation_6d(R_vel_aug).view(-1,6),t_vel_aug.view(-1,3)],1)) 
        

    results["flatten_joints3d_in_cam_gt"]=torch.cat(list_flatten_j3d_cam,1).view(num_frames,-1,3)
    results["flatten_joints3d_in_local_gt"]=torch.cat(list_flatten_j3d_local,1).view(num_frames,-1,3)
    results["flatten_joints3d_in_base_gt"]=torch.cat(list_flatten_j3d_base,1).view(num_frames,-1,3)
    results["flatten_local2base_gt"]=torch.cat(list_flatten_local2base,1)
    if compute_local2first:
        results["flatten_local2first_gt"]=torch.cat(list_flatten_local2first,1)

    
    #local left2right, in original meter space
    R_cam2left,t_cam2left_meter=batch_flatten[f"R_cam2local_left"].cuda(), batch_flatten["t_cam2local_left"].cuda()
    R_cam2right,t_cam2right_meter=batch_flatten[f"R_cam2local_right"].cuda(), batch_flatten["t_cam2local_right"].cuda()
    R_left2right, t_left2right_meter=compose_Rt_a2b(batch_R_c2a=R_cam2left,batch_t_c2a=t_cam2left_meter, batch_R_c2b=R_cam2right,batch_t_c2b=t_cam2right_meter,is_c2a=True)
    flatten_left2right=torch.cat([matrix_to_rotation_6d(R_left2right).view(-1,6),t_left2right_meter.view(-1,3)],1) 

    flatten_comps={}
    flatten_comps["gt"]=torch.cat([torch.cat(list_flatten_j3d_local_normed,1),torch.cat(list_flatten_vel,1),flatten_left2right],1).float()#
    if with_augmentation:
        R_left2right_aug,t_left2right_aug=augment_rotation_translation(R_left2right,t_left2right_meter,noise_factor_angle=noise_factor_angle_L2R,noise_factor_trans=noise_factor_trans_L2R)
        flatten_left2right_aug=torch.cat([matrix_to_rotation_6d(R_left2right_aug).view(-1,6),t_left2right_aug.view(-1,3)],1) 
        flatten_comps["aug"]=torch.cat([torch.cat(list_flatten_j3d_local_aug,1),torch.cat(list_flatten_vel_aug,1),flatten_left2right_aug],1).float()#
        if verbose:
            print("flatten_hand_feature_aug",flatten_comps["aug"].shape)#[bs,144]
            ssid=15
            print("diff-L",(flatten_comps["aug"]-flatten_comps["gt"])[ssid,:63])
            print("GT-L",flatten_comps["gt"][ssid,:63])
            print(torch.mean(torch.abs(flatten_comps["gt"][:,:63]),dim=0))
            print(torch.std(flatten_comps["gt"][:,:63],dim=0))
            print("diff-R",(flatten_comps["aug"]-flatten_comps["gt"])[ssid,63:126])
            print("GT-R",flatten_comps["gt"][ssid,63:126])            
            print(torch.mean(torch.abs(flatten_comps["gt"][:,63:126]),dim=0))
            print(torch.std(flatten_comps["gt"][:,63:126],dim=0))
            print("diff-vel",(flatten_comps["aug"]-flatten_comps["gt"])[ssid,126:])
            print("GT-vel/L2R",flatten_comps["gt"][ssid,126:])
            print(torch.mean(torch.abs(flatten_comps["gt"][:,126:]),dim=0))
            print(torch.std(flatten_comps["gt"][:,126:],dim=0))

            
    list_flatten_valid_features.append(torch.ones_like(torch.cat(list_flatten_vel,1)))
    list_flatten_valid_features.append(torch.ones_like(flatten_left2right))
    results["flatten_valid_features"]=torch.cat(list_flatten_valid_features,1) 
        
    if verbose:
        print('**check pred**') 
        hand_left_size=batch_flatten["hand_size_left"].view(-1,len_seq)[:,base_frame_id+1:]
        hand_right_size=batch_flatten["hand_size_right"].view(-1,len_seq)[:,base_frame_id+1:]
        flatten_hand_size=[torch.flatten(hand_left_size),torch.flatten(hand_right_size)] 
        
        recov_comp=from_comp_to_joints(flatten_comps["gt"].view(num_frames//len_seq,len_seq,-1)[:,base_frame_id+1:], flatten_hand_size,factor_scaling, trans_info=results,verbose=verbose)
        recov_base_joints3d, recov_trj=recov_comp['joints_in_base'],recov_comp['local2base']

        ori_base_joints3d=torch.flatten(results["flatten_joints3d_in_base_gt"].view(-1,len_seq,42,3)[:,base_frame_id+1:],0,1)#
        ori_trj=torch.flatten(results["flatten_local2base_gt"].view(-1,len_seq,18)[:,base_frame_id+1:],0,1)#
        print('check ori vs pre-post process',torch.abs(recov_base_joints3d-ori_base_joints3d).max(),torch.abs(recov_trj-ori_trj).max())


        print('**check seq by first aligning to first frame and then converting to base frame**')
        hand_left_size=batch_flatten["hand_size_left"]
        hand_right_size=batch_flatten["hand_size_right"]
        flatten_hand_size=[hand_left_size,hand_right_size] 
            
        recov_comp=from_comp_to_joints(flatten_comps["gt"].view(num_frames//len_seq,len_seq,-1), flatten_hand_size,factor_scaling,trans_info=results,verbose=verbose)
        recov_inframe1_joints3d=recov_comp['joints_in_base'].view(-1,42,3) #base frame for this output is the first observed frame.
        
        R_frame1_2base=results['flatten_R_local2base_left'][0::len_seq].clone().view(-1,1,3,3).repeat(1,len_seq,1,1).view(-1,3,3)
        t_frame1_2base=results['flatten_t_local2base_left'][0::len_seq].clone().view(-1,1,1,3).repeat(1,len_seq,1,1).view(-1,1,3)
        recov_inframe1_joints3d[:,:21]=torch.bmm(recov_inframe1_joints3d[:,:21].double(),R_frame1_2base)+t_frame1_2base

        R_frame1_2base=results['flatten_R_local2base_right'][0::len_seq].clone().view(-1,1,3,3).repeat(1,len_seq,1,1).view(-1,3,3)
        t_frame1_2base=results['flatten_t_local2base_right'][0::len_seq].clone().view(-1,1,1,3).repeat(1,len_seq,1,1).view(-1,1,3)
        recov_inframe1_joints3d[:,21:]=torch.bmm(recov_inframe1_joints3d[:,21:].double(),R_frame1_2base)+t_frame1_2base
        ori_base_joints3d=results["flatten_joints3d_in_base_gt"].view(-1,42,3)
        print('check ori vs pre-post process',torch.abs(recov_inframe1_joints3d-ori_base_joints3d).max())
        print('num_frames/len_seq',num_frames,num_frames//len_seq,len_seq)

    return flatten_comps, results



def from_comp_to_joints(batch_seq_comp,flatten_hand_size, factor_scaling, trans_info=None, num_hands=2,num_joints=21,dim_joint=3,dim_rot=6,dim_tra=3,verbose=False):
    batch_seq_comp=batch_seq_comp.double()
    
    bs,len_out=batch_seq_comp.shape[0],batch_seq_comp.shape[1]
    dev=num_hands*num_joints*dim_joint

    to_return_trans_info={}
    #in factor_scaling*meter space
    list_j3d_in_local,list_j3d_in_cam, list_j3d_in_base,list_local2base=[],[],[],[]

    for hid in range(0,num_hands):
        hand_tag="left" if hid==0 else "right"
        #recover in local frame, unit is meter*factor_scaling
        hand_a=batch_seq_comp[:,:,hid*num_joints*dim_joint:(hid+1)*num_joints*dim_joint].reshape(bs*len_out,num_joints,dim_joint)
        hand_a=hand_a*flatten_hand_size[hid].view(-1,1,1).to(hand_a.device)*factor_scaling
        list_j3d_in_local.append(hand_a)
        
        #with velocity, recover global trajectory in base-frame space
        flatten_Rt_vel=batch_seq_comp[:,:,dev+hid*(dim_rot+dim_tra):dev+(hid+1)*(dim_rot+dim_tra)].reshape(bs*len_out,dim_rot+dim_tra)
        results_local2base=accumulate_flatten_local2base(flatten_Rt_vel=flatten_Rt_vel,len_out=len_out,bs=bs,dim_rot=dim_rot)
        
        for ttag in ["R","t"]:
            ttag2=f"batch_seq_{ttag}_local2base"
            to_return_trans_info[f"{ttag2}_{hand_tag}"]=results_local2base[ttag2]                
        list_local2base.append(torch.cat([results_local2base["flatten_R_local2base_6d"],results_local2base["flatten_t_local2base"].view(-1,3)],dim=1))

        if verbose:
            results_R=trans_info[f'flatten_R_local2base_{hand_tag}'].view(bs,-1,3,3)[:,-len_out:]
            results_t=trans_info[f'flatten_t_local2base_{hand_tag}'].view(bs,-1,1,3)[:,-len_out:]
            
            print("check trjectory local2base, ori vs post-recov",torch.abs(results_R-results_local2base["batch_seq_R_local2base"]).max(), torch.abs(results_t-results_local2base["batch_seq_t_local2base"]).max()) 

        #recover in base frame,
        hand_in_base=torch.bmm(hand_a,results_local2base["flatten_R_local2base"])+results_local2base["flatten_t_local2base"]
        list_j3d_in_base.append(hand_in_base)

        #recover in camera space    
        if f"batch_nextclip_R_base2cam_{hand_tag}" in trans_info.keys(): 
            #for num of iterations>1, check in pmodel_ite
            R_cbase2cam=trans_info[f"batch_nextclip_R_base2cam_{hand_tag}"].clone()
            t_cbase2cam=trans_info[f"batch_nextclip_t_base2cam_{hand_tag}"].clone()
        else:
            #first ite
            R_cbase2cam = trans_info[f'flatten_firstclip_R_base2cam_{hand_tag}'].view(bs,-1,3,3)[:,0].clone().cuda()#
            t_cbase2cam = trans_info[f'flatten_firstclip_t_base2cam_{hand_tag}'].view(bs,-1,1,3)[:,0].clone().cuda()#
        
        R_cbase2cam=R_cbase2cam.view(bs,-1,3,3).repeat(1,len_out,1,1).clone()
        t_cbase2cam=t_cbase2cam.view(bs,-1,1,3).repeat(1,len_out,1,1).clone()
        to_return_trans_info[f'batch_seq_R_base2cam_{hand_tag}']=R_cbase2cam
        to_return_trans_info[f'batch_seq_t_base2cam_{hand_tag}']=t_cbase2cam
        
        R_base2cam=torch.flatten(R_cbase2cam,start_dim=0,end_dim=1)
        t_base2cam=torch.flatten(t_cbase2cam,start_dim=0,end_dim=1)
        hand_in_cam=torch.bmm(hand_in_base,R_base2cam)+t_base2cam
        list_j3d_in_cam.append(hand_in_cam)
    
    to_return={'joints_in_local':torch.cat(list_j3d_in_local,1), 
                'joints_in_base': torch.cat(list_j3d_in_base,dim=1).float(), 
                'joints_in_cam': torch.cat(list_j3d_in_cam,dim=1).float(), 
                'local2base':torch.cat(list_local2base,dim=1).float(),
                'batch_seq_trans_info':to_return_trans_info}
                
    return to_return



def project_hand_3d2img(skel,cam_intr,cam_extr=None):    
    if cam_extr is not None:
        skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
        skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
    else:
        skel_camcoords=skel
    
    hom_2d = cam_intr.dot(skel_camcoords.transpose()).transpose()
    hom_2d[:,2]=np.where(hom_2d[:,2]<1e-8,1,hom_2d[:,2])
    skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]

    return skel2d
        


def recover_3d_proj_pinhole_wo_rescale(camintr, est_2D, est_Z0,verbose=False):
    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = est_2D.shape[0]
    num_joints = est_2D.shape[1]
    focal = focal.view(batch_size, 1, 1)
    est_Z0 = est_Z0.view(batch_size, -1, 1)#
    est_2D = est_2D.view(batch_size, -1, 2)#

    if verbose:
        print('recover_3d_proj_pinhole, est_Z0/2D',est_Z0.shape,est_2D.shape)#[128,21,1],[128,21,2]
    
    # est_scale is homogeneous to object scale change in pixels
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    
    if verbose:
        print('focal,cam_centers',focal.shape,cam_centers.shape)#[128,21,2],[128,21,2]

    est_XY0=(est_2D-cam_centers) * est_Z0 / focal
    if verbose:
        print('est_Z0/2D/XY0',est_Z0.shape,est_2D.shape,est_XY0.shape)#[128,21,1],[128,21,2],[128,21,2]
    est_c3d = torch.cat([est_XY0, est_Z0], -1)
    return est_c3d
        

def recover_3d_proj_pinhole(camintr, est_scale, est_trans,off_z=0.4, input_res=(128, 128), verbose=False):
    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = est_trans.shape[0]
    num_joints = est_trans.shape[1]
    focal = focal.view(batch_size, 1, 1)
    est_scale = est_scale.view(batch_size, -1, 1)# z factor
    est_trans = est_trans.view(batch_size, -1, 2)# 2D x,y, img_center as 0,0

    if verbose:
        print('recover_3d_proj_pinhole, est_scale/est_trans',est_scale.shape,est_trans.shape)#[128,21,1],[128,21,2]
    
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    img_centers = (cam_centers.new(input_res) / 2).view(1, 1, 2).repeat(batch_size,num_joints, 1)

    if verbose:
        #print('focal/cam/img',focal,cam_centers,img_centers)
        print('focal,cam_centers,img_centers',focal.shape,cam_centers.shape,img_centers.shape)#[128,21,2],[128,21,2]

    est_xy0= est_trans+img_centers
    est_XY0=(est_xy0-cam_centers) * est_Z0 / focal
    if verbose:
        print('est_Z0, est_xy0,est_XY0',est_Z0.shape,est_xy0.shape,est_XY0.shape)#[128,21,1],[128,21,2],[128,21,2]
    #joints25d[:,:,i]=(joints25d[:,:,i]-cam_intr[i,2])*joints25d[:,:,2]/cam_intr[i,i] 
    est_c3d = torch.cat([est_XY0, est_Z0], -1)
    return est_xy0,est_Z0, est_c3d


def transfer_to_feature_from_2_5D(pose_25d, camintr,off_z=0.4,input_res=(128,128),verbose=False):
    focal = camintr[:, :1, :1]
    batch_size, num_joints = pose_25d.shape[:2]
    focal = focal.view(batch_size, 1, 1)

    est_xy0, est_Z0=pose_25d[:,:,:2],pose_25d[:,:,2:]
    feature_z=(1./focal)*(est_Z0-off_z)
    
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    img_centers = (cam_centers.new(input_res) / 2).view(1, 1, 2).repeat(batch_size,num_joints, 1)

    feature_xy=est_xy0-img_centers
    return feature_xy, feature_z


class To25DBranch(nn.Module):
    def __init__(self, trans_factor=1, scale_factor=1):
        """
        Args:
            trans_factor: Scaling parameter to insure translation and scale
                are updated similarly during training (if one is updated 
                much more than the other, training is slowed down, because
                for instance only the variation of translation or scale
                significantly influences the final loss variation)
            scale_factor: Scaling parameter to insure translation and scale
                are updated similarly during training
        """
        super(To25DBranch, self).__init__()
        self.trans_factor = trans_factor
        self.scale_factor = scale_factor
        #self.inp_res = [256, 256]

    def forward(self, flatten_camintr, flatten_scaletrans,height,width, verbose=False):        
        batch_size = flatten_scaletrans.shape[0]
        trans = flatten_scaletrans[:, :, :2]
        scale = flatten_scaletrans[:, :, 2]
        final_trans = trans.view(batch_size,-1, 2)* self.trans_factor
        final_scale = scale.view(batch_size,-1, 1)* self.scale_factor
        #height, width = tuple(flatten_imgs.shape[2:])#sample["image"].shape[2:])
        camintr = flatten_camintr#sample["cam_intr"].cuda() 
        
        est_xy0,est_Z0, est_c3d=recover_3d_proj_pinhole(camintr=camintr,est_scale=final_scale,est_trans=final_trans,input_res=(width,height), verbose=verbose)
        if verbose:
            print('25DBranch- est_xy0',est_xy0.shape) #[-1,29,2]
            print('est_Z0',est_Z0.shape) #[-1,29,1]
            print('est_c3d',est_c3d.shape) #[-1,29,3]
        return {
            "rep2d": est_xy0, 
            "rep_absz": est_Z0,
            "rep3d": est_c3d,
        }
    def reproject_to_latent(self, flatten_camintr,flatten_pose25d, height=None, width=None):
        trans,scale=transfer_to_feature_from_2_5D(pose_25d=flatten_pose25d,camintr=flatten_camintr,input_res=(width,height))
        trans=trans/self.trans_factor
        scale=scale/self.scale_factor
        return torch.cat([trans,scale],-1)


        

def compute_hand_loss(est2d,gt2d,estz,gtz,est3d,gt3d,weights,pose_loss,verbose):
    hand_losses={}
    sum_weights=torch.where(torch.sum(weights)>0,torch.sum(weights),torch.Tensor([1]).cuda())[0]
    if not (est2d is None):
        loss2d=pose_loss(est2d,gt2d,reduction='none')
        
        if verbose:
            print('Before bmm loss2d/weights',loss2d.shape,weights.shape,weights)#[bs,42,2],[bs]
        loss2d_1=torch.mean(torch.flatten(loss2d,start_dim=1,end_dim=-1),dim=1,keepdim=False)
        loss2d=torch.mul(loss2d_1,weights)
        if verbose:
            print('After bmm loss2d_1/loss2d',loss2d_1.shape,loss2d.shape)#[bs],[bs]
        
        hand_losses["recov_joints2d"]=torch.sum(loss2d)/sum_weights
    if not (estz is None):
        lossz=pose_loss(estz,gtz,reduction='none')
        
        if verbose:
            print('Before bmm lossz',lossz.shape)#[bs,42,1]
        lossz_1=torch.mean(torch.flatten(lossz,start_dim=1,end_dim=-1),dim=1,keepdim=False)
        lossz=torch.mul(lossz_1,weights)
        if verbose:
            print('After bmm lossz_1/lossz',lossz_1.shape,lossz.shape)#[bs],[bs]
        
        hand_losses["recov_joints_absz"]=torch.sum(lossz)/sum_weights

        hand_losses["recov_joints_zroot"]=pose_loss(estz[:,0],gtz[:,0],reduction='mean')
        hand_losses["recov_joints_zrel"]=pose_loss(estz[:,1:]-estz[:,0:1],gtz[:,1:]-gtz[:,0:1],reduction='mean')

    if not (est3d is None):
        loss3d= pose_loss(est3d,gt3d,reduction='none')
        if verbose:
            print('Before bmm loss3d',loss3d.shape)#[bs,42,3]
        loss3d_1=torch.mean(torch.flatten(loss3d,start_dim=1,end_dim=-1),dim=1,keepdim=False)
        loss3d=torch.mul(loss3d_1,weights)
        if verbose:
            print('After bmm loss3d_1/loss3d',loss3d_1.shape,loss3d.shape)#[bs],[bs]
            
        hand_losses["recov_joint3d"] = torch.sum(loss3d)/sum_weights
    return hand_losses



#Geometry
def get_inverse_Rt(batch_R_a2b,batch_t_a2b):
    #batch_R_a2b in [-1,3,3]; batch_t_a2b in [-1,1,3]
    batch_R_a2b,batch_t_a2b=batch_R_a2b.double(),batch_t_a2b.double()

    batch_R_b2a=batch_R_a2b.transpose(1,2)
    batch_t_b2a=-torch.bmm(batch_t_a2b,batch_R_b2a)
    return batch_R_b2a, batch_t_b2a

def compose_Rt_a2b(batch_R_c2a,batch_t_c2a,batch_R_c2b,batch_t_c2b, is_c2a):
    batch_R_c2a,batch_t_c2a=batch_R_c2a.double(),batch_t_c2a.double()
    batch_R_c2b,batch_t_c2b=batch_R_c2b.double(),batch_t_c2b.double()

    batch_R_a2c=batch_R_c2a.transpose(1,2) if is_c2a else batch_R_c2a
    batch_R_a2b=torch.bmm(batch_R_a2c,batch_R_c2b)

    if not is_c2a:
        batch_t_a2c=batch_t_c2a
    batch_t_a2b=batch_t_c2b+(-torch.bmm(batch_t_c2a,batch_R_a2b) if is_c2a else torch.bmm(batch_t_a2c,batch_R_c2b))

    return batch_R_a2b, batch_t_a2b



def solve_pnp_and_transform(cam1_joints3d,cam1_joints2d,cam2_K):
    _, rvec, tvec=cv2.solvePnP(cam1_joints3d,cam1_joints2d,cam2_K,None,flags=cv2.SOLVEPNP_EPNP)

    #ca for cam1, cb for cam2
    R_ca2cb=cv2.Rodrigues(rvec)[0]
    t_ca2cb=tvec.reshape(1,3)

    cam2_joints3d=R_ca2cb.dot(cam1_joints3d.transpose()).transpose()+t_ca2cb
    cam2_hom2d=cam2_K.dot(cam2_joints3d.transpose()).transpose()
    cam2_joints2d =(cam2_hom2d / cam2_hom2d[:, 2:])[:, :2]
    
    return {"R_ca2cb":R_ca2cb,"t_ca2cb":t_ca2cb,"cam2_joints3d":cam2_joints3d,"cam2_joints2d":cam2_joints2d}

def solve_pnp_ransac_and_translate(cam1_joints3d,cam1_joints2d,cam2_K):
    _, _, tvec, inliers = cv2.solvePnPRansac(cam1_joints3d,cam1_joints2d,cam2_K,None,flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        print("Inliers is None")
        return {"cam2_joints3d":cam1_joints3d}
    else:
        t_ca2cb=tvec.reshape(1,3)
    
        cam2_joints3d=cam1_joints3d+t_ca2cb
        return {"cam2_joints3d":cam2_joints3d,"t_ca2cb":t_ca2cb}
    

def compute_bert_embedding_for_taxonomy(model_bert,datasets, is_action,verbose=False):
    name_to_idx={}
    for cset in datasets:
        key_sets= cset.action_to_idx.keys() if is_action else cset.object_to_idx.keys()
        for cname in key_sets:
            if cname not in name_to_idx:
                name_to_idx[cname]=len(name_to_idx)
    if verbose:
        print("compute_bert_embedding_for_taxonomy",name_to_idx)            

    list_name=list(name_to_idx.keys())
    tokens=compute_berts_for_strs(model_bert,list_name,verbose)
    tokens=torch.transpose(tokens,0,1)

    if verbose:
        print('Embedding-tokens',tokens.shape)#[512,-1]
    return name_to_idx,tokens
    
def compute_berts_for_strs(model_bert, list_strs, verbose=False):
    tokens=open_clip.tokenizer.tokenize(list_strs).cuda()
    if verbose:
        print("tokens",tokens.shape)#[-1,77]
    with torch.no_grad():
        tokens=model_bert.encode_text(tokens).float()
    if verbose:
        print("Bert-tokens",tokens.shape)#[-1,512]
    tokens/=tokens.norm(dim=-1,keepdim=True)    
    return tokens


def embedding_lookup(query, embedding, verbose=False):
    returns={}
    flat_input=torch.flatten(query,start_dim=0,end_dim=-2)#
    if verbose:
        print('Embedding_lookup- flat_input',flat_input.shape)#[bs,512]
        print('embedding',embedding.shape)#[512,num_embeddings]
    
    #first normalized
    flat_input=flat_input/flat_input.norm(dim=-1,keepdim=True)
    w=embedding/embedding.norm(dim=0,keepdim=True)
    
    if verbose:
        print('Embedding_lookup- flat_input',flat_input.shape,torch.abs(flat_input.norm(dim=1)-1.).max())#[bs,512]
        print('w',w.shape,torch.abs(w.norm(dim=0)-1.).max())#[512,num_embeddings]
        
    cosine_similarity = torch.matmul(flat_input,w)
    _, flat_1nn_indices = cosine_similarity.max(dim=1)
    
    if verbose:
        print("cosine_similarity",cosine_similarity.shape)#[bs,embeddings]
        print('flat_1nn_indices',flat_1nn_indices.shape)#[bs]
        
    return cosine_similarity,flat_1nn_indices





def compute_root_aligned_and_palmed_aligned(flatten_cent_joints_out, flatten_cent_joints_gt,align_to_gt_size, palm_joints=[0,5,9,13,17],verbose=False):
    root_idx=palm_joints[0]
    return_results={}
    for tag in ["left","right"]:
        flatten_cent_chand_out=(flatten_cent_joints_out[:,:21] if tag=="left" else flatten_cent_joints_out[:,21:]).clone().contiguous()
        flatten_cent_chand_gt=(flatten_cent_joints_gt[:,:21] if tag=="left" else flatten_cent_joints_gt[:,21:]).clone().contiguous()

        if align_to_gt_size:
            palm_size_out=torch.mean(torch.norm(flatten_cent_chand_out[:,palm_joints[1:]]-flatten_cent_chand_out[:,root_idx:root_idx+1],p=2,dim=-1),dim=1).view(-1,1,1)
            palm_size_gt=torch.mean(torch.norm(flatten_cent_chand_gt[:,palm_joints[1:]]-flatten_cent_chand_gt[:,root_idx:root_idx+1],p=2,dim=-1),dim=1).view(-1,1,1)#
            
            if verbose:
                print("flatten_cent_chand_out/gt",flatten_cent_chand_out.shape,flatten_cent_chand_gt.shape)#[bs,21,3]
                print("palm_size",palm_size_out.shape,palm_size_gt.shape)#[bs,1,1]
                print("computation",(flatten_cent_chand_out[:,palm_joints[1:]]-flatten_cent_chand_out[:,root_idx:root_idx+1]).shape)
                print((flatten_cent_chand_gt[:,palm_joints[1:]]-flatten_cent_chand_gt[:,root_idx:root_idx+1]).shape)#[bs,4,3]
                print(torch.norm(flatten_cent_chand_out[:,palm_joints[1:]]-flatten_cent_chand_out[:,root_idx:root_idx+1],p=2,dim=-1).shape)
                print(torch.norm(flatten_cent_chand_gt[:,palm_joints[1:]]-flatten_cent_chand_gt[:,root_idx:root_idx+1],p=2,dim=-1).shape)#[bs,4]
            
            assert (palm_size_out>0).all()
            flatten_cent_chand_out=(1./palm_size_out)*palm_size_gt*flatten_cent_chand_out
            
        return_results[f"flatten_{tag}_ra_out"]=flatten_cent_chand_out
        return_results[f"flatten_{tag}_ra_gt"]=flatten_cent_chand_gt
        return_results[f"flatten_{tag}_pa_out"]=transform_by_align_a2b(flatten_cam_a=flatten_cent_chand_out,flatten_cam_b=flatten_cent_chand_gt)
    return return_results
