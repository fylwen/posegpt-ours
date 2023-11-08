import argparse
from datetime import datetime
import os
import pickle
import numpy as np

from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from libyana.exputils.argutils import save_args
from libyana.modelutils import modelio
from libyana.modelutils import freeze
from libyana.randomutils import setseeds
from libyana.datautils import concatloader

from meshreg.datasets import collate
from torch.utils.data._utils.collate import default_collate

from meshreg.netscripts import epochpass_amodel as epochpass
from meshreg.netscripts import reloadmodel,get_dataset

from meshreg.netscripts.position_evaluator import calculate_activation_statistics,calculate_frechet_distance
from meshreg.models.fid_net import FIDNet


from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

plt.switch_backend("agg")


print('*********Sucessfully import*************')

extend_queries = []
def collate_fn(seq, extend_queries=extend_queries):
    return collate.seq_extend_flatten_collate(seq,extend_queries)#seq_extend_collate(seq, extend_queries)


def main(args):
    setseeds.set_all_seeds(args.manual_seed)
    # Initialize hosting
    now = datetime.now()
    
        

    kwargs={"action_taxonomy_to_use": "fine", "load_untrimmed_videos": False, "view_ids":["e3"],"get_velocity": False,"max_samples":-1,"nclips_dev":args.nclips_dev}
    print('**** Get val set from: ', args.val_dataset, args.val_split)

    use_same_action=True
    assert args.nclips_dev==0
    
    
    val_dataset = get_dataset.get_dataset_motion([args.val_dataset],
                    list_splits=[args.val_split],
                    list_view_ids=[args.val_view_id],
                    dataset_folder=args.dataset_folder,
                    use_same_action=use_same_action,
                    ntokens_per_clip=args.ntokens_per_clip*args.nclips_pred,
                    spacing=args.spacing,
                    nclips=1,
                    min_window_sec=args.min_window_sec,
                    is_shifting_window=False,
                    dict_is_aug={},
                    **kwargs)
    
    val_loader = get_dataset.DataLoaderX(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=False,
        drop_last=False,
        collate_fn= collate_fn,)
    
    '''
      
    val_dataset = get_dataset.get_dataset_fid([args.val_dataset],
                    list_splits=[args.val_split],
                    dataset_folder=args.dataset_folder,
                    spacing=args.spacing,
                    capacity_ntokens=256,#args.ntokens_per_video,
                    const_ntokens_obsv=256,#args.ntokens_per_video,
                    all_queries=None,
                    is_shifting_window=True,
                    **kwargs,)    
                                
    val_loader = get_dataset.DataLoaderX(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=False,
        drop_last=False,
        collate_fn= default_collate)#collate_fn,)###################
    
    '''
    
    model=FIDNet(transformer_d_model=args.hidden_dim,
                        transformer_nhead=args.nheads,
                        transformer_dim_feedforward=args.dim_feedforward,
                        transformer_nlayers_enc=args.nlayers_enc,
                        transformer_activation=args.transformer_activation,
                        
                        ntokens_per_video=args.nclips_pred*args.ntokens_per_clip,
                        spacing=args.spacing, 
                        code_loss="l1")


    
    epoch=reloadmodel.reload_model(model,args.resume_path) 
    model=model.cuda()
    model.compute_bert_embedding_for_taxonomy(val_dataset.list_pose_datasets,is_action=True,verbose=False)

    freeze.freeze_batchnorm_stats(model)  # Freeze batchnorm 
    
    tag_to_save = f"{args.val_dataset}_{args.val_split}_minwin{int(args.min_window_sec)}_ntokens{args.nclips_pred*args.ntokens_per_clip}_3"
    
    
    epochpass.epoch_pass_fid_eval_for_gt(val_loader,
                                        model,
                                        num_prefix_frames_to_remove=0,
                                        tag_to_save=tag_to_save)
    
    '''
    epochpass.epoch_pass_fid(val_loader,
                        model,
                        optimizer=None,
                        scheduler=None,
                        epoch=0,
                        split_tag="val",
                        tensorboard_writer=None)
    '''
    

def calculate_fid_for_two_distributions(args, nsamples=-1):
    kwargs={"action_taxonomy_to_use": "fine", "load_untrimmed_videos": False, "view_ids":["e3"],"get_velocity": False,"max_samples":-1}    
        
    val_dataset = get_dataset.get_dataset_motion([args.val_dataset],
                    list_splits=["val"],
                    list_view_ids=[1],
                    dataset_folder=args.dataset_folder,
                    use_same_action=True,
                    ntokens_per_clip=256,
                    spacing=args.spacing,
                    nclips=1,
                    min_window_sec=0.5,
                    is_shifting_window=True,
                    dict_is_aug={},
                    **kwargs)


    path_pkl1="fid_vis_asshand_val_view_id-1_minwindow1.0_seqlen112_ckpt_30.pkl"
    path_pkl2="fidgt_asshand_val_minwin1_ntokens112_3.pkl"

    
    with open(path_pkl1, 'rb') as f:
        dict_pkl1=pickle.load(f)
    
    with open(path_pkl2, 'rb') as f:
        dict_pkl2=pickle.load(f)

    activations1=dict_pkl1["batch_enc_out_global_feature"]
    activations2=dict_pkl2["batch_enc_out_global_feature"]

    obsv_actions1=dict_pkl1["batch_action_name_obsv"]
    obsv_actions2=dict_pkl2["batch_action_name_obsv"]

    print(len(obsv_actions1),len(obsv_actions2))
    #for x in range(len(obsv_actions1)):
    #    assert obsv_actions1[x]==obsv_actions2[x],obsv_actions1[x]+","+obsv_actions2[x]

    if nsamples>0:
        idx1=np.random.choice(len(activations1),size=nsamples,replace=True)
        idx2=np.random.choice(len(activations2),size=nsamples,replace=True)
        print("after sampling with replacement", activations1.shape,activations2.shape, "to", activations1[idx1].shape,activations2[idx2].shape)
        activation1=activations1[idx1]
        activation2=activations2[idx2]

        obsv_actions1=[obsv_actions1[i] for i in idx1]
        obsv_actions2=[obsv_actions2[i] for i in idx2]
        print(len(obsv_actions1),len(obsv_actions2))


    mu1, sigma1 = calculate_activation_statistics(activations1)#[idx1])
    mu2, sigma2 = calculate_activation_statistics(activations2)#[idx2])
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    print("mu1,sigma1","mu2,sigma2",mu1.shape,sigma1.shape,mu2.shape,sigma2.shape)
    print("--",path_pkl1,path_pkl2)
    print("general fid",fid_value)

    exit(0)

    fid_by_class=[]
    aname_by_class=[]
    activation1_by_class={k:[] for k in val_dataset.list_pose_datasets[0].action_to_idx.keys()}
    activation2_by_class={k:[] for k in val_dataset.list_pose_datasets[0].action_to_idx.keys()}

    for i in range(len(obsv_actions1)):
        activation1_by_class[obsv_actions1[i]].append(activations1[i])
    for i in range(len(obsv_actions2)):
        activation2_by_class[obsv_actions2[i]].append(activations2[i])

    for aname in activation1_by_class.keys():
        #print(aname, np.array(activation1_by_class[aname]).shape, np.array(activation2_by_class[aname]).shape)
        cactivation1,cactivation2=np.array(activation1_by_class[aname]),np.array(activation2_by_class[aname])
        #assert np.fabs(cactivation1-cactivation2).max()<1e-4
        #cidx1=np.random.choice(len(cactivation1),size=1000,replace=True)
        #cidx2=np.random.choice(len(cactivation2),size=1000,replace=True)

        cmu1, csigma1 = calculate_activation_statistics(cactivation1)#[cidx1])
        cmu2, csigma2 = calculate_activation_statistics(cactivation2)#[cidx2])

        #print(cmu1.shape,csigma1.shape,cmu2.shape,csigma2.shape)
        #print(aname, np.fabs(cmu1-cmu2).max(), np.fabs(csigma1-csigma2).max())

        cfid_value = calculate_frechet_distance(cmu1, csigma1, cmu2, csigma2)
        print(aname,cfid_value)
        fid_by_class.append(cfid_value)
        aname_by_class.append(aname)

    print("fid by class",np.array(fid_by_class).mean())





if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("forkserver")
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser()
    # Base params    
    parser.add_argument('--dataset_folder',default='../')
    parser.add_argument('--resume_path',default='')
    
    #Transformer parameters
    parser.add_argument("--ntokens_per_clip", type=int, default=15, help="Sequence length")
    parser.add_argument("--spacing",type=int,default=2, help="Sample space for temporal sequence, assuming fps=30")
    parser.add_argument("--min_window_sec",type=float,default=2,help="min window length in sec to the end of video/trimmed action")
    parser.add_argument("--nclips_dev",type=int,default=0,help="dev for starting each sample, for alignment with different observation len")
    parser.add_argument("--nclips_pred",type=int,default=0,help="num of clips to predict")
    
    
    parser.add_argument("--val_dataset", choices=["h2o","ass101","asshand"], default="fhbhands",)
    parser.add_argument("--val_split", default="val", choices=["test", "train", "val", "trainval"])
    parser.add_argument("--val_view_id", type=int, default=-1)


    # Training parameters
    parser.add_argument("--manual_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for multiprocessing")
    parser.add_argument("--pyapt_id")

    #Transformer    
    parser.add_argument('--nlayers_enc', default=9, type=int,help="Number of Encoder layers in the transformer")
    parser.add_argument('--nlayers_dec', default=9, type=int,help="Number of Decoder layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,help="Size of the embeddings (dimension of the transformer)")#
    parser.add_argument('--nheads', default=8, type=int,help="Number of attention heads inside the transformer's attentions")    
    parser.add_argument("--transformer_activation", choices=["relu","gelu"], default="gelu")



    # Evaluation params

    args = parser.parse_args()
    for key, val in sorted(vars(args).items(), key=lambda x: x[0]):
        print(f"{key}: {val}")

    #main(args)
    calculate_fid_for_two_distributions(args)
