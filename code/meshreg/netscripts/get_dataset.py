import numpy as np

from meshreg.datasets import (#seqset_htt,
    seqset_motion,
    seqset_context,
    ass101,
    asshand,
    h2ohands_untrimmed
)

from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import torch
import itertools


class ConcatLoader:
    def __init__(self, dataloaders):
        self.loaders = dataloaders
    
    def __iter__(self):        
        self.idx_cycle = itertools.cycle(list(range(self.__len__())))
        self.iters = [iter(loader) for loader in self.loaders]
        return self
    
    def __next__(self):
        subbatches=[]
        for idx in range(len(self.loaders)):
            batch=next(self.iters[idx])
            subbatches.append(batch)
        
        ret_batch={}

        for key in subbatches[0].keys():
            if isinstance(subbatches[0][key],torch.Tensor):
                ret_batch[key]=torch.cat([subbatches[i][key] for i in range(len(subbatches))],dim=0)
                #print(key,subbatches[0][key].shape,ret_batch[key].shape)
            else:
                ret_batch[key]=[]
                for i in range(0,len(subbatches)):
                    ret_batch[key]+=subbatches[i][key]                
                #print(key,len(ret_batch[key]),ret_batch[key])
        return ret_batch

    def __len__(self):
        return min(len(loader) for loader in self.loaders)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



def get_dataset_motion(
    list_datasets_name,
    list_splits,
    list_view_ids,
    dataset_folder,
    use_same_action,
    ntokens_per_clip,
    spacing,
    nclips,
    min_window_sec,
    is_shifting_window,
    dict_is_aug,
    **kwargs,
):
    mode="context" if use_same_action else "motion"
    nclips_dev=0 if "nclips_dev" not in kwargs else kwargs["nclips_dev"]
    buf_sec=0 if "buf_sec" not in kwargs else kwargs["buf_sec"]
    shift_as_htt=False if "shift_as_htt" not in kwargs else kwargs["shift_as_htt"]

    aug_obsv_len=False if "aug_obsv_len" not in dict_is_aug else dict_is_aug["aug_obsv_len"]
    aug_img=False if "aug_img" not in dict_is_aug else dict_is_aug["aug_img"]


    pose_datasets=[]
    for dataset_name,split,view_id in zip(list_datasets_name,list_splits,list_view_ids):
        if dataset_name=='ass101':
            pose_dataset=ass101.Ass101(dataset_folder=dataset_folder,
                                    split=split,
                                    view_id=view_id,
                                    mode=mode,
                                    ntokens_per_seq=ntokens_per_clip,
                                    spacing=spacing,
                                    is_shifting_window=is_shifting_window,
                                    action_taxonomy_to_use=kwargs["action_taxonomy_to_use"],
                                    max_samples=kwargs["max_samples"],
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)
            
            pose_dataset.load_dataset()
        elif dataset_name=='asshand':
            pose_dataset=asshand.AssHand(dataset_folder=dataset_folder,
                                    split=split,
                                    view_id=view_id,
                                    mode=mode,
                                    ntokens_per_seq=ntokens_per_clip,
                                    spacing=spacing,
                                    is_shifting_window=is_shifting_window,
                                    action_taxonomy_to_use=kwargs["action_taxonomy_to_use"],
                                    max_samples=kwargs["max_samples"],
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)
            
            pose_dataset.load_dataset()
        elif dataset_name=="h2o":
            pose_dataset=h2ohands_untrimmed.H2OHands(dataset_folder=dataset_folder,
                                    split=split, 
                                    view_id=view_id,
                                    mode=mode,
                                    ntokens_per_seq=ntokens_per_clip,
                                    spacing=spacing, 
                                    is_shifting_window=is_shifting_window,
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)


            
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        pose_datasets.append(pose_dataset)

    dataset = seqset_motion.SeqSet(        
        list_pose_datasets=pose_datasets,
        queries=pose_datasets[0].all_queries,
        ntokens_per_clip=ntokens_per_clip,
        spacing=spacing,
        nclips=nclips,
        nclips_dev=nclips_dev,
        aug_obsv_len=aug_obsv_len,
        aug_img=aug_img)

    return dataset




def get_dataset_context(
    list_datasets_name,
    list_splits,
    list_view_ids,
    dataset_folder,
    ntokens_per_clip,
    spacing,
    max_ntokens_per_video,
    const_ntokens_obsv,
    const_ntokens_pred,
    min_window_sec,
    is_shifting_window,
    strict_untrimmed_video,
    **kwargs,  
):
    nclips_dev=0 if "nclips_dev" not in kwargs else kwargs["nclips_dev"]
    buf_sec=0 if "buf_sec" not in kwargs else kwargs["buf_sec"]
    shift_as_htt=False if "shift_as_htt" not in kwargs else kwargs["shift_as_htt"]
    all_queries=None if "all_queries" not in kwargs else kwargs["all_queries"]

    pose_datasets=[]
    ntokens_per_clip2=ntokens_per_clip if not is_shifting_window or max_ntokens_per_video!=const_ntokens_obsv else ntokens_per_clip*max_ntokens_per_video
    for dataset_name,split,view_id in zip(list_datasets_name,list_splits,list_view_ids):
        if dataset_name=='ass101':
            pose_dataset=ass101.Ass101(dataset_folder=dataset_folder,
                                    split=split,
                                    view_id=view_id,
                                    mode="context",
                                    ntokens_per_seq= ntokens_per_clip2,
                                    spacing=spacing,
                                    is_shifting_window=is_shifting_window,
                                    action_taxonomy_to_use=kwargs["action_taxonomy_to_use"],
                                    max_samples=kwargs["max_samples"],
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)
            pose_dataset.load_dataset()
        elif dataset_name=='asshand':
            pose_dataset=asshand.AssHand(dataset_folder=dataset_folder,
                                    split=split,
                                    view_id=view_id,
                                    mode="context",
                                    ntokens_per_seq=ntokens_per_clip2,
                                    spacing=spacing,
                                    is_shifting_window=is_shifting_window,
                                    action_taxonomy_to_use=kwargs["action_taxonomy_to_use"],
                                    max_samples=kwargs["max_samples"],
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)
            
            pose_dataset.load_dataset()
        elif dataset_name=="h2o":
            pose_dataset=h2ohands_untrimmed.H2OHands(dataset_folder=dataset_folder,
                                    split=split, 
                                    view_id=view_id,
                                    mode="context",
                                    ntokens_per_seq=ntokens_per_clip2,
                                    spacing=spacing, 
                                    is_shifting_window=is_shifting_window,
                                    buf_sec=buf_sec,
                                    min_window_sec=min_window_sec,
                                    shift_as_htt=shift_as_htt)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        pose_datasets.append(pose_dataset)


    dataset = seqset_context.SeqSet(        
        list_pose_datasets=pose_datasets,
        queries=pose_datasets[0].all_queries if all_queries is None else all_queries,
        ntokens_per_clip=ntokens_per_clip,
        spacing=spacing,
        capacity_ntokens=max_ntokens_per_video,
        const_ntokens_obsv=const_ntokens_obsv,
        const_ntokens_pred=const_ntokens_pred,
        nclips_dev=nclips_dev,
        strict_untrimmed_video=strict_untrimmed_video,)

    return dataset