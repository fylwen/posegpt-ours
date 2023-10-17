from enum import Enum, auto
import torch


class BaseQueries(Enum):
    NCAM_JOINTS3d=auto()
    JOINTS3D = auto()
    RESNET_JOINTS3D=auto()

    MIDPE=auto()

    JOINTS2D = auto()
    IMAGE = auto()
    
    ACTIONIDX = auto()
    ACTIONNAME = auto()
    VERBIDX=auto()
    VERBNAME=auto()
    OBJIDX=auto()
    OBJNAME=auto()
    
 



def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False

def map_key(qkey, is_writing_board):
    if not is_writing_board:
        return qkey
    elif qkey in BaseQueries:
        return qkey.value
    else:
        len_base=len(BaseQueries)
        return len_base+qkey.value


def transfer_batch_enum_to_int(batch):    
    t_batch={}
    
    for ckey in BaseQueries:
        t_batch[ckey.value]=batch[ckey] if ckey in batch.keys() and torch.is_tensor(batch[ckey]) \
            else torch.Tensor([0])
            
    len_base=len(BaseQueries)
    for ckey in TransQueries:
        t_batch[len_base+ckey.value]=batch[ckey] if ckey in batch.keys() and torch.is_tensor(batch[ckey]) \
            else torch.Tensor([0])
    return t_batch
    
def transfer_batch_int_to_enum(batch):
    t_batch={}
    for ckey in BaseQueries:
        if list(batch[ckey.value].size())!=[1] or ckey in [BaseQueries.OBJPOSEWEIGHT, BaseQueries.ACTIONIDX]:
            t_batch[ckey]=batch[ckey.value]
    
    len_base=len(BaseQueries)
    for ckey in TransQueries:
        if list(batch[ckey.value+len_base].size())!=[1]:            
            t_batch[ckey]=batch[ckey.value+len_base]
    
    #Forced to be right
    batch_size=list(t_batch[BaseQueries.CAMINTR].size())[0]
    t_batch[BaseQueries.SIDE]=['right']*batch_size
    return t_batch
    


    


def get_trans_queries(base_queries):
    trans_queries = []
    #if BaseQueries.OBJVERTS3D in base_queries:
    #    trans_queries.append(TransQueries.OBJVERTS3D)
    if BaseQueries.IMAGE in base_queries:
        trans_queries.append(TransQueries.IMAGE)
        trans_queries.append(TransQueries.AFFINETRANS)
        trans_queries.append(TransQueries.ROTMAT)
        trans_queries.append(TransQueries.JITTERMASK)
    if BaseQueries.JOINTS2D in base_queries:
        trans_queries.append(TransQueries.JOINTS2D)
    if BaseQueries.JOINTS3D in base_queries:
        trans_queries.append(TransQueries.JOINTS3D)
    #if BaseQueries.HANDVERTS3D in base_queries:
    #    trans_queries.append(TransQueries.HANDVERTS3D)
    #    trans_queries.append(TransQueries.CENTER3D)
    #if BaseQueries.HANDVERTS2D in base_queries:
    #    trans_queries.append(TransQueries.HANDVERTS2D)
    #if BaseQueries.OBJVERTS3D in base_queries:
    #    trans_queries.append(TransQueries.OBJVERTS3D)
    #if BaseQueries.OBJVERTS2D in base_queries:
    #    trans_queries.append(TransQueries.OBJVERTS2D)
    #if BaseQueries.OBJCORNERS3D in base_queries:
    #    trans_queries.append(TransQueries.OBJCORNERS3D)
    #if BaseQueries.OBJCORNERS2D in base_queries:
    #    trans_queries.append(TransQueries.OBJCORNERS2D)
    #if BaseQueries.OBJCANROTCORNERS in base_queries:
    #    trans_queries.append(TransQueries.OBJCANROTCORNERS)
    #if BaseQueries.OBJCANROTVERTS in base_queries:
    #    trans_queries.append(TransQueries.OBJCANROTVERTS)
    if BaseQueries.CAMINTR in base_queries:
        trans_queries.append(TransQueries.CAMINTR)
    #if BaseQueries.OBJCANVERTS in base_queries or BaseQueries.OBJCANCORNERS in base_queries:
    #    trans_queries.append(BaseQueries.OBJCANSCALE)
    #    trans_queries.append(BaseQueries.OBJCANTRANS)
    if BaseQueries.JOINTSABS25D in base_queries:
        trans_queries.append(TransQueries.JOINTSABS25D)
    return trans_queries