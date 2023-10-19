import torch
from torch import nn
from torch.nn import functional as torch_f
import numpy as np
from meshreg.models.utils import loss_str2func

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class Quantize(nn.Module):
    def __init__(self, embedding_dim, num_embeddings,code_loss, temperature=0.07, epsilon=1e-5, decay=0.99):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.temperature=temperature
        self.decay = decay
        self.epsilon = epsilon

        _w = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("batch_cnt",torch.zeros(1))
        self.register_buffer("_w", _w)
        self.register_buffer("_ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("_ema_w", _w.clone())

        #max num of consecutive batches that a code is not used
        #self.max_unused_batches=max_unused_batches
        #self.register_buffer("usage_count", self.max_unused_batches*torch.ones(num_embeddings))
        self.register_buffer("usage_count",torch.zeros(num_embeddings))
        self.register_buffer("eval_usage_count",torch.zeros(num_embeddings))
        self.last_batch = None
        
        self.code_loss= loss_str2func()[code_loss]
        

    def forward(self, input, weights_loss, verbose=False):
        returns={}
        if verbose:
            print('Before-sync-ema',self._ema_cluster_size,)#'\n',self._ema_w[:2,:5],'\n',self._w[:2,:5])
        
        flat_input = input.reshape(-1, self.embedding_dim)
        w=self._w
        
        if verbose:
            print('********forward*********')
            #print(self._w[:3])
            print('Quantize- flat_input',flat_input.shape)#[bs,128]
            print('w',w.shape)#[128,num_embeddings]
        
        #retrieve 1nn embedding
        distances = torch.sum(torch.pow(flat_input,2),dim=1,keepdim=True) \
            - 2*torch.matmul(flat_input, w) + torch.sum(torch.pow(w,2),dim=0,keepdim=True) 
        _, flat_encoding_1nn_indices = (-distances).max(dim=1)
        
        if verbose:
            print('distances',distances.shape)#[bs,num_embeddings]
            print(torch.sum(torch.pow(flat_input,2),dim=1,keepdim=True).shape)
            print((- 2*torch.matmul(flat_input, w)).shape)
            print(torch.sum(torch.pow(w,2),dim=0,keepdim=True).shape)
            print('flat_encoding_1nn_indices',flat_encoding_1nn_indices.shape)#[bs]

        #Get quantize and compute loss
        encoding_1nn_indices = flat_encoding_1nn_indices.view(*input.shape[:-1])
        quantize = self.quantize(encoding_1nn_indices)

        #if weights_loss is None:
        #    loss=self.code_loss(quantize.detach(),flat_input)
        #else:
        #    sum_weights=torch.where(torch.sum(weights_loss)>0,torch.sum(weights_loss),torch.Tensor([1]).cuda())[0]
        #    loss=self.code_loss(quantize.detach(),flat_input,reduction='none')
        #    loss=torch.bmm(loss.view(loss.shape[0],-1,1),weights_loss.view(-1,1,1))
        #    #print('code loss',torch.sum(loss),torch.sum(weights_loss),loss.shape)
        #    loss=torch.sum(loss)/(loss.shape[1]*sum_weights)#torch.sum(weights_loss))

        assert weights_loss is None, 'weights loss shd be None'
        loss=self.code_loss(quantize.detach(),flat_input)

        quantize=input+(quantize-input).detach()


        if verbose:
            print('encoding_1nn_indices',encoding_1nn_indices.shape)#[bs]
            print('input',input.shape)#[bs,128]
            print('quantize',quantize.shape)#[bs,128]
            print('loss',loss)
        
        #to update ema
        flat_encodings = torch_f.one_hot(flat_encoding_1nn_indices, self.num_embeddings).type(flat_input.dtype)
        avg_probs=flat_encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        
        #print(avg_probs,perplexity)
        dema_cluster_size=flat_encodings.sum(0)
        dw=torch.matmul(flat_input.transpose(0,1),flat_encodings)

        if verbose:
            print('flat_encodings',flat_encodings.shape)#[bs,num_embeddings]
            print('dema_cluster_size',dema_cluster_size.shape)#[num_embeddings]
            print('dw',dw.shape)#[128,num_embeddings]
            print('avg_probs',avg_probs.shape,avg_probs)#[num_embeddings]
            #print(distances)
            print('perplexity',perplexity)

        
        returns['loss']=loss
        returns['quantize']=quantize
        returns['dema_cluster_size']=dema_cluster_size.view(1,self.num_embeddings)
        returns['dw']=dw.view(1,self.embedding_dim,self.num_embeddings)
        returns['encoding_1nn_indices']=encoding_1nn_indices
        returns['perplexity']=perplexity

        if verbose:
            print('check gradient')
            print(self._w.requires_grad,self._ema_cluster_size.requires_grad,self._ema_w.requires_grad)
            print(self.usage_count.requires_grad,self.eval_usage_count.requires_grad)
        return returns

       

    def quantize(self, encoding_indices):
        return torch_f.embedding(encoding_indices, self._w.transpose(0, 1))
    

    ###EMA updation
    def update_w_from_list(self,query_codes,tower_dema_cluster_size,tower_dw,verbose=False): 
        #use multiple gpu ;single gpu is a tensor
        #tower_dema_cluster_size=tower_dema_cluster_size.view(-1,self.num_embeddings)
        #tower_dw=tower_dw.view()
        #first update tracker
        #then update ema
        dema_cluster_size=torch.sum(tower_dema_cluster_size,dim=0)
        dw=torch.sum(tower_dw,dim=0)
        self.update_tracker(flatten_inputs=query_codes, dema_cluster_size=dema_cluster_size,verbose=verbose)
        
        if verbose:
            print('tower_dema_cluster_size/dema_cluster_size',tower_dema_cluster_size.shape,dema_cluster_size.shape)
            print('tower_dw/dw',tower_dw.shape,dw.shape)
            pre_ema_cluster_size=self._ema_cluster_size.clone().detach()
            pre_ema_w=self._ema_w.clone().detach()
        
        self._ema_cluster_size.data.mul_(self.decay).add_(dema_cluster_size, alpha=1-self.decay)
        self._ema_w.data.mul_(self.decay).add_(dw,alpha=1-self.decay)

        normalized_updated_ema_w=self._ema_w/self._ema_cluster_size.unsqueeze(0)
        self._w.data.copy_(normalized_updated_ema_w)
        if verbose:
            updated_ema_cluster_size=self.decay*pre_ema_cluster_size+(1-self.decay)*dema_cluster_size
            updated_ema_w=self.decay*pre_ema_w+(1-self.decay)*dw
            print('check ema_cluster_size',(updated_ema_cluster_size-self._ema_cluster_size).max(),(updated_ema_cluster_size-self._ema_cluster_size).min())
            print('check ema_w',(updated_ema_w-self._ema_w).max(),(updated_ema_w-self._ema_w).min())
            print('check shape',self._ema_w.shape,self._ema_cluster_size.unsqueeze(0).shape)
            print('After-sync-ema',self._ema_cluster_size,'\n',self._ema_w[:2,:5],'\n',self._w[:2,:5])
            #print(self._w[:3])
            print('*******End Update********')

    # This is to check number of activation for codes in embedding (self._w)， for training stage, utilized also for dead code revival
    def update_tracker(self,flatten_inputs,dema_cluster_size,verbose=False):
        #cur_batch=flatten_inputs.clone().detach().cpu().numpy()
        #self.last_batches.insert(0,cur_batch)
        #if len(self.last_batches)>self.max_last_batches:
        #    self.last_batches.pop()
        self.last_batch=flatten_inputs.clone().detach()

        dcount=dema_cluster_size.clone().detach()
        if verbose:
            print('last_batch',self.last_batch.shape)
            print('dcount',dcount.shape,dcount.flatten())
            print('usage_count_before',self.usage_count.shape,self.usage_count.flatten())
        self.usage_count.data.add_(dcount)
        if verbose:
            print('usage_count_after',self.usage_count.shape, self.usage_count.flatten())
    

    #Dead code revival, if necessary
    def revive_dead_entries(self,reset_cnt,verbose=False):
        assert self.last_batch is not None, 'Quantize, last batch should not be None'
        np_input=self.last_batch.detach().cpu().numpy()
        num_input_codes=np_input.shape[0]

        np_count=self.usage_count.detach().cpu().numpy()
        np_ema_cluster_size=self._ema_cluster_size.detach().cpu().numpy()
        np_ema_w=self._ema_w.detach().cpu().numpy()
        np_ema_w=np_ema_w.transpose(1,0).copy()  
        np_w=self._w.detach().cpu().numpy()
        np_w=np_w.transpose(1,0).copy()

        if verbose:
            print('np_input',np_input.shape)
            print('np_count',np_count.shape,np_count.flatten())
            print('np_ema_w',np_ema_w.shape)
            print('np_ema_cluster_size',np_ema_cluster_size.shape)

        toupdate_idx =np.where(np_count<1e-4)[0]
        print(toupdate_idx,toupdate_idx.shape)
        
        for wcode_id in toupdate_idx:  
            rep_id=np.random.choice(num_input_codes)
            np_w[wcode_id]=np_input[rep_id].copy()
            
            np_ema_w[wcode_id]=np_w[wcode_id].copy()
            np_ema_cluster_size[wcode_id]=1

        self._w.data.copy_(torch.from_numpy(np_w.transpose(1,0)))
        self._ema_w.data.copy_(torch.from_numpy(np_ema_w.transpose(1,0)))
        self._ema_cluster_size.data.copy_(torch.from_numpy(np_ema_cluster_size))
        if reset_cnt:
            self.usage_count.data.copy_(torch.from_numpy(np.zeros(self.num_embeddings)))

        if verbose:
            print('usage_count',self.usage_count.shape,self.usage_count.flatten())
            print('_ema_cluster_size',self._ema_cluster_size.shape,self._ema_cluster_size.flatten())
            print('_ema_w',self._ema_w.shape,torch.transpose(self._ema_w[:5,:],0,1))
            print('_w',self._w.shape,torch.transpose(self._w[:5,:],0,1))
            #print(self._w[:3])
            print('**********end revival************')



    # This is to check number of activation for codes in embedding (self._w), for eval stage, only to record 
    def update_eval_usage_count(self,tower_dema_cluster_size,verbose=False):
        dema_cluster_size=torch.sum(tower_dema_cluster_size,dim=0)
        dcount=dema_cluster_size.clone().detach()
        if verbose:
            print('tower_dema_cluster_size',tower_dema_cluster_size.shape)
            print('dema_cluster_size',dema_cluster_size.shape)
        self.eval_usage_count.data.add_(dcount)

    # Reset counter
    def aggregate_and_reset_eval_usage_count(self,verbose=False):
        np_count=self.eval_usage_count.detach().cpu().numpy()
        if verbose:
            print('np_count',np_count)
        num_activated=(np.where(np_count>1e-4)[0]).shape[0]
        self.eval_usage_count.data.copy_(torch.from_numpy(np.zeros(self.num_embeddings)))
        if verbose:
            print('num_activated',num_activated)
            print('reset eval_usage_count',self.eval_usage_count)
        return num_activated


    
