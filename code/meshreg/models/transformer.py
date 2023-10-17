# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy, math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable

class Transformer_Encoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        print('Transformer- Use pre-normalize', normalize_before)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_pos, key_padding_mask=None, attn_mask=None, verbose=False):
        if verbose:
            print('Transformer- before permute, src/src_pos',src.size(),src_pos.size())
            #should be([batch_size, len_seq, d_model])
        src=src.permute(1,0,2)
        src_pos=src_pos.permute(1,0,2)

        if verbose:
            print('Transformer- after permute, src/src_pos',src.size(),src_pos.size())
            #should be([len_seq, batch_size, d_model])
            print('Transformer- attn_mask','None' if attn_mask is None else attn_mask.size())#[batch_size, len_seq]
            print('Transformer- key_padding_mask','None' if key_padding_mask is None else key_padding_mask.size())#[batch_size, len_seq]
        

        memory,list_attn_maps = self.encoder(src, src_attn_mask=attn_mask, src_key_padding_mask=key_padding_mask, src_pos=src_pos, verbose=verbose)


        if verbose:
            print('Transformer- before permute, memory',memory.size())#[len_seq, batch_size,d_model],
        memory=memory.permute(1,0,2)
        if verbose:
            print('Transformer- after permute memory',memory.size())#[batch_size, len_seq, d_model],
        return memory, list_attn_maps#memory.permute(1,2,0)hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None, verbose=False):
        output = src
        list_attn_maps=[]

        for layer in self.layers:
            output,attn_map = layer(output, src_attn_mask=src_attn_mask,
                           src_key_padding_mask=src_key_padding_mask, src_pos=src_pos,verbose=verbose)
            list_attn_maps.append(attn_map)
        if self.norm is not None:
            output = self.norm(output)

        return output,list_attn_maps

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        #q is with (L,N,E):L is number of words, N is batch size, E is embedding dimension,
        #k (S,N,E), value(S,N,E); attn_mask:(L,S); key_padding_mask(N,S)
        #output is attn_output (L,N,E) and attn_output_weights(N,L,S)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_attn_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None, verbose=False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, src_pos)
        
        if verbose:
            print('****Start TransformerEncoderLayer****')
            print('q/k/src2',q.size(),k.size(),src2.size())# should be [L,N,E], [S,N,E], [S,N,E],
            #L is number of words, N is batch size, E is embedding dimension,
            print('src_mask/src_key_padding_mask','None' if src_mask is None else src_mask.size(),
                        'None' if src_key_padding_mask is None else src_key_padding_mask.size())#key padding is [N,S]


        src2,attn_map = self.self_attn(q, k, value=src2, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)#[0]
 
        src = src + self.dropout1(src2)
        
        #print(torch.nonzero(self.dropout1(src2)).shape)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if verbose:
            print('output-src',src.size())#should be [L,N,E] 
            print('****End TransformerEncoderLayer****')
        return src,attn_map

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None, verbose=False):
        if self.normalize_before:
            return self.forward_pre(src, src_attn_mask, src_key_padding_mask, src_pos, verbose=verbose)
        return self.forward_post(src, src_attn_mask, src_key_padding_mask, src_pos)


class Transformer_Decoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, tgt_pos, memory_pos,  
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, verbose=False):
       
        
        if verbose:
            print('*****Transformer- before permute, tgt/tgt_pos',tgt.size(), tgt_pos.size())#[bs,seq_len,512],[bs,seq_len,512]
            print('memory/memory_pos',memory.size(),memory_pos.size())#[bs,2*seq_len+1,512],[bs,2*seq_len+1,512]
            
        tgt=tgt.permute(1,0,2)
        tgt_pos=tgt_pos.permute(1,0,2)
        memory=memory.permute(1,0,2)
        memory_pos=memory_pos.permute(1,0,2)


        if verbose:
            print('Transformer- after permute, tgt/memory',tgt.size(),memory.size())#should be([len_seq, batch_size, d_model])
            

        hs, list_attn_maps = self.decoder(tgt, memory, 
                        tgt_attn_mask=tgt_attn_mask, memory_attn_mask=memory_attn_mask, 
                        tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                        tgt_pos=tgt_pos, memory_pos=memory_pos,verbose=verbose)

        if verbose:
            print('Transformer- before permute, hs',hs.size())#[len_seq, batch_size,d_model]
        hs=hs.permute(1,0,2)
        if verbose:
            print('Transformer- after permute hs',hs.size())#[batch_size, len_seq, d_model]
        return hs, list_attn_maps#.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None, verbose=False):
        output = tgt

        list_attn_maps=[]

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_attn_mask=tgt_attn_mask,
                           memory_attn_mask=memory_attn_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_pos=tgt_pos, memory_pos=memory_pos, verbose=verbose)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
            
        return output,list_attn_maps#.unsqueeze(0)



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,
                     memory_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, tgt_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_attn_mask: Optional[Tensor] = None,
                    memory_attn_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None,verbose=False):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)

        if verbose:
            print('****Start TransformerDecoderLayer****')
            print('q/k/tgt2',q.size(),k.size(),tgt.size())# should be [L,N,E], [S,N,E], [S,N,E], S=L here
            #L is number of words, N is batch size, E is embedding dimension,
            print('tgt_attn_mask/tgt_key_padding_mask','None' if tgt_attn_mask is None else tgt_attn_mask.size(),
                        'None' if tgt_key_padding_mask is None else tgt_key_padding_mask.size())

        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_attn_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]#tgt2 shd be [L,N,E]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if verbose:
            print('tgt2/memory',tgt2.size(),memory.size())#should be [L,N,E], [S,N,E],           
            print('memory_attn_mask/memory_key_padding_mask','None' if memory_attn_mask is None else memory_attn_mask.size(),
                        'None' if memory_key_padding_mask is None else memory_key_padding_mask.size())

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                   key=self.with_pos_embed(memory, memory_pos),
                                   value=memory, attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if verbose:
            print('return tgt',tgt.shape)#shd be [L,N,E]
        return tgt

    def forward(self, tgt, memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None, verbose=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_attn_mask, memory_attn_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, tgt_pos, memory_pos, verbose)
        return self.forward_post(tgt, memory, tgt_attn_mask, memory_attn_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, tgt_pos, memory_pos)








def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


#Below is the position encoding
#Attention is all you need
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):#dropout, 
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x_pe = Variable(self.pe[:, :x.size(1)],requires_grad=False)
        x_pe = x_pe.repeat(x.size(0),1,1)
        return x_pe

        #x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        #return self.dropout(x)


    def forward_idx(self,idx):
        ori_idx_shape=idx.shape
        idx=torch.flatten(idx)

        x_pe=Variable(torch.index_select(self.pe,1,idx),requires_grad=False)
        x_pe=x_pe.view(ori_idx_shape+(x_pe.shape[-1],))

        return x_pe
        
        

class PositionalEncoding2D(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):#dropout, 
        super(PositionalEncoding2D, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #x_pe = Variable(self.pe[:, :x.size(1)],requires_grad=False)
        #x_pe = x_pe.repeat(x.size(0),1,1)
        h, w = x.shape[-2:]
        #i = torch.arange(w, device=x.device)
        #j = torch.arange(h, device=x.device)
        w_emb = Variable(self.pe[:, :w],requires_grad=False)#self.col_embed(i)
        h_emb = Variable(self.pe[:, :h],requires_grad=False)#self.row_embed(j)
        #print(self.pe.shape, w_emb.shape,h_emb.shape)
        pos = torch.cat([
            w_emb.repeat(h, 1, 1),
            h_emb.permute(1,0,2).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

        #x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        #return self.dropout(x)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        #x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
