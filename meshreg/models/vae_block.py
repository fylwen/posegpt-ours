import torch
import torch.nn.functional as torch_f
from torch import nn

from einops import repeat
import numpy as np


from meshreg.models.transformer import Transformer_Encoder,Transformer_Decoder, PositionalEncoding

from meshreg.models.mlp import MultiLayerPerceptron

class VAE(torch.nn.Module):
    def __init__(self,  transformer_d_model,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_nlayers_enc,
                        transformer_nlayers_dec,
                        transformer_activation,
                        num_mids):

        super().__init__()
        
        self.code_dim=transformer_d_model 
        self.num_mids=num_mids
        self.mid_token_mu_enc_in=torch.nn.Parameter(torch.randn(1,num_mids,transformer_d_model))
        self.mid_token_logvar_enc_in=torch.nn.Parameter(torch.randn(1,num_mids,transformer_d_model))

        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model)
        self.encoder=Transformer_Encoder(d_model=transformer_d_model, 
                                        nhead=transformer_nhead, 
                                        num_encoder_layers=transformer_nlayers_enc,
                                        dim_feedforward=transformer_dim_feedforward,
                                        dropout=0.0,#transformer_dropout,
                                        activation=transformer_activation, 
                                        normalize_before=True)
        
        self.decoder=Transformer_Decoder(d_model=transformer_d_model, 
                                        nhead=transformer_nhead, 
                                        num_decoder_layers=transformer_nlayers_dec,
                                        dim_feedforward=transformer_dim_feedforward,
                                        dropout=0.0,#transformer_dropout,
                                        activation=transformer_activation, 
                                        normalize_before=True,
                                        return_intermediate=False)
 
    def feed_encoder(self,batch_seq_enc_in_tokens,batch_seq_enc_mask_tokens, verbose):
        batch_size=batch_seq_enc_in_tokens.shape[0]        
        batch_enc_in_mu=repeat(self.mid_token_mu_enc_in,'() n d -> b n d',b=batch_size)
        batch_enc_in_logvar=repeat(self.mid_token_logvar_enc_in,'() n d -> b n d',b=batch_size)
        batch_enc_mask_global=batch_seq_enc_mask_tokens[:,0:self.num_mids].detach().clone()
        
        #to cope with passing padded clips to P-Block when training A-Block
        batch_enc_mask_global[:,0]=False
        
        batch_seq_enc_in=torch.cat((batch_enc_in_mu,batch_enc_in_logvar,batch_seq_enc_in_tokens),dim=1)
        batch_seq_enc_pe=self.transformer_pe(batch_seq_enc_in)
        batch_seq_enc_mask=torch.cat((batch_enc_mask_global,batch_enc_mask_global,batch_seq_enc_mask_tokens),dim=1)
        
        if verbose:
            print('batch_enc_in_mu/bath_enc_in_logvar/batch_enc_mask_global',
                batch_enc_in_mu.shape,batch_enc_in_logvar.shape,batch_enc_mask_global.shape)#[bs,num_mids,512],[bs,num_mids,512],[bs,num_mids]
            print('batch_seq_enc_in/batch_seq_enc_pe/batch_seq_enc_mask',
                batch_seq_enc_in.shape,batch_seq_enc_pe.shape,batch_seq_enc_mask.shape)#[bs,2*num_mids+3*len_tokens,512]x2,[bs,2*num_mids+3*len_tokens]
            print(batch_seq_enc_mask[0])
        
        
        batch_seq_enc_out, _ =self.encoder(src=batch_seq_enc_in, src_pos=batch_seq_enc_pe, key_padding_mask=batch_seq_enc_mask,verbose=False)      
        
        batch_enc_out_mu =batch_seq_enc_out[:,:self.num_mids]
        batch_enc_out_logvar=batch_seq_enc_out[:,self.num_mids:2*self.num_mids]
        batch_seq_enc_out_tokens=batch_seq_enc_out[:,2*self.num_mids:]
        
        if verbose:
            print("batch_seq_enc_out",batch_seq_enc_out.shape)#[bs,2*num_mids+3*len_tokens,512]
            print("batch_enc_out_mu/batch_enc_out_logvar/batch_seq_enc_out_tokens",batch_enc_out_mu.shape,batch_enc_out_logvar.shape,batch_seq_enc_out_tokens.shape)#[bs,num_mids,512],[bs,num_mids,512],[bs,3*len_tokens,512]

        return batch_enc_out_mu,batch_enc_out_logvar,batch_seq_enc_out_tokens
    


    def feed_decoder(self,batch_seq_dec_query,batch_seq_dec_mem,batch_seq_dec_mem_mask,batch_seq_dec_tgt_key_padding_mask,verbose):
        batch_size=batch_seq_dec_mem.shape[0]        
        batch_seq_dec_mem_pe=self.transformer_pe(batch_seq_dec_mem)

        if batch_seq_dec_query is None:
            batch_seq_dec_query=torch.zeros(batch_size,self.ntokens_pred,self.code_dim,
                                    dtype=batch_seq_dec_mem.dtype,device=batch_seq_dec_mem.device)
            batch_seq_dec_query=self.transformer_pe(batch_seq_dec_query)
        batch_seq_dec_query_pe=self.transformer_pe(batch_seq_dec_query)
        if verbose:            
            print('batch_seq_dec_query/pe',batch_seq_dec_query.shape,batch_seq_dec_query_pe.shape)#[bs,len_p,512]x2
            print('batch_seq_dec_mem/pe/mask',batch_seq_dec_mem.shape,batch_seq_dec_mem_pe.shape,batch_seq_dec_mem_mask.shape)
            #[bs,1+len_o,512]x2, [bs,1+len_o]; #[bs,1,dim]x2, [bs,1]
            print(batch_seq_dec_mem_mask[0]) 
            print(batch_seq_dec_tgt_key_padding_mask[1])
            
        batch_seq_dec_out_tokens,_=self.decoder(tgt=batch_seq_dec_query, memory=batch_seq_dec_mem,\
                            tgt_key_padding_mask=batch_seq_dec_tgt_key_padding_mask, memory_key_padding_mask=batch_seq_dec_mem_mask, \
                            tgt_pos=batch_seq_dec_query_pe, memory_pos=batch_seq_dec_mem_pe)
        
        if verbose:
            print("batch_seq_dec_out_tokens",batch_seq_dec_out_tokens.shape)#[bs,len_p,512]
        return batch_seq_dec_out_tokens

    
    def reparameterize(self, mu, logvar,factor=1.):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        std = std * factor
        eps = torch.randn_like(std)
        return eps * std + mu
            

        

    def compute_kl_loss(self, mu1, logvar1,verbose=False):
        #assume \var2=1
        kld_loss = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1)
        
        if verbose:
            print("mu1/logvar1",mu1.shape,logvar1.shape)#[bs,512]x2
            print("kld_loss",kld_loss.shape)#[bs]
        assert (kld_loss>-1e-8).all(),"kld_loss shd >0"

        kld_loss=torch.mean(kld_loss)
        return kld_loss
