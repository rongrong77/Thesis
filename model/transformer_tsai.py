# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/108b_models.TST.ipynb (unless otherwise specified).
from typing import Optional
import torch
import torch.nn.functional as F
from fastai.layers import SigmoidRange, Flatten
from fastcore.basics import ifnone
from torch import Tensor
from torch.nn import Module
from fastai.torch_core import Module, default_device
from torch import nn
import math
from collections import OrderedDict

__all__ = ['TST']

# Cell
from tsai import *
from tsai.utils import *
from tsai.models.layers import *
from tsai.models.utils import *



# Internal Cell
class _ScaledDotProductAttention(Module):
    def __init__(self, d_k:int): self.d_k = d_k


    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor]=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                         # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Mask (optional)
        if mask is not None: scores.masked_fill_(mask, -1e9)

        # SoftMax
        attn = F.softmax(scores, dim=-1)                                    # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                     # context: [bs x n_heads x q_len x d_v]

        return context, attn

# Internal Cell
class _MultiHeadAttention(Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int):
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Optional[Tensor]=None):
        device = Q.device
        self.W_Q = self.W_Q.to(device)
        self.W_K = self.W_K.to(device)
        self.W_V = self.W_V.to(device)
        self.W_O = self.W_O.to(device)

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)          # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                  # context: [bs x q_len x d_model]

        return output, attn

# Internal Cell
class _TSTEncoderLayer(Module):
    def __init__(self, q_len:int, d_model:int, n_heads:int, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, res_dropout:float=0.1,
                 activation:str="gelu"):

        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.batchnorm_attn = nn.BatchNorm1d(q_len)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(q_len)

    def forward(self, src:Tensor, mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        device = src.device
        self.to(device)
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_attn(src)      # Norm: batchnorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_ffn(src) # Norm: batchnorm

        return src

    def _get_activation_fn(self, activation):
        if activation == "relu": return nn.ReLU()
        elif activation == "gelu": return nn.GELU()
        else: return activation()
#         raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')

# Internal Cell
class _TSTEncoder(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu', n_layers=1):

        self.layers = nn.ModuleList([_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                                            activation=activation) for i in range(n_layers)])

    def forward(self, src):
        device = src.device
        output = src
        for mod in self.layers: 
            mod = mod.to(device)
            output = mod(output)
        return output


class DecoderFC(Module):
    def __init__(self, c_out, output_size1):
        """Set the hyper-parameters and build the layers."""
        self.fc_classifier = nn.Linear(output_size1, c_out)

    def forward(self, x):
        if x.ndim>2:
            x = torch.flatten(x, 1)
        out = F.softmax(self.fc_classifier(x))
        return out

# Cell
class TransformerTSAI(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int, max_seq_len:Optional[int]=None,
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, res_dropout:float=0.1, act:str="gelu", fc_dropout:float=0.,
                 y_range:Optional[tuple]=None, verbose:bool=False, classification:bool=False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            res_dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len: # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding), Conv1d(c_in, d_model, kernel_size=tr_factor, stride=tr_factor))
            pv(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n', verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs) # Eq 2
            pv(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        else:
            self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        W_pos = torch.zeros((q_len, d_model), device=default_device())
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.res_dropout = nn.Dropout(res_dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout, activation=act, n_layers=n_layers)
        self.flatten = Flatten()

        # Encoder emmbedding length
        self.head_nf = q_len * d_model

        # FC dropout
        self.fc_dropout = nn.Dropout(fc_dropout)

        # HEADS Regression
        # 1) Linear Head: [bs , q_len , d_model] --> [bs , q_len , c_out]
        # self._linear = nn.Linear(d_model, c_out)
        # 2) Create head based on TSAI [bs , q_len x d_model] --> [bs , q_len , c_out]
        # self.head = self.create_head(self.head_nf, c_out, fc_dropout=fc_dropout, y_range=y_range)
        # 3) Create multi-heads based on nn.Seq function [bs , q_len x d_model] --> [bs , q_len , c_out]
        # self.heads = self.create_heads(self.c_out, self.head_nf, self.seq_len)
        # 4) Create multi-heads [bs , q_len x d_model] --> [bs , q_len , c_out]
        self.create_multiheads(self.c_out, self.head_nf, self.seq_len)


    def create_multiheads(self, c_out, output_size1, output_size2):
        if c_out == 1:
            self.fc2_1 = nn.Linear(output_size1, output_size2)
        elif c_out == 2:
            self.fc2_1 = nn.Linear(output_size1, output_size2)
            self.fc2_2 = nn.Linear(output_size1, output_size2)
        elif c_out == 3:
            self.fc2_1 = nn.Linear(output_size1, output_size2)
            self.fc2_2 = nn.Linear(output_size1, output_size2)
            self.fc2_3 = nn.Linear(output_size1, output_size2)

    def create_head(self, nf, c_out, fc_dropout=0., y_range=None, **kwargs):
        layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def create_head_block(self, h, input_length, output_length):
        head_block = nn.Sequential()
        head_block.add_module('relu{}'.format(h), nn.ReLU(inplace=True))
        head_block.add_module('fc2_{}'.format(h), nn.Linear(input_length, output_length))
        return head_block

    def create_heads(self, c_out, input_length, output_length):
        heads = []
        for c in range(c_out):
            heads.append(self.create_head_block(c, input_length, output_length))
        return heads

    def forward(self, x:Tensor, mask:Optional[Tensor]=None) -> Tensor:  # x: [bs x nvars x q_len]
        x = x.permute(0,2,1)
        # Input encoding
        device = x.device
        self.W_pos.data = self.W_pos.data.to(device)
        if self.new_q_len: u = self.W_P(x).transpose(2,1).to(device) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)).to(device) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]
        # Positional encoding
        u = self.res_dropout(u + self.W_pos)
        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        if self.flatten is not None: z = self.flatten(z)                # z: [bs x q_len * d_model]

        out = self.fc_dropout(z)
        if self.c_out == 1:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out = [out1]
        elif self.c_out == 2:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out = [out1, out2]
        elif self.c_out == 3:
            out1 = F.relu(out)
            out1 = self.fc2_1(out1)
            out2 = F.relu(out)
            out2 = self.fc2_2(out2)
            out3 = F.relu(out)
            out3 = self.fc2_3(out3)
            out = [out1, out2, out3]
        regression_out = torch.stack(out, dim=0).permute(1, 2, 0)
        return regression_out