import torch
from . import mlp
from math import sqrt
import math

SCALE_TYPES = ["1/sqrt(d)", "1/d"]

# Pure PyTorch implementation of torch.nn.functional.scaled_dot_product_attention that returns the attention weights (lower triangular) after softmax W instead (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight

# Exposes scale and is faster (https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) than PyTorch's torch.nn.MultiheadAttention
class MHSA(torch.nn.Module):
    # Remember that d_q=d_k. Moreover, assume that d_x=d_q=d_v:=d
    def __init__(self, heads, d_head, is_causal, scale_type="1/sqrt(d)"):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.is_causal = is_causal
        self.scale_type = scale_type
        if scale_type=="1/sqrt(d)":
            self.scale = 1/sqrt(self.d)
        elif scale_type=="1/d":
            self.scale = 1/self.d

        # We fuse Q, K and V (and different heads) for better parallelization, as well as less code
        self.QKV = torch.nn.Linear(self.d, 3*self.d, bias=False)
        
        # First time I implemented MHSA, I forgot the outputs' projection :P
        self.O = torch.nn.Linear(self.d, self.d, bias=False)

    # (batches*)context*d
    def forward(self, X):
        # (batches*)context*(3d)
        QKV = self.QKV(X)

        # (batches*)context*3*heads*d_head
        QKV = QKV.unflatten(dim=-1, sizes=(3, self.heads, self.d_head))
        # (batches*)3*heads*context*d_head
        QKV = QKV.movedim(-4,-2)
        # (batches*)heads*context*d_head
        Q, K, V = QKV.unbind(-4)

        # (batches*)heads*context*d_head
        Y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal, scale=self.scale)
        # (batches*)context*heads*d_head
        Y = torch.movedim(Y, source=-3, destination=-2)
        # (batches*)context*d
        Y = torch.flatten(Y, start_dim=-2, end_dim=-1)

        Y = self.O(Y)

        return Y

    # (batches*)context*d
    def W(self, X):
        # (batches*)context*(3d)
        QKV = self.QKV(X)

        # (batches*)context*3*heads*d_head
        QKV = QKV.unflatten(dim=-1, sizes=(3, self.heads, self.d_head))
        # (batches*)3*heads*context*d_head
        QKV = QKV.movedim(-4,-2)
        # (batches*)heads*context*d_head
        Q, K, V = QKV.unbind(-4)

        # (batches*)heads*context*context
        W = scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal, scale=self.scale)

        return W

# Pre-Norm
class TransformerBlock(torch.nn.Module):
    def __init__(self, heads, d_head, is_causal, scale_type="1/sqrt(d)", exp_factor=4, dropout=0, norm_type="layer", bias=True, act=torch.nn.ReLU(), l1_type="linear"):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.is_causal = is_causal
        self.scale_type = scale_type
        self.exp_factor = exp_factor
        self.d_hidden = exp_factor*self.d
        self.dropout = dropout
        self.norm_type = norm_type
        self.bias = bias
        self.act = act
        self.l1_type = l1_type

        self.mhsa = MHSA(heads, d_head, is_causal, scale_type)
        if norm_type=="layer":
            self.norm1 = torch.nn.LayerNorm(self.d, bias=bias)
            self.norm2 = torch.nn.LayerNorm(self.d, bias=bias)
        elif norm_type=="rms":
            self.norm1 = torch.nn.RMSNorm(self.d, elementwise_affine=False)
            self.norm2 = torch.nn.RMSNorm(self.d, elementwise_affine=False)
        self.mlp = mlp.MLP2L(self.d, self.d_hidden, self.d, bias, act=act, dropout=dropout, l1_type=l1_type)

    def forward(self, X):
        Y = self.mhsa(self.norm1(X))
        Y_ = torch.nn.functional.dropout(Y, p=self.dropout, training=self.training)
        Y__ = X + Y_

        Z = self.mlp(self.norm2(Y__))
        Z_ = torch.nn.functional.dropout(Z, p=self.dropout, training=self.training)
        Z__ = Y__ + Z_

        return Z__

    def W(self, X):
        W = self.mhsa.W(self.norm1(X))

        return W

class TransformerEncBlock(TransformerBlock):
    def __init__(self, heads, d_head, scale_type="1/sqrt(d)", exp_factor=4, dropout=0, norm_type="layer", bias=True, act=torch.nn.ReLU(), l1_type="linear"):
        super().__init__(heads, d_head, False, scale_type, exp_factor, dropout, norm_type, bias, act, l1_type)

class TransformerDecBlock(TransformerBlock):
    def __init__(self, heads, d_head, scale_type="1/sqrt(d)", exp_factor=4, dropout=0, norm_type="layer", bias=True, act=torch.nn.ReLU(), l1_type="linear"):
        super().__init__(heads, d_head, True, scale_type, exp_factor, dropout, norm_type, bias, act, l1_type)

def get_sin(max_context, d):
    # [pos=0, pos=1, ...]
    poss = torch.arange(0., max_context)
    # [i=0, i=1, ...]
    js = torch.arange(0., d/2)
    # [ω0, ω1, ...]
    ωs = 1/10_000**(2*js/d)
    
    # [pos=0*ω0, pos=0*ω1, ...]
    # [pos=1*ω0, pos=1*ω1, ...]
    φs = poss[...,None] @ ωs[None,...]
    
    # max_context*d
    sin = torch.empty((max_context, d))
    sin[:,0::2] = torch.sin(φs)
    sin[:,1::2] = torch.cos(φs)

    return sin

def get_rot(max_context, d):
    # [m=0, m=1, ...]
    ms = torch.arange(0., max_context)
    # [i=0, i=1, ...]
    js = torch.arange(0., d/2)
    # [θ0, θ1, ...]
    θs = 1/10_000**(2*js/d)
    
    # [m=0*θ0, m=0*θ1, ...]
    # [m=1*θ0, m=1*θ1, ...]
    φs = ms[...,None] @ θs[None,...]
    
    # max_context*d/2
    cos = torch.cos(φs)
    sin = torch.sin(φs)
    # max_context*d
    cos = cos.repeat_interleave(2, dim=1)
    sin = sin.repeat_interleave(2, dim=1)
    
    # 2*max_context*d
    rot = torch.stack((cos,sin))

    return rot

def get_pos(pos_type, max_context, d):
    if pos_type == "sin":
        pos = get_sin(max_context, d)
    elif pos_type == "learned":
        pos = torch.randn((max_context, d))
    elif pos_type == "rot":
        pos = get_rot(max_context, d)
    
    return pos

def apply_pos(pos_type, emb, pos): 
    if pos_type in ("sin", "learned"):
        X = emb+pos
    elif pos_type == "rot":
        # (batches*)context*d
        emb_ = torch.empty_like(emb)
        emb_[...,0::2] = -emb[...,1::2]
        emb_[...,1::2] = emb[...,0::2]

        # context*d
        cos = pos[0]
        sin = pos[1]

        X = emb*cos + emb_*sin
        
    return X

def get_attention_header(transformer, blocks_interval):
    attention_header = ""
    
    for block in range(transformer.num_blocks):
        if block % blocks_interval == 0:
            for head in range(transformer.heads):
                attention_header += f"block{block}.head{head} "

    # Remove last space
    attention_header = attention_header[:-1]

    return attention_header

def get_attention(W, x, y, blocks_interval):
    attention = ""
    
    for block in range(W.shape[0]):
        if block % blocks_interval == 0:
            for head in range(W.shape[1]):
                # rows->y, columns->x
                attention +=  "%.2f " % W[block, head, y, x]

    # Remove last space
    attention = attention[:-1]

    return attention

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size=50257, num_blocks=6, heads=8, d_head=4, scale_type="1/sqrt(d)", exp_factor=4, dropout=0, pos_type="sin", max_context=128, all_pos=False, norm_type="layer", bias=True, act=torch.nn.ReLU(), l1_type="linear"):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale_type = scale_type
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.pos_type = pos_type
        self.max_context = max_context
        self.all_pos = all_pos
        self.norm_type = norm_type
        self.bias = bias
        self.act = act
        self.l1_type = l1_type

        self.emb = torch.nn.Embedding(vocab_size, self.d)
        
        pos = get_pos(pos_type, max_context, self.d)
        if pos_type in ("sin", "rot"):
            self.register_buffer("pos", pos)
        elif pos_type == "learned":
            self.pos = torch.nn.Parameter(pos)
        
        self.blocks = torch.nn.Sequential(*[TransformerDecBlock(heads, d_head, scale_type, exp_factor, dropout, norm_type, bias, act, l1_type) for _ in range(num_blocks)])
        
        if norm_type=="layer":
            self.norm = torch.nn.LayerNorm(self.d, bias=bias)
        elif norm_type=="rms":
            self.norm = torch.nn.RMSNorm(self.d, elementwise_affine=False)
        
        self.linear = torch.nn.Linear(self.d, vocab_size, bias=False)

    # (batches*)context
    def forward(self, ids):
        context = ids.shape[-1]

        # (batches*)context*d
        X = apply_pos(self.pos_type, self.emb(ids), self.pos[...,:context,:])
        X_ = torch.nn.functional.dropout(X, p=self.dropout, training=self.training)

        Y = X_
        for block in self.blocks:
            Y_ = block(Y)
            Y = apply_pos(self.pos_type, Y_, self.pos[...,:context,:]) if self.all_pos else Y_
            
        Y__ = self.norm(Y_)

        # (batches*)context*vocab_size
        Z = self.linear(Y__)

        return Z
    
    # Attention weights (after softmax)
    # (batches*)context
    def W(self, ids):
        context = ids.shape[-1]
        
        # (batches*)num_blocks*heads*context*context
        W = torch.empty(*ids.shape[:-1], self.num_blocks, self.heads, context, context)
        
        # (batches*)context*d
        X = apply_pos(self.pos_type, self.emb(ids), self.pos[...,:context,:])
        X_ = torch.nn.functional.dropout(X, p=self.dropout, training=self.training)

        Y = X_
        for i, block in enumerate(self.blocks):
            # (batches*)heads*context*context
            W[...,i,:,:,:] = block.W(Y)

            Y_ = block(Y)
            Y = apply_pos(self.pos_type, Y_, self.pos[...,:context,:]) if self.all_pos else Y_

        return W 

    # # Embeddings (before positional bias)
    # # (batches*)context
    # def Y_(self, ids):
    #     return
