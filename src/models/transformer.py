import torch
import huggingface_hub
from . import mlp

# Exposes scale and is faster (https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) than PyTorch's torch.nn.MultiheadAttention
class MHSA(torch.nn.Module):
    # Remember that d_q=d_k. Moreover, assume that d_x=d_q=d_v:=d
    # scale=None <=> scale=1/d**0.5 (typical)
    def __init__(self, d, heads, is_causal, scale=None):
        super().__init__()

        self.d = d
        self.heads = heads
        self.d_head = int(d/heads)
        self.is_causal = is_causal
        self.scale = scale

        # We fuse Q, K and V (and different heads) for better parallelization, as well as less code
        self.QKV = torch.nn.Linear(d, 3*d, bias=False)

    # (batches*)context*d
    def forward(self, X):
        # (batches*)context*(3d)
        QKV = self.QKV(X)

        # (batches*)context*3*heads*d_head
        QKV = torch.unflatten(QKV, dim=-1, sizes=(3,self.heads,self.d_head))
        # (batches*)3*heads*context*d_head
        QKV = torch.movedim(QKV, source=-4, destination=-2)
        # (batches*)heads*context*d_head
        Q, K, V = torch.unbind(QKV, dim=-4)

        # (batches*)heads*context*d_head
        Y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal, scale=self.scale)
        # (batches*)context*heads*d_head
        Y = torch.movedim(Y, source=-3, destination=-2)
        # (batches*)context*d
        Y = torch.flatten(Y, start_dim=-2, end_dim=-1)

        return Y

# Pre-Norm
class TransformerBlock(torch.nn.Module):
    def __init__(self, d, heads, is_causal, scale=None, exp_factor=4, dropout=0):
        super().__init__()

        self.d = d
        self.heads = heads
        self.is_causal = is_causal
        self.scale = scale
        self.exp_factor = exp_factor
        self.d_hidden = exp_factor*d
        self.dropout = dropout

        self.norm1 = torch.nn.LayerNorm(d)
        self.mhsa = MHSA(d, heads, is_causal, scale)

        self.norm2 = torch.nn.LayerNorm(d)
        self.mlp = mlp.MLP2L(d, self.d_hidden, d)

    def forward(self, X):
        Y = self.mhsa(self.norm1(X))
        Y_ = torch.nn.functional.dropout(Y, p=self.dropout, training=self.training)
        Y__ = X + Y_

        Z = self.mlp(self.norm2(Y__))
        Z_ = torch.nn.functional.dropout(Z, p=self.dropout, training=self.training)
        Z__ = Y__ + Z_

        return Z__

class TransformerEncBlock(TransformerBlock):
    def __init__(self, d, heads, scale=None, exp_factor=4, dropout=0):
        super().__init__(d, heads, False, scale, exp_factor, dropout)

class TransformerDecBlock(TransformerBlock):
    def __init__(self, d, heads, scale=None, exp_factor=4, dropout=0):
        super().__init__(d, heads, True, scale, exp_factor, dropout)

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

class Transformer(torch.nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, vocab_size=50257, num_blocks=6, d=32, heads=8, scale=None, exp_factor=4, dropout=0, pos_type="sin", max_context=128, all_pos=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.d = d
        self.heads = heads
        self.exp_factor = exp_factor
        self.scale = scale
        self.dropout = dropout
        self.pos_type = pos_type
        self.max_context = max_context
        self.all_pos = all_pos

        self.emb = torch.nn.Embedding(vocab_size, d)
        
        pos = get_pos(pos_type, max_context, d)
        if pos_type in ("sin", "rot"):
            self.register_buffer("pos", pos)
        elif pos_type == "learned":
            self.pos = torch.nn.Parameter(pos)

        self.blocks = torch.nn.Sequential(*[TransformerDecBlock(d, heads, scale, exp_factor, dropout) for _ in range(num_blocks)])

        self.norm = torch.nn.LayerNorm(d)

        self.linear = torch.nn.Linear(d, vocab_size)

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
