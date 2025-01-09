import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# -------------------------------------------------------------------------
# Simple sphere normalization function
# -------------------------------------------------------------------------
def SphereNorm(self, x):
    res = x / x.norm(p=2, dim=-1, keepdim=True)
    return res

# -------------------------------------------------------------------------
# Stackable nGPTBlock from https://arxiv.org/abs/2410.01131
# -------------------------------------------------------------------------
class nGPTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        d = config.d

        self.k = nn.Linear(d, d, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.att_proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 2 * 4 * d, bias=False)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * d, d, bias=False)

        # For scaling vector Aa
        self.Aa_init = 0.05
        self.Aa_scale = 1/sqrt(d)
        self.Aa = torch.nn.Parameter(self.Aa_scale*torch.ones(d))
        # For scaling vector Am
        self.Am_init = 0.05
        self.Am_scale = 1/sqrt(d)
        self.Am = torch.nn.Parameter(self.Am_scale*torch.ones(d))
        # For scaling vector sqk
        self.sqk_init = 1.0       
        self.sqk_scale = 1/sqrt(d)
        self.sqk = torch.nn.Parameter(self.sqk_scale*torch.ones(d))
        # For scaling vector suv
        self.suv_init = 1.0
        self.suv_scale = 1.0
        self.suv = torch.nn.Parameter(self.suv_scale*torch.ones(2 * 4 * d))

    def forward(self, h):
        B, T, C = h.size()
        
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = q.view(B, T, heads, d // heads) 
        k = k.view(B, T, heads, d // heads)
        v = v.view(B, T, heads, d // heads)

        q, k = RoPE(q.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        sqk = (self.sqk * (self.sqk_init/self.sqk_scale)).view(1, 1, heads, d // heads)
        q = sqk * self.SphereNorm(q)  
        k = sqk * self.SphereNorm(k)  

        sqrt_head_dim = (d / heads) ** 0.5
        
        y = attention()
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, d)

        h_att = self.att_proj(y)

        alpha = self.Aa * (self.Aa_init / self.Aa_scale)
        alpha = torch.abs(alpha)
        
        A_norm = self.SphereNorm(h)
        B_norm = self.SphereNorm(h_att)
            
        residual = A_norm + alpha * (B_norm - A_norm)
        h = self.SphereNorm(residual)
        
        uv = self.fc(h)
        suv = (self.suv * ((self.suv_init/self.suv_scale) * (d ** 0.5))) 
        uv = suv * uv  
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        alpha = self.Am * (self.Am_init / self.Am_scale)
        alpha = torch.abs(alpha)

        A_norm = self.SphereNorm(h)
        B_norm = self.SphereNorm(h_mlp)
            
        residual = A_norm + alpha * (B_norm - A_norm)
        h = self.SphereNorm(residual)

        return h

# -------------------------------------------------------------------------
# nGPT model from https://arxiv.org/abs/2410.01131
# -------------------------------------------------------------------------
class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleDict(dict(Ein = nn.Embedding(vocab_size, d),h = nn.ModuleList([nGPTBlock() for i in range(config.n_layer)])))
        self.Eout = nn.Linear(d, vocab_size, bias=False)

        self.sz_init = 1.00
        self.sz_scale = 1/sqrt(d)
        self.sz = torch.nn.Parameter(self.sz_scale*torch.ones(config.vocab_size, dtype=torch.float32))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        x = self.blocks.Ein(idx)
        for block in self.blocks.h:
            x = block(x)

            logits = self.Eout(x)
            sz = self.sz * (self.sz_init/self.sz_scale)
            logits = sz * logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
