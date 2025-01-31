import torch
from torch import arccos, sin
import matplotlib.pyplot
from math import sqrt
from . import transformer

# Normalizes along last dimension on the hypersphere
# (s1*...*)s-1
def sphere_norm(X):
    return torch.nn.functional.normalize(X, dim=-1)

# Samples from uniform distribution on the hypersphere
def rands(*size):
    return sphere_norm(torch.randn(*size))

def test_rands(samples=50):
    xy = rands((samples, 2))
    
    # Orthogonal aspect ratio
    matplotlib.pyplot.axis("equal")
    matplotlib.pyplot.gca().add_artist(matplotlib.pyplot.Circle((0,0), radius=1, fill=False))
    matplotlib.pyplot.scatter(xy[:,0],xy[:,1])
    matplotlib.pyplot.show()
    matplotlib.pyplot.clf()

# (s1*...*)s-1
def angle(a, b, keepdim=False):
    # (s1*...*)(1)
    cosθ = (a*b).sum(dim=-1, keepdim=keepdim)
    θ = arccos(cosθ)

    return θ

class Slerp(torch.nn.Module):
    def __init__(self, α=0.5):
        super().__init__()

        self.α = α
    
    # (s1*...*)s-1
    def forward(self, a, b):
        # (s1*...*)1
        θ = angle(a, b, keepdim=True)

        slerp = ( sin((1-self.α)*θ)*a + sin(self.α*θ)*b )/sin(θ)
        
        return slerp

    def test(self, s1=10, s2=2):
        a = rands((s1, s2))
        b = rands((s1, s2))

        slerp = self(a,b)
        
        # s1
        norms = torch.linalg.vector_norm(slerp, ord=2, dim=-1)
        
        a_slerp = angle(a, slerp).rad2deg()
        slerp_b = angle(slerp, b).rad2deg()
        a_b = angle(a, b).rad2deg()
        
        print("\x1b[1m%8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("‖slerp‖", "a_slerp", "slerp_b", "a_b"))
        for i in range(s1):
            print("%8.8s %8.8s %8.8s %8.8s" % ("%f" % norms[i], "%.0f°" % a_slerp[i], "%.0f°" % slerp_b[i], "%.0f°" % a_b[i]))

        if s2==2:
            # Orthogonal aspect ratio
            matplotlib.pyplot.axis("equal")
            matplotlib.pyplot.axis([-1.5, 1.5, -1.5, 1.5])
            matplotlib.pyplot.gca().add_artist(matplotlib.pyplot.Circle((0,0), radius=1, fill=False))
            matplotlib.pyplot.scatter(a[-1,0], a[-1,1], c="red")
            matplotlib.pyplot.scatter(slerp[-1,0], slerp[-1,1], c="green")
            matplotlib.pyplot.scatter(b[-1,0], b[-1,1], c="blue")
            matplotlib.pyplot.show()
            matplotlib.pyplot.clf()

class NormLerp(torch.nn.Module):
    def __init__(self, d):
        super().__init__()

        self.d = d
        self.α = torch.nn.Parameter(torch.full((d,), 0.5))    

    # (s1*...*)s-1
    def forward(self, a, b):
        lerp = a + self.α*(b-a)
        
        return sphere_norm(lerp)

# Sec. 2.3.2 of https://arxiv.org/abs/2410.01131
class NormMHSA(torch.nn.Module):
    # Remember that d_q=d_k. Moreover, assume that d_x=d_q=d_v:=d
    def __init__(self, heads, d_head, is_causal):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.is_causal = is_causal

        # We fuse Q, K and V (and different heads) for better parallelization, as well as less code
        self.Wqkv = torch.nn.Linear(self.d, 3*self.d, False)

        self.sqk = torch.nn.Parameter((heads, d_head))
        
        # DO NOT FORGET THE OUTPUT PROJECTION
        self.Wo = torch.nn.Linear(self.d, self.d, False)

    # (batches*)context*d
    def forward(self, X):
        # (batches*)context*(3d)
        QKV = self.Wqkv(X)

        # (batches*)context*3*heads*d_head
        QKV = QKV.unflatten(dim=-1, sizes=(3, self.heads, self.d_head))
        # (batches*)3*heads*context*d_head
        QKV = QKV.movedim(-4,-2)
        # (batches*)2*heads*context*d_head, (batches*)heads*context*d_head
        QK, V = QKV[...,:1,:,:,:], QKV[...,2,:,:,:]

        QK = self.sqk.unsqueeze(1) * sphere_norm(QK)
        Q, K = QK[...,0,:,:,:], QK[...,1,:,:,:]

        # In the original paper, V is NOT normalized

        # Attention scale is handled by sqk
        # (batches*)heads*context*d_head
        pos = transformer.get_pos("rot", X.shape[-2], X.shape[-1]):
        Q_ = transformer.apply_pos("rot", Q, pos): 
        K_ = transformer.apply_pos("rot", K, pos): 
        Y = torch.nn.functional.scaled_dot_product_attention(Q_, K_, V, is_causal=self.is_causal, scale=1)
        # (batches*)context*heads*d_head
        Y = torch.movedim(Y, source=-3, destination=-2)
        # (batches*)context*d
        Y = torch.flatten(Y, start_dim=-2, end_dim=-1)

        # In the original paper, Y is NOT normalized

        Y = self.Wo(Y)

        return Y

class NormGLU(torch.nn.Module):
    def __init__(self, d0, d1):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        
        self.Wv = torch.nn.Linear(d0, d1, False)
        self.sv = torch.nn.Parameter()

        self.Wu = torch.nn.Linear(d0, d1, False)
        self.su = torch.nn.Parameter()

    def forward(self, x):
        v = sqrt(self.d0) * self.sv * self.Wv(x)

        u = self.su * self.Wu(x)

        y = torch.nn.functional.silu(v) * u

        return y

class NormMLP2L(torch.nn.Module):
    def __init__(self, d0, d1, d2):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2

        self.d1_ = (2*d1)//3
        self.l1 = NormGLU(d0, self.d1_)
        self.l2 = torch.nn.Linear(self.d1_, d2, bias)

    def forward(self, x):
        # In the original paper, a1 is NOT normalized
        a1 = self.l1(x)

        y = self.l2(a1)

        return y

class Block(torch.nn.Module):
    def __init__(self, heads, d_head, is_causal, exp_factor=4, interp="lerp"):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.is_causal = is_causal
        self.exp_factor = exp_factor
        self.d_hidden = exp_factor*self.d
        self.interp = interp

        self.norm_mhsa = NormMHSA(heads, d_head, is_causal)
        if interp=="slerp":
            self.res1 = Slerp()
            self.res2 = Slerp()
        elif interp=="lerp":
            self.res1 = NormLerp(self.d)
            self.res2 = NormLerp(self.d)
        self.norm_mlp = mlp.MLP2L(self.d, self.d_hidden, self.d, bias, act=act, dropout=dropout, l1_type=l1_type)

    def forward(self, X):
        HA = sphere_norm(self.norm_mhsa(X))
        H = self.res1(X, HA)

        HM = sphere_norm(self.norm_mlp(H))
        H = self.res2(H, HM)

        return H   

class EncBlock(Block):
    def __init__(self, heads, d_head, exp_factor=4, interp="lerp"):
        super().__init__(heads, d_head, False, exp_factor, interp)

class DecBlock(Block):
    def __init__(self, heads, d_head, exp_factor=4, interp="lerp"):
        super().__init__(heads, d_head, True, exp_factor, interp)

class nGPT(torch.nn.Module):
    def __init__(self, vocab_size=50257, num_blocks=6, heads=8, d_head=4, exp_factor=4, max_context=128, all_pos=False, interp="lerp"):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.exp_factor = exp_factor
        self.max_context = max_context
        self.all_pos = all_pos
        self.interp = interp

        self.Ein = torch.nn.Embedding(vocab_size, self.d)

        self.blocks = torch.nn.Sequential(*[DecBlock(heads, d_head, exp_factor, interp) for _ in range(num_blocks)])

        self.Eout = torch.nn.Linear(self.d, vocab_size, bias=False)
        
        self.sz = torch.nn.Parameter()

    # (batches*)context
    def forward(self, ids):
        context = ids.shape[-1]

        # (batches*)context*d
        X = self.Ein(ids)

        H = X
        for block in self.blocks:
            H_ = block(H)
            Y = apply_pos(self.pos_type, Y_, self.pos[...,:context,:]) if self.all_pos else Y_

        # (batches*)context*vocab_size
        Z = self.Eout(H)

        Z = self.sz * Z

        return Z

    # # (batches*)context
    # def forward(self, ids):
    #     context = ids.shape[-1]
    #
    #     # (batches*)context*d
    #     X = apply_pos(self.pos_type, self.emb(ids), self.pos[...,:context,:])
    #     X_ = torch.nn.functional.dropout(X, p=self.dropout, training=self.training)
    #
    #     Y = X_
    #     for block in self.blocks:
    #         Y_ = block(Y)
    #         Y = apply_pos(self.pos_type, Y_, self.pos[...,:context,:]) if self.all_pos else Y_
    #
    #     Y__ = self.norm(Y_)
    #
    #     # (batches*)context*vocab_size
    #     Z = self.linear(Y__)
    #
    #     return Z
