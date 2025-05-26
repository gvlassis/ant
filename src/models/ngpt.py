import torch
from torch import arccos, sin
import matplotlib.pyplot
from math import sqrt
from . import mlp
from . import transformer

# Samples from uniform distribution on the hypersphere
def rands(*size):
    return mlp.sphere_norm(torch.randn(*size))

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
    def __init__(self):
        super().__init__()

        self.α = 0.05
    
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
        
        self.α_init = 0.05
        self.α_scale = 1/sqrt(d)
        α = torch.full((d,), self.α_scale)
        self.α = torch.nn.Parameter(α)

    # (s1*...*)s-1
    def forward(self, a, b):
        α = (self.α_init/self.α_scale)*self.α
        lerp = a + α*(b-a)
        
        return mlp.sphere_norm(lerp)

# Sec. 2.3.2 of https://arxiv.org/abs/2410.01131
class NormMHSA(torch.nn.Module):
    def __init__(self, heads, d_head):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale = sqrt(d_head)
        
        # Packing QKV gives negligible (~1%) speed gains, while not allowing GQA, hurting code clarity and having side effects with μP
        self.lq = torch.nn.Linear(self.d, self.d, bias=False)
        self.lk = torch.nn.Linear(self.d, self.d, bias=False)
        self.lv = torch.nn.Linear(self.d, self.d, bias=False)
        
        self.sqk_init = 1
        self.sqk_scale = 1/sqrt(d_head)
        sqk = torch.full((heads, d_head), self.sqk_scale)
        self.sqk = torch.nn.Parameter(sqk)
        
        self.lo = torch.nn.Linear(self.d, self.d, bias=False)

    # (batches*)context*d
    def forward(self, X, causal=None, rope=None, swa=None, return_A=False, backend="flash"):
        # (batches*)context*d
        Q = self.lq(X)
        K = self.lk(X)
        V = self.lv(X)

        # (batches*)context*heads*d_head
        Q = Q.unflatten(dim=-1, sizes=(self.heads, self.d_head))
        K = K.unflatten(dim=-1, sizes=(self.heads, self.d_head))
        V = V.unflatten(dim=-1, sizes=(self.heads, self.d_head))

        # (batches*)heads*context*d_head
        Q = Q.movedim(-3,-2)
        K = K.movedim(-3,-2)
        V = V.movedim(-3,-2)
    
        Q = transformer.apply_rope(Q,rope)
        K = transformer.apply_rope(K,rope)

        # Normalize after RoPE
        sqk = (self.sqk_init/self.sqk_scale)*self.sqk
        Q = sqk.unsqueeze(1) * mlp.sphere_norm(Q)
        K = sqk.unsqueeze(1) * mlp.sphere_norm(K)
        # In the original paper, V is NOT normalized
        
        # (batches*)heads*context*d_head
        if not return_A:
            Y = transformer.sdpa_wrapper(Q, K, V, causal, swa=swa, scale=self.scale, return_A=return_A, backend=backend)
        else:
            Y, A__, A_, A = transformer.sdpa_wrapper(Q, K, V, causal, swa=swa, scale=self.scale, return_A=return_A, backend=backend)
        # (batches*)context*heads*d_head
        Y = Y.movedim(-3,-2)
        # (batches*)context*d
        Y = Y.flatten(-2,-1)

        Y = self.lo(Y)
        
        if not return_A:
            return Y
        else:
            return Y, A__, A_, A

class NormGLU(torch.nn.Module):
    def __init__(self, d0, d1):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        
        self.lv = torch.nn.Linear(d0, d1, bias=False)
        self.sv_init = 1.0
        self.sv_scale = 1.0
        sv = torch.full((self.d1,), self.sv_scale)
        self.sv = torch.nn.Parameter(sv)

        self.lu = torch.nn.Linear(d0, d1, bias=False)
        self.su_init = 1.0
        self.su_scale = 1.0
        su = torch.full((self.d1,), self.su_scale)
        self.su = torch.nn.Parameter(su)

    def forward(self, x):
        sv = (self.sv_init/self.sv_scale)*self.sv
        v = sqrt(self.d0) * sv * self.lv(x)
        
        su = (self.su_init/self.su_scale)*self.su
        u = su * self.lu(x)

        y = torch.nn.functional.silu(v) * u

        return y

class NormMLP2L(torch.nn.Module):
    def __init__(self, d0, d1, d2):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2

        self.l1 = NormGLU(d0, d1)
        self.l2 = torch.nn.Linear(d1, d2, bias=False)

    def forward(self, x):
        # In the original paper, a1 is NOT normalized
        a1 = self.l1(x)

        y = self.l2(a1)

        return y

class Block(torch.nn.Module):
    def __init__(self, heads, d_head, exp_factor=4, interp="lerp"):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.exp_factor = exp_factor
        self.d_hidden = exp_factor*self.d
        self.interp = interp
        
        self.norm_mhsa = NormMHSA(heads, d_head)
        if interp=="slerp":
            self.interp1 = Slerp()
            self.interp2 = Slerp()
        elif interp=="lerp":
            self.interp1 = NormLerp(self.d)
            self.interp2 = NormLerp(self.d)
        self.norm_mlp = NormMLP2L(self.d, self.d_hidden, self.d)

    def forward(self, X, causal=None, rope=None, swa=None, return_A=False, backend="flash"):
        if not return_A:
            HA = mlp.sphere_norm(self.norm_mhsa(X, causal, rope, swa, return_A, backend))
        else:
            HA, A__, A_, A = mlp.sphere_norm(self.norm_mhsa(X, causal, rope, swa, return_A, backend))
        H = self.interp1(X, HA)
        
        HM = mlp.sphere_norm(self.norm_mlp(H))
        H = self.interp2(H, HM)

        if not return_A:
            return H
        else:
            return H, A__, A_, A

class nGPT(torch.nn.Module):
    def __init__(self, vocab_size=50304, num_blocks=12, interp="lerp", heads=12, d_head=64, is_causal=True, window=None, backend="flash", exp_factor=4, std=0.02, test=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.interp = interp
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.is_causal = is_causal
        self.window = window
        self.backend = backend
        self.exp_factor = exp_factor

        self.Ein = torch.nn.Embedding(vocab_size, self.d)
        
        self.blocks = torch.nn.Sequential(*[Block(heads, d_head, exp_factor, interp) for _ in range(num_blocks)])
        
        self.Eout = torch.nn.Linear(self.d, vocab_size, bias=False)
        
        self.sz_init = 1
        self.sz_scale = 1/sqrt(self.d)
        sz = torch.full((vocab_size,), self.sz_scale)
        self.sz = torch.nn.Parameter(sz)

        self.init(std, test)

    def init(self, std=0.02, test=False):
        if test: print("\x1b[1m%36.36s %8.8s %8.8s %8.8s\x1b[0m" % ("parameter_name", "suffix", "mean", "std"))
        for parameter_name, parameter in self.named_parameters():
            parent_name, _, suffix = parameter_name.rpartition(".")
            parent = self.get_submodule(parent_name)
            
            torch.nn.init.normal_(parameter, 0, std)
            
            if test:
                print("%36.36s %8.8s %8.8s %8.8s\x1b[0m" % (parameter_name, suffix, "%f" % parameter.mean(), "%f" % parameter.std()))

    # (batches*)context
    def forward(self, ids, return_A=False, return_emb=False):
        context = ids.shape[-1]

        if return_A:
            # (batches*)num_blocks*heads*context*context
            A__ = torch.empty(*ids.shape[:-1], self.num_blocks, self.heads, context, context)
            A_ = torch.empty_like(A__)
            A = torch.empty_like(A__)
        
        # (batches*)context*d
        X = self.Ein(ids)

        if return_emb:
            # (batches*)(num_blocks+1)*context*d
            embeddings = torch.empty(*ids.shape[:-1], self.num_blocks+1, context, self.d)
            embeddings[...,0,:,:] = X
        
        # Recompute in every batch in case context changes
        if self.is_causal:
            if self.backend=="pytorch":
                causal = transformer.get_causal(context).to(ids.device)
            elif self.backend=="flash":
                causal = True
            elif self.backend=="flex":
                causal = causal_mod
            elif self.backend=="cudnn":
                # right_bound
                causal = 0
        else: causal = None

        rope = transformer.get_rope(context, self.d_head).to(ids.device)

        if self.window is not None:
            if self.backend=="pytorch":
                swa = transformer.get_swa(context, self.window).to(ids.device)
            elif self.backend=="flash":
                swa = (self.window, self.window)
            elif self.backend=="flex":
                swa = swa_mod
            elif self.backend=="cudnn":
                # left_bound
                swa = self.window
        else: swa = None

        H = X
        for i, block in enumerate(self.blocks):
            if not return_A:
                H = block(H, causal, rope, swa, return_A, self.backend)
            else:
                H, A__[...,i,:,:,:], A_[...,i,:,:,:], A[...,i,:,:,:] = block(H, causal, rope, swa, return_A, self.backend)

            if return_emb:
                embeddings[...,i+1,:,:] = H

        # (batches*)context*vocab_size
        Z = self.Eout(H)
        
        sz = (self.sz_init/self.sz_scale)*self.sz
        Z = sz * Z

        if not return_A:
            if not return_emb:
                return Z
            else:
                return Z, embeddings
        else:
            if not return_emb:
                return Z, A__, A_, A
            else:
                return Z, A__, A_, A, embeddings
