import torch
from torch import arccos, sin
import matplotlib.pyplot

# Normalizes along last dimension on the hypersphere
# (s1*...*)s-1
def sphere_norm(X):
    return X/torch.linalg.vector_norm(X, ord=2, dim=-1, keepdim=True)

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

# class Block(torch.nn.Module):
#     def __init__(self, heads, d_head, is_causal, exp_factor=4, act=torch.nn.ReLU()):
#         super().__init__()
#
#         self.heads = heads
#         self.d_head = d_head
#         self.d = heads * d_head
#         self.is_causal = is_causal
#         self.exp_factor = exp_factor
#         self.d_hidden = exp_factor*self.d
#         self.act = act
#
#         self.mhsa = MHSA(heads, d_head, is_causal, scale_type)
#         if norm_type=="layer":
#             self.norm1 = torch.nn.LayerNorm(self.d, bias=bias)
#             self.norm2 = torch.nn.LayerNorm(self.d, bias=bias)
#         elif norm_type=="rms":
#             self.norm1 = torch.nn.RMSNorm(self.d, elementwise_affine=False)
#             self.norm2 = torch.nn.RMSNorm(self.d, elementwise_affine=False)
#         self.mlp = mlp.MLP2L(self.d, self.d_hidden, self.d, bias, act=act, dropout=dropout, l1_type=l1_type)
#
#     def forward(self, X):
#         Y = self.mhsa(self.norm1(X))
#         Y_ = torch.nn.functional.dropout(Y, p=self.dropout, training=self.training)
#         Y__ = X + Y_
#
#         Z = self.mlp(self.norm2(Y__))
#         Z_ = torch.nn.functional.dropout(Z, p=self.dropout, training=self.training)
#         Z__ = Y__ + Z_
#
#         return Z__
#
# class EncBlock(Block):
#     def __init__(self, heads, d_head, exp_factor=4, dropout=0, act=torch.nn.ReLU()):
#         super().__init__(heads, d_head, False, exp_factor, act)
#
# class DecBlock(Block):
#     def __init__(self, heads, d_head, exp_factor=4, act=torch.nn.ReLU()):
#         super().__init__(heads, d_head, True, exp_factor, act)
#
# class nGPT(torch.nn.Module):
#     def __init__(self, vocab_size=50257, num_blocks=6, heads=8, d_head=4, exp_factor=4, max_context=128, all_pos=False, act=torch.nn.ReLU()):
#         super().__init__()
#
#         self.vocab_size = vocab_size
#         self.num_blocks = num_blocks
#         self.heads = heads
#         self.d_head = d_head
#         self.d = heads * d_head
#         self.exp_factor = exp_factor
#         self.dropout = dropout
#         self.max_context = max_context
#         self.all_pos = all_pos
#         self.act = act
#
#         self.Ein = torch.nn.Embedding(vocab_size, self.d)
#
#         self.blocks = torch.nn.Sequential(*[DecBlock(heads, d_head, exp_factor, act) for _ in range(num_blocks)])
#
#         self.Eout = torch.nn.Linear(self.d, vocab_size, bias=False)
#
#         self.sz = torch.nn.Parameter()
#
#     # (batches*)context
#     def forward(self, ids):
#         context = ids.shape[-1]
#
#         # (batches*)context*d
#         H = 
#
#         # (batches*)context*vocab_size
#         Z = self.Eout(H)
#
#         Z = self.sz * Z
#
#         return Z
