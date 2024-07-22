import torch
from . import transformer

class ViT(torch.nn.Module):
    def __init__(self, res=32, patch_size=4, num_blocks=6, d=32, heads=8, scale=None, exp_factor=1, dropout=0.1, classes=10):
        super().__init__()
        
        self.res = res
        self.patch_size = patch_size
        patches = (res//patch_size)*(res//patch_size)
        self.seq_len = patches+1
        self.num_blocks = num_blocks
        self.d = d
        self.heads = heads
        self.scale = scale
        self.exp_factor = exp_factor
        self.dropout = dropout

        # Linear projection
        self.emb = torch.nn.Conv2d(in_channels=3, out_channels=d, kernel_size=patch_size, stride=patch_size, bias=False)

        self.class_emb = torch.nn.Parameter(torch.randn(d))

        self.blocks = torch.nn.Sequential(*[transformer.TransformerEncBlock(d, heads, scale, exp_factor, dropout) for _ in range(num_blocks)])

        self.norm = torch.nn.LayerNorm(d)

        self.linear = torch.nn.Linear(d, classes)

    # (batches*)3*32*32
    def forward(self, x):

        # (batches*)d*(32/patch_size)*(32/patch_size)
        patch_emb = self.emb(x)

        # (batches*)d*patches
        patch_emb = patch_emb.flatten(start_dim=-2, end_dim=-1)

        # (batches*)patches*d
        patch_emb = patch_emb.swapdims(-1,-2)

        # (batches*)patches*d
        class_emb = self.class_emb.expand_as(patch_emb)
        # (batches*)1*d
        # 0:1 instead of 0 in the slicing maintains the dimensions (otherwise, it would drop the middle dimension)
        class_emb = class_emb[...,0:1,:]

        # (batches*)seq_len*d
        emb = torch.cat((class_emb, patch_emb), dim=-2)
        pos_emb = transformer.sin_pos_enc(self.seq_len, self.d, x.device)

        X = torch.nn.functional.dropout(emb+pos_emb, p=self.dropout, training=self.training)

        Y = self.blocks(X)

        # (batches*)d
        y0 = Y[...,0,:]

        y0_ = self.norm(y0)

        # (batches*)d
        z = self.linear(y0_)

        return z
