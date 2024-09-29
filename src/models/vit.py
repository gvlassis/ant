import torch
from . import transformer

class ViT(torch.nn.Module):
    def __init__(self, channels=3, max_res=32, patch_size=4, num_blocks=6, heads=8, d_head=4, scale_type="1/sqrt(d)", exp_factor=1, dropout=0.1, pos_type="sin", all_pos=False, norm_type="layer", bias=True, act=torch.nn.ReLU(), l1_type="linear", classes=10):
        super().__init__()

        self.channels = channels
        self.max_res = max_res
        self.patch_size = patch_size
        self.max_patches = (max_res//patch_size)*(max_res//patch_size)
        self.max_context = self.max_patches+1
        self.num_blocks = num_blocks
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale_type = scale_type
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.pos_type = pos_type
        self.all_pos = all_pos
        self.norm_type = norm_type
        self.bias = bias
        self.act = act
        self.l1_type = l1_type
        self.classes = classes

        # Linear projection
        self.emb = torch.nn.Conv2d(in_channels=channels, out_channels=self.d, kernel_size=patch_size, stride=patch_size, bias=False)

        self.cls = torch.nn.Parameter(torch.randn(self.d))

        pos = transformer.get_pos(pos_type, self.max_context, self.d)
        if pos_type in ("sin", "rot"):
            self.register_buffer("pos", pos)
        elif pos_type == "learned":
            self.pos = torch.nn.Parameter(pos)
        
        self.blocks = torch.nn.Sequential(*[transformer.TransformerEncBlock(heads, d_head, scale_type, exp_factor, dropout, norm_type, bias, act, l1_type) for _ in range(num_blocks)])

        if norm_type=="layer":
            self.norm = torch.nn.LayerNorm(self.d, bias=bias)
        elif norm_type=="rms":
            self.norm = torch.nn.RMSNorm(self.d, elementwise_affine=False)

        self.linear = torch.nn.Linear(self.d, classes, bias=False)

    # (batches*)channels*res*res
    def forward(self, x):
        res = x.shape[-1]
        patches = (res//self.patch_size)*(res//self.patch_size)
        context = patches+1

        # (batches*)d*(res/patch_size)*(res/patch_size)
        y = self.emb(x)

        # (batches*)d*patches
        X = y.flatten(start_dim=-2, end_dim=-1)

        # (batches*)patches*d
        X = X.swapdims(-1,-2)

        # (batches*)patches*d
        cls = self.cls.expand_as(X)
        # (batches*)1*d
        # 0:1 instead of 0 in the slicing maintains the dimensions (otherwise, it would drop the middle dimension)
        cls = cls[...,0:1,:]

        # (batches*)context*d
        X = torch.cat((cls, X), dim=-2)
        
        X = transformer.apply_pos(self.pos_type, X, self.pos[...,:context,:])
        X_ = torch.nn.functional.dropout(X, p=self.dropout, training=self.training)

        Y = X_
        for block in self.blocks:
            Y_ = block(Y)
            Y = transformer.apply_pos(self.pos_type, Y_, self.pos[...,:context,:]) if self.all_pos else Y_

        # (batches*)d
        y0_ = Y_[...,0,:]

        y0__ = self.norm(y0_)

        # (batches*)classes
        z = self.linear(y0__)

        return z
