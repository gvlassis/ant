import torch
from . import mlp
from math import sqrt
from .quantization import QuantizedLinear
from .mlp import get_quantizers

SCALE_TYPES = ["1/sqrt(d)", "1/d"]
POS_TYPES = ["learned", "sinusoidal", "rope", "alibi"]
BACKENDS = ["pytorch", "flash", "flex", "cudnn"]
NORM_TYPES = ["layer", "rms_learned", "rms_const", "sphere"]

def get_causal(context):
    causal = torch.full((context,context), True)

    causal = causal.tril()

    return causal

def get_sinusoidal(context, d, base=1024):
    # [pos=0, pos=1, ...]
    poss = torch.arange(0., context)
    # [i=0, i=1, ...]
    js = torch.arange(0., d//2)
    # [ω0, ω1, ...]
    ωs = 1/base**(2*js/d)
    
    # [pos=0*ω0, pos=0*ω1, ...]
    # [pos=1*ω0, pos=1*ω1, ...]
    φs = poss[...,None] @ ωs[None,...]
    
    # context*d
    sinusoidal = torch.empty((context, d))
    sinusoidal[:,0::2] = torch.sin(φs)
    sinusoidal[:,1::2] = torch.cos(φs)

    return sinusoidal

def get_rope(context, d, base=1024):
    # [m=0, m=1, ...]
    ms = torch.arange(0., context)
    # [i=0, i=1, ...]
    js = torch.arange(0., d//2)
    # [θ0, θ1, ...]
    θs = 1/base**(2*js/d)
    
    # [m=0*θ0, m=0*θ1, ...]
    # [m=1*θ0, m=1*θ1, ...]
    φs = ms[...,None] @ θs[None,...]
    
    # context*d/2
    cos = torch.cos(φs)
    sin = torch.sin(φs)
    # context*d
    cos = cos.repeat_interleave(repeats=2, dim=1)
    sin = sin.repeat_interleave(repeats=2, dim=1)
    
    # 2*context*d
    rope = torch.stack((cos,sin))

    return rope

# (batches*)context*d
def apply_rope(X, rope):
    X_ = torch.empty_like(X)
    X_[...,0::2] = -X[...,1::2]
    X_[...,1::2] = X[...,0::2]

    # context*d
    cos = rope[0]
    sin = rope[1]

    Y = X*cos + X_*sin

    return Y

def get_m(heads, base=2, exp=8):
    m = base**( (-exp/heads)*torch.arange(1,heads+1) )

    return m

def get_alibi(heads, context):
    # 1*context*1
    i = torch.arange(0, context)[None,:,None]
    # 1*1*context
    j = i.mT
    # heads*1*1
    m = get_m(heads)[:,None,None]

    alibi = -torch.abs(i - j)*m

    return alibi

def get_swa(context, window):
    # context*1
    i = torch.arange(0, context).unsqueeze(-1)
    # 1*context
    j = i.T

    swa = torch.abs(i - j) <= window

    return swa  

# (batches*)heads/groups*context*d_head
def sdpa_pytorch(Q, K, V, causal=None, alibi=None, swa=None, scale=None, return_A=False):
    if scale is None:
        d_head = Q.shape[-1]
        scale = 1/sqrt(d_head)
    
    # (batches*)heads*context*d_head
    heads = Q.shape[-3]
    groups = K.shape[-3]
    ratio = heads//groups
    # PyTorch only broadcasts when the operation is not defined otherwise. MM does not involve the batch dimensions, and hence PyTorch does not broadcast them.
    K = K.repeat_interleave(repeats=ratio, dim=-3)
    V = V.repeat_interleave(repeats=ratio, dim=-3)

    # (batches*)heads*context*context
    A__ = Q @ K.mT
    
    # batches*heads*context*context
    A_ = scale*A__
    # (batches*)heads*context*context
    A_ = A_.reshape(A__.shape)

    if alibi is not None:
        A_ = A_ + alibi
    if causal is not None:
        A_.masked_fill_(~causal, -float("inf"))
    if swa is not None:
        A_.masked_fill_(~swa, -float("inf"))

    A = torch.softmax(A_, dim=-1)

    # (batches*)heads*context*d_head
    Y = A @ V
    
    if not return_A:
        return Y
    else:
        return Y, A__, A_, A

# (batches*)heads/groups*context*d_head
def sdpa_flash(Q, K, V, causal=False, alibi=None, swa=None, scale=None):
    import flash_attn
    
    # FlashAttention2 only supports float scale
    if isinstance(scale, torch.Tensor):
        Q_shape = Q.shape
        # batches*heads*context*d_head
        Q = scale*Q
        # (batches*)heads*context*d_head
        Q = Q.reshape(Q_shape)

        scale = 1
    
    # FlashAttention2 only supports BF16 and FP16
    if Q.dtype in [torch.bfloat16, torch.float16]:
        dtype = Q.dtype
    else: 
        dtype = torch.bfloat16

    heads = Q.shape[-3]
    groups = K.shape[-3]
    context = Q.shape[-2]
    d_head = Q.shape[-1]

    # CAUTION: FlashAttention2 expects batches*context*heads/groups*d_head
    Q = Q.movedim(-3,-2).reshape(-1,context,heads,d_head)
    K = K.movedim(-3,-2).reshape(-1,context,groups,d_head)
    V = V.movedim(-3,-2).reshape(-1,context,groups,d_head)
    
    if swa is None:
        swa = (-1,-1)
    
    Y = flash_attn.flash_attn_func(Q.to(dtype), K.to(dtype), V.to(dtype), causal=causal, alibi_slopes=alibi,  window_size=swa, softmax_scale=scale)
    Y = Y.to(Q.dtype)
    
    # Restore the shape to: (batches*)heads*context*d_head
    Y = Y.movedim(-3,-2).squeeze(0)

    return Y

# (batches*)heads/groups*context*d_head
def sdpa_flex():
    return None

# (batches*)heads/groups*context*d_head
def sdpa_cudnn():
    return None

def sdpa_wrapper(Q, K, V, causal=None, alibi=None, swa=None, scale=None, return_A=False, backend="flash"):
    if backend=="pytorch":
        return sdpa_pytorch(Q, K, V, causal, alibi, swa, scale, return_A)
    elif backend=="flash":
        return sdpa_flash(Q, K, V, causal, alibi, swa, scale)
    elif backend=="flex":
        return sdpa_flex()
    elif backend=="cudnn":
        return sdpa_cudnn()

def test_sdpa():
    batches = 32
    heads = 12
    context = 1024
    d_head = 64
    window = 256
    groups = 4
    dtype = torch.bfloat16
    
    print("\x1b[1mbfloat16\x1b[0m",end="")
    Q = torch.rand((batches, heads, context, d_head)).to("cuda:0", dtype)
    K = torch.rand((batches, heads, context, d_head)).to("cuda:0", dtype)
    V = torch.rand((batches, heads, context, d_head)).to("cuda:0", dtype)
    pytorch = sdpa_wrapper(Q, K, V, backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

    print("\x1b[1mcausal\x1b[0m",end="")
    pytorch = sdpa_wrapper(Q, K, V, causal=get_causal(context).to("cuda:0"), backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, causal=True, backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

    print("\x1b[1malibi\x1b[0m",end="")
    pytorch = sdpa_wrapper(Q, K, V, alibi=get_alibi(heads,context).to("cuda:0",dtype), backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, alibi=get_m(heads).to("cuda:0"), backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

    print("\x1b[1mswa\x1b[0m",end="")
    pytorch = sdpa_wrapper(Q, K, V, swa=get_swa(context,window).to("cuda:0"), backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, swa=(window,window), backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")
    
    print("\x1b[1mcausal+alibi\x1b[0m",end="")
    pytorch = sdpa_wrapper(Q, K, V, causal=get_causal(context).to("cuda:0"), alibi=get_alibi(heads,context).to("cuda:0",dtype), backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, causal=True, alibi=get_m(heads).to("cuda:0"), backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

    print("\x1b[1mcausal+swa\x1b[0m",end="")
    pytorch = sdpa_wrapper(Q, K, V, causal=get_causal(context).to("cuda:0"), swa=get_swa(context,window).to("cuda:0"), backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, causal=True, swa=(window,window), backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

    print("\x1b[1mGQA\x1b[0m",end="")
    Q = torch.rand((batches, heads, context, d_head)).to("cuda:0", dtype)
    K = torch.rand((batches, groups, context, d_head)).to("cuda:0", dtype)
    V = torch.rand((batches, groups, context, d_head)).to("cuda:0", dtype)
    pytorch = sdpa_wrapper(Q, K, V, backend="pytorch")
    flash = sdpa_wrapper(Q, K, V, backend="flash")
    torch.testing.assert_close(flash, pytorch, check_dtype=False)
    print("\x1b[32m ✔\x1b[0m")

class MHSA(torch.nn.Module):
    def __init__(self, heads, d_head, scale_type="1/sqrt(d)", ratio=1, qk_norm=True, quantization_bits=16):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale_type = scale_type
        self.ratio = ratio
        self.groups = heads//ratio
        self.d_KV = self.groups * d_head
        self.qk_norm = qk_norm
        if qk_norm:
            # (batches*)heads*context*d_head
            scale = torch.full((1,heads,1,1), sqrt(d_head))
            self.scale = torch.nn.Parameter(scale)
        else:
            if scale_type=="1/sqrt(d)":
                self.scale = 1/sqrt(d_head)
            elif scale_type=="1/d":
                self.scale = 1/d_head
        
        # Get quantizers
        weight_quantizer, activation_quantizer = get_quantizers(quantization_bits)
        
        # Packing QKV gives negligible speed gains, while not allowing GQA, hurting code clarity and having side effects with μP
        self.lq = QuantizedLinear(self.d, self.d, bias=False, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
        self.lk = QuantizedLinear(self.d, self.d_KV, bias=False, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
        self.lv = QuantizedLinear(self.d, self.d_KV, bias=False, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
        
        self.lo = QuantizedLinear(self.d, self.d, bias=False, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)

    # (batches*)context*d
    def forward(self, X, causal=None, rope=None, alibi=None, swa=None, return_A=False, backend="flash"):
        # (batches*)context*d
        Q = self.lq(X)
        # (batches*)context*d_KV
        K = self.lk(X)
        V = self.lv(X)

        # (batches*)context*heads*d_head
        Q = Q.unflatten(dim=-1, sizes=(self.heads, self.d_head))
        # (batches*)context*groups*d_head
        K = K.unflatten(dim=-1, sizes=(self.groups, self.d_head))
        V = V.unflatten(dim=-1, sizes=(self.groups, self.d_head))

        # (batches*)heads*context*d_head
        Q = Q.movedim(-3,-2)
        # (batches*)groups*context*d_head
        K = K.movedim(-3,-2)
        V = V.movedim(-3,-2)
        
        if rope is not None:
            Q = apply_rope(Q,rope)
            K = apply_rope(K,rope)
        
        # After RoPE
        if self.qk_norm:
            Q = mlp.sphere_norm(Q)
            K = mlp.sphere_norm(K)

        # (batches*)heads*context*d_head
        if not return_A:
            Y = sdpa_wrapper(Q, K, V, causal, alibi, swa, self.scale, return_A, backend)
        else:
            Y, A__, A_, A = sdpa_wrapper(Q, K, V, causal, alibi, swa, self.scale, return_A, backend)
        # (batches*)context*heads*d_head
        Y = Y.movedim(-3,-2)
        # (batches*)context*d
        Y = Y.flatten(-2,-1)

        Y = self.lo(Y)
        
        if not return_A:
            return Y
        else:
            return Y, A__, A_, A

# Pre-Norm
class Block(torch.nn.Module):
    def __init__(self, heads, d_head, scale_type="1/sqrt(d)", ratio=1, exp_factor=3, dropout=0, norm_type="rms_learned", bias=False, act=mlp.ReLU2(), l1_type="linear", pre_att_norm=False, qk_norm=True, out_att_norm=True, pre_mlp_norm=False, act_norm=False, out_mlp_norm=True, quantization_bits=16):
        super().__init__()

        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale_type = scale_type
        self.ratio = ratio
        self.groups = heads//ratio
        self.exp_factor = exp_factor
        self.d_hidden = exp_factor*self.d
        self.dropout = dropout
        self.norm_type = norm_type
        self.bias = bias
        self.act = act
        self.l1_type = l1_type
        
        self.mhsa = MHSA(heads, d_head, scale_type, ratio, qk_norm, quantization_bits)
        self.pre_att_norm = mlp.get_norm(pre_att_norm, norm_type, self.d, bias)
        self.out_att_norm = mlp.get_norm(out_att_norm, norm_type, self.d, bias)

        self.mlp = mlp.MLP2L(self.d, self.d_hidden, self.d, bias, act, dropout, l1_type, norm_type, act_norm, quantization_bits)
        self.pre_mlp_norm = mlp.get_norm(pre_mlp_norm, norm_type, self.d, bias)
        self.out_mlp_norm = mlp.get_norm(out_mlp_norm, norm_type, self.d, bias)
        
    def forward(self, X, causal=None, rope=None, alibi=None, swa=None, return_A=False, backend="flash"):
        mhsa = self.mhsa(self.pre_att_norm(X) if self.pre_att_norm else X, causal, rope, alibi, swa, return_A, backend)
        if not return_A:
            Y = mhsa
        else:
            Y, A__, A_, A = mhsa

        if self.out_att_norm: Y = self.out_att_norm(Y)

        Y_ = torch.nn.functional.dropout(Y, p=self.dropout, training=self.training)
        Y__ = X + Y_
        
        Z = self.mlp(self.pre_mlp_norm(Y__) if self.pre_mlp_norm else Y__)

        if self.out_mlp_norm: Z = self.out_mlp_norm(Z)

        Z_ = torch.nn.functional.dropout(Z, p=self.dropout, training=self.training)
        Z__ = Y__ + Z_

        if not return_A:
            return Z__
        else:
            return Z__, A__, A_, A

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size=50304, num_blocks=12, heads=12, d_head=64, scale_type="1/sqrt(d)", ratio=1, is_causal=True, window=None, backend="flash", exp_factor=4, dropout=0, pos_type="rope", max_context=128, norm_type="rms_learned", bias=False, act=mlp.ReLU2(), l1_type="linear", std=0.02, test=False, weight_tying=True, emb_norm=False, pre_att_norm=False, qk_norm=True, out_att_norm=True, pre_mlp_norm=False, act_norm=False, out_mlp_norm=True, out_norm=True, fix_norm=False, quantization_bits=16):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.heads = heads
        self.d_head = d_head
        self.d = heads * d_head
        self.scale_type = scale_type
        self.ratio = ratio
        self.groups = heads//ratio
        self.is_causal = is_causal
        self.window = window
        self.backend = backend
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.pos_type = pos_type
        self.max_context = max_context
        self.norm_type = norm_type
        self.bias = bias
        self.act = act
        self.l1_type = l1_type
        self.weight_tying = weight_tying
        self.fix_norm = fix_norm

        self.emb = torch.nn.Embedding(vocab_size, self.d)

        self.emb_norm = mlp.get_norm(emb_norm, norm_type, self.d, bias)

        if pos_type == "learned":
            pos = torch.rand((max_context, self.d))
            self.pos = torch.nn.Parameter(pos)
        
        self.blocks = torch.nn.Sequential(*[Block(heads, d_head, scale_type, ratio, exp_factor, dropout, norm_type, bias, act, l1_type, pre_att_norm, qk_norm, out_att_norm, pre_mlp_norm, act_norm, out_mlp_norm, quantization_bits) for _ in range(num_blocks)])
        
        self.out_norm = mlp.get_norm(out_norm, norm_type, self.d, bias)
        
        # Get quantizers for output linear layer
        weight_quantizer, activation_quantizer = get_quantizers(quantization_bits)
        self.linear = QuantizedLinear(self.d, vocab_size, bias=False, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)

        if weight_tying: self.emb.weight = self.linear.weight
        
        self.init(std, test)

    def init(self, std=0.02, test=False):
        if test: print("\x1b[1m%36.36s %8.8s %8.8s %8.8s\x1b[0m" % ("parameter_name", "suffix", "mean", "std"))
        for parameter_name, parameter in self.named_parameters():
            parent_name, _, suffix = parameter_name.rpartition(".")
            parent = self.get_submodule(parent_name)

            if isinstance(parent, (torch.nn.Linear, torch.nn.Embedding, QuantizedLinear)) and suffix=="weight":
                torch.nn.init.normal_(parameter, 0, std)
            elif isinstance(parent, (torch.nn.Linear, torch.nn.LayerNorm, QuantizedLinear)) and suffix=="bias":
                torch.nn.init.zeros_(parameter)
            elif isinstance(parent, (torch.nn.LayerNorm, torch.nn.RMSNorm)) and suffix=="weight":
                torch.nn.init.ones_(parameter)
            else:
                # pos
                if parameter.ndim == 2:
                    torch.nn.init.zeros_(parameter)
                # scale
                elif parameter.ndim == 4:
                    torch.nn.init.constant_(parameter, sqrt(self.d_head))
            
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
        X = self.emb(ids)

        if return_emb:
            # (batches*)(num_blocks+1)*context*d
            embeddings = torch.empty(*ids.shape[:-1], self.num_blocks+1, context, self.d)
            embeddings[...,0,:,:] = X
        
        # Recompute in every batch in case context changes
        if self.is_causal:
            if self.backend=="pytorch":
                causal = get_causal(context).to(ids.device)
            elif self.backend=="flash":
                causal = True
            elif self.backend=="flex":
                causal = causal_mod
            elif self.backend=="cudnn":
                # right_bound
                causal = 0
        else: causal = None

        if self.pos_type == "sinusoidal":
            pos = get_sinusoidal(context, self.d).to(ids.device)
            X = X + pos
            
        if self.pos_type == "learned":
            X = X + self.pos[:context,:]

        if self.pos_type == "rope":
            rope = get_rope(context, self.d_head).to(ids.device)
        else: rope = None

        if self.pos_type == "alibi":
            if self.backend=="pytorch":
                alibi = get_alibi(self.heads, context).to(ids.device)
            elif self.backend=="flash":
                alibi = get_m(self.heads).to(ids.device)
            elif self.backend=="flex":
                alibi = alibi_mod
            elif self.backend=="cudnn":
                alibi = True
        else: alibi = None

        if self.window is not None:
            if self.backend=="pytorch":
                swa = get_swa(context, self.window).to(ids.device)
            elif self.backend=="flash":
                swa = (self.window, self.window)
            elif self.backend=="flex":
                swa = swa_mod
            elif self.backend=="cudnn":
                # left_bound
                swa = self.window
        else: swa = None
        
        # After positional encoding
        if self.emb_norm: X = self.emb_norm(X)

        X_ = torch.nn.functional.dropout(X, p=self.dropout, training=self.training)

        Y = X_
        for i, block in enumerate(self.blocks):
            if not return_A:
                Y = block(Y, causal, rope, alibi, swa, return_A, self.backend)
            else:
                Y, A__[...,i,:,:,:], A_[...,i,:,:,:], A[...,i,:,:,:] = block(Y, causal, rope, alibi, swa, return_A, self.backend)

            if return_emb:
                embeddings[...,i+1,:,:] = Y
        
        if self.out_norm: Y = self.out_norm(Y)

        # (batches*)context*vocab_size
        if self.fix_norm:
            Z = torch.nn.functional.linear(Y, mlp.sphere_norm(self.linear.weight))
        else:
            Z = self.linear(Y)
        
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

def get_attention_header(transformer):
    attention_header = ""
    
    for block in range(transformer.num_blocks):
        for head in range(transformer.heads):
            attention_header += f"block{block}.head{head} "

    # Remove last space
    attention_header = attention_header[:-1]

    return attention_header

def get_attention(W):
    attention = ""
    
    for block in range(W.shape[0]):
        for head in range(W.shape[1]):
            # rows->y, columns->x
            attention +=  "%.2f " % W[block, head]

    # Remove last space
    attention = attention[:-1]

    return attention

def get_similarity_header(transformer):
    similarity_header = "embedding "
    
    for block in range(transformer.num_blocks):
        similarity_header += f"block{block} "

    # Remove last space
    similarity_header = similarity_header[:-1]

    return similarity_header

def get_similarity(embeddings_x, embeddings_y):
    similarity = ""

    for block in range(embeddings_x.shape[0]):
        similarity +=  "%.2f " % torch.nn.functional.cosine_similarity(embeddings_x[block,:], embeddings_y[block,:], dim=0)

    # Remove last space
    similarity = similarity[:-1]

    return similarity

def get_clustering_header(transformer):
    clustering_header = "embedding.random.x embedding.random.y "\
                        "embedding.pca.x embedding.pca.y "\
                        "embedding.mds.x embedding.mds.y "\
                        "embedding.tsne.x embedding.tsne.y "\
                        "embedding.umap.x embedding.umap.y "

    for block in range(transformer.num_blocks):
        clustering_header += f"block{block}.random.x block{block}.random.y "\
                             f"block{block}.pca.x block{block}.pca.y "\
                             f"block{block}.mds.x block{block}.mds.y "\
                             f"block{block}.tsne.x block{block}.tsne.y "\
                             f"block{block}.umap.x block{block}.umap.y "

    # Remove last space
    clustering_header = clustering_header[:-1]

    return clustering_header

def get_clustering(random, pca, mds, tsne, umap):
    clustering = ""

    for block in range(random.shape[0]):
        clustering += "%f %f %f %f %f %f %f %f %f %f " % (random[block,0], random[block,1], pca[block,0], pca[block,1], mds[block,0], mds[block,1], tsne[block,0], tsne[block,1], umap[block,0], umap[block,1])

    # Remove last space
    clustering = clustering[:-1]

    return clustering
