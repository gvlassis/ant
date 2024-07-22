import torch
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

    # (batches*)seq_len*d
    def forward(self, X):
        # (batches*)seq_len*(3d)
        QKV = self.QKV(X)

        # (batches*)seq_len*3*heads*d_head
        QKV = torch.unflatten(QKV, dim=-1, sizes=(3,self.heads,self.d_head))
        # (batches*)3*heads*seq_len*d_head
        QKV = torch.movedim(QKV, source=-4, destination=-2)
        # (batches*)heads*seq_len*d_head
        Q, K, V = torch.unbind(QKV, dim=-4)

        # (batches*)heads*seq_len*d_head
        Y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=self.is_causal, scale=self.scale)
        # (batches*)seq_len*heads*d_head
        Y = torch.movedim(Y, source=-3, destination=-2)
        # (batches*)seq_len*d
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

def sin_pos_enc(seq_len, d, device=None):
    sin_pos_enc = torch.empty((seq_len, d), device=device)

    ts = torch.arange(1.0, seq_len+1, dtype=torch.float32, device=device)

    ks = torch.arange(1.0, int(d/2)+1, device=device)
    ωs = 1/10000**(2*ks/d)

    # seq_len*(d/2)
    tsxωs = ts[:,None] @ ωs[None,:]

    sin_pos_enc[:,0::2] = torch.sin(tsxωs)
    sin_pos_enc[:,1::2] = torch.cos(tsxωs)

    return sin_pos_enc

# model d heads num_blocks
# xt 256 4 2
# t 384 6 4
# Vaswani/xs 512 8 6
# GPT2-S/s 768 12 12
# GPT2-M/m 1024 16 24
# GPT2-L/l 1280 20 36
# GPT2-XL/xl 1600 25 48
class Transformer(torch.nn.Module):
    def __init__(self, vocab_size=50257, num_blocks=6, d=32, heads=8, scale=None, exp_factor=4, dropout=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.d = d
        self.heads = heads
        self.exp_factor = exp_factor
        self.scale = scale
        self.dropout = dropout

        self.embeddings = torch.nn.Embedding(vocab_size, d)

        self.blocks = torch.nn.Sequential(*[TransformerDecBlock(d, heads, scale, exp_factor, dropout) for _ in range(num_blocks)])

        self.norm = torch.nn.LayerNorm(d)

        self.linear = torch.nn.Linear(d, vocab_size)

    # (batches*)seq_len
    def forward(self, ids):
        seq_len = ids.shape[-1]

        # (batches*)seq_len*d
        embeddings = self.embeddings(ids)
        pos_enc = sin_pos_enc(seq_len, self.d, ids.device)

        X = torch.nn.functional.dropout(embeddings+pos_enc, p=self.dropout, training=self.training)

        Y = self.blocks(X)

        Y_ = self.norm(Y)

        # (batches*)seq_len*vocab_size
        Z = self.linear(Y_)

        return Z

    def speak(self, starting_string, tokenizer, eot_id, context=1024):
        string = starting_string
        print(starting_string, end="")

        ids = tokenizer.encode(starting_string).ids

        while True:
            X = torch.tensor(ids[-context:])

            self.eval()
            with torch.no_grad():
                Y = self( X.to(device=next(self.parameters()).device) )

            probs = torch.nn.functional.softmax(Y[-1], dim=0)

            new_id = torch.multinomial(probs, num_samples=1).item()
            ids = ids + [new_id]

            if new_id == eot_id:
                break
            else:
                new_substring = tokenizer.decode([new_id])
                string = string + new_substring
                print(new_substring, end="")

        return string
