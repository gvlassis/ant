import argparse
import os
import torch
import models.utils_models
import utils
import transformers
import numpy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522)", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=16)
parser.add_argument("--context", type=int, default=1024)

parser.add_argument("--string", help="The string to be visualized", default="Alice is a kind and caring nurse who always puts others first. She works long hours at the hospital, helping patients feel better and supporting their families. Alice has a warm smile that makes people feel safe, and she listens carefully to their worries.")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gpt2")

parser.add_argument("--thresh", help="Attention weights threshold above which a token is considered 'attended'", type=float, default=0.1)
parser.add_argument("--print_tokens", help="Print string tokens for debugging", type=utils.str_to_bool, default=False)
args=parser.parse_args()

attention_path = args.PATH.split(".")[0]+"_attention.dat"

device = "cuda:0"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.scale_type, args.ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, args.context, False, True)
model.load_state_dict(torch.load(args.PATH, weights_only=True))
model = model.to(device)

ids = args.tokenizer.encode(args.string).ids
if args.print_tokens:
    print(f"\x1b[1m{len(ids)} tokens\x1b[0m")
    
    for token in ids:
        print("%16.16s: %6.6s" % (args.tokenizer.id_to_token(token), token))

X = torch.tensor(ids)

model.eval()
with torch.no_grad():
    W = model.W( X.to(device) )

print("\x1b[1m%6.6s %12.12s %6.6s %12.12s %12.12s\x1b[0m" % ("block", "entropy↑", "l2↓", "invsimp↑", "thresh↑"))
entropies = []
l2s = []
invsimps = []
threshs = []
for block in range(model.num_blocks):
    entropy = utils.entropy(W[block])
    l2 = utils.l2(W[block])
    invsimp = utils.invsimp(W[block])
    thresh = utils.thresh(W[block], args.thresh)

    entropies.append(entropy)
    l2s.append(l2)
    invsimps.append(invsimp)
    threshs.append(thresh)
    print("%6.6s %12.12s %6.6s %12.12s %12.12s" % (block, "%.2f" % entropy, "%.2f" % l2, "%.2f" % invsimp, "%.2f" % thresh))
print("━"*55)
print("%6.6s %12.12s %6.6s %12.12s %12.12s" % ("mean", "%.2f" % numpy.mean(entropies), "%.2f" % numpy.mean(l2s), "%.2f" % numpy.mean(invsimps), "%.2f" % numpy.mean(threshs)))

attention_header = models.transformer.get_attention_header(model)
with open(attention_path,"w") as file:
    file.write(f"x y token1 token2 {attention_header}\n")

# How much token1 (x in matrix plot) contributes to the context of token2 (y in matrix plot)
for y, token2 in enumerate(ids):

    for x, token1 in enumerate(ids):
        # rows->y, columns->x
        attention = models.transformer.get_attention(W[..., y, x])
        with open(attention_path,"a") as file:
            file.write("%d %d %s %s %s\n" % (x, y, args.tokenizer.id_to_token(token1), args.tokenizer.id_to_token(token2), attention))
    
    with open(attention_path,"a") as file:
            file.write("\n")

