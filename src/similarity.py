import argparse
import os
import torch
import models.utils_models
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522)", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=16)
parser.add_argument("--context", type=int, default=1024)

parser.add_argument("--string", help="The string to be visualized", default="Bob is a pilot and Alice is a nurse. He works in a plane and she works in a hospital.")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gpt2")

parser.add_argument("--blocks_interval", help="Every how many transformer blocks to check", type=int, default=1)
args=parser.parse_args()

similarity_path = args.PATH.split(".")[0]+"_similarity.dat"

device = "cuda:0"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.scale_type, args.ζ, 0.02, 0.5, 0.5, 0.001, 0.001, 0.001, "adam", 0, False, (0.9, 0.95), 0, args.context, False, True)
model.load_state_dict(torch.load(args.PATH, weights_only=True))
model = model.to(device)

ids = args.tokenizer.encode(args.string).ids
print(f"\x1b[1m{len(ids)} tokens\x1b[0m")

X = torch.tensor(ids)

model.eval()
with torch.no_grad():
    embeddings = model.get_embeddings( X.to(device) )

similarity_header = models.transformer.get_similarity_header(model, args.blocks_interval)
with open(similarity_path,"w") as file:
    file.write(f"x y token1 token2 {similarity_header}\n")

# How similar token1 (x in matrix plot) is to token2 (y in matrix plot)
for y, token2 in enumerate(ids):
    print("%16.16s: %6.6s" % (args.tokenizer.id_to_token(token2), token2))
    
    for x, token1 in enumerate(ids):
        similarity = models.transformer.get_similarity(embeddings[:, x, :], embeddings[:, y, :], args.blocks_interval)
        with open(similarity_path,"a") as file:
            file.write("%d %d %s %s %s\n" % (x, y, args.tokenizer.id_to_token(token1), args.tokenizer.id_to_token(token2), similarity))

    with open(similarity_path,"a") as file:
            file.write("\n")

