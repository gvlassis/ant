import argparse
import os
import tokenizers
import torch
import models.transformer
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)
parser.add_argument("--starting_string", help="The string that the model will continue", default="ΑΝΤΙΓΟΝΗ")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=tokenizers.Tokenizer.from_pretrained ,default="gvlassis/culturay_el_32000")
parser.add_argument("--unk_id", help="Unknown special token id", type=int, default=2)
parser.add_argument("--eot_id", help="End-of-text special token id", type=int, default=6)
parser.add_argument("--context", type=int, default=128)
parser.add_argument("--max_tokens", help="Max number of tokens to be generated (if [EOT] is not generated)", type=int, default=128)
parser.add_argument("--T", help="Softmax temperature", type=int, default=1)
parser.add_argument("--K", help="Top-K sampling", type=int, default=50)
parser.add_argument("--P", help="Top-P sampling", type=int, default=0.95)
args=parser.parse_args()

device="cpu"

print("🧠 Initializing model")
ζ = 8
model = models.transformer.Transformer(vocab_size=args.tokenizer.get_vocab_size(), d=32*ζ, scale=1/(32*ζ)).to(device)
model.load_state_dict(torch.load(args.PATH, map_location=device))

utils.generate_text(args.starting_string, args.tokenizer, args.unk_id, args.eot_id, model, args.context, args.max_tokens, args.T, args.K, args.P)
