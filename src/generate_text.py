import argparse
import os
import tokenizers
import torch
import models.transformer
import utils
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)
parser.add_argument("--starting_string", help="The string that the model will continue", default="ΑΝΤΙΓΟΝΗ")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gvlassis/culturay_el_32000")
parser.add_argument("--unk_id", help="Unknown special token id", type=int, default=2)
parser.add_argument("--eot_id", help="End-of-text special token id", type=int, default=6)
parser.add_argument("--context", type=int, default=128)
parser.add_argument("--max_tokens", help="Max number of tokens to be generated (if [EOT] is not generated)", type=int, default=128)
parser.add_argument("--T", help="Softmax temperature", type=int, default=1)
parser.add_argument("--K", help="Top-K sampling", type=int, default=50)
parser.add_argument("--P", help="Top-P sampling", type=int, default=0.95)
args=parser.parse_args()

device="cuda"

print("🧠 Initializing model")
ζ = 24
model = models.transformer.Transformer.from_pretrained(args.PATH).to(device)

utils.generate_text(args.starting_string, args.tokenizer, args.unk_id, args.eot_id, model, args.context, args.max_tokens, args.T, args.K, args.P)
