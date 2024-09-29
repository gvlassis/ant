import argparse
import os
import tokenizers
import torch
import models.utils_models
import utils
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--starting_string", help="The string that the model will continue", default="On Christmas Day, the gifts were brought by Santa")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default="gpt2")
parser.add_argument("--unk_id", help="Unknown special token id", type=int, default=50257)
parser.add_argument("--eot_id", help="End-of-text special token id", type=int, default=50256)
parser.add_argument("--max_tokens", help="Max number of tokens to be generated (if [EOT] is not generated)", type=int, default=128)
parser.add_argument("--T", help="Softmax temperature", type=int, default=1)
parser.add_argument("--K", help="Top-K sampling", type=int, default=50)
parser.add_argument("--P", help="Top-P sampling", type=int, default=0.95)
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=12)

parser.add_argument("--context", type=int, default=1024)
args=parser.parse_args()

model_device = "cuda:0"
dataset_device = "cpu"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, ζ=args.ζ, max_context=args.context)
model.load_state_dict(torch.load(args.PATH))
model = model.to(model_device)

utils.generate_text(args.starting_string, args.tokenizer, args.unk_id, args.eot_id, model, args.context, args.max_tokens, args.T, args.K, args.P)
