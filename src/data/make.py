import os
import argparse
import utils_data
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("DIR", help="The directory which will contain train_X.pt, train_Y.pt, val_X.pt, val_Y.pt, test_X.pt, test_Y.pt", type=os.path.abspath)
parser.add_argument("--dataset", choices=utils_data.DATASETS, default="shakespearefirstfolio")
parser.add_argument("--tokenizer_type", choices=utils_data.TOKENIZER_TYPES, help="Tokenizer library to use", default="tokenizers")
parser.add_argument("--tokenizer", help="Name/URL/File of the tokenizer", default="gpt2")
parser.add_argument("--eot_id", help="End-Of-Text token id", type=int, default=50256)
parser.add_argument("--cores", help="CPU cores used by datasets.Dataset.map(num_proc=).", type=int, default=os.cpu_count()//2)
args=parser.parse_args()

os.makedirs(args.DIR, exist_ok=True)

if args.tokenizer_type=="tokenizers":
    import transformers
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(args.tokenizer).backend_tokenizer
elif args.tokenizer_type=="tokenmonster":
    import tokenmonster
    tokenizer = tokenmonster.load_multiprocess_safe(args.tokenizer)

print("💾 Loading splits")
train_dataset, val_dataset, test_dataset = utils_data.get_splits(args.dataset)

print("✏️ Preprocessing")
train_X, train_Y = utils_data.dataset_to_tensors(train_dataset, args.tokenizer_type, tokenizer, args.eot_id, args.cores)
test_X, test_Y = utils_data.dataset_to_tensors(test_dataset, args.tokenizer_type, tokenizer, args.eot_id, args.cores)
val_X, val_Y = utils_data.dataset_to_tensors(val_dataset, args.tokenizer_type, tokenizer, args.eot_id, args.cores)

print("💾 Saving splits")
torch.save(train_X, args.DIR+"/train_X.pt")
if train_Y is not None: torch.save(train_Y, args.DIR+"/train_Y.pt")
torch.save(val_X, args.DIR+"/val_X.pt")
if val_Y is not None: torch.save(val_Y, args.DIR+"/val_Y.pt")
torch.save(test_X, args.DIR+"/test_X.pt")
if test_Y is not None: torch.save(test_Y, args.DIR+"/test_Y.pt")

print("🍾 Done!")
