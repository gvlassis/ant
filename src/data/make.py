import os
import argparse
import transformers
import utils_data
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("DIR", help="The directory which will contain train_X.pt, train_Y.pt, val_X.pt, val_Y.pt, test_X.pt, test_Y.pt", type=os.path.abspath)
parser.add_argument("--dataset", choices=utils_data.DATASETS, default="ancient_greek_theatre")
parser.add_argument("--tokenizer", help="Hugging Face repository of the tokenizer to be used", type=lambda x: None if x is None else transformers.PreTrainedTokenizerFast.from_pretrained(x).backend_tokenizer, default=None)
parser.add_argument("--eot_id", type=int, default=6)
args=parser.parse_args()

os.makedirs(args.DIR, exist_ok=True)

print("💾 Loading splits")
train_dataset, val_dataset, test_dataset = utils_data.get_splits(args.dataset)

print("✏️ Preprocessing")
train_X, train_Y = utils_data.dataset_to_tensors(train_dataset, args.tokenizer, args.eot_id)
test_X, test_Y = utils_data.dataset_to_tensors(test_dataset, args.tokenizer, args.eot_id)
val_X, val_Y = utils_data.dataset_to_tensors(val_dataset, args.tokenizer, args.eot_id)

print("💾 Saving splits")
torch.save(train_X, args.DIR+"/train_X.pt")
if train_Y is not None: torch.save(train_Y, args.DIR+"/train_Y.pt")
torch.save(val_X, args.DIR+"/val_X.pt")
if val_Y is not None: torch.save(val_Y, args.DIR+"/val_Y.pt")
torch.save(test_X, args.DIR+"/test_X.pt")
if test_Y is not None: torch.save(test_Y, args.DIR+"/test_Y.pt")

print("🍾 Done!")
