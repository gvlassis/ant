import os
import argparse
import tokenizers
import datasets
datasets.logging.set_verbosity_error()
import utils
import torch

script_path = os.path.abspath(__file__)
datasets_path = os.path.dirname(script_path)
src_path = os.path.dirname(datasets_path)
root_path = os.path.dirname(src_path)
openwebtext_path = root_path+"/openwebtext"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_path", metavar="PATH", help="The directory which will contain train_X.pt, val_X.pt, test_X.pt. They are gonna be stored as torch.int16, because there is not torch.uint16", type=os.path.abspath, default=openwebtext_path)
args=parser.parse_args()

if not os.path.isdir(args.out_path):
    os.makedirs(args.out_path)

tokenizer = tokenizers.Tokenizer.from_pretrained("gpt2")
tokenizer.encode_special_tokens = False
eot_id = 50256

print("💾 Loading datasets")
openwebtext_train_dataset = datasets.load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
# openwebtext only offers a train split
openwebtext_train_dataset = openwebtext_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
train_val_dataset = openwebtext_train_dataset["train"]
train_val_dataset = train_val_dataset.train_test_split(train_size=None, test_size=500, shuffle=True)
train_dataset = train_val_dataset["train"]
val_dataset = train_val_dataset["test"]
test_dataset = openwebtext_train_dataset["test"]

print("🪙  Tokenization")
train_X = utils.text_dataset_to_tensor(train_dataset, tokenizer, eot_id)
val_X = utils.text_dataset_to_tensor(val_dataset, tokenizer, eot_id)
test_X = utils.text_dataset_to_tensor(test_dataset, tokenizer, eot_id)

print("💾 Saving datasets")
torch.save(train_X, args.out_path+"/train_X.pt")
torch.save(val_X, args.out_path+"/val_X.pt")
torch.save(test_X, args.out_path+"/test_X.pt")

print("🍾 Done!")