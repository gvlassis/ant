import os
import argparse
import datasets
datasets.logging.set_verbosity_error()
import utils
import torch

script_path = os.path.abspath(__file__)
datasets_path = os.path.dirname(script_path)
src_path = os.path.dirname(datasets_path)
root_path = os.path.dirname(src_path)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--out_path", metavar="PATH", help="The directory which will contain train_X.pt, train_Y.pt, val_X.pt, val_Y.pt, test_X.pt, test_Y.pt", type=os.path.abspath, default=root_path+"/cifar10")
args=parser.parse_args()

if not os.path.isdir(args.out_path):
    os.makedirs(args.out_path)

print("💾 Loading datasets")
cifar10_train_dataset = datasets.load_dataset("cifar10", split="train", trust_remote_code=True)
# cifar10 only offers a train and test split
cifar10_train_dataset = cifar10_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)

train_dataset = cifar10_train_dataset["train"]
val_dataset = cifar10_train_dataset["test"]
test_dataset = datasets.load_dataset("cifar10", split="test", trust_remote_code=True)

print("✏️ Preprocessing")
train_X, train_Y = utils.image_dataset_to_tensors(train_dataset)
test_X, test_Y = utils.image_dataset_to_tensors(test_dataset)
val_X, val_Y = utils.image_dataset_to_tensors(val_dataset)

print("💾 Saving datasets")
torch.save(train_X, args.out_path+"/train_X.pt")
torch.save(train_Y, args.out_path+"/train_Y.pt")
torch.save(val_X, args.out_path+"/val_X.pt")
torch.save(val_Y, args.out_path+"/val_Y.pt")
torch.save(test_X, args.out_path+"/test_X.pt")
torch.save(test_Y, args.out_path+"/test_Y.pt")

print("🍾 Done!")