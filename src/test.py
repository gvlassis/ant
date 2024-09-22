import argparse
import os
import models.transformer
import data.utils_data
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="openwebtext")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--context", type=int, default=1024)
parser.add_argument("--batches", help="The number of batches used", type=int, default=100)
args=parser.parse_args()

model_device = "cuda:0"
dataset_device = "cpu"

print("🧠 Initializing model")
model = models.transformer.Transformer.from_pretrained(args.PATH).to(model_device)
# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

print("💾 Loading dataset")
train_iterator = data.utils_data.get_iterator(args.dataset, "train", dataset_device, args.batch_size, args.context)
val_iterator = data.utils_data.get_iterator(args.dataset, "val", dataset_device, args.batch_size, args.context)
test_iterator = data.utils_data.get_iterator(args.dataset, "test", dataset_device, args.batch_size, args.context)

train_loss = data.utils_data.approximate_loss(args.batches, train_iterator, args.dataset, model)
val_loss = data.utils_data.approximate_loss(args.batches, val_iterator, args.dataset, model)
test_loss = data.utils_data.approximate_loss(args.batches, test_iterator, args.dataset, model)

print(f"train_loss: {train_loss}")
print(f"val_loss: {val_loss}")
print(f"test_loss: {test_loss}")
