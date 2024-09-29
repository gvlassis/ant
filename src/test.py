import argparse
import os
import models.utils_models
import torch
import data.utils_data
import transformers

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Path of the model to be used", type=os.path.abspath)

parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="openwebtext")
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522)", choices=models.parametrizations.PARAMETRIZATIONS, default="sp")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=12)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--context", type=int, default=1024)
parser.add_argument("--batches", help="The number of batches used", type=int, default=100)
args=parser.parse_args()

model_device = "cuda:0"
dataset_device = "cpu"

print("🧠 Initializing model")
model, _ = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, ζ=args.ζ, max_context=args.context)
model.load_state_dict(torch.load(args.PATH))

# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-medium")
# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large")
# model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-xl")

model = model.to(model_device)

print("💾 Loading dataset")
train_iterator = data.utils_data.get_iterator(args.dataset, "train", dataset_device, args.batch_size, args.context)
val_iterator = data.utils_data.get_iterator(args.dataset, "val", dataset_device, args.batch_size, args.context)
test_iterator = data.utils_data.get_iterator(args.dataset, "test", dataset_device, args.batch_size, args.context)

train_loss = data.utils_data.approximate_loss(args.batches, train_iterator, args.dataset, model)
print("train_loss: %f" % train_loss)

val_loss = data.utils_data.approximate_loss(args.batches, val_iterator, args.dataset, model)
print("val_loss: %f" % val_loss)

test_loss = data.utils_data.approximate_loss(args.batches, test_iterator, args.dataset, model)
print("test_loss: %f" % test_loss)
