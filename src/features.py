import argparse
import os
import torch
import models.utils_models
import data.utils_data
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="openwebtext")
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--context", type=int, default=1024)

parser.add_argument("--block", help="Transformer block whose output we are checking", type=int, default=-1)
parser.add_argument("--start", help="Starting x for the CDF visualization", type=float, default=0.01)
parser.add_argument("--stop", help="Stopping x for the CDF visualization", type=float, default=1000)
parser.add_argument("--num", help="Number of samples for the CDF visualization", type=int, default=1000)
args=parser.parse_args()

device = "cuda:0"

print("💾 Loading dataset")
iterator = data.utils_data.get_iterator(args.dataset, "train", "cpu", args.batch_size, args.context)
batch_X, _ = next(iterator)

print("🧠 Initializing models")
for ζ in [1,2,4,8,16]:
    print(f"\x1b[1mζ={ζ}\x1b[0m")
    
    utils.write_features(args.vocab_size, args.family, "sp", args.scale_type, ζ, args.context, "sp", device, args.dataset, batch_X, args.block, args.start, args.stop, args.num)

    utils.write_features(args.vocab_size, args.family, "mup", args.scale_type, ζ, args.context, "mup", device, args.dataset, batch_X, args.block, args.start, args.stop, args.num)
