import torch
import os
import argparse
import data.utils
import models.utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("SUBPATH", help="Training log will be saved in SUBPATH.dat")
parser.add_argument("--dataset", choices=data.utils.DATASETS_TABULAR+data.utils.DATASETS_IMAGE+data.utils.DATASETS_TEXT, default="california")
parser.add_argument("--family", choices=models.utils.FAMILIES, default="mlp")
parser.add_argument("--parametrization", choices=["sp", "mup"], default="mup")
parser.add_argument("--c", metavar="FLOAT", help="Initial standard deviation coefficient", type=float, default=1/10)
parser.add_argument("--k", metavar="FLOAT", help="Learning rate", type=float, default=5e-4)
parser.add_argument("--weight_decay", metavar="FLOAT", type=float, default=0)
parser.add_argument("--batch_size", metavar="INT", type=int, default=16)
parser.add_argument("--batches", metavar="INT", help="The number of batches used", type=int, default=5)
parser.add_argument("--update_freq", metavar="INT", help="Every how many batches the norm will be printed", type=int, default=1)
args=parser.parse_args()

device="cuda:0"

subpath_dir = os.path.dirname(args.SUBPATH)
if not os.path.isdir(subpath_dir):
    os.makedirs(subpath_dir)
log_path = args.SUBPATH+".dat"

print("💾 Loading dataset")
dataloader = data.utils.get_train_dataloader(args.dataset, device, args.batch_size)

suffix=""
for batch in range(0, args.batches, args.update_freq):
    suffix+=" batch=%d" % batch

print("\x1b[1mζ%s\x1b[0m" % suffix)
with open(log_path,"w") as file:
    file.write("ζ%s\n" % suffix)
    
for ζ in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    print("%d" % ζ, end="")
    with open(log_path,"a") as file:
        file.write("%d" % ζ)
    
    model, optimizer = models.utils.get_model_optimizer(args.family, args.parametrization, ζ, device, args.c, args.k, args.weight_decay)

    for batch in range(args.batches):
        try:
            batch_X, batch_Y = next(iterator)
        except (NameError, StopIteration):
            iterator = iter(dataloader)
            batch_X, batch_Y = next(iterator)

        model.train()
        
        batch_Y_ = model(batch_X)
            
        loss = data.utils.get_loss(args.dataset, batch_Y_, batch_Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%args.update_freq==0:
            quantity = model.layer2.weight.grad.abs().mean().item()
            # quantity = model.linear.weight.grad.abs().mean().item()
            # quantity = model.linear.weight.data.abs().mean().item()
            # quantity = batch_Y_.abs().mean().item()
            print(" %f" % quantity, end="")
            with open(log_path,"a") as file:
                file.write(" %f" % quantity)

    print()
    with open(log_path,"a") as file:
        file.write("\n")