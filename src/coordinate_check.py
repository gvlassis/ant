import torch
import os
import argparse
import data.utils_data
import models.utils_models
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("SUBPATH", help="Training log will be saved in SUBPATH.dat", type=os.path.abspath)

parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="california_housing")
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="mlp")
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")

parser.add_argument("--decoupling", help="Decouples c/k_input, c/k_hidden and c/k_output. If coupled, they are controlled by c/k_input.", type=utils.str_to_bool, default=False)
parser.add_argument("--c_input", type=float, default=0.05)
parser.add_argument("--c_hidden", type=float, default=0.05)
parser.add_argument("--c_output", type=float, default=0.05)
parser.add_argument("--optimizer", choices=models.parametrizations.OPTIMIZERS, default="adam")
parser.add_argument("--k_input", type=float, default=1e-3)
parser.add_argument("--k_hidden", type=float, default=1e-3)
parser.add_argument("--k_output", type=float, default=1e-3)
parser.add_argument("--β1", type=float, default=0.9)
parser.add_argument("--β2", type=float, default=0.999)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--context", type=int, default=512)
parser.add_argument("--batches", help="The number of batches used", type=int, default=15)
parser.add_argument("--update_freq", help="Every how many batches the norm will be printed", type=int, default=3)
args=parser.parse_args()

subpath_dir = os.path.dirname(args.SUBPATH)
os.makedirs(subpath_dir, exist_ok=True)
log_path = args.SUBPATH+".dat"

if args.decoupling:
    c_input = args.c_input
    c_hidden = args.c_hidden
    c_output = args.c_output
    k_input = args.k_input
    k_hidden = args.k_hidden
    k_output = args.k_output
else:
    c_input = args.c_input
    c_hidden = args.c_input
    c_output = args.c_input
    k_input = args.k_input
    k_hidden = args.k_input
    k_output = args.k_input

device = "cuda:0"

print("💾 Loading dataset")
iterator = data.utils_data.get_iterator(args.dataset, "train", "cpu", args.batch_size, args.context)

print("\x1b[1m%6.6s" % "ζ", end="")
with open(log_path,"w") as file:
    file.write("ζ")

for parametrization in models.parametrizations.PARAMETRIZATIONS:
    for batch in range(0, args.batches, args.update_freq):
        print(" %10.10s %16.16s %16.16s" % (f"{parametrization}.{batch}.out", f"{parametrization}.{batch}.grad_mean", f"{parametrization}.{batch}.data_mean"), end="")
        with open(log_path,"a") as file:
            file.write(f" {parametrization}.{batch}.out {parametrization}.{batch}.grad_mean {parametrization}.{batch}.data_mean")

print("\x1b[0m")
with open(log_path, "a") as file:
    file.write("\n")

for ζ in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    print("%6.6s" % ζ, end="")
    with open(log_path, "a") as file:
        file.write("%d" % ζ)

    for parametrization in models.parametrizations.PARAMETRIZATIONS:
        model, optimizer = models.utils_models.get_model_optimizer(args.vocab_size, args.family, parametrization, args.scale_type, ζ, c_input, c_hidden, c_output, k_input, k_hidden, k_output, args.optimizer, False, False, (args.β1, args.β2), 0, args.context, False, True)
        model = model.to(device)
        
        model.train()
        for batch in range(args.batches):
            batch_X, batch_Y = next(iterator)
                
            batch_Y_, loss = data.utils_data.get_loss(args.dataset, model, batch_X.to(device), batch_Y.to(device), False)
            
            loss.backward()
            optimizer.step()

            # Access gradients before they become None
            if batch % args.update_freq == 0:
                out, grad_mean, data_mean = models.utils_models.get_batch_stats(args.family, model, batch_Y_)
                
                print(" %10.10s %16.16s %16.16s" % ("%f" % out, "%f" % grad_mean, "%f" % data_mean), end="")
                with open(log_path, "a") as file:
                    file.write(" %f %f %f" % (out, grad_mean, data_mean))

            optimizer.zero_grad()

    print()
    with open(log_path,"a") as file:
        file.write("\n")
