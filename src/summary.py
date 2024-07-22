import argparse
import os
import utils
import numpy
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Training logs are expected in PATH/hyper=·/seed.dat", type=os.path.abspath)
args=parser.parse_args()

summary_path = args.PATH+"/summary.dat"

ks = sorted([float(child.split("=")[-1]) for child in utils.get_subdir(args.PATH)])
seeds = [ sorted([int(child) for child in utils.get_subdat("%s/k=%f" % (args.PATH,k))]) for k in ks ]

print("\x1b[1mk min_train_loss_mean min_train_loss_top min_train_loss_bot min_val_loss_mean min_val_loss_top min_val_loss_bot\x1b[0m")
with open(summary_path,"w") as file:
    file.write("k min_train_loss_mean min_train_loss_top min_train_loss_bot min_val_loss_mean min_val_loss_top min_val_loss_bot\n")

for i, k in enumerate(ks):
    k_path = "%s/k=%f" % (args.PATH, k)
    
    min_train_losses = []
    min_val_losses = []
    for seed in seeds[i]:
        log_path = "%s/%d.dat" % (k_path, seed)
        
        with open(log_path,"r") as file:
            # Skip header
            header = file.readline()

            min_train_loss = float("+inf")
            min_val_loss = float("+inf")
            for line in file:
                cols = line.rstrip().split(' ')
                train_batch, train_loss, val_loss = int(cols[0]), float(cols[1]), float(cols[2])
                
                if min_train_loss>train_loss:
                    min_train_loss = train_loss
                    
                if min_val_loss>val_loss:
                    min_val_loss = val_loss

        min_train_losses.append(min_train_loss)
        min_val_losses.append(min_val_loss)
    
    min_train_loss_mean = numpy.mean(min_train_losses)
    min_train_loss_std = numpy.std(min_train_losses)
    min_train_loss_top = min_train_loss_mean+min_train_loss_std
    min_train_loss_bot = min_train_loss_mean-min_train_loss_std

    min_val_loss_mean = numpy.mean(min_val_losses)
    min_val_loss_std = numpy.std(min_val_losses)
    min_val_loss_top = min_val_loss_mean+min_val_loss_std
    min_val_loss_bot = min_val_loss_mean-min_val_loss_std

    print("%f %f %f %f %f %f %f" % (k, min_train_loss_mean, min_train_loss_top, min_train_loss_bot, min_val_loss_mean, min_val_loss_top, min_val_loss_bot))
    with open(summary_path,"a") as file:
        file.write("%s %f %f %f %f %f %f\n" % (k, min_train_loss_mean, min_train_loss_top, min_train_loss_bot, min_val_loss_mean, min_val_loss_top, min_val_loss_bot))
