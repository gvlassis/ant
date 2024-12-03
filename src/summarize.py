import argparse
import os
import utils
import natsort
import math
import numpy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("DIR", help="Training logs are expected in DIR/**/hyper=·/seed.dat", type=os.path.abspath)
parser.add_argument("HYPER", help="The hyperparameter swept")
args=parser.parse_args()

files = utils.get_files(args.DIR)
archs = natsort.natsorted(utils.match_list(files, f"^(.*)/{args.HYPER}=.*/.*\.dat", 1))
print(f"🏛️ archs: \x1b[33;1m{len(archs)}\x1b[0m")

min_train_loss_min_archs = utils.INF
min_train_loss_mean_archs = utils.INF
min_val_loss_min_archs = utils.INF
min_val_loss_mean_archs = utils.INF
for arch in archs:
    summary = arch+"/summary.dat"
    
    hypers = sorted( [float(child.split("=")[-1]) for child in utils.get_subdir(arch) if child.split("=")[0]==args.HYPER] )
    seeds = [ utils.get_subdat("%s/%s=%f" % (arch, args.HYPER, hyper)) for hyper in hypers ]

    print(f"{arch} (hypers: \x1b[33;3m{len(hypers)}\x1b[0m, seeds: \x1b[33;3m{utils.numel(seeds)}\x1b[0m)")

    print("\x1b[1m%8.8s %20.20s %20.20s %20.20s %20.20s %20.20s %20.20s %20.20s %20.20s\x1b[0m" % (args.HYPER,"min_train_loss_min","min_train_loss_mean","min_train_loss_top","min_train_loss_bot","min_val_loss_min","min_val_loss_mean","min_val_loss_top","min_val_loss_bot"))
    with open(summary,"w") as file:
        file.write(f"{args.HYPER} min_train_loss_min min_train_loss_mean min_train_loss_top min_train_loss_bot min_val_loss_min min_val_loss_mean min_val_loss_top min_val_loss_bot\n")
    
    min_train_loss_min_hypers = utils.INF
    min_train_loss_mean_hypers = utils.INF
    min_val_loss_min_hypers = utils.INF
    min_val_loss_mean_hypers = utils.INF
    arch_diverged = 0
    for i, hyper in enumerate(hypers):
        hyper_path = "%s/%s=%f" % (arch, args.HYPER, hyper)
        
        min_train_losses = []
        min_val_losses = []
        for seed in seeds[i]:
            log_path = "%s/%s.dat" % (hyper_path, seed)
            
            with open(log_path,"r") as file:
                # Skip header
                header = file.readline()

                min_train_loss = utils.INF
                min_val_loss = utils.INF
                seed_diverged = False
                for line in file:
                    cols = line.rstrip().split(' ')
                    train_batch, train_loss, val_loss = int(cols[0]), float(cols[2]), float(cols[3])
                    
                    if min_train_loss>train_loss:
                        min_train_loss = train_loss
                        
                    if min_val_loss>val_loss:
                        min_val_loss = val_loss
                    
                    if math.isnan(train_loss):
                        arch_diverged+=1
                        seed_diverged = True
                        break
            
            if not seed_diverged:
                min_train_losses.append(min_train_loss)
                min_val_losses.append(min_val_loss)
        
        if min_train_losses == []:
            break
        
        min_train_loss_min = numpy.min(min_train_losses)
        min_train_loss_mean = numpy.mean(min_train_losses)
        min_train_loss_std = numpy.std(min_train_losses)
        min_train_loss_top = min_train_loss_mean+min_train_loss_std
        # Loss cannot be negative
        min_train_loss_bot = max(min_train_loss_mean-min_train_loss_std,0)

        min_val_loss_min = numpy.min(min_val_losses)
        min_val_loss_mean = numpy.mean(min_val_losses)
        min_val_loss_std = numpy.std(min_val_losses)
        min_val_loss_top = min_val_loss_mean+min_val_loss_std
        # Loss cannot be negative
        min_val_loss_bot = max(min_val_loss_mean-min_val_loss_std,0)
        
        hyper_decorated = "%8.8s" % ("%f" % hyper)
        # min_train_loss_min
        if min_train_loss_min<min_train_loss_min_archs:
            min_train_loss_min_archs = min_train_loss_min
            min_train_loss_min_hypers = min_train_loss_min
            min_train_loss_min_decorated = "\x1b[35;1m%20.20s\x1b[0m" % ("%f" % min_train_loss_min)
        elif min_train_loss_min<min_train_loss_min_hypers:
            min_train_loss_min_hypers = min_train_loss_min
            min_train_loss_min_decorated = "\x1b[35m%20.20s\x1b[0m" % ("%f" % min_train_loss_min)
        else:
            min_train_loss_min_decorated = "%20.20s" % ("%f" % min_train_loss_min)
        # min_train_loss_mean
        if min_train_loss_mean<min_train_loss_mean_archs:
            min_train_loss_mean_archs = min_train_loss_mean
            min_train_loss_mean_hypers = min_train_loss_mean
            min_train_loss_mean_decorated = "\x1b[95;1m%20.20s\x1b[0m" % ("%f" % min_train_loss_mean)
        elif min_train_loss_mean<min_train_loss_mean_hypers:
            min_train_loss_mean_hypers = min_train_loss_mean
            min_train_loss_mean_decorated = "\x1b[95m%20.20s\x1b[0m" % ("%f" % min_train_loss_mean)
        else:
            min_train_loss_mean_decorated = "%20.20s" % ("%f" % min_train_loss_mean)
        min_train_loss_top_decorated = "%20.20s" % ("%f" % min_train_loss_top)
        min_train_loss_bot_decorated = "%20.20s" % ("%f" % min_train_loss_bot)
        # min_val_loss_min
        if min_val_loss_min<min_val_loss_min_archs:
            min_val_loss_min_archs = min_val_loss_min
            min_val_loss_min_hypers = min_val_loss_min
            min_val_loss_min_decorated = "\x1b[36;1m%20.20s\x1b[0m" % ("%f" % min_val_loss_min)
        elif min_val_loss_min<min_val_loss_min_hypers:
            min_val_loss_min_hypers = min_val_loss_min
            min_val_loss_min_decorated = "\x1b[36m%20.20s\x1b[0m" % ("%f" % min_val_loss_min)
        else:
            min_val_loss_min_decorated = "%20.20s" % ("%f" % min_val_loss_min)
        # min_val_loss_mean
        if min_val_loss_mean<min_val_loss_mean_archs:
            min_val_loss_mean_archs = min_val_loss_mean
            min_val_loss_mean_hypers = min_val_loss_mean
            min_val_loss_mean_decorated = "\x1b[96;1m%20.20s\x1b[0m" % ("%f" % min_val_loss_mean)
        elif min_val_loss_mean<min_val_loss_mean_hypers:
            min_val_loss_mean_hypers = min_val_loss_mean
            min_val_loss_mean_decorated = "\x1b[96m%20.20s\x1b[0m" % ("%f" % min_val_loss_mean)
        else:
            min_val_loss_mean_decorated = "%20.20s" % ("%f" % min_val_loss_mean)
        min_val_loss_top_decorated = "%20.20s" % ("%f" % min_val_loss_top)
        min_val_loss_bot_decorated = "%20.20s" % ("%f" % min_val_loss_bot)

        print("%s %s %s %s %s %s %s %s %s" % (hyper_decorated, min_train_loss_min_decorated, min_train_loss_mean_decorated, min_train_loss_top_decorated, min_train_loss_bot_decorated, min_val_loss_min_decorated, min_val_loss_mean_decorated, min_val_loss_top_decorated, min_val_loss_bot_decorated))
        with open(summary,"a") as file:
            file.write("%f %f %f %f %f %f %f %f %f\n" % (hyper, min_train_loss_min, min_train_loss_mean, min_train_loss_top, min_train_loss_bot, min_val_loss_min, min_val_loss_mean, min_val_loss_top, min_val_loss_bot))
    
    if arch_diverged > 0:
        arch_diverged_decorated = "\x1b[31;1m%d\x1b[0m" % arch_diverged
    else:
        arch_diverged_decorated = "\x1b[32m%d\x1b[0m" % arch_diverged

    print(f"arch_diverged: {arch_diverged_decorated}")
    print()
