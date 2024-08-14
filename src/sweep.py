import torch
import os
import argparse
import uuid

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Training logs and models will be saved in PATH/hyper=·/seed.dat and PATH/hyper=·/seed.pt respectively", type=os.path.abspath)
parser.add_argument("SEEDS", help="The number of models trained for each hyperparameter value", type=int)
parser.add_argument("TRAIN_ARGS", nargs=argparse.REMAINDER, help="Optional arguments for train.py")
args=parser.parse_args()

# for k in [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1]:
for k in [1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2, 1e-1]:
    print("⚙️ k=%f" % (k))
    k_path = "%s/k=%f" % (args.PATH, k)

    for _ in range(args.SEEDS):
        # μs since Epoch
        seed = str(uuid.uuid4())
        seed_subpath = k_path+"/"+seed

        print("🌱 seed=%s" % (seed))
        
        os.system("python %s/train.py %s --k %f %s" % (src_path, seed_subpath, k, " ".join(args.TRAIN_ARGS)))
