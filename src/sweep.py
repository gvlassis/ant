import torch
import os
import argparse
import numpy
import uuid

script_path = os.path.abspath(__file__)
src_path = os.path.dirname(script_path)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("PATH", help="Training logs will be saved in PATH/hyper=·/seed.dat", type=os.path.abspath)

# Cannot have optional arguments with argparse.REMAINDER
parser.add_argument("HYPER", help="The hyperparameter swept")
parser.add_argument("START", help="Smallest value of the hyperparameter sweep", type=float)
parser.add_argument("STOP", help="Largest value of the hyperparameter sweep", type=float)
parser.add_argument("NUM", help="Number of geometrically spaced values between start and stop", type=int)

parser.add_argument("SEEDS", help="The number of models trained for each hyperparameter value", type=int)

parser.add_argument("TRAIN_ARGS", nargs=argparse.REMAINDER, help="Optional arguments for train.py")
args=parser.parse_args()

for hyper in numpy.geomspace(args.START, args.STOP, args.NUM):
    print("⚙️ %s=%f" % (args.HYPER, hyper))
    hyper_path = "%s/%s=%f" % (args.PATH, args.HYPER, hyper)

    for _ in range(args.SEEDS):
        seed = str(uuid.uuid4())
        seed_subpath = hyper_path+"/"+seed

        command = "python %s/train.py --%s %f %s %s" % (src_path, args.HYPER, hyper, " ".join(args.TRAIN_ARGS), seed_subpath)

        print("🌱 "+ command)
        os.system(command)
