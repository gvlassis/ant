import torch
import os

def get_subdir(path):
    return [child for child in os.listdir(path) if os.path.isdir(path+"/"+child)]

def get_subdat(path):
    return [child[:-4] for child in os.listdir(path) if child[-4:]==".dat"]
