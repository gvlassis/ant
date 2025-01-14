import os
import datasets
datasets.logging.set_verbosity_error()
import torch
import torchvision.transforms.v2
import sys
from math import sqrt
import collections

script_path = os.path.abspath(__file__)
data_path = os.path.dirname(script_path)
src_path = os.path.dirname(data_path)
root_path = os.path.dirname(src_path)

DATASETS_TABULAR = ["california_housing"]
DATASETS_IMAGE = ["mnist", "cifar10", "tinyimagenet"]
DATASETS_TEXT = ["shakespearefirstfolio", "wikitext", "minipile", "openwebtext", "finewebedu", "ancient_greek_theatre", "culturay_el"]
DATASETS = DATASETS_TABULAR + DATASETS_IMAGE + DATASETS_TEXT

def get_splits(dataset):
    if dataset=="california_housing":
        train_dataset = datasets.load_dataset("gvlassis/california_housing", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("gvlassis/california_housing", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("gvlassis/california_housing", split="test", trust_remote_code=True)
    elif dataset=="mnist":
        mnist_train_dataset = datasets.load_dataset("ylecun/mnist", split="train", trust_remote_code=True)
        mnist_train_dataset = mnist_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_dataset = mnist_train_dataset["train"]
        val_dataset = mnist_train_dataset["test"]
        test_dataset = datasets.load_dataset("ylecun/mnist", split="test", trust_remote_code=True)
    elif dataset=="cifar10":
        cifar10_train_dataset = datasets.load_dataset("uoft-cs/cifar10", split="train", trust_remote_code=True)
        cifar10_train_dataset = cifar10_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_dataset = cifar10_train_dataset["train"]
        val_dataset = cifar10_train_dataset["test"]
        test_dataset = datasets.load_dataset("uoft-cs/cifar10", split="test", trust_remote_code=True)
    elif dataset=="tinyimagenet":
        tinyimagenet_train_dataset = datasets.load_dataset("zh-plus/tiny-imagenet", split="train", trust_remote_code=True)
        tinyimagenet_train_dataset = tinyimagenet_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_dataset = tinyimagenet_train_dataset["train"]
        test_dataset = tinyimagenet_train_dataset["test"]
        val_dataset = datasets.load_dataset("zh-plus/tiny-imagenet", split="valid", trust_remote_code=True)
    elif dataset=="shakespearefirstfolio":
        train_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="test", trust_remote_code=True)
    elif dataset=="wikitext":
        train_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-103-v1", split="test", trust_remote_code=True)
    elif dataset=="minipile":
        train_dataset = datasets.load_dataset("JeanKaddour/minipile", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("JeanKaddour/minipile", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("JeanKaddour/minipile", split="test", trust_remote_code=True)
    elif dataset=="openwebtext":
        openwebtext_train_dataset = datasets.load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)
        openwebtext_train_dataset = openwebtext_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_val_dataset = openwebtext_train_dataset["train"]
        train_val_dataset = train_val_dataset.train_test_split(train_size=None, test_size=500, shuffle=True)
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = openwebtext_train_dataset["test"]
    elif dataset=="finewebedu":
        finewebedu_train_dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", trust_remote_code=True)
        finewebedu_train_dataset = finewebedu_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_val_dataset = finewebedu_train_dataset["train"]
        train_val_dataset = train_val_dataset.train_test_split(train_size=None, test_size=500, shuffle=True)
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = finewebedu_train_dataset["test"]
    elif dataset=="ancient_greek_theatre":
        train_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("gvlassis/ancient_greek_theatre", split="test", trust_remote_code=True)
    elif dataset=="culturay_el":
        culturay_el_train_dataset = datasets.load_dataset("ontocord/CulturaY", name="el", split="train", trust_remote_code=True)
        culturay_el_train_dataset = culturay_el_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_val_dataset = culturay_el_train_dataset["train"]
        train_val_dataset = train_val_dataset.train_test_split(train_size=None, test_size=500, shuffle=True)
        train_dataset = train_val_dataset["train"]
        val_dataset = train_val_dataset["test"]
        test_dataset = culturay_el_train_dataset["test"]

    return train_dataset, val_dataset, test_dataset

class TabularDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        Y_path = "%s/%s_Y.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device, weights_only=True)
        self.Y = torch.load(Y_path, map_location=device, weights_only=True)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def tabular_dataset_to_tensors(dataset):
    dataset = torch.tensor(dataset.to_pandas().values)
    
    # Target is last column
    X = dataset[:,:-1]
    Y = dataset[:,-1]

    print("\x1b[36m%d\x1b[0m samples, %d feats" % (X.shape[0], X.shape[1]))

    return X, Y

class ImageDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        Y_path = "%s/%s_Y.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device, weights_only=True)
        self.Y = torch.load(Y_path, map_location=device, weights_only=True)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def image_dataset_to_tensors(dataset):
    cores = os.cpu_count()
    
    # Unpacking set to single element
    name, = set(["img", "image"]) & set(dataset.column_names)
    
    # Image datasets may contain mixed modes (e.g. ImageNet contains greyscale images)
    print("🌄 Modes")
    modes = [ image.mode for image in dataset[name] ]
    modes = collections.Counter(modes)
    modes = dict(sorted(modes.items(), key=lambda item: item[1], reverse=True))
    for mode in modes.keys():
        print("%6.6s: %10.10s" % (mode,modes[mode]))
    dataset = dataset.filter(lambda sample: sample[name].mode==next(iter(modes)))
    
    X_dataset = dataset.map(lambda sample: {"tensor": torchvision.transforms.v2.functional.pil_to_tensor(sample[name])}, remove_columns=["label"], num_proc=cores)
    X = torch.tensor(X_dataset["tensor"], dtype=torch.uint8)
    Y = torch.tensor(dataset["label"])

    print("\x1b[36m%d\x1b[0m images, %d×%d×%d" % (X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    return X, Y

class TextDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None, context=1024):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split
        self.context = context

        X_path = "%s/%s_X.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device, weights_only=True)

    # i is the first index of the context
    def __getitem__(self, i):
        X = self.X[i:i+self.context]
        Y = self.X[i+1:i+1+self.context]

        return X, Y

    def __len__(self):
        return self.X.numel()-self.context

def text_dataset_to_tensor(dataset, tokenizer, eot_id):
    cores = os.cpu_count()
    # PyTorch does not support uint16
    dataset = dataset.map(lambda sample: {"ids": tokenizer.encode(sample["text"], add_special_tokens=False).ids+[eot_id]}, remove_columns=["text"], num_proc=cores)

    ids = [element for sublist in dataset["ids"] for element in sublist]

    X = torch.tensor(ids)
    X = X + torch.iinfo(torch.int16).min
    X = X.to(torch.int16)

    print("\x1b[36m%d\x1b[0m tokens" % X.numel())

    return X

def dataset_to_tensors(dataset, tokenizer=None, eot_id=None):

    # Auto-detection
    if set(["img", "image"]) & set(dataset.column_names):
        tensors = image_dataset_to_tensors(dataset)
    elif "text" in dataset.column_names:
        tensors = (text_dataset_to_tensor(dataset, tokenizer, eot_id), None)
    else:
        tensors = tabular_dataset_to_tensors(dataset)

    return tensors

def get_iterator(dataset, split, device, batch_size, context):
    dataset_path = "%s/%s" % (root_path, dataset)

    if dataset in DATASETS_TABULAR:
        dataset = TabularDataset(dataset_path, split=split, device=device)
    elif dataset in DATASETS_IMAGE:
        dataset = ImageDataset(dataset_path, split=split, device=device)
    elif dataset in DATASETS_TEXT:
        dataset = TextDataset(dataset_path, split=split, device=device, context=context)
    
    # shuffle=True hangs, num_samples: Maximum samples (default: len(dataset))
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=sys.maxsize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    iterator = iter(dataloader)

    return iterator

def transform(dataset, x):
    if dataset in DATASETS_TABULAR:
        x = x.to(torch.float32)
    elif dataset in DATASETS_IMAGE:
        x = x.to(torch.float32)
    elif dataset in DATASETS_TEXT:
        x = x.to(torch.int32) - torch.iinfo(torch.int16).min
        
    return x

def get_loss(dataset, model, batch_X, batch_Y, label_smoothing=0):
    batch_Y_ = model(transform(dataset, batch_X))

    # huggingface
    # batch_Y_ = model(transform(dataset, batch_X)).logits

    if dataset in DATASETS_TABULAR:
        loss = torch.nn.functional.mse_loss(
            batch_Y_.flatten(),
            batch_Y.to(torch.float32)
        )

    elif dataset in DATASETS_IMAGE:
        loss = torch.nn.functional.cross_entropy(
            batch_Y_,
            batch_Y.to(torch.int64),
            label_smoothing = label_smoothing
        )

    elif dataset in DATASETS_TEXT:
        loss = torch.nn.functional.cross_entropy(
            torch.reshape(batch_Y_, (-1,batch_Y_.shape[-1])),
            torch.flatten(batch_Y.to(torch.int64) - torch.iinfo(torch.int16).min),
            label_smoothing = label_smoothing
        )
        
    return batch_Y_, loss

@torch.no_grad()
def approximate_loss(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    loss = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        loss += get_loss(dataset, model, batch_X.to(device), batch_Y.to(device), 0)[1].item()
    loss = loss/batches

    return loss

@torch.no_grad()
def approximate_rmse(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    mse = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mse += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
    batch_size = batch_X.shape[0]
    mse = mse/(batches*batch_size)
    rmse = sqrt(mse)

    return rmse

@torch.no_grad()
def approximate_nrmse(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    mse = 0
    minimum = float("+inf")
    maximum = float("-inf")
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        minimum = min(minimum, batch_Y.min().item())
        maximum = max(maximum, batch_Y.max().item())
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mse += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
    batch_size = batch_X.shape[0]
    mse = mse/(batches*batch_size)
    rmse = sqrt(mse)
    nrmse = rmse/(maximum-minimum)

    return nrmse

@torch.no_grad()
def approximate_mae(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    mae = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mae += (batch_Y.to(device) - batch_Y_).abs().sum().item()
    batch_size = batch_X.shape[0]
    mae = mae/(batches*batch_size)

    return mae

@torch.no_grad()
def approximate_nmae(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    mae = 0
    minimum = float("+inf")
    maximum = float("-inf")
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        minimum = min(minimum, batch_Y.min().item())
        maximum = max(maximum, batch_Y.max().item())
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        mae += (batch_Y.to(device) - batch_Y_).abs().sum().item()
    batch_size = batch_X.shape[0]
    mae = mae/(batches*batch_size)
    nmae = mae/(maximum-minimum)

    return nmae

@torch.no_grad()
def approximate_r2(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    rss = 0
    ss = 0
    mean = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()

        rss += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
        ss += (batch_Y.to(device)**2).sum().item()
        mean += batch_Y.to(device).sum().item()
    batch_size = batch_X.shape[0]
    mean = mean/(batches*batch_size)
    tss = ss - batches*batch_size*(mean**2)
    r2 = 1 - rss/tss
    
    return r2

@torch.no_grad()
def approximate_acc(batches, iterator, dataset, model):
    model.eval()

    device = next(model.parameters()).device
    
    correct = 0
    for batch in range(batches):
        batch_X, batch_Y = next(iterator)
        
        batch_Y_ = model(transform(dataset, batch_X.to(device))).argmax(dim=-1)
        
        correct += (batch_Y_ == batch_Y.to(device)).sum().item()
    batch_size = batch_X.shape[0]
    acc = correct/(batches*batch_size)

    return acc

# @torch.no_grad()
# def approximate_ppl(batches, iterator, dataset, model):
#     model.eval()
#
#     device = next(model.parameters()).device
#
#     ce = 0
#     for batch in range(batches):
#         batch_X, batch_Y = next(iterator)
#
#         batch_Y_ = model(transform(dataset, batch_X.to(device))).flatten()
#
#         ce += ((batch_Y.to(device) - batch_Y_)**2).sum().item()
#     batch_size = batch_X.shape[0]
#     mse = mse/(batches*batch_size)
#     ppl = sqrt(ce)
#
#     return ppl
#
#     batch_Y.to(torch.int64) - torch.iinfo(torch.int16).min
