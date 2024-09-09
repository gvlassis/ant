import os
import datasets
datasets.logging.set_verbosity_error()
import torch
import torchvision.transforms.v2

script_path = os.path.abspath(__file__)
data_path = os.path.dirname(script_path)
src_path = os.path.dirname(data_path)
root_path = os.path.dirname(src_path)

DATASETS_TABULAR = ["california_housing"]
DATASETS_IMAGE = ["cifar10"]
DATASETS_TEXT = ["shakespearefirstfolio", "finewebedu", "ancient_greek_theatre", "culturay_el"]
DATASETS = DATASETS_TABULAR + DATASETS_IMAGE + DATASETS_TEXT

def get_splits(dataset):
    if dataset=="california_housing":
        train_dataset = datasets.load_dataset("gvlassis/california_housing", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("gvlassis/california_housing", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("gvlassis/california_housing", split="test", trust_remote_code=True)
    elif dataset=="cifar10":
        cifar10_train_dataset = datasets.load_dataset("cifar10", split="train", trust_remote_code=True)
        cifar10_train_dataset = cifar10_train_dataset.train_test_split(train_size=None, test_size=10_000, shuffle=True)
        train_dataset = cifar10_train_dataset["train"]
        val_dataset = cifar10_train_dataset["test"]
        test_dataset = datasets.load_dataset("cifar10", split="test", trust_remote_code=True)
    elif dataset=="shakespearefirstfolio":
        train_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="train", trust_remote_code=True)
        val_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="validation", trust_remote_code=True)
        test_dataset = datasets.load_dataset("gvlassis/shakespearefirstfolio", split="test", trust_remote_code=True)
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

        self.X = torch.load(X_path, map_location=device)
        self.Y = torch.load(Y_path, map_location=device)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def tabular_dataset_to_tensors(dataset):
    dataset = torch.tensor(dataset.to_pandas().values)
    
    # Target is last column
    X = dataset[:,:-1]
    Y = dataset[:,-1]

    print("\x1b[36m%d\x1b[0m samples, %d features" % (X.shape[0], X.shape[1]))

    return X, Y

class ImageDataset(torch.utils.data.Dataset):
    # device is only used for initialization
    def __init__(self, dataset_path, split="train", device=None):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        X_path = "%s/%s_X.pt" % (dataset_path, split)
        Y_path = "%s/%s_Y.pt" % (dataset_path, split)

        self.X = torch.load(X_path, map_location=device)
        self.Y = torch.load(Y_path, map_location=device)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return len(self.X)

def image_dataset_to_tensors(dataset):
    cores = os.cpu_count()

    X_dataset = dataset.map(lambda sample: {"tensor": torchvision.transforms.v2.functional.pil_to_tensor(sample["img"])}, remove_columns=["label"], num_proc=cores)
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

        self.X = torch.load(X_path, map_location=device)

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
    if "img" in dataset.column_names:
        tensors = image_dataset_to_tensors(dataset)
    elif "text" in dataset.column_names:
        tensors = (text_dataset_to_tensor(dataset, tokenizer, eot_id), None)
    else:
        tensors = tabular_dataset_to_tensors(dataset)

    return tensors

def get_train_dataloader(dataset, device, batch_size, context=1024):
    dataset_path = "%s/%s" % (root_path, dataset)

    if dataset in DATASETS_TABULAR:
        train_dataset = TabularDataset(dataset_path, split="train", device=device)
    elif dataset in DATASETS_IMAGE:
        train_dataset = ImageDataset(dataset_path, split="train", device=device)
    elif dataset in DATASETS_TEXT:
        train_dataset = TextDataset(dataset_path, split="train", device=device, context=context)
    
    # shuffle=True hangs
    train_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)

    return train_dataloader

def get_val_dataloader(dataset, device, batch_size, context=1024):
    dataset_path = "%s/%s" % (root_path, dataset)

    if dataset in DATASETS_TABULAR:
        val_dataset = TabularDataset(dataset_path, split="val", device=device)
    elif dataset in DATASETS_IMAGE:
        val_dataset = ImageDataset(dataset_path, split="val", device=device)
    elif dataset in DATASETS_TEXT:
        val_dataset = TextDataset(dataset_path, split="val", device=device, context=context)
    
    # shuffle=True hangs
    val_sampler = torch.utils.data.RandomSampler(val_dataset, replacement=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)

    return val_dataloader

def transform(dataset, x):
    if dataset in DATASETS_TABULAR:
        x = x.to(torch.float32)
    elif dataset in DATASETS_IMAGE:
        x = x.to(torch.float32)
    elif dataset in DATASETS_TEXT:
        x = x.to(torch.int32) - torch.iinfo(torch.int16).min
        
    return x

def get_loss(dataset, batch_Y_, batch_Y):
    if dataset in DATASETS_TABULAR:
        loss = torch.nn.functional.mse_loss(
            batch_Y_[...,0],
            batch_Y.to(torch.float32)
        )

    elif dataset in DATASETS_IMAGE:
        loss = torch.nn.functional.cross_entropy(
            batch_Y_,
            batch_Y.to(torch.int64)
        )

    elif dataset in DATASETS_TEXT:
        loss = torch.nn.functional.cross_entropy(
            torch.reshape(batch_Y_, (-1,batch_Y_.shape[-1])),
            torch.flatten(batch_Y.to(torch.int64) - torch.iinfo(torch.int16).min)
        )
        
    return loss
