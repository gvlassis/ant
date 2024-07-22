import torch
import os
import argparse
import data.utils
import models.utils

def str_to_bool(string):
    if string == "True":
        boolean = True
    elif string == "False":
        boolean = False

    return boolean
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("SUBPATH", help="Training log will be saved in SUBPATH.dat")
parser.add_argument("--dataset", choices=data.utils.DATASETS_TABULAR+data.utils.DATASETS_IMAGE+data.utils.DATASETS_TEXT, default="california")
parser.add_argument("--family", choices=models.utils.FAMILIES, default="mlp")
parser.add_argument("--parametrization", choices=["sp", "mup"], default="mup")
parser.add_argument("--ζ", metavar="INT", help="Width scaling factor", type=int, default=8)
parser.add_argument("--c", metavar="FLOAT", help="Initial standard deviation coefficient", type=float, default=1/10)
parser.add_argument("--k", metavar="FLOAT", help="Learning rate", type=float, default=5e-4)
parser.add_argument("--weight_decay", metavar="FLOAT", type=float, default=0)
parser.add_argument("--batch_size", metavar="INT", type=int, default=16)
parser.add_argument("--context", metavar="INT", type=int, default=128)
parser.add_argument("--train_batches", metavar="INT", help="The number of batches used during training", type=int, default=50000)
parser.add_argument("--val_batches", metavar="INT", help="The number of batches used during validation", type=int, default=100)
parser.add_argument("--update_freq", metavar="INT", help="Every how many batches the train and the validation loss will be printed", type=int, default=50)
parser.add_argument("--save_model", help="Save the model with the min validation loss in SUBPATH.pt", type=str_to_bool, default=False)
parser.add_argument("--device_index", metavar="INT", help="CUDA device that stores the dataset and the models", type=int, default=0)
parser.add_argument("--dtype", metavar="DTYPE", help="torch.DTYPE for Automatic Mixed Precision (AMP)", type=lambda x: getattr(torch, x), default="float32")
parser.add_argument("--compile", help="Use or not torch.compile()", type=str_to_bool, default=False)
args=parser.parse_args()

device_type="cuda"
device="%s:%d" % (device_type, args.device_index)

subpath_dir = os.path.dirname(args.SUBPATH)
if not os.path.isdir(subpath_dir):
    os.makedirs(subpath_dir)
log_path = args.SUBPATH+".dat"
model_path = args.SUBPATH+".pt"

print("💾 Loading dataset")
train_dataloader = data.utils.get_train_dataloader(args.dataset, device, args.batch_size, context=args.context)
val_dataloader = data.utils.get_val_dataloader(args.dataset, device, args.batch_size, context=args.context)

print("🧠 Initializing model")
model, optimizer = models.utils.get_model_optimizer(args.family, args.parametrization, args.ζ, device, args.c, args.k, args.weight_decay)

suffix=""
for name, _ in model.named_parameters():
    suffix+=" %s.grad_mean %s.grad_top %s.grad_bot %s.grad_max %s.data_mean %s.data_top %s.data_bot %s.data_max" % (name, name, name, name, name, name, name, name)

print("\x1b[1mtrain_batch train_loss val_loss\x1b[0m")
with open(log_path,"w") as file:
    file.write("train_batch train_loss val_loss%s\n" % suffix)

scaler = torch.cuda.amp.GradScaler()
if args.compile:
    torch._inductor.config.compile_threads=1
    model = torch.compile(model, mode="max-autotune")

min_train_loss = float("+inf")
min_val_loss = float("+inf")
for train_batch in range(args.train_batches):
    try:
        batch_train_X, batch_train_Y = next(train_iterator)
    except (NameError, StopIteration):
        train_iterator = iter(train_dataloader)
        batch_train_X, batch_train_Y = next(train_iterator)

    model.train()
    with torch.autocast(device_type=device_type, dtype=args.dtype), torch.cuda.device(args.device_index):
        batch_train_Y_ = model(data.utils.transform(args.dataset, batch_train_X))
        
        train_loss = data.utils.get_loss(args.dataset, batch_train_Y_, batch_train_Y)

    optimizer.zero_grad()
    scaler.scale(train_loss).backward()

    if train_batch%args.update_freq==0:
        val_loss_sum = 0
        for _ in range(args.val_batches):
            try:
                batch_val_X, batch_val_Y = next(val_iterator)
            except (NameError, StopIteration):
                val_iterator = iter(val_dataloader)
                batch_val_X, batch_val_Y = next(val_iterator)

            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=args.dtype), torch.cuda.device(args.device_index):
                batch_val_Y_ = model(data.utils.transform(args.dataset, batch_val_X))
                
                batch_val_loss = data.utils.get_loss(args.dataset, batch_val_Y_, batch_val_Y)

            val_loss_sum += batch_val_loss.item()

        val_loss = val_loss_sum/args.val_batches

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if args.save_model: torch.save(model.state_dict(), model_path)
            val_loss_decorated = "\x1b[36;1m%f\x1b[0m" % val_loss
        else:
            val_loss_decorated = "%f" % val_loss

        if train_loss.item() < min_train_loss:
            min_train_loss = train_loss.item()
            train_loss_decorated = "\x1b[35;1m%f\x1b[0m" % train_loss.item()
        else:
            train_loss_decorated = "%f" % train_loss.item()

        suffix=""
        for name, parameter in model.named_parameters():
            # Numerically, I only care about the absolute values
            grad = parameter.grad.abs()
            grad_mean = grad.mean().item()
            # https://github.com/pytorch/pytorch/issues/29372
            if grad.numel()==1:
                grad_std = 0
            else:
                grad_std = grad.std().item()
            grad_top = grad_mean+grad_std
            grad_bot = grad_mean-grad_std
            grad_max = grad.max().item()
            
            # Numerically, I only care about the absolute values
            parameter_data = parameter.data.abs()
            data_mean = parameter_data.mean().item()
            # https://github.com/pytorch/pytorch/issues/29372
            if parameter_data.numel()==1:
                data_std = 0
            else:
                data_std = parameter_data.std().item()
            data_top = data_mean+data_std
            data_bot = data_mean-data_std
            data_max = parameter_data.max().item()

            suffix+=" %f %f %f %f %f %f %f %f" % (grad_mean, grad_top, grad_bot, grad_max, data_mean, data_top, data_bot, data_max)

        print("%d %s %s" % (train_batch, train_loss_decorated, val_loss_decorated))
        with open(log_path,"a") as file:
            file.write("%d %f %f%s\n" % (train_batch, train_loss.item(), val_loss, suffix))

    scaler.step(optimizer)
    scaler.update()
