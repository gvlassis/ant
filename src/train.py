import torch
import os
import argparse
import data.utils_data
import models.utils_models
import utils
import torchinfo
import torchview

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("SUBPATH", help="Training log will be saved in SUBPATH.dat", type=os.path.abspath)
parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="california_housing")
parser.add_argument("--vocab_size", type=int, default=50257)
parser.add_argument("--family", choices=models.utils_models.FAMILIES, default="mlp")
parser.add_argument("--parametrization", choices=models.utils_models.PARAMETRIZATIONS, default="mup")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=8)
parser.add_argument("--c", help="Initial standard deviation coefficient", type=float, default=1/10)
parser.add_argument("--k", help="Learning rate", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--context", type=int, default=128)
parser.add_argument("--train_batches", help="The number of batches used during training", type=int, default=50000)
parser.add_argument("--val_batches", help="The number of batches used during validation", type=int, default=10)
parser.add_argument("--update_freq", help="Every how many batches the train and the validation loss will be printed", type=int, default=200)
parser.add_argument("--device_index", help="CUDA device that stores the dataset and the models", type=int, default=0)
parser.add_argument("--dtype", help="torch.dtype for Automatic Mixed Precision (AMP)", type=lambda x: getattr(torch, x), default="float32")
parser.add_argument("--compile", help="Use or not torch.compile()", type=utils.str_to_bool, default=False)
parser.add_argument("--save_model", help="Save the model with the min validation loss in SUBPATH.pt", type=utils.str_to_bool, default=False)
parser.add_argument("--info", help="Print or not information about the model", type=utils.str_to_bool, default=False)
parser.add_argument("--graph", help="SUBPATH.", type=utils.str_to_bool, default=False)
args=parser.parse_args()

device_type="cuda"
device="%s:%d" % (device_type, args.device_index)

subpath_dir = os.path.dirname(args.SUBPATH)
os.makedirs(subpath_dir, exist_ok=True)
log_path = args.SUBPATH+".dat"
graph_path = args.SUBPATH+".pdf"

print("💾 Loading dataset")
train_dataloader = data.utils_data.get_train_dataloader(args.dataset, device, args.batch_size, args.context)
val_dataloader = data.utils_data.get_val_dataloader(args.dataset, device, args.batch_size, args.context)

print("🧠 Initializing model")
model, optimizer = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.ζ, args.c, args.k, args.weight_decay, args.context, device)
if args.info:
    batch_X, _ = next(iter(train_dataloader))
    input_data = data.utils_data.transform(args.dataset, batch_X)
    torchinfo.summary(model, input_data=input_data, col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"), col_width=18, depth=3)
if args.graph:
    batch_X, _ = next(iter(train_dataloader))
    input_data = data.utils_data.transform(args.dataset, batch_X)
    torchview.draw_graph(model, input_data=input_data, depth=1, expand_nested=True, graph_dir="TB", show_shapes=True).visual_graph.render(cleanup=True, format="pdf", outfile=graph_path)

suffix=""
for name, _ in model.named_parameters():
    suffix+=" %s.grad_mean %s.grad_top %s.grad_bot %s.grad_max %s.data_mean %s.data_top %s.data_bot %s.data_max" % (name, name, name, name, name, name, name, name)

print("\x1b[1m%12.12s %12.12s %12.12s %12.12s %12.12s %12.12s\x1b[0m" % ("train_batch", "train_loss", "val_loss", "forward", "backward", "total"))
with open(log_path,"w") as file:
    file.write("train_batch train_loss val_loss%s\n" % suffix)

scaler = torch.cuda.amp.GradScaler()
if args.compile:
    torch._inductor.config.compile_threads=1
    model = torch.compile(model, mode="max-autotune")

min_train_loss = float("+inf")
min_val_loss = float("+inf")
for train_batch in range(args.train_batches):
    total_start = utils.get_sync_time(device)

    try:
        batch_train_X, batch_train_Y = next(train_iterator)
    except (NameError, StopIteration):
        train_iterator = iter(train_dataloader)
        batch_train_X, batch_train_Y = next(train_iterator)

    model.train()
    with torch.autocast(device_type=device_type, dtype=args.dtype), torch.cuda.device(args.device_index):
        forward_start = utils.get_sync_time(device)
        batch_train_Y_ = model(data.utils_data.transform(args.dataset, batch_train_X))
        
        train_loss = data.utils_data.get_loss(args.dataset, batch_train_Y_, batch_train_Y)
        forward_end = utils.get_sync_time(device)

    optimizer.zero_grad()
    backward_start = utils.get_sync_time(device)
    scaler.scale(train_loss).backward()
    backward_end = utils.get_sync_time(device)

    total_end = utils.get_sync_time(device)

    if train_batch%args.update_freq==0:
        train_batch_decorated = "%12.12s" % train_batch
        
        val_loss_sum = 0
        for _ in range(args.val_batches):
            try:
                batch_val_X, batch_val_Y = next(val_iterator)
            except (NameError, StopIteration):
                val_iterator = iter(val_dataloader)
                batch_val_X, batch_val_Y = next(val_iterator)

            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=args.dtype), torch.cuda.device(args.device_index):
                batch_val_Y_ = model(data.utils_data.transform(args.dataset, batch_val_X))
                
                batch_val_loss = data.utils_data.get_loss(args.dataset, batch_val_Y_, batch_val_Y)

            val_loss_sum += batch_val_loss.item()

        val_loss = val_loss_sum/args.val_batches

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if args.save_model: model.save_pretrained(args.SUBPATH)
            val_loss_decorated = "\x1b[36;1m%12.12s\x1b[0m" % ("%f" % val_loss)
        else:
            val_loss_decorated = "%12.12s" % ("%f" % val_loss)

        if train_loss.item() < min_train_loss:
            min_train_loss = train_loss.item()
            train_loss_decorated = "\x1b[35;1m%12.12s\x1b[0m" % ("%f" % train_loss.item())
        else:
            train_loss_decorated = "%12.12s" % ("%f" % train_loss.item())
    
        forward_decorated = "%12.12s" % utils.us_to_human_friendly(forward_end-forward_start)
        backward_decorated = "%12.12s" % utils.us_to_human_friendly(backward_end-backward_start)
        total_decorated = "\x1b[33;3m%12.12s\x1b[0m" % utils.us_to_human_friendly(total_end-total_start)

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
        
        print("%s %s %s %s %s %s" % (train_batch_decorated, train_loss_decorated, val_loss_decorated, forward_decorated, backward_decorated, total_decorated))
        with open(log_path,"a") as file:
            file.write("%d %f %f%s\n" % (train_batch, train_loss.item(), val_loss, suffix))

    scaler.step(optimizer)
    scaler.update()
