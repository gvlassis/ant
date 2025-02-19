import torch
import os
import argparse
import data.utils_data
import models.utils_models
import models.transformer
import utils
import fvcore.nn
import torchview
import logging
torch._logging.set_logs(all=logging.ERROR)
import contextlib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("SUBPATH", help="Training log will be saved in SUBPATH.dat", type=os.path.abspath)
parser.add_argument("--save_model", help="Save the model with the min validation loss in SUBPATH.pt", type=utils.str_to_bool, default=False)
parser.add_argument("--info", help="Print information about the model", type=utils.str_to_bool, default=False)
parser.add_argument("--graph", help="Draw computational graph in SUBPATH.pdf", type=utils.str_to_bool, default=False)
parser.add_argument("--test_parametrization", help="Print parametrization information", type=utils.str_to_bool, default=False)
parser.add_argument("--print_schedule", help="Print learning rate schedule", type=utils.str_to_bool, default=False)
parser.add_argument("--warning", type=utils.str_to_bool, default=True)
parser.add_argument("--verbose", help="If True, print a pretty log in the stdout. If False, only print the final val_loss. Useful for piping.", type=utils.str_to_bool, default=True)
parser.add_argument("--extra_freq", help="Every how many batches to perform the extra evaluations", type=int, default=utils.INF)
parser.add_argument("--rmse", help="As an extra, evaluate the train and validation Root Mean Squared Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--nrmse", help="As an extra, evaluate the train and validation Normalized Root Mean Squared Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--mae", help="As an extra, evaluate the train and validation Mean Absolute Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--nmae", help="As an extra, evaluate the train and validation Normalized Mean Absolute Errors", type=utils.str_to_bool, default=False)
parser.add_argument("--r2", help="As an extra, evaluate the train and validation Coefficients of Determination", type=utils.str_to_bool, default=False)
parser.add_argument("--acc", help="As an extra, evaluate the train and validation accuracies", type=utils.str_to_bool, default=False)
parser.add_argument("--ppl", help="As an extra, evaluate the train and validation perplexities", type=utils.str_to_bool, default=False)

parser.add_argument("--dataset", choices=data.utils_data.DATASETS, default="openwebtext")
parser.add_argument("--vocab_size", type=int, default=50304)
parser.add_argument("--family", help="Model architecture", choices=models.utils_models.FAMILIES, default="transformer")
parser.add_argument("--parametrization", help="(a)bc parametrization, as defined in Tensor Programs IV (https://arxiv.org/abs/2011.14522). np (No Parametrization) means that the initialization is handled internally by the model.", choices=models.parametrizations.PARAMETRIZATIONS, default="mup")
parser.add_argument("--ζ", help="Width scaling factor", type=int, default=16)
parser.add_argument("--scale_type", help="Scaling factor applied prior to softmax", choices=models.transformer.SCALE_TYPES, default="1/sqrt(d)")
parser.add_argument("--pos_type", help="Positional embeddings", choices=models.transformer.POS_TYPES, default="learned")

parser.add_argument("--decoupling", help="Decouples c/k_input, c/k_hidden and c/k_output. If coupled, they are controlled by c/k_input.", type=utils.str_to_bool, default=True)
parser.add_argument("--c_input", type=float, default=0.02)
parser.add_argument("--c_hidden", type=float, default=0.4)
parser.add_argument("--c_output", type=float, default=0.4)
parser.add_argument("--optimizer", choices=models.parametrizations.OPTIMIZERS, default="adam")
parser.add_argument("--k_input", type=float, default=1e-3)
parser.add_argument("--k_hidden", type=float, default=6e-3)
parser.add_argument("--k_output", type=float, default=6e-3)
parser.add_argument("--scheduler", help="Learning rate schedule", choices=utils.SCHEDULERS, default="trapezoidal")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--nesterov", help="Use Nesterov momentum", type=utils.str_to_bool, default=False)
parser.add_argument("--β1", type=float, default=0.85)
parser.add_argument("--β2", type=float, default=0.95)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--label_smoothing", type=float, default=0)
parser.add_argument("--pre_norm", help="Normalize the weights on the unit hypersphere after initialization", type=utils.str_to_bool, default=False)
parser.add_argument("--post_norm", help="Normalize the weights on the unit hypersphere after each training step", type=utils.str_to_bool, default=False)

parser.add_argument("--batch_size", help="Total batch size, over all GPUs and accumulations, for one gradient update", type=int, default=1024)
parser.add_argument("--micro_batch_size", help="Batch size that fits in every GPU", type=int, default=32)
parser.add_argument("--context", type=int, default=1024)
parser.add_argument("--train_batches", help="The number of batches used during training", type=int, default=100_000)
parser.add_argument("--thresh", help="Artificially stop training when the validation loss first crosses this threshold", type=float, default=0)
parser.add_argument("--val_batches", help="The number of batches used during validation", type=int, default=100)
parser.add_argument("--update_freq", help="Every how many batches the train and the validation loss will be evaluated", type=int, default=500)

parser.add_argument("--model_device_index", help="CUDA device that stores the model", type=int, default=0)
parser.add_argument("--dataset_device_type", choices=["cpu", "cuda"], help="Device type that preloads the dataset", default="cpu")
parser.add_argument("--dtype", help="torch.dtype for Automatic Mixed Precision (AMP)", type=lambda x: getattr(torch, x), default="bfloat16")
parser.add_argument("--compile", help="Use torch.compile()", type=utils.str_to_bool, default=True)
parser.add_argument("--backend", help="Scaled Dot Product Attention (SDPA) backend", choices=models.transformer.BACKENDS, default="flash")
args=parser.parse_args()

torchelastic = os.getenv("TORCHELASTIC_RUN_ID") != None
if torchelastic:
    # If the backend is not provided, then both a gloo and nccl backend will be created
    torch.distributed.init_process_group(backend="nccl")
    
    # Get environment variables set by torchrun
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
    RANK = int(os.getenv("RANK"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
    
    master = RANK == 0
    
    model_device_index = LOCAL_RANK

    accumulation = args.batch_size//(WORLD_SIZE*args.micro_batch_size)
else:
    master = True

    model_device_index = args.model_device_index
    
    accumulation = args.batch_size//args.micro_batch_size

subpath_dir = os.path.dirname(args.SUBPATH)
if master: os.makedirs(subpath_dir, exist_ok=True)
log_path = args.SUBPATH+".dat"
model_path = args.SUBPATH+".pt"
graph_path = args.SUBPATH+".pdf"
extra_path = args.SUBPATH+"_extra.dat"

extra = False if args.extra_freq==utils.INF else True

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

model_device_type = "cuda"
model_device = f"{model_device_type}:{model_device_index}"
if args.dataset_device_type == "cpu":
    dataset_device = "cpu"
elif args.dataset_device_type == "cuda":
    dataset_device = model_device

if master and args.verbose: print("💾 Loading dataset")
train_iterator = data.utils_data.get_iterator(args.dataset, "train", dataset_device, args.micro_batch_size, args.context)
val_iterator = data.utils_data.get_iterator(args.dataset, "val", dataset_device, args.micro_batch_size, args.context)

if master and args.verbose: print("🧠 Initializing model")
model, optimizer = models.utils_models.get_model_optimizer(args.vocab_size, args.family, args.parametrization, args.ζ, args.scale_type, args.pos_type, c_input, c_hidden, c_output, k_input, k_hidden, k_output, args.optimizer, args.momentum, args.nesterov, (args.β1, args.β2), args.weight_decay, args.context, args.test_parametrization and master, args.warning and master, args.backend)
if args.pre_norm: model = models.utils_models.weight_norm(model)
model = model.to(model_device)
if args.info:
    batch_X, _ = next(train_iterator)
    # Not having batch dimension can cause problems (e.g. BatchNorm)
    X = batch_X[:1]
    input_data = data.utils_data.transform(args.dataset, X.to(model_device))
    if master: print(fvcore.nn.flop_count_table(fvcore.nn.FlopCountAnalysis(model, input_data), max_depth=3, show_param_shapes=False))
if args.graph:
    batch_X, _ = next(train_iterator)
    input_data = data.utils_data.transform(args.dataset, batch_X.to(model_device))
    if master: torchview.draw_graph(model, input_data=input_data, depth=1, expand_nested=True, graph_dir="TB", show_shapes=True).visual_graph.render(cleanup=True, format="pdf", outfile=graph_path)
# Get the parameters' names before DDP/compile
train_stats_header = models.utils_models.get_train_stats_header(model)

# float16 (not bfloat16) needs scaling
scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype==torch.float16))

if torchelastic: model = torch.nn.parallel.DistributedDataParallel(model)

# Compile after DDP
if args.compile:
    if master and args.verbose: print("⏳ Compiling")
    # mode=max-autotune gives NaN
    get_loss = torch.compile(data.utils_data.get_loss)
else:
    get_loss = data.utils_data.get_loss

scheduler = utils.get_scheduler(args.scheduler, optimizer, args.train_batches)
if args.print_schedule and master: utils.print_schedule(args.train_batches, scheduler)

if master:
    if args.verbose: print("\x1b[1m%12.12s %12.12s %12.12s %12.12s %18.18s\x1b[0m" % ("train_batch", "lr0", "train_loss", "val_loss", "train_batch_time"))
    with open(log_path,"w") as file:
        file.write(f"train_batch lr0 train_loss val_loss train_time {train_stats_header}\n")
    if extra:
        with open(extra_path,"w") as file:
            file.write("train_batch train_time")

            if args.rmse: file.write(" train_rmse val_rmse")
            if args.nrmse: file.write(" train_nrmse val_nrmse")
            if args.mae: file.write(" train_mae val_mae")
            if args.nmae: file.write(" train_nmae val_nmae")
            if args.r2: file.write(" train_r2 val_r2")
            if args.acc: file.write(" train_acc val_acc")

            file.write("\n")

train_time = 0
min_train_loss = utils.INF
min_val_loss = utils.INF
terminate = torch.tensor(False).to(model_device)
for train_batch in range(args.train_batches):
    if torchelastic: torch.distributed.broadcast(terminate, src=0)
    if terminate:
        if torchelastic: torch.distributed.destroy_process_group()
        exit(1)
    
    train_batch_start = utils.get_sync_time(model_device)
    
    model.train()
    train_loss = torch.tensor(0.0).to(model_device)
    for micro_train_batch in range(accumulation):
        batch_train_X, batch_train_Y = next(train_iterator)

        # Only sync gradients in the last micro_train_batch
        with (model.no_sync() if torchelastic and micro_train_batch<accumulation-1 else contextlib.nullcontext()):
            with torch.autocast(device_type=model_device_type, dtype=args.dtype):
                micro_train_loss = get_loss(args.dataset, model, batch_train_X.to(model_device), batch_train_Y.to(model_device), args.label_smoothing)[1]/accumulation
                train_loss += micro_train_loss.detach()
            scaler.scale(micro_train_loss).backward()
    
    train_batch_end = utils.get_sync_time(model_device)
    train_batch_time = train_batch_end-train_batch_start
    train_time += train_batch_time
    
    if torchelastic: torch.distributed.all_reduce(train_loss, op=torch.distributed.ReduceOp.AVG)
    train_loss = train_loss.item()

    if (train_batch % args.update_freq == 0 or train_batch == args.train_batches-1) and master:
        train_batch_decorated = "%12.12s" % train_batch

        lr0_decorated = "%12.12s" % ("%f" % scheduler.get_last_lr()[0])
        
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            train_loss_decorated = "\x1b[35;1m%12.12s\x1b[0m" % ("%f" % train_loss)
        else:
            train_loss_decorated = "%12.12s" % ("%f" % train_loss)

        val_loss = data.utils_data.approximate_loss(args.val_batches, val_iterator, args.dataset, model, args.dtype)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if args.save_model:
                if torchelastic:
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
            val_loss_decorated = "\x1b[36;1m%12.12s\x1b[0m" % ("%f" % val_loss)
        else:
            val_loss_decorated = "%12.12s" % ("%f" % val_loss)

        train_batch_time_decorated = "\x1b[33;3m%18.18s\x1b[0m" % utils.us_to_human_friendly(train_batch_time)

        train_stats = models.utils_models.get_train_stats(model)

        if args.verbose: print("%s %s %s %s %s" % (train_batch_decorated, lr0_decorated, train_loss_decorated, val_loss_decorated, train_batch_time_decorated))
        with open(log_path,"a") as file:
            file.write("%d %f %f %f %d %s\n" % (train_batch, scheduler.get_last_lr()[0], train_loss, val_loss, train_time, train_stats))

        if val_loss < args.thresh:
            terminate = torch.tensor(True).to(model_device)
    
    if extra and (train_batch % args.extra_freq == 0 or train_batch == args.train_batches-1) and master:
        print("━"*70)

        with open(extra_path,"a") as file:
            file.write("%d %d" % (train_batch, train_time))
        
        if args.rmse:
            train_rmse = data.utils_data.approximate_rmse(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_rmse", "%f" % train_rmse))
            val_rmse = data.utils_data.approximate_rmse(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_rmse", "%f" % val_rmse))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_rmse, val_rmse))

        if args.nrmse:
            train_nrmse = data.utils_data.approximate_nrmse(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_nrmse", "%f" % train_nrmse))
            val_nrmse = data.utils_data.approximate_nrmse(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_nrmse", "%f" % val_nrmse))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_nrmse, val_nrmse))

        if args.mae:
            train_mae = data.utils_data.approximate_mae(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_mae", "%f" % train_mae))
            val_mae = data.utils_data.approximate_mae(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_mae", "%f" % val_mae))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_mae, val_mae))

        if args.nmae:
            train_nmae = data.utils_data.approximate_nmae(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_nmae", "%f" % train_nmae))
            val_nmae = data.utils_data.approximate_nmae(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_nmae", "%f" % val_nmae))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_nmae, val_nmae))

        if args.r2:
            train_r2 = data.utils_data.approximate_r2(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_r2", "%f" % train_r2))
            val_r2 = data.utils_data.approximate_r2(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_r2", "%f" % val_r2))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_r2, val_r2))

        if args.acc:
            train_acc = data.utils_data.approximate_acc(args.val_batches, train_iterator,  args.dataset, model, args.dtype)*100
            print("%12.12s: %6.6s%%" % ("train_acc", "%.2f" % train_acc))
            val_acc = data.utils_data.approximate_acc(args.val_batches, val_iterator,  args.dataset, model, args.dtype)*100
            print("%12.12s: %6.6s%%" % ("val_acc", "%.2f" % val_acc))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_acc, val_acc))

        if args.ppl:
            train_ppl = data.utils_data.approximate_ppl(args.val_batches, train_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("train_ppl", "%f" % train_ppl))
            val_ppl = data.utils_data.approximate_ppl(args.val_batches, val_iterator,  args.dataset, model, args.dtype)
            print("%12.12s: %12.12s" % ("val_ppl", "%f" % val_ppl))

            with open(extra_path,"a") as file:
                file.write(" %f %f" % (train_ppl, val_ppl))

        print("━"*70)

        with open(extra_path,"a") as file:
            file.write("\n")

    scaler.step(optimizer)
    if args.post_norm: model = models.utils_models.weight_norm(model)
    optimizer.zero_grad()
    scaler.update()
    scheduler.step()

if master and (not args.verbose): print("%f" % val_loss, end="")

if torchelastic: torch.distributed.destroy_process_group()
exit(0)
