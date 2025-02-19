import torch
import warnings
from . import mlp
from . import vgg
from . import resnet
from . import vit
from . import transformer
from . import ngpt
from . import parametrizations

FAMILIES=["mlp", "mlp_image", "vgg", "resnet", "vit", "transformer", "ngpt"]

def get_model_optimizer(vocab_size, family, parametrization, ζ, scale_type, pos_type, c_input, c_hidden, c_output, k_input, k_hidden, k_output, optimizer, momentum, nesterov, betas, weight_decay, max_context, test_parametrization, warning, backend):
    if warning and ((parametrization != "mup" and scale_type == "1/d") or (parametrization == "mup" and scale_type == "1/sqrt(d)")): warnings.warn(f"You use {scale_type} attention scaling even though the parametrization is {parametrization}", UserWarning)
    
    if family=="mlp":
        model0 = mlp.MLP3L(8, 16, 16, 1)
        model = mlp.MLP3L(8, 16*ζ, 16*ζ, 1)
        model_ = mlp.MLP3L(8, 16*2, 16*2, 1)

    elif family=="mlp_image":
        model0 = mlp.MLP3L_image(d1=16, d2=16)
        model = mlp.MLP3L_image(d1=16*ζ, d2=16*ζ)
        model_ = mlp.MLP3L_image(d1=16*2, d2=16*2)

    elif family=="vgg":
        model0 = vgg.VGG(out_channels0=4)
        model = vgg.VGG(out_channels0=4*ζ)
        model_ = vgg.VGG(out_channels0=4*2)

    elif family=="resnet":
        model0 = resnet.ResNet(out_channels0=4)
        model = resnet.ResNet(out_channels0=4*ζ)
        model_ = resnet.ResNet(out_channels0=4*2)

    elif family=="vit":
        channels = 3
        max_res = 32
        patch_size = 4
        num_blocks = 6
        heads = 8
        exp_factor = 1
        dropout = 0.1
        pos_type = "sin"
        all_pos = False
        norm_type = "layer"
        bias = False
        act = torch.nn.GELU()
        l1_type = "linear"
        classes = 10
        model0 = vit.ViT(channels, max_res, patch_size, num_blocks, heads, 4, scale_type, exp_factor, dropout, pos_type, all_pos, norm_type, bias, act, l1_type, classes)
        model = vit.ViT(channels, max_res, patch_size, num_blocks, heads, 4*ζ, scale_type, exp_factor, dropout, pos_type, all_pos, norm_type, bias, act, l1_type, classes)
        model_ = vit.ViT(channels, max_res, patch_size, num_blocks, heads, 4*2, scale_type, exp_factor, dropout, pos_type, all_pos, norm_type, bias, act, l1_type, classes)

    elif family=="transformer":
        num_blocks = 12
        heads = 12
        d_head0 = 4
        model0 = transformer.Transformer(vocab_size, num_blocks, heads, d_head0, scale_type, backend=backend, pos_type=pos_type, max_context=max_context)
        model = transformer.Transformer(vocab_size, num_blocks, heads, ζ*d_head0, scale_type, backend=backend, pos_type=pos_type, max_context=max_context)
        model_ = transformer.Transformer(vocab_size, num_blocks, heads, 2*d_head0, scale_type, backend=backend, pos_type=pos_type, max_context=max_context)

    elif family=="ngpt":
        num_blocks = 12
        heads = 12
        d_head0 = 4
        model0 = None
        model = ngpt.nGPT(vocab_size, num_blocks, heads=heads, d_head=ζ*d_head0, backend=backend, std=c_input, test=test_parametrization)
        model_ = None

    optimizer = parametrizations.parametrize(model0, model, model_, parametrization, c_input, c_hidden, c_output, k_input, k_hidden, k_output, optimizer, momentum, nesterov, betas, weight_decay, test_parametrization, warning)

    return model, optimizer

def weight_norm(model):
    for parameter_name, parameter in model.named_parameters():
        parent_name, _, suffix = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        if parent_name.endswith(".lo") and suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=0)
        elif parent_name.endswith(".l2") and suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=0)
        elif suffix=="weight":
            parameter.data = ngpt.sphere_norm(parameter.data, dim=1)

    return model

def get_train_stats_header(model):
    train_stats_header = ""

    for name, _ in model.named_parameters():
        train_stats_header += f"{name}.grad_mean {name}.grad_top {name}.grad_bot {name}.grad_max {name}.data_mean {name}.data_top {name}.data_bot {name}.data_max "

    # Remove last space
    train_stats_header = train_stats_header[:-1]

    return train_stats_header

def get_stats_abs(tensor):
    mean = tensor.mean().item()

    # https://github.com/pytorch/pytorch/issues/29372
    std = 0 if tensor.numel()==1 else tensor.std().item()

    top = mean+std
    # Absolute value cannot be negative
    bot = max(mean-std,0)
    _max = tensor.max().item()

    return mean, top, bot, _max

def get_train_stats(model):
    train_stats = ""

    for parameter in model.parameters():
        grad_mean, grad_top, grad_bot, grad_max = get_stats_abs(parameter.grad.abs())
        
        data_mean, data_top, data_bot, data_max = get_stats_abs(parameter.data.abs())

        train_stats += f"{grad_mean} {grad_top} {grad_bot} {grad_max} {data_mean} {data_top} {data_bot} {data_max} "
    
    # Remove last space
    train_stats = train_stats[:-1]

    return train_stats

def get_batch_stats(family, model, batch_Y_):
    out = batch_Y_.abs().mean().item()

    if family == "mlp":
        grad_mean = model.l2.weight.grad.abs().mean().item()
        data_mean = model.l2.weight.data.abs().mean().item()

    return out, grad_mean, data_mean
