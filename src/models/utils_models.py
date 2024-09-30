import torch
from . import mlp
from . import vgg
from . import vit
from . import transformer
from . import parametrizations

FAMILIES=["mlp", "vgg", "vit", "transformer"]

def get_model_optimizer(vocab_size=50304, family="transformer", parametrization="sp", ζ=1, c=0.5, k=1e-3, betas=(0.9, 0.95), weight_decay=0, max_context=512, test_parametrization=False):
    if parametrization=="mup":
        scale_type = "1/d"
    else:
        scale_type = "1/sqrt(d)"
    
    if family=="mlp":
        model0 = mlp.MLP3L(8, 16, 16, 1)
        model = mlp.MLP3L(8, 16*ζ, 16*ζ, 1)
        model_ = mlp.MLP3L(8, 16*2, 16*2, 1)

    elif family=="vgg":
        model0 = vgg.VGG(out_channels0=4)
        model = vgg.VGG(out_channels0=4*ζ)
        model_ = vgg.VGG(out_channels0=4*2)

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
        model_ = vit.ViT(channels, max_res, patch_size, num_blocks, heads, 8, scale_type, exp_factor, dropout, pos_type, all_pos, norm_type, bias, act, l1_type, classes)

    elif family=="transformer":
        num_blocks = 12
        heads = 12
        exp_factor = 4
        dropout = 0
        pos_type = "learned"
        all_pos = False
        norm_type = "layer"
        bias = False
        act = torch.nn.GELU()
        l1_type = "linear"
        model0 = transformer.Transformer(vocab_size, num_blocks, heads, 4, scale_type, exp_factor, dropout, pos_type, max_context, all_pos, norm_type, bias, act, l1_type)
        model = transformer.Transformer(vocab_size, num_blocks, heads, 4*ζ, scale_type, exp_factor, dropout, pos_type, max_context, all_pos, norm_type, bias, act, l1_type)
        model_ = transformer.Transformer(vocab_size, num_blocks, heads, 4*2, scale_type, exp_factor, dropout, pos_type, max_context, all_pos, norm_type, bias, act, l1_type)

    optimizer = parametrizations.parametrize(model0, model, model_, parametrization, "adam", c, k, betas=betas, weight_decay=weight_decay, test=test_parametrization, warning=True)

    return model, optimizer

def get_train_stats_header(model):
    train_stats_header = ""

    for name, _ in model.named_parameters():
        train_stats_header += f"{name}.grad_mean {name}.grad_top {name}.grad_bot {name}.grad_max {name}.data_mean {name}.data_top {name}.data_bot {name}.data_max "

    # Remove last space
    train_stats_header = train_stats_header[:-1]

    return train_stats_header

def get_stats(tensor):
    mean = tensor.mean().item()

    # https://github.com/pytorch/pytorch/issues/29372
    std = 0 if tensor.numel()==1 else tensor.std().item()

    top = mean+std
    bot = mean-std
    _max = tensor.max().item()

    return mean, top, bot, _max

def get_train_stats(model):
    train_stats = ""

    for parameter in model.parameters():
        grad_mean, grad_top, grad_bot, grad_max = get_stats(parameter.grad.abs())
        
        data_mean, data_top, data_bot, data_max = get_stats(parameter.data.abs())

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
