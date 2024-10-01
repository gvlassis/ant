from math import sqrt
import torch
import os
import warnings
# More compact and discreet
def get_formatwarning(message, category, filename, lineno, line=None):
    return f"\x1b[90;3m[{category.__name__}] {os.path.basename(filename)} ({lineno}L): {message}\x1b[0m\n"
warnings.formatwarning = get_formatwarning

PARAMETRIZATIONS=["sp", "ntk", "mup", "mf"]
OPTIMIZERS=["sgd","adam"]

def lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout):
    if parametrization == "sp":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1
        elif layer == "output":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1
            C2_adam = 1

    elif parametrization == "ntk":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)
        elif layer == "output":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = sqrt(fanin0)
            C2_adam = 1 / sqrt(fanin)

    elif parametrization == "mup":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1 / fanout0
            C2_sgd = fanout
            C1_adam = 1
            C2_adam = 1
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = fanin0
            C2_adam = 1 / fanin
        elif layer == "output":
            B1 = sqrt(fanin0)
            B2 = 1 / fanin
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = fanin0
            C2_adam = 1 / fanin

    elif parametrization == "mf":
        if layer == "input":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1 / sqrt(fanin0)
            C2_adam = sqrt(fanin)
        elif layer == "hidden":
            B1 = 1
            B2 = 1 / sqrt(fanin)
            C1_sgd = 1
            C2_sgd = 1
            C1_adam = 1 / sqrt(fanin0)
            C2_adam = sqrt(fanin)
        elif layer == "output":
            B1 = sqrt(fanin0)
            B2 = 1 / fanin
            C1_sgd = fanin0
            C2_sgd = 1 / fanin
            C1_adam = 1
            C2_adam = 1

    return B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam

def test_table1(fanin0, fanin, fanout0, fanout):
    print("\x1b[1m%16.16s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("parametrization", "layer", "B1", "B2", "C1_sgd", "C2_sgd", "C1_adam", "C2_adam"))
    for parametrization in ["sp", "ntk", "mup", "mf"]:
        for layer in ["input", "hidden", "output"]:
            B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam = lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout)

            B1_decorated = "%8.8s" % ("%f" % B1)
            B2_decorated = "%8.8s" % ("%f" % B2)
            C1_sgd_decorated = "%8.8s" % ("%f" % C1_sgd)
            C2_sgd_decorated = "%8.8s" % ("%f" % C2_sgd)
            C1_adam_decorated = "%8.8s" % ("%f" % C1_adam)
            C2_adam_decorated = "%8.8s" % ("%f" % C2_adam)

            print("%16.16s %8.8s %s %s %s %s %s %s" % (parametrization, layer, B1_decorated, B2_decorated, C1_sgd_decorated, C2_sgd_decorated, C1_adam_decorated, C2_adam_decorated))
        print("━"*80)

def lookup_table2(fanin, parameter_type, c, k):
    if fanin == 1:
        if parameter_type == "Linear/Conv.bias":
            μ = 0
            B0 = 0
            C0 = k
        elif parameter_type == "LayerNorm.weight":
            μ = 1
            B0 = 0
            C0 = k
        elif parameter_type == "LayerNorm.bias":
            μ = 0
            B0 = 0
            C0 = k
        elif parameter_type == "class":
            μ = 0
            B0 = 0.02
            C0 = k

    elif fanin > 1:
        if parameter_type == "Linear/Conv.weight":
            μ = 0
            B0 = c
            C0 = k
        elif parameter_type == "emb/pos":
            μ = 0
            B0 = 0.02 * sqrt(fanin)
            C0 = k

    return μ, B0, C0

def test_table2(c, k):
    print("\x1b[1m%12.12s %18.18s %8.8s %8.8s %8.8s\x1b[0m" % ("fanin", "parameter_type", "μ", "B0", "C0"))

    μ, B0, C0 = lookup_table2(1, "Linear/Conv.bias", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=1", "Linear/Conv.bias", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(1, "LayerNorm.weight", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=1", "LayerNorm.weight", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(1, "LayerNorm.bias", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=1", "LayerNorm.bias", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(1, "class", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=1", "class", "%f" % μ, "%f" % B0, "%f" % C0))
    
    print("━"*60)

    μ, B0, C0 = lookup_table2(4, "Linear/Conv.weight", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=4", "Linear/Conv.weight", "%f" % μ, "%f" % B0, "%f" % C0))
    μ, B0, C0 = lookup_table2(4, "emb/pos", c, k)
    print("%12.12s %18.18s %8.8s %8.8s %8.8s" % ("fanin=4", "emb", "%f" % μ, "%f" % B0, "%f" % C0))

    print("━"*60)

def get_fan(parameter, suffix, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="weight":
        fanin = parameter.shape[1]
        fanout = parameter.shape[0]
    elif isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="bias":
        fanin = 1
        fanout = len(parameter)
    elif isinstance(parent, torch.nn.LayerNorm) and (suffix=="weight" or suffix=="bias"):
        fanin = 1
        fanout = len(parameter)
    else:
        # class
        if parameter.ndim == 1:
            fanin = 1
            fanout = len(parameter)
        # emb, pos
        elif parameter.ndim == 2:
            fanin = parameter.shape[0]
            fanout = parameter.shape[1]

    return fanin, fanout

# model: base
# model_: A scaled (up or down) version of model
def get_layers(model, model_, warning=True):
    layers = {}

    for parameter_name, parameter in model.named_parameters():
        parameter_ = model_.get_parameter(parameter_name)

        parent_name, _, suffix = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        fanin, fanout = get_fan(parameter, suffix, parent)
        fanin_, fanout_ = get_fan(parameter_, suffix, parent)
        
        if fanin == fanin_ and fanout != fanout_:
            layers[parameter_name] = "input"
        elif fanin != fanin_ and fanout != fanout_:
            layers[parameter_name] = "hidden"
        elif fanin != fanin_ and fanout == fanout_:
            layers[parameter_name] = "output"
        else:
            if warning: warnings.warn(f"{parameter_name} is not \"input\", \"hidden\" or \"output\". This means that you are either not scaling the WHOLE model_ (in which case parametrizations do not work), or that {parameter_name} is an output bias, and hence an \"input\" layer. We are going to assume tha latter.", UserWarning)
            layers[parameter_name] = "input"

    return layers

def get_parameter_type(parameter, suffix, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="weight":
        parameter_type = "Linear/Conv.weight"
    elif isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and suffix=="bias":
        parameter_type = "Linear/Conv.bias"
    elif isinstance(parent, torch.nn.LayerNorm) and suffix=="weight":
        parameter_type = "LayerNorm.weight"
    elif isinstance(parent, torch.nn.LayerNorm) and suffix=="bias":
        parameter_type = "LayerNorm.bias"
    else:
        # class
        if parameter.ndim == 1:
            parameter_type = "LayerNorm.bias"
        # emb, pos
        elif parameter.ndim == 2:
            parameter_type = "emb/pos"

    return parameter_type

# model0: proxy
# model: target
# model_: A scaled (up or down) version of model0
def parametrize(model0, model, model_, parametrization="sp", c=0.5, k=1e-3, optimizer="sgd", momentum=0, nesterov=False, betas=(0.9, 0.95), weight_decay=0, test=False, warning=True):
    layers = get_layers(model0, model_, warning)
    
    params = []
    
    if test: print("\x1b[1m%36.36s %8.8s %20.20s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % ("parameter_name", "layer", "parameter_type", "fanin0", "fanin", "fanout0", "fanout", "mean", "B0", "B1", "B2", "std", "C0", "C1", "C2", "lr"))
    for parameter_name, parameter in model.named_parameters():
        parameter0 = model0.get_parameter(parameter_name)

        parent_name, _, suffix = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        layer = layers[parameter_name]

        fanin0, fanout0 = get_fan(parameter0, suffix, parent)
        fanin, fanout = get_fan(parameter, suffix, parent)

        B1, B2, C1_sgd, C2_sgd, C1_adam, C2_adam = lookup_table1(parametrization, layer, fanin0, fanin, fanout0, fanout)
        if optimizer == "sgd":
            C1 = C1_sgd
            C2 = C2_sgd
        elif optimizer == "adam":
            C1 = C1_adam
            C2 = C2_adam
        
        parameter_type = get_parameter_type(parameter, suffix, parent)

        μ, B0, C0 = lookup_table2(fanin, parameter_type, c, k)

        mean = μ
        std = B0*B1*B2
        torch.nn.init.normal_(parameter, mean, std)

        lr = C0*C1*C2
        params.append({"params": parameter, "lr": lr})

        if test:
            print("%36.36s %8.8s %20.20s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s %8.8s\x1b[0m" % (parameter_name, layer, parameter_type, fanin0, fanin, fanout0, fanout, mean, "%f" % B0, "%f" % B1, "%f" % B2, "%f" % std, "%f" % C0, "%f" % C1, "%f" % C2, "%f" % lr))

    if optimizer=="sgd":
        # fused=True is negligibly faster
        return torch.optim.SGD(params, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, fused=True)
    elif optimizer=="adam":
        # fused=True is negligibly faster
        return torch.optim.AdamW(params, betas=betas, weight_decay=weight_decay, fused=True)
