import torch
from . import mlp
from . import vgg
from . import vit
from . import transformer

FAMILIES=["mlp", "vgg", "vit", "transformer"]
PARAMETRIZATIONS=["sp", "mup"]

def get_fanin_fanout(parameter, kind, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and kind=="weight":
        fanin = parameter.shape[1]
        fanout = parameter.shape[0]
    elif isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and kind=="bias":
        fanin = 1
        fanout = len(parameter)
    elif isinstance(parent, torch.nn.LayerNorm) and (kind=="weight" or kind=="bias"):
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

def get_c(parameter, kind, parent, c, target_fanin):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and kind=="bias":
        c = 0
    elif isinstance(parent, torch.nn.LayerNorm) and (kind=="weight" or kind=="bias"):
        c = 0
    else:
        # class
        if parameter.ndim == 1:
            c = 0.02*(target_fanin)**0.5
        # emb, pos
        elif parameter.ndim == 2:
            c = 0.02*(target_fanin)**0.5

    return c

def get_inf(proxy, target):
    if target>proxy:
        inf = True
    else:
        inf = False

    return inf

def get_s_c_const(proxy_fanin, target_fanin, proxy_fanout, target_fanout):
    fanin_inf = get_inf(proxy_fanin, target_fanin)
    fanout_inf = get_inf(proxy_fanout, target_fanout)

    if (not fanin_inf):
        s = 1/(target_fanin)**0.5
        c_const = 1
    elif fanin_inf and fanout_inf:
        s = 1/(target_fanin)**0.5
        c_const = 1
    elif fanin_inf and (not fanout_inf):
        s = 1/target_fanin
        c_const = (proxy_fanin)**0.5
        
    return s, c_const

def get_μ(kind, parent):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and kind=="bias":
        μ = 0
    elif isinstance(parent, torch.nn.LayerNorm) and kind=="weight":
        μ = 1
    elif isinstance(parent, torch.nn.LayerNorm) and kind=="bias":
        μ = 0
    # class
    else:
        μ = 0

    return μ

def get_γ_γ_const(proxy_fanin, target_fanin, proxy_fanout, target_fanout):
    fanin_inf = get_inf(proxy_fanin, target_fanin)
    fanout_inf = get_inf(proxy_fanout, target_fanout)

    if (not fanin_inf):
        γ = 1
        γ_const = 1
    elif fanin_inf and fanout_inf:
        γ = 1/target_fanin
        γ_const = proxy_fanin
    elif fanin_inf and (not fanout_inf):
        γ = 1/target_fanin
        γ_const = proxy_fanin
        
    return γ, γ_const

def init_sp(model, c=1): 
    for parameter_name, parameter in model.named_parameters():
        parent_name,_,kind = parameter_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        
        fanin, _ = get_fanin_fanout(parameter, kind, parent)

        if fanin==1:
            μ = get_μ(kind, parent)
            torch.nn.init.normal_(parameter, mean=μ , std=get_c(parameter, kind, parent, c, fanin))
        elif fanin>1:
            torch.nn.init.normal_(parameter, mean=0 , std=get_c(parameter, kind, parent, c, fanin)/fanin**0.5)

def init_mup(proxy, target, c=1):
    for parameter_name, target_parameter in target.named_parameters():
        proxy_parameter = proxy.get_parameter(parameter_name)
        parent_name,_,kind = parameter_name.rpartition(".")
        target_parent = target.get_submodule(parent_name)

        proxy_fanin, proxy_fanout = get_fanin_fanout(proxy_parameter, kind, target_parent)
        target_fanin, target_fanout = get_fanin_fanout(target_parameter, kind, target_parent)
        
        if target_fanin==1:
            μ = get_μ(kind, target_parent)
            torch.nn.init.normal_(target_parameter, mean=μ , std=get_c(target_parameter, kind, target_parent, c, target_fanin))
        elif target_fanin>1:
            s, c_const = get_s_c_const(proxy_fanin, target_fanin, proxy_fanout, target_fanout)
            torch.nn.init.normal_(target_parameter, mean=0 , std=c_const*get_c(target_parameter, kind, target_parent, c, target_fanin)*s)

class AdamW_mup(torch.optim.AdamW):
    def __init__(self, proxy, target, k=0.001, weight_decay=0.0):
        params = []

        for parameter_name, target_parameter in target.named_parameters():
            proxy_parameter = proxy.get_parameter(parameter_name)
            parent_name,_,kind = parameter_name.rpartition(".")
            target_parent = target.get_submodule(parent_name)

            proxy_fanin, proxy_fanout = get_fanin_fanout(proxy_parameter, kind, target_parent)
            target_fanin, target_fanout = get_fanin_fanout(target_parameter, kind, target_parent)
            
            γ, γ_const = get_γ_γ_const(proxy_fanin, target_fanin, proxy_fanout, target_fanout)

            params.append({"params": target_parameter, "lr": k*γ_const*γ, "weight_decay": weight_decay})

        super().__init__(params)

def get_model_optimizer(vocab_size, family, parametrization, ζ, c, k, weight_decay, max_context, device):
    if family=="mlp":
        if parametrization=="sp":
            model = mlp.MLP3L(8, 16*ζ, 16*ζ, 1).to(device)
        elif parametrization=="mup":
            proxy = mlp.MLP3L(8, 16, 16, 1).to(device)
            target = mlp.MLP3L(8, 16*ζ, 16*ζ, 1).to(device)

    elif family=="vgg":
        if parametrization=="sp":
            model = vgg.VGG(out_channels0=4*ζ).to(device)
        elif parametrization=="mup":
            proxy = vgg.VGG(out_channels0=4).to(device)
            target = vgg.VGG(out_channels0=4*ζ).to(device)
    
    elif family=="vit":
        if parametrization=="sp":
            model = vit.ViT(d=32*ζ).to(device)
        elif parametrization=="mup":
            proxy = vit.ViT(d=32, scale=1/32).to(device)
            target = vit.ViT(d=32*ζ, scale=1/(32*ζ)).to(device)

    elif family=="transformer":
        if parametrization=="sp":
            model = transformer.Transformer(vocab_size=vocab_size, num_blocks=12, d=32*ζ, heads=8, scale=None, exp_factor=4, dropout=0, pos_type="sin", max_context=max_context, all_pos=False).to(device)
        elif parametrization=="mup":
            proxy = transformer.Transformer(vocab_size=vocab_size, num_blocks=12, d=32, heads=8, scale=1/32, exp_factor=4, dropout=0, pos_type="sin", max_context=max_context, all_pos=False).to(device)
            target = transformer.Transformer(vocab_size=vocab_size, num_blocks=12, d=32*ζ, heads=8, scale=1/(32*ζ), exp_factor=4, dropout=0, pos_type="sin", max_context=max_context, all_pos=False).to(device)

    if parametrization=="sp":
        init_sp(model, c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=k, weight_decay=weight_decay)
        return model, optimizer
    elif parametrization=="mup":
        init_mup(proxy, target, c)
        optimizer = AdamW_mup(proxy, target, k, weight_decay)
        return target, optimizer
