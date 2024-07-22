import torch

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
    elif kind=="class_emb":
        fanin = 1
        fanout = len(parameter)
    elif isinstance(parent, torch.nn.Embedding):
        fanin = parameter.shape[0]
        fanout = parameter.shape[1]

    return fanin, fanout

def get_c(kind, parent, c, target_fanin):
    if isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)) and kind=="bias":
        c = 0
    elif isinstance(parent, torch.nn.LayerNorm) and (kind=="weight" or kind=="bias"):
        c = 0
    elif kind=="class_emb":
        c = 0.02*(target_fanin)**0.5
    elif isinstance(parent, torch.nn.Embedding):
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
    elif kind=="class_emb":
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
            torch.nn.init.normal_(parameter, mean=μ , std=get_c(kind, parent, c, fanin))
        elif fanin>1:
            torch.nn.init.normal_(parameter, mean=0 , std=get_c(kind, parent, c, fanin)/(fanin)**0.5)

def init_mup(proxy, target, c=1):
    for parameter_name, target_parameter in target.named_parameters():
        proxy_parameter = proxy.get_parameter(parameter_name)
        parent_name,_,kind = parameter_name.rpartition(".")
        target_parent = target.get_submodule(parent_name)

        proxy_fanin, proxy_fanout = get_fanin_fanout(proxy_parameter, kind, target_parent)
        target_fanin, target_fanout = get_fanin_fanout(target_parameter, kind, target_parent)
        
        if target_fanin==1:
            μ = get_μ(kind, target_parent)
            torch.nn.init.normal_(target_parameter, mean=μ , std=get_c(kind, target_parent, c, target_fanin))
        elif target_fanin>1:
            s, c_const = get_s_c_const(proxy_fanin, target_fanin, proxy_fanout, target_fanout)
            torch.nn.init.normal_(target_parameter, mean=0 , std=c_const*get_c(kind, target_parent, c, target_fanin)*s)

class AdamW(torch.optim.AdamW):
    def __init__(self, proxy, target, k=0.001, weight_decay=0.01):
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
