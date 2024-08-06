import torch
from . import mlp
from . import vgg
from . import vit
from . import transformer
from . import mup

FAMILIES=["mlp", "vgg", "vit", "transformer"]

def get_model_optimizer(vocab_size, family, parametrization, ζ, c, k, weight_decay, device):
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
            model = transformer.Transformer(vocab_size=vocab_size, d=32*ζ).to(device)
        elif parametrization=="mup":
            proxy = transformer.Transformer(vocab_size=vocab_size, d=32, scale=1/32).to(device)
            target = transformer.Transformer(vocab_size=vocab_size, d=32*ζ, scale=1/(32*ζ)).to(device)

    if parametrization=="sp":
        mup.init_sp(model, c)
        optimizer = torch.optim.AdamW(model.parameters(), lr=k, weight_decay=weight_decay)
        return model, optimizer
    elif parametrization=="mup":
        mup.init_mup(proxy, target, c)
        optimizer = mup.AdamW(proxy, target, k, weight_decay)
        return target, optimizer
