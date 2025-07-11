import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform


class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = int(2**bits)


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x


class UniformQuantizer(BaseQuantizer):
    def forward(self, x):
        if not self.training:
            return x
        scale, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
        scale = scale + 1e-8
        step = scale * 2 / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()


class ClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = (
                (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
            ).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None or self.matrix.shape[0] != x.shape[-1]:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None or self.matrix.shape[0] != x.shape[-1]:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardTrustQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True)
        self.matrix = None
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        if self.matrix is None or self.matrix.shape[0] != x.shape[-1]:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()

QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    "HadamardClipQuantizer": HadamardClipQuantizer,

    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
}


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

        if hasattr(weight_quantizer, "initialize_blocks"):
            weight_quantizer.initialize_blocks(in_features)


    def forward(self, x):
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)
