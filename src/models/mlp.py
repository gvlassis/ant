import torch
from .quantization import QuantizedLinear, QUANTIZER_CLASSES


# QuEST quantizer set as the default
DEFAULT_QUANTIZER = "HalfHadamardTrustQuantizer"


def get_quantizers(quantization_bits):
    """
    16/32 bits → NoQuantizer; otherwise `HalfHadamardTrustQuantizer` as default.
    To plug in other quantizers, modify this function.
    """
    if isinstance(quantization_bits, tuple):
        weight_bits, activation_bits = quantization_bits
    else:
        weight_bits = activation_bits = quantization_bits

    def _select(bits):
        return (
            QUANTIZER_CLASSES["NoQuantizer"]()
            if bits in (16, 32)
            else QUANTIZER_CLASSES[DEFAULT_QUANTIZER](bits=bits)
        )

    weight_quantizer = _select(weight_bits)
    activation_quantizer = _select(activation_bits)

    return weight_quantizer, activation_quantizer

# Normalizes on the hypersphere along dim
# (s1*...*)s-1
def sphere_norm(X, dim=-1):
    return torch.nn.functional.normalize(X, dim=dim)

class SphereNorm(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()

        self.dim = dim

    def forward(self, X):
        Y = sphere_norm(X, dim=self.dim)

        return Y

def get_norm(enable, norm_type, d, bias):
    if enable:
        if norm_type=="layer":
            norm = torch.nn.LayerNorm(d, bias=bias)
        elif norm_type=="rms_learned":
            norm = torch.nn.RMSNorm(d, elementwise_affine=True)
        elif norm_type=="rms_const":
            norm = torch.nn.RMSNorm(d, elementwise_affine=False)
        elif norm_type=="sphere":
            norm = SphereNorm(dim=-1)
    else:
        norm = None

    return norm

class ReLU2(torch.nn.Module):
    def forward(self, x):
        y = torch.nn.functional.relu(x)**2

        return y

class Abs(torch.nn.Module):
    def forward(self, x):
        y = x.abs()

        return y

class GLU(torch.nn.Module):
    def __init__(self, d0, d1, bias=True, act=torch.nn.ReLU(), quantization_bits=16):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.bias = bias
        self.act = act
        
        # Get quantizers
        weight_quantizer, activation_quantizer = get_quantizers(quantization_bits)
        
        gate_linear = QuantizedLinear(d0, d1, bias=bias, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
        self.gate = torch.nn.Sequential(gate_linear, act)

        self.proj = QuantizedLinear(d0, d1, bias=bias, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)

    def forward(self, x):
        y = self.gate(x) * self.proj(x)

        return y

class MLP2L(torch.nn.Module):
    def __init__(self, d0, d1, d2, bias=True, act=torch.nn.ReLU(), dropout=0, l1_type="linear", norm_type="rms_learned", norm=False, quantization_bits=16):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.bias = bias
        self.act = act
        self.dropout = dropout
        self.l1_type = l1_type
        self.norm_type = norm_type

        # Get quantizers
        weight_quantizer, activation_quantizer = get_quantizers(quantization_bits)

        if l1_type=="linear":
            l1_linear = QuantizedLinear(d0, d1, bias=bias, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
            self.l1 = torch.nn.Sequential(l1_linear, act)
        elif l1_type=="glu":
            self.l1 = GLU(d0, d1, bias, act, quantization_bits)

        self.norm = get_norm(norm, norm_type, d1, bias)

        self.l2 = QuantizedLinear(d1, d2, bias=bias, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)

    def forward(self, x):
        a1 = self.l1(x)
        if self.norm: a1 = self.norm(a1)
        a1 = torch.nn.functional.dropout(a1, p=self.dropout, training=self.training)

        y = self.l2(a1)

        return y

class MLP3L(torch.nn.Module):
    def __init__(self, d0, d1, d2, d3, bias=True, act=torch.nn.ReLU(), dropout=0):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.bias = bias
        self.act = act
        self.dropout=dropout

        self.l1 = torch.nn.Linear(d0, d1, bias)
        self.l2 = torch.nn.Linear(d1, d2, bias)
        self.l3 = torch.nn.Linear(d2, d3, bias)

    def forward(self, x):
        z1 = self.l1(x)
        a1 = self.act(z1)
        a1 = torch.nn.functional.dropout(a1, p=self.dropout, training=self.training)

        z2 = self.l2(a1)
        a2 = self.act(z2)
        a2 = torch.nn.functional.dropout(a2, p=self.dropout, training=self.training)

        y = self.l3(a2)

        return y

class MLP3L_image(torch.nn.Module):
    def __init__(self, res=28, d1=16, d2=16, dropout=0, classes=10):
        super().__init__()

        self.res = res
        self.d1 = d1
        self.d2 = d2
        self.dropout = dropout
        self.classes = classes

        self.mlp = MLP3L(res*res, d1, d2, classes, dropout=dropout)

    def forward(self, x):
        x = x.flatten(start_dim=-3, end_dim=-1)

        y = self.mlp(x)

        return y
