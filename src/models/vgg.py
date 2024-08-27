import torch
import huggingface_hub
from . import mlp

class ConvStageA(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv(x)

        x = torch.nn.functional.relu(x)
        
        x = torch.nn.functional.max_pool2d(x , kernel_size=2, stride=2, padding=0)

        return x

class ConvStageB(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)

        x = torch.nn.functional.relu(x)

        x = self.conv2(x)

        x = torch.nn.functional.relu(x)

        x = torch.nn.functional.max_pool2d(x , kernel_size=2, stride=2, padding=0)

        return x

class VGG(torch.nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, res=32, num_a=2, num_b=2, out_channels0=4, dropout=0.5, classes=10):
        super().__init__()
        
        self.res = res
        self.num_a = num_a
        self.num_b = num_b
        self.out_channels0 = out_channels0
        self.dropout = dropout
        self.classes=10
        
        self.conv_stages_a = [ConvStageA(3, out_channels0)]
        res//=2
        for i in range(num_a-1):
            conv_stage = ConvStageA(self.conv_stages_a[i].out_channels, 2*self.conv_stages_a[i].out_channels)
            self.conv_stages_a.append(conv_stage)
            res//=2
        self.conv_stages_a = torch.nn.Sequential(*self.conv_stages_a)

        self.conv_stages_b = [ConvStageB(self.conv_stages_a[-1].out_channels, 2*self.conv_stages_a[-1].out_channels)]
        res//=2
        for i in range(num_b-1):
            conv_stage = ConvStageB(self.conv_stages_b[i].out_channels, 2*self.conv_stages_b[i].out_channels)
            self.conv_stages_b.append(conv_stage)
            res//=2
        self.conv_stages_b = torch.nn.Sequential(*self.conv_stages_b)
        
        self.mlp = mlp.MLP3L(res*res*self.conv_stages_b[-1].out_channels, 5*out_channels0, 5*out_channels0, classes, dropout)

    def forward(self, x):
        xa = self.conv_stages_a(x)

        xb = self.conv_stages_b(xa)
        xb = torch.flatten(xb, start_dim=-3, end_dim=-1)
        
        y = self.mlp(xb)

        return y
