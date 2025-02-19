import torch
from . import mlp

class BlockA(torch.nn.Module):
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

class StageA(torch.nn.Module):
    def __init__(self, res, num, in_channels, out_channels0):
        super().__init__()
        
        self.num = num
        self.in_channels = in_channels
        self.out_channels0 = out_channels0

        self.blocks = [BlockA(in_channels, out_channels0)]
        for i in range(num-1):
            block = BlockA(self.blocks[-1].out_channels, self.blocks[-1].out_channels*2)
            self.blocks.append(block)
        self.blocks = torch.nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)

class BlockB(torch.nn.Module):
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

class StageB(torch.nn.Module):
    def __init__(self, num, in_channels, out_channels0):
        super().__init__()
        
        self.num = num
        self.in_channels = in_channels
        self.out_channels0 = out_channels0

        self.blocks = [BlockB(in_channels, out_channels0)]
        for i in range(num-1):
            block = BlockB(self.blocks[-1].out_channels, self.blocks[-1].out_channels*2)
            self.blocks.append(block)
        self.blocks = torch.nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)

class VGG(torch.nn.Module):
    def __init__(self, res=32, num1=2, num2=2, out_channels0=4, dropout=0.5, classes=10):
        super().__init__()
        
        self.res = res
        self.num1 = num1
        self.num2 = num2
        self.num = num1+num2
        self.out_channels0 = out_channels0
        self.dropout = dropout
        self.classes=10
        
        self.stage1 = StageA(res, num1, 3, out_channels0)
        res = res//2**num1

        self.stage2 = StageB(num2, self.stage1.blocks[-1].out_channels, self.stage1.blocks[-1].out_channels*2)
        res = res//2**num2

        self.mlp = mlp.MLP3L(res*res*self.stage2.blocks[-1].out_channels, 5*out_channels0, 5*out_channels0, classes, dropout)

    def forward(self, x):
        x1 = self.stage1(x)

        x2 = self.stage2(x1)

        x2 = x2.flatten(start_dim=-3, end_dim=-1)
        
        y = self.mlp(x2)

        return y
