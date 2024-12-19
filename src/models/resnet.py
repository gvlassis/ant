import torch

class BlockA(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = torch.nn.functional.relu(y)

        y = self.conv2(y)
        y = self.norm2(y)
        
        z = y+x
        z = torch.nn.functional.relu(z)

        return z

class StageA(torch.nn.Module):
    def __init__(self, num, in_channels):
        super().__init__()
        
        self.num = num
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.blocks = torch.nn.Sequential(*[BlockA(in_channels) for _ in range(num)])
    
    def forward(self, x):
        return self.blocks(x)

class BlockDownsamplingA(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = 2*in_channels
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(2*in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=2*in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(2*in_channels)
        self.proj = torch.nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=1, stride=2, padding=0)
        self.norm = torch.nn.BatchNorm2d(2*in_channels)
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = torch.nn.functional.relu(y)

        y = self.conv2(y)
        y = self.norm2(y)

        # Matching the dimensions of y and x
        z = y+self.norm(self.proj(x))
        z = torch.nn.functional.relu(z)

        return z

class StageDownsamplingA(torch.nn.Module):
    def __init__(self, num, in_channels):
        super().__init__()
        
        self.num = num
        self.in_channels = in_channels
        self.out_channels = 2*in_channels
        
        blocks = [BlockDownsamplingA(in_channels)]
        for i in range(num-1):
            block = BlockA(2*in_channels)
            blocks.append(block)
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
        
class ResNet(torch.nn.Module):
    def __init__(self, res=64, num2=2, num3=2, num4=2, num5=2, out_channels0=4, dropout=0.5, classes=200):
        super().__init__()

        self.res = res
        self.num2 = num2
        self.num3 = num3
        self.num4 = num4
        self.num5 = num5
        self.num = num2+num3+num4+num5
        self.out_channels0 = out_channels0
        self.dropout = dropout
        self.classes = classes

        self.stage1 = torch.nn.Conv2d(in_channels=3, out_channels=out_channels0, kernel_size=7, stride=2, padding=3)
        self.stage2 = StageA(num2, out_channels0)
        self.stage3 = StageDownsamplingA(num3, out_channels0)
        self.stage4 = StageDownsamplingA(num4, out_channels0*2)
        self.stage5 = StageDownsamplingA(num5, out_channels0*4)

        self.linear = torch.nn.Linear(out_channels0*8, classes)
        
    def forward(self, x):
        y1 = self.stage1(x)
        y1 = torch.nn.functional.max_pool2d(y1, kernel_size=3, stride=2, padding=1)

        y2 = self.stage2(y1)

        y3 = self.stage3(y2)

        y4 = self.stage4(y3)

        y5 = self.stage5(y4)

        z = y5.mean(dim=(-2,-1))
        z = self.linear(z)

        return z
