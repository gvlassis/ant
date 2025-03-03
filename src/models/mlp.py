import torch

class GLU(torch.nn.Module):
    def __init__(self, d0, d1, bias=True, act=torch.nn.ReLU()):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.bias = bias
        self.act = act
        
        self.gate = torch.nn.Sequential(torch.nn.Linear(d0, d1, bias), act)

        self.proj = torch.nn.Linear(d0, d1, bias)

    def forward(self, x):
        y = self.gate(x) * self.proj(x)

        return y

class MLP2L(torch.nn.Module):
    def __init__(self, d0, d1, d2, bias=True, act=torch.nn.ReLU(), dropout=0, l1_type="linear"):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.bias = bias
        self.act = act
        self.dropout = dropout
        self.l1_type = l1_type

        if l1_type=="linear":
            self.l1 = torch.nn.Sequential(torch.nn.Linear(d0, d1, bias), act)
        elif l1_type=="glu":
            self.l1 = GLU(d0, d1, bias, act)
        self.l2 = torch.nn.Linear(d1, d2, bias)

    def forward(self, x):
        a1 = self.l1(x)
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
