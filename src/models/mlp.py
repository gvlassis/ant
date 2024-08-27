import torch
import huggingface_hub

class MLP2L(torch.nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, d0, d1, d2, dropout=0):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.dropout=dropout

        self.l1 = torch.nn.Linear(d0, d1)
        self.l2 = torch.nn.Linear(d1, d2)

    def forward(self, x):
        z1 = self.l1(x)
        a1 = torch.nn.functional.relu(z1)
        a1 = torch.nn.functional.dropout(a1, p=self.dropout, training=self.training)

        y = self.l2(a1)

        return y

class MLP3L(torch.nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, d0, d1, d2, d3, dropout=0):
        super().__init__()

        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.dropout=dropout

        self.l1 = torch.nn.Linear(d0, d1)
        self.l2 = torch.nn.Linear(d1, d2)
        self.l3 = torch.nn.Linear(d2, d3)

    def forward(self, x):
        z1 = self.l1(x)
        a1 = torch.nn.functional.relu(z1)
        a1 = torch.nn.functional.dropout(a1, p=self.dropout, training=self.training)

        z2 = self.l2(a1)
        a2 = torch.nn.functional.relu(z2)
        a2 = torch.nn.functional.dropout(a2, p=self.dropout, training=self.training)

        y = self.l3(a2)

        return y
