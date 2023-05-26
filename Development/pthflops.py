import torch
from torch import nn
from pthflops import count_ops

class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        self.conv1 = nn.Conv2d(5, 5, 1, 1, 0)
        # ... other layers present inside will also be ignored

    def forward(self, x):
        return self.conv1(x)

# Create a network and a corresponding input
inp = torch.rand(1,5,7,7)
net = nn.Sequential(
    nn.Conv2d(5, 5, 1, 1, 0),
    nn.ReLU(inplace=True),
    CustomLayer()
)

# Count the number of FLOPs
count_ops(net, inp, ignore_layers=['CustomLayer'])