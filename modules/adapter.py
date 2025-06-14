import torch.nn as nn
import torch

class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck)
        self.nonlin = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, hidden_size)

    def forward(self, x):
        return self.up_proj(self.nonlin(self.down_proj(x)))
