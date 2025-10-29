from __future__ import annotations
import torch
from torch import nn

class DWConv2d(nn.Module):
    def __init__(self, c: int, k: int = 3, s: int = 1, d: int = 1):
        super().__init__()
        p = (k//2)*d
        self.dw = nn.Conv2d(c, c, kernel_size=k, stride=s, padding=p, dilation=d, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class RefineCore(nn.Module):
    """Two-layer depthwise-separable conv refiner with residual."""
    def __init__(self, c_in: int, c_hidden: int):
        super().__init__()
        self.conv1 = DWConv2d(c_in, 3)
        self.channel_reduce = nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(c_hidden)
        self.act_reduce = nn.SiLU(inplace=True)
        self.conv2 = DWConv2d(c_hidden, 3)
        self.proj  = nn.Conv2d(c_in, c_hidden, 1, bias=False) if c_in!=c_hidden else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.channel_reduce(y)
        y = self.bn_reduce(y)
        y = self.act_reduce(y)
        y = self.conv2(y)
        return y + self.proj(x)
