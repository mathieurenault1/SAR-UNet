import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DS_parts import *
from models.unet_parts import *

class RecurrentConvDS_unit(nn.Module):
    def __init__(self,out_channels, t=2,kernels_per_layer=1):
        super(RecurrentConvDS_unit, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            ConvDS(out_channels, out_channels, kernels_per_layer=kernels_per_layer),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class ResDoubleConvDS(nn.Module):
    def __init__(self, in_channels, out_channels,kernels_per_layer=1):
        super(ResDoubleConvDS, self).__init__()
        self.doubleconv = nn.Sequential(
            DoubleConvDS(in_channels, out_channels,kernels_per_layer=kernels_per_layer)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x2 = self.doubleconv(x)
        return x1 + x2

class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDoubleConv, self).__init__()
        self.doubleconv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv_1x1(x)
        x2 = self.doubleconv(x)
        return x1 + x2

class RecConvDS(nn.Module):
    def __init__(self,in_channels,  out_channels, t=2,kernels_per_layer=1):
        super(RecConvDS, self).__init__()
        self.RecCNN = nn.Sequential(
            RecurrentConvDS_unit(out_channels, t=t,kernels_per_layer=kernels_per_layer),
            RecurrentConvDS_unit(out_channels, t=t,kernels_per_layer=kernels_per_layer)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x = self.RecCNN(x)
        return x

class RecResConvDS_block(nn.Module):
    def __init__(self, in_channels, out_channels, t=2,kernels_per_layer=1):
        super(RecResConvDS_block, self).__init__()
        self.RecCNN = nn.Sequential(
            RecurrentConvDS_unit(out_channels, t=t,kernels_per_layer=kernels_per_layer),
            RecurrentConvDS_unit(out_channels, t=t,kernels_per_layer=kernels_per_layer)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RecCNN(x)
        return x + x1

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
