import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as dataset

import utils
from Build_Dataset import _data_transforms_cifar10
from utils import count_parameters_in_MB, Calculate_flops

BASE_DIR = os.getcwd()
INPLACE = False
BIAS = False


class ConvBNReLU6(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1):
        super(ConvBNReLU6, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation, group, bias=BIAS),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=INPLACE)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1):
        super(ConvBNReLU, self).__init__()
        if isinstance(kernel_size, int):
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation, group, bias=BIAS),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=INPLACE)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=(k1, k2), stride=(1, stride), padding=padding[0], bias=BIAS),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(c_out, c_out, kernel_size=(k2, k1), stride=(stride, 1), padding=padding[1], bias=BIAS),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=INPLACE)
            )

    def forward(self, x):
        return self.conv(x)


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1):
        super(ReLUConvBN, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                          groups=group, bias=BIAS),
                nn.BatchNorm2d(c_out)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]  # (1, 3) & (3, 1)
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(c_in, c_out, kernel_size=(k1, k2), stride=(1, stride), padding=padding[0], bias=BIAS),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(c_out, c_out, kernel_size=(k2, k1), stride=(stride, 1), padding=padding[1], bias=BIAS),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        return self.ops(x)


# 按照我的理解的空洞和可分离
class DilateRCB(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation):
        super(DilateRCB, self).__init__()
        self.ops = ReLUConvBN(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation)

    def forward(self, x):
        return self.ops(x)


class DilateCBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation):
        super(DilateCBR, self).__init__()
        self.ops = ConvBNReLU(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation)

    def forward(self, x):
        return self.ops(x)


class SepRCB(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation=1):
        super(SepRCB, self).__init__()
        self.relu6 = nn.ReLU6(inplace=INPLACE)
        self.dw = nn.Conv2d(c_in, c_in, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation,
                            groups=c_in)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        out = self.relu6(x)
        out = self.dw(out)
        out = self.bn1(out)

        out = self.relu6(out)
        out = self.pw(out)
        out = self.bn2(out)
        return out


class SepCBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation=1):
        super(SepCBR, self).__init__()
        self.dw = nn.Conv2d(c_in, c_in, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation,
                            groups=c_in)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu6 = nn.ReLU6(inplace=INPLACE)

    def forward(self, x):
        out = self.dw(x)
        out = self.bn1(out)
        out = self.relu6(out)

        out = self.pw(out)
        out = self.bn2(out)
        out = self.relu6(out)
        return out


# DARTs中的空洞和可分离
class DARTsDilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, c_in, c_out, kernel_size, stride, dilation):
        super(DARTsDilConv, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(c_in, c_in, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation,
                      groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.net(x)


class DARTsSepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, c_in, c_out, kernel_size, stride, dilation=1):
        super(DARTsSepConv, self).__init__()
        self.net = nn.Sequential(
            DARTsDilConv(c_in, c_in, kernel_size, stride, dilation=dilation),
            DARTsDilConv(c_in, c_out, kernel_size, 1, dilation=1)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.downsample = None
        if c_in != c_out or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in ,c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out)
            )
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.conv1 = ConvBNReLU(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation[0]+1)//2, dilation=dilation[0])
        # self.conv1 = nn.Conv2d(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation[0]+1)//2, dilation=dilation[0])
        # self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out ,c_out ,kernel_size, 1, padding=((kernel_size-1)*dilation[1]+1)//2, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=INPLACE)

        # self.conv1 = Conv(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation)
        # self.conv2 = Conv(c_out, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation)
        # self.relu = nn.ReLU(inplace=INPLACE)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, groups=1, dilation=1):
        super(BottleNeck, self).__init__()
        self.downsample = None
        if c_in != c_out or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out)
            )
        width = c_out * groups
        self.conv1 = ConvBNReLU(c_in, width, 1, 1, 0)
        # self.conv1 = nn.Conv2d(c_in, width, 1, 1, 0)
        # self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = ConvBNReLU(width, width, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2,
                                dilation=dilation, group=groups)
        # self.conv2 = nn.Conv2d(width, width, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2,
        #                        dilation=dilation, groups=groups)
        # self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, c_in, 1, 1 ,0)
        self.bn3 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=INPLACE)

        # self.pw1 = Conv(c_in, int(c_in*ex_ratio), 1, 1, 0)
        # self.dw = Conv(int(c_in*ex_ratio), int(c_in*ex_ratio), kernel_size, stride,
        #                padding=((kernel_size-1)*dilation+1)//2, dilation=dilation)
        # self.pw2 = Conv(int(c_in*ex_ratio), c_out, 1, 1, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class IRB(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation=1, ex_ratio=3):
        super(IRB, self).__init__()
        hidden_dim = int(round(c_in * ex_ratio))
        self.use_res_connect = stride == 1 and c_in == c_out

        layers = []
        if ex_ratio != 1:
            layers.append(ConvBNReLU6(c_in, hidden_dim, 1, 1, 0))
        layers.extend([
            ConvBNReLU6(hidden_dim, hidden_dim, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2,
                        dilation=dilation, group=hidden_dim),
            nn.Conv2d(hidden_dim, c_out, 1, 1, 0, bias=BIAS),
            nn.BatchNorm2d(c_out)
        ])
        self.conv = nn.Sequential(*layers)

        # self.relu6 = nn.ReLU6(inplace=INPLACE)
        # self.pw1 = nn.Conv2d(c_in, int(c_in*ex_ratio), 1, 1, 0)
        # self.bn1 = nn.BatchNorm2d(int(c_in*ex_ratio), affine=affine)
        # self.dw = nn.Conv2d(int(c_in*ex_ratio), int(c_in*ex_ratio), kernel_size, stride,
        #                     padding=((kernel_size-1)*dilation+1)//2, dilation=dilation, groups=c_in*ex_ratio)
        # self.pw2 = nn.Conv2d(int(c_in*ex_ratio), c_out, 1, 1, 0)
        # self.bn2 = nn.BatchNorm2d(c_out, affine=affine)
        # self.relu = nn.ReLU(inplace=INPLACE)

    def forward(self, x):
        residual = x
        if self.use_res_connect:
            return residual + self.conv(x)
        else:
            return self.conv(x)

        # out = self.bn1(self.pw1(self.relu6(x)))
        # out = self.bn1(self.dw(self.relu6(out)))
        # out = self.bn2(self.pw2(self.relu6(out)))
        # out += residual
        # out = self.relu(out)
        # return out


class SeLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SeLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.op = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.op(y).view(batch, channel, 1, 1)
        return x * y.expand_as(x)


# class ECALayer(nn.Module):
#     """
#     Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=3):
#         super(ECALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)


class ECALayer(nn.Module):
    def __init__(self, channel, k_size):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LocalBinaryConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1, groups=1):
        super(LocalBinaryConv, self).__init__()
        self.nInputPlane = c_in
        self.nOutputPlane = c_out
        self.kW = kernel_size
        self.LBCNN = nn.Conv2d(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2, dilation=dilation, groups=groups, bias=BIAS)
        self.LBCNN.weight.requires_grad = False
        # init weight
        numElements = self.nInputPlane * self.nOutputPlane * self.kW * self.kW
        index = torch.randperm(numElements)
        self.LBCNN.weight.copy = (torch.Tensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kW).uniform_(0, 1))
        temp = (torch.bernoulli(self.LBCNN.weight.copy) * 2 - 1).view(-1)
        for i in range(1, numElements // 2):
            temp[index[i]] = 0
        self.LBCNN.weight.copy = temp.view(self.nOutputPlane, self.nInputPlane, self.kW, self.kW)
        self.LBCNN.weight = nn.Parameter(self.LBCNN.weight.copy)

    def forward(self, input):
        return self.LBCNN(input)


class LocalBinaryConvm(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1, groups=1):
        super(LocalBinaryConvm, self).__init__()
        self.nInputPlane = c_in
        self.nOutputPlane = c_out
        self.kW = kernel_size
        self.LBCNN = nn.Conv2d(c_in, c_out, kernel_size, stride, padding=((kernel_size-1)*dilation+1)//2,
                               dilation=dilation, groups=groups, bias=BIAS)
        # self.LBCNN.weight.requires_grad = False
        # init weight
        numElements = self.nInputPlane * self.nOutputPlane * self.kW * self.kW
        index = torch.randperm(numElements)
        self.LBCNN.weight.copy = (torch.Tensor(self.nOutputPlane, self.nInputPlane, self.kW, self.kW).uniform_(0, 1))
        temp = (torch.bernoulli(self.LBCNN.weight.copy) * 2 - 1).view(-1)
        for i in range(1, numElements // 2):
            temp[index[i]] = 0
        self.LBCNN.weight.copy = temp.view(self.nOutputPlane, self.nInputPlane, self.kW, self.kW)
        self.LBCNN.weight = nn.Parameter(self.LBCNN.weight.copy)
        self.LBCNN.weight.requires_grad = True

    def forward(self, input):
        return self.LBCNN(input)


class Involution(nn.Module):
    def __init__(self, c_in, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = c_in
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvBNReLU(c_in, c_in // reduction_ratio, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels=c_in // reduction_ratio, out_channels=kernel_size**2 * self.groups,
                               kernel_size=1, stride=1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        # nn.Unfold(kernel_size, dilation, padding, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        # unsqueeze在特定维度添加一维
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


# Operations_test = {
#     0: lambda c_in, c_out: ReLUConvBN(c_in, c_out, 3, 1, 1),
#     1: lambda c_in, c_out: ReLUConvBN(c_in, c_out, (1, 3), 1, ((0, 1), (1, 0))),
#     2: lambda c_in, c_out: ConvBNReLU(c_in, c_out, 3, 1, 1),
#     3: lambda c_in, c_out: ConvBNReLU(c_in, c_out, (1, 3), 1, ((0, 1), (1, 0))),
#     4: lambda c_in, c_out: DilateRCB(c_in, c_out, 3, 1, 2),
#     5: lambda c_in, c_out: DilateCBR(c_in, c_out, 3, 1, 2),
#     6: lambda c_in, c_out: SepRCB(c_in, c_out, 3, 1, 2),
#     7: lambda c_in, c_out: SepCBR(c_in, c_out, 3, 1, 2),
#     8: lambda c_in, c_out: DARTsDilConv(c_in, c_out, 3, 1, 2),
#     9: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 3, 1, 2),
#     10: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, 1),
#     11: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, 2),
#     12: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, (1,2)),
#     13: lambda c_in, c_out: BottleNeck(c_in, c_out, 3, 1, 2),
#     14: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 1, 3),
#     15: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 2, 3),
#     16: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 3, 1, 2)
#     17: lambda c_in, c_out: Involution(c_in, 3, 1),
#     18: lambda c_in, c_out: Involution(c_in, 5, 1)
# }
#
#
# Operations_name_test = [
#     'RCB-k3',
#     'RCB-k1*3+3*1',
#     'CBR-k3',
#     'CBR-k1*3+3*1',
#     'DilRCB-k3-d2',
#     'DilCBR-k3-d2',
#     'SepRCB-k3-d2',
#     'SepCBR-k3-d2',
#     'DARTs-Dil-k3-d2',
#     'DARTs-Sep-k3-d2',
#     'rb-k3-d1',
#     'rb-k3-d2',
#     'rb-k3-d12',
#     'Bottleneck-k3-d2',
#     'irb-k3-d1-e3',
#     'irb-k3-d2-e3',
#     'LBC-k3-d2'
#     'inv-3',
#     'inv-5'
# ]


Operations = {
    0: lambda c_in, c_out: ReLUConvBN(c_in, c_out, 3, 1, 1),
    1: lambda c_in, c_out: ReLUConvBN(c_in, c_out, 5, 1, 2),
    2: lambda c_in, c_out: ReLUConvBN(c_in, c_out, (1, 3), 1, ((0, 1), (1, 0))),
    3: lambda c_in, c_out: ReLUConvBN(c_in, c_out, (1, 5), 1, ((0, 2), (2, 0))),

    4: lambda c_in, c_out: ConvBNReLU(c_in, c_out, 3, 1, 1),
    5: lambda c_in, c_out: ConvBNReLU(c_in, c_out, 5, 1, 2),
    6: lambda c_in, c_out: ConvBNReLU(c_in, c_out, (1, 3), 1, ((0, 1), (1, 0))),
    7: lambda c_in, c_out: ConvBNReLU(c_in, c_out, (1, 5), 1, ((0, 2), (2, 0))),

    8: lambda c_in, c_out: DilateRCB(c_in, c_out, 3, 1, 2),
    9: lambda c_in, c_out: DilateRCB(c_in, c_out, 3, 1, 3),
    10: lambda c_in, c_out: DilateRCB(c_in, c_out, 5, 1, 2),
    11: lambda c_in, c_out: DilateRCB(c_in, c_out, 5, 1, 3),

    12: lambda c_in, c_out: DilateCBR(c_in, c_out, 3, 1, 2),
    13: lambda c_in, c_out: DilateCBR(c_in, c_out, 3, 1, 3),
    14: lambda c_in, c_out: DilateCBR(c_in, c_out, 5, 1, 2),
    15: lambda c_in, c_out: DilateCBR(c_in, c_out, 5, 1, 3),

    16: lambda c_in, c_out: SepRCB(c_in, c_out, 3, 1, 1),
    17: lambda c_in, c_out: SepRCB(c_in, c_out, 3, 1, 2),
    18: lambda c_in, c_out: SepRCB(c_in, c_out, 3, 1, 3),
    19: lambda c_in, c_out: SepRCB(c_in, c_out, 5, 1, 1),
    20: lambda c_in, c_out: SepRCB(c_in, c_out, 5, 1, 2),
    21: lambda c_in, c_out: SepRCB(c_in, c_out, 5, 1, 3),

    22: lambda c_in, c_out: SepCBR(c_in, c_out, 3, 1, 1),
    23: lambda c_in, c_out: SepCBR(c_in, c_out, 3, 1, 2),
    24: lambda c_in, c_out: SepCBR(c_in, c_out, 3, 1, 3),
    25: lambda c_in, c_out: SepCBR(c_in, c_out, 5, 1, 1),
    26: lambda c_in, c_out: SepCBR(c_in, c_out, 5, 1, 2),
    27: lambda c_in, c_out: SepCBR(c_in, c_out, 5, 1, 3),

    28: lambda c_in, c_out: DARTsDilConv(c_in, c_out, 3, 1, 2),
    29: lambda c_in, c_out: DARTsDilConv(c_in, c_out, 3, 1, 3),
    30: lambda c_in, c_out: DARTsDilConv(c_in, c_out, 5, 1, 2),
    31: lambda c_in, c_out: DARTsDilConv(c_in, c_out, 5, 1, 3),

    32: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 3, 1, 1),
    33: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 3, 1, 2),
    34: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 3, 1, 3),
    35: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 5, 1, 1),
    36: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 5, 1, 2),
    37: lambda c_in, c_out: DARTsSepConv(c_in, c_out, 5, 1, 3),

    38: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, 1),
    39: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, 2),
    40: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, 3),
    41: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, (1,2)),
    42: lambda c_in, c_out: ResidualBlock(c_in, c_out, 3, 1, (2,1)),
    43: lambda c_in, c_out: ResidualBlock(c_in, c_out, 5, 1, 1),
    44: lambda c_in, c_out: ResidualBlock(c_in, c_out, 5, 1, 2),
    45: lambda c_in, c_out: ResidualBlock(c_in, c_out, 5, 1, 3),

    46: lambda c_in, c_out: BottleNeck(c_in, c_out, 3, 1, 1),
    47: lambda c_in, c_out: BottleNeck(c_in, c_out, 3, 1, 2),
    48: lambda c_in, c_out: BottleNeck(c_in, c_out, 3, 1, 3),
    49: lambda c_in, c_out: BottleNeck(c_in, c_out, 5, 1, 1),
    50: lambda c_in, c_out: BottleNeck(c_in, c_out, 5, 1, 2),
    51: lambda c_in, c_out: BottleNeck(c_in, c_out, 5, 1, 3),

    52: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 1, 1),
    53: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 1, 3),
    54: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 1, 6),
    55: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 2, 1),
    56: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 2, 3),
    57: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 2, 6),
    58: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 3, 1),
    59: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 3, 3),
    60: lambda c_in, c_out: IRB(c_in, c_out, 3, 1, 3, 6),
    61: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 1, 1),
    62: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 1, 3),
    63: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 1, 6),
    64: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 2, 1),
    65: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 2, 3),
    66: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 2, 6),
    67: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 3, 1),
    68: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 3, 3),
    69: lambda c_in, c_out: IRB(c_in, c_out, 5, 1, 3, 6),

    70: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 3, 1, 1),
    71: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 3, 1, 2),
    72: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 3, 1, 3),
    73: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 5, 1, 1),
    74: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 5, 1, 2),
    75: lambda c_in, c_out: LocalBinaryConv(c_in, c_out, 5, 1, 3),
}

Operations_name = [
    'RCB-k3',
    'RCB-k5',
    'RCB-k13-31',
    'RCB-k15-51',
    'CBR-k3',
    'CBR-k5',
    'CBR-k13-31',
    'CBR-k15-51',
    'DilRCB-k3-d2',
    'DilRCB-k3-d3',
    'DilRCB-k5-d2',
    'DilRCB-k5-d3',
    'DilCBR-k3-d2',
    'DilCBR-k3-d3',
    'DilCBR-k5-d2',
    'DilCBR-k5-d3',
    'SepRCB-k3-d1',
    'SepRCB-k3-d2',
    'SepRCB-k3-d3',
    'SepRCB-k5-d1',
    'SepRCB-k5-d2',
    'SepRCB-k5-d3',
    'SepCBR-k3-d1',
    'SepCBR-k3-d2',
    'SepCBR-k3-d3',
    'SepCBR-k5-d1',
    'SepCBR-k5-d2',
    'SepCBR-k5-d3',
    'DARTs-Dil-k3-d2',
    'DARTs-Dil-k3-d3',
    'DARTs-Dil-k5-d2',
    'DARTs-Dil-k5-d3',
    'DARTs-Sep-k3-d1',
    'DARTs-Sep-k3-d2',
    'DARTs-Sep-k3-d3',
    'DARTs-Sep-k5-d1',
    'DARTs-Sep-k5-d2',
    'DARTs-Sep-k5-d3',
    'rb-k3-d1',
    'rb-k3-d2',
    'rb-k3-d3',
    'rb-k3-d12',
    'rb-k3-d21',
    'rb-k5-d1',
    'rb-k5-d2',
    'rb-k5-d3',
    'Bottleneck-k3-d1',  # 先1*1降通道，然后深度卷积，然后1*1恢复通道
    'Bottleneck-k3-d2',
    'Bottleneck-k3-d3',
    'Bottleneck-k5-d1',
    'Bottleneck-k5-d2',
    'Bottleneck-k5-d3',
    'irb-k3-d1-e1',
    'irb-k3-d1-e3',
    'irb-k3-d1-e6',
    'irb-k3-d2-e1',
    'irb-k3-d2-e3',
    'irb-k3-d2-e6',
    'irb-k3-d3-e1',
    'irb-k3-d3-e3',
    'irb-k3-d3-e6',
    'irb-k5-d1-e1',
    'irb-k5-d1-e3',
    'irb-k5-d1-e6',
    'irb-k5-d2-e1',
    'irb-k5-d2-e3',
    'irb-k5-d2-e6',
    'irb-k5-d3-e1',
    'irb-k5-d3-e3',
    'irb-k5-d3-e6',
    'LBC-k3-d1',
    'LBC-k3-d2',
    'LBC-k3-d3',
    'LBC-k5-d1',
    'LBC-k5-d2',
    'LBC-k5-d3',
]


# CIFAR10
def build_train_cifar10(args, cutout_size=None):
    # used for training process, so valid_data "train=False"

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size)

    train_data = dataset.CIFAR10(root=os.path.join(BASE_DIR, 'data'), train=True, download=True, transform=train_transform)
    valid_data = dataset.CIFAR10(root=os.path.join(BASE_DIR, 'data'), train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    inference_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=16)

    return train_queue, valid_queue, inference_queue


def build_train_Optimizer_Loss(model, args, epoch=-1):
    model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_max,
        momentum=args.momentum,
        weight_decay=args.l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)

    return train_criterion, eval_criterion, optimizer, scheduler


def train(train_queue, model, args, train_criterion, optimizer, scheduler, validate_queue, inference_queue, eval_criterion):
    total = len(train_queue)
    best_top1 = 0
    best_top5 = 0
    best_loss = 0
    best_dict = None

    for epoch in range(args.epochs):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (inputs, targets) in enumerate(train_queue):
            print('\r[Epoch:{}/{}, Training:{}/{}]'.format(epoch+1, args.epochs, step+1, total), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = train_criterion(outputs, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
            optimizer.step()

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

        scheduler.step()

        if (epoch+1) == args.epochs:
            print('Training accuracy top1:{:.4f}, top5:{:.4f}, loss:{:.6f}'.format(top1.avg, top5.avg, objs.avg))
            logging.info('Training accuracy top1:{:.4f}, top5:{:.4f}, loss:{:.6f}'.format(top1.avg, top5.avg, objs.avg))
        valid_top1, valid_top5, valid_loss, dict_params = eval(validate_queue, model, args, eval_criterion, epoch)
        if valid_top1 > best_top1:
            best_top1 = valid_top1
            best_top5 = valid_top5
            best_loss = valid_loss
            best_dict = dict_params


    avg_time = inference(inference_queue, model)

    return top1.avg, top5.avg, objs.avg, best_top1, best_top5, best_loss, best_dict, avg_time


def eval(valid_queue, model, args, eval_criterion, epoch):
    total = len(valid_queue)

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    with torch.no_grad():
        model.eval()
        for step, (inputs, targets) in enumerate(valid_queue):
            print('\r[-------------Validating {0:>2d}/{1:>2d}]'.format(step + 1, total), end='')

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)

            loss = eval_criterion(outputs, targets)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    if (epoch+1) == args.epochs:
        print('valid accuracy top1:{0:.4f}, top5:{1:.4f}, loss:{2:.6f}'.format(top1.avg, top5.avg, objs.avg))
        logging.info('valid accuracy top1:{0:.4f}, top5:{1:.4f}, loss:{2:.6f}'.format(top1.avg, top5.avg, objs.avg))

    return top1.avg, top5.avg, objs.avg, model.state_dict()


def inference(inference_queue, model):
    total_infer = len(inference_queue)
    avg_time = 0
    # for inference time
    with torch.no_grad():
        model.eval()
        print('[-------------inferring ({0:>2d}images)]'.format(total_infer), end=" ")
        for step, (inputs, targets) in enumerate(inference_queue):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            since = time.time()
            _ = model(inputs)
            end = time.time()
            avg_time += (end - since) / total_infer
    print('average inference time:{:.6f} ms'.format(avg_time*1000))

    return avg_time * 1000


# baseline
class NetWork(nn.Module):
    def __init__(self, args, op_id):
        super(NetWork, self).__init__()
        self.args = args
        self.op_id = op_id
        self.pool_layers = [self.args.layers, self.args.layers*2 + 1]
        self.total_layers = self.args.layers * 3 + 2

        self.stem = ConvBNReLU(3, args.channels, 3, 1, 1)
        self._make_layer()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(args.channels, args.classes)

        self.init_parameters()

    def _make_layer(self):
        self.layers = nn.ModuleList()
        for i in range(self.total_layers):
            if i in self.pool_layers:
                self.layers.append(nn.MaxPool2d(2, 2))
            else:
                layer = Operations[self.op_id](self.args.channels, self.args.channels)
                self.layers.append(layer)

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, x):
        out = self.stem(x)
        for i in range(self.total_layers):
            out = self.layers[i](out)
        out = self.avgpool(out)
        out = self.fc(out.view(out.shape[0], -1))
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='operation test')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-pth_path', type=str, default='pthfiles', help='save model parameters')
    parser.add_argument('-channels', type=int, default=64)  # 64
    parser.add_argument('-layers', type=int, default=4, help='operation repeat times')
    parser.add_argument('-epochs', type=int, default=100)  # 100
    parser.add_argument('-classes', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-lr_max', type=float, default=0.025)
    parser.add_argument('-lr_min', type=float, default=0)
    parser.add_argument('-l2_reg', type=float, default=5e-4, help='weight decay')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-grad_bound', type=float, default=6.0, help='for grad clip')
    args = parser.parse_args()

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/operation_test-test.log'.format(BASE_DIR),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    train_queue, validate_queue, inference_queue = build_train_cifar10(args)

    model_names = []
    accs = []
    infer_times = []
    num_paramss = []
    Flopss = []

    for i in range(len(Operations)):
        model = NetWork(args, i).to(args.device)
        train_criterion, eval_criterion, optimizer, scheduler = build_train_Optimizer_Loss(model, args)

        # top1.avg, top5.avg, objs.avg, best_top1, best_top5, best_loss, best_dict, avg_time
        result = train(train_queue, model, args, train_criterion, optimizer, scheduler, validate_queue, inference_queue, eval_criterion)
        num_params = count_parameters_in_MB(model)
        Flops = Calculate_flops(model)
        logging.info('Model Name: {0}'.format(Operations_name[i]))
        logging.info('Model Accuracy: {0:.4f}, Model infer time: {1:.6f} ms, '
                     'Model Parameters: {2:.4f} MB, Model Flops: {3:.6f} MB\n'.format(result[3],
                                                                                      result[7],
                                                                                      num_params,
                                                                                      Flops))
        # torch.save(result[6], os.path.join(BASE_DIR, args.pth_path, Operations_name[i] + '.pth'))
        model_names.append(Operations_name[i])
        accs.append(result[3])
        infer_times.append(result[7])
        num_paramss.append(num_params)
        Flopss.append(Flops)

    logging.info('Model_name\tAccuracy(%)\tInference_time(ms)\tParameters(MB)\tFLOPs(MB)')

    for i in range(len(model_names)):
        logging.info('{0}\t{1:.6f}\t{2:.6f}\t{3:.4f}\t{4:.6f}'.format(model_names[i],
                                                          accs[i],
                                                          infer_times[i],
                                                          num_paramss[i],
                                                          Flopss[i]))
