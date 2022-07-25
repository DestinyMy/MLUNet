import numpy as np
import torch
import torch.nn as nn


INPLACE = False
BIAS = False


class ConvBNReLU6(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1, shape=None):
        super(ConvBNReLU6, self).__init__()
        self.shape = shape

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation, group, bias=BIAS),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=INPLACE)
        )
    def forward(self, x):
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1, shape=None):
        super(ConvBNReLU, self).__init__()
        self.shape = shape
        self.out_channel = c_out

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, dilation, group, bias=BIAS),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=INPLACE)
        )
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1, shape=None):
        super(ResidualBlock, self).__init__()
        self.downsample = None
        self.shape = shape
        self.out_channel = c_out

        if c_in != c_out or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out)
            )

        self.conv1 = ConvBNReLU(c_in, c_out, kernel_size, stride, padding=(kernel_size - 1) // 2 * dilation,
                                dilation=dilation, shape=shape)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=INPLACE)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, groups=1, dilation=1, shape=None):
        super(BottleNeck, self).__init__()
        self.downsample = None

        width = c_out * groups
        self.out_channel = c_out

        if c_in != self.out_channel or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, self.out_channel, 1, stride, 0),
                nn.BatchNorm2d(self.out_channel)
            )

        self.conv1 = ConvBNReLU(c_in, width, 1, 1, 0, shape=shape)
        self.conv2 = ConvBNReLU(width, width, kernel_size, stride, padding=(kernel_size - 1) // 2 * dilation,
                                dilation=dilation, group=groups, shape=shape)
        self.conv3 = nn.Conv2d(width, self.out_channel, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=INPLACE)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class IRB(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, dilation=1, ex_ratio=3, shape=None):
        super(IRB, self).__init__()
        hidden_dim = int(round(c_in * ex_ratio))
        self.use_res_connect = stride == 1 and c_in == c_out
        self.out_channel = c_out

        layers = []
        if ex_ratio != 1:
            layers.append(ConvBNReLU6(c_in, hidden_dim, 1, 1, 0, shape=shape))
        layers.extend([
            ConvBNReLU6(hidden_dim, hidden_dim, kernel_size, stride, padding=(kernel_size - 1) // 2 * dilation,
                        dilation=dilation, group=hidden_dim, shape=shape),
            nn.Conv2d(hidden_dim, c_out, 1, 1, 0, bias=BIAS),
            nn.BatchNorm2d(c_out)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        if self.use_res_connect:
            return residual + self.conv(x)
        else:
            return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# -------------------------------------------------------------------
class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x):
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(768, classes)

    def forward(self, x):
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


Operations = {
    0: lambda c_in, c_out, stride, shape: ConvBNReLU(c_in, c_out, 3, stride, 1, shape=shape),
    1: lambda c_in, c_out, stride, shape: ResidualBlock(c_in, c_out, 3, stride, shape=shape),
    2: lambda c_in, c_out, stride, shape: BottleNeck(c_in, c_out, 3, stride, dilation=2, shape=shape),
    3: lambda c_in, c_out, stride, shape: BottleNeck(c_in, c_out, 5, stride, dilation=2, shape=shape),
    4: lambda c_in, c_out, stride, shape: IRB(c_in, c_out, 3, stride, shape=shape),
    5: lambda c_in, c_out, stride, shape: IRB(c_in, c_out, 3, stride, ex_ratio=6, shape=shape),
    6: lambda c_in, c_out, stride, shape: IRB(c_in, c_out, 5, stride, shape=shape),
    7: lambda c_in, c_out, stride, shape: IRB(c_in, c_out, 5, stride, ex_ratio=6, shape=shape),
}

Operations_name = [
    'CBR-k3',
    'RB-k3-d1',
    'BN-k3-d2',
    'BN-k5-d2',

    'IRB-k3-d1-e3',
    'IRB-k3-d1-e6',
    'IRB-k5-d1-e3',
    'IRB-k5-d1-e6',
]

Operations_len = np.array([1, 2, 3, 3, 3, 3, 3, 3])
Init_Channel = [16, 24, 32, 40, 48, 64]
