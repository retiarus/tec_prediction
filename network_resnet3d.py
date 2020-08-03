import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import CLSTM_cell as Recurrent_cell

torch.set_default_dtype(torch.float32)


def swish(x):
    return x * torch.sigmoid(x)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['swish', swish],
        ['none', nn.Identity()]
    ])[activation]


class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        # dynamic add padding based on the kernel_size
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2,
                        self.kernel_size[0] // 2,
                        self.kernel_size[1] // 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='swish'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        if self.should_apply_shortcut:
            residual = self.shortcut(x)

        x = self.blocks(x)
        x += residual
        x = self.activate(x)

        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 downsampling=1,
                 conv=conv3x3x3,
                 *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv

        if self.should_apply_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(self.in_channels,
                          self.expanded_channels,
                          kernel_size=1,
                          stride=self.downsampling,
                          bias=False),
                          nn.BatchNorm3d(self.expanded_channels))
        else:
            self.shortcut = None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels,
                              out_channels,
                              *args,
                              **kwargs),
                         nn.BatchNorm3d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels,
                    self.out_channels,
                    conv=self.conv,
                    bias=False,
                    stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels,
                    self.expanded_channels,
                    conv=self.conv,
                    bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )
