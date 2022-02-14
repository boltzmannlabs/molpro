from typing import List
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv3d(in_dim, out_dim, kernel_size=(k, k, k), stride=(s, s, s), padding=(p, p, p))]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class InnerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InnerBlock, self).__init__()

        self.conv = ConvBlock(in_channels, 4 * out_channels, k=1, s=1, p=0, norm=False, non_linear=None)
        self.block1 = ConvBlock(in_channels, out_channels, k=3, s=1, p=1, norm=False, non_linear='relu')
        self.block2 = ConvBlock(out_channels, out_channels, k=1, s=1, p=0, norm=False, non_linear='relu')
        self.block3 = ConvBlock(out_channels, 4 * out_channels, k=1, s=1, p=0, norm=False, non_linear='relu')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.block3(self.block2(self.block1(x)))
        out = x1 + x2
        return out


class StartBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StartBlock, self).__init__()

        self.conv = ConvBlock(in_channels, 4 * out_channels, k=1, s=2, p=0, norm=False, non_linear=None)
        self.block1 = ConvBlock(in_channels, out_channels, k=3, s=2, p=1, norm=False, non_linear='relu')
        self.block2 = ConvBlock(out_channels, out_channels, k=1, s=1, p=0, norm=False, non_linear='relu')
        self.block3 = ConvBlock(out_channels, 4 * out_channels, k=1, s=1, p=0, norm=False, non_linear='relu')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.block3(self.block2(self.block1(x)))
        out = x1 + x2
        return out


class ResNet(nn.Module):
    def __init__(self,  input_channel: int, output_channel: int, intermediate_channel: int, layers: List[int]):
        super(ResNet, self).__init__()

        self.conv0 = ConvBlock(input_channel, 32, k=3, s=2)
        self.conv = ConvBlock(32, intermediate_channel, k=7, s=2, p=2, norm=False, non_linear='relu')
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layers = layers
        self.inner_blocks = []
        self.inner_blocks.append(InnerBlock(intermediate_channel, intermediate_channel))
        for i in range(1, self.layers[0]):
            self.inner_blocks.append(InnerBlock(4 * intermediate_channel, intermediate_channel))
        self.inner_blocks = nn.Sequential(*self.inner_blocks)

        self.start_inner_1 = []
        self.start_inner_1.append(StartBlock(4 * intermediate_channel, intermediate_channel * (2 ** 1)))
        for j in range(self.layers[1] - 1):
            self.start_inner_1.append(InnerBlock(4 * intermediate_channel * (2 ** 1), intermediate_channel * (2 ** 1)))
        self.start_inner_1 = nn.Sequential(*self.start_inner_1)

        self.start_inner_2 = []
        self.start_inner_2.append(StartBlock(4 * intermediate_channel * (2 ** 1), intermediate_channel * (2 ** 2)))
        for j in range(self.layers[2] - 1):
            self.start_inner_2.append(InnerBlock(4 * intermediate_channel * (2 ** 2), intermediate_channel * (2 ** 2)))
        self.start_inner_2 = nn.Sequential(*self.start_inner_2)

        self.start_inner_3 = []
        self.start_inner_3.append(StartBlock(4 * intermediate_channel * (2 ** 2), intermediate_channel * (2 ** 3)))
        for j in range(self.layers[3] - 1):
            self.start_inner_3.append(InnerBlock(4 * intermediate_channel * (2 ** 3), intermediate_channel * (2 ** 3)))
        self.start_inner_3 = nn.Sequential(*self.start_inner_3)

        self.fc1 = nn.Linear(2048, output_channel)

    def forward(self, x):
        x = self.pool(self.conv(self.conv0(x)))
        x = self.inner_blocks(x)
        x = self.start_inner_1(x)
        x = self.start_inner_2(x)
        x = self.start_inner_3(x)
        x = x.flatten(1)
        return self.fc1(x)
