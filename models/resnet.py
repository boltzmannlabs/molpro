import torch
from torch import nn
import torch.nn.functional as F
    
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(* layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class InnerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InnerBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, 4*out_channels,k=1,s=1,p=0,norm=False,non_linear=None)
        self.block1 = ConvBlock(in_channels, out_channels,k=3,s=1,p=1,norm=False,non_linear='relu')
        self.block2 = ConvBlock(out_channels, out_channels,k=1,s=1,p=0,norm=False,non_linear='relu')
        self.block3 = ConvBlock(out_channels, 4*out_channels,k=1,s=1,p=0,norm=False,non_linear='relu')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.block3(self.block2(self.block1(x)))
        out = x1+x2
        return out
    
class StartBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StartBlock, self).__init__()
        
        self.conv = ConvBlock(in_channels, 4*out_channels,k=1,s=2,p=0,norm=False,non_linear=None)
        self.block1 = ConvBlock(in_channels, out_channels,k=3,s=2,p=1,norm=False,non_linear='relu')
        self.block2 = ConvBlock(out_channels, out_channels,k=1,s=1,p=0,norm=False,non_linear='relu')
        self.block3 = ConvBlock(out_channels, 4*out_channels,k=1,s=1,p=0,norm=False,non_linear='relu')

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.block3(self.block2(self.block1(x)))
        out = x1+x2
        return out
        
class ResNet(nn.Module):
    def __init__(self, intermediate_channel, layers = [3, 4, 6, 3]):
        super(ResNet, self).__init__()
        
        self.conv0 = ConvBlock(18,32,k=3,s=2)
        self.conv = ConvBlock(32, intermediate_channel,k=7,s=2,p=2,norm=False,non_linear='relu')
        self.pool = nn.MaxPool3d(kernel_size = 3, stride = 2, padding=1)
        self.layers = layers
        self.inner_blocks = nn.ModuleList()
        self.inner_blocks.append(InnerBlock(intermediate_channel,intermediate_channel))
        for i in range(1,self.layers[0]):
            self.inner_blocks.append(InnerBlock(4*intermediate_channel,intermediate_channel))
        self.start_inner_1 = nn.ModuleList()
        for j in range(self.layers[1]-1):
            self.start_inner_1.append(InnerBlock(4*intermediate_channel*(2**1),intermediate_channel*(2**1)))
        self.start_inner_2 = nn.ModuleList()
        for j in range(self.layers[2]-1):
            self.start_inner_2.append(InnerBlock(4*intermediate_channel*(2**2),intermediate_channel*(2**2)))
        self.start_inner_3 = nn.ModuleList()
        for j in range(self.layers[3]-1):
            self.start_inner_3.append(InnerBlock(4*intermediate_channel*(2**3),intermediate_channel*(2**3)))
        self.start_1 = nn.ModuleDict({
                'start': StartBlock(4*intermediate_channel,intermediate_channel*(2**1)),
                'inner': self.start_inner_1
        })
        self.start_2 = nn.ModuleDict({
                'start': StartBlock(4*intermediate_channel*(2**1),intermediate_channel*(2**2)),
                'inner': self.start_inner_2
        })
        self.start_3 = nn.ModuleDict({
                'start': StartBlock(4*intermediate_channel*(2**2),intermediate_channel*(2**3)),
                'inner': self.start_inner_3
        })
        
        self.fc1 = nn.Linear(2048,1)
        self.fc2 = nn.Linear(2048,2)

    def forward(self, x):
        x = self.pool(self.conv(self.conv0(x)))
        for inner_layer in self.inner_blocks:
            x = inner_layer(x)
        x = self.start_1['start'](x)
        for inner_start_layer in self.start_1['inner']:
            x = inner_start_layer(x)
        x = self.start_2['start'](x)
        for inner_start_layer in self.start_2['inner']:
            x = inner_start_layer(x)
        x = self.start_3['start'](x)
        for inner_start_layer in self.start_3['inner']:
            x = inner_start_layer(x)
        x = x.flatten(1)
        return self.fc1(x), self.fc2(x)

