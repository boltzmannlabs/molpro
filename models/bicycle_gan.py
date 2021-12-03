
import torch
from torch import nn


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


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1,o_p=1, norm=True, non_linear='relu'):
        super(DeconvBlock, self).__init__()
        layers = []

        # Transpose Convolution Layer
        layers += [nn.ConvTranspose3d(in_dim, out_dim, kernel_size=k, stride=s, padding=p, output_padding=o_p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-Linearity Layer
        if non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]
        elif non_linear == 'sigmoid':
            layers += [nn.Sigmoid()]

        self.deconv_block = nn.Sequential(* layers)

    def forward(self, x):
        out = self.deconv_block(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=8):
        super(Generator, self).__init__()

        self.downsample_1 = ConvBlock(35 + z_dim, 64, k=3, s=1, p=1, norm=False, non_linear='leaky_relu')
        self.downsample_2 = ConvBlock(64, 128, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_3 = ConvBlock(128, 256, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_4 = ConvBlock(256, 512, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_5 = ConvBlock(512, 512, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_6 = ConvBlock(512, 512, k=3, s=2, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_7 = ConvBlock(512, 512, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')
        self.downsample_8 = ConvBlock(512, 512, k=3, s=1, p=1, norm=True, non_linear='leaky_relu')

        # Need concatenation when upsampling, see foward function for details
        self.upsample_1 = DeconvBlock(512, 512, k=3, s=1, p=1,o_p=0, norm=True, non_linear='relu')
        self.upsample_2 = DeconvBlock(1024, 512, k=3, s=1, p=1,o_p=0, norm=True, non_linear='relu')
        self.upsample_3 = DeconvBlock(1024, 512, k=3, s=2, p=1,o_p=1, norm=True, non_linear='relu')
        self.upsample_4 = DeconvBlock(1024, 512, k=3, s=1, p=1,o_p=0, norm=True, non_linear='relu')
        self.upsample_5 = DeconvBlock(1024, 256, k=3, s=2, p=1,o_p=1, norm=True, non_linear='relu')
        self.upsample_6 = DeconvBlock(512, 128, k=3, s=1, p=1,o_p=0, norm=True, non_linear='relu')
        self.upsample_7 = DeconvBlock(256, 64, k=3, s=2, p=1,o_p=1, norm=True, non_linear='relu')
        self.upsample_8 = DeconvBlock(128, 34, k=3, s=1, p=1,o_p=0, norm=False, non_linear='sigmoid')

    def forward(self, x, z):
        # z : (N, z_dim) -> (N, z_dim, 1, 1) -> (N, z_dim, H, W)
        # x_with_z : (N, 3 + z_dim, H, W)
        z = z.unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3), x.size(4))
        x_with_z = torch.cat([x, z], dim=1)

        down_1 = self.downsample_1(x_with_z)
        down_2 = self.downsample_2(down_1)
        down_3 = self.downsample_3(down_2)
        down_4 = self.downsample_4(down_3)
        down_5 = self.downsample_5(down_4)
        down_6 = self.downsample_6(down_5)
        down_7 = self.downsample_7(down_6)
        down_8 = self.downsample_8(down_7)

        up_1 = self.upsample_1(down_8)
        up_2 = self.upsample_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.upsample_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.upsample_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.upsample_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.upsample_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.upsample_7(torch.cat([up_6, down_2], dim=1))
        out = self.upsample_8(torch.cat([up_7, down_1], dim=1))

        return out


class Encoder(nn.Module):
    def __init__(self, z_dim=8):
        super(Encoder, self).__init__()

        self.conv1 = ConvBlock(34, 64, k=3, s=1, p=1, norm=True, non_linear='relu')
        self.conv2 = ConvBlock(64, 128, k=3, s=1, p=1, norm=True, non_linear='relu')
        self.conv3 = ConvBlock(128, 128, k=3, s=1, p=1, norm=True, non_linear='relu')
        self.conv4 = ConvBlock(128, 256, k=3, s=1, p=1, norm=True, non_linear='relu')
        self.conv5 = ConvBlock(256, 256, k=3, s=1, p=1, norm=True, non_linear='relu')

        self.pool = nn.AvgPool3d(kernel_size=2,stride=2)

        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x):
        # (N, 3, 128, 128) -> (N, 64, 64, 64)
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))

        out = x.flatten(1)

        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)

        return mu, log_var


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d1 = nn.Sequential(ConvBlock(69, 64, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(64, 128, k=3, s=2, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(128, 256, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(256, 512, k=3, s=2, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(512, 512, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(512, 512, k=3, s=1, p=1, norm=False, non_linear=None))

        self.d2 = nn.Sequential(ConvBlock(69, 32, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(32, 64, k=3, s=2, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(64, 128, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(128, 256, k=3, s=2, p=1, norm=True, non_linear='leaky_relu'),
                                ConvBlock(256, 256, k=3, s=1, p=1, norm=True, non_linear='leaky_relu'), ConvBlock(256, 256, k=3, s=1, p=1, norm=False, non_linear=None))

    def forward(self, x):
        out_1 = self.d1(x)
        out_2 = self.d2(x)
        return out_1, out_2
