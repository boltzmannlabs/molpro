import torch


class UnNetDownBlock(torch.nn.Module):
    def __init__(self, input_channel: int, output_channel: int, pool_size: int, down_size: bool) -> None:
        super(UnNetDownBlock, self).__init__()
        """UNet convolution block with downscaling
            Parameters
            ----------
            input_channel: int,
                Number of features in input grid
            output_channel: int,
                Desired number of features for output grid
            pool_size: int,
                Factor to downscale the 3D grid
            down_size: bool,
                Whether to downscale the 3D grid or not
        """
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.max_pool = torch.nn.MaxPool3d(pool_size)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNetUpBlock(torch.nn.Module):
    def __init__(self, prev_channel: int, input_channel: int, output_channel: int, size: int) -> None:
        super(UNetUpBlock, self).__init__()
        """UNet convolution block with upsampling and skip connection
            Parameters
            ----------
            input_channel: int,
                Number of features in input grid
            output_channel: int,
                Desired number of features for output grid
            prev_channel: int,
                Number of features in grid for skip connection
            size: int,
                Factor to upscale the 3D grid
        """
        self.up_sampling = torch.nn.Upsample(scale_factor=size)
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet(torch.nn.Module):
    def __init__(self, input_channel: int, output_channel: int) -> None:
        super(UNet, self).__init__()
        """3D UNet model with 3D convolutions, downscaling, upsampling and skip connections
            Parameters
            ----------
            input_channel: int,
                Number of features in input grid
            output_channel: int,
                Desired number of features for output grid
        """
        self.down_block1 = UnNetDownBlock(input_channel, 32, 2, False)
        self.down_block2 = UnNetDownBlock(32, 64, 2, True)
        self.down_block3 = UnNetDownBlock(64, 128, 2, True)
        self.down_block4 = UnNetDownBlock(128, 256, 3, True)
        self.max_pool = torch.nn.MaxPool3d(3)

        self.mid_conv1 = torch.nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.mid_conv2 = torch.nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.up_block1 = UNetUpBlock(256, 512, 256, 3)
        self.up_block2 = UNetUpBlock(128, 256, 128, 3)
        self.up_block3 = UNetUpBlock(64, 128, 64, 2)
        self.up_block4 = UNetUpBlock(32, 64, 32, 2)

        self.last_conv1 = torch.nn.Conv3d(32, output_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.max_pool(x4)
        x6 = self.relu(self.mid_conv1(x5))
        x6 = self.relu(self.mid_conv2(x6))
        x = self.up_block1(x4, x6)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)
        x = self.sigmoid(self.last_conv1(x))
        return x


def compute_per_channel_dice(inputs: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute the dice metric for 3D grids
        Parameters
        ----------
        inputs: torch.Tensor,
            3D grid predicted by the model
        target: int,
            Target 3D grid with labels
        epsilon: float,
            smoothing factor
    """
    assert inputs.size() == target.size()

    inputs = torch.flatten(inputs)
    target = torch.flatten(target)
    target = target.float()
    intersect = (inputs * target).sum(-1)

    denominator = (inputs * inputs).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class DiceLoss(torch.nn.Module):
    def __init__(self, epsilon: int = 1e-6):
        super(DiceLoss, self).__init__()
        """Compute the dice loss (1 - dice_metric) for 3D grids
            Parameters
            ----------
            epsilon: float,
                smoothing factor
         """
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_channel_dice = compute_per_channel_dice(inputs, target, self.epsilon)
        return 1. - torch.mean(per_channel_dice)

