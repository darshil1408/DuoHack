import torch
import torch.nn as nn


class CALayer(nn.Module):
    """
    Channel Attention Layer.

    Learns which feature channels are important and
    reweights them accordingly.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention = self.body(self.pool(x))
        return x * attention


class ResidualCABlock(nn.Module):
    """
    Residual block with Channel Attention.

    The residual connection helps preserve useful
    low-level image information while learning restoration.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            CALayer(channels, reduction),
        )

    def forward(self, x):
        return x + 0.2 * self.body(x)


class RestorationSRNet(nn.Module):
    """
    Compact residual image restoration + x2 super-resolution network.

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]

    The network does not normalize or clip the input.

    This is important because the KLA NoisyLR data can
    contain values below 0 and above 1.

    The final prediction is clamped to [0, 1] because
    the ground-truth images are in that range.
    """

    def __init__(self, channels=64, blocks=12):
        super().__init__()

        # Initial feature extraction
        self.head = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1
        )

        # Residual feature extraction
        self.body = nn.Sequential(
            *[
                ResidualCABlock(channels)
                for _ in range(blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # 2x spatial upsampling
        self.up = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),
        )

        # Final image reconstruction
        self.tail = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        shallow = self.head(x)

        deep = self.body(shallow)

        deep = self.body_conv(deep) + shallow

        up = self.up(deep)

        output = self.tail(up)

        return output.clamp(0.0, 1.0)


def count_parameters(model):
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
