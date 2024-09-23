import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class MultiStitchModule(nn.Module):
    def __init__(self, in_features):
        super(MultiStitchModule, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_features, in_features, 3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        color_attention = self.conv1(x)
        texture_attention = self.conv2(x)
        source_attention = self.conv3(x)
        attention = self.softmax(color_attention + texture_attention + source_attention)
        return attention * x

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Multi-stitch module
        model += [MultiStitchModule(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.Conv2d(64, output_nc, 7, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)