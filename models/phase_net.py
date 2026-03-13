"""
U-Net based phase retrieval network.
Used as a deep prior for untrained phase retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial sizes don't match
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PhaseUNet(nn.Module):
    """
    U-Net that maps amplitude spectrum (or noise) to phase spectrum.

    In Deep Prior mode:
        - Input: random noise (1xHxW)
        - Output: phase image (1xHxW), values in [-pi, pi]
        - Network is optimized so that FFT(sigmoid_output) amplitude ≈ target amplitude

    In supervised mode (if pre-trained weights available):
        - Input: amplitude spectrum image (1xHxW)
        - Output: phase spectrum image (1xHxW)
    """

    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()

        self.down1 = DownBlock(in_channels, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, dropout=0.3)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        b = self.bottleneck(x4)

        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # Output in [-pi, pi] using tanh * pi
        return torch.tanh(self.out_conv(x)) * torch.pi


class PhaseRetrievalNet(nn.Module):
    """
    Supervised phase retrieval network.
    Input: log-normalized amplitude spectrum
    Output: phase spectrum in [-pi, pi]
    """

    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bridge
        self.bridge = ConvBlock(base_ch * 8, base_ch * 16, dropout=0.5)

        # Decoder
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.final = nn.Conv2d(base_ch, out_channels, 1)

        # Residual connection from input to output (helps phase alignment)
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bridge(self.pool(s4))

        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)

        out = self.final(d)
        res = self.residual(x)

        return torch.tanh(out + res) * torch.pi
