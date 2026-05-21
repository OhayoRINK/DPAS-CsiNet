"""
Original CsiNet (Wen et al., 2018) - PyTorch 재구현
비교 기준선(baseline)으로 사용
"""

import torch
import torch.nn as nn

img_channels = 2
img_height   = 32
img_width    = 32


class CsiNetEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * img_height * img_width, feedback_bits),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


class RefineBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.LeakyReLU(0.1))
        self.conv3 = nn.Conv2d(16, img_channels, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(img_channels)

    def forward(self, x):
        r = self.conv1(x)
        r = self.conv2(r)
        r = self.bn3(self.conv3(r))
        return torch.relu(x + r)


class CsiNetDecoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.fc = nn.Linear(feedback_bits, img_channels * img_height * img_width)
        self.refine1 = RefineBlock()
        self.refine2 = RefineBlock()
        self.sigmoid = nn.Sigmoid()

    def forward(self, codeword):
        x = self.fc(codeword)
        x = x.reshape(x.size(0), img_channels, img_height, img_width)
        x = self.refine1(x)
        x = self.refine2(x)
        return self.sigmoid(x)


class CsiNet(nn.Module):
    def __init__(self, feedback_bits=512):
        super().__init__()
        self.encoder = CsiNetEncoder(feedback_bits)
        self.decoder = CsiNetDecoder(feedback_bits)

    def forward(self, x):
        codeword = self.encoder(x)
        recon    = self.decoder(codeword)
        return recon


if __name__ == "__main__":
    model = CsiNet(feedback_bits=512)
    x = torch.randn(4, 2, 32, 32)
    out = model(x)
    print(f"CsiNet | Input: {x.shape}, Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
