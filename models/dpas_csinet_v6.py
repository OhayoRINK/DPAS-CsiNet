"""
DPAS-CsiNet v6
==============
v5 대비 변경 (구조만, 학습 기법 동일):
1. Spatial Attention (CBAM-style 7×7) — "어디에" 집중할지 학습
2. Multi-scale Decoder skip connection
   x_init + 중간 DualPath feature(proj) 모두 최종 출력에 합산
   → 복원 경로 다양화, 후반 진동 억제, gradient flow 개선
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext

img_channels = 2
img_height   = 32
img_width    = 32
feedback_bits_map = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}


class AdaptiveFrequencyGating(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) \
              if x.is_cuda else nullcontext()
        with ctx:
            x_fft = torch.fft.rfft2(x.float())
            mag   = x_fft.abs().mean(dim=(-2, -1))
            gate  = self.gate_fc(mag).unsqueeze(-1).unsqueeze(-1)
        return x * gate.to(dtype=x.dtype)


class SpectralRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_real = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_real   = nn.BatchNorm2d(channels)
        self.bn_imag   = nn.BatchNorm2d(channels)
        self.alpha     = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        orig_dtype = x.dtype
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) \
              if x.is_cuda else nullcontext()
        with ctx:
            x32    = x.float()
            x_fft  = torch.fft.rfft2(x32)
            r      = self.bn_real(F.relu(self.conv_real(x_fft.real.float())))
            im     = self.bn_imag(F.relu(self.conv_imag(x_fft.imag.float())))
            x_fft2 = torch.complex(r.float(), im.float())
            res    = torch.fft.irfft2(x_fft2, s=(x.shape[-2], x.shape[-1]))
            out    = x32 + self.alpha.float() * res
        return out.to(orig_dtype)


class DualPathBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pathA = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pathB = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, img_width),
                      padding=(0, img_width // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.afg  = AdaptiveFrequencyGating(out_ch)

    def forward(self, x):
        a = self.pathA(x)
        b = self.pathB(x)
        if a.shape != b.shape:
            b = b[..., :a.shape[-2], :a.shape[-1]]
        out = self.fuse(torch.cat([a, b], dim=1))
        return self.afg(out)


# ── Channel Attention (v5) ────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


# ── Spatial Attention (v6 추가 ①) ────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# ── Encoder ──────────────────────────────────────────────────
class DPASEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.feature = nn.Sequential(
            DualPathBlock(img_channels, 16),
            DualPathBlock(16, 16),
        )
        self.ca = ChannelAttention(16, reduction=4)
        self.sa = SpatialAttention(kernel_size=7)

        self.aux_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, feedback_bits)
        )
        self.fc = nn.Linear(16 * img_height * img_width, feedback_bits)
        self.feedback_bits = feedback_bits

    def forward(self, x):
        B    = x.shape[0]
        feat = self.feature(x)
        feat = self.ca(feat)
        feat = self.sa(feat)
        aux  = self.aux_proj(feat)
        codeword = torch.sigmoid(self.fc(feat.reshape(B, -1)))
        return codeword, aux


# ── Decoder (v6 추가 ②: multi-scale skip) ────────────────────
class DPASDecoder(nn.Module):
    """
    x_init → dp1(16ch) → srb(16ch) → dp2(2ch) → out
    skip: out += x_init + proj(dp1_feat)
          ↑ 두 경로 동시 합산 → 복원 정보 손실 최소화
    proj: 16ch → 2ch (1×1 conv, 파라미터 32개)
    """
    def __init__(self, feedback_bits):
        super().__init__()
        self.fc   = nn.Linear(feedback_bits,
                              img_channels * img_height * img_width)
        self.dp1  = DualPathBlock(img_channels, 16)
        self.srb  = SpectralRefinementBlock(16)
        self.dp2  = DualPathBlock(16, img_channels)
        # ★ 중간 feature (16ch) → 2ch projection
        self.mid_proj = nn.Conv2d(16, img_channels, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, codeword):
        B      = codeword.shape[0]
        x_init = self.fc(codeword).reshape(B, img_channels, img_height, img_width)
        f1     = self.dp1(x_init)          # (B,16,H,W)
        f2     = self.srb(f1)              # (B,16,H,W)
        out    = self.dp2(f2)              # (B, 2,H,W)
        # multi-scale skip: x_init + projected mid feature
        out    = out + x_init + self.mid_proj(f1)
        return self.sigmoid(out)


# ── Full model ────────────────────────────────────────────────
class DPASCsiNet(nn.Module):
    def __init__(self, feedback_bits=512):
        super().__init__()
        self.encoder = DPASEncoder(feedback_bits)
        self.decoder = DPASDecoder(feedback_bits)

    def forward(self, x):
        codeword, aux = self.encoder(x)
        recon         = self.decoder(codeword)
        return recon, aux


# ── Loss ──────────────────────────────────────────────────────
class ProgressiveFeedbackLoss(nn.Module):
    def __init__(self, aux_weight=0.0, recon_loss_fn=None):
        super().__init__()
        self.aux_weight    = aux_weight
        self.recon_loss_fn = recon_loss_fn if recon_loss_fn else nn.MSELoss()
        self.mse           = nn.MSELoss()

    def forward(self, recon, target, aux, feedback_bits):
        main_loss = self.recon_loss_fn(recon, target)
        if self.aux_weight <= 0:
            return main_loss, float(main_loss.detach().cpu()), 0.0
        target_flat = target.reshape(target.shape[0], -1)
        M           = aux.shape[1]
        aux_target  = target_flat[:, :M].detach()
        aux_loss    = self.mse(aux.sigmoid(), aux_target)
        total       = main_loss + self.aux_weight * aux_loss
        return total, float(main_loss.detach().cpu()), float(aux_loss.detach().cpu())


if __name__ == "__main__":
    for cr, bits in [(4, 512), (8, 256), (16, 128)]:
        m = DPASCsiNet(bits)
        x = torch.randn(2, 2, 32, 32)
        r, a = m(x)
        p = sum(v.numel() for v in m.parameters())
        print(f"CR=1/{cr:<2} bits={bits} | recon={tuple(r.shape)} | params={p:,}")
