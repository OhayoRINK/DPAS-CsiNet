"""
DPAS-CsiNet v4
===============
목표:
- 원본 DPAS 대비 파라미터 감소
- v3 대비 성능 손실 최소화
- 경량화 + 정보 보존 + 디코더 복원력 강화의 절충안

핵심 아이디어
1) Encoder 경량화는 너무 공격적으로 하지 않음
   - v3: 16x32x32 -> DW(s=2) -> PW(16->8) -> FC(2048->M)
     => 성능 손실이 큼
   - v4: 16x32x32 -> PW(16->8) -> FC(8192->M/2)
          + GAP skip -> FC(16->M/2)
          -> concat -> merge(M->M)
     => 공간 정보는 유지하고, 글로벌 통계 경로를 함께 전달

2) Multi-scale skip fusion encoder
   - main path: spatial-rich compressed feature
   - skip path: global context feature
   - 두 경로를 합쳐 codeword 구성

3) Decoder refinement 강화
   - v2/v3의 skip connection 유지
   - RefineStage를 2회 반복하여 복원력 보강

예상 효과
- 원본 대비 파라미터 대폭 절감
- v3 대비 NMSE 회복
- 원본과 v3 사이의 절충안
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

img_channels = 2
img_height = 32
img_width = 32
feedback_bits_map = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}


class AdaptiveFrequencyGating(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x_fft = torch.fft.rfft2(x.float())
            mag = x_fft.abs().mean(dim=(-2, -1))
            gate = self.gate_fc(mag).unsqueeze(-1).unsqueeze(-1)
        return x * gate.to(dtype=x.dtype)


class SpectralRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_real = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_real = nn.BatchNorm2d(channels)
        self.bn_imag = nn.BatchNorm2d(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        orig_dtype = x.dtype
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x32 = x.float()
            x_fft = torch.fft.rfft2(x32)
            r = self.bn_real(F.relu(self.conv_real(x_fft.real.float())))
            im = self.bn_imag(F.relu(self.conv_imag(x_fft.imag.float())))
            x_fft2 = torch.complex(r.float(), im.float())
            residual = torch.fft.irfft2(x_fft2, s=(x.shape[-2], x.shape[-1]))
            out = x32 + self.alpha.float() * residual
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
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, img_width), padding=(0, img_width // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.afg = AdaptiveFrequencyGating(out_ch)

    def forward(self, x):
        a = self.pathA(x)
        b = self.pathB(x)
        if a.shape != b.shape:
            b = b[..., :a.shape[-2], :a.shape[-1]]
        out = self.fuse(torch.cat([a, b], dim=1))
        return self.afg(out)




class DPASEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.feedback_bits = feedback_bits
        half_bits = feedback_bits // 2
        rem_bits = feedback_bits - half_bits

        self.feature = nn.Sequential(
            DualPathBlock(img_channels, 16),
            DualPathBlock(16, 16),
        )

        self.channel_compress = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.main_fc = nn.Linear(8 * img_height * img_width, half_bits)
        self.skip_fc = nn.Linear(16, rem_bits)
        self.merge_fc = nn.Linear(feedback_bits, feedback_bits)

        self.aux_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, feedback_bits)
        )

    def forward(self, x):
        B = x.shape[0]
        feat = self.feature(x)
        aux = self.aux_proj(feat)

        main_feat = self.channel_compress(feat).reshape(B, -1)
        main_code = self.main_fc(main_feat)

        skip_feat = F.adaptive_avg_pool2d(feat, 1).reshape(B, -1)
        skip_code = self.skip_fc(skip_feat)

        codeword = torch.cat([main_code, skip_code], dim=1)
        codeword = torch.sigmoid(self.merge_fc(codeword))
        return codeword, aux


class DPASDecoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.fc = nn.Linear(feedback_bits, img_channels * img_height * img_width)
        self.dp1 = DualPathBlock(img_channels, 16)
        self.srb = SpectralRefinementBlock(16)
        self.dp2 = DualPathBlock(16, img_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, codeword):
        B = codeword.shape[0]
        x_init = self.fc(codeword).reshape(B, img_channels, img_height, img_width)
        x = self.dp1(x_init)
        x = self.srb(x)
        x = self.dp2(x)
        return self.sigmoid(x + x_init)


class DPASCsiNet(nn.Module):
    def __init__(self, feedback_bits=512):
        super().__init__()
        self.encoder = DPASEncoder(feedback_bits)
        self.decoder = DPASDecoder(feedback_bits)

    def forward(self, x):
        codeword, aux = self.encoder(x)
        recon = self.decoder(codeword)
        return recon, aux


class ProgressiveFeedbackLoss(nn.Module):
    def __init__(self, aux_weight=0.0, recon_loss_fn=None):
        super().__init__()
        self.aux_weight = aux_weight
        self.recon_loss_fn = recon_loss_fn if recon_loss_fn is not None else nn.MSELoss()
        self.mse = nn.MSELoss()

    def forward(self, recon, target, aux, feedback_bits):
        main_loss = self.recon_loss_fn(recon, target)
        if self.aux_weight <= 0:
            return main_loss, float(main_loss.detach().cpu()), 0.0
        target_flat = target.reshape(target.shape[0], -1)
        M = aux.shape[1]
        aux_target = target_flat[:, :M].detach()
        aux_loss = self.mse(aux.sigmoid(), aux_target)
        total = main_loss + self.aux_weight * aux_loss
        return total, float(main_loss.detach().cpu()), float(aux_loss.detach().cpu())


if __name__ == "__main__":
    for cr, bits in [(4, 512), (8, 256), (16, 128), (32, 64), (64, 32)]:
        model = DPASCsiNet(bits)
        x = torch.randn(2, 2, 32, 32)
        recon, aux = model(x)
        total = sum(p.numel() for p in model.parameters())
        print(f"CR=1/{cr:<2} bits={bits:<3} | recon={tuple(recon.shape)} aux={tuple(aux.shape)} | params={total:,}")
