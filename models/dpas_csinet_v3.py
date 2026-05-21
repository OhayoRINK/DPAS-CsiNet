"""
DPAS-CsiNet v3 (경량화 + 성능향상 통합 버전)
=============================================
v2 대비 추가 변경사항:

[개선 4] Encoder FC 앞 MobileNet 스타일 경량화
  DW Conv(stride=2) + PW Conv(16→8) 삽입
  → FC 입력: 16×32×32=16384 → 8×16×16=2048 (1/8 축소)
  → 파라미터 약 75~78% 절감 (전 압축률 공통)
  → 공간 위치 정보 보존 + 중복 정보 제거로 정규화 효과

v2에서 유지된 사항:
[개선 1] AFG reduction=4 → 2 (게이팅 표현력 향상)
[개선 2] Decoder Skip Connection (FC 출력 잔차 보존)
[개선 3] MultiScaleBlock (3×3 + 5×5 병렬 Conv)

논문 원본에서 유지된 사항:
- DualPathBlock 구조 (pathA/pathB/fuse/afg)
- SpectralRefinementBlock 구조
- ProgressiveFeedbackLoss 구조
- 학습 하이퍼파라미터 (aux_weight=0.0 기본)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

img_channels = 2
img_height   = 32
img_width    = 32
feedback_bits_map = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Adaptive Frequency Gating  [개선 1: reduction=2]
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveFrequencyGating(nn.Module):
    def __init__(self, channels, reduction=2):   # v2: 4→2
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
            mag   = x_fft.abs().mean(dim=(-2, -1))
            gate  = self.gate_fc(mag).unsqueeze(-1).unsqueeze(-1)
        return x * gate.to(dtype=x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral Refinement Block  [논문 원본과 동일]
# ─────────────────────────────────────────────────────────────────────────────
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
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x32      = x.float()
            x_fft    = torch.fft.rfft2(x32)
            r        = self.bn_real(F.relu(self.conv_real(x_fft.real.float())))
            im       = self.bn_imag(F.relu(self.conv_imag(x_fft.imag.float())))
            x_fft2   = torch.complex(r.float(), im.float())
            residual = torch.fft.irfft2(x_fft2, s=(x.shape[-2], x.shape[-1]))
            out      = x32 + self.alpha.float() * residual
        return out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dual-Path Block  [논문 원본 + AFG reduction=2 자동 적용]
# ─────────────────────────────────────────────────────────────────────────────
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
        self.afg  = AdaptiveFrequencyGating(out_ch)  # reduction=2 적용

    def forward(self, x):
        a = self.pathA(x)
        b = self.pathB(x)
        if a.shape != b.shape:
            b = b[..., :a.shape[-2], :a.shape[-1]]
        out = self.fuse(torch.cat([a, b], dim=1))
        return self.afg(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Scale Block  [v2에서 유지, 개선 3]
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, half, 5, padding=2, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.fuse(torch.cat([self.branch3(x), self.branch5(x)], dim=1))
        return self.act(out + x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Encoder
# [개선 4] FC 앞에 DW(stride=2) + PW(16→8) 삽입으로 파라미터 ~78% 절감
#
# 구조 변화:
#   기존: feature(16×32×32) ─────── FC(16384→M) ──→ codeword
#   v3:   feature(16×32×32) → DW(s=2) → PW(16→8) → FC(2048→M) ──→ codeword
#
# 파라미터 (CR=1/4, M=512):
#   기존 FC: 16×32×32×512 = 8,388,608
#   v3  FC:  8×16×16×512  = 1,048,576  (-87.5%)
# ─────────────────────────────────────────────────────────────────────────────
class DPASEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.feature = nn.Sequential(
            DualPathBlock(img_channels, 16),
            DualPathBlock(16, 16),
        )

        # [개선 4] MobileNet 스타일 공간 압축
        self.compress = nn.Sequential(
            # Depthwise Conv: 공간 32×32 → 16×16 (stride=2), 채널 유지
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1,
                      groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # Pointwise Conv: 채널 16 → 8
            nn.Conv2d(16, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # aux_proj는 compress 이후 feature 기준으로 수정
        self.aux_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, feedback_bits)
        )

        # FC: 8×16×16 = 2048 → feedback_bits
        self.fc            = nn.Linear(8 * (img_height // 2) * (img_width // 2), feedback_bits)
        self.sig           = nn.Identity()
        self.feedback_bits = feedback_bits

    def forward(self, x):
        B         = x.shape[0]
        feat      = self.feature(x)           # (B, 16, 32, 32)
        feat_c    = self.compress(feat)        # (B, 8, 16, 16)
        aux       = self.aux_proj(feat_c)      # (B, M)
        feat_flat = feat_c.reshape(B, -1)      # (B, 8×16×16=2048)
        codeword  = torch.sigmoid(self.fc(feat_flat))
        return codeword, aux


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decoder  [v2 구조 유지: Skip Connection + MultiScaleBlock]
# ─────────────────────────────────────────────────────────────────────────────
class DPASDecoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.fc  = nn.Linear(feedback_bits, img_channels * img_height * img_width)
        self.dp1 = DualPathBlock(img_channels, 16)
        self.msb = MultiScaleBlock(16)
        self.srb = SpectralRefinementBlock(16)
        self.dp2 = DualPathBlock(16, img_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, codeword):
        B      = codeword.shape[0]
        x_init = self.fc(codeword).reshape(B, img_channels, img_height, img_width)
        x = self.dp1(x_init)
        x = self.msb(x)
        x = self.srb(x)
        x = self.dp2(x)
        x = x + x_init      # Skip Connection [개선 2]
        return self.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full DPAS-CsiNet v3
# ─────────────────────────────────────────────────────────────────────────────
class DPASCsiNet(nn.Module):
    def __init__(self, feedback_bits=512):
        super().__init__()
        self.encoder = DPASEncoder(feedback_bits)
        self.decoder = DPASDecoder(feedback_bits)

    def forward(self, x):
        codeword, aux = self.encoder(x)
        recon         = self.decoder(codeword)
        return recon, aux


# ─────────────────────────────────────────────────────────────────────────────
# 8. Progressive Feedback Loss  [논문 원본과 동일]
# ─────────────────────────────────────────────────────────────────────────────
class ProgressiveFeedbackLoss(nn.Module):
    def __init__(self, aux_weight=0.0, recon_loss_fn=None):
        super().__init__()
        self.aux_weight    = aux_weight
        self.recon_loss_fn = recon_loss_fn if recon_loss_fn is not None else nn.MSELoss()
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
    print("=" * 60)
    print(" DPAS-CsiNet v3 구조 검증")
    print("=" * 60)

    results = []
    for cr, bits in [(4, 512), (16, 128), (32, 64), (64, 32)]:
        model  = DPASCsiNet(feedback_bits=bits)
        x      = torch.randn(4, 2, 32, 32)
        recon, aux = model(x)
        assert recon.shape == x.shape,    f"recon shape 불일치"
        assert aux.shape   == (4, bits),  f"aux shape 불일치"
        params = sum(p.numel() for p in model.parameters())
        results.append((cr, bits, params))
        print(f"  CR=1/{cr:<2} | bits={bits:<3} | params={params:>10,} ✓")

    baseline = {4:9_469_685, 16:2_391_413, 32:1_211_701, 64:621_845}
    print()
    print(f"  {'CR':<8} {'기존':>10} {'v3':>10} {'절감율':>8}")
    for cr, bits, p in results:
        b = baseline[cr]
        print(f"  CR=1/{cr:<4} {b:>10,} {p:>10,} {(b-p)/b*100:>7.1f}%")

    print()
    # Encoder compress 구조 확인
    model  = DPASCsiNet(512)
    enc    = model.encoder
    dw_out = enc.compress[0]
    pw_out = enc.compress[3]
    print(f"  Encoder.compress:")
    print(f"    DW Conv: in={dw_out.in_channels}, out={dw_out.out_channels}, "
          f"groups={dw_out.groups}, stride={dw_out.stride}")
    print(f"    PW Conv: in={pw_out.in_channels}, out={pw_out.out_channels}")
    print(f"  Encoder.fc: in={enc.fc.in_features}, out={enc.fc.out_features}")
    print()
    print("모든 검증 통과 ✓")
