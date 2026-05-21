"""
DualPath Adaptive Spectral CsiNet (DPAS-CsiNet)
================================================
독창적 아이디어:
  1. Dual-Path Encoder: 공간(spatial) 경로 + 주파수(spectral) 경로를 병렬로 처리
     → CsiNet은 단일 CNN 경로만 사용. 각 경로가 서로 다른 물리적 특성을 포착.
  2. Adaptive Frequency Gating (AFG): 압축률(CR)에 따라 인코더가 스스로 중요한
     주파수 성분에 가중치를 부여 (학습 가능한 게이팅). Attention과 달리,
     물리 채널의 지연-각도(delay-angle) 도메인 구조를 활용.
  3. Spectral Refinement Block (SRB): 디코더에서 FFT 도메인에서의 잔차 보정을
     학습 → 기존 공간 도메인 복원의 고주파 손실 보완.
  4. Progressive Feedback Loss: 단순 MSE 대신, 인코딩 중간 표현과 최종 복원
     모두에 손실을 적용 (Auxiliary loss) → 인코더의 표현력 강화.

References avoided (to be novel):
  - CsiNet (Wen 2018): 기준선
  - CsiNet-LSTM (Wang 2019): 시계열 확장
  - CsiNet+ (Guo 2020): 다중 해상도 CNN
  - TransNet / SwinCFNet: Transformer 기반
  - CRNet: 멀티-레졸루션
  → 이 모델은 "주파수 이중 경로 + 적응형 게이팅"의 조합으로 차별화.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext

# ─────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 (논문 기본 세팅과 동일)
# ─────────────────────────────────────────────────────────────────────────────
img_channels = 2   # real/imag
img_height   = 32  # 안테나 수
img_width    = 32  # 서브캐리어 (2×Nc/4 = 32, 지연 도메인)
feedback_bits_map = {4: 512, 8: 256, 16: 128, 32: 64}  # gamma → M


# ─────────────────────────────────────────────────────────────────────────────
# 1. Adaptive Frequency Gating (AFG)
#    - 인코딩된 벡터에서 각 frequency bin의 중요도를 동적으로 학습
#    - CR에 따라 다른 게이팅 패턴을 자동 생성
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveFrequencyGating(nn.Module):
    """
    입력 특징 맵을 FFT 도메인에서 게이팅.
    채널 어텐션과 달리, 실제 DFT 변환 후 magnitude 기반 게이팅 적용.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # channels=2인 decoder 후단에서도 hidden dim이 0이 되지 않게 보정합니다.
        # 이 처리가 없으면 PyTorch에서 "Initializing zero-element tensors is a no-op" 경고가 납니다.
        hidden = max(1, channels // reduction)
        self.gate_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        # AMP 사용 시 FFT가 complex half로 들어가는 것을 피하기 위해
        # FFT와 gate 생성 구간은 float32/autocast off로 고정합니다.
        ctx = torch.amp.autocast(device_type=x.device.type, enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x_fft = torch.fft.rfft2(x.float())          # (B, C, H, W//2+1) complex64
            mag   = x_fft.abs().mean(dim=(-2, -1))      # (B, C), float32
            gate  = self.gate_fc(mag).unsqueeze(-1).unsqueeze(-1)
        return x * gate.to(dtype=x.dtype)   # 공간 도메인에 게이트 적용


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral Refinement Block (SRB)
#    - 복원된 CSI에 FFT 도메인에서 잔차(residual)를 추가로 학습
#    - 고주파 성분 손실 보완
# ─────────────────────────────────────────────────────────────────────────────
class SpectralRefinementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 실수·허수 성분을 concat해서 처리
        self.conv_real = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_imag = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_real   = nn.BatchNorm2d(channels)
        self.bn_imag   = nn.BatchNorm2d(channels)
        self.alpha     = nn.Parameter(torch.tensor(0.1))  # 학습 가능한 잔차 스케일

    def forward(self, x):
        # x: (B, C, H, W)
        # AMP 사용 시 complex half 경고/연산 미지원 문제를 피하기 위해
        # FFT 및 complex 연산 구간은 float32로 고정합니다.
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dual-Path Feature Extractor
#    - Path A: 공간 도메인 CNN (기존 CsiNet 스타일, 3×3 conv)
#    - Path B: 지연 도메인 CNN (1×W 수평 conv → 안테나 상관 무시, 지연 포착)
#    - 두 경로 융합 후 AFG 적용
# ─────────────────────────────────────────────────────────────────────────────
class DualPathBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Path A: spatial
        self.pathA = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Path B: delay-domain (horizontal, captures delay-domain correlations)
        self.pathB = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, img_width), padding=(0, img_width//2),
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.afg  = AdaptiveFrequencyGating(out_ch)

    def forward(self, x):
        a = self.pathA(x)
        b = self.pathB(x)
        # 크기 맞추기 (pathB의 padding 때문에 약간 다를 수 있음)
        if a.shape != b.shape:
            b = b[..., :a.shape[-2], :a.shape[-1]]
        out = self.fuse(torch.cat([a, b], dim=1))
        out = self.afg(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Encoder
# ─────────────────────────────────────────────────────────────────────────────
class DPASEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.feature = nn.Sequential(
            DualPathBlock(img_channels, 16),
            DualPathBlock(16, 16),
        )
        # 보조 손실을 위한 중간 투영 (Auxiliary)
        self.aux_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, feedback_bits)
        )
        # 메인 인코딩
        # 중요: 기존 버전은 feature를 계산하고도 codeword에는 원본 x_flat만 넣어서
        # Dual-Path/AFG encoder가 실제 feedback codeword에 반영되지 않았습니다.
        # 여기서는 DualPathBlock의 출력 feature를 직접 압축합니다.
        self.fc = nn.Linear(16 * img_height * img_width, feedback_bits)
        # Sigmoid → [0,1] codeword
        self.sig = nn.Identity()  # CsiNet-style linear feedback codeword

        # 잔차 경로: 원본 flatten → feedback_bits
        self.feedback_bits = feedback_bits

    def forward(self, x):
        B = x.shape[0]
        feat = self.feature(x)           # (B, 16, H, W)
        # Aux loss input
        aux  = self.aux_proj(feat)       # (B, M)
        # Main: Dual-Path/AFG feature를 feedback codeword로 압축
        # 기존처럼 원본 x를 바로 flatten하면 제안 encoder 모듈이 reconstruction에 기여하지 않습니다.
        feat_flat = feat.reshape(B, -1)       # (B, 16*H*W)
        codeword = 0.5 + 0.5 * torch.tanh(self.fc(feat_flat))  # (B, M)
        return codeword, aux


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decoder
# ─────────────────────────────────────────────────────────────────────────────
class DPASDecoder(nn.Module):
    def __init__(self, feedback_bits):
        super().__init__()
        self.fc = nn.Linear(feedback_bits, img_channels * img_height * img_width)
        # Refinement blocks
        self.refine = nn.Sequential(
            DualPathBlock(img_channels, 16),
            SpectralRefinementBlock(16),
            DualPathBlock(16, img_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, codeword):
        B = codeword.shape[0]
        x = self.fc(codeword)
        x = x.reshape(B, img_channels, img_height, img_width)
        x = self.refine(x)
        x = self.sigmoid(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full DPAS-CsiNet
# ─────────────────────────────────────────────────────────────────────────────
class DPASCsiNet(nn.Module):
    def __init__(self, feedback_bits=512):
        super().__init__()
        self.encoder = DPASEncoder(feedback_bits)
        self.decoder = DPASDecoder(feedback_bits)

    def forward(self, x):
        codeword, aux = self.encoder(x)
        recon = self.decoder(codeword)
        return recon, aux


# ─────────────────────────────────────────────────────────────────────────────
# 7. Progressive Feedback Loss
#    - Main: MSE(복원, 원본)
#    - Aux:  MSE(중간 표현의 통계, 원본의 요약 통계)
#    → 인코더 표현력 강화
# ─────────────────────────────────────────────────────────────────────────────
class ProgressiveFeedbackLoss(nn.Module):
    def __init__(self, aux_weight=0.0, recon_loss_fn=None):
        super().__init__()
        self.aux_weight = aux_weight
        self.recon_loss_fn = recon_loss_fn if recon_loss_fn is not None else nn.MSELoss()
        self.mse = nn.MSELoss()

    def forward(self, recon, target, aux, feedback_bits):
        main_loss = self.recon_loss_fn(recon, target)

        # 기존 aux_target = sigmoid(target_flat[:, :M])는 물리적 의미가 약하고
        # reconstruction 학습을 방해할 수 있어 기본 aux_weight=0으로 둡니다.
        if self.aux_weight <= 0:
            aux_loss = torch.zeros((), device=recon.device, dtype=main_loss.dtype)
            return main_loss, float(main_loss.detach().cpu()), 0.0

        target_flat = target.reshape(target.shape[0], -1)
        M = aux.shape[1]
        aux_target = target_flat[:, :M].detach()  # [0,1] target slice, sigmoid 재적용 금지
        aux_loss = self.mse(aux.sigmoid(), aux_target)
        total = main_loss + self.aux_weight * aux_loss
        return total, float(main_loss.detach().cpu()), float(aux_loss.detach().cpu())


if __name__ == "__main__":
    # 빠른 동작 확인
    model = DPASCsiNet(feedback_bits=512)
    x = torch.randn(4, 2, 32, 32)
    recon, aux = model(x)
    print(f"Input: {x.shape}, Recon: {recon.shape}, Aux: {aux.shape}")
    criterion = ProgressiveFeedbackLoss()
    loss, ml, al = criterion(recon, x.sigmoid(), aux, 512)
    print(f"Total loss={loss:.4f}, Main={ml:.4f}, Aux={al:.4f}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
