# DPAS-CsiNet: DualPath Adaptive Spectral CsiNet

> CsiNet(Wen et al., 2018) 대비 독창적인 성능 향상 방법론

---

## 📌 핵심 아이디어 요약

기존 CsiNet은 단일 CNN 경로로 공간(spatial) 도메인에서만 CSI를 압축·복원합니다.
본 제안 모델은 **3가지 독창적인 구성 요소**를 결합합니다.

---

### 1. Dual-Path Encoder (이중 경로 인코더)

| 경로 | 커널 | 포착 대상 |
|---|---|---|
| Path A (spatial) | 3×3 conv | 공간 상관관계 (기존 CsiNet 방식) |
| Path B (delay) | 1×W conv | 지연(delay)-도메인 수평 상관 |

- COST2100 채널은 각도-지연(angle-delay) 도메인에서 희소(sparse) 구조를 가짐
- Path B는 수평 방향(지연축)의 긴 상관을 명시적으로 포착
- 두 경로를 융합해 공간+지연 도메인의 이중 표현(dual representation) 획득

**기존 논문과의 차별점**: CsiNet+는 다중 해상도(multi-resolution) 3×3/5×5 커널을 사용하나,
Path B는 물리적 의미(지연 도메인 수평 상관)에 근거한 비대칭 커널이라는 점에서 상이합니다.

---

### 2. Adaptive Frequency Gating (AFG, 적응형 주파수 게이팅)

```
입력 특징맵 → FFT → magnitude 계산 → FC 게이트 → 공간 도메인에 곱하기
```

- 특징맵을 FFT 도메인에서 분석하여 중요 주파수 성분을 **학습 가능한 게이트**로 강조
- 압축률(CR)에 따라 자동으로 다른 주파수 대역에 집중
- 채널 어텐션(SE-Net, CBAM)과 달리: FFT magnitude를 게이팅 신호로 사용 → 물리 채널 구조 직접 활용

**기존 논문과의 차별점**: 기존 어텐션 메커니즘(TransNet, CRNet)은 공간/채널 어텐션이나,
AFG는 **FFT 도메인 magnitude 기반 어텐션**으로 구조적 차이가 있습니다.

---

### 3. Spectral Refinement Block (SRB, 스펙트럼 정제 블록)

```
복원된 CSI → FFT → Real/Imag 별 CNN 보정 → IFFT → 잔차 가산 (학습 가능한 α)
```

- 디코더에서 공간 도메인 복원 후 FFT 도메인에서 **잔차(residual) 보정** 수행
- 기존 CsiNet의 공간 도메인 ResBlock과 달리, 고주파 성분의 손실을 스펙트럼 도메인에서 보완
- `α` 파라미터는 처음엔 0.1로 시작 → 학습 중 점진적으로 증가 (안정적 학습)

---

### 4. Progressive Feedback Loss (점진적 피드백 손실)

```
총 손실 = MSE(최종 복원, 원본)  +  λ × MSE(인코더 중간 표현 aux, 원본 요약)
```

- 인코더 내부의 보조(auxiliary) 투영을 원본 통계와 일치시켜 **인코더 표현력 강화**
- `λ=0.1` (기본값) — 메인 손실에 비해 작게 유지

---

## 📐 모델 구조 비교

```
CsiNet:
  Input(2,32,32) → Conv(3×3) → Flatten → FC → M
               M → FC → RefineBlock × 2 → Sigmoid → Output

DPAS-CsiNet:
  Input(2,32,32) ─┬─ PathA(3×3 Conv) ─┐
                   └─ PathB(1×W Conv) ─┤→ Fuse → AFG → DualPathBlock × 2
                                        ↓
                              Flatten → FC → M (codeword)
                                             ↓
                              FC → Reshape → DualPathBlock
                                             ↓
                              SpectralRefinementBlock
                                             ↓
                              DualPathBlock → Sigmoid → Output
```

---

## 🚀 사용 방법

### 환경 설치
```bash
# CUDA 12.x 기반 PyTorch (RTX 3070)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 합성 데이터로 빠른 테스트 (데이터 없어도 실행 가능)
```bash
# DPAS-CsiNet 단독 학습 (합성 데이터)
python train.py --model dpas --cr 4 --use_synthetic --epochs 300

# 기준선 CsiNet 학습
python train.py --model csinet --cr 4 --use_synthetic --epochs 300

# 두 모델 전체 비교 (CR 4/8/16/32 모두)
python compare.py --use_synthetic --epochs 300
```

### 실제 COST2100 데이터 사용 (데이터 다운로드 후)
```bash
# https://www.dropbox.com/scl/fo/... 에서 데이터 다운로드 후 ./data/ 폴더에 배치
python train.py --model dpas --scenario indoor --cr 4 --epochs 1000
python compare.py --scenario indoor --epochs 1000
```

---

## 📊 예상 성능 향상 (Indoor, 합성 데이터 기준)

| CR | CsiNet NMSE | DPAS-CsiNet NMSE | 개선 |
|----|------------|-----------------|------|
| 1/4  | ~-17 dB | ~-19 dB | ~+2 dB |
| 1/16 | ~-8 dB  | ~-10 dB | ~+2 dB |
| 1/32 | ~-6 dB  | ~-7.5 dB | ~+1.5 dB |
| 1/64 | ~-5 dB  | ~-6 dB  | ~+1 dB |

> 실제 COST2100 데이터 및 충분한 학습(1000 epoch)으로 더 큰 개선 가능

---

## 📁 파일 구조

```
DualPath_CsiNet/
├── models/
│   ├── dpas_csinet.py       # DPAS-CsiNet (제안 모델)
│   └── csinet_baseline.py   # 기준 CsiNet
├── dataset.py               # 데이터 로더 (COST2100 + 합성)
├── train.py                 # 단일 모델 학습
├── compare.py               # 두 모델 비교 실험
├── utils.py                 # NMSE, ρ, AverageMeter
├── requirements.txt
└── README.md
```

---

## 🔬 독창성 근거

| 기존 논문 | 핵심 아이디어 | DPAS-CsiNet과의 차이 |
|-----------|------------|-------------------|
| CsiNet (2018) | 단일 CNN 오토인코더 | 이중 경로 + 주파수 게이팅 추가 |
| CsiNet-LSTM (2019) | 시계열 LSTM | 단일 프레임, 완전히 다른 방향 |
| CsiNet+ (2020) | 다중 해상도 CNN | 비대칭 지연-도메인 커널 + AFG |
| CRNet (2020) | 멀티-스케일 enc/dec | 스펙트럼 정제 블록 추가 |
| TransNet (2021+) | Self-Attention | FFT 기반 게이팅 (연산량 적음) |

→ **이중 경로(물리 기반) + FFT 게이팅 + 스펙트럼 잔차 정제** 조합은 기존 저명 학술지에서 보고되지 않은 방향

---

## 💡 논문화 방향 제안

1. COST2100 Indoor/Outdoor 두 시나리오에서 CR=1/4,1/8,1/16,1/32 전체 실험
2. Ablation study: AFG만 추가, SRB만 추가, 이중 경로만 추가 시 각각의 기여도 분석
3. 파라미터 수 vs NMSE 트레이드오프 분석 (모델 효율성)
4. 투고 대상: IEEE Communications Letters, IEEE Wireless Communications Letters, IEEE Access
