\# DPAS-CsiNet (Dual-Path Adaptive Spectral CsiNet)

&nbsp;

\> \*\*이중 경로 및 적응형 주파수 게이팅 기반 대규모 MIMO CSI 피드백 압축 기법\*\*&nbsp;&nbsp;

\> \*Dual-Path and Adaptive Frequency Gating Based CSI Feedback Compression for Massive MIMO Systems\*

&nbsp;

\---

&nbsp;

\#\# 📌 개요 (Overview)

&nbsp;

대규모 MIMO(Massive MIMO) FDD(Frequency Division Duplex) 시스템에서는 기지국이 최적의 빔포밍 벡터를 형성하기 위해 하향링크 채널 상태 정보(CSI, Channel State Information)를 단말로부터 피드백받아야 합니다. 하지만 송수신 안테나 및 OFDM 부반송파 개수가 증가할수록 CSI 피드백으로 인한 상향링크 오버헤드가 급격히 증가합니다.

&nbsp;

기존 대표적인 딥러닝 기반 압축 기법인 \*\*CsiNet\*\*은 오토인코더 구조를 통해 유의미한 성능을 보였으나, 다음과 같은 구조적 한계점이 존재합니다:

1\. \*\*단일 공간 합성곱 경로\*\*: 인코더가 3x3 단일 합성곱에 의존하여 각도-지연(Angle-Delay) 영역 내 지연축(Delay axis)의 장거리 상관관계를 충분히 추출하지 못함.

2\. \*\*주파수 성분 적응성 부족\*\*: 공간 도메인 중심의 합성곱 연산으로 인해 채널 특성에 따른 주파수 성분별 중요도를 적응적으로 가중하지 못함.

3\. \*\*디코더 내 스펙트럼 잔차 보정 부재\*\*: 복원 후단에서 주파수 영역 구조를 직접 참조하지 않아 잔차(Residual) 성분의 완벽한 복원이 어려움.

&nbsp;

\*\*DPAS-CsiNet\*\*은 이러한 한계를 해결하기 위해 \*\*이중 경로 인코더(Dual-Path Encoder)\*\*, \*\*적응형 주파수 게이팅(Adaptive Frequency Gating, AFG)\*\*, \*\*스펙트럼 정제 블록(Spectral Refinement Block, SRB)\*\*을 제안 및 통합하여 저압축부터 고압축 환경 전 구간에서 우수한 복원 정확도를 달성합니다.

&nbsp;

\---

&nbsp;

\#\# 🏗️ 모델 구조 (Architecture)

&nbsp;

\`\`\`

\[입력 CSI H (2 x 32 x 32)\]

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

┌───────────────────────────────────────────────┐

│ 1\. 이중 경로 인코더 (Dual-Path Encoder)        │

│  \- Path 1: 3x3 Conv (국소 공간 특징 포착)     │

│  \- Path 2: 1xW Conv (지연축 장거리 상관성 포착) │

│  \- 채널 Concat 후 1x1 Conv 융합               │

└───────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

┌───────────────────────────────────────────────┐

│ 2\. 적응형 주파수 게이팅 (AFG Module)          │

│  \- 2D-FFT 진폭(Magnitude) 기반 채널별 게이트   │

│  \- 중요 주파수 성분 강조 & 불필요 성분 감쇄   │

└───────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

┌───────────────────────────────────────────────┐

│ 저차원 압축 코드워드 s (Dense Layer)          │

└───────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│ (피드백 채널 전송)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

┌───────────────────────────────────────────────┐

│ 디코더 기본 복원 (Dense Layer \+ RefineNet)    │

└───────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

┌───────────────────────────────────────────────┐

│ 3\. 스펙트럼 정제 블록 (SRB Module)            │

│  \- FFT 도메인에서 실수/허수부 스펙트럼 보정   │

│  \- IFFT 역변환 후 원본 복원 신호에 잔차 합성  │

└───────────────────────────────────────────────┘

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼

\[최종 복원 CSI H' (2 x 32 x 32)\]

\`\`\`

&nbsp;

\#\#\# 핵심 모듈 상세

1\. \*\*이중 경로 인코더 (Dual-Path Encoder)\*\*

&nbsp;&nbsp;&nbsp;\- 국소 공간 패턴을 학습하는 3x3 합성곱과 지연축 방향의 상관성을 강화하는 1xW 비대칭(Asymmetric) 합성곱을 병렬로 배치하였습니다.

&nbsp;&nbsp;&nbsp;\- 두 경로의 특징맵을 결합(Concatenate)한 뒤 1x1 합성곱으로 융합(Fusion)하여 지연축과 공간축의 특징을 동시에 활용합니다.

2\. \*\*적응형 주파수 게이팅 (Adaptive Frequency Gating, AFG)\*\*

&nbsp;&nbsp;&nbsp;\- 융합된 특징맵을 2차원 FFT 도메인으로 변환한 뒤 진폭(Magnitude) 정보를 바탕으로 게이트 벡터를 도출합니다.

&nbsp;&nbsp;&nbsp;\- 시그모이드 활성화 함수를 통해 채널별 유효 주파수 성분을 증폭하고 노이즈성 성분을 억제합니다.

3\. \*\*스펙트럼 정제 블록 (Spectral Refinement Block, SRB)\*\*

&nbsp;&nbsp;&nbsp;\- 디코더 복원 후단에 위치하여 CSI를 주파수 영역으로 변환하고 잔차 보정을 수행합니다.

&nbsp;&nbsp;&nbsp;\- 역 FFT(IFFT)를 통해 공간 도메인으로 복원 후 덧셈 연산(Residual connection)을 수행하여 세부 스펙트럼 복원 정밀도를 극대화합니다.

&nbsp;

\---

&nbsp;

\#\# 📊 실험 성능 (Performance Comparison)

&nbsp;

COST2100 채널 모델의 Indoor(실내) 및 Outdoor(실외) 시나리오에서 평가된 결과입니다 (2 x 32 x 32 입력 크기, Adam Optimizer, 1000 Epochs 기준).

&nbsp;

\#\#\# 1\. Indoor 환경 (COST2100)

| 압축률 (gamma) | 모델 (Method) | NMSE (dB) ↓ | 성능 개선 폭 | 상관계수 (rho) ↑ |

|:---:|:---|:---:|:---:|:---:|

| \*\*1/4\*\* | CsiNet | \-17.36 | 기준 | 0.9882 |

| | \*\*DPAS-CsiNet\*\* | \*\*-24.72\*\* | \*\*+7.36 dB 향상\*\* | \*\*0.9983\*\* |

| \*\*1/16\*\* | CsiNet | \-8.65 | 기준 | 0.9346 |

| | \*\*DPAS-CsiNet\*\* | \*\*-13.52\*\* | \*\*+4.87 dB 향상\*\* | \*\*0.9776\*\* |

| \*\*1/32\*\* | CsiNet | \-6.24 | 기준 | 0.8860 |

| | \*\*DPAS-CsiNet\*\* | \*\*-9.35\*\* | \*\*+3.11 dB 향상\*\* | \*\*0.9384\*\* |

| \*\*1/64\*\* | CsiNet | \-5.84 | 기준 | 0.8700 |

| | \*\*DPAS-CsiNet\*\* | \*\*-6.64\*\* | \*\*+0.80 dB 향상\*\* | \*\*0.8851\*\* |

&nbsp;

\#\#\# 2\. Outdoor 환경 (COST2100)

| 압축률 (gamma) | 모델 (Method) | NMSE (dB) ↓ | 성능 개선 폭 | 상관계수 (rho) ↑ |

|:---:|:---|:---:|:---:|:---:|

| \*\*1/4\*\* | CsiNet | \-8.75 | 기준 | 0.9141 |

| | \*\*DPAS-CsiNet\*\* | \*\*-15.15\*\* | \*\*+6.40 dB 향상\*\* | \*\*0.9847\*\* |

| \*\*1/16\*\* | CsiNet | \-4.51 | 기준 | 0.7855 |

| | \*\*DPAS-CsiNet\*\* | \*\*-6.56\*\* | \*\*+2.05 dB 향상\*\* | \*\*0.8798\*\* |

| \*\*1/32\*\* | CsiNet | \-2.81 | 기준 | 0.6655 |

| | \*\*DPAS-CsiNet\*\* | \*\*-4.37\*\* | \*\*+1.56 dB 향상\*\* | \*\*0.7908\*\* |

| \*\*1/64\*\* | CsiNet | \-1.93 | 기준 | 0.5933 |

| | \*\*DPAS-CsiNet\*\* | \*\*-2.49\*\* | \*\*+0.56 dB 향상\*\* | \*\*0.6519\*\* |

&nbsp;

\---

&nbsp;

\#\# 💻 설치 및 환경 설정 (Installation)

&nbsp;

\`\`\`bash

git clone OhayoRINK/DPAS-CsiNet repository

cd DPAS-CsiNet

pip install \-r requirements.txt

\`\`\`

&nbsp;

\#\#\# 권장 환경 (Dependencies)

\- Python \>= 3.8

\- PyTorch \>= 1.10.0 (CUDA 환경 권장)

\- SciPy, NumPy, Matplotlib

&nbsp;

\---

&nbsp;

\#\# 🚀 실행 방법 (Usage)

&nbsp;

\#\#\# 1\. 데이터 준비

COST2100 데이터셋(.mat)을 \`data/\` 경로에 저장합니다:

\- Indoor: \`DATA\_Htrainin.mat\`, \`DATA\_Hvalin.mat\`, \`DATA\_Htestin.mat\`

\- Outdoor: \`DATA\_Htrainout.mat\`, \`DATA\_Hvalout.mat\`, \`DATA\_Htestout.mat\`

&nbsp;

\#\#\# 2\. 모델 학습 (Training)

\`\`\`bash

\# 실내 환경, 압축률 1/4 기준

python main.py \--scenario indoor \--cr 4 \--epochs 1000 \--batch\_size 200

&nbsp;

\# 실외 환경, 압축률 1/16 기준

python main.py \--scenario outdoor \--cr 16 \--epochs 1000 \--batch\_size 200

\`\`\`

&nbsp;

\#\#\# 3\. 평가 및 테스트 (Evaluation)

\`\`\`bash

python test.py \--scenario indoor \--cr 4 \--checkpoint checkpoints/best\_model\_cr4.pth

\`\`\`

&nbsp;

\---

&nbsp;

\#\# 📑 인용 및 사사 (Citation & Acknowledgement)

&nbsp;

본 연구는 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원-대학ICT연구센터(ITRC)의 지원을 받아 수행된 연구입니다 (IITP-2026-RS-2024-00437886).

&nbsp;

\`\`\`bibtex

@inproceedings{park2026dpascsinet,

&nbsp;&nbsp;title     \= {Dual-Path and Adaptive Frequency Gating Based CSI Feedback Compression for Massive MIMO Systems},

&nbsp;&nbsp;author    \= {Park, Jinhyeong and Shin, Donghui and Cho, Junyong and Noh, Hyeonho},

&nbsp;&nbsp;booktitle \= {Proceedings of the Korean Institute of Communications and Information Sciences (KICS) Winter Conference},

&nbsp;&nbsp;year      \= {2026}

}

\`\`\`

&nbsp;