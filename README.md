# DPAS-CsiNet

Dual-Path and Adaptive Frequency Gating Based CSI Feedback Compression for Massive MIMO Systems

PyTorch implementation of **DPAS-CsiNet**, a deep learning-based CSI feedback compression model for Massive MIMO FDD systems.

---

## Overview

This repository implements **DPAS-CsiNet (Dual-Path Adaptive Spectral CsiNet)**, proposed for improving CSI feedback compression performance in Massive MIMO FDD systems.

The proposed architecture extends the original CsiNet by introducing:

* **Dual-Path Encoder**
* **Adaptive Frequency Gating (AFG)**
* **Spectral Refinement Block (SRB)**

These modules are designed to better exploit both spatial and spectral structures in angle-delay domain CSI.

---

## Motivation

In Massive MIMO FDD systems, accurate CSI feedback is essential for effective beamforming. However, CSI feedback overhead increases significantly as the number of antennas and subcarriers grows.

Although CsiNet successfully formulates CSI feedback as an autoencoder-based compression problem, the original architecture has several limitations:

* Single-path convolution cannot sufficiently capture long-range delay-domain correlations.
* Spatial-domain convolutions alone cannot adaptively emphasize important frequency components.
* Decoder refinement is limited to spatial-domain processing.

DPAS-CsiNet addresses these issues by combining dual-path feature extraction with spectral-domain processing.

---

## Proposed Architecture

### 1. Dual-Path Encoder

The encoder consists of two parallel convolution paths:

| Path         | Kernel   | Purpose                                    |
| ------------ | -------- | ------------------------------------------ |
| Spatial Path | 3×3 Conv | Local spatial feature extraction           |
| Delay Path   | 1×W Conv | Long-range delay-axis correlation modeling |

The outputs are concatenated and fused using 1×1 convolution.

```text
Input CSI
   ├── 3×3 Conv Path
   └── 1×W Conv Path
            ↓
      Feature Fusion
```

This structure allows the model to simultaneously capture spatial structures and delay-domain correlations in angle-delay CSI.

---

### 2. Adaptive Frequency Gating (AFG)

AFG applies FFT-domain analysis to feature maps and generates adaptive gating weights using frequency magnitude information.

```text
Feature Map → FFT → Magnitude → Gate Generation → Feature Reweighting
```

Unlike conventional channel attention mechanisms, AFG directly utilizes spectral-domain characteristics of CSI.

Key characteristics:

* FFT magnitude-based gating
* Adaptive frequency emphasis
* Compression-ratio-aware feature selection
* Spectral-domain attention mechanism

---

### 3. Spectral Refinement Block (SRB)

SRB refines reconstructed CSI in the spectral domain.

```text
Recovered CSI
      ↓
     FFT
      ↓
Residual Spectral Refinement
      ↓
    IFFT
      ↓
Residual Fusion
```

The SRB compensates for spectral residual errors that are difficult to recover using spatial convolutions alone.

---

## Overall Pipeline

```text
Input CSI (2×32×32)
        ↓
Dual-Path Encoder
        ↓
Adaptive Frequency Gating
        ↓
Latent Codeword
        ↓
Decoder
        ↓
Spectral Refinement Block
        ↓
Recovered CSI
```

---

## Experimental Setup

### Dataset

* COST2100

### Scenarios

* Indoor
* Outdoor

### Input Shape

* `(2, 32, 32)`

### Compression Ratios

* 1/4
* 1/16
* 1/32
* 1/64

### Training Configuration

* Optimizer: Adam
* Epochs: 1000
* Framework: PyTorch

---

## Performance Comparison

### Indoor Scenario

| Compression Ratio | CsiNet NMSE | DPAS-CsiNet NMSE | Improvement |
| ----------------- | ----------- | ---------------- | ----------- |
| 1/4               | -17.36 dB   | -24.72 dB        | +7.36 dB    |
| 1/16              | -8.65 dB    | -13.52 dB        | +4.87 dB    |
| 1/32              | -6.24 dB    | -9.35 dB         | +3.11 dB    |
| 1/64              | -5.84 dB    | -6.64 dB         | +0.80 dB    |

### Outdoor Scenario

| Compression Ratio | CsiNet NMSE | DPAS-CsiNet NMSE | Improvement |
| ----------------- | ----------- | ---------------- | ----------- |
| 1/4               | -8.75 dB    | -15.15 dB        | +6.40 dB    |
| 1/16              | -4.51 dB    | -6.56 dB         | +2.05 dB    |
| 1/32              | -2.81 dB    | -4.37 dB         | +1.56 dB    |
| 1/64              | -1.93 dB    | -2.49 dB         | +0.56 dB    |

The proposed DPAS-CsiNet consistently achieves lower NMSE and higher correlation coefficients than the original CsiNet across all compression ratios.

---

## Installation

### Environment

* Python 3.10+
* CUDA 12.x
* PyTorch

### Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Training

### Synthetic Dataset (Quick Test)

```bash
# Train DPAS-CsiNet
python train.py --model dpas --cr 4 --use_synthetic --epochs 300

# Train baseline CsiNet
python train.py --model csinet --cr 4 --use_synthetic --epochs 300
```

---

### COST2100 Dataset

Download COST2100 dataset and place it inside the `./data/` directory. Can download Python_CsiNet github repository.

```bash
python train.py --model dpas --scenario indoor --cr 4 --epochs 1000
```

## 📁 Repository Structure

```text
DualPath_CsiNet/
├── models/
│   ├── dpas_csinet.py       # Proposed DPAS-CsiNet
│   └── csinet_baseline.py   # Original CsiNet baseline
├── dataset.py               # COST2100 / Synthetic dataset loader
├── train.py                 # Single-model training
├── utils.py                 # Metrics and utilities
├── requirements.txt
└── README.md
```

---

## References

1. C. K. Wen, W. T. Shih, and S. Jin,
   “Deep Learning for Massive MIMO CSI Feedback,”
   *IEEE Wireless Communications Letters*, vol. 7, no. 5, pp. 748–751, 2018.

2. L. Liu et al.,
   “The COST 2100 MIMO Channel Model,”
   *IEEE Wireless Communications*, vol. 19, no. 6, pp. 92–99, 2012.

---

## Paper

**Dual-Path and Adaptive Frequency Gating Based CSI Feedback Compression for Massive MIMO Systems**

* Hanbat National University
* Department of Information and Communication Engineering

---

## Future Work

Planned future extensions include:

* Ablation studies for each module
* FLOPs and parameter efficiency analysis
* Lightweight deployment optimization
* Transformer-based spectral modeling
* Real-world channel generalization experiments

---

