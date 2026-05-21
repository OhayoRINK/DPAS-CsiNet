"""
유틸리티: 평가 지표 및 학습 도구
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 평가 지표: 원본 CsiNet 방식에 가까운 centered complex metric
# ─────────────────────────────────────────────────────────────────────────────
def _as_centered_complex(arr):
    """
    CsiNet COST2100 데이터는 보통 real/imag 2채널이 [0, 1]로 저장됩니다.
    원본 CsiNet 평가 코드와 맞추기 위해 평가 시에는 각 채널에서 0.5를 빼서
    복소 CSI H = (real - 0.5) + j(imag - 0.5) 로 복원합니다.

    Input:
        arr: (N, 2, H, W) numpy array, values usually in [0, 1]
    Return:
        complex array: (N, H, W)
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2, H, W), got {arr.shape}")
    return (arr[:, 0] - 0.5) + 1j * (arr[:, 1] - 0.5)


def nmse(recon, target):
    """
    원본 CsiNet 스타일 NMSE(dB) 계산.

    기존 코드처럼 [0,1] 값을 그대로 complex로 보지 않고,
    real/imag 채널을 각각 0.5-centered 값으로 되돌린 뒤 계산합니다.
    이렇게 해야 CsiNet 원 코드의 COST2100 평가 방식과 비교가 가능합니다.

    recon, target: (N, 2, H, W) numpy array
    return: scalar float, dB. 더 낮고 더 음수일수록 좋음.
    """
    r_complex = _as_centered_complex(recon)
    t_complex = _as_centered_complex(target)

    err = r_complex - t_complex
    num = np.sum(np.abs(err) ** 2, axis=(-2, -1))
    den = np.sum(np.abs(t_complex) ** 2, axis=(-2, -1))
    nmse_linear = np.mean(num / (den + 1e-12))
    return float(10 * np.log10(nmse_linear + 1e-12))


def nmse_raw(recon, target):
    """
    참고용 raw NMSE(dB).
    [0,1] 값을 그대로 complex로 보고 계산합니다.
    보고서/논문 비교에는 nmse() centered metric 사용을 권장합니다.
    """
    recon = np.asarray(recon, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r_complex = recon[:, 0] + 1j * recon[:, 1]
    t_complex = target[:, 0] + 1j * target[:, 1]
    err = r_complex - t_complex
    num = np.sum(np.abs(err) ** 2, axis=(-2, -1))
    den = np.sum(np.abs(t_complex) ** 2, axis=(-2, -1))
    nmse_linear = np.mean(num / (den + 1e-12))
    return float(10 * np.log10(nmse_linear + 1e-12))


def rho_metric(recon, target):
    """
    원본 CsiNet 스타일 correlation coefficient ρ 계산.
    평가 시 real/imag 채널에서 0.5를 빼서 centered complex CSI로 변환합니다.

    recon, target: (N, 2, H, W) numpy array
    return: scalar float, 클수록 좋음.
    """
    r_complex = _as_centered_complex(recon).reshape(len(recon), -1)
    t_complex = _as_centered_complex(target).reshape(len(target), -1)

    num = np.abs(np.sum(np.conj(t_complex) * r_complex, axis=-1))
    den = (
        np.sqrt(np.sum(np.abs(r_complex) ** 2, axis=-1))
        * np.sqrt(np.sum(np.abs(t_complex) ** 2, axis=-1))
    )
    rho = np.mean(num / (den + 1e-12))
    return float(np.clip(rho, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch 학습 손실: 평가 metric과 맞춘 centered NMSE loss
# ─────────────────────────────────────────────────────────────────────────────
def centered_nmse_loss_torch(recon, target, eps=1e-8):
    """
    recon/target: torch.Tensor (B, 2, 32, 32), values in [0,1]
    원본 CsiNet NMSE와 같은 centered complex 해석을 학습 loss에 사용합니다.
    반환값은 linear NMSE 평균이며, log10은 학습 안정성을 위해 적용하지 않습니다.
    """
    import torch
    rr = recon[:, 0] - 0.5
    ri = recon[:, 1] - 0.5
    tr = target[:, 0] - 0.5
    ti = target[:, 1] - 0.5
    num = torch.sum((rr - tr) ** 2 + (ri - ti) ** 2, dim=(-2, -1))
    den = torch.sum(tr ** 2 + ti ** 2, dim=(-2, -1))
    return torch.mean(num / (den + eps))


def raw_mse_loss_torch(recon, target):
    import torch.nn.functional as F
    return F.mse_loss(recon, target)


# ─────────────────────────────────────────────────────────────────────────────
# 학습 도구
# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


class EarlyStopping:
    """검증 지표가 개선되지 않으면 학습 중단"""
    def __init__(self, patience=50, min_delta=1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def __call__(self, metric):
        """metric이 클수록 좋다고 가정 (NMSE의 경우 -NMSE를 넣을 것)"""
        if self.best is None or metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# 복소수 채널 재구성 (평가용)
# ─────────────────────────────────────────────────────────────────────────────
def to_complex(arr):
    """(N, 2, H, W) → centered complex CSI. 원본 CsiNet 평가 방식과 동일하게 0.5를 뺍니다."""
    return _as_centered_complex(arr)


def print_result_table(results):
    """결과를 논문 테이블 형식으로 출력"""
    print("\n" + "="*65)
    print(f"{'Method':<20} {'CR':>6} {'Scenario':>10} {'NMSE(dB)':>10} {'ρ':>8}")
    print("-"*65)
    for r in results:
        cr_str = f"1/{r['cr']}"
        print(f"{r['model']:<20} {cr_str:>6} {r['scenario']:>10} "
              f"{r['test_nmse_dB']:>10.2f} {r['test_rho']:>8.4f}")
    print("="*65)
