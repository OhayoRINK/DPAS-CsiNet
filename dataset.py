"""
COST2100 채널 데이터셋 로더
원본 CsiNet과 동일한 data/*.mat 파일 구성을 명시적으로 사용합니다.

기대 파일 구조:
  data/DATA_Htrainin.mat   key: HT   indoor train
  data/DATA_Hvalin.mat     key: HT   indoor validation
  data/DATA_Htestin.mat    key: HT   indoor test
  data/DATA_Htrainout.mat  key: HT   outdoor train
  data/DATA_Hvalout.mat    key: HT   outdoor validation
  data/DATA_Htestout.mat   key: HT   outdoor test

주의:
  A32.mat, A64.mat, A128.mat, A512.mat은 sensing matrix 파일입니다.
  이 로더는 해당 파일들을 학습 CSI 데이터로 읽지 않습니다.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio


DATA_DIR = "./data"


# ─────────────────────────────────────────────────────────────────────────────
# .mat → (N, 2, 32, 32) 변환 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _select_mat_array(mat_dict, filename):
    """MATLAB 메타 키를 제외하고 실제 CSI 배열을 선택합니다."""
    # 원본 CsiNet train/val/test 파일은 HT를 사용합니다.
    preferred_keys = ["HT", "HF_all", "H", "Hdata", "H_train", "H_test"]
    for key in preferred_keys:
        if key in mat_dict:
            return mat_dict[key], key

    valid = [
        (k, v) for k, v in mat_dict.items()
        if not k.startswith("__") and isinstance(v, np.ndarray)
    ]
    if len(valid) == 1:
        return valid[0][1], valid[0][0]

    keys = [k for k, _ in valid]
    raise KeyError(
        f"{filename}에서 CSI 배열 키를 찾지 못했습니다. keys={keys}. "
        "원본 CsiNet 데이터는 보통 'HT' 키를 사용합니다."
    )


def _to_channels_first_2ch(H, filename):
    """여러 COST2100 저장 형식을 (N, 2, 32, 32) float32로 변환합니다."""
    H = np.asarray(H)

    # 원본 CsiNet DATA_H*.mat: 보통 (N, 2048), 실수형, 이미 real/imag 2채널이 펼쳐진 형태
    if H.ndim == 2 and H.shape[1] == 2048 and not np.iscomplexobj(H):
        x = H.astype(np.float32).reshape(-1, 2, 32, 32)

    # 복소수 벡터형: (N, 1024) complex → real/imag 분리
    elif H.ndim == 2 and H.shape[1] == 1024 and np.iscomplexobj(H):
        N = H.shape[0]
        x = np.stack(
            [np.real(H).reshape(N, 32, 32), np.imag(H).reshape(N, 32, 32)],
            axis=1,
        ).astype(np.float32)

    # 복소수 이미지형: (N, 32, 32) complex → real/imag 분리
    elif H.ndim == 3 and H.shape[1:] == (32, 32) and np.iscomplexobj(H):
        x = np.stack([np.real(H), np.imag(H)], axis=1).astype(np.float32)

    # 이미 channels-first: (N, 2, 32, 32)
    elif H.ndim == 4 and H.shape[1:] == (2, 32, 32):
        x = H.astype(np.float32)

    # channels-last: (N, 32, 32, 2)
    elif H.ndim == 4 and H.shape[1:] == (32, 32, 2):
        x = np.transpose(H, (0, 3, 1, 2)).astype(np.float32)

    # 마지막 안전장치: 전체 원소 수가 2*32*32로 나누어떨어지면 reshape 시도
    elif H.size % (2 * 32 * 32) == 0 and not np.iscomplexobj(H):
        x = H.astype(np.float32).reshape(-1, 2, 32, 32)

    else:
        raise ValueError(
            f"지원하지 않는 데이터 shape입니다: {filename}, shape={H.shape}, "
            f"complex={np.iscomplexobj(H)}"
        )

    # CsiNet 계열 모델 decoder가 sigmoid이므로 입력도 [0, 1] 범위가 안전합니다.
    # 원본 DATA_H*.mat이 이미 [0,1]이면 그대로 둡니다.
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin < -1e-6 or xmax > 1.0 + 1e-6:
        x = (x - xmin) / (xmax - xmin + 1e-8)

    return x.astype(np.float32)


def _load_one_mat(filename):
    path = os.path.join(DATA_DIR, filename)
    # Windows에서 확장자 숨김/누락으로 DATA_Htrainin처럼 저장된 경우도 허용합니다.
    if not os.path.isfile(path) and filename.endswith(".mat"):
        alt_path = os.path.join(DATA_DIR, filename[:-4])
        if os.path.isfile(alt_path):
            path = alt_path
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} 파일이 없습니다. data 폴더에 COST2100 .mat 파일을 넣어주세요."
        )

    data = sio.loadmat(path)
    H, key = _select_mat_array(data, filename)
    x = _to_channels_first_2ch(H, filename)
    print(
        f"[Load] {path} | key={key} | shape={x.shape} | "
        f"range=[{x.min():.4f}, {x.max():.4f}]"
    )
    return x


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드: 원본 CsiNet train/val/test split 사용
# ─────────────────────────────────────────────────────────────────────────────
def load_cost2100(scenario="indoor"):
    """
    scenario: 'indoor' or 'outdoor'
    Returns:
        x_train, x_val, x_test: numpy arrays (N, 2, 32, 32)
    """
    if scenario not in ["indoor", "outdoor"]:
        raise ValueError("scenario는 'indoor' 또는 'outdoor'만 가능합니다.")

    if scenario == "indoor":
        train_file = "DATA_Htrainin.mat"
        val_file = "DATA_Hvalin.mat"
        test_file = "DATA_Htestin.mat"
        fallback_all_file = "DATA_HtestFin_all.mat"
    else:
        train_file = "DATA_Htrainout.mat"
        val_file = "DATA_Hvalout.mat"
        test_file = "DATA_Htestout.mat"
        fallback_all_file = "DATA_HtestFout_all.mat"

    split_files = [train_file, val_file, test_file]
    split_paths = [os.path.join(DATA_DIR, f) for f in split_files]

    # 1순위: 원본 CsiNet의 train/val/test 파일을 명시적으로 사용
    if all(os.path.isfile(p) for p in split_paths):
        x_train = _load_one_mat(train_file)
        x_val = _load_one_mat(val_file)
        x_test = _load_one_mat(test_file)

    # 2순위: split 파일이 없고 *_all 파일만 있을 때 8:1:1로 분할
    elif os.path.isfile(os.path.join(DATA_DIR, fallback_all_file)):
        print(
            f"[Warn] {split_files} 중 일부가 없습니다. "
            f"{fallback_all_file} 하나를 8:1:1로 분할합니다."
        )
        x_all = _load_one_mat(fallback_all_file)
        N = x_all.shape[0]
        n_train = int(N * 0.8)
        n_val = int(N * 0.1)
        x_train = x_all[:n_train]
        x_val = x_all[n_train:n_train + n_val]
        x_test = x_all[n_train + n_val:]

    else:
        raise FileNotFoundError(
            "COST2100 데이터 파일을 찾지 못했습니다. data 폴더에 다음 파일들을 넣어주세요:\n"
            f"  indoor: DATA_Htrainin.mat, DATA_Hvalin.mat, DATA_Htestin.mat\n"
            f"  outdoor: DATA_Htrainout.mat, DATA_Hvalout.mat, DATA_Htestout.mat\n"
            f"현재 DATA_DIR={os.path.abspath(DATA_DIR)}"
        )

    print(
        f"[Dataset] {scenario}: train={x_train.shape}, "
        f"val={x_val.shape}, test={x_test.shape}"
    )
    return x_train, x_val, x_test


# ─────────────────────────────────────────────────────────────────────────────
# 합성 데이터 생성 (실제 데이터 없을 때 테스트용)
# ─────────────────────────────────────────────────────────────────────────────
def make_synthetic_cost2100(n_samples=50000, seed=42):
    """
    COST2100 채널을 현실적으로 모사하는 개선된 합성 데이터.
    - Cluster 기반 다중 경로 (각도/지연 spread 포함)
    - 경로별 진폭 감쇠 (Rayleigh fading)
    - 샘플마다 다른 noise level
    """
    rng = np.random.default_rng(seed)
    H_out = np.zeros((n_samples, 2, 32, 32), dtype=np.float32)

    for i in range(n_samples):
        H_ad = np.zeros((32, 32), dtype=complex)

        n_clusters = rng.integers(1, 5)
        for _ in range(n_clusters):
            c_row = rng.integers(0, 32)
            c_col = rng.integers(0, 6)
            n_scatter = rng.integers(3, 9)
            cluster_amp = rng.uniform(0.3, 1.0)

            for _ in range(n_scatter):
                row = int(np.clip(c_row + rng.integers(-2, 3), 0, 31))
                col = int(np.clip(c_col + rng.integers(-1, 2), 0, 7))
                amp = cluster_amp * rng.rayleigh(0.5)
                phase = rng.uniform(0, 2 * np.pi)
                H_ad[row, col] += amp * np.exp(1j * phase)

        H_spatial = np.fft.ifft2(H_ad)

        snr_db = rng.uniform(20, 40)
        snr_lin = 10 ** (snr_db / 10)
        sig_pwr = np.mean(np.abs(H_spatial) ** 2)
        noise = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))) * np.sqrt(sig_pwr / (2 * snr_lin))
        H_spatial = H_spatial + noise

        H_out[i, 0] = np.real(H_spatial).astype(np.float32)
        H_out[i, 1] = np.imag(H_spatial).astype(np.float32)

    for i in range(n_samples):
        vmin, vmax = H_out[i].min(), H_out[i].max()
        H_out[i] = (H_out[i] - vmin) / (vmax - vmin + 1e-8)

    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)
    print(
        f"[Dataset] 합성 데이터 생성 완료: train={n_train}, "
        f"val={n_val}, test={n_samples - n_train - n_val}"
    )
    return (
        H_out[:n_train],
        H_out[n_train:n_train + n_val],
        H_out[n_train + n_val:],
    )


class CSIDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloaders(scenario="indoor", batch_size=200, use_synthetic=False):
    if use_synthetic:
        print("[Info] 합성 데이터 사용 (실제 .mat 파일 없음)")
        x_train, x_val, x_test = make_synthetic_cost2100()
    else:
        x_train, x_val, x_test = load_cost2100(scenario)

    # Windows 환경 안정성을 위해 persistent_workers는 사용하지 않습니다.
    train_loader = DataLoader(
        CSIDataset(x_train), batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        CSIDataset(x_val), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        CSIDataset(x_test), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders(scenario="indoor", use_synthetic=False)
    batch = next(iter(train_l))
    print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
    print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
