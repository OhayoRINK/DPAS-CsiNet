"""
flops_count.py - FLOPs & Params 측정
사용법: python flops_count.py --cr 4
"""

import argparse
import importlib
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cr', type=int, default=4, choices=[4,8,16,32,64])
    p.add_argument('--models', type=str, nargs='+',
                   default=['dpas_v5', 'csinet'],
                   help='측정할 모델 목록')
    return p.parse_args()

MODEL_MAP = {
    'dpas_v5': ('models.dpas_csinet_v5', 'DPASCsiNet'),
    'dpas_v4': ('models.dpas_csinet',    'DPASCsiNet'),
    'csinet':  ('models.csinet_baseline','CsiNet'),
}

def measure(model_name, feedback_bits):
    try:
        from thop import profile
    except ImportError:
        raise ImportError("pip install thop 먼저 실행하세요")

    module_path, cls_name = MODEL_MAP[model_name]
    mod = importlib.import_module(module_path)
    model = getattr(mod, cls_name)(feedback_bits=feedback_bits)
    model.eval()

    x = torch.randn(1, 2, 32, 32)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return flops, params

def main():
    args = parse_args()
    bits_map = {4:512, 8:256, 16:128, 32:64, 64:32}
    feedback_bits = bits_map[args.cr]

    print(f"\n[FLOPs & Params] CR=1/{args.cr}, bits={feedback_bits}")
    print(f"{'모델':<14} {'FLOPs':>10} {'Params':>12} {'파라미터 수':>14}")
    print("-" * 55)

    for name in args.models:
        if name not in MODEL_MAP:
            print(f"{name:<14} 알 수 없는 모델 (MODEL_MAP 확인)")
            continue
        flops, params = measure(name, feedback_bits)
        print(f"{name:<14} {flops/1e6:>9.2f}M {params/1e6:>11.2f}M ({int(params):>12,})")

    print()
    print("[참고] CRNet 논문 Table I 기준: CsiNet=5.41M FLOPs, CRNet=5.12M FLOPs")

if __name__ == '__main__':
    main()
