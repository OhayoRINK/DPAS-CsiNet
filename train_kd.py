"""
train_kd.py — Knowledge Distillation
======================================
Teacher : 원본 DPAS (dpas_csinet.py, 학습된 best checkpoint)
Student : v4 DPAS  (dpas_csinet_v4.py, 처음부터 학습)

Loss = α·MSE(recon_s, target)           ← student 복원 손실
     + β·MSE(code_s, code_t.detach())   ← codeword 레벨 증류
     + γ·MSE(enc_feat_s, enc_feat_t.detach())  ← encoder feature 증류
     + δ·MSE(dec_feat_s, dec_feat_t.detach())  ← decoder feature 증류

실행 예시:
  python train_kd.py --scenario indoor --cr 4 --epochs 100 \
      --teacher_ckpt ./checkpoints/dpas_indoor_cr4_best.pth

가중치 기본값: α=1.0, β=0.5, γ=0.5, δ=0.3
"""

import os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models.dpas_csinet import DPASCsiNet as TeacherNet
from models.dpas_csinet_v4 import DPASCsiNet as StudentNet
from utils import nmse, rho_metric


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario',      type=str,   default='indoor', choices=['indoor','outdoor'])
    p.add_argument('--cr',            type=int,   default=4,        choices=[4,8,16,32,64])
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch_size',    type=int,   default=200)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=0.0)
    p.add_argument('--scheduler',     type=str,   default='none',   choices=['none','cosine'])
    p.add_argument('--teacher_ckpt',  type=str,   default='',       help='원본 best.pth 경로')
    # KD 가중치
    p.add_argument('--alpha',  type=float, default=1.0,  help='복원 MSE 가중치')
    p.add_argument('--beta',   type=float, default=0.5,  help='codeword 증류 가중치')
    p.add_argument('--gamma',  type=float, default=0.5,  help='encoder feature 증류 가중치')
    p.add_argument('--delta',  type=float, default=0.3,  help='decoder feature 증류 가중치')
    p.add_argument('--save_dir',   type=str, default='./checkpoints')
    p.add_argument('--save_every', type=int, default=50)
    p.add_argument('--amp',        action='store_true', default=True)
    return p.parse_args()


# ──────────────────────────────────────────────
# Feature Hook
# ──────────────────────────────────────────────
class FeatureHook:
    """register_forward_hook으로 중간 feature를 캡처."""
    def __init__(self):
        self.feat = None

    def hook_fn(self, module, input, output):
        self.feat = output

    def register(self, module):
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        self._handle.remove()


# ──────────────────────────────────────────────
# Projection heads (feature 차원 맞추기)
# ──────────────────────────────────────────────
class ProjHead(nn.Module):
    """student feature 채널을 teacher 채널로 맞추는 1×1 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.proj(x)


# ──────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    xs, rs = [], []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        recon, _ = model(x)
        xs.append(x.cpu().numpy())
        rs.append(recon.cpu().numpy())
    xs = np.concatenate(xs)
    rs = np.concatenate(rs)
    return nmse(rs, xs), rho_metric(rs, xs)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if device.type == 'cuda':
        print(f"[GPU]  {torch.cuda.get_device_name(0)}")

    bits_map = {4:512, 8:256, 16:128, 32:64, 64:32}
    feedback_bits = bits_map[args.cr]
    print(f"[Config] KD | scenario={args.scenario} CR=1/{args.cr} bits={feedback_bits}")
    print(f"[KD weights] α={args.alpha} β={args.beta} γ={args.gamma} δ={args.delta}")

    train_loader, val_loader, test_loader = get_dataloaders(
        scenario=args.scenario, batch_size=args.batch_size)

    # ── Teacher 로드 (eval, no grad) ──
    teacher = TeacherNet(feedback_bits=feedback_bits).to(device)
    if args.teacher_ckpt and os.path.isfile(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        state = ckpt.get('model_state', ckpt)
        teacher.load_state_dict(state)
        print(f"[Teacher] 로드: {args.teacher_ckpt}")
    else:
        print("[Teacher] ⚠️  checkpoint 없음 → 랜덤 초기화 (teacher_ckpt 경로 확인 필요)")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Student ──
    student = StudentNet(feedback_bits=feedback_bits).to(device)
    s_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"[Teacher] 파라미터: {t_params:,}")
    print(f"[Student] 파라미터: {s_params:,}")

    # ── Hook 등록 ──
    # Teacher: encoder feature (DualPathBlock 2번째 출력), decoder 첫 DualPath 출력
    t_enc_hook = FeatureHook().register(teacher.encoder.feature[1])   # (B,16,H,W)
    t_dec_hook = FeatureHook().register(teacher.decoder.refine[0])    # (B,16,H,W) 첫 DualPath

    # Student: encoder feature, decoder 첫 DualPath
    s_enc_hook = FeatureHook().register(student.encoder.feature[1])   # (B,16,H,W)
    s_dec_hook = FeatureHook().register(student.decoder.dp1)          # (B,16,H,W)

    # Student encoder feature는 16ch, teacher도 16ch → projection 불필요
    # (만약 채널이 다르면 ProjHead 사용)

    mse = nn.MSELoss()

    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) \
                if args.scheduler == 'cosine' else None
    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"dpas_kd_{args.scenario}_cr{args.cr}"
    best_ckpt = os.path.join(args.save_dir, f"{tag}_best.pth")
    last_ckpt = os.path.join(args.save_dir, f"{tag}_last.pth")

    best_nmse_val = float('inf')
    history = {'train_loss': [], 'val_nmse': [], 'val_rho': [],
               'loss_recon': [], 'loss_code': [], 'loss_enc': [], 'loss_dec': []}

    print(f"\n{'='*60}\n 학습 시작: {args.epochs} epochs\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        teacher.eval()

        sum_total = sum_recon = sum_code = sum_enc = sum_dec = 0.0
        count = 0

        for batch in train_loader:
            x = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def forward_pass():
                # Teacher forward (no grad)
                with torch.no_grad():
                    recon_t, _ = teacher(x)
                    t_enc_feat = t_enc_hook.feat.detach()
                    t_dec_feat = t_dec_hook.feat.detach()
                    t_codeword, _ = teacher.encoder(x)
                    t_codeword = t_codeword.detach()

                # Student forward + explicit codeword path once
                feat_s = student.encoder.feature(x)
                aux_s = student.encoder.aux_proj(feat_s)
                main_s = student.encoder.channel_compress(feat_s).reshape(x.shape[0], -1)
                main_code_s = student.encoder.main_fc(main_s)
                skip_s = torch.nn.functional.adaptive_avg_pool2d(feat_s, 1).reshape(x.shape[0], -1)
                skip_code_s = student.encoder.skip_fc(skip_s)
                s_codeword = torch.sigmoid(student.encoder.merge_fc(torch.cat([main_code_s, skip_code_s], dim=1)))
                recon_s = student.decoder(s_codeword)

                s_enc_feat = s_enc_hook.feat
                s_dec_feat = s_dec_hook.feat

                l_recon = mse(recon_s, x)
                l_code  = mse(s_codeword, t_codeword)
                l_enc   = mse(s_enc_feat, t_enc_feat)
                l_dec   = mse(s_dec_feat, t_dec_feat)

                loss = (args.alpha * l_recon
                      + args.beta  * l_code
                      + args.gamma * l_enc
                      + args.delta * l_dec)
                return loss, l_recon, l_code, l_enc, l_dec

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss, l_recon, l_code, l_enc, l_dec = forward_pass()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, l_recon, l_code, l_enc, l_dec = forward_pass()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            bs = x.size(0)
            sum_total += loss.item()  * bs
            sum_recon += l_recon.item() * bs
            sum_code  += l_code.item()  * bs
            sum_enc   += l_enc.item()   * bs
            sum_dec   += l_dec.item()   * bs
            count     += bs

        if scheduler:
            scheduler.step()

        n = max(1, count)
        val_nmse_v, val_rho_v = evaluate(student, val_loader, device)
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

        history['train_loss'].append(sum_total / n)
        history['val_nmse'].append(float(val_nmse_v))
        history['val_rho'].append(float(val_rho_v))
        history['loss_recon'].append(sum_recon / n)
        history['loss_code'].append(sum_code  / n)
        history['loss_enc'].append(sum_enc   / n)
        history['loss_dec'].append(sum_dec   / n)

        print(f"[Epoch {epoch:4d}/{args.epochs}] "
              f"total={sum_total/n:.5f} "
              f"recon={sum_recon/n:.5f} code={sum_code/n:.5f} "
              f"enc={sum_enc/n:.5f} dec={sum_dec/n:.5f} | "
              f"val NMSE={val_nmse_v:.2f}dB ρ={val_rho_v:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if val_nmse_v < best_nmse_val:
            best_nmse_val = val_nmse_v
            torch.save({'epoch': epoch,
                        'model_state': student.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_nmse': float(best_nmse_val),
                        'history': history,
                        'args': vars(args)}, best_ckpt)
            print(f"  → Best 저장 (NMSE={best_nmse_val:.2f}dB)")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({'epoch': epoch,
                        'model_state': student.state_dict(),
                        'best_nmse': float(best_nmse_val),
                        'history': history,
                        'args': vars(args)}, last_ckpt)
            print(f"  → Last 저장: epoch {epoch}")

    # 최종 테스트
    print(f"\n{'='*60}\n 최종 테스트 평가\n{'='*60}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt['model_state'])
    test_nmse_v, test_rho_v = evaluate(student, test_loader, device)
    print(f"[Test] NMSE={test_nmse_v:.2f}dB  ρ={test_rho_v:.4f}")

    result = {
        'method': 'KD', 'scenario': args.scenario, 'cr': args.cr,
        'feedback_bits': feedback_bits,
        'test_nmse_dB': float(test_nmse_v), 'test_rho': float(test_rho_v),
        'best_val_nmse_dB': float(best_nmse_val),
        'student_params': int(s_params), 'teacher_params': int(t_params),
        'kd_weights': {'alpha': args.alpha, 'beta': args.beta,
                       'gamma': args.gamma, 'delta': args.delta},
        'history': history,
    }
    rpath = os.path.join(args.save_dir, f"{tag}_result.json")
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {rpath}")

    t_enc_hook.remove(); t_dec_hook.remove()
    s_enc_hook.remove(); s_dec_hook.remove()
    return result


if __name__ == '__main__':
    train(parse_args())
