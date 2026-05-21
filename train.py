"""
train.py for DPAS-CsiNet v5
- 학습 기법: Adam + 선택적 cosine annealing (--scheduler cosine)
- 모델: dpas_csinet_v5
"""

import os, time, json, argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models.dpas_csinet_v5 import DPASCsiNet, ProgressiveFeedbackLoss
from models.csinet_baseline import CsiNet
from utils import nmse, rho_metric, raw_mse_loss_torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str,   default='dpas', choices=['dpas','csinet'])
    p.add_argument('--scenario',   type=str,   default='indoor', choices=['indoor','outdoor'])
    p.add_argument('--cr',         type=int,   default=4,   choices=[4,8,16,32,64])
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=200)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--scheduler',  type=str,   default='none', choices=['none','cosine'],
                   help='none=lr고정(const), cosine=cosine annealing')
    p.add_argument('--T_max',      type=int,   default=2500,
                   help='cosine annealing 주기 (CRNet 논문 기준 2500)')
    p.add_argument('--save_dir',   type=str,   default='./checkpoints')
    p.add_argument('--save_every', type=int,   default=50)
    p.add_argument('--amp',        action='store_true', default=True)
    p.add_argument('--resume',     type=str,   default='')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    xs, rs = [], []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        out = model(x)
        recon = out[0] if isinstance(out, tuple) else out
        xs.append(x.cpu().numpy())
        rs.append(recon.cpu().numpy())
    return nmse(np.concatenate(rs), np.concatenate(xs)), \
           rho_metric(np.concatenate(rs), np.concatenate(xs))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    if device.type == 'cuda':
        print(f"[GPU]  {torch.cuda.get_device_name(0)}")
        print(f"[VRAM] {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    bits_map = {4:512, 8:256, 16:128, 32:64, 64:32}
    feedback_bits = bits_map[args.cr]
    print(f"[Config] model={args.model} scenario={args.scenario} "
          f"CR=1/{args.cr} bits={feedback_bits}")
    print(f"[Scheduler] {args.scheduler}")

    train_loader, val_loader, test_loader = get_dataloaders(
        scenario=args.scenario, batch_size=args.batch_size)

    if args.model == 'dpas':
        model     = DPASCsiNet(feedback_bits=feedback_bits)
        criterion = ProgressiveFeedbackLoss(aux_weight=0.0,
                                            recon_loss_fn=raw_mse_loss_torch)
    else:
        model     = CsiNet(feedback_bits=feedback_bits)
        criterion = raw_mse_loss_torch

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 파라미터: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-5) \
                if args.scheduler == 'cosine' else None

    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type == 'cuda') else None

    os.makedirs(args.save_dir, exist_ok=True)
    tag       = f"{args.model}_v5_{args.scheduler}_{args.scenario}_cr{args.cr}"
    best_ckpt = os.path.join(args.save_dir, f"{tag}_best.pth")
    last_ckpt = os.path.join(args.save_dir, f"{tag}_last.pth")

    best_nmse_val = float('inf')
    history = {'train_loss': [], 'val_nmse': [], 'val_rho': [], 'lr': []}
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck.get('optimizer_state', {}))
        start_epoch = int(ck.get('epoch', 0)) + 1
        best_nmse_val = float(ck.get('best_nmse', best_nmse_val))
        history = ck.get('history', history)
        print(f"[Resume] epoch {start_epoch}부터 재개")

    print(f"\n{'='*60}\n 학습 시작: {args.epochs} epochs\n{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, count = 0.0, 0

        for batch in train_loader:
            x = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out   = model(x)
                    recon = out[0] if isinstance(out, tuple) else out
                    loss  = criterion(recon, x) if args.model == 'csinet' \
                            else criterion(recon, x, out[1], feedback_bits)[0]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out   = model(x)
                recon = out[0] if isinstance(out, tuple) else out
                loss  = criterion(recon, x) if args.model == 'csinet' \
                        else criterion(recon, x, out[1], feedback_bits)[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum += loss.item() * x.size(0)
            count    += x.size(0)

        if scheduler:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        val_nmse_v, val_rho_v = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        history['train_loss'].append(loss_sum / max(1, count))
        history['val_nmse'].append(float(val_nmse_v))
        history['val_rho'].append(float(val_rho_v))
        history['lr'].append(float(lr_now))

        print(f"[Epoch {epoch:4d}/{args.epochs}] loss={loss_sum/max(1,count):.5f} | "
              f"val NMSE={val_nmse_v:.2f}dB ρ={val_rho_v:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        if val_nmse_v < best_nmse_val:
            best_nmse_val = val_nmse_v
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'best_nmse': float(best_nmse_val),
                        'history': history,
                        'args': vars(args)}, best_ckpt)
            print(f"  → Best 저장 (NMSE={best_nmse_val:.2f}dB)")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'best_nmse': float(best_nmse_val),
                        'history': history,
                        'args': vars(args)}, last_ckpt)
            print(f"  → Last 저장: epoch {epoch}")

    print(f"\n{'='*60}\n 최종 테스트 평가\n{'='*60}")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'])
    test_nmse_v, test_rho_v = evaluate(model, test_loader, device)
    print(f"[Test] NMSE={test_nmse_v:.2f}dB  ρ={test_rho_v:.4f}")

    result = {
        'model': args.model + '_v5_' + args.scheduler,
        'scenario': args.scenario, 'cr': args.cr,
        'feedback_bits': feedback_bits,
        'test_nmse_dB': float(test_nmse_v),
        'test_rho': float(test_rho_v),
        'best_val_nmse_dB': float(best_nmse_val),
        'total_params': int(total_params),
        'scheduler': args.scheduler,
        'history': history,
    }
    rpath = os.path.join(args.save_dir, f"{tag}_result.json")
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {rpath}")
    return result


if __name__ == '__main__':
    train(parse_args())
