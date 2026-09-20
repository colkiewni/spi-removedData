"""
Train policy network on ISMCTS self-play data.

Usage:
  python scripts/train_policy.py
  python scripts/train_policy.py --data data/ismcts_experiences
  python scripts/train_policy.py --epochs 100 --hidden 256

Requires GPU for reasonable speed.
Saves best model to models/policy_best.pt
"""

import argparse
import os
import pickle
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import (
    FEATURE_DIM, HIDDEN_DIM, N_LAYERS,
    BATCH_SIZE, VAL_BATCH_SIZE, LR, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, VAL_FRACTION, GRAD_CLIP,
    MODELS_DIR, DRIVE_DIR
)
from neural.features import obs_to_features
from neural.policy_net import PolicyNet, count_params


class PolicyDataset(Dataset):
    def __init__(self, data_dirs, is_val=False,
                 val_fraction=VAL_FRACTION, seed=42):
        all_files = []
        for d in data_dirs:
            if os.path.exists(d):
                found = sorted([
                    os.path.join(d, f)
                    for f in os.listdir(d)
                    if f.endswith('.pkl')
                ])
                all_files += found
                print(f"  {d}: {len(found)} shards")

        rng   = random.Random(seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        files = (all_files[:n_val] if is_val
                 else all_files[n_val:])
        label = 'Val' if is_val else 'Train'
        print(f"  {label}: {len(files)} shards")

        feats, acts = [], []
        for path in files:
            try:
                with open(path, 'rb') as f:
                    exps = pickle.load(f)
                for exp in exps:
                    if exp.get('phase') != 'PLAY':
                        continue
                    feats.append(obs_to_features(exp['obs']))
                    acts.append(exp['action'])
            except Exception:
                pass

        self.features = torch.tensor(
            np.array(feats), dtype=torch.float32
        )
        self.actions  = torch.tensor(
            np.array(acts), dtype=torch.long
        )
        print(f"  Samples: {len(self.features):,}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.actions[idx]


def save_checkpoint(model, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = model.state_dict()
    torch.save(payload, path)


def backup_model_to_drive(path):
    if not os.path.exists('/content/drive'):
        return
    dst_dir = os.path.join(DRIVE_DIR, 'models')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(path))
    shutil.copy2(path, dst)
    print(f"  → Drive: {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   nargs='+',
                        default=['data/ismcts_experiences'])
    parser.add_argument('--epochs', type=int,
                        default=MAX_EPOCHS)
    parser.add_argument('--hidden', type=int,
                        default=HIDDEN_DIM)
    parser.add_argument('--layers', type=int,
                        default=N_LAYERS)
    parser.add_argument('--out',    type=str,
                        default='models/policy_best.pt')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    train_ds = PolicyDataset(args.data, is_val=False)
    print()
    val_ds   = PolicyDataset(args.data, is_val=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    model     = PolicyNet(
        hidden_dim=args.hidden, n_layers=args.layers
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\nModel params: {count_params(model):,}")
    print(f"Training for up to {args.epochs} epochs...\n")

    best_acc = best_epoch = patience_count = 0
    t_total  = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP
            )
            optimizer.step()
            tl += loss.item() * len(x)
            tc += (out.argmax(1) == y).sum().item()
            n  += len(x)
        tl /= max(n, 1)
        ta  = tc / max(n, 1)

        model.eval()
        vl, vc, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out  = model(x)
                loss = criterion(out, y)
                vl += loss.item() * len(x)
                vc += (out.argmax(1) == y).sum().item()
                n  += len(x)
        vl /= max(n, 1)
        va  = vc / max(n, 1)

        scheduler.step(vl)

        improved = ""
        if va > best_acc + 0.001:
            best_acc, best_epoch = va, epoch
            patience_count       = 0
            improved             = " ← best"
            save_checkpoint(model, args.out)
        else:
            patience_count += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {tl:.4f} ({ta:.3f}) | "
              f"Val: {vl:.4f} ({va:.3f}) | "
              f"{time.time()-t0:.1f}s{improved}")

        if patience_count >= PATIENCE:
            print(f"\nEarly stop. Best: epoch {best_epoch} "
                  f"val_acc={best_acc:.3f}")
            break

    # Save final and backup
    final_path = args.out.replace('_best', '_latest')
    save_checkpoint(model, final_path)
    backup_model_to_drive(args.out)
    backup_model_to_drive(final_path)

    print(f"\nBest val acc: {best_acc:.3f}")
    print(f"Best epoch:   {best_epoch}")
    print(f"Total time:   {time.time()-t_total:.0f}s")
    print(f"Saved:        {args.out}")


if __name__ == '__main__':
    main()