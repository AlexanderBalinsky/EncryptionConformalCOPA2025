#!/usr/bin/env python3
"""
Evaluate intersections of BB prediction sets across multiple encryption keys.

For each fixed (key, IV) pair, this script:
  1) AES-CBC encrypts MNIST deterministically,
  2) trains the same feed-forward model,
  3) computes BB threshold on a calibration split,
  4) builds BB prediction sets on a disjoint CP-test split.

Then, for each CP-test image, it intersects prediction sets across keys and reports:
  - intersection mean size,
  - intersection size distribution,
  - how often the intersection still contains the correct label.

Requires:
  pip install torch torchvision numpy pycryptodome

Example:
  python intersection_bb_keys.py --num-keys 10 --alpha 0.4 --n-cal 5000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from torchvision import datasets

try:
    from Crypto.Cipher import AES
except ModuleNotFoundError:
    from Cryptodome.Cipher import AES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForwardNet(nn.Module):
    def __init__(self, image_size=28 * 28, hidden=128, num_classes=10, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(image_size, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def aes_fixed(images, key, iv):
    """Deterministic AES-CBC encryption with a single fixed (key, iv)."""
    out = np.empty_like(images, dtype=np.uint8)
    for i, img in enumerate(images):
        ct = AES.new(key, AES.MODE_CBC, iv).encrypt(img.astype(np.uint8).tobytes())
        out[i] = np.frombuffer(ct, dtype=np.uint8).reshape(img.shape)
    return out


def build_model(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return FeedForwardNet().to(DEVICE)


def train_model(model, x_train, y_train, epochs, batch_size, seed):
    train_x = torch.as_tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    train_y = torch.as_tensor(y_train, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def per_label_loss(model, x):
    """Return [num_samples, 10] where entry (i,c) = -log softmax_c(logits_i)."""
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(1) / 255.0
        logits = model(x_t)
        return (-F.log_softmax(logits, dim=1)).cpu().numpy()


def bb_threshold(cal_losses, alpha):
    n = len(cal_losses)
    lbar = cal_losses.mean()
    return (1.0 / alpha) * (1.0 / (1.0 + (1.0 - 1.0 / alpha) / n)) * lbar


def build_keys(rng, num_keys):
    keys = [(b"abs2kas126oZbdXs", b"1nsdjah72MdnJ12a")]
    keys += [(rng.bytes(16), rng.bytes(16)) for _ in range(num_keys - 1)]
    return keys


def main():
    parser = argparse.ArgumentParser(description="Intersection analysis for BB prediction sets across keys")
    parser.add_argument("--num-keys", type=int, default=10, help="Number of fixed encryption keys")
    parser.add_argument("--alpha", type=float, default=0.4, help="Miscoverage level for BB predictor")
    parser.add_argument("--n-cal", type=int, default=5000, help="Calibration size")
    parser.add_argument("--epochs", type=int, default=32, help="Training epochs per key")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=2024, help="Global seed")
    args = parser.parse_args()

    if args.num_keys < 2:
        raise ValueError("--num-keys must be at least 2 for intersection analysis")

    rng = np.random.RandomState(args.seed)

    train_ds = datasets.MNIST(root=".", train=True, download=True)
    test_ds = datasets.MNIST(root=".", train=False, download=True)
    xtr = train_ds.data.numpy()
    ytr = train_ds.targets.numpy()
    xte = test_ds.data.numpy()
    yte = test_ds.targets.numpy()

    idx = np.random.RandomState(args.seed).permutation(len(xte))
    cal_idx = idx[: args.n_cal]
    cp_idx = idx[args.n_cal : 2 * args.n_cal]
    y_cp = yte[cp_idx]

    keys = build_keys(rng, args.num_keys)

    all_sets = []
    print(f"Running BB sets for {args.num_keys} keys on device={DEVICE} ...")
    for i, (key, iv) in enumerate(keys, 1):
        xtr_enc = aes_fixed(xtr, key, iv)
        xte_enc = aes_fixed(xte, key, iv)

        model = build_model(args.seed)
        train_model(model, xtr_enc, ytr, args.epochs, args.batch, args.seed)

        losses = per_label_loss(model, xte_enc)
        cal_losses = losses[cal_idx, yte[cal_idx]]
        tau = bb_threshold(cal_losses, args.alpha)

        sets_cp = losses[cp_idx] <= tau  # [n_cp, 10]
        all_sets.append(sets_cp)

        cover_i = sets_cp[np.arange(len(cp_idx)), y_cp].mean()
        size_i = sets_cp.sum(axis=1).mean()
        print(
            f"key {i:2d}: tau={tau:.4f}, BB-cover={100*cover_i:6.2f}%, mean-set-size={size_i:5.2f}"
        )

    stack_sets = np.stack(all_sets, axis=0)  # [K, n_cp, 10]
    inter_sets = np.logical_and.reduce(stack_sets, axis=0)  # [n_cp, 10]

    inter_sizes = inter_sets.sum(axis=1)
    inter_cover = inter_sets[np.arange(len(cp_idx)), y_cp].mean()

    print("\nIntersection across keys (per image):")
    print(f"  Mean intersection size: {inter_sizes.mean():.4f}")
    print(f"  Median intersection size: {np.median(inter_sizes):.4f}")
    print(f"  Empty intersections: {(inter_sizes == 0).mean() * 100:.2f}%")
    print(f"  Coverage of true label by intersection: {inter_cover * 100:.2f}%")

    values, counts = np.unique(inter_sizes, return_counts=True)
    print("\nIntersection size distribution:")
    for v, c in zip(values, counts):
        print(f"  size={int(v):2d}: {c:5d} ({100*c/len(inter_sizes):6.2f}%)")


if __name__ == "__main__":
    main()
