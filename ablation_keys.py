#!/usr/bin/env python3
"""
Robustness-to-key-variation ablation for
"Conformal Prediction for Privacy-Preserving Machine Learning".

For each of K fixed (key, IV) pairs this script:
  1. AES-CBC encrypts MNIST deterministically (same key/IV for every image),
  2. trains the fixed feed-forward network,
  3. splits the test set into Calibration / CP-Test (n = 5000 each),
  4. computes the p-value and e-value (BB) thresholds at alpha = 0.4,
  5. builds prediction sets over the 10 candidate labels,
  6. records test accuracy, realized coverage and mean prediction-set size.
It also runs the plaintext and per-image-randomized-key boundary cases, then
prints the rows you can paste into Table 3 (Robustness to key variation).

This is a *reference* harness that reproduces the pipeline described in the
paper. Adapt the model / preprocessing to match your own repository exactly so
the numbers line up with Sections 5.1-5.5.

Requires:  pip install torch torchvision numpy pycryptodome
Run:       python ablation_keys.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from torchvision import datasets

try:
    from Crypto.Cipher import AES
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    from Cryptodome.Cipher import AES

ALPHA  = 0.4      # miscoverage level  (target coverage = 1 - ALPHA = 0.6)
N      = 5000     # size of Calibration set and of CP-Test set
SEED   = 2024
EPOCHS = 32
BATCH  = 64
NUM_FIXED_KEYS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 28*28 = 784 bytes per image, which is exactly 49 AES blocks (16 bytes) -> no padding.

def aes_fixed(images, key, iv):
    """Deterministic AES-CBC encryption with a single fixed (key, iv)."""
    out = np.empty_like(images, dtype=np.uint8)
    for i, img in enumerate(images):
        ct = AES.new(key, AES.MODE_CBC, iv).encrypt(img.astype(np.uint8).tobytes())
        out[i] = np.frombuffer(ct, dtype=np.uint8).reshape(img.shape)
    return out

def aes_per_image(images, rng):
    """Randomized encryption: a fresh random (key, iv) per image."""
    out = np.empty_like(images, dtype=np.uint8)
    for i, img in enumerate(images):
        key, iv = rng.bytes(16), rng.bytes(16)
        ct = AES.new(key, AES.MODE_CBC, iv).encrypt(img.astype(np.uint8).tobytes())
        out[i] = np.frombuffer(ct, dtype=np.uint8).reshape(img.shape)
    return out

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_model():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    return FeedForwardNet().to(DEVICE)

def per_label_loss(model, x):
    """Return [num_samples, 10]: cross-entropy loss of x under each candidate label.
       loss for label c = -log softmax_c(logits)."""
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(x, dtype=torch.float32, device=DEVICE) / 255.0
        logits = model(inputs)
        return (-F.log_softmax(logits, dim=1)).cpu().numpy()

def evaluate(x_train, y_train, x_test, y_test):
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_x = torch.as_tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    train_y = torch.as_tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model.train()
    for _ in range(EPOCHS):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    test_x = torch.as_tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
    test_y = torch.as_tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        logits = model(test_x.to(DEVICE))
        predictions = logits.argmax(dim=1).cpu()
        acc = (predictions == test_y).float().mean().item()

    # Split the test set into Calibration and CP-Test (disjoint, n each).
    idx = np.random.RandomState(SEED).permutation(len(x_test))
    cal, cp = idx[:N], idx[N:2 * N]

    L = per_label_loss(model, x_test)                 # [10000, 10]
    cal_loss = L[cal, y_test[cal]]                    # calibration scores at true labels

    # p-value threshold: ceil((1-alpha)(n+1))-th smallest calibration loss.
    k_index = math.ceil((1 - ALPHA) * (N + 1))        # = 3001 for n = 5000
    tau_p = np.sort(cal_loss)[k_index - 1]            # 1-indexed -> 0-indexed

    # e-value (BB) threshold (Corollary 1 in the paper).
    Lbar  = cal_loss.mean()
    tau_e = (1 / ALPHA) * (1 / (1 + (1 - 1 / ALPHA) / N)) * Lbar

    def cover_and_size(tau):
        sets  = L[cp] <= tau                          # [n, 10] boolean
        cover = sets[np.arange(N), y_test[cp]].mean()
        size  = sets.sum(axis=1).mean()
        return 100 * cover, size

    cov_p, size_p = cover_and_size(tau_p)
    cov_e, size_e = cover_and_size(tau_e)
    return dict(acc=100 * acc, cov_p=cov_p, size_p=size_p,
                cov_e=cov_e, size_e=size_e, tau_p=tau_p, tau_e=tau_e)

def fmt(r):
    return (f"{r['acc']:6.2f} {r['cov_p']:6.1f} {r['size_p']:7.2f} "
            f"{r['cov_e']:6.1f} {r['size_e']:7.2f}")

def main():
    train_ds = datasets.MNIST(
        root=".",
        train=True,
        download=True,
    )
    test_ds = datasets.MNIST(
        root=".",
        train=False,
        download=True,
    )
    xtr = train_ds.data.numpy()
    ytr = train_ds.targets.numpy()
    xte = test_ds.data.numpy()
    yte = test_ds.targets.numpy()
    rng = np.random.RandomState(SEED)

    # First fixed key = the paper's key; the rest are random 16-byte (key, iv) pairs.
    keys = [(b"abs2kas126oZbdXs", b"1nsdjah72MdnJ12a")]
    keys += [(rng.bytes(16), rng.bytes(16)) for _ in range(NUM_FIXED_KEYS - 1)]

    header = f"{'regime':26s} {'acc':>6s} {'p-cov':>6s} {'p|set|':>7s} {'e-cov':>6s} {'e|set|':>7s}"
    print(header); print("-" * len(header))

    rows = []
    for j, (key, iv) in enumerate(keys, 1):
        r = evaluate(aes_fixed(xtr, key, iv), ytr, aes_fixed(xte, key, iv), yte)
        rows.append(r)
        tag = "fixed key 1 (paper)" if j == 1 else f"fixed key {j}"
        print(f"{tag:26s} {fmt(r)}   (tau_p={r['tau_p']:.4f}, tau_e={r['tau_e']:.4f})")

    A = np.array([[r['acc'], r['cov_p'], r['size_p'], r['cov_e'], r['size_e']] for r in rows])
    m, s = A.mean(0), A.std(0)
    print("-" * len(header))
    print(f"{'MEAN (fixed keys)':26s} {m[0]:6.2f} {m[1]:6.1f} {m[2]:7.2f} {m[3]:6.1f} {m[4]:7.2f}")
    print(f"{'SD   (fixed keys)':26s} {s[0]:6.2f} {s[1]:6.1f} {s[2]:7.2f} {s[3]:6.1f} {s[4]:7.2f}")
    print("-" * len(header))

    # Boundary cases.
    rp = evaluate(xtr, ytr, xte, yte)  # plaintext (CP columns are reported for completeness)
    print(f"{'plaintext':26s} {fmt(rp)}")
    ptr, pte = aes_per_image(xtr, rng), aes_per_image(xte, rng)
    rr = evaluate(ptr, ytr, pte, yte)
    print(f"{'per-image random key':26s} {fmt(rr)}")

if __name__ == "__main__":
    main()
