#!/usr/bin/env python3
"""
Intersection of BB-prediction sets across K independent encryption keys.

Research question
-----------------
Train K independent pipelines, one per fixed encryption key. Each produces a
BB (e-value) prediction set C_k(x) for every test image, with high per-key
coverage (P(y in C_k) > 0.9) but possibly large sets. For each image take the
intersection  C_cap(x) = intersection_k C_k(x).
  * How large is C_cap(x)?
  * How often does it contain the true label?

Two reference points frame the answer (see the write-up):
  * Bonferroni / union lower bound:  P(y in C_cap) >= 1 - sum_k alpha_k,
        where alpha_k = 1 - (realized per-key coverage).
  * Independence estimate (errors independent across keys):
        P(y in C_cap) ~ prod_k (1 - alpha_k).
  Reality sits between these; positive error correlation across keys (the same
  hard images are missed regardless of key) pushes it toward the upper side.

The analysis functions (intersection_metrics, sweep_over_K) operate on plain
boolean arrays and are unit-tested at the bottom (run:  python intersection_bb.py --selftest).

Requires:  pip install torch torchvision numpy pycryptodome  (matplotlib optional)
Run:       python intersection_bb.py
"""

import sys
import math
import itertools
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

ALPHA   = 0.4      # miscoverage level for the BB threshold (BB is conservative,
                   # so realized per-key coverage is typically ~0.95-0.98 > 0.9)
N       = 5000     # Calibration size and CP-Test size
SEED    = 2024
EPOCHS  = 32
BATCH   = 64
NUM_KEYS = 10
MAX_SUBSETS_PER_K = 50   # cap on random subsets averaged per K in the sweep
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


def train_model(model, x_train, y_train, epochs=EPOCHS, batch=BATCH, seed=SEED):
    train_x = torch.as_tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    train_y = torch.as_tensor(y_train, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch,
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
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(1) / 255.0
        logits = model(x_t)
        return (-F.log_softmax(logits, dim=1)).cpu().numpy()

# ----------------------------------------------------------------------------
# Analysis layer  (pure numpy; unit-tested -- no ML here)
# ----------------------------------------------------------------------------
def intersection_metrics(sets, y_true):
    """
    sets   : bool array [K, n, C]  -- sets[k, i, c] = (label c in C_k(image i))
    y_true : int array  [n]        -- true labels of the n images
    Returns metrics for the intersection over ALL K sets.
    """
    K, n, C = sets.shape
    inter = sets.all(axis=0)                       # [n, C]  label kept iff in every set
    size  = inter.sum(axis=1)                      # [n]     intersection size per image
    covered = inter[np.arange(n), y_true]          # [n]     true label survived?
    size_hist = np.bincount(size, minlength=C + 1) # counts of size 0,1,...,C
    return dict(
        mean_size = size.mean(),
        coverage  = covered.mean(),
        empty_frac = (size == 0).mean(),
        size_hist = size_hist,
    )

def per_key_metrics(sets, y_true):
    """Per-key coverage and mean size (one entry per key)."""
    K, n, C = sets.shape
    cov  = sets[:, np.arange(n), y_true].mean(axis=1)   # [K]
    size = sets.sum(axis=2).mean(axis=1)                # [K]
    return cov, size

def union_metrics(sets, y_true):
    K, n, C = sets.shape
    uni = sets.any(axis=0)
    size = uni.sum(axis=1)
    covered = uni[np.arange(n), y_true]
    return dict(mean_size=size.mean(), coverage=covered.mean())

def sweep_over_K(sets, y_true, max_subsets=MAX_SUBSETS_PER_K, rng=None):
    """
    For each K' = 1..K, estimate the EXPECTED intersection behaviour over random
    subsets of K' keys (averages out the dependence on which keys are chosen).
    Returns dict K' -> (mean_size, coverage, empty_frac).
    """
    if rng is None:
        rng = np.random.RandomState(0)
    K = sets.shape[0]
    out = {}
    for k in range(1, K + 1):
        total = math.comb(K, k)
        if total <= max_subsets:
            combos = itertools.combinations(range(K), k)
        else:
            sampled = set()
            while len(sampled) < max_subsets:
                combo = tuple(sorted(rng.choice(K, size=k, replace=False).tolist()))
                sampled.add(combo)
            combos = sampled
        ms, cov, emp = [], [], []
        n = sets.shape[1]
        for combo in combos:
            m = intersection_metrics(sets[list(combo)], y_true)
            ms.append(m['mean_size']); cov.append(m['coverage']); emp.append(m['empty_frac'])
        out[k] = (float(np.mean(ms)), float(np.mean(cov)), float(np.mean(emp)))
    return out

def theory_bounds(per_key_cov):
    """Bonferroni lower bound and independence estimate for full intersection."""
    alphas = 1.0 - np.asarray(per_key_cov)
    bonferroni_lb = max(0.0, 1.0 - alphas.sum())
    independence  = float(np.prod(1.0 - alphas))
    return bonferroni_lb, independence


def parse_checkpoints(text, max_k):
    vals = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    vals = sorted(set(vals))
    vals = [k for k in vals if 1 <= k <= max_k]
    if not vals:
        vals = [max_k]
    return vals

# ----------------------------------------------------------------------------
# ML layer  (builds the boolean `sets` array from MNIST + AES + BB thresholds)
# ----------------------------------------------------------------------------
def build_sets(num_keys=NUM_KEYS, n=N, seed=SEED, epochs=EPOCHS, batch=BATCH):
    train_ds = datasets.MNIST(root=".", train=True, download=True)
    test_ds = datasets.MNIST(root=".", train=False, download=True)
    xtr = train_ds.data.numpy()
    ytr = train_ds.targets.numpy()
    xte = test_ds.data.numpy()
    yte = test_ds.targets.numpy()

    idx = np.random.RandomState(seed).permutation(len(xte))
    cal, cp = idx[:n], idx[n:2 * n]                          # SAME split for every key
    y_cp = yte[cp]

    rng = np.random.RandomState(seed)
    keys = [(b"abs2kas126oZbdXs", b"1nsdjah72MdnJ12a")]
    keys += [(rng.bytes(16), rng.bytes(16)) for _ in range(num_keys - 1)]

    sets = np.zeros((num_keys, n, 10), dtype=bool)
    for k, (key, iv) in enumerate(keys):
        etr, ete = aes_fixed(xtr, key, iv), aes_fixed(xte, key, iv)
        m = build_model(seed)
        train_model(m, etr, ytr, epochs=epochs, batch=batch, seed=seed)
        L = per_label_loss(m, ete)                           # [10000, 10]
        cal_loss = L[cal, yte[cal]]
        tau_e = (1 / ALPHA) * (1 / (1 + (1 - 1 / ALPHA) / n)) * cal_loss.mean()
        sets[k] = L[cp] <= tau_e
        cov = sets[k, np.arange(n), y_cp].mean()
        print(f"  key {k+1:2d}: tau_e={tau_e:6.4f}  coverage={100*cov:5.2f}%  "
              f"mean|set|={sets[k].sum(1).mean():4.2f}")
    return sets, y_cp

# ----------------------------------------------------------------------------
def report(sets, y_cp):
    cov_k, size_k = per_key_metrics(sets, y_cp)
    inter = intersection_metrics(sets, y_cp)
    uni   = union_metrics(sets, y_cp)
    bonf, indep = theory_bounds(cov_k)

    print("\n=== Per-key (BB) ===")
    print(f"  coverage  mean {100*cov_k.mean():.2f}%  (min {100*cov_k.min():.2f}%, "
          f"max {100*cov_k.max():.2f}%)")
    print(f"  set size  mean {size_k.mean():.2f}")

    print("\n=== Intersection of all "
          f"{sets.shape[0]} keys ===")
    print(f"  coverage           : {100*inter['coverage']:.2f}%")
    print(f"  mean size          : {inter['mean_size']:.3f}")
    print(f"  empty intersections: {100*inter['empty_frac']:.2f}%")
    print(f"  size distribution (0..10): {inter['size_hist'].tolist()}")

    print("\n=== Theory references ===")
    print(f"  Bonferroni lower bound on coverage : {100*bonf:.2f}%")
    print(f"  independence estimate of coverage  : {100*indep:.2f}%")
    print(f"  (realized {100*inter['coverage']:.2f}% should exceed the independence "
          "estimate if key errors are positively correlated)")

    print("\n=== Union of all keys (for contrast) ===")
    print(f"  coverage {100*uni['coverage']:.2f}%   mean size {uni['mean_size']:.2f}")

    print("\n=== Sweep: intersection of first K keys (averaged over random K-subsets) ===")
    print(f"  {'K':>2s}  {'mean|set|':>9s}  {'coverage%':>9s}  {'empty%':>7s}")
    sw = sweep_over_K(sets, y_cp)
    for k in sorted(sw):
        s, c, e = sw[k]
        print(f"  {k:2d}  {s:9.3f}  {100*c:9.2f}  {100*e:7.2f}")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = sorted(sw)
        sizes = [sw[k][0] for k in ks]; covs = [100*sw[k][1] for k in ks]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(ks, sizes, "o-", color="tab:blue"); ax1.set_xlabel("number of keys K")
        ax1.set_ylabel("mean intersection size", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ks, covs, "s--", color="tab:red"); ax2.set_ylabel("coverage (%)", color="tab:red")
        fig.tight_layout(); fig.savefig("intersection_tradeoff.png", dpi=150)
        print("\nSaved plot: intersection_tradeoff.png")
    except Exception as ex:
        print(f"\n(plot skipped: {ex})")


def report_checkpoints(sets, y_cp, checkpoints):
    print("\n=== Key-Scaling Checkpoints ===")
    print("This section is intended for direct inclusion in a paper/report.")
    print(f"  {'K':>3s}  {'cover%':>8s}  {'mean|int|':>9s}  {'empty%':>7s}  {'Bonf%':>7s}  {'Indep%':>8s}")
    for k in checkpoints:
        sub_sets = sets[:k]
        sub_cov, _ = per_key_metrics(sub_sets, y_cp)
        inter = intersection_metrics(sub_sets, y_cp)
        bonf, indep = theory_bounds(sub_cov)
        print(
            f"  {k:3d}  {100*inter['coverage']:8.2f}  {inter['mean_size']:9.3f}  "
            f"{100*inter['empty_frac']:7.2f}  {100*bonf:7.2f}  {100*indep:8.2f}"
        )

    print("\n=== Theoretical Interpretation (for LaTeX write-up) ===")
    print("Let C_k(x) be the BB prediction set under key k and C_cap^(K)(x)=intersection_{k=1}^K C_k(x).")
    print("As K increases, |C_cap^(K)(x)| is non-increasing by construction, so ambiguity should shrink.")
    print("Coverage of C_cap^(K) can decrease with K, but empirically often remains high due to correlated errors.")
    print("Bonferroni bound: P(y in C_cap^(K)) >= 1 - sum_{k=1}^K alpha_k, where alpha_k=1-coverage_k.")
    print("Independence estimate: P(y in C_cap^(K)) ~= prod_{k=1}^K (1-alpha_k).")
    print("When realized coverage is much higher than the independence estimate, key-wise misses are positively correlated.")
    print("This means many hard samples are shared across keys rather than missed independently.")
    print("Practical takeaway: increasing K can substantially reduce set size while preserving useful coverage.")

# ----------------------------------------------------------------------------
def selftest():
    """Validate the analysis layer on hand-checkable synthetic sets."""
    # 2 keys, 3 images, 4 labels. y = [0, 1, 2].
    y = np.array([0, 1, 2])
    s0 = np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1]], dtype=bool)   # key 0
    s1 = np.array([[1,0,1,0],[0,1,0,1],[1,0,0,1]], dtype=bool)   # key 1
    sets = np.stack([s0, s1])                                    # [2,3,4]
    # intersections: img0 {0}; img1 {1}; img2 {3}
    m = intersection_metrics(sets, y)
    assert np.isclose(m['mean_size'], (1+1+1)/3), m
    assert np.isclose(m['coverage'], 2/3), m            # img0 yes(0), img1 yes(1), img2 no(true 2, inter {3})
    assert np.isclose(m['empty_frac'], 0.0), m
    cov, size = per_key_metrics(sets, y)
    assert np.allclose(cov, [3/3, 2/3]), cov            # key0 covers all; key1 misses img2
    bonf, indep = theory_bounds(cov)
    assert np.isclose(indep, 1.0 * (2/3)), indep
    assert np.isclose(bonf, max(0.0, 1 - (0 + 1/3))), bonf
    u = union_metrics(sets, y)
    assert np.isclose(u['coverage'], 1.0), u            # unions contain the truth for all 3
    sw = sweep_over_K(sets, y)
    assert set(sw.keys()) == {1, 2}
    print("selftest passed: analysis layer is correct.")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intersection analysis for BB sets across multiple keys")
    parser.add_argument("--selftest", action="store_true", help="Run analysis-layer self-test")
    parser.add_argument("--num-keys", type=int, default=NUM_KEYS, help="Number of fixed keys to train")
    parser.add_argument("--n", type=int, default=N, help="Calibration size (also CP-test size)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs per key")
    parser.add_argument("--batch", type=int, default=BATCH, help="Training batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="10,20,30,40,50",
        help="Comma-separated K values for checkpoint summary (e.g., 10,20,30,40,50)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest()
    else:
        if args.num_keys < 1:
            raise ValueError("--num-keys must be >= 1")
        checkpoints = parse_checkpoints(args.checkpoints, args.num_keys)
        print(
            f"Building BB prediction sets for {args.num_keys} keys "
            f"(alpha={ALPHA}, n={args.n}, epochs={args.epochs}); "
            f"this trains {args.num_keys} models..."
        )
        sets, y_cp = build_sets(
            num_keys=args.num_keys,
            n=args.n,
            seed=args.seed,
            epochs=args.epochs,
            batch=args.batch,
        )
        report(sets, y_cp)
        report_checkpoints(sets, y_cp, checkpoints)