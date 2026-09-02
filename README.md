# Conformal Prediction for Privacy-Preserving Machine Learning

Code accompanying the paper

> A. D. Balinsky, D. Krzemiński and A. Balinsky,
> *Conformal Prediction for Privacy-Preserving Machine Learning:
> Uncertainty Quantification on Encrypted Data*,
> Applied Mathematics & Information Sciences (2026).

The paper shows that a fixed deterministic cipher preserves exchangeability,
so conformal prediction (both the classical *p*-value predictor and the
*e*-value BB-predictor) remains exactly valid on AES-encrypted data, and
studies the resulting compactness–reliability trade-off on encrypted MNIST.

## Contents

| File | What it does | Produces (paper) |
|---|---|---|
| `AlexDominikCOPA2025.ipynb` | Main pipeline: AES-CBC encryption of MNIST (fixed key and per-image keys), training the feed-forward model, inductive CP with *e*-test (BB) and *p*-value statistics | Abstract numbers, Sections 5.2–5.4, Tables 1–2, Figures 1, 3, 4 |
| `ablation_keys.py` | Robustness-to-key-variation ablation: repeats the full pipeline for 10 fixed (key, IV) pairs plus plaintext and per-image-key boundary cases | Table 3 |
| `intersection_claude.py` | Intersection of BB prediction sets across K independent encryption keys, with Bonferroni / independence reference bounds, checkpoint summaries and plots; includes a unit-tested analysis layer (`--selftest`) | Table 4, Figures 5–6 |
| `intersection_bb_keys.py` | Smaller standalone version of the intersection experiment | — |
| `make_model_figure.py` | Draws the network-architecture diagram | Figure 2 |
| `ablation_keys_output.txt` | Saved output of `ablation_keys.py` (the run reported in the paper) | Table 3 |
| `intersection_claude_50keys_complete.txt` | Saved output of the completed K=50 run of `intersection_claude.py` | Table 4, Figures 5–6 |
| `key_scaling_50keys.png` | Per-key coverage / set-size stability across 50 keys | Figure 6 |

## Requirements

```bash
pip install torch torchvision numpy pycryptodome scikit-learn matplotlib
```

Python ≥ 3.10. A GPU is used automatically when available (strongly
recommended for the multi-key scripts, which train one model per key).

## Reproducing the paper

**Main results (Tables 1–2, Sections 5.2–5.4).** Run
`AlexDominikCOPA2025.ipynb` top to bottom. The notebook fixes the seed
(`2024`), the AES key/IV
(`key = b'abs2kas126oZbdXs'`, `iv = b'1nsdjah72MdnJ12a'`) and uses AES in CBC
mode (28×28 = 784 bytes = 49 AES blocks, so no padding is needed).

**Key-robustness ablation (Table 3).**

```bash
python ablation_keys.py
```

**Key-scaling / intersection study (Table 4, Figures 5–6).**

```bash
python intersection_claude.py --selftest        # fast: checks the analysis layer
python intersection_claude.py --num-keys 50     # trains 50 models (long)
```

The K=50 run writes `intersection_tradeoff.png` (Figure 5) and prints the
checkpoint table used in the paper.

**Architecture figure (Figure 2).**

```bash
python make_model_figure.py
```

## Method in one paragraph

Every record is encrypted with the *same* fixed (key, IV), so encryption is a
single fixed measurable map applied identically to each observation. Such a map
commutes with permutations and therefore preserves exchangeability — the only
assumption conformal prediction needs — so coverage guarantees transfer to the
encrypted domain unchanged (Lemma 1 of the paper). The BB *e*-value predictor
admits a label when its loss is below
`(1/α) · (1 / (1 + (1 − 1/α)/n)) · mean(calibration losses)`;
the *p*-value predictor uses the ⌈(1−α)(n+1)⌉-th smallest calibration loss.
Determinism is what makes learning on ciphertext possible, and is also the
security price: the scheme is not IND-CPA and is intended against a
curious-but-passive observer (Section 3.4 of the paper).

## Note on data

MNIST is downloaded automatically by `torchvision` on first run; the raw data
is not tracked in this repository.
