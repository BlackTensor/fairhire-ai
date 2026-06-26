"""FairHire AI — Phase 2 baseline (no privacy).

Pipeline:
  1. Train a plain Encoder + Hiring Predictor on the main task (income >50K) to
     convergence, keeping the best-val checkpoint.
  2. Record hiring accuracy on val and test.
  3. Freeze the encoder, extract the latent ``z``, and train a standalone
     attacker probe per sensitive attribute (gender / age / ethnicity) to
     measure how much each leaks out of the *unsanitized* latent.
  4. Save the "Before" numbers to ``logs/baseline.csv`` and the model +
     metrics to ``models/baseline.pt``.

This establishes the "Before" column that Phase 3's adversarial sanitization is
measured against. The probes deliberately reuse the same MLP head class as the
Phase 3 auditors, so any leakage drop later is due to the sanitization, not a
weaker attacker.

Run:  python -m src.train_baseline
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import SENSITIVE_KEYS, make_dataloaders
from .models import SENSITIVE_N_CLASSES, Z_DIM, BaselineModel, MLPHead

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters.
MAX_EPOCHS = 60
PATIENCE = 8          # early-stop on val accuracy
LR = 1e-3
WEIGHT_DECAY = 1e-4
PROBE_EPOCHS = 40     # standalone attacker probe budget
PROBE_LR = 1e-3


def _seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _accuracy(model: BaselineModel, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def _extract_latent(encoder: nn.Module, loader) -> dict[str, np.ndarray]:
    """Run the frozen encoder over a loader, returning z and all labels."""
    encoder.eval()
    zs, mains, sens = [], [], {k: [] for k in SENSITIVE_KEYS}
    for x, y, s in loader:
        z = encoder(x.to(DEVICE)).cpu().numpy()
        zs.append(z)
        mains.append(y.numpy())
        for k in SENSITIVE_KEYS:
            sens[k].append(s[k].numpy())
    out = {"z": np.concatenate(zs), "main": np.concatenate(mains)}
    for k in SENSITIVE_KEYS:
        out[k] = np.concatenate(sens[k])
    return out


# --------------------------------------------------------------------------- #
# Main-task training
# --------------------------------------------------------------------------- #
def train_main_task(loaders) -> tuple[BaselineModel, dict[str, float]]:
    in_dim = loaders["train"].dataset.n_features
    model = BaselineModel(in_dim, z_dim=Z_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y, _ in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        val_acc = _accuracy(model, loaders["val"])
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = "  *best"
        else:
            epochs_without_improvement += 1
        print(f"  epoch {epoch:3d}  val_acc={val_acc:.4f}{marker}")
        if epochs_without_improvement >= PATIENCE:
            print(f"  early stop at epoch {epoch} (no val gain in {PATIENCE})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    test_acc = _accuracy(model, loaders["test"])
    return model, {"main_acc_val": best_val, "main_acc_test": test_acc}


# --------------------------------------------------------------------------- #
# Leakage probes on the frozen latent
# --------------------------------------------------------------------------- #
def _train_probe(z_train, y_train, n_classes) -> MLPHead:
    probe = MLPHead(z_train.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    zt = torch.as_tensor(z_train, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
    n = zt.shape[0]
    batch = 256
    for _ in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = criterion(probe(zt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return probe


@torch.no_grad()
def _probe_accuracy(probe: MLPHead, z_test, y_test) -> float:
    probe.eval()
    zt = torch.as_tensor(z_test, dtype=torch.float32, device=DEVICE)
    pred = probe(zt).argmax(1).cpu().numpy()
    return float((pred == y_test).mean())


def measure_leakage(encoder, loaders) -> dict[str, dict[str, float]]:
    """Train one attacker probe per sensitive attribute on the frozen latent."""
    train_lat = _extract_latent(encoder, loaders["train"])
    test_lat = _extract_latent(encoder, loaders["test"])

    results: dict[str, dict[str, float]] = {}
    for key in SENSITIVE_KEYS:
        n_classes = SENSITIVE_N_CLASSES[key]
        probe = _train_probe(train_lat["z"], train_lat[key], n_classes)
        acc = _probe_accuracy(probe, test_lat["z"], test_lat[key])
        baseline = float(np.bincount(test_lat[key]).max() / len(test_lat[key]))
        results[key] = {
            "probe_acc": acc,
            "majority_baseline": baseline,
            "lift": acc - baseline,
        }
    return results


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_baseline(model, main_metrics, leakage, in_dim) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = LOGS_DIR / "baseline.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "attribute", "value"])
        w.writerow(["main_acc_val", "", f"{main_metrics['main_acc_val']:.6f}"])
        w.writerow(["main_acc_test", "", f"{main_metrics['main_acc_test']:.6f}"])
        for key in SENSITIVE_KEYS:
            r = leakage[key]
            w.writerow(["leakage_probe_acc", key, f"{r['probe_acc']:.6f}"])
            w.writerow(["leakage_majority", key, f"{r['majority_baseline']:.6f}"])
            w.writerow(["leakage_lift", key, f"{r['lift']:.6f}"])
    print(f"Saved baseline metrics -> {csv_path}")

    ckpt_path = MODELS_DIR / "baseline.pt"
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "predictor_state": model.predictor.state_dict(),
            "in_dim": in_dim,
            "z_dim": Z_DIM,
            "main_metrics": main_metrics,
            "leakage": leakage,
        },
        ckpt_path,
    )
    print(f"Saved baseline checkpoint -> {ckpt_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    _seed_everything()
    print("=" * 70)
    print("PHASE 2 BASELINE (no privacy)")
    print(f"device={DEVICE}  z_dim={Z_DIM}")
    print("=" * 70)

    loaders, _ = make_dataloaders(batch_size=256)
    in_dim = loaders["train"].dataset.n_features

    print("\n[1] Training Encoder + Hiring Predictor")
    model, main_metrics = train_main_task(loaders)
    print(f"\n  main-task accuracy  val={main_metrics['main_acc_val']:.4f}  "
          f"test={main_metrics['main_acc_test']:.4f}")

    print("\n[2] Baseline leakage — attacker probes on the frozen latent z")
    leakage = measure_leakage(model.encoder, loaders)
    print("    (probe acc well above majority baseline = the latent leaks)")
    for key in SENSITIVE_KEYS:
        r = leakage[key]
        print(f"    {key:10s}: probe acc={r['probe_acc']:.4f}   "
              f"majority={r['majority_baseline']:.4f}   "
              f"lift=+{r['lift']:.4f}")

    save_baseline(model, main_metrics, leakage, in_dim)
    print("\nPhase 2 baseline complete. This is the 'Before' column.")


if __name__ == "__main__":
    main()
