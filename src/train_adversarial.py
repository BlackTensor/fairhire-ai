"""FairHire AI — Phase 3 adversarial sanitization (the core contribution).

Trains the Encoder + Hiring Predictor jointly with two adversarial auditors
(gender and age) attacking a shared latent ``z`` through a Gradient Reversal
Layer. The encoder is pushed to keep ``z`` useful for the hiring task while
making it useless for the auditors -- "Multi-Adversarial Representation
Sanitization."

Why gender + age only (no ethnicity): the Phase 2 baseline showed an ethnicity
leakage lift of +0.3% over a 0.859 majority floor (the dataset is ~86% White),
so there is essentially no recoverable signal to scrub. Ethnicity is still
*measured* in the "After" column for completeness, but it is not a sanitization
target. See the CLAUDE.md decisions log.

Pipeline:
  1. For each lambda in a small sweep grid, train the sanitized model. Each
     batch does ``N_CRITIC`` auditor-only updates on the current (detached)
     latent so the adversaries stay near-optimal, then one encoder+predictor
     update whose gradient is reversed through the GRL. A near-optimal critic
     is what forces the encoder to *actually remove* the attribute rather than
     just fool one weak auditor instance. Lambda is ramped from 0 to its max
     over training (DANN-style warmup) for stability.
  2. Freeze the encoder and re-train fresh standalone probes per attribute --
     the same identical-capacity attackers used for the "Before" column -- to
     get the honest leakage of the *sanitized* latent.
  3. Select the lambda at the accuracy-vs-leakage knee: lowest mean adversarial
     leakage (gender + age) subject to main accuracy staying within a tolerance
     of the baseline. Save that model + its "After" numbers.

Outputs:
  * ``logs/adversarial_sweep.csv``  -- per-lambda main acc + leakage.
  * ``logs/adversarial.csv``        -- the chosen "After" column (mirrors
                                       baseline.csv, plus the chosen lambda).
  * ``models/sanitized.pt``         -- the chosen sanitized model + metrics.

Run:  python -m src.train_adversarial
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import SENSITIVE_KEYS, make_dataloaders
from .models import Z_DIM, SanitizedModel
from .train_baseline import DEVICE, _seed_everything, measure_leakage

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

# Which attributes get an adversary. Ethnicity is intentionally excluded as a
# sanitization target (see module docstring); it is still measured downstream.
ADVERSARIES: tuple[str, ...] = ("gender", "age")

# Training hyperparameters.
MAX_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
ADV_LR = 1e-3         # auditor (critic) learning rate
N_CRITIC = 5          # auditor updates per encoder update -> a strong adversary

# The tunable adversarial weight. We sweep these and pick the knee.
LAMBDA_GRID = (0.5, 1.0, 2.0, 4.0)
RAMP_GAMMA = 10.0          # steepness of the 0 -> lambda_max warmup

# Model selection: allow main accuracy to drop at most this far below the
# baseline val accuracy in exchange for the largest leakage reduction.
BASELINE_VAL_ACC = 0.859405      # from logs/baseline.csv (Phase 2)
MAIN_ACC_TOLERANCE = 0.02
# Only consider epochs after the warmup has largely completed for the best-acc
# checkpoint, so the saved encoder is actually sanitized (lambda near max).
SELECT_AFTER_FRAC = 0.5


def _lambda_at(epoch: int, max_epochs: int, lambda_max: float) -> float:
    """DANN warmup: ramp lambda from 0 to ``lambda_max`` over training."""
    p = epoch / max(1, max_epochs - 1)
    ramp = 2.0 / (1.0 + math.exp(-RAMP_GAMMA * p)) - 1.0
    return lambda_max * ramp


@torch.no_grad()
def _main_accuracy(model: SanitizedModel, loader) -> float:
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        main_logits, _, _ = model(x)
        correct += (main_logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def _adversary_accuracies(model: SanitizedModel, loader) -> dict[str, float]:
    """In-training auditor accuracy (a weakened lower bound on leakage -- the
    GRL is actively undermining these heads, which is the point)."""
    model.eval()
    correct = {k: 0 for k in model.adversaries}
    total = 0
    for x, _, s in loader:
        x = x.to(DEVICE)
        _, _, aud = model(x)
        for k in model.adversaries:
            pred = aud[k].argmax(1).cpu()
            correct[k] += (pred == s[k]).sum().item()
        total += x.shape[0]
    return {k: correct[k] / total for k in model.adversaries}


def train_one_lambda(loaders, lambda_max: float) -> SanitizedModel:
    """Train a sanitized model for a single lambda_max, returning the best
    sanitized checkpoint (best main val acc among the warmed-up epochs).

    Two optimizers: ``opt_enc`` owns the encoder + hiring predictor (it sees the
    main-task gradient plus the GRL-reversed adversarial gradient), and
    ``opt_adv`` owns the auditors and is stepped ``N_CRITIC`` times per batch on
    the detached latent so the adversaries stay strong.
    """
    in_dim = loaders["train"].dataset.n_features
    model = SanitizedModel(
        in_dim, adversaries=ADVERSARIES, z_dim=Z_DIM, lambda_=0.0
    ).to(DEVICE)
    enc_params = (list(model.encoder.parameters())
                  + list(model.predictor.parameters()))
    opt_enc = torch.optim.Adam(enc_params, lr=LR, weight_decay=WEIGHT_DECAY)
    opt_adv = torch.optim.Adam(
        model.auditors.parameters(), lr=ADV_LR, weight_decay=WEIGHT_DECAY
    )
    main_criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss()

    select_after = int(MAX_EPOCHS * SELECT_AFTER_FRAC)
    best_val = -1.0
    best_state: dict | None = None

    for epoch in range(MAX_EPOCHS):
        model.grl.lambda_ = _lambda_at(epoch, MAX_EPOCHS, lambda_max)
        model.train()
        for x, y, s in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            s = {k: v.to(DEVICE) for k, v in s.items()}

            # (a) Strengthen the auditors on the current latent (detached, so
            #     only the auditor heads move).
            for _ in range(N_CRITIC):
                with torch.no_grad():
                    z_det = model.encoder(x)
                opt_adv.zero_grad()
                adv_loss = sum(adv_criterion(head(z_det), s[k])
                               for k, head in model.auditors.items())
                adv_loss.backward()
                opt_adv.step()

            # (b) Update encoder + predictor: minimize main loss while the GRL
            #     reverses the adversarial gradient into the encoder.
            opt_enc.zero_grad()
            main_logits, _, aud_logits = model(x)
            loss = main_criterion(main_logits, y)
            for k in model.adversaries:
                loss = loss + adv_criterion(aud_logits[k], s[k])
            loss.backward()
            opt_enc.step()

        val_acc = _main_accuracy(model, loaders["val"])
        adv_acc = _adversary_accuracies(model, loaders["val"])
        adv_str = "  ".join(f"{k}={a:.3f}" for k, a in adv_acc.items())
        marker = ""
        # Only snapshot once the GRL has warmed up, so the saved encoder is
        # genuinely sanitized rather than an early under-pressure epoch.
        if epoch >= select_after and val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            marker = "  *best"
        print(f"    epoch {epoch + 1:3d}  lambda={model.grl.lambda_:.3f}  "
              f"val_acc={val_acc:.4f}  aud[{adv_str}]{marker}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.grl.lambda_ = lambda_max
    return model


def evaluate(model: SanitizedModel, loaders) -> dict:
    """Main accuracy (val/test) + honest leakage on the frozen sanitized z."""
    main_val = _main_accuracy(model, loaders["val"])
    main_test = _main_accuracy(model, loaders["test"])
    leakage = measure_leakage(model.encoder, loaders)
    return {
        "main_acc_val": main_val,
        "main_acc_test": main_test,
        "leakage": leakage,
    }


def _mean_adv_lift(leakage: dict[str, dict[str, float]]) -> float:
    return float(np.mean([leakage[k]["lift"] for k in ADVERSARIES]))


def select_best(results: list[dict]) -> dict:
    """Pick the accuracy-vs-leakage knee: lowest mean adversarial leakage lift
    among runs whose main val acc is within tolerance of the baseline; if none
    qualify, fall back to the highest-accuracy run."""
    floor = BASELINE_VAL_ACC - MAIN_ACC_TOLERANCE
    eligible = [r for r in results if r["metrics"]["main_acc_val"] >= floor]
    pool = eligible if eligible else results
    key = ((lambda r: _mean_adv_lift(r["metrics"]["leakage"])) if eligible
           else (lambda r: -r["metrics"]["main_acc_val"]))
    return min(pool, key=key)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_sweep(results: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "adversarial_sweep.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["lambda", "main_acc_val", "main_acc_test"]
        for k in SENSITIVE_KEYS:
            header += [f"leak_acc_{k}", f"leak_lift_{k}"]
        header += ["mean_adv_lift"]
        w.writerow(header)
        for r in results:
            m = r["metrics"]
            row = [f"{r['lambda']:.3f}",
                   f"{m['main_acc_val']:.6f}", f"{m['main_acc_test']:.6f}"]
            for k in SENSITIVE_KEYS:
                row += [f"{m['leakage'][k]['probe_acc']:.6f}",
                        f"{m['leakage'][k]['lift']:.6f}"]
            row += [f"{_mean_adv_lift(m['leakage']):.6f}"]
            w.writerow(row)
    print(f"Saved lambda sweep -> {path}")


def save_after(chosen: dict, in_dim: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    m = chosen["metrics"]
    leakage = m["leakage"]

    csv_path = LOGS_DIR / "adversarial.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "attribute", "value"])
        w.writerow(["chosen_lambda", "", f"{chosen['lambda']:.6f}"])
        w.writerow(["adversaries", "", "+".join(ADVERSARIES)])
        w.writerow(["main_acc_val", "", f"{m['main_acc_val']:.6f}"])
        w.writerow(["main_acc_test", "", f"{m['main_acc_test']:.6f}"])
        for key in SENSITIVE_KEYS:
            r = leakage[key]
            w.writerow(["leakage_probe_acc", key, f"{r['probe_acc']:.6f}"])
            w.writerow(["leakage_majority", key, f"{r['majority_baseline']:.6f}"])
            w.writerow(["leakage_lift", key, f"{r['lift']:.6f}"])
    print(f"Saved 'After' metrics -> {csv_path}")

    model = chosen["model"]
    ckpt_path = MODELS_DIR / "sanitized.pt"
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "predictor_state": model.predictor.state_dict(),
            "auditors_state": model.auditors.state_dict(),
            "adversaries": list(ADVERSARIES),
            "lambda": chosen["lambda"],
            "in_dim": in_dim,
            "z_dim": Z_DIM,
            "main_metrics": {"main_acc_val": m["main_acc_val"],
                             "main_acc_test": m["main_acc_test"]},
            "leakage": leakage,
        },
        ckpt_path,
    )
    print(f"Saved sanitized checkpoint -> {ckpt_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    _seed_everything()
    print("=" * 70)
    print("PHASE 3 ADVERSARIAL SANITIZATION")
    print(f"device={DEVICE}  z_dim={Z_DIM}  adversaries={'+'.join(ADVERSARIES)}")
    print(f"lambda grid={LAMBDA_GRID}")
    print("=" * 70)

    loaders, _ = make_dataloaders(batch_size=256)
    in_dim = loaders["train"].dataset.n_features

    results: list[dict] = []
    for lambda_max in LAMBDA_GRID:
        print(f"\n[lambda={lambda_max}] training sanitized model")
        _seed_everything()  # identical init/shuffle per lambda for a fair sweep
        model = train_one_lambda(loaders, lambda_max)
        metrics = evaluate(model, loaders)
        print(f"  -> main val={metrics['main_acc_val']:.4f}  "
              f"test={metrics['main_acc_test']:.4f}")
        for k in SENSITIVE_KEYS:
            r = metrics["leakage"][k]
            tag = "ADV" if k in ADVERSARIES else "measured-only"
            print(f"     {k:10s} [{tag:13s}]: probe acc={r['probe_acc']:.4f}  "
                  f"majority={r['majority_baseline']:.4f}  lift=+{r['lift']:.4f}")
        results.append({"lambda": lambda_max, "model": model, "metrics": metrics})

    save_sweep(results)

    chosen = select_best(results)
    print("\n" + "=" * 70)
    print(f"CHOSEN lambda={chosen['lambda']}  "
          f"(knee: lowest gender+age leakage within "
          f"{MAIN_ACC_TOLERANCE:.0%} of baseline acc)")
    m = chosen["metrics"]
    print(f"  main acc  val={m['main_acc_val']:.4f}  test={m['main_acc_test']:.4f}"
          f"  (baseline val={BASELINE_VAL_ACC:.4f})")
    print("  After leakage:")
    for k in SENSITIVE_KEYS:
        r = m["leakage"][k]
        print(f"    {k:10s}: lift=+{r['lift']:.4f}")
    print("=" * 70)

    save_after(chosen, in_dim)
    print("\nPhase 3 complete. This is the 'After' column.")


if __name__ == "__main__":
    main()
