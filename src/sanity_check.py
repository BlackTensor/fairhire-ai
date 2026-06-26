"""FairHire AI — Phase 1 sanity checks.

Reports, for the processed Adult bundle:
  1. Split sizes and main-task (income >50K) class balance per split.
  2. Sensitive-attribute distributions (gender, age bucket, ethnicity).
  3. Proxy leakage: how well a simple linear probe recovers each sensitive
     attribute *from the input features alone* (sensitive columns are excluded
     from the features). High accuracy vs. the majority baseline confirms the
     premise of the project -- proxies exist, which is why adversarial
     scrubbing of the latent is needed.
  4. Basic feature stats (shape, NaN/inf check, continuous-column ranges).

Run:  python -m src.sanity_check
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import (
    AGE_BUCKET_CLASSES, ETHNICITY_CLASSES, GENDER_CLASSES, SENSITIVE_KEYS,
    build_processed,
)

_CLASS_NAMES = {
    "gender": GENDER_CLASSES,
    "age": AGE_BUCKET_CLASSES,
    "ethnicity": ETHNICITY_CLASSES,
}


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _dist(arr: np.ndarray, names: list[str]) -> str:
    counts = np.bincount(arr, minlength=len(names))
    total = counts.sum()
    parts = [f"{n}={c} ({_pct(c / total)})" for n, c in zip(names, counts)]
    return "  ".join(parts)


def main() -> None:
    bundle = build_processed()

    print("=" * 70)
    print("PHASE 1 SANITY CHECK - UCI Adult")
    print("=" * 70)

    feature_names = bundle["feature_names"]
    print(f"\nFeature matrix width: {len(feature_names)} columns "
          f"(sensitive cols excluded from input)")

    # --- 1. Split sizes + main-task balance ------------------------------- #
    print("\n[1] Split sizes and main-task (income >50K) balance")
    for split in ("train", "val", "test"):
        X = bundle[f"X_{split}"]
        y = bundle[f"main_{split}"]
        pos = y.mean()
        print(f"    {split:5s}: n={X.shape[0]:6d}   >50K rate={_pct(pos)}")

    # --- 2. Sensitive distributions (train) ------------------------------- #
    print("\n[2] Sensitive-attribute distribution (train split)")
    for key in SENSITIVE_KEYS:
        arr = bundle[f"{key}_train"]
        print(f"    {key:10s}: {_dist(arr, _CLASS_NAMES[key])}")

    # --- 3. Proxy leakage via linear probe -------------------------------- #
    print("\n[3] Proxy leakage - linear probe on input features (test acc)")
    print("    A score well above the majority baseline = proxy info present.")
    Xtr = bundle["X_train"]
    Xte = bundle["X_test"]
    for key in SENSITIVE_KEYS:
        ytr = bundle[f"{key}_train"]
        yte = bundle[f"{key}_test"]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        acc = clf.score(Xte, yte)
        baseline = np.bincount(yte).max() / len(yte)
        lift = acc - baseline
        print(f"    {key:10s}: probe acc={_pct(acc)}   "
              f"majority baseline={_pct(baseline)}   lift=+{_pct(lift)}")

    # --- 4. Basic feature stats ------------------------------------------- #
    print("\n[4] Feature-matrix health check")
    all_X = np.concatenate(
        [bundle[f"X_{s}"] for s in ("train", "val", "test")], axis=0
    )
    print(f"    combined shape: {all_X.shape}")
    print(f"    NaNs: {np.isnan(all_X).sum()}   Infs: {np.isinf(all_X).sum()}")
    print(f"    global min/max: {all_X.min():.3f} / {all_X.max():.3f}")
    print("    continuous columns (train, post-standardization):")
    for i, name in enumerate(bundle["continuous_features"]):
        col = bundle["X_train"][:, i]
        print(f"      {name:16s} mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"min={col.min():+.2f}  max={col.max():+.2f}")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()
