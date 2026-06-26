"""FairHire AI — Phase 5 export: bundle the trained models + data for the web UI.

The UI (``ui/``) is a static HTML/CSS/JS shell hosted on GitHub Pages, so it
cannot run PyTorch. Instead we export everything the browser needs to run the
model itself:

  * ``ui/model.json``  — both encoders + hiring predictors (Before / After) as
    plain nested-array weights, plus the preprocessing schema (scaler stats,
    categorical vocabularies, feature order) and the gender-axis direction used
    by the counterfactual-flip demo. The networks are tiny MLPs, so a ~40-line
    JS forward pass reproduces them exactly (parity checked below).
  * ``ui/data.json``   — precomputed t-SNE coordinates of the latent Before and
    After (the signature separated -> merged animation), the four headline plot
    numbers (accuracy, leakage, fairness), and the counterfactual swing
    distributions for the honest-caveat histogram.

Run:  python -m src.export_ui
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import make_dataloaders
from .evaluate import (
    _gender_axis,
    _axis_swing,
    _read_leakage,
    encode_test,
    fairness_metrics,
    load_encoder_predictor,
    score_test,
    _tsne,
)

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

SEED = 42
TSNE_N = 2000          # subsample for the latent animation (speed + payload)
SWING_N = 1500         # subsample for the counterfactual histogram


# --------------------------------------------------------------------------- #
# Weight extraction
# --------------------------------------------------------------------------- #
def _round(arr: np.ndarray, ndigits: int = 6):
    """Round + convert to nested python lists to keep model.json compact."""
    return np.round(np.asarray(arr, dtype=np.float64), ndigits).tolist()


def _linear(state: dict, prefix: str) -> dict:
    return {"W": _round(state[f"{prefix}.weight"]),   # [out, in]
            "b": _round(state[f"{prefix}.bias"])}


def _batchnorm(state: dict, prefix: str, eps: float = 1e-5) -> dict:
    return {
        "weight": _round(state[f"{prefix}.weight"]),
        "bias": _round(state[f"{prefix}.bias"]),
        "mean": _round(state[f"{prefix}.running_mean"]),
        "var": _round(state[f"{prefix}.running_var"]),
        "eps": eps,
    }


def _export_model(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc, pred = ckpt["encoder_state"], ckpt["predictor_state"]
    return {
        "encoder": {
            "l1": _linear(enc, "net.0"),
            "bn1": _batchnorm(enc, "net.1"),
            "l2": _linear(enc, "net.4"),
            "bn2": _batchnorm(enc, "net.5"),
            "l3": _linear(enc, "net.8"),
        },
        "predictor": {
            "l1": _linear(pred, "net.0"),
            "l2": _linear(pred, "net.3"),
        },
    }


# --------------------------------------------------------------------------- #
# NumPy reference forward (mirrors the JS we are about to write) — used to
# verify parity against PyTorch so the browser scores match the paper.
# --------------------------------------------------------------------------- #
def _np_linear(x, lin):
    return x @ np.asarray(lin["W"]).T + np.asarray(lin["b"])


def _np_bn(x, bn):
    return (x - np.asarray(bn["mean"])) / np.sqrt(
        np.asarray(bn["var"]) + bn["eps"]
    ) * np.asarray(bn["weight"]) + np.asarray(bn["bias"])


def _np_relu(x):
    return np.maximum(x, 0.0)


def _np_score(model: dict, X: np.ndarray) -> np.ndarray:
    e = model["encoder"]
    h = _np_relu(_np_bn(_np_linear(X, e["l1"]), e["bn1"]))
    h = _np_relu(_np_bn(_np_linear(h, e["l2"]), e["bn2"]))
    z = _np_linear(h, e["l3"])
    p = model["predictor"]
    logits = _np_linear(_np_relu(_np_linear(z, p["l1"])), p["l2"])
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm = ex / ex.sum(axis=1, keepdims=True)
    return sm[:, 1]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("PHASE 5 — EXPORT MODELS + DATA FOR THE WEB UI")
    print("=" * 70)

    loaders, bundle = make_dataloaders(batch_size=256)
    ds = loaders["test"].dataset
    X = ds.X
    Xn = X.numpy().astype(np.float64)
    y_true = ds.main.numpy()
    gender = ds.sensitive["gender"].numpy()
    feature_names = [str(s) for s in bundle["feature_names"]]
    continuous = [str(s) for s in bundle["continuous_features"]]

    # Categorical vocabularies, reconstructed from the one-hot feature names so
    # the UI dropdowns offer exactly the categories the model was trained on.
    categories: dict[str, list[str]] = {}
    cat_cols: dict[str, list[int]] = {}
    for j, name in enumerate(feature_names):
        if "=" in name:
            col, val = name.split("=", 1)
            categories.setdefault(col, []).append(val)
            cat_cols.setdefault(col, []).append(j)

    # Most-common (mode) category per field, from the TRAIN split. The UI uses
    # these as silent defaults for any field it deliberately does NOT expose
    # (e.g. 'relationship', a direct gender proxy that is excluded from the
    # candidate form on principle — see CLAUDE.md).
    X_train = bundle["X_train"]
    silent_defaults: dict[str, str] = {}
    for col, cols in cat_cols.items():
        counts = X_train[:, cols].sum(axis=0)
        silent_defaults[col] = categories[col][int(np.argmax(counts))]

    print("\n[1] Exporting model weights (Before / After)")
    models = {
        "baseline": _export_model(MODELS_DIR / "baseline.pt"),
        "sanitized": _export_model(MODELS_DIR / "sanitized.pt"),
    }

    print("[2] Gender axis for the counterfactual-flip demo")
    w_axis, step = _gender_axis(Xn, gender)

    model_json = {
        "meta": {
            "in_dim": len(feature_names),
            "z_dim": int(bundle["X_test"].shape[1] and 32),
            "note": "Encoder + hiring predictor weights for in-browser scoring.",
        },
        "scaler": {
            "mean": _round(bundle["scaler_mean"], 8),
            "scale": _round(bundle["scaler_scale"], 8),
        },
        "continuous_features": continuous,
        "categorical_features": list(categories.keys()),
        "categories": categories,
        "silent_defaults": silent_defaults,
        "feature_names": feature_names,
        "gender_axis": {"w": _round(w_axis, 8), "step": round(float(step), 8)},
        "models": models,
    }
    (UI_DIR / "model.json").write_text(json.dumps(model_json), encoding="utf-8")
    size_kb = (UI_DIR / "model.json").stat().st_size / 1024
    print(f"    wrote ui/model.json ({size_kb:.0f} KB)")

    print("\n[3] Parity check: JS-equivalent numpy forward vs PyTorch")
    for name, ckpt in (("baseline", "baseline.pt"), ("sanitized", "sanitized.pt")):
        enc, pred = load_encoder_predictor(MODELS_DIR / ckpt)
        torch_s = score_test(enc, pred, X)
        np_s = _np_score(models[name], Xn)
        max_diff = float(np.max(np.abs(torch_s - np_s)))
        flag = "OK" if max_diff < 1e-4 else "FAIL"
        print(f"    {name:10s} max|p_torch - p_js| = {max_diff:.2e}  [{flag}]")
        assert max_diff < 1e-4, f"{name} forward parity failed"

    print("\n[4] t-SNE of the latent Before / After (signature animation)")
    enc_b, pred_b = load_encoder_predictor(MODELS_DIR / "baseline.pt")
    enc_a, pred_a = load_encoder_predictor(MODELS_DIR / "sanitized.pt")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xn))[:TSNE_N]
    z_before = encode_test(enc_b, X)[idx]
    z_after = encode_test(enc_a, X)[idx]
    emb_b = _norm_xy(_tsne(z_before))
    emb_a = _norm_xy(_tsne(z_after))
    g = gender[idx].astype(int).tolist()

    print("[5] Headline plot numbers (accuracy / leakage / fairness)")
    acc_b = float(((score_test(enc_b, pred_b, X) >= 0.5).astype(int) == y_true).mean())
    acc_a = float(((score_test(enc_a, pred_a, X) >= 0.5).astype(int) == y_true).mean())
    leak_b = _read_leakage(LOGS_DIR / "baseline.csv")
    leak_a = _read_leakage(LOGS_DIR / "adversarial.csv")
    fair_b = fairness_metrics(score_test(enc_b, pred_b, X), y_true, gender)
    fair_a = fairness_metrics(score_test(enc_a, pred_a, X), y_true, gender)

    print("[6] Counterfactual swing distributions (honest-caveat histogram)")
    sw_b = _axis_swing(enc_b, pred_b, Xn, w_axis, step)
    sw_a = _axis_swing(enc_a, pred_a, Xn, w_axis, step)
    sidx = rng.permutation(len(sw_b))[:SWING_N]

    data_json = {
        "tsne": {
            "gender": g,
            "before": emb_b,
            "after": emb_a,
            "gender_classes": [str(s) for s in bundle["gender_classes"]],
        },
        "accuracy": {"before": acc_b, "after": acc_a},
        "leakage": {
            "attrs": ["gender", "age"],
            "before": [leak_b["gender"]["lift"], leak_b["age"]["lift"]],
            "after": [leak_a["gender"]["lift"], leak_a["age"]["lift"]],
        },
        "fairness": {
            "labels": ["Demographic parity gap", "Equal opportunity (TPR) gap",
                       "FPR gap"],
            "before": [fair_b["dp_gap"], fair_b["tpr_gap"], fair_b["fpr_gap"]],
            "after": [fair_a["dp_gap"], fair_a["tpr_gap"], fair_a["fpr_gap"]],
        },
        "counterfactual": {
            "before": _round(sw_b[sidx], 5),
            "after": _round(sw_a[sidx], 5),
            "mean_abs_before": float(np.mean(np.abs(sw_b))),
            "mean_abs_after": float(np.mean(np.abs(sw_a))),
        },
    }
    (UI_DIR / "data.json").write_text(json.dumps(data_json), encoding="utf-8")
    size_kb = (UI_DIR / "data.json").stat().st_size / 1024
    print(f"    wrote ui/data.json ({size_kb:.0f} KB)")

    print("\nExport complete. The UI can now score candidates fully in-browser.")


def _norm_xy(emb: np.ndarray) -> list[list[float]]:
    """Center + scale a 2-D embedding into roughly [-1, 1] for stable plotting
    across the Before/After morph."""
    emb = emb - emb.mean(axis=0)
    span = np.abs(emb).max() + 1e-9
    emb = emb / span
    return np.round(emb, 4).tolist()


if __name__ == "__main__":
    main()
