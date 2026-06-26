"""FairHire AI — Phase 4 evaluation and the headline plots.

Reads the trained "Before" (``models/baseline.pt``) and "After"
(``models/sanitized.pt``) encoders, runs them over the held-out test split, and
produces the visual thesis as clean PNGs in ``plots/``:

  * ``plot1_tsne_before.png``  — t-SNE of the unsanitized latent, coloured by
    gender. Gender skews across regions (it is decodable at +15pts over chance).
  * ``plot2_tsne_after.png``   — t-SNE of the sanitized latent. The genders are
    mixed throughout (gender decodability drops to ~chance).
  * ``plot3_accuracy.png``     — hiring accuracy Before vs After (small drop).
  * ``plot4_leakage.png``      — sensitive-attribute leakage Before vs After
    (large drop, coral bars).
  * ``plot5_counterfactual.png`` — the counterfactual sensitivity audit. Gender
    is not a model input (Phase 1), so we nudge each candidate along the
    gender-predictive direction in feature space and plot the resulting score
    swing. Honest finding: the sanitized model swings MORE, not less. Three
    independent instruments (this axis nudge, a Husband<->Wife token flip, and
    nearest opposite-gender matched pairs; logged to ``logs/counterfactual.csv``)
    all agree. Reported plainly rather than hidden.

It also computes group-fairness metrics (demographic parity gap, equalized-odds
gaps) Before vs After and writes them to ``logs/fairness.csv`` with a companion
``plot6_fairness.png``.

Key caveat (see CLAUDE.md decisions log): adversarial *latent* sanitization
removes gender's decodability from z (Plot 4) and shrinks group-fairness gaps
(Plot 6), but it optimizes for *representational* invariance on the data
manifold -- not *counterfactual* invariance to input perturbations. The two are
different, and Plot 5 shows the gap.

Run:  python -m src.evaluate
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from .data import make_dataloaders
from .models import MAIN_N_CLASSES, Encoder, MLPHead
from .train_baseline import DEVICE, _seed_everything

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"

# --------------------------------------------------------------------------- #
# Locked design tokens (CLAUDE.md)
# --------------------------------------------------------------------------- #
BG = "#13151C"        # background (lightened from #0D0F14)
SURFACE = "#1C1F28"   # surface / axes face (lightened from #16191F)
TEXT = "#E8EAF0"      # primary text
GRID = "#2A2F3A"      # subtle gridlines (derived, between bg and text)
MUTED = "#8A90A0"     # secondary labels
ACCENT_A = "#A855F7"  # violet  — demographic cluster 1 (Female)
ACCENT_B = "#F59E0B"  # amber   — demographic cluster 2 (Male)
CORAL = "#E8614A"     # danger / leakage / bias — reserved

# Counterfactual subsample for t-SNE (speed + legibility).
TSNE_N = 2500
SEED = 42


def _setup_style() -> None:
    """Dark premium matplotlib defaults. Fonts fall back gracefully if the
    locked typefaces are not installed on the machine."""
    warnings.filterwarnings("ignore", message="findfont")
    plt.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Space Grotesk", "DejaVu Sans"],
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------- #
# Model loading + latent / prediction extraction
# --------------------------------------------------------------------------- #
def load_encoder_predictor(path: Path) -> tuple[Encoder, MLPHead]:
    """Reconstruct a frozen Encoder + hiring predictor from a checkpoint."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    enc = Encoder(ckpt["in_dim"], z_dim=ckpt["z_dim"]).to(DEVICE)
    enc.load_state_dict(ckpt["encoder_state"])
    pred = MLPHead(ckpt["z_dim"], MAIN_N_CLASSES).to(DEVICE)
    pred.load_state_dict(ckpt["predictor_state"])
    enc.eval()
    pred.eval()
    return enc, pred


@torch.no_grad()
def encode_test(enc: Encoder, X: torch.Tensor) -> np.ndarray:
    enc.eval()
    return enc(X.to(DEVICE)).cpu().numpy()


@torch.no_grad()
def score_test(enc: Encoder, pred: MLPHead, X: torch.Tensor) -> np.ndarray:
    """Return P(income > 50K) for each row."""
    enc.eval()
    pred.eval()
    logits = pred(enc(X.to(DEVICE)))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


# --------------------------------------------------------------------------- #
# Plots 1 & 2 — t-SNE of the latent, coloured by gender
# --------------------------------------------------------------------------- #
def _tsne(z: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=2, perplexity=30, init="pca",
        random_state=SEED, max_iter=1000,
    ).fit_transform(z)


def plot_tsne(z: np.ndarray, gender: np.ndarray, gender_classes, title: str,
              subtitle: str, out: Path) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(z))[:TSNE_N]
    emb = _tsne(z[idx])
    g = gender[idx]

    fig, ax = plt.subplots(figsize=(7, 6.2))
    colors = {0: ACCENT_A, 1: ACCENT_B}  # Female=violet, Male=amber
    for code, color in colors.items():
        m = g == code
        ax.scatter(emb[m, 0], emb[m, 1], s=7, c=color, alpha=0.55,
                   linewidths=0, label=str(gender_classes[code]))
    ax.set_title(title, fontsize=16, fontweight="bold", pad=30, loc="left")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10,
                    title="Gender")
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_title().set_color(TEXT)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


# --------------------------------------------------------------------------- #
# Plot 3 — hiring accuracy Before vs After
# --------------------------------------------------------------------------- #
def plot_accuracy(before: float, after: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    labels = ["Before\n(no privacy)", "After\n(sanitized)"]
    vals = [before * 100, after * 100]
    bars = ax.bar(labels, vals, width=0.55,
                  color=[MUTED, ACCENT_A], edgecolor="none")
    drop = (before - after) * 100
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hiring accuracy (test, %)")
    ax.set_title("The useful signal survives", fontsize=16,
                 fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012,
            f"Sanitization costs only {drop:.1f} pts of accuracy",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=TEXT, family="monospace")
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


# --------------------------------------------------------------------------- #
# Plot 4 — sensitive-attribute leakage Before vs After
# --------------------------------------------------------------------------- #
def plot_leakage(before: dict, after: dict, attrs, out: Path) -> None:
    """Leakage = probe accuracy lift over the majority baseline, in points.
    Coral is reserved for this bias/leakage indicator."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(attrs))
    w = 0.38
    b_vals = [before[a]["lift"] * 100 for a in attrs]
    a_vals = [after[a]["lift"] * 100 for a in attrs]

    ax.bar(x - w / 2, b_vals, w, label="Before", color=CORAL, edgecolor="none")
    ax.bar(x + w / 2, a_vals, w, label="After", color=CORAL, alpha=0.32,
           edgecolor="none")

    for xi, v in zip(x - w / 2, b_vals):
        ax.text(xi, v + 0.4, f"+{v:.1f}", ha="center", va="bottom",
                fontsize=11, color=TEXT, family="monospace")
    for xi, v in zip(x + w / 2, a_vals):
        ax.text(xi, v + 0.4, f"+{v:.1f}", ha="center", va="bottom",
                fontsize=11, color=MUTED, family="monospace")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in attrs])
    ax.set_ylabel("Recoverable signal above chance (pts)")
    ax.set_ylim(0, max(b_vals) * 1.25)
    ax.set_title("Leakage falls off a cliff", fontsize=16,
                 fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012,
            "How much a probe can read each attribute out of the latent",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


# --------------------------------------------------------------------------- #
# Plot 5 — the counterfactual audit
#
# Honest finding (see CLAUDE.md decisions log). Adversarial *latent*
# sanitization removes gender's decodability from z (Plot 4) and shrinks the
# group-fairness gaps (Plot 6) -- but it never trains for *counterfactual*
# invariance to input perturbations. Across three independent gender-flip
# instruments the sanitized model's hiring score is in fact MORE sensitive to a
# gender perturbation than the baseline. We report this plainly: representational
# invariance on the data manifold is not the same as counterfactual invariance.
# Gender is not a model input (Phase 1), so every instrument perturbs proxies.
# --------------------------------------------------------------------------- #
def _gender_axis(X: np.ndarray, gender: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit gender-predictive direction in feature space (logistic regression)
    plus the female->male distance along it -- a calibrated, on-axis nudge."""
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, gender)
    w = lr.coef_[0]
    w = w / (np.linalg.norm(w) + 1e-12)
    proj = X @ w
    step = float(proj[gender == 1].mean() - proj[gender == 0].mean())
    return w, step


def _axis_swing(enc, pred, X: np.ndarray, w: np.ndarray, step: float) -> np.ndarray:
    """Signed score change when a candidate is nudged toward 'male' vs 'female'
    along the gender axis (+ means the more-male version scores higher)."""
    sM = score_test(enc, pred, torch.as_tensor(X + w * step, dtype=torch.float32))
    sF = score_test(enc, pred, torch.as_tensor(X - w * step, dtype=torch.float32))
    return sM - sF


def _token_flip_swing(enc, pred, X: np.ndarray, feature_names) -> np.ndarray:
    """Crude but interpretable: swap the gender-locked relationship proxy
    (Husband <-> Wife) on the candidates that have one."""
    h = feature_names.index("relationship=Husband")
    w = feature_names.index("relationship=Wife")
    Xf = X.copy()
    Xf[:, [h, w]] = Xf[:, [w, h]]
    mask = (X[:, h] == 1.0) | (X[:, w] == 1.0)
    s0 = score_test(enc, pred, torch.as_tensor(X, dtype=torch.float32))
    s1 = score_test(enc, pred, torch.as_tensor(Xf, dtype=torch.float32))
    return np.abs(s1 - s0)[mask]


def _matched_pair_swing(enc, pred, X: np.ndarray, gender: np.ndarray) -> np.ndarray:
    """On-manifold: each candidate vs their nearest real opposite-gender
    neighbour in feature space; |score difference|."""
    fem, mal = np.where(gender == 0)[0], np.where(gender == 1)[0]
    s = score_test(enc, pred, torch.as_tensor(X, dtype=torch.float32))
    im = mal[NearestNeighbors(n_neighbors=1).fit(X[mal]).kneighbors(X[fem])[1][:, 0]]
    iff = fem[NearestNeighbors(n_neighbors=1).fit(X[fem]).kneighbors(X[mal])[1][:, 0]]
    return np.concatenate([np.abs(s[fem] - s[im]), np.abs(s[mal] - s[iff])])


def plot_counterfactual(swing_before: np.ndarray, swing_after: np.ndarray,
                        mab: float, maa: float, out: Path) -> None:
    """Distribution of the gender-axis score swing, Before vs After. The wider
    (sanitized) distribution is the bias indicator, so it gets coral."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lim = float(np.percentile(np.abs(np.concatenate(
        [swing_before, swing_after])), 99))
    lim = max(lim, 0.02)
    bins = np.linspace(-lim, lim, 61)

    ax.hist(swing_before, bins=bins, color=ACCENT_A, alpha=0.75,
            label="Before (no privacy)", edgecolor="none")
    ax.hist(swing_after, bins=bins, color=CORAL, alpha=0.7,
            label="After (sanitized)", edgecolor="none")
    ax.axvline(0, color=MUTED, lw=1, ls=":")

    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Score change when a candidate is nudged along the gender axis")
    ax.set_ylabel("Candidates")
    ax.set_title("An honest caveat: scrubbing z is not counterfactual invariance",
                 fontsize=14, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012,
            f"Mean absolute swing rises {mab * 100:.1f} -> {maa * 100:.1f} pts "
            f"after sanitization",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def save_counterfactual_csv(rows: list[tuple[str, float, float]]) -> None:
    path = LOGS_DIR / "counterfactual.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instrument", "mean_abs_swing_before", "mean_abs_swing_after"])
        for name, b, a in rows:
            w.writerow([name, f"{b:.6f}", f"{a:.6f}"])
    print(f"  saved {path.name}")


# --------------------------------------------------------------------------- #
# Fairness metrics (group fairness on the actual gender labels)
# --------------------------------------------------------------------------- #
def fairness_metrics(scores: np.ndarray, y_true: np.ndarray,
                     gender: np.ndarray, thresh: float = 0.5) -> dict:
    """Demographic parity gap and equalized-odds (TPR/FPR) gaps across the two
    gender groups, using a 0.5 decision threshold."""
    yhat = (scores >= thresh).astype(int)
    g0, g1 = gender == 0, gender == 1  # Female, Male

    def rate(mask):
        return float(yhat[mask].mean()) if mask.any() else float("nan")

    def cond_rate(mask, label):
        sel = mask & (y_true == label)
        return float(yhat[sel].mean()) if sel.any() else float("nan")

    sr0, sr1 = rate(g0), rate(g1)
    tpr0, tpr1 = cond_rate(g0, 1), cond_rate(g1, 1)
    fpr0, fpr1 = cond_rate(g0, 0), cond_rate(g1, 0)
    return {
        "dp_gap": abs(sr0 - sr1),
        "tpr_gap": abs(tpr0 - tpr1),
        "fpr_gap": abs(fpr0 - fpr1),
        "sel_rate_female": sr0,
        "sel_rate_male": sr1,
    }


def plot_fairness(before: dict, after: dict, out: Path) -> None:
    metrics = [("dp_gap", "Demographic\nparity gap"),
               ("tpr_gap", "Equal opportunity\n(TPR gap)"),
               ("fpr_gap", "FPR gap")]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(metrics))
    w = 0.38
    b_vals = [before[k] * 100 for k, _ in metrics]
    a_vals = [after[k] * 100 for k, _ in metrics]
    ax.bar(x - w / 2, b_vals, w, label="Before", color=CORAL, edgecolor="none")
    ax.bar(x + w / 2, a_vals, w, label="After", color=CORAL, alpha=0.32,
           edgecolor="none")
    for xi, v in zip(x - w / 2, b_vals):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=11, color=TEXT, family="monospace")
    for xi, v in zip(x + w / 2, a_vals):
        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=11, color=MUTED, family="monospace")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Gap between gender groups (pts)")
    ax.set_ylim(0, max(b_vals + a_vals) * 1.3)
    ax.set_title("Hiring decisions even out across gender", fontsize=16,
                 fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012, "Smaller gaps = fairer outcomes (0.5 threshold)",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def save_fairness_csv(before: dict, after: dict) -> None:
    path = LOGS_DIR / "fairness.csv"
    keys = ["dp_gap", "tpr_gap", "fpr_gap", "sel_rate_female", "sel_rate_male"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after"])
        for k in keys:
            w.writerow([k, f"{before[k]:.6f}", f"{after[k]:.6f}"])
    print(f"  saved {path.name}")


# --------------------------------------------------------------------------- #
# Leakage helper — read the recorded Before/After numbers from the CSV logs
# --------------------------------------------------------------------------- #
def _read_leakage(csv_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            metric, attr, val = row["metric"], row["attribute"], row["value"]
            if metric.startswith("leakage_") and attr:
                out.setdefault(attr, {})[metric.replace("leakage_", "")] = \
                    float(val)
    return {a: {"lift": d.get("lift", d.get("probe_acc", 0.0) -
                                 d.get("majority", 0.0))} for a, d in out.items()}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    _seed_everything(SEED)
    _setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 4 — EVALUATION & PLOTS")
    print(f"device={DEVICE}")
    print("=" * 70)

    loaders, bundle = make_dataloaders(batch_size=256)
    ds = loaders["test"].dataset
    X = ds.X                                  # (N, in_dim) float32 tensor
    y_true = ds.main.numpy()
    gender = ds.sensitive["gender"].numpy()
    feature_names = [str(s) for s in bundle["feature_names"]]
    gender_classes = list(bundle["gender_classes"])

    print("\n[1] Loading Before / After models")
    enc_b, pred_b = load_encoder_predictor(MODELS_DIR / "baseline.pt")
    enc_a, pred_a = load_encoder_predictor(MODELS_DIR / "sanitized.pt")

    print("\n[2] t-SNE of the latent (Plots 1 & 2)")
    z_before = encode_test(enc_b, X)
    z_after = encode_test(enc_a, X)
    plot_tsne(z_before, gender, gender_classes,
              "Before: gender leaks into the latent",
              "t-SNE of z (no privacy), coloured by gender — note the "
              "amber/violet skew across regions",
              PLOTS_DIR / "plot1_tsne_before.png")
    plot_tsne(z_after, gender, gender_classes,
              "After: gender is scrubbed from the latent",
              "t-SNE of z (sanitized) — the two genders are now mixed "
              "throughout",
              PLOTS_DIR / "plot2_tsne_after.png")

    print("\n[3] Hiring accuracy (Plot 3)")
    acc_b = float(((score_test(enc_b, pred_b, X) >= 0.5).astype(int)
                   == y_true).mean())
    acc_a = float(((score_test(enc_a, pred_a, X) >= 0.5).astype(int)
                   == y_true).mean())
    print(f"    before={acc_b:.4f}  after={acc_a:.4f}")
    plot_accuracy(acc_b, acc_a, PLOTS_DIR / "plot3_accuracy.png")

    print("\n[4] Leakage Before vs After (Plot 4)")
    leak_before = _read_leakage(LOGS_DIR / "baseline.csv")
    leak_after = _read_leakage(LOGS_DIR / "adversarial.csv")
    plot_leakage(leak_before, leak_after, ["gender", "age"],
                 PLOTS_DIR / "plot4_leakage.png")

    print("\n[5] Counterfactual audit — gender-flip sensitivity (Plot 5)")
    Xn = X.numpy()
    w_axis, step = _gender_axis(Xn, gender)
    sw_b = _axis_swing(enc_b, pred_b, Xn, w_axis, step)
    sw_a = _axis_swing(enc_a, pred_a, Xn, w_axis, step)
    mab, maa = float(np.mean(np.abs(sw_b))), float(np.mean(np.abs(sw_a)))
    plot_counterfactual(sw_b, sw_a, mab, maa,
                        PLOTS_DIR / "plot5_counterfactual.png")
    # Corroborating instruments (logged, not plotted): all three agree the
    # sanitized model is MORE gender-sensitive, not less.
    tok_b = _token_flip_swing(enc_b, pred_b, Xn, feature_names)
    tok_a = _token_flip_swing(enc_a, pred_a, Xn, feature_names)
    mp_b = _matched_pair_swing(enc_b, pred_b, Xn, gender)
    mp_a = _matched_pair_swing(enc_a, pred_a, Xn, gender)
    rows = [
        ("gender_axis_perturbation", mab, maa),
        ("husband_wife_token_flip", float(np.mean(tok_b)), float(np.mean(tok_a))),
        ("matched_opposite_gender_pair", float(np.mean(mp_b)), float(np.mean(mp_a))),
    ]
    for nm, b, a in rows:
        print(f"    {nm:30s} before={b * 100:5.2f}pts  after={a * 100:5.2f}pts")
    save_counterfactual_csv(rows)

    print("\n[6] Group-fairness metrics Before vs After")
    fair_b = fairness_metrics(score_test(enc_b, pred_b, X), y_true, gender)
    fair_a = fairness_metrics(score_test(enc_a, pred_a, X), y_true, gender)
    for name, fm in (("before", fair_b), ("after", fair_a)):
        print(f"    {name:6s}: DP gap={fm['dp_gap']:.4f}  "
              f"TPR gap={fm['tpr_gap']:.4f}  FPR gap={fm['fpr_gap']:.4f}")
    save_fairness_csv(fair_b, fair_a)
    plot_fairness(fair_b, fair_a, PLOTS_DIR / "plot6_fairness.png")

    print("\nPhase 4 complete. Six PNGs in plots/, fairness in logs/.")


if __name__ == "__main__":
    main()
