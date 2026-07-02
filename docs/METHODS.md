# Methods — FairHire AI

Technical depth for the [FairHire AI](../README.md) project. See `plots/` for the full visual results.

## Architecture

```
        Input features
              |
          [ Encoder ]
              |
              z  (sanitized latent representation)
              |
   +----------+-----------+-----------+------------+
   |          |           |           |            |
[Hiring]  [Gender]    [Age]      [Ethnicity]   (auditors attack z)
[Predictor] [Auditor]  [Auditor]  [Auditor]
   |          |           |           |
 main loss  adversarial losses via Gradient Reversal Layer (GRL)
```

## Multi-adversary sanitization

A small MLP encoder maps each candidate's features to a latent `z`. A hiring predictor
learns income suitability from `z` (the main loss). Simultaneously, a gender auditor and
an age auditor try to recover those attributes from `z`. Each auditor is connected through
a **Gradient Reversal Layer (GRL)**: in the forward pass it is the identity, but in the
backward pass it multiplies the gradient by `−λ`, so the encoder is pushed to make `z`
*useless* to the auditors while keeping it useful for hiring. Running **two adversaries at
once** — "Multi-Adversarial Representation Sanitization" — is the contribution over a
textbook single-adversary GRL setup. Ethnicity is *measured* but not targeted: in UCI
Adult it is ~unrecoverable (the data is ~86% one group), so an ethnicity adversary would
optimize against noise.

## Keeping the adversary strong

A single jointly-optimized GRL barely scrubbed (gender only +15.4 → +12.4) because the
encoder learns to fool one weak auditor instance while a fresh probe still recovers the
attribute. Fix: per batch we run `N_CRITIC=5` auditor-only updates on the *detached*
latent (a separate optimizer) so the adversary stays near-optimal, then take one
encoder+predictor step through the GRL. `λ` is ramped from 0 to its max (DANN-style
warmup) for stability. Auditor capacity is held identical to the Phase-2 leakage probe so
the before/after comparison measures scrubbing, not a bigger attacker.

## Measuring leakage

"Leakage" is the accuracy of a freshly-trained probe reading a sensitive attribute off the
frozen `z`, reported as points *above* that attribute's majority-class baseline. Gender at
chance is ~66.6%; after sanitization the probe reaches 68.7% (+2.0). The before/after
probes are identical in architecture so the drop reflects the representation, not the
attacker.

## Counterfactual audit (a sensitivity audit, not causal inference)

Gender is excluded from the model input, so to test counterfactual sensitivity we perturb
*proxies* and measure the score drift across three independent instruments: (1) a nudge
along the gender-predictive logistic direction in feature space, (2) a Husband↔Wife
relationship-token flip, (3) nearest opposite-gender matched pairs. No DAGs, no
do-calculus — we flip the sensitive direction in the input and measure output drift. All
three agree that the sanitized model is *more* gender-sensitive: adversarial scrubbing
enforces *representational* invariance on the data manifold, which is not the same as
*counterfactual* invariance of the score to input perturbations.

## Data & training

UCI Adult Income (48,842 rows), stratified 70/15/15 split on the income label; scaler and
one-hot vocabularies fit on train only. Sensitive columns (`sex`, `race`, `age`) are kept
only as labels and dropped from the feature matrix, so the auditors must recover them from
proxy correlations rather than a direct copy. Chosen `λ = 1.0` (the knee of a
`{0.5, 1, 2, 4}` sweep: lowest mean gender+age leakage among the runs that held accuracy
within 2% of baseline). Experiment tracking is local CSV (`logs/`), no external accounts.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.data              # download + build splits  -> data/processed/adult.npz
python -m src.train_baseline    # "Before" model + leakage -> models/baseline.pt, logs/baseline.csv
python -m src.train_adversarial # "After"  model           -> models/sanitized.pt, logs/adversarial.csv
python -m src.evaluate          # the six plots + fairness/counterfactual CSVs -> plots/, logs/
python -m src.export_ui         # serialize both models for the browser -> ui/model.json, ui/data.json
```

Then open `ui/index.html` (or serve the `ui/` folder) for the live, client-side demo.

## Project layout

```
data/        UCI Adult Income dataset + processed splits
src/         encoder, predictor, auditors, GRL, training loops, evaluation, UI export
notebooks/   exploration and experiment notebooks
plots/       the exported PNGs (four headline + counterfactual + fairness)
models/      trained weights (baseline + sanitized)
ui/          premium dark-themed web interface (pure HTML/CSS/JS, runs the models in-browser)
logs/        CSV metrics (baseline, adversarial sweep, fairness, counterfactual)
```

## Stack

Python, PyTorch, scikit-learn, numpy, pandas. Matplotlib/seaborn for static plots; pure
HTML/CSS/JS + Chart.js for the web UI. Dataset: UCI Adult Income. Every tool free /
open-source; the demo runs entirely client-side with no server or paid hosting.
