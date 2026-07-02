# FairHire AI

**An AI hiring model that scrubs gender and age from how it "thinks" — shrinking demographic bias at almost no accuracy cost.**

[**Live demo →**](#) &nbsp;·&nbsp; [**Repo →**](https://github.com/BlackTensor/fairhire-ai) &nbsp;·&nbsp; built by Shayan Ansari

---

## What this does and why

Hiring models can quietly encode a candidate's gender or age even when you never feed those
attributes in — the model picks them up from correlated signals and can carry that bias into
its decisions. FairHire AI trains the model against built-in "auditors" whose only job is to
guess gender and age from the model's internal representation. By making that representation
useless to the auditors, we cut how much those attributes can be recovered and we shrink the
hiring gaps between groups — while keeping accuracy essentially unchanged. In short: the model
still predicts well, but it becomes much harder to read a candidate's gender or age out of it.

## Results

| Measure | Before | After |
|---|---|---|
| Hiring accuracy | 85.8% | 85.7% |
| Gender leakage (points above chance) | +15.4 | +2.0 |
| Equal-opportunity (TPR) gap | 10.4 | 2.6 |

See `plots/` for the full visual results (latent-space maps, leakage, accuracy, and fairness).

## The honest caveat

Scrubbing the model's internal representation is not the same as making its score immune to
a candidate's gender. When we nudge a candidate along gender-predictive directions in the
input and watch the score move, the sanitized model actually moves *more*, not less. So this
work makes gender hard to *read* from the representation and narrows group-level gaps, but it
does not guarantee an individual's score would be unchanged if only their gender were
different. We report this openly rather than hide it.

## Future work

Train for score-level invariance directly with a counterfactual-consistency penalty, moving
toward full counterfactual fairness (Kusner et al., 2017).

---

Technical depth — architecture, the adversarial training loop, how leakage is measured, data,
and reproduce steps — lives in [`docs/METHODS.md`](docs/METHODS.md).
