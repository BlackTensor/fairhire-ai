# LinkedIn post — draft

> Attach the four plots in this order: `plot1_tsne_before.png`, `plot2_tsne_after.png`,
> `plot4_leakage.png`, `plot5_counterfactual.png`. (Swap plot5 for `plot3_accuracy.png`
> if you'd rather lead with the "accuracy survives" story than the honest caveat.)

---

Can an AI hiring model "forget" gender and age — and does forgetting actually make it fair?

I built FairHire AI to find out: a Privacy-Preserving Functional Anonymizer via Adversarial Representation.

An encoder turns a candidate into an internal representation. A hiring head learns from it — while two adversaries fight to recover gender and age from that same representation through a Gradient Reversal Layer. The encoder is pushed to make those traits unreadable while keeping the signal that predicts suitability.

The result, on held-out data:

- Gender leakage from a probe: +15.4 → +2.0 points above chance (basically gone)
- Age leakage: +17.6 → +9.0
- Hiring accuracy: 85.8% → 85.7% (it barely moves)
- Equal-opportunity gap between groups: 10.4 → 2.6 points

You can watch gender literally dissolve out of the latent space (plots 1 and 2).

But here's the part most demos quietly skip. Scrubbing a *representation* is not the same as a counterfactually invariant *score*. When I nudge a candidate along a gender proxy and measure how much the score moves, the sanitized model swings MORE, not less. Three independent tests agree.

So I'm reporting it honestly: adversarial training buys representational invariance and smaller group-fairness gaps — both real — but not counterfactual invariance. The honest limit is the most interesting finding, and closing it (a counterfactual-consistency objective) is the next step.

Built entirely with free, open-source tools; the demo runs the models live in your browser, no server.

Live demo + write-up: [link]

#MachineLearning #FairnessInAI #ResponsibleAI #DeepLearning #PyTorch
</content>
