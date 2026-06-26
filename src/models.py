"""FairHire AI — model components.

The pieces here are deliberately small and composable so they carry across
phases:

  * ``Encoder``      input features -> sanitized latent ``z``.
  * ``MLPHead``      ``z`` -> class logits. Used three ways:
                       - the hiring predictor (main task, 2 classes),
                       - the standalone leakage probes (Phase 2 "Before"),
                       - the adversarial auditors (Phase 3, behind a GRL).

Keeping a single head class means the Phase 2 probe and the Phase 3 auditor have
identical capacity, so any drop in recoverability between them is attributable
to the sanitization, not to a weaker attacker.
"""

from __future__ import annotations

import torch
from torch import nn

# Latent width. Small enough that t-SNE (Phase 4) is meaningful, wide enough to
# carry the main-task signal.
Z_DIM = 32

# Number of classes per sensitive attribute (matches src/data.py vocabularies).
SENSITIVE_N_CLASSES = {"gender": 2, "age": 3, "ethnicity": 5}
MAIN_N_CLASSES = 2


class Encoder(nn.Module):
    """Maps the raw feature vector to a latent ``z``.

    Two hidden blocks with BatchNorm + ReLU + Dropout, then a linear projection
    to ``z_dim``. No activation on ``z`` itself -- downstream heads apply their
    own nonlinearities.
    """

    def __init__(
        self,
        in_dim: int,
        z_dim: int = Z_DIM,
        hidden: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, z_dim),
        )
        self.in_dim = in_dim
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """Generic one-hidden-layer classification head: ``z`` -> logits."""

    def __init__(self, z_dim: int, n_classes: int, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.z_dim = z_dim
        self.n_classes = n_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BaselineModel(nn.Module):
    """Encoder + hiring predictor with no adversaries (Phase 2 'Before')."""

    def __init__(self, in_dim: int, z_dim: int = Z_DIM):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim=z_dim)
        self.predictor = MLPHead(z_dim, MAIN_N_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.predictor(z)
        return logits, z


# --------------------------------------------------------------------------- #
# Phase 3 — Gradient Reversal Layer + the adversarial (sanitized) model
# --------------------------------------------------------------------------- #
class _GradReverse(torch.autograd.Function):
    """Identity on the forward pass; negates and scales the gradient on the way
    back. This is the mechanism that turns an ordinary classifier into an
    adversary: the auditor head still minimizes its own loss, but the gradient
    that reaches the *encoder* is flipped, so the encoder is pushed to make the
    latent ``z`` as useless as possible for that auditor.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale; no gradient flows to the lambda scalar.
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wraps :class:`_GradReverse`. ``lambda_`` is a plain mutable attribute so
    the training loop can ramp it over epochs (DANN-style warmup)."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverse.apply(x, self.lambda_)


class SanitizedModel(nn.Module):
    """Encoder + hiring predictor + one auditor per adversarial attribute, all
    sharing a single GRL on the latent ``z`` (Phase 3 'After').

    ``forward`` returns ``(main_logits, z, auditor_logits)`` where
    ``auditor_logits`` is a dict keyed by attribute. The auditor logits are
    computed from ``grl(z)``, so a single ``backward`` on
    ``main_loss + sum(adv_losses)`` trains the auditors to recover the
    attributes while pushing the encoder to defeat them.
    """

    def __init__(
        self,
        in_dim: int,
        adversaries: tuple[str, ...] = ("gender", "age"),
        z_dim: int = Z_DIM,
        lambda_: float = 1.0,
    ):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim=z_dim)
        self.predictor = MLPHead(z_dim, MAIN_N_CLASSES)
        self.grl = GradientReversalLayer(lambda_)
        self.auditors = nn.ModuleDict(
            {k: MLPHead(z_dim, SENSITIVE_N_CLASSES[k]) for k in adversaries}
        )
        self.adversaries = tuple(adversaries)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encoder(x)
        main_logits = self.predictor(z)
        z_rev = self.grl(z)
        aud_logits = {k: head(z_rev) for k, head in self.auditors.items()}
        return main_logits, z, aud_logits
