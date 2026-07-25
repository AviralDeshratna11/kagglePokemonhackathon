"""Stochastic self-play driver.

``BCPolicy`` (Week 1) is deterministic -- it always ranks the top-scoring
option first, exactly like ``HeuristicPolicy``. That is correct for a
submission, but self-play training needs exploration: without variance in
which actions get taken, there is nothing for a policy-gradient update to
learn from.

Rather than touching ``SafetyShell``'s selection code (it sorts candidates by
score and takes the top-k -- proven, safety-critical, not something to
special-case), this perturbs the *scores* with Gumbel noise before handing
them to that same deterministic sort. Gumbel-max is an exact sampler:
``argmax(logits/T + Gumbel noise)`` is a draw from ``softmax(logits/T)``, and
taking the top-k of Gumbel-perturbed logits (Gumbel-Top-k) is sampling
without replacement -- exactly what a multi-select decision needs. So
``SafetyShell``'s existing "sort by score, take top-k" *is* the sampler; no
new selection logic, same zero-illegal-action guarantee as every other
policy.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..core.actions import ActionCandidate
from ..core.clock import Deadline
from ..core.obs import ObsView
from .base import default_desired_count
from .network import BeliefSetDMCLite, CardTable, encode_view

__all__ = ["NetworkPolicy"]


class NetworkPolicy:
    """Same shape as :class:`~ptcg.agents.bc_policy.BCPolicy`; ``score()``
    returns Gumbel-perturbed logits instead of raw ones. ``temperature<=0``
    disables perturbation entirely (pure argmax, for evaluation/deployment of
    a self-play checkpoint without exploration noise)."""

    name = "network-selfplay"

    def __init__(
        self,
        deck_ids: Sequence[int],
        model: BeliefSetDMCLite,
        table: CardTable,
        card_mode: str,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._deck = list(deck_ids)
        self.model = model
        self.model.eval()
        self.table = table
        self.card_mode = card_mode
        self.temperature = temperature
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def deck(self) -> list[int]:
        return list(self._deck)

    def score(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        deadline: Deadline,
    ) -> list[float]:
        if not candidates:
            return []
        inp = encode_view(view, candidates, self.table, self.card_mode)
        with torch.no_grad():
            out = self.model(inp)
        logits = out["policy_logits"][0][: len(candidates)]

        if self.temperature <= 0:
            return [float(x) for x in logits.tolist()]

        u = torch.rand(logits.shape, generator=self._gen).clamp_(1e-9, 1.0 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        perturbed = logits / self.temperature + gumbel
        return [float(x) for x in perturbed.tolist()]

    def desired_count(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        scores: Sequence[float],
    ) -> int:
        return default_desired_count(view, candidates, scores)
