"""Sparring partners.

Week 2 builds a scripted league; Week 0 needs the two ends of it now, because
"the heuristic is better than nothing" is a claim that has to be measured, not
assumed.

* :class:`RandomPolicy` and :class:`FirstPolicy` mirror the two example agents
  shipped in ``kaggle_environments/envs/cabt/cabt.py``. Community timing
  analysis suggests a large share of the live ladder is running exactly these
  or light variations, so they are the honest floor to beat.
* :class:`GreedyAttackPolicy` is the obvious naive strategy -- always take the
  biggest attack available -- and is the more interesting baseline, because
  beating it is what shows the setup, benching and energy-routing logic is
  earning its keep rather than just the damage comparison.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..core.actions import ActionCandidate
from ..core.clock import Deadline
from ..core.enums import OptionType, SelectType
from ..core.obs import ObsView
from .base import default_desired_count

__all__ = ["RandomPolicy", "FirstPolicy", "GreedyAttackPolicy"]


class _Base:
    name = "base"

    def __init__(self, deck_ids: Sequence[int]) -> None:
        self._deck = list(deck_ids)

    def deck(self) -> list[int]:
        return list(self._deck)

    def desired_count(self, view, candidates, scores) -> int:
        return default_desired_count(view, candidates, scores)


class RandomPolicy(_Base):
    """Uniform over legal options."""

    name = "random"

    def __init__(self, deck_ids: Sequence[int], seed: int | None = None) -> None:
        super().__init__(deck_ids)
        self._rng = random.Random(seed)

    def score(self, view: ObsView, candidates: Sequence[ActionCandidate], deadline: Deadline):
        return [self._rng.random() for _ in candidates]

    def desired_count(self, view, candidates, scores) -> int:
        lo, hi = view.min_count, view.max_count
        return self._rng.randint(lo, hi) if hi > lo else lo


class FirstPolicy(_Base):
    """Always take the earliest options, like the shipped ``first_agent``."""

    name = "first"

    def score(self, view: ObsView, candidates: Sequence[ActionCandidate], deadline: Deadline):
        return [float(len(candidates) - c.index) for c in candidates]

    def desired_count(self, view, candidates, scores) -> int:
        return view.max_count


class GreedyAttackPolicy(_Base):
    """Attack for the most damage available; otherwise develop arbitrarily.

    No setup planning, no energy routing, no retreat logic -- the ablation
    baseline for "does the rest of the heuristic actually matter".
    """

    name = "greedy-attack"

    def score(self, view: ObsView, candidates: Sequence[ActionCandidate], deadline: Deadline):
        out = []
        for c in candidates:
            if view.select_type == SelectType.MAIN:
                if c.option_type == OptionType.ATTACK:
                    s = 500.0 + c.est_damage + (5000.0 if c.is_knockout else 0.0)
                elif c.option_type == OptionType.END:
                    s = 0.0
                else:
                    s = 100.0
            else:
                s = float(len(candidates) - c.index)
            out.append(s)
        return out
