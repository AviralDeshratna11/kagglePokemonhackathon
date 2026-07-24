"""The policy interface every brain in this project implements."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..core.actions import ActionCandidate
from ..core.clock import Deadline
from ..core.enums import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS, SelectType
from ..core.obs import ObsView

__all__ = ["Policy", "PolicyResult", "default_desired_count"]


class PolicyResult(list):
    """Scores plus optional diagnostics, kept list-compatible for simplicity."""

    def __init__(self, scores: Sequence[float], info: dict | None = None) -> None:
        super().__init__(scores)
        self.info = info or {}


class Policy(Protocol):
    """A brain.

    ``score`` returns one real number per candidate; higher is better. It must
    never raise, never mutate the view, and must respect ``deadline``.
    """

    name: str

    def deck(self) -> list[int]:
        """The 60 card IDs to register at the deck-selection phase."""
        ...

    def score(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        deadline: Deadline,
    ) -> Sequence[float]:
        ...

    def desired_count(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        scores: Sequence[float],
    ) -> int:
        ...


def default_desired_count(
    view: ObsView,
    candidates: Sequence[ActionCandidate],
    scores: Sequence[float],
) -> int:
    """How many options to select when ``maxCount > minCount``.

    The engine's multi-select is genuinely ambiguous and a wrong count is a
    silent strategic leak rather than a crash, so the rule is explicit:

    * energy payment and damage-counter placement are *quantity* selections
      driven by ``remainEnergyCost`` / ``remainDamageCounter``;
    * contexts that cost us resources (discard, shuffle away, bottom-deck) take
      the legal minimum;
    * contexts that gain us resources (search, draw, heal, put into play) take
      the legal maximum;
    * otherwise take the minimum, but never fewer than one positively-scored
      option when the minimum is zero -- declining a free effect is usually
      wrong, and this keeps the agent from passing on its own abilities.
    """
    lo, hi = view.min_count, view.max_count
    if hi <= lo:
        return max(0, lo)

    ctx = view.context

    if view.select_type == SelectType.ENERGY and view.remain_energy_cost > 0:
        need = view.remain_energy_cost
        taken = 0
        total = 0
        for i in sorted(range(len(candidates)), key=lambda k: -scores[k]):
            if total >= need or taken >= hi:
                break
            total += int(candidates[i].count or 1)
            taken += 1
        return max(lo, min(hi, taken))

    if view.remain_damage_counter > 0:
        return max(lo, min(hi, view.remain_damage_counter))

    if ctx in NEGATIVE_CONTEXTS:
        return lo

    if ctx in POSITIVE_CONTEXTS:
        return hi

    if lo == 0:
        positive = sum(1 for s in scores if s > 0.0)
        return min(hi, max(0, positive))
    return lo
