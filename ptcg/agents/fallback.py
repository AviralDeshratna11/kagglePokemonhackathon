"""The floor of the policy stack.

Deliberately trivial: no board reasoning, no card database lookups beyond what
is already attached to the candidate, no allocations of consequence. Its only
job is to produce a *sane legal* move when the real policy has failed or the
clock is nearly gone, in a handful of microseconds and with no way to throw.

If this file ever needs a bug fix, something upstream is wrong.
"""

from __future__ import annotations

from typing import Sequence

from ..core.actions import ActionCandidate
from ..core.clock import Deadline
from ..core.enums import OptionType, SelectContext, SelectType
from ..core.obs import ObsView
from .base import default_desired_count

__all__ = ["FallbackPolicy"]

#: Static preference over option types in the main phase. Attacking beats
#: passing; using a free ability beats attacking; ending the turn is last.
_MAIN_PRIORITY = {
    OptionType.ABILITY: 60.0,
    OptionType.EVOLVE: 55.0,
    OptionType.PLAY: 50.0,
    OptionType.ATTACH: 45.0,
    OptionType.ATTACK: 40.0,
    OptionType.RETREAT: 5.0,
    OptionType.DISCARD: 1.0,
    OptionType.END: 0.0,
}


class FallbackPolicy:
    name = "fallback-v1"

    def __init__(self, deck_ids: Sequence[int]) -> None:
        self._deck = list(deck_ids)

    def deck(self) -> list[int]:
        return list(self._deck)

    def score(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        deadline: Deadline,
    ) -> list[float]:
        out: list[float] = []
        negative = view.context in (
            SelectContext.DISCARD,
            SelectContext.TO_DECK,
            SelectContext.TO_DECK_BOTTOM,
            SelectContext.DISCARD_ENERGY,
            SelectContext.DISCARD_ENERGY_CARD,
        )
        for c in candidates:
            s = 0.0
            if view.select_type == SelectType.MAIN:
                s = _MAIN_PRIORITY.get(c.option_type, 10.0)
                if c.would_win_game:
                    s = 1e6
                elif c.is_knockout:
                    s += 1000.0
                else:
                    s += c.est_damage * 0.01
            elif c.option_type == OptionType.YES:
                s = 1.0
            elif c.option_type == OptionType.NO:
                s = 0.5
            else:
                # Prefer cheap, low-value cards when giving things up.
                base = float(c.card.hp if c.card else 0) * 0.01
                s = -base if negative else base
            out.append(s)
        return out

    def desired_count(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        scores: Sequence[float],
    ) -> int:
        return default_desired_count(view, candidates, scores)
