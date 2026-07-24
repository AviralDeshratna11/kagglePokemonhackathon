"""Time budgeting against the tournament's cumulative 600-second clock.

Verified from ``kaggle_environments/envs/cabt/cabt.json``::

    "configuration": {"episodeSteps": ..., "actTimeout": 0, "runTimeout": ...},
    "observation":   {"remainingOverageTime": 600}

``actTimeout`` is **0**, which means there is no separate per-move allowance:
*every* millisecond an agent spends comes out of the single 600 s overage pool,
and exhausting it is an immediate loss. The engine helpfully reports the live
remaining figure as ``obs["remainingOverageTime"]``, so we never have to guess.

The manager does three things:

* keeps a private safety margin so we can never be the player who times out;
* divides the remaining pool by an *estimate of remaining decisions*, rather
  than a fixed per-move constant, so that early cheap decisions bank time for
  the complicated mid-game ones;
* exposes a panic level that downstream policies can use to degrade gracefully
  (Week 3: skip tree search; Week 0: skip the expensive scoring paths).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = ["Deadline", "BudgetManager"]


@dataclass
class Deadline:
    """A monotonic deadline you can cheaply poll."""

    started: float
    allotted: float

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        return self.allotted - self.elapsed

    def expired(self, margin: float = 0.0) -> bool:
        return self.remaining <= margin

    def fraction_used(self) -> float:
        return 0.0 if self.allotted <= 0 else min(1.0, self.elapsed / self.allotted)


@dataclass
class BudgetManager:
    """Adaptive allocator over the cumulative per-player clock."""

    total: float = 600.0
    #: Never spend below this. A forfeit costs a whole game; a fast move costs
    #: a few Elo at worst.
    reserve: float = 25.0
    #: Hard ceiling for a single decision even when we are rich in time.
    max_per_decision: float = 3.0
    #: Floor, so that a bad estimate cannot starve us into a nonsense move.
    min_per_decision: float = 0.010
    #: Decisions one player makes in a full game. Measured over 60 heuristic and
    #: random self-play games (``tools/engine_report.py``): mean 55.8 per
    #: player, worst observed game ~113. The default sits above the mean so the
    #: allocator is not systematically starved on long games, and the engine's
    #: live ``remainingOverageTime`` corrects any drift anyway.
    expected_decisions: int = 90

    used: float = 0.0
    decisions: int = 0
    _ewma_cost: float = 0.0
    _last_reported_remaining: float | None = None
    history: list[float] = field(default_factory=list)

    # -- syncing with the engine -------------------------------------------

    def sync(self, remaining_overage: float | None) -> None:
        """Trust the engine's own figure over our local accounting.

        Local timing misses interpreter start-up, model loading and any time
        the harness spends outside ``agent()``. ``remainingOverageTime`` does
        not, so whenever it is present it wins.
        """
        if remaining_overage is None:
            return
        self._last_reported_remaining = float(remaining_overage)
        self.used = max(self.used, self.total - float(remaining_overage))

    @property
    def remaining(self) -> float:
        if self._last_reported_remaining is not None:
            return max(0.0, self._last_reported_remaining - (0.0))
        return max(0.0, self.total - self.used)

    @property
    def spendable(self) -> float:
        return max(0.0, self.remaining - self.reserve)

    # -- allocation ---------------------------------------------------------

    def estimate_remaining_decisions(self, turn: int) -> int:
        """How many more decisions will we be asked to make?

        Games are roughly 12-25 turns each side; decision density is highest in
        the mid-game. A linear decay from ``expected_decisions`` with a floor of
        20 is deliberately crude -- it only has to be right to within a factor
        of two for the allocator to behave sensibly, and being crude keeps it
        free.
        """
        made = self.decisions
        by_turn = max(20, self.expected_decisions - 4 * max(0, turn))
        by_count = max(20, self.expected_decisions - made)
        return max(20, min(by_turn, by_count))

    def allot(self, turn: int = 0, importance: float = 1.0) -> Deadline:
        """Open a deadline for the decision that is about to be made.

        ``importance`` lets a policy ask for more time on genuinely hard
        decisions (Week 3 uses the policy-entropy gate for exactly this).
        """
        n = self.estimate_remaining_decisions(turn)
        fair_share = self.spendable / n if n else self.min_per_decision
        allot = fair_share * max(0.25, min(4.0, importance))
        allot = max(self.min_per_decision, min(self.max_per_decision, allot))
        if self.panic_level >= 2:
            allot = self.min_per_decision
        return Deadline(time.monotonic(), allot)

    def close(self, deadline: Deadline) -> float:
        """Record actual spend. Returns the elapsed seconds."""
        elapsed = deadline.elapsed
        self.used += elapsed
        self.decisions += 1
        self.history.append(elapsed)
        self._ewma_cost = (
            elapsed if self._ewma_cost == 0.0 else 0.9 * self._ewma_cost + 0.1 * elapsed
        )
        return elapsed

    # -- health -------------------------------------------------------------

    @property
    def panic_level(self) -> int:
        """0 = healthy, 1 = economise, 2 = emergency (fallback policy only)."""
        r = self.remaining
        if r <= self.reserve:
            return 2
        if r <= self.reserve * 3:
            return 1
        return 0

    def stats(self) -> dict[str, float | int]:
        return {
            "used": round(self.used, 3),
            "remaining": round(self.remaining, 3),
            "decisions": self.decisions,
            "mean_ms": round(1000 * (self.used / self.decisions), 3) if self.decisions else 0.0,
            "ewma_ms": round(1000 * self._ewma_cost, 3),
            "max_ms": round(1000 * max(self.history), 3) if self.history else 0.0,
            "panic": self.panic_level,
        }
