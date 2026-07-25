"""League registry and Prioritized Fictitious Self-Play (PFSP) opponent
sampling for Week 2.

Deliberately scoped to what already exists and is proven: the two registered
decks, the four baseline bots (``ptcg/agents/baselines.py``), the Week-0
heuristic, and the Week-1 BC network as the frozen teacher. The plan's own
Week 2 spec names scripted Dragapult/Dusknoir and Crustle-wall bots, but no
deck data exists for those archetypes on this machine -- building them is
real deck-construction work, not something to improvise here (see the plan
file's "explicitly deferred" section).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy, RandomPolicy
from ..agents.bc_policy import load_bc_policy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.carddb import CardDB

__all__ = ["League"]

PolicyFactory = Callable[[], object]


class League:
    """Tracks an exponential-moving-average win rate of the *current trainee*
    against each opponent, and samples opponents skewed toward whoever the
    trainee is currently losing to (PFSP), so training time concentrates on
    weak matchups instead of re-grinding easy wins -- directly targeting H5
    (flattening the matchup spread) rather than uniform self-play.
    """

    def __init__(
        self,
        db: CardDB,
        deck: list[int],
        frozen_teacher_ckpt: str | Path,
        max_past_selves: int = 3,
        ema_alpha: float = 0.25,
        pfsp_power: float = 2.0,
    ) -> None:
        self.db = db
        self.deck = deck
        self.frozen_teacher_ckpt = str(frozen_teacher_ckpt)
        self.max_past_selves = max_past_selves
        self.ema_alpha = ema_alpha
        self.pfsp_power = pfsp_power
        self.past_selves: list[str] = []
        self.win_rate: dict[str, float] = {}

    # -- membership -----------------------------------------------------

    def base_members(self) -> dict[str, PolicyFactory]:
        d, db = self.deck, self.db
        return {
            "random": lambda: RandomPolicy(d, seed=None),
            "first": lambda: FirstPolicy(d),
            "greedy": lambda: GreedyAttackPolicy(d),
            "heuristic": lambda: HeuristicPolicy(d, db, HeuristicConfig()),
            "frozen-teacher": lambda: load_bc_policy(self.frozen_teacher_ckpt, d, db),
        }

    def all_members(self) -> dict[str, PolicyFactory]:
        members = dict(self.base_members())
        for i, ckpt in enumerate(self.past_selves):
            members[f"past-self-{i}"] = (lambda c=ckpt: load_bc_policy(c, self.deck, self.db))
        return members

    def make_opponent(self, name: str):
        members = self.all_members()
        if name not in members:
            raise KeyError(f"unknown league member {name!r}")
        return members[name]()

    # -- PFSP -------------------------------------------------------------

    def update_win_rate(self, opponent: str, outcome: float) -> None:
        """``outcome`` is the trainee's result: 1.0 win / 0.0 loss / 0.5 draw."""
        prev = self.win_rate.get(opponent, 0.5)
        self.win_rate[opponent] = (1 - self.ema_alpha) * prev + self.ema_alpha * outcome

    def sample_opponent(self, rng: random.Random) -> str:
        names = list(self.all_members().keys())
        weights = [max(1e-3, (1.0 - self.win_rate.get(n, 0.5) + 0.05)) ** self.pfsp_power for n in names]
        total = sum(weights)
        probs = [w / total for w in weights]
        return rng.choices(names, weights=probs, k=1)[0]

    def worst_case(self) -> tuple[str, float] | None:
        if not self.win_rate:
            return None
        name = min(self.win_rate, key=self.win_rate.get)
        return name, self.win_rate[name]

    # -- checkpoint pool ----------------------------------------------------

    def push_past_self(self, ckpt_path: str | Path) -> None:
        self.past_selves.append(str(ckpt_path))
        if len(self.past_selves) > self.max_past_selves:
            self.past_selves.pop(0)
