"""Week-8 verification bench: does the updated heuristic (search-before-draw
ordering, recovery-aware discarding, pre-shuffle clearing, prized-resource
awareness, late-game bench thinning) actually help, or at least not regress,
relative to the pre-Week-8 heuristic -- on both registered decks.

``_PreWeek8HeuristicPolicy`` reconstructs the exact pre-Week-8 scoring by
overriding only the methods Week 8 touched, rather than relying on git state
(there is no clean commit boundary between Week 7 and Week 8 in this
session's history).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.obs import ObsView
from ..core.safety import SafetyShell
from ..decks.registry import load_deck
from ..eval.arena import round_robin

__all__ = ["heuristic_ab_bench", "main"]


class _PreWeek8HeuristicPolicy(HeuristicPolicy):
    """The heuristic exactly as it stood at the end of Week 7."""

    name = "heuristic-pre-week8"

    def _score_discard(self, view: ObsView, c) -> float:  # type: ignore[override]
        return -50.0

    def _pre_shuffle_clearing_bonus(self, view: ObsView) -> float:  # type: ignore[override]
        return 0.0

    def _bench_thinning_bonus(self, view: ObsView, card) -> float:  # type: ignore[override]
        return 0.0

    def _all_other_copies_prized(self, view: ObsView, name: str) -> bool:  # type: ignore[override]
        return False


def _pre_week8_config() -> HeuristicConfig:
    cfg = HeuristicConfig()
    cfg.search_bonus = 240.0  # the pre-Week-8 value, below draw_bonus
    return cfg


def _shell(policy_factory, deck):
    def make():
        pol = policy_factory(deck)
        return SafetyShell(pol, get_card_db(), FallbackPolicy(deck), BudgetManager()).act

    return make


def heuristic_ab_bench(deck_name: str, games: int, seed: int = 0) -> dict[str, Any]:
    db = get_card_db()
    deck = load_deck(deck_name)

    entries: dict[str, Any] = {
        "new_heuristic": (
            _shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), deck),
            deck,
        ),
        "pre_week8_heuristic": (
            _shell(lambda d: _PreWeek8HeuristicPolicy(d, db, _pre_week8_config()), deck),
            deck,
        ),
        "greedy": (_shell(lambda d: GreedyAttackPolicy(d), deck), deck),
        "first": (_shell(lambda d: FirstPolicy(d), deck), deck),
    }

    t0 = time.time()
    table, ladder = round_robin(entries, games=games, progress=print)
    return {
        "deck": deck_name,
        "games_per_pair": games,
        "seconds": round(time.time() - t0, 2),
        "ladder": ladder.table(),
        **table.report(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="artifacts/week8")
    args = ap.parse_args(argv)

    results = {}
    for deck_name in ("kangaskhan_crustle", "lucario_fighting"):
        info = heuristic_ab_bench(deck_name, args.games, args.seed)
        results[deck_name] = info
        print(f"-- {deck_name} done ({info['seconds']}s) --")

    out = Path(f"{args.out_prefix}_ab_bench.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
