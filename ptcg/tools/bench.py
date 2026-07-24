"""Benchmark the heuristic against the baseline panel and both decks.

Produces the Week-0 acceptance evidence: win rates with Wilson intervals, a
matchup matrix, worst-case matchup per agent, and latency percentiles measured
against the real 600 s clock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy, RandomPolicy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import available_decks, load_deck
from ..eval.arena import Ladder, round_robin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", default="lucario_fighting")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--cross-deck", action="store_true", help="also run deck vs deck")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    db = get_card_db()
    deck = load_deck(args.deck)

    def shell(policy, d):
        return SafetyShell(policy, db, FallbackPolicy(d), BudgetManager()).act

    entries = {
        "heuristic": (lambda: shell(HeuristicPolicy(deck, db, HeuristicConfig()), deck), deck),
        "greedy": (lambda: shell(GreedyAttackPolicy(deck), deck), deck),
        "first": (lambda: shell(FirstPolicy(deck), deck), deck),
        "random": (lambda: shell(RandomPolicy(deck, seed=7), deck), deck),
    }

    if args.cross_deck:
        for name in available_decks():
            if name == args.deck:
                continue
            other = load_deck(name)
            entries[f"heuristic@{name}"] = (
                lambda o=other: shell(HeuristicPolicy(o, db, HeuristicConfig()), o),
                other,
            )

    table, ladder = round_robin(
        entries,
        games=args.games,
        progress=lambda s: print(f"  {s}", file=sys.stderr),
    )

    report = {
        "deck": args.deck,
        "games_per_pair": args.games,
        "ladder": ladder.table(),
        **table.report(),
    }
    blob = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    print(blob)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
