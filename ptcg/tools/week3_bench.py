"""Week-3 verification benches: does distillation actually cost quality, and
does the confidence-gated router actually help?

Two separate round robins, both against the same panel shape prior weeks'
benches used (heuristic-piloted own deck, greedy, first), so results are
comparable across weeks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..agents.bc_policy import load_bc_policy
from ..agents.baselines import FirstPolicy, GreedyAttackPolicy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..agents.router_policy import RouterPolicy
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import load_deck
from ..eval.arena import round_robin

__all__ = ["distillation_bench", "router_bench", "main"]


def _shell(policy_factory, deck):
    def make():
        pol = policy_factory(deck)
        return SafetyShell(pol, get_card_db(), FallbackPolicy(deck), BudgetManager()).act

    return make


def distillation_bench(games: int, seed: int = 0) -> dict[str, Any]:
    db = get_card_db()
    deck = load_deck("lucario_fighting")

    entries = {
        "teacher_bc_text": (_shell(lambda d: load_bc_policy("artifacts/bc_text.pt", d, db), deck), deck),
        "distill_medium": (_shell(lambda d: load_bc_policy("artifacts/distill_medium.pt", d, db), deck), deck),
        "distill_small": (_shell(lambda d: load_bc_policy("artifacts/distill_small.pt", d, db), deck), deck),
        "heuristic": (_shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), deck), deck),
        "greedy": (_shell(lambda d: GreedyAttackPolicy(d), deck), deck),
        "first": (_shell(lambda d: FirstPolicy(d), deck), deck),
    }

    t0 = time.time()
    table, ladder = round_robin(entries, games=games, progress=print)
    return {
        "games_per_pair": games,
        "seconds": round(time.time() - t0, 2),
        "ladder": ladder.table(),
        **table.report(),
    }


def router_bench(games: int, seed: int = 0) -> dict[str, Any]:
    db = get_card_db()
    deck = load_deck("lucario_fighting")

    def make_router(d):
        net = load_bc_policy("artifacts/bc_text.pt", d, db)
        heur = HeuristicPolicy(d, db, HeuristicConfig())
        return RouterPolicy(d, net, heur)

    entries = {
        "router": (_shell(make_router, deck), deck),
        "network_only": (_shell(lambda d: load_bc_policy("artifacts/bc_text.pt", d, db), deck), deck),
        "heuristic_only": (_shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), deck), deck),
        "greedy": (_shell(lambda d: GreedyAttackPolicy(d), deck), deck),
        "first": (_shell(lambda d: FirstPolicy(d), deck), deck),
    }

    t0 = time.time()
    table, ladder = round_robin(entries, games=games, progress=print)
    return {
        "games_per_pair": games,
        "seconds": round(time.time() - t0, 2),
        "ladder": ladder.table(),
        **table.report(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["distill", "router", "both"], default="both")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="artifacts/week3")
    args = ap.parse_args(argv)

    if args.which in ("distill", "both"):
        info = distillation_bench(args.games, args.seed)
        out = Path(f"{args.out_prefix}_distill_bench.json")
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if args.which in ("router", "both"):
        info = router_bench(args.games, args.seed)
        out = Path(f"{args.out_prefix}_router_bench.json")
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
