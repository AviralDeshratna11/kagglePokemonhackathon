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

__all__ = ["distillation_bench", "router_bench", "replay_bench", "archetype_bench", "main"]


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


def replay_bench(games: int, seed: int = 0) -> dict[str, Any]:
    """Week 4: does real-ladder imitation learning actually beat the
    self-play-derived teacher, not just tie it like everything else has."""
    db = get_card_db()
    deck = load_deck("lucario_fighting")

    def make_router(d):
        net = load_bc_policy("artifacts/bc_text.pt", d, db)
        heur = HeuristicPolicy(d, db, HeuristicConfig())
        return RouterPolicy(d, net, heur)

    entries = {
        "bc_replay": (_shell(lambda d: load_bc_policy("artifacts/bc_replay.pt", d, db), deck), deck),
        "bc_replay_blend": (_shell(lambda d: load_bc_policy("artifacts/bc_replay_blend.pt", d, db), deck), deck),
        "teacher_bc_text": (_shell(lambda d: load_bc_policy("artifacts/bc_text.pt", d, db), deck), deck),
        "heuristic": (_shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), deck), deck),
        "router": (_shell(make_router, deck), deck),
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


def deck_bench(
    new_deck: str,
    baseline_deck: str,
    games: int,
    seed: int = 0,
    new_bc_checkpoint: str | None = None,
    baseline_bc_checkpoint: str | None = None,
    include_crustle: bool = True,
    include_router: bool = True,
) -> dict[str, Any]:
    """Week 6: the general form of Week 5's ``archetype_bench`` -- any newly
    onboarded deck (optionally with its own BC checkpoint) vs. any baseline
    deck (optionally with its own checkpoint), each side piloting its own
    deck exactly like two real submissions would. Parameterized so a future
    deck pivot reuses this bench instead of a new hardcoded copy
    (``ptcg/tools/onboard_deck.py`` calls this directly).

    Week 9: a confidence-gated ``RouterPolicy`` (heuristic + BC) is built
    for *either* side whenever that side has a BC checkpoint -- previously
    only ``baseline_deck`` got one, so there was no way to see whether a
    router beats a raw heuristic/BC split for the deck actually being
    onboarded."""
    from ..tools.deck_search import build_crustle_test_deck

    db = get_card_db()
    new_d = load_deck(new_deck)
    base_d = load_deck(baseline_deck)

    entries: dict[str, Any] = {
        f"{new_deck}_heuristic": (_shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), new_d), new_d),
        f"{baseline_deck}_heuristic": (_shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), base_d), base_d),
        "greedy": (_shell(lambda d: GreedyAttackPolicy(d), new_d), new_d),
        "first": (_shell(lambda d: FirstPolicy(d), new_d), new_d),
    }

    if new_bc_checkpoint:
        entries[f"{new_deck}_bc"] = (
            _shell(lambda d, ckpt=new_bc_checkpoint: load_bc_policy(ckpt, d, db), new_d),
            new_d,
        )
        if include_router:
            def make_new_router(d, ckpt=new_bc_checkpoint):
                net = load_bc_policy(ckpt, d, db)
                heur = HeuristicPolicy(d, db, HeuristicConfig())
                return RouterPolicy(d, net, heur)

            entries[f"{new_deck}_router"] = (_shell(make_new_router, new_d), new_d)
    if baseline_bc_checkpoint:
        entries[f"{baseline_deck}_bc"] = (
            _shell(lambda d, ckpt=baseline_bc_checkpoint: load_bc_policy(ckpt, d, db), base_d),
            base_d,
        )
        if include_router:
            def make_router(d, ckpt=baseline_bc_checkpoint):
                net = load_bc_policy(ckpt, d, db)
                heur = HeuristicPolicy(d, db, HeuristicConfig())
                return RouterPolicy(d, net, heur)

            entries[f"{baseline_deck}_router"] = (_shell(make_router, base_d), base_d)

    if include_crustle:
        crustle_deck = build_crustle_test_deck(db)
        entries["crustle_wall"] = (
            _shell(lambda d: HeuristicPolicy(d, db, HeuristicConfig()), crustle_deck),
            crustle_deck,
        )

    t0 = time.time()
    table, ladder = round_robin(entries, games=games, progress=print)
    return {
        "games_per_pair": games,
        "seconds": round(time.time() - t0, 2),
        "ladder": ladder.table(),
        **table.report(),
    }


def archetype_bench(games: int, seed: int = 0) -> dict[str, Any]:
    """Week 5's original bench, kept as a thin, backward-compatible call
    into the now-general :func:`deck_bench`."""
    return deck_bench(
        "rocket_mewtwo",
        "lucario_fighting",
        games,
        seed,
        new_bc_checkpoint="artifacts/bc_rocket_mewtwo.pt",
        baseline_bc_checkpoint="artifacts/bc_text.pt",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["distill", "router", "replay", "archetype", "both"], default="both")
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

    if args.which == "replay":
        info = replay_bench(args.games, args.seed)
        out = Path(f"{args.out_prefix}_replay_bench.json")
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    if args.which == "archetype":
        info = archetype_bench(args.games, args.seed)
        out = Path(f"{args.out_prefix}_archetype_bench.json")
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
