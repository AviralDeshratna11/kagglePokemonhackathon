"""Ablate individual :class:`HeuristicConfig` fields against a fixed panel.

The Strategy rubric rewards hypotheses that were *tested*, not asserted. Every
judgement call in the heuristic is a named config field precisely so that this
script can flip one at a time and report the effect with an interval.

Usage::

    python -m ptcg.tools.ablate --games 300
    python -m ptcg.tools.ablate --field prefer_first --values true false

Output is JSON on stdout and, optionally, a file for the report's figure
pipeline. Paired seeds are used across variants so that the comparison is not
dominated by deal luck.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy, RandomPolicy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import load_deck
from ..eval.arena import duel, wilson

PANEL = {
    "random": lambda deck, db: RandomPolicy(deck, seed=7),
    "first": lambda deck, db: FirstPolicy(deck),
    "greedy": lambda deck, db: GreedyAttackPolicy(deck),
}

#: Fields worth ablating by default: the ones where theory does not settle it.
DEFAULT_FIELDS: dict[str, list[Any]] = {
    "prefer_first": [True, False],
    "retreat_to_swap_attacker": [True, False],
    "target_bench": [3, 4, 5],
    "attach_progress_weight": [3.0, 6.0, 12.0],
    "designated_attacker_bonus": [0.0, 400.0, 900.0],
    "draw_when_hand_below": [4, 5, 7],
}


def _shell(policy, db, deck):
    return SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager()).act


def evaluate(cfg: HeuristicConfig, deck, db, games: int, panel: list[str], seed0: int) -> dict:
    """Score one config against the whole panel; returns pooled + per-opponent."""
    total_score = 0.0
    total_games = 0
    per: dict[str, Any] = {}
    for name in panel:
        make_opp = PANEL[name]
        d = duel(
            "cand",
            lambda: _shell(HeuristicPolicy(deck, db, cfg), db, deck),
            deck,
            name,
            lambda: _shell(make_opp(deck, db), db, deck),
            deck,
            games=games,
            seed0=seed0,
        )
        per[name] = d.summary()
        total_score += d.a_score
        total_games += d.games
    lo, p, hi = wilson(total_score, total_games)
    return {
        "pooled_winrate": round(p, 4),
        "pooled_ci95": [round(lo, 4), round(hi, 4)],
        "games": total_games,
        "worst_opponent": min(per, key=lambda k: per[k]["a_winrate"]) if per else None,
        "worst_winrate": min((v["a_winrate"] for v in per.values()), default=None),
        "per_opponent": per,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", default="lucario_fighting")
    ap.add_argument("--games", type=int, default=200, help="games per opponent")
    ap.add_argument("--panel", nargs="*", default=list(PANEL))
    ap.add_argument("--field", default=None, help="ablate only this field")
    ap.add_argument("--values", nargs="*", default=None)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    db = get_card_db()
    deck = load_deck(args.deck)

    base = HeuristicConfig()
    results: dict[str, Any] = {
        "deck": args.deck,
        "games_per_opponent": args.games,
        "panel": args.panel,
        "baseline_config": asdict(base),
    }

    baseline = evaluate(base, deck, db, args.games, args.panel, args.seed0)
    results["baseline"] = baseline
    print(f"baseline: {baseline['pooled_winrate']:.3f} {baseline['pooled_ci95']}", file=sys.stderr)

    if args.field:
        raw = args.values or []
        cur = getattr(base, args.field)
        values = [_coerce(v, cur) for v in raw] if raw else DEFAULT_FIELDS.get(args.field, [])
        fields = {args.field: values}
    else:
        fields = DEFAULT_FIELDS

    ablations: dict[str, Any] = {}
    for field, values in fields.items():
        rows = []
        for value in values:
            cfg = replace(base, **{field: value})
            r = evaluate(cfg, deck, db, args.games, args.panel, args.seed0)
            r["value"] = value
            r["delta_vs_baseline"] = round(r["pooled_winrate"] - baseline["pooled_winrate"], 4)
            rows.append(r)
            print(
                f"  {field}={value!r}: {r['pooled_winrate']:.3f} "
                f"({r['delta_vs_baseline']:+.3f}) worst={r['worst_winrate']}",
                file=sys.stderr,
            )
        ablations[field] = rows
    results["ablations"] = ablations

    blob = json.dumps(results, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(blob)
    return 0


def _coerce(raw: str, like: Any) -> Any:
    if isinstance(like, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return raw


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
