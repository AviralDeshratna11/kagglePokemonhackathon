"""Engine introspection and empirical schema mining.

Two jobs.

**Doctor.** Report the platform, which shared library was loaded, its SHA-256,
and which exported symbols the shipped Python bindings leave unbound. Run this
first when anything behaves oddly; the hash in particular is what detects the
organisers shipping a new ``libcg`` under the same package version, which would
silently invalidate every offline measurement in the report.

**Schema mining.** The official docs enumerate 11 ``SelectType``s, 49
``SelectContext``s, 17 ``OptionType``s and 24 ``LogType``s. Documentation and
implementation are not the same thing, so this plays games and records which
values actually occur, how often, how many options each decision offers, and
which ``(SelectType, SelectContext)`` pairs are real.

That distribution is directly useful:

* it tells the heuristic which contexts are worth hand-writing (a handful cover
  the overwhelming majority of decisions) and which are long-tail;
* the option-count distribution sizes the Week-1 pointer head;
* the decisions-per-game figure calibrates ``BudgetManager.expected_decisions``;
* any value observed that is *not* in the documented enum is a finding.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.enums import LogType, OptionType, SelectContext, SelectType
from ..core.engine import describe_platform, get_engine

__all__ = ["doctor", "mine_schema"]


def doctor() -> dict[str, Any]:
    eng = get_engine()
    fp = eng.fingerprint()
    fp["card_count"] = None
    fp["attack_count"] = None
    try:
        fp["card_count"] = len(eng.all_card_data())
        fp["attack_count"] = len(eng.all_attack())
    except Exception as exc:  # noqa: BLE001
        fp["table_error"] = str(exc)

    try:
        import kaggle_environments

        spec_path = (
            Path(kaggle_environments.__file__).parent / "envs" / "cabt" / "cabt.json"
        )
        spec = json.loads(spec_path.read_text())
        fp["kaggle_environments_version"] = getattr(kaggle_environments, "__version__", "?")
        fp["clock"] = {
            "remainingOverageTime": spec["observation"]["remainingOverageTime"],
            "actTimeout": spec["configuration"]["actTimeout"],
            "runTimeout": spec["configuration"]["runTimeout"],
            "episodeSteps": spec["configuration"]["episodeSteps"],
        }
    except Exception as exc:  # noqa: BLE001
        fp["spec_error"] = str(exc)
    return fp


def _name(enum_cls, value: int) -> str:
    try:
        return enum_cls(value).name
    except ValueError:
        return f"UNDOCUMENTED({value})"


def mine_schema(games: int = 40, seed: int = 0) -> dict[str, Any]:
    """Play games and record what the engine actually emits."""
    from ..agents.baselines import RandomPolicy
    from ..agents.fallback import FallbackPolicy
    from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
    from ..core.carddb import get_card_db
    from ..core.clock import BudgetManager
    from ..core.safety import SafetyShell
    from ..decks.registry import available_decks, load_deck
    from ..eval.harness import play_match

    db = get_card_db()
    decks = {n: load_deck(n) for n in available_decks()}
    names = list(decks)

    select_types: Counter = Counter()
    contexts: Counter = Counter()
    option_types: Counter = Counter()
    log_types: Counter = Counter()
    pairs: Counter = Counter()
    option_counts: list[int] = []
    multi_select: Counter = Counter()
    decisions_per_game: list[int] = []
    turns_per_game: list[int] = []
    undocumented: Counter = Counter()

    doc_select = {int(v) for v in SelectType}
    doc_ctx = {int(v) for v in SelectContext}
    doc_opt = {int(v) for v in OptionType}
    doc_log = {int(v) for v in LogType}

    for g in range(games):
        d0 = decks[names[g % len(names)]]
        d1 = decks[names[(g + 1) % len(names)]]
        counter = {"n": 0}

        def observe(_p: int, obs: dict, _a: list[int]) -> None:
            counter["n"] += 1
            sel = obs.get("select") or {}
            st = int(sel.get("type", -1))
            ctx = int(sel.get("context", -1))
            select_types[_name(SelectType, st)] += 1
            contexts[_name(SelectContext, ctx)] += 1
            pairs[(_name(SelectType, st), _name(SelectContext, ctx))] += 1
            if st not in doc_select:
                undocumented[f"SelectType={st}"] += 1
            if ctx not in doc_ctx:
                undocumented[f"SelectContext={ctx}"] += 1

            opts = sel.get("option") or []
            option_counts.append(len(opts))
            lo, hi = int(sel.get("minCount") or 0), int(sel.get("maxCount") or 0)
            if hi > lo:
                multi_select[f"{lo}..{hi}"] += 1
            for o in opts:
                ot = int(o.get("type", -1))
                option_types[_name(OptionType, ot)] += 1
                if ot not in doc_opt:
                    undocumented[f"OptionType={ot}"] += 1
            for L in obs.get("logs") or []:
                lt = int(L.get("type", -1))
                log_types[_name(LogType, lt)] += 1
                if lt not in doc_log:
                    undocumented[f"LogType={lt}"] += 1

        def make(deck, s):
            pol = (
                HeuristicPolicy(deck, db, HeuristicConfig())
                if s % 2 == 0
                else RandomPolicy(deck, seed=s)
            )
            return SafetyShell(pol, db, FallbackPolicy(deck), BudgetManager()).act

        r = play_match(make(d0, seed + g), make(d1, seed + g + 1), d0, d1,
                       seed=seed + g, on_step=observe)
        decisions_per_game.append(counter["n"])
        turns_per_game.append(r.turns)

    def pct(c: Counter) -> list[dict[str, Any]]:
        total = sum(c.values()) or 1
        return [
            {"value": k, "count": v, "share": round(v / total, 4)}
            for k, v in c.most_common()
        ]

    counts_sorted = sorted(option_counts)
    return {
        "games": games,
        "decisions_per_game": {
            "mean": round(statistics.mean(decisions_per_game), 1),
            "median": statistics.median(decisions_per_game),
            "max": max(decisions_per_game),
            "per_player_mean": round(statistics.mean(decisions_per_game) / 2, 1),
        },
        "turns_per_game": {
            "mean": round(statistics.mean(turns_per_game), 1),
            "max": max(turns_per_game),
        },
        "options_per_decision": {
            "mean": round(statistics.mean(option_counts), 2),
            "median": statistics.median(option_counts),
            "p95": counts_sorted[int(0.95 * (len(counts_sorted) - 1))],
            "max": max(option_counts),
        },
        "select_types": pct(select_types),
        "contexts": pct(contexts),
        "option_types": pct(option_types),
        "log_types": pct(log_types),
        "multi_select_ranges": pct(multi_select),
        "observed_type_context_pairs": [
            {"select_type": a, "context": b, "count": n} for (a, b), n in pairs.most_common()
        ],
        "undocumented_values": dict(undocumented),
        "documented_but_never_seen": {
            "contexts": sorted(
                c.name for c in SelectContext if c.name not in contexts
            ),
            "option_types": sorted(
                o.name for o in OptionType if o.name not in option_types
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="doctor only, no games")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    report: dict[str, Any] = {"engine": doctor()}
    if not args.quick:
        report["schema"] = mine_schema(games=args.games)

    blob = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    print(blob)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
