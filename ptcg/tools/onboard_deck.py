"""Week 6: the reusable per-deck onboarding pipeline.

Weeks 4-5 built every piece needed to adopt a new real deck and imitate real
players who run it (identity/deck-matched pulling, multi-directory BC
training, Week-0-standard fuzzing, a head-to-head bench) -- but as one-off
scripting reproduced by hand for `rocket_mewtwo` specifically. This composes
those *existing, unchanged* pieces into a single command so the next deck
pivot (if the metagame shifts again, or a stronger real archetype turns up)
is one documented call, not another ad hoc session.

Nothing here is new infrastructure: every step below already exists and was
already proven this session --

* legality        -> ``CardDB.validate_deck``            (Week 0)
* smoke test       -> ``ptcg.eval.harness.play_match``    (Week 0)
* deck-matched pull-> ``ptcg.tools.pull_episodes``         (Weeks 4-5)
* BC training      -> ``ptcg.train.bc``                    (Week 1, Week 4 multi-dir)
* adversarial fuzz -> ``ptcg.eval.fuzz``                    (Week 0, Week 3)
* head-to-head bench-> ``ptcg.tools.week3_bench.deck_bench`` (Week 5, generalized Week 6)

This file is glue and an honest report, not a new algorithm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..agents.bc_policy import load_bc_policy
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import load_deck
from ..eval.fuzz import fuzz_observations, fuzz_selfplay_checkpoint
from ..eval.harness import play_match
from ..train import bc as bc_module
from . import pull_episodes
from .week3_bench import deck_bench

__all__ = ["onboard_deck", "main"]


def _legality_and_smoke(deck_name: str, smoke_games: int = 5) -> dict[str, Any]:
    db = get_card_db()
    deck = load_deck(deck_name)
    problems = db.validate_deck(deck)
    result: dict[str, Any] = {"deck": deck_name, "size": len(deck), "legal": not problems, "problems": problems}
    if problems:
        return result

    exceptions = 0
    for seed in range(smoke_games):
        shell = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
        r = play_match(shell.act, shell.act, deck, deck, seed=seed)
        exceptions += shell.report.policy_exceptions + shell.report.fallback_exceptions
        if r.reason != "engine_result":
            result.setdefault("bad_games", []).append({"seed": seed, "reason": r.reason})
    result["smoke_games"] = smoke_games
    result["smoke_exceptions"] = exceptions
    result["smoke_ok"] = exceptions == 0 and not result.get("bad_games")
    return result


def onboard_deck(
    deck_name: str,
    baseline_deck: str,
    baseline_bc_checkpoint: str,
    archetype_anchor_cards: list[int] | None = None,
    leaderboard_csv: str | None = None,
    pull_days: list[str] | None = None,
    traces_dir: str | None = None,
    max_files_per_day: int = 300,
    bc_epochs: int = 6,
    bc_max_records: int = 120000,
    fuzz_cases: int = 20000,
    fuzz_games: int = 150,
    bench_games: int = 150,
    out_prefix: str | None = None,
    bc_checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one deck and return a single consolidated,
    honest report. Any step's failure is recorded and stops the pipeline
    rather than silently continuing on bad data."""
    out_prefix = out_prefix or f"artifacts/onboard_{deck_name}"
    traces_dir = traces_dir or f"artifacts/traces/replays_{deck_name}"
    report: dict[str, Any] = {"deck": deck_name, "baseline_deck": baseline_deck}
    t0 = time.time()

    print(f"[1/5] legality + smoke test: {deck_name}")
    step1 = _legality_and_smoke(deck_name)
    report["legality_and_smoke"] = step1
    if not step1["legal"] or not step1.get("smoke_ok", False):
        report["stopped_at"] = "legality_and_smoke"
        report["seconds"] = round(time.time() - t0, 1)
        return report

    # Defaults to the canonical per-deck checkpoint path -- callers doing a
    # throwaway/smoke run (e.g. tiny epoch/record counts) must pass an
    # explicit ``bc_checkpoint_path`` so they never silently clobber a real,
    # fully-trained checkpoint of the same deck name (this bit once, during
    # this file's own smoke test -- see Week 6 report).
    bc_checkpoint = bc_checkpoint_path or f"artifacts/bc_{deck_name}.pt"

    if pull_days and leaderboard_csv and archetype_anchor_cards:
        print(f"[2/5] deck-matched pull: {len(pull_days)} day(s), anchors={archetype_anchor_cards}")
        target_names = pull_episodes.load_target_names(leaderboard_csv, top_n=150)
        pull_summary = pull_episodes.pull_and_convert(
            pull_days,
            target_names,
            Path(traces_dir),
            Path(f"artifacts/onboard_raw_scratch_{deck_name}"),
            max_files_per_day=max_files_per_day,
            archetype_anchor_cards=set(archetype_anchor_cards),
        )
        report["pull"] = {k: v for k, v in pull_summary.items() if k != "decks_found"}
        report["pull"]["decks_captured"] = len(pull_summary.get("decks_found", []))
    else:
        print("[2/5] deck-matched pull: skipped (no --pull-days/--leaderboard-csv/--archetype-anchor-cards)")
        report["pull"] = {"skipped": True}

    if not Path(traces_dir).exists() or not any(Path(traces_dir).glob("*.jsonl.gz")):
        report["stopped_at"] = "training"
        report["training"] = {"error": f"no trace files in {traces_dir}"}
        report["seconds"] = round(time.time() - t0, 1)
        return report

    print(f"[3/5] BC training -> {bc_checkpoint}")
    bc_module.main(
        [
            "--traces", traces_dir,
            "--card-mode", "text",
            "--max-records", str(bc_max_records),
            "--epochs", str(bc_epochs),
            "--out", bc_checkpoint,
        ]
    )
    report["training"] = {"checkpoint": bc_checkpoint}

    print(f"[4/5] adversarial fuzz: {fuzz_cases} observations + {fuzz_games} games")
    db = get_card_db()
    deck = load_deck(deck_name)

    def make_shell():
        pol = load_bc_policy(bc_checkpoint, deck, db)
        return SafetyShell(pol, db, FallbackPolicy(deck), BudgetManager())

    obs_report = fuzz_observations(make_shell, cases=fuzz_cases, seed=0)
    sp_report = fuzz_selfplay_checkpoint(bc_checkpoint, games=fuzz_games, seed=0, opponent="random", batch=10)
    report["fuzz"] = {"observation_fuzz": obs_report, "selfplay_fuzz": sp_report}

    print(f"[5/5] bench vs {baseline_deck} (+ Crustle wall)")
    report["bench"] = deck_bench(
        deck_name, baseline_deck, bench_games,
        new_bc_checkpoint=bc_checkpoint, baseline_bc_checkpoint=baseline_bc_checkpoint,
    )

    report["seconds"] = round(time.time() - t0, 1)
    Path(f"{out_prefix}_report.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out_prefix}_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_prefix}_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--baseline-deck", default="lucario_fighting")
    ap.add_argument("--baseline-bc-checkpoint", default="artifacts/bc_text.pt")
    ap.add_argument("--archetype-anchor-cards", default=None, help="comma-separated card IDs")
    ap.add_argument("--leaderboard-csv", default=None)
    ap.add_argument("--pull-days", default=None, help="comma-separated dataset slugs")
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--max-files-per-day", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--bc-max-records", type=int, default=120000)
    ap.add_argument("--fuzz-cases", type=int, default=20000)
    ap.add_argument("--fuzz-games", type=int, default=150)
    ap.add_argument("--bench-games", type=int, default=150)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument(
        "--bc-checkpoint-path", default=None,
        help="override the trained checkpoint's output path (default: artifacts/bc_<deck>.pt); "
        "always set this for a smoke/dry run so it can't overwrite a real checkpoint",
    )
    args = ap.parse_args(argv)

    onboard_deck(
        args.deck,
        args.baseline_deck,
        args.baseline_bc_checkpoint,
        archetype_anchor_cards=[int(x) for x in args.archetype_anchor_cards.split(",")] if args.archetype_anchor_cards else None,
        leaderboard_csv=args.leaderboard_csv,
        pull_days=args.pull_days.split(",") if args.pull_days else None,
        traces_dir=args.traces_dir,
        max_files_per_day=args.max_files_per_day,
        bc_epochs=args.bc_epochs,
        bc_max_records=args.bc_max_records,
        fuzz_cases=args.fuzz_cases,
        fuzz_games=args.fuzz_games,
        bench_games=args.bench_games,
        out_prefix=args.out_prefix,
        bc_checkpoint_path=args.bc_checkpoint_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
