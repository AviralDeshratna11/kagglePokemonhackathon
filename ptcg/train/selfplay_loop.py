"""Week-2 iteration loop: generate self-play games via PFSP opponent
sampling -> train a few epochs (policy gradient + value + frozen-teacher KL +
belief) -> update the league's win-rate table -> checkpoint -> repeat, for a
bounded number of rounds.

Two fixes here address a real failure mode found across two prior runs: the
*training-time* win rates (measured while the trainee explores stochastically
via ``NetworkPolicy``'s Gumbel sampling) looked strong every round, but a
clean, deterministic, no-exploration bench of the resulting checkpoint
repeatedly came back at a coin flip against both the frozen BC teacher and
the Week-0 heuristic. A policy can win games through randomness while its
*deterministic* (argmax) behavior -- what actually gets deployed -- never
really improves.

1. **Temperature annealing.** Exploration temperature now decays linearly
   from ``temperature_start`` to ``temperature_end`` across rounds instead of
   staying fixed at 1.0 for the whole run. Late rounds train on data closer
   to what the deployed deterministic policy will actually do, closing the
   gap between "wins under exploration" and "wins for real."
2. **A trustworthy signal every round.** After each round's training, a
   small *deterministic* duel (no exploration, real ``BCPolicy`` loaded from
   the just-saved checkpoint) is run against the frozen teacher and the
   heuristic, with a Wilson CI. This is what gets trusted -- the noisy
   training-time EMA win rates are still logged for diagnostics but are no
   longer the only signal available until the very end of the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..agents.bc_policy import load_bc_policy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..agents.network import BeliefSetDMCLite, CardTable
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import load_deck
from ..eval.arena import duel
from ..tools.generate_league_selfplay import generate_round
from .dataset import load_records
from .league import League
from .selfplay_rl import train_selfplay_round

__all__ = ["run", "deterministic_check"]


def deterministic_check(
    ckpt_path: str | Path,
    frozen_teacher_ckpt: str | Path,
    db,
    deck: list[int],
    games: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    """The trustworthy number: a real, no-exploration duel against the frozen
    teacher and the heuristic, through the exact same ``SafetyShell`` +
    ``BCPolicy`` path a submission would use. Small ``games`` by design --
    this runs every round, so it has to be cheap; the final gate-check bench
    (a separate, larger round-robin) is what actually confirms a result."""

    def shell(policy):
        return SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager()).act

    trainee = shell(load_bc_policy(ckpt_path, deck, db))
    teacher = shell(load_bc_policy(frozen_teacher_ckpt, deck, db))
    heuristic = shell(HeuristicPolicy(deck, db, HeuristicConfig()))

    d_teacher = duel("trainee", lambda: trainee, deck, "frozen-teacher", lambda: teacher, deck, games=games, seed0=seed)
    d_heur = duel("trainee", lambda: trainee, deck, "heuristic", lambda: heuristic, deck, games=games, seed0=seed + 1000)

    return {
        "vs_frozen_teacher": d_teacher.summary(),
        "vs_heuristic": d_heur.summary(),
    }


def run(
    deck_name: str = "lucario_fighting",
    frozen_teacher_ckpt: str = "artifacts/bc_text.pt",
    out_prefix: str = "artifacts/selfplay_v2",
    rounds: int = 3,
    games_per_round: int = 200,
    epochs_per_round: int = 2,
    batch_size: int = 64,
    lr: float = 2e-4,
    temperature_start: float = 1.0,
    temperature_end: float = 0.15,
    replay_window: int = 2,
    max_records_per_round: int = 40000,
    max_past_selves: int = 3,
    pfsp_power: float = 2.0,
    ema_alpha: float = 0.25,
    eval_games: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    db = get_card_db()
    deck = load_deck(deck_name)
    table = CardTable.build(db, with_text=True)

    # Warm start from the frozen teacher's weights: self-play refines an
    # already-competent policy instead of rediscovering "illegal-feeling
    # moves are bad" from a random init.
    teacher_ckpt = torch.load(frozen_teacher_ckpt, map_location="cpu", weights_only=False)
    model = BeliefSetDMCLite(card_mode="text", n_cards=teacher_ckpt["n_cards"])
    model.load_state_dict(teacher_ckpt["state_dict"])

    teacher_model = BeliefSetDMCLite(card_mode="text", n_cards=teacher_ckpt["n_cards"])
    teacher_model.load_state_dict(teacher_ckpt["state_dict"])
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    league = League(
        db, deck, frozen_teacher_ckpt,
        max_past_selves=max_past_selves, ema_alpha=ema_alpha, pfsp_power=pfsp_power,
    )

    round_dirs: list[Path] = []
    history: list[dict[str, Any]] = []
    ckpt_path: Path | None = None

    for r in range(rounds):
        temperature = temperature_start + (temperature_end - temperature_start) * (
            r / max(1, rounds - 1)
        )
        print(f"=== round {r}/{rounds} (temperature={temperature:.3f}) ===")
        round_dir = Path(out_prefix) / f"round_{r}"
        gen_info = generate_round(
            league,
            model,
            table,
            "text",
            round_dir,
            games_per_round,
            temperature=temperature,
            seed_start=seed * 100_000 + r * 10_000,
            round_tag=f"r{r}",
        )
        round_dirs.append(round_dir)
        print(f"  generated {gen_info['total_records']} records in {gen_info['seconds']}s")
        print(f"  opponents played: {gen_info['opponent_counts']}")

        window_dirs = round_dirs[-replay_window:]
        per_dir_cap = max(1, max_records_per_round // len(window_dirs))
        records: list[dict[str, Any]] = []
        for i, d in enumerate(window_dirs):
            records.extend(load_records(d, max_records=per_dir_cap, seed=seed + i))
        print(f"  training on {len(records)} records from {len(window_dirs)} round(s)")

        train_info = train_selfplay_round(
            records,
            table,
            db,
            model,
            teacher_model,
            card_mode="text",
            epochs=epochs_per_round,
            batch_size=batch_size,
            lr=lr,
            seed=seed + r,
        )

        ckpt_path = Path(out_prefix) / f"checkpoint_round_{r}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "card_mode": "text",
                "n_cards": table.n_cards,
                "card_ids": table.card_ids,
                "card_struct": table.struct,
                "card_text": table.text,
                "round": r,
            },
            ckpt_path,
        )
        league.push_past_self(ckpt_path)

        print("  running deterministic check (the trustworthy number)...")
        real = deterministic_check(ckpt_path, frozen_teacher_ckpt, db, deck, games=eval_games, seed=seed * 1000 + r)
        print(
            f"  REAL (deterministic) vs frozen-teacher: {real['vs_frozen_teacher']['a_winrate']:.3f} "
            f"ci95={real['vs_frozen_teacher']['ci95']}"
        )
        print(
            f"  REAL (deterministic) vs heuristic:      {real['vs_heuristic']['a_winrate']:.3f} "
            f"ci95={real['vs_heuristic']['ci95']}"
        )

        worst = league.worst_case()
        history.append(
            {
                "round": r,
                "temperature": temperature,
                "generation": gen_info,
                "training": train_info,
                "training_time_win_rate_noisy": dict(league.win_rate),
                "worst_case_noisy": worst,
                "deterministic_check": real,
                "checkpoint": str(ckpt_path),
            }
        )
        print(f"  (noisy) training-time win rates vs league: {league.win_rate}")

    return {
        "deck": deck_name,
        "rounds": rounds,
        "games_per_round": games_per_round,
        "history": history,
        "final_checkpoint": str(ckpt_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", default="lucario_fighting")
    ap.add_argument("--frozen-teacher", default="artifacts/bc_text.pt")
    ap.add_argument("--out-prefix", default="artifacts/selfplay_v2")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games-per-round", type=int, default=200)
    ap.add_argument("--epochs-per-round", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temperature-start", type=float, default=1.0)
    ap.add_argument("--temperature-end", type=float, default=0.15)
    ap.add_argument("--replay-window", type=int, default=2)
    ap.add_argument("--max-records-per-round", type=int, default=40000)
    ap.add_argument("--max-past-selves", type=int, default=3)
    ap.add_argument("--pfsp-power", type=float, default=2.0)
    ap.add_argument("--ema-alpha", type=float, default=0.25)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-out", default="artifacts/week2_report.json")
    args = ap.parse_args(argv)

    info = run(
        deck_name=args.deck,
        frozen_teacher_ckpt=args.frozen_teacher,
        out_prefix=args.out_prefix,
        rounds=args.rounds,
        games_per_round=args.games_per_round,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        replay_window=args.replay_window,
        max_records_per_round=args.max_records_per_round,
        max_past_selves=args.max_past_selves,
        pfsp_power=args.pfsp_power,
        ema_alpha=args.ema_alpha,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
