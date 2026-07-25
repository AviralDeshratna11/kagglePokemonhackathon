"""Week-2 iteration loop: generate self-play games via PFSP opponent
sampling -> train a few epochs (policy gradient + value + frozen-teacher KL +
belief) -> update the league's win-rate table -> checkpoint -> repeat, for a
bounded number of rounds. Default 3 rounds x 200 games -- a quick pass sized
to finish in a session and show a real trend, not to fully converge (see the
Week 2 plan's "explicitly deferred" section for what a full run would need).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..agents.network import BeliefSetDMCLite, CardTable
from ..core.carddb import get_card_db
from ..decks.registry import load_deck
from ..tools.generate_league_selfplay import generate_round
from .dataset import load_records
from .league import League
from .selfplay_rl import train_selfplay_round

__all__ = ["run"]


def run(
    deck_name: str = "lucario_fighting",
    frozen_teacher_ckpt: str = "artifacts/bc_text.pt",
    out_prefix: str = "artifacts/selfplay_v2",
    rounds: int = 3,
    games_per_round: int = 200,
    epochs_per_round: int = 2,
    batch_size: int = 64,
    lr: float = 5e-4,
    temperature: float = 1.0,
    replay_window: int = 2,
    max_records_per_round: int = 40000,
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

    league = League(db, deck, frozen_teacher_ckpt)

    round_dirs: list[Path] = []
    history: list[dict[str, Any]] = []
    ckpt_path: Path | None = None

    for r in range(rounds):
        print(f"=== round {r}/{rounds} ===")
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

        worst = league.worst_case()
        history.append(
            {
                "round": r,
                "generation": gen_info,
                "training": train_info,
                "win_rate": dict(league.win_rate),
                "worst_case": worst,
                "checkpoint": str(ckpt_path),
            }
        )
        print(f"  win rates vs league: {league.win_rate}")
        print(f"  worst-case matchup: {worst}")

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
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--replay-window", type=int, default=2)
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
        temperature=args.temperature,
        replay_window=args.replay_window,
        seed=args.seed,
    )
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
