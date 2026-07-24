"""Generate a BC training corpus via real self-play through the real engine.

Week 1 needs an imitation corpus. The official daily Kaggle episode replays
are the intended primary source (see ``pull_episodes.py``), but this machine
has no Kaggle API credentials configured, so this script produces a genuine
substitute: ``HeuristicPolicy`` vs itself, driven through the actual ``libcg``
engine via ``ptcg.eval.harness.play_match`` (not mocked, not synthetic).
``trace.py``'s own docstring anticipated exactly this gap -- self-play and
real replay data are meant to share one schema so they can be mixed later.

Two writers per game (one per side), each getting only that side's decisions
plus that side's own win/loss/draw outcome -- the same one-recorder-per-agent
pattern ``main.py`` uses for ``PTCG_TRACE_DIR``.

Records are richer than ``SafetyShell``'s built-in recorder: they also carry
enough board state (card IDs per zone, log tail, globals) to reconstruct a
:class:`~ptcg.agents.network.ModelInput` at training time, plus a belief
label for the opponent-belief head.

Belief label. The opponent's hand is legitimately hidden from the mover's own
observation -- that is the whole point of the game. This script tracks, per
player, the most recently *fully observed* true hand (available every time
that player was themselves the mover) and uses the opponent's most recent
snapshot as the training target when the other player decides. That target is
exact for the duration of the opponent's last turn and only possibly stale
during the current player's own turn (e.g. an ability that secretly modifies
the opponent's hand mid-turn) -- an acceptable approximation for a bonus
auxiliary head, not used at inference time.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.actions import build_candidates
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.obs import ObsView
from ..core.safety import SafetyShell
from ..core.trace import TraceWriter
from ..decks.registry import available_decks, load_deck
from ..eval.harness import play_match

HIST_LEN = 16


class _GameRecorder:
    """One instance per game; feeds two per-side :class:`TraceWriter`s from a
    single ``on_step`` callback."""

    def __init__(self, out_dir: Path, tag: str, db) -> None:
        self.db = db
        self.writers = [
            TraceWriter(out_dir, tag=f"{tag}-p0"),
            TraceWriter(out_dir, tag=f"{tag}-p1"),
        ]
        self.last_hand: list[list[int]] = [[], []]  # true card ids, per player
        self.last_hand_seen: list[bool] = [False, False]
        self.n_records = [0, 0]

    def on_step(self, player: int, obs: dict[str, Any], action: list[int]) -> None:
        view = ObsView(obs, self.db)
        if view.is_deck_selection:
            return

        me = view.me
        opp = view.opp

        my_hand_ids = [c.card_id if c else None for c in me.hand_cards]
        self.last_hand[player] = [cid for cid in my_hand_ids if cid is not None]
        self.last_hand_seen[player] = True

        options = view.options
        n = len(options)
        if n == 0:
            return
        candidates = build_candidates(view, self.db)

        def zone_ids(cards) -> list[int | None]:
            return [c.card_id if c is not None else None for c in cards]

        board = {
            "my_active": zone_ids([me.active.card] if me.active else []),
            "my_bench": zone_ids([p.card for p in me.bench]),
            "my_hand": zone_ids([c for c in me.hand_cards]),
            "my_discard": [d.get("id") for d in me.discard],
            "my_prize": [p.get("id") if p else None for p in me.prize],
            "opp_active": zone_ids([opp.active.card] if opp.active else []),
            "opp_bench": zone_ids([p.card for p in opp.bench]),
            "opp_discard": [d.get("id") for d in opp.discard],
            "opp_prize": [p.get("id") if p else None for p in opp.prize],
            "opp_hand_count": opp.hand_count,
            "opp_deck_count": opp.deck_count,
            "opp_prizes_remaining": opp.prizes_remaining,
            "my_deck_count": me.deck_count,
        }
        globals_ = {
            "turn": view.turn,
            "prize_diff": view.prize_diff,
            "remaining_time": view.remaining_time,
            "supporter_played": view.supporter_played,
            "stadium_played": view.stadium_played,
            "energy_attached": view.energy_attached,
            "i_am_first": view.i_am_first,
        }

        opp_idx = 1 - player
        belief_valid = self.last_hand_seen[opp_idx]
        belief_hand_ids = list(self.last_hand[opp_idx]) if belief_valid else []

        record = {
            "player": player,
            "turn": view.turn,
            "select_type": view.select_type,
            "context": view.context,
            "n_options": n,
            "chosen": list(action),
            "candidate_card_ids": [c.card_id for c in candidates],
            "features": [c.features for c in candidates],
            "board": board,
            "globals": globals_,
            "logs_tail": list(view.logs[-HIST_LEN:]),
            "belief_valid": belief_valid,
            "belief_hand_card_ids": belief_hand_ids,
        }
        self.writers[player](record)
        self.n_records[player] += 1

    def finish(self, winner: int) -> None:
        for p, w in enumerate(self.writers):
            outcome = 0.5 if winner == -1 else (1.0 if winner == p else 0.0)
            w.finish({"winner": winner, "player": p, "outcome": outcome, "n_records": self.n_records[p]})


def generate(
    out_dir: Path,
    n_games: int,
    decks: list[str],
    seed_start: int = 0,
    log_every: int = 100,
) -> dict[str, Any]:
    db = get_card_db()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_records = 0
    outcomes: dict[str, int] = {"p0_win": 0, "p1_win": 0, "draw": 0}
    reasons: dict[str, int] = {}

    rng = random.Random(seed_start)
    for i in range(n_games):
        deck_name = decks[i % len(decks)]
        deck = load_deck(deck_name)
        cfg_a = HeuristicConfig()
        cfg_b = HeuristicConfig()
        # A little config jitter so the corpus isn't one deterministic policy
        # replayed with only engine-shuffle randomness.
        if rng.random() < 0.3:
            cfg_a.prefer_first = not cfg_a.prefer_first
        if rng.random() < 0.3:
            cfg_b.prefer_first = not cfg_b.prefer_first

        shell_a = SafetyShell(HeuristicPolicy(deck, db, cfg_a), db, FallbackPolicy(deck), BudgetManager())
        shell_b = SafetyShell(HeuristicPolicy(deck, db, cfg_b), db, FallbackPolicy(deck), BudgetManager())

        rec = _GameRecorder(out_dir, tag=f"selfplay-{deck_name}", db=db)
        seed = seed_start + i
        result = play_match(shell_a.act, shell_b.act, deck, deck, seed=seed, on_step=rec.on_step)
        rec.finish(result.winner)

        total_records += sum(rec.n_records)
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        if result.winner == 0:
            outcomes["p0_win"] += 1
        elif result.winner == 1:
            outcomes["p1_win"] += 1
        else:
            outcomes["draw"] += 1

        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{n_games} games, {total_records} records, {dt:.1f}s ({(i + 1) / dt:.1f} games/s)")

    return {
        "games": n_games,
        "decks": decks,
        "total_records": total_records,
        "seconds": round(time.time() - t0, 2),
        "outcomes": outcomes,
        "reasons": reasons,
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--decks", default=",".join(available_decks()))
    ap.add_argument("--out", default="artifacts/traces/selfplay")
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args(argv)

    decks = [d.strip() for d in args.decks.split(",") if d.strip()]
    info = generate(Path(args.out), args.games, decks, seed_start=args.seed_start)
    import json

    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
