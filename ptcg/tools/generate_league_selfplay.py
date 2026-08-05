"""Week-2 self-play generation: the current trainee (a stochastic
:class:`~ptcg.agents.network_policy.NetworkPolicy`) against a PFSP-sampled
league opponent, through the real engine.

Only the trainee's decisions are recorded -- the opponents are fixed
(baselines, the frozen Week-1 teacher, or older frozen self-play checkpoints),
so there is nothing to learn from recording their moves. The record schema is
identical to ``ptcg/tools/generate_selfplay.py``'s (board/globals/features
/candidate_card_ids/chosen/logs_tail/belief_*), so ``ptcg/train/dataset.py``'s
loader applies completely unchanged.

Belief labels use the same mechanism as Week 1: the generator observes both
players' own-turn observations (each player's hand is fully visible to
itself), so it can log the opponent's true hand as of their last visible turn
-- a training target only, never a model input.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from ..agents.fallback import FallbackPolicy
from ..agents.network import BeliefSetDMCLite, CardTable
from ..agents.network_policy import NetworkPolicy
from ..core.actions import build_candidates
from ..core.carddb import CardDB
from ..core.clock import BudgetManager
from ..core.obs import ObsView
from ..core.safety import SafetyShell
from ..core.trace import TraceWriter
from ..eval.harness import play_match
from ..train.league import League

__all__ = ["generate_round"]

HIST_LEN = 16


class _TraineeRecorder:
    """Tracks true hands for *both* players (needed for belief labels) but
    only writes decision records for the trainee's seat."""

    def __init__(self, out_dir: Path, tag: str, db: CardDB, trainee_player: int) -> None:
        self.db = db
        self.trainee_player = trainee_player
        self.writer = TraceWriter(out_dir, tag=tag)
        self.n_records = 0
        self.last_hand: list[list[int]] = [[], []]
        self.last_hand_seen: list[bool] = [False, False]

    def on_step(self, player: int, obs: dict[str, Any], action: list[int]) -> None:
        view = ObsView(obs, self.db)
        if view.is_deck_selection:
            return

        me = view.me
        my_hand_ids = [c.card_id for c in me.hand_cards if c is not None]
        self.last_hand[player] = my_hand_ids
        self.last_hand_seen[player] = True

        if player != self.trainee_player:
            return

        opp = view.opp
        options = view.options
        n = len(options)
        if n == 0:
            return
        candidates = build_candidates(view, self.db)

        def zone_ids(cards):
            return [c.card_id if c is not None else None for c in cards]

        board = {
            "my_active": zone_ids([me.active.card] if me.active else []),
            "my_bench": zone_ids([p.card for p in me.bench]),
            "my_hand": zone_ids(list(me.hand_cards)),
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

        self.writer(
            {
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
                "belief_hand_card_ids": list(self.last_hand[opp_idx]) if belief_valid else [],
            }
        )
        self.n_records += 1

    def finish(self, outcome: float) -> None:
        self.writer.finish({"outcome": outcome, "n_records": self.n_records})


def generate_round(
    league: League,
    model: BeliefSetDMCLite,
    table: CardTable,
    card_mode: str,
    out_dir: Path,
    n_games: int,
    temperature: float = 1.0,
    seed_start: int = 0,
    round_tag: str = "round",
    log_every: int = 100,
) -> dict[str, Any]:
    """Play ``n_games``, each trainee-vs-PFSP-sampled-opponent, alternating
    seats for fairness. Updates ``league``'s win-rate table and returns a
    summary including per-opponent outcome counts."""
    db = league.db
    deck = league.deck
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed_start)

    t0 = time.time()
    total_records = 0
    opponent_counts: dict[str, int] = {}
    opponent_results: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}

    for i in range(n_games):
        opponent_name = league.sample_opponent(rng)
        opponent_deck = league.member_deck(opponent_name)
        opponent_policy = league.make_opponent(opponent_name)

        trainee_policy = NetworkPolicy(deck, model, table, card_mode, temperature=temperature, seed=seed_start + i)

        trainee_player = i % 2  # alternate seats
        shells = [None, None]
        decks = [None, None]
        shells[trainee_player] = SafetyShell(trainee_policy, db, FallbackPolicy(deck), BudgetManager())
        decks[trainee_player] = deck
        shells[1 - trainee_player] = SafetyShell(opponent_policy, db, FallbackPolicy(opponent_deck), BudgetManager())
        decks[1 - trainee_player] = opponent_deck

        rec = _TraineeRecorder(out_dir, tag=f"{round_tag}-vs-{opponent_name}", db=db, trainee_player=trainee_player)
        result = play_match(
            shells[0].act, shells[1].act, decks[0], decks[1], seed=seed_start + i, on_step=rec.on_step
        )

        outcome = 0.5 if result.winner == -1 else (1.0 if result.winner == trainee_player else 0.0)
        rec.finish(outcome)
        league.update_win_rate(opponent_name, outcome)

        total_records += rec.n_records
        opponent_counts[opponent_name] = opponent_counts.get(opponent_name, 0) + 1
        oc = opponent_results.setdefault(opponent_name, {"win": 0, "loss": 0, "draw": 0})
        oc["win" if outcome == 1.0 else "draw" if outcome == 0.5 else "loss"] += 1
        reasons[result.reason] = reasons.get(result.reason, 0) + 1

        if (i + 1) % log_every == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/{n_games} games, {total_records} records, {dt:.1f}s")

    return {
        "games": n_games,
        "total_records": total_records,
        "seconds": round(time.time() - t0, 2),
        "opponent_counts": opponent_counts,
        "opponent_results": opponent_results,
        "reasons": reasons,
        "out_dir": str(out_dir),
    }
