"""Direct match loop against ``libcg``.

``kaggle_environments.make("cabt").run(...)`` works and is what the ladder uses,
but it wraps every decision in the generic Kaggle agent machinery: process
isolation, JSON round-trips, step bookkeeping and a full visualiser dump at the
end. For offline evaluation -- where Week 4 needs thousands of games per
matchup to get Wilson intervals down to a couple of points -- that overhead
dominates.

This harness drives ``battle_start`` / ``battle_select`` / ``battle_finish``
directly. It is roughly an order of magnitude faster, and more importantly it
gives us three things the generic runner cannot:

* an honest simulation of the cumulative 600 s clock, including forfeits;
* per-decision timing, so latency claims are measured rather than asserted;
* a place to hang the trace recorder.

One caveat is baked into the design: ``Battle.battle_ptr`` is a module-level
global in the engine bindings, so exactly one battle can be live per process.
Parallelism is therefore process-level (see ``run_many``), never threads.
"""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.engine import bootstrap_paths, get_engine

__all__ = ["MatchResult", "play_match", "TimeoutForfeit"]


class TimeoutForfeit(Exception):
    """A player exhausted the cumulative clock."""


@dataclass
class MatchResult:
    winner: int          # 0, 1, or -1 for a draw
    reason: str
    turns: int
    decisions: tuple[int, int]
    time_used: tuple[float, float]
    max_decision_ms: tuple[float, float]
    error: str = ""
    invalid_player: int = -1
    logs_tail: list[dict[str, Any]] = field(default_factory=list)

    @property
    def draw(self) -> bool:
        return self.winner == -1

    def score_for(self, player: int) -> float:
        if self.winner == -1:
            return 0.5
        return 1.0 if self.winner == player else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "turns": self.turns,
            "decisions": list(self.decisions),
            "time_used": [round(t, 3) for t in self.time_used],
            "max_decision_ms": [round(t, 3) for t in self.max_decision_ms],
            "error": self.error,
            "invalid_player": self.invalid_player,
        }


def _import_game():
    """Import the engine's game API, having already ensured single init."""
    bootstrap_paths()
    get_engine()  # guarantees GameInitialize ran exactly once
    try:
        from cg import game as g  # type: ignore
    except Exception:  # noqa: BLE001
        from kaggle_environments.envs.cabt.cg import game as g  # type: ignore
    return g


def play_match(
    agent0: Callable[[dict], list[int]],
    agent1: Callable[[dict], list[int]],
    deck0: Sequence[int],
    deck1: Sequence[int],
    time_budget: float = 600.0,
    max_steps: int = 20000,
    seed: int | None = None,
    on_step: Callable[[int, dict, list[int]], None] | None = None,
) -> MatchResult:
    """Play one complete game and return the outcome.

    ``seed`` seeds Python's RNG, which is what any stochastic policy should be
    drawing from. The C++ engine shuffles internally with its own generator, so
    a seed makes *our* behaviour reproducible but not the deal -- which is the
    honest situation and the reason Week 4's protocol relies on large paired
    samples rather than on seed matching.
    """
    g = _import_game()
    if seed is not None:
        random.seed(seed)

    agents = (agent0, agent1)
    used = [0.0, 0.0]
    decisions = [0, 0]
    max_ms = [0.0, 0.0]

    obs, start = g.battle_start(list(deck0), list(deck1))
    if obs is None:
        bad = start.errorPlayer if start.errorPlayer >= 0 else 0
        return MatchResult(
            winner=1 - bad,
            reason="illegal_deck",
            turns=0,
            decisions=(0, 0),
            time_used=(0.0, 0.0),
            max_decision_ms=(0.0, 0.0),
            error=f"battle_start rejected player {bad} (errorType={start.errorType})",
            invalid_player=bad,
        )

    winner, reason, err, invalid = -1, "unknown", "", -1
    turns = 0
    try:
        for _ in range(max_steps):
            state = obs.get("current") or {}
            turns = int(state.get("turn", turns) or turns)
            result = int(state.get("result", -1))
            if result >= 0:
                winner = -1 if result == 2 else result
                reason = "engine_result"
                break

            p = int(state.get("yourIndex", 0))

            # Present the observation the way the ladder does, including the
            # live clock, so agents can budget identically offline and online.
            view = dict(obs)
            view["remainingOverageTime"] = max(0.0, time_budget - used[p])

            t0 = time.monotonic()
            try:
                action = agents[p](view)
            except BaseException as exc:  # noqa: BLE001
                winner, reason, err, invalid = 1 - p, "agent_exception", f"{type(exc).__name__}: {exc}", p
                break
            dt = time.monotonic() - t0

            used[p] += dt
            decisions[p] += 1
            max_ms[p] = max(max_ms[p], dt * 1000)

            if used[p] >= time_budget:
                winner, reason, invalid = 1 - p, "timeout", p
                break

            if on_step is not None:
                try:
                    on_step(p, view, action)
                except BaseException:  # noqa: BLE001
                    pass

            try:
                obs = g.battle_select(list(action))
            except BaseException as exc:  # noqa: BLE001
                winner, reason, err, invalid = 1 - p, "illegal_action", f"{type(exc).__name__}: {exc}", p
                break
        else:
            reason = "max_steps"
    except BaseException as exc:  # noqa: BLE001 - engine crash containment
        winner, reason, err = -1, "engine_crash", traceback.format_exc(limit=3)
        _ = exc
    finally:
        try:
            g.battle_finish()
        except BaseException:  # noqa: BLE001
            pass
        try:
            from cg.sim import Battle  # type: ignore
        except Exception:  # noqa: BLE001
            try:
                from kaggle_environments.envs.cabt.cg.sim import Battle  # type: ignore
            except Exception:  # noqa: BLE001
                Battle = None  # type: ignore
        if Battle is not None:
            Battle.battle_ptr = None

    return MatchResult(
        winner=winner,
        reason=reason,
        turns=turns,
        decisions=(decisions[0], decisions[1]),
        time_used=(used[0], used[1]),
        max_decision_ms=(max_ms[0], max_ms[1]),
        error=err,
        invalid_player=invalid,
    )
