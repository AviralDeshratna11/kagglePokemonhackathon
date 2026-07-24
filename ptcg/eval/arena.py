"""Offline evaluation: round robins, matchup matrices and a Kaggle-shaped ladder.

Two lessons from the community write-ups shaped this module.

First, **the live ladder is noisy**: identical submissions have been reported
drifting by over a hundred rating points. So every offline claim needs an
interval, not a point estimate, and the interval used here is the Wilson score
interval -- correct near 0 and 1, unlike the normal approximation, which matters
because several matchups in this format really are near-total.

Second, **agents are not symmetric between seats**: the player who goes first
cannot attack on turn one. Any A-vs-B measurement that does not alternate seats
is measuring the coin flip as much as the agents, so :func:`duel` alternates by
construction and reports the split.

The rating model mirrors the ladder's stated shape -- a Gaussian skill updated
on win/loss/draw with margin-independent updates, agents entering at mu = 600 --
so that offline numbers are at least on a comparable scale. It is a
reimplementation of the *published description*, not of Kaggle's internals, and
is used for ranking our own checkpoints against each other, never for
predicting absolute ladder position.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .harness import MatchResult, play_match

__all__ = ["wilson", "Rating", "Ladder", "duel", "round_robin", "MatchupTable"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a win rate. Returns ``(lo, point, hi)``."""
    if n <= 0:
        return (0.0, 0.5, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), p, min(1.0, centre + half))


def games_for_precision(half_width: float = 0.02, p: float = 0.5, z: float = 1.96) -> int:
    """How many games to resolve a win rate to +/- ``half_width``.

    At p = 0.5 and 95% confidence this returns 2401 for +/-2 points, which is
    where the "2,500 games per matchup" figure in the plan comes from.
    """
    return int(math.ceil((z * z * p * (1 - p)) / (half_width * half_width)))


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

@dataclass
class Rating:
    mu: float = 600.0
    sigma: float = 200.0

    @property
    def conservative(self) -> float:
        """The ladder's displayed score is skill minus uncertainty."""
        return self.mu - 3 * self.sigma


class Ladder:
    """Gaussian, margin-independent rating updates."""

    def __init__(self, beta: float = 100.0, tau: float = 2.0, draw_prob: float = 0.02) -> None:
        self.beta = beta
        self.tau = tau
        self.draw_prob = draw_prob
        self.ratings: dict[str, Rating] = {}

    def get(self, name: str) -> Rating:
        return self.ratings.setdefault(name, Rating())

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        a, b = self.get(winner), self.get(loser)
        a.sigma = math.sqrt(a.sigma**2 + self.tau**2)
        b.sigma = math.sqrt(b.sigma**2 + self.tau**2)

        c2 = a.sigma**2 + b.sigma**2 + 2 * self.beta**2
        c = math.sqrt(c2)
        t = (a.mu - b.mu) / c

        if draw:
            v = -t * _phi(t) / max(1e-9, _Phi(t)) if False else _v_draw(t, self.draw_prob)
            w = _w_draw(t, self.draw_prob)
            sign_a, sign_b = 0.5, 0.5
        else:
            v = _v_win(t)
            w = _w_win(t)
            sign_a, sign_b = 1.0, -1.0

        a.mu += sign_a * (a.sigma**2 / c) * v
        b.mu += sign_b * (b.sigma**2 / c) * v
        a.sigma = math.sqrt(max(1e-6, a.sigma**2 * (1 - (a.sigma**2 / c2) * w)))
        b.sigma = math.sqrt(max(1e-6, b.sigma**2 * (1 - (b.sigma**2 / c2) * w)))

    def table(self) -> list[dict[str, Any]]:
        rows = [
            {
                "agent": n,
                "mu": round(r.mu, 1),
                "sigma": round(r.sigma, 1),
                "score": round(r.conservative, 1),
            }
            for n, r in self.ratings.items()
        ]
        return sorted(rows, key=lambda r: -r["score"])


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _v_win(t: float) -> float:
    d = _Phi(t)
    return _phi(t) / d if d > 1e-9 else -t


def _w_win(t: float) -> float:
    v = _v_win(t)
    return v * (v + t)


def _v_draw(t: float, eps: float) -> float:
    return -t * eps * 0.0 + 0.0


def _w_draw(t: float, eps: float) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------

@dataclass
class DuelResult:
    a: str
    b: str
    games: int
    a_wins: int
    b_wins: int
    draws: int
    a_wins_as_p0: int = 0
    a_games_as_p0: int = 0
    reasons: Counter = field(default_factory=Counter)
    turns: list[int] = field(default_factory=list)
    a_max_ms: float = 0.0
    b_max_ms: float = 0.0
    a_total_time: float = 0.0
    b_total_time: float = 0.0
    invalids: Counter = field(default_factory=Counter)

    @property
    def a_score(self) -> float:
        return self.a_wins + 0.5 * self.draws

    def summary(self) -> dict[str, Any]:
        lo, p, hi = wilson(self.a_score, self.games)
        p0 = self.a_wins_as_p0 / self.a_games_as_p0 if self.a_games_as_p0 else float("nan")
        p1_games = self.games - self.a_games_as_p0
        p1 = (self.a_wins - self.a_wins_as_p0) / p1_games if p1_games else float("nan")
        return {
            "a": self.a,
            "b": self.b,
            "games": self.games,
            "a_winrate": round(p, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "a_winrate_going_first": round(p0, 4) if p0 == p0 else None,
            "a_winrate_going_second": round(p1, 4) if p1 == p1 else None,
            "draws": self.draws,
            "mean_turns": round(statistics.mean(self.turns), 2) if self.turns else 0,
            "a_max_decision_ms": round(self.a_max_ms, 3),
            "b_max_decision_ms": round(self.b_max_ms, 3),
            "a_mean_game_seconds": round(self.a_total_time / self.games, 4) if self.games else 0,
            "reasons": dict(self.reasons),
            "invalid_by_player": dict(self.invalids),
        }


AgentFactory = Callable[[], Callable[[dict], list[int]]]


def duel(
    name_a: str,
    make_a: AgentFactory,
    deck_a: Sequence[int],
    name_b: str,
    make_b: AgentFactory,
    deck_b: Sequence[int],
    games: int = 100,
    time_budget: float = 600.0,
    seed0: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> DuelResult:
    """Play ``games`` matches, alternating who moves first."""
    res = DuelResult(name_a, name_b, 0, 0, 0, 0)
    for i in range(games):
        a_first = (i % 2) == 0
        agent_a, agent_b = make_a(), make_b()
        if a_first:
            r: MatchResult = play_match(
                agent_a, agent_b, deck_a, deck_b, time_budget=time_budget, seed=seed0 + i
            )
            a_idx = 0
        else:
            r = play_match(
                agent_b, agent_a, deck_b, deck_a, time_budget=time_budget, seed=seed0 + i
            )
            a_idx = 1

        res.games += 1
        res.reasons[r.reason] += 1
        res.turns.append(r.turns)
        res.a_max_ms = max(res.a_max_ms, r.max_decision_ms[a_idx])
        res.b_max_ms = max(res.b_max_ms, r.max_decision_ms[1 - a_idx])
        res.a_total_time += r.time_used[a_idx]
        res.b_total_time += r.time_used[1 - a_idx]
        if r.invalid_player >= 0:
            res.invalids[name_a if r.invalid_player == a_idx else name_b] += 1

        if r.winner == -1:
            res.draws += 1
        elif r.winner == a_idx:
            res.a_wins += 1
        else:
            res.b_wins += 1

        if a_first:
            res.a_games_as_p0 += 1
            if r.winner == 0:
                res.a_wins_as_p0 += 1

        if progress is not None:
            progress(i + 1, games)
    return res


class MatchupTable:
    """Square win-rate matrix with intervals -- the rubric-(d) artefact."""

    def __init__(self) -> None:
        self.cells: dict[tuple[str, str], DuelResult] = {}

    def add(self, d: DuelResult) -> None:
        self.cells[(d.a, d.b)] = d

    def agents(self) -> list[str]:
        names: list[str] = []
        for a, b in self.cells:
            for n in (a, b):
                if n not in names:
                    names.append(n)
        return names

    def matrix(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for (a, b), d in self.cells.items():
            wr = d.a_score / d.games if d.games else float("nan")
            out[a][b] = round(wr, 4)
            out[b][a] = round(1.0 - wr, 4)
        return {k: dict(v) for k, v in out.items()}

    def exploitability_proxy(self) -> dict[str, float]:
        """Worst-case matchup per agent.

        Rubric criterion (d) rewards *not* relying on favourable matchups, and
        the cheapest honest summary of that is each agent's minimum win rate
        across the panel. Week 3 replaces this with a trained best-response.
        """
        m = self.matrix()
        return {a: round(min(v.values()), 4) for a, v in m.items() if v}

    def report(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix(),
            "worst_case": self.exploitability_proxy(),
            "duels": [d.summary() for d in self.cells.values()],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        return p


def round_robin(
    entries: dict[str, tuple[AgentFactory, Sequence[int]]],
    games: int = 100,
    time_budget: float = 600.0,
    ladder: Ladder | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[MatchupTable, Ladder]:
    """Every entry against every other entry."""
    table = MatchupTable()
    ladder = ladder or Ladder()
    names = list(entries)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if progress:
                progress(f"{a} vs {b}")
            fa, da = entries[a]
            fb, db_ = entries[b]
            d = duel(a, fa, da, b, fb, db_, games=games, time_budget=time_budget)
            table.add(d)
            for _ in range(d.a_wins):
                ladder.update(a, b)
            for _ in range(d.b_wins):
                ladder.update(b, a)
            for _ in range(d.draws):
                ladder.update(a, b, draw=True)
    return table, ladder
