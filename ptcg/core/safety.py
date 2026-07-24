"""The shell that stands between any policy and the engine.

Week-0's hard requirement is *zero invalid submissions*. On this ladder a crash
is not a lost game, it is a lost submission: ``cabt.py``'s interpreter marks the
player ``INVALID`` and awards the opponent the win, and because Kaggle validates
every upload by playing the agent against a copy of itself, a single unhandled
exception invalidates the whole submission.

So the shell is written to the standard that ``act()`` **cannot raise and cannot
return an illegal action**, no matter what the policy does. Four independent
layers:

1. **Structural validation** of the incoming observation. A malformed or empty
   ``select`` short-circuits to the trivial legal answer.
2. **Action masking.** Selections are only ever produced by indexing into
   ``obs["select"]["option"]``; indices are then clipped, de-duplicated and
   padded to satisfy ``minCount``/``maxCount``. There is no code path that can
   emit an index the engine did not offer.
3. **A loop guard.** The one non-crash way to lose is the clock, and the way an
   agent burns the clock is by re-triggering a free repeatable Ability forever.
   Repeated identical decisions inside a single turn get masked out, with
   ``END`` as the guaranteed escape hatch.
4. **Layered fallbacks.** Policy exception, deadline overrun or budget panic all
   degrade to a rule-based fallback, and that in turn degrades to
   ``list(range(minCount))`` -- which is always legal by construction.

Every rescue is counted, not swallowed silently: :meth:`SafetyShell.report`
returns the incident tally that Week 4's reliability figure is built from.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .actions import ActionCandidate, build_candidates
from .carddb import CardDB
from .clock import BudgetManager, Deadline
from .enums import OptionType
from .obs import ObsView

__all__ = ["SafetyShell", "SafetyReport", "sanitize_selection"]


# ---------------------------------------------------------------------------
# Selection sanitising
# ---------------------------------------------------------------------------

def sanitize_selection(
    selection: Any,
    n_options: int,
    min_count: int,
    max_count: int,
) -> list[int]:
    """Coerce anything into a selection the engine will accept.

    ``battle_select`` raises ``ValueError`` unless it receives ``list[int]``,
    and ``IndexError`` for out-of-range indices; both are fatal to the
    submission. This function is total: for any input it returns a list of
    valid indices whose length lies in ``[min_count, max_count]``.
    """
    lo = max(0, int(min_count or 0))
    hi = int(max_count or 0)
    if hi < lo:
        hi = lo
    if n_options <= 0:
        return []
    lo = min(lo, n_options if hi <= n_options else lo)

    # 1. coerce to a list of ints
    raw: list[int] = []
    if isinstance(selection, (list, tuple)):
        for x in selection:
            try:
                raw.append(int(x))
            except (TypeError, ValueError):
                continue
    elif selection is not None:
        try:
            raw.append(int(selection))
        except (TypeError, ValueError):
            pass

    # 2. keep only offered indices
    raw = [i for i in raw if 0 <= i < n_options]

    # 3. de-duplicate unless the engine is asking for more picks than there are
    #    distinct options (damage-counter placement legitimately repeats).
    allow_repeats = lo > n_options
    if not allow_repeats:
        seen: set[int] = set()
        deduped: list[int] = []
        for i in raw:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        raw = deduped

    # 4. clip to the maximum
    if hi and len(raw) > hi:
        raw = raw[:hi]

    # 5. pad to the minimum, preferring unused options
    if len(raw) < lo:
        for i in range(n_options):
            if len(raw) >= lo:
                break
            if i not in raw:
                raw.append(i)
    while len(raw) < lo:  # only reachable when lo > n_options
        raw.append(0)

    return raw


# ---------------------------------------------------------------------------
# Incident reporting
# ---------------------------------------------------------------------------

@dataclass
class SafetyReport:
    decisions: int = 0
    policy_exceptions: int = 0
    fallback_exceptions: int = 0
    deadline_overruns: int = 0
    panic_moves: int = 0
    loop_breaks: int = 0
    sanitised: int = 0
    empty_selects: int = 0
    max_decision_ms: float = 0.0
    last_error: str = ""
    context_counts: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "context_counts"}
        d["context_counts"] = dict(self.context_counts)
        d["clean"] = (
            self.policy_exceptions == 0
            and self.fallback_exceptions == 0
            and self.sanitised == 0
        )
        return d


# ---------------------------------------------------------------------------
# Loop guard
# ---------------------------------------------------------------------------

class LoopGuard:
    """Detects a decision loop and masks the offending action.

    Repeatable abilities (Iono's Bellibolt's *Electric Streamer* is the canonical
    example: "as often as you like during your turn") are re-offered by the
    engine immediately after use. A greedy scorer that always ranks the ability
    first will select it until the resource runs out -- which is correct -- but a
    scoring bug, or an ability whose precondition the scorer misreads, produces
    an unbounded loop that silently drains the 600 s clock.
    """

    def __init__(self, max_repeats: int = 12, window: int = 512) -> None:
        self.max_repeats = max_repeats
        self.window: deque[tuple[int, int, tuple]] = deque(maxlen=window)
        self._turn = -1

    def note(self, turn: int, context: int, signature: tuple) -> None:
        if turn != self._turn:
            self.window.clear()
            self._turn = turn
        self.window.append((turn, context, signature))

    def banned(self, turn: int) -> set[tuple]:
        if turn != self._turn:
            return set()
        counts: Counter = Counter(sig for t, _c, sig in self.window if t == turn)
        return {sig for sig, n in counts.items() if n >= self.max_repeats}


# ---------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------

class SafetyShell:
    """Wrap a policy so it is safe to hand to Kaggle."""

    def __init__(
        self,
        policy,
        db: CardDB,
        fallback=None,
        budget: BudgetManager | None = None,
        recorder: Callable[[dict[str, Any]], None] | None = None,
        max_repeats: int = 12,
        debug: bool = False,
    ) -> None:
        self.policy = policy
        self.fallback = fallback
        self.db = db
        self.budget = budget or BudgetManager()
        self.recorder = recorder
        self.guard = LoopGuard(max_repeats=max_repeats)
        self.report = SafetyReport()
        self.debug = debug

    # -- entry point --------------------------------------------------------

    def act(self, obs: dict[str, Any]) -> list[int]:
        """The function handed to kaggle-environments. Never raises."""
        try:
            return self._act(obs)
        except BaseException as exc:  # noqa: BLE001 - last line of defence
            self.report.fallback_exceptions += 1
            self.report.last_error = f"{type(exc).__name__}: {exc}"
            if self.debug:
                traceback.print_exc()
            return self._structural_minimum(obs)

    # -- internals ----------------------------------------------------------

    def _structural_minimum(self, obs: Any) -> list[int]:
        """The answer that is legal for *any* observation."""
        try:
            sel = (obs or {}).get("select")
            if not sel:
                return list(self.policy.deck())
            n = len(sel.get("option") or ())
            return sanitize_selection([], n, sel.get("minCount", 0), sel.get("maxCount", 0))
        except BaseException:  # noqa: BLE001
            return []

    def _act(self, obs: dict[str, Any]) -> list[int]:
        if not isinstance(obs, dict):
            return self._structural_minimum(obs)

        self.budget.sync(obs.get("remainingOverageTime"))

        view = ObsView(obs, self.db)

        # Deck-selection phase: `select` is None and the engine wants 60 IDs.
        if view.is_deck_selection:
            deck = list(self.policy.deck())
            if len(deck) != 60:
                self.report.sanitised += 1
                deck = (deck + deck)[:60] if deck else []
            return deck

        options = view.options
        n = len(options)
        if n == 0:
            self.report.empty_selects += 1
            return sanitize_selection([], 0, view.min_count, view.max_count)

        self.report.decisions += 1
        self.report.context_counts[int(view.context)] += 1

        deadline = self.budget.allot(turn=view.turn, importance=self._importance(view, n))
        t0 = time.monotonic()

        candidates = build_candidates(view, self.db)

        # -- score ----------------------------------------------------------
        scores: Sequence[float]
        used_fallback = False
        if self.budget.panic_level >= 2 and self.fallback is not None:
            self.report.panic_moves += 1
            used_fallback = True
            scores = self.fallback.score(view, candidates, deadline)
        else:
            try:
                scores = self.policy.score(view, candidates, deadline)
                if len(scores) != n:
                    raise ValueError(f"policy returned {len(scores)} scores for {n} options")
            except BaseException as exc:  # noqa: BLE001
                self.report.policy_exceptions += 1
                self.report.last_error = f"policy {type(exc).__name__}: {exc}"
                if self.debug:
                    traceback.print_exc()
                used_fallback = True
                if self.fallback is not None:
                    try:
                        scores = self.fallback.score(view, candidates, deadline)
                    except BaseException:  # noqa: BLE001
                        self.report.fallback_exceptions += 1
                        scores = [0.0] * n
                else:
                    scores = [0.0] * n

        scores = list(scores)

        # -- loop guard -----------------------------------------------------
        banned = self.guard.banned(view.turn)
        if banned:
            escape = None
            for c in candidates:
                if c.option_type == OptionType.END:
                    escape = c.index
                if c.signature() in banned:
                    scores[c.index] = -1e12
            if max(scores) <= -1e11:
                # Everything is banned: force the escape hatch, else index 0.
                self.report.loop_breaks += 1
                idx = escape if escape is not None else 0
                scores[idx] = 1.0
            elif escape is not None:
                self.report.loop_breaks += 1

        # -- choose ---------------------------------------------------------
        try:
            k = int(self.policy.desired_count(view, candidates, scores))
        except BaseException:  # noqa: BLE001
            k = view.min_count
        k = max(view.min_count, min(view.max_count, k))

        order = sorted(range(n), key=lambda i: (-scores[i], i))
        selection = order[:k] if k > 0 else []

        clean = sanitize_selection(selection, n, view.min_count, view.max_count)
        if clean != selection:
            self.report.sanitised += 1

        for i in clean:
            self.guard.note(view.turn, view.context, candidates[i].signature())

        # -- accounting -----------------------------------------------------
        elapsed = self.budget.close(deadline)
        self.report.max_decision_ms = max(self.report.max_decision_ms, elapsed * 1000)
        if deadline.expired():
            self.report.deadline_overruns += 1

        if self.recorder is not None:
            try:
                self.recorder(
                    {
                        "t": time.time(),
                        "turn": view.turn,
                        "select_type": view.select_type,
                        "context": view.context,
                        "n_options": n,
                        "chosen": clean,
                        "scores": [round(float(s), 4) for s in scores],
                        "features": [c.features for c in candidates],
                        "signatures": [list(map(_jsonable, c.signature())) for c in candidates],
                        "prize_diff": view.prize_diff,
                        "elapsed_ms": round(elapsed * 1000, 3),
                        "used_fallback": used_fallback,
                        "search_begin_input": view.search_begin_input,
                    }
                )
            except BaseException:  # noqa: BLE001 - telemetry must never break play
                pass

        _ = t0
        return clean

    @staticmethod
    def _importance(view: ObsView, n_options: int) -> float:
        """Ask for more clock on decisions that plausibly matter more.

        This is the seam the Week-3 entropy gate plugs into: today it is a
        cheap proxy (main-phase decisions with many branches, and decisions
        taken while the prize race is close), later it becomes the policy's own
        top-1/top-2 margin.
        """
        if n_options <= 2:
            return 0.35
        imp = 1.0
        if view.select_type == 0:  # MAIN
            imp *= 1.6
        if abs(view.prize_diff) <= 1:
            imp *= 1.2
        if n_options >= 12:
            imp *= 1.3
        return imp


def _jsonable(x: Any) -> Any:
    return x if x is None or isinstance(x, (int, float, str, bool)) else str(x)
