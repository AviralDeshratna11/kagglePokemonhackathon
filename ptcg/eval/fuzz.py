"""Adversarial fuzzing of the agent and the engine.

Week 0's acceptance criterion is "zero invalid submissions", and the only way to
have any confidence in that is to attack the agent deliberately rather than
watch it play nicely against itself.

Three distinct things get fuzzed:

**The observation contract.** :func:`fuzz_observations` feeds the shell garbage
-- ``None``, empty dicts, negative counts, ``minCount`` greater than the number
of options, options with missing or nonsensical fields, deeply wrong types --
and asserts that it still returns a list of valid indices of legal length. This
is where a Kaggle-side schema change would show up as a caught failure instead
of an invalidated submission.

**Live legality.** :func:`fuzz_selfplay` plays games with a deliberately chaotic
policy and verifies every action against the engine's own option list before it
is submitted, so an illegal index is caught here rather than as an
``IndexError`` from ``battle_select`` on the ladder.

**Engine stability.** Random decks and random play explore effect interactions
that curated decks never reach. The blueprint flags one such interaction as a
reproducible segfault; since a native crash takes the whole process down, this
runs each batch in a child process so a crash is *recorded* rather than fatal.

**The network path, specifically** (Week 3). Week 0 fuzzed only
``HeuristicPolicy``; the network policies (``BCPolicy`` and anything built on
``BeliefSetDMCLite``) parse a materially different, more structured slice of
the observation (``encode_view``'s zone extraction) that synthetic garbage
observations don't always exercise the same way. :func:`fuzz_selfplay_checkpoint`
plays real (if randomized) games with a checkpoint-loaded policy on one seat,
in the same child-process containment as :func:`fuzz_selfplay` -- the
checkpoint is reconstructed *inside* the worker from a plain file path rather
than passed as a live object, because Windows' ``spawn`` multiprocessing
context cannot pickle a closure over a loaded ``torch.nn.Module``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.carddb import CardDB, get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell, sanitize_selection
from ..agents.fallback import FallbackPolicy

__all__ = [
    "fuzz_observations", "fuzz_selfplay", "fuzz_selfplay_checkpoint",
    "random_legal_deck", "FuzzReport",
]


@dataclass
class FuzzReport:
    cases: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    crashes: list[dict[str, Any]] = field(default_factory=list)
    illegal_actions: int = 0
    ok: bool = True

    def fail(self, **info: Any) -> None:
        self.ok = False
        self.failures.append(info)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "ok": self.ok,
            "illegal_actions": self.illegal_actions,
            "n_failures": len(self.failures),
            "failures": self.failures[:25],
            "crashes": self.crashes[:10],
        }


# ---------------------------------------------------------------------------
# 1. Observation fuzzing
# ---------------------------------------------------------------------------

_HOSTILE_SCALARS = [None, 0, -1, 1, 10**9, "x", [], {}, 1.5, True, float("nan")]


def _random_option(rng: random.Random, db: CardDB) -> dict[str, Any]:
    opt: dict[str, Any] = {"type": rng.choice([*range(-2, 20)])}
    for key in ("area", "index", "playerIndex", "inPlayArea", "inPlayIndex",
                "toolIndex", "energyIndex", "count", "number", "attackId",
                "cardId", "serial", "specialConditionType"):
        if rng.random() < 0.45:
            opt[key] = rng.choice(_HOSTILE_SCALARS + [rng.randint(-3, 40)])
    return opt


def _random_observation(rng: random.Random, db: CardDB) -> Any:
    if rng.random() < 0.06:
        return rng.choice([None, [], "nonsense", 42, {"select": "bad"}])

    n = rng.randint(0, 12)
    options = [_random_option(rng, db) for _ in range(n)]
    lo = rng.choice([0, 0, 1, 1, 2, rng.randint(0, 6)])
    hi = rng.choice([lo, lo + rng.randint(0, 4), rng.randint(0, 8)])

    player = {
        "active": rng.choice([[], [None], [{"id": rng.randint(1, 1300), "serial": 1,
                                            "hp": rng.randint(-50, 400), "maxHp": rng.randint(0, 400),
                                            "energies": [rng.randint(0, 11) for _ in range(rng.randint(0, 6))],
                                            "tools": [], "preEvolution": []}]]),
        "bench": [],
        "benchMax": rng.choice([0, 5, -1]),
        "hand": rng.choice([None, [], [{"id": rng.randint(1, 1300)} for _ in range(rng.randint(0, 9))]]),
        "handCount": rng.randint(-2, 12),
        "deckCount": rng.randint(-2, 60),
        "discard": [],
        "prize": [None] * rng.randint(0, 6),
        "poisoned": rng.choice([True, False, None]),
        "burned": False, "asleep": False, "paralyzed": False, "confused": False,
    }
    return {
        "remainingOverageTime": rng.choice([600.0, 12.0, 0.0, -5.0, None]),
        "logs": [],
        "current": rng.choice([
            None,
            {
                "turn": rng.randint(-1, 40),
                "turnActionCount": rng.randint(0, 50),
                "yourIndex": rng.choice([0, 1, 5, -1]),
                "firstPlayer": rng.choice([-1, 0, 1]),
                "supporterPlayed": False, "stadiumPlayed": False,
                "energyAttached": False, "retreated": False,
                "result": -1, "stadium": [], "looking": None,
                "players": [player, dict(player)],
            },
        ]),
        "select": {
            "type": rng.choice([*range(-1, 13)]),
            "context": rng.choice([*range(-1, 52)]),
            "minCount": lo,
            "maxCount": hi,
            "remainEnergyCost": rng.randint(-1, 5),
            "remainDamageCounter": rng.randint(-1, 8),
            "option": options,
            "deck": None, "contextCard": None, "effect": None,
        },
    }


def fuzz_observations(
    make_shell: Callable[[], SafetyShell],
    cases: int = 4000,
    seed: int = 0,
) -> FuzzReport:
    """Assert the shell never raises and never returns an illegal selection."""
    rng = random.Random(seed)
    db = get_card_db()
    rep = FuzzReport()
    shell = make_shell()

    for i in range(cases):
        obs = _random_observation(rng, db)
        rep.cases += 1
        try:
            action = shell.act(obs)
        except BaseException as exc:  # noqa: BLE001
            rep.fail(case=i, kind="raised", error=f"{type(exc).__name__}: {exc}",
                     trace=traceback.format_exc(limit=4))
            continue

        if not isinstance(action, list) or not all(isinstance(x, int) for x in action):
            rep.fail(case=i, kind="bad_type", action=repr(action)[:200])
            continue

        sel = obs.get("select") if isinstance(obs, dict) else None
        if not isinstance(sel, dict) or not sel:
            # No usable select block: the only defensible answers are the
            # 60-card deck (deck-selection phase) or an empty selection.
            if len(action) not in (0, 60):
                rep.fail(case=i, kind="deck_phase_length", n=len(action))
            continue

        n = len(sel.get("option") or ())
        lo = max(0, int(sel.get("minCount") or 0))
        hi = int(sel.get("maxCount") or 0)
        if any(x < 0 or x >= n for x in action):
            rep.illegal_actions += 1
            rep.fail(case=i, kind="out_of_range", action=action, n_options=n)
        elif n > 0 and (len(action) < min(lo, n) or (hi and len(action) > max(hi, lo))):
            rep.fail(case=i, kind="bad_length", action=action, lo=lo, hi=hi, n=n)
    return rep


# ---------------------------------------------------------------------------
# 2. Random legal decks
# ---------------------------------------------------------------------------

def random_legal_deck(db: CardDB, rng: random.Random, tries: int = 400) -> list[int]:
    """Build a random deck that passes :meth:`CardDB.validate_deck`.

    Random decks reach card interactions no curated list will, which is exactly
    what an engine-stability fuzz wants.
    """
    basics = [c for c in db.all_cards() if c.is_pokemon and c.basic and not c.ace_spec]
    energies = [c for c in db.all_cards() if c.is_basic_energy]
    trainers = [c for c in db.all_cards() if c.is_trainer and not c.ace_spec]

    for _ in range(tries):
        deck: list[int] = []
        for _ in range(rng.randint(3, 6)):
            c = rng.choice(basics)
            deck += [c.card_id] * rng.randint(1, 4)
        for _ in range(rng.randint(4, 9)):
            c = rng.choice(trainers)
            deck += [c.card_id] * rng.randint(1, 4)
        e = rng.choice(energies)
        while len(deck) < 60:
            deck.append(e.card_id)
        deck = deck[:60]
        if len(deck) == 60 and not db.validate_deck(deck):
            return deck
    # Guaranteed-legal fallback.
    c = rng.choice(basics)
    e = rng.choice(energies)
    return [c.card_id] * 4 + [e.card_id] * 56


# ---------------------------------------------------------------------------
# 3. Self-play legality + engine stability
# ---------------------------------------------------------------------------

def _selfplay_worker(games: int, seed: int, random_decks: bool, q) -> None:
    """Run in a child process so a native crash is observable, not fatal."""
    try:
        from ..agents.baselines import RandomPolicy
        from ..eval.harness import play_match

        rng = random.Random(seed)
        db = get_card_db()
        out: dict[str, Any] = {"games": 0, "illegal": 0, "reasons": {}, "error": ""}

        for i in range(games):
            if random_decks:
                d0, d1 = random_legal_deck(db, rng), random_legal_deck(db, rng)
            else:
                from ..decks.registry import available_decks, load_deck

                names = available_decks()
                d0, d1 = load_deck(rng.choice(names)), load_deck(rng.choice(names))

            checked = {"bad": 0}

            def make(deck):
                shell = SafetyShell(
                    RandomPolicy(deck, seed=rng.randint(0, 10**6)),
                    db,
                    FallbackPolicy(deck),
                    BudgetManager(),
                )

                def act(obs):
                    action = shell.act(obs)
                    sel = obs.get("select")
                    if sel:
                        n = len(sel.get("option") or ())
                        lo = int(sel.get("minCount") or 0)
                        hi = int(sel.get("maxCount") or 0)
                        if any(x < 0 or x >= n for x in action) or not (
                            min(lo, n) <= len(action) <= max(hi, lo, 0) or n == 0
                        ):
                            checked["bad"] += 1
                    return action

                return act

            r = play_match(make(d0), make(d1), d0, d1, seed=seed + i)
            out["games"] += 1
            out["illegal"] += checked["bad"]
            out["reasons"][r.reason] = out["reasons"].get(r.reason, 0) + 1
        q.put(out)
    except BaseException as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)})


def fuzz_selfplay(
    games: int = 60,
    seed: int = 0,
    random_decks: bool = True,
    batch: int = 20,
    timeout: float = 300.0,
) -> FuzzReport:
    """Play chaotic games in child processes, recording any native crash."""
    rep = FuzzReport()
    ctx = mp.get_context("spawn")
    done = 0
    batch_i = 0
    while done < games:
        n = min(batch, games - done)
        q = ctx.Queue()
        p = ctx.Process(target=_selfplay_worker, args=(n, seed + 1000 * batch_i, random_decks, q))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            rep.crashes.append({"batch": batch_i, "kind": "hang", "games": n})
            rep.ok = False
        elif p.exitcode not in (0, None):
            rep.crashes.append(
                {"batch": batch_i, "kind": "native_crash", "exitcode": p.exitcode, "games": n}
            )
            rep.ok = False
        else:
            try:
                res = q.get_nowait()
            except Exception:  # noqa: BLE001
                res = {"error": "no result"}
            if res.get("error"):
                rep.fail(batch=batch_i, kind="worker_error", **res)
            else:
                rep.cases += res.get("games", 0)
                rep.illegal_actions += res.get("illegal", 0)
                if res.get("illegal"):
                    rep.fail(batch=batch_i, kind="illegal_action", n=res["illegal"])
        done += n
        batch_i += 1
    return rep


def _selfplay_checkpoint_worker(checkpoint_path: str, games: int, seed: int, random_decks: bool, opponent: str, q) -> None:
    """Same containment pattern as :func:`_selfplay_worker`, but one seat is
    the checkpoint-loaded network policy instead of ``RandomPolicy``. Runs in
    a spawned child process, so ``checkpoint_path`` (a plain string) is what
    gets reconstructed here -- not a live model passed across the process
    boundary, which spawn cannot pickle."""
    try:
        from ..agents.baselines import RandomPolicy
        from ..agents.bc_policy import load_bc_policy
        from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
        from ..eval.harness import play_match

        rng = random.Random(seed)
        db = get_card_db()
        out: dict[str, Any] = {"games": 0, "illegal": 0, "reasons": {}, "error": ""}

        for i in range(games):
            if random_decks:
                deck = random_legal_deck(db, rng)
            else:
                from ..decks.registry import available_decks, load_deck

                deck = load_deck(rng.choice(available_decks()))

            checked = {"bad": 0}

            def make_network_act():
                policy = load_bc_policy(checkpoint_path, deck, db)
                shell = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())

                def act(obs):
                    action = shell.act(obs)
                    sel = obs.get("select")
                    if sel:
                        n = len(sel.get("option") or ())
                        lo = int(sel.get("minCount") or 0)
                        hi = int(sel.get("maxCount") or 0)
                        if any(x < 0 or x >= n for x in action) or not (
                            min(lo, n) <= len(action) <= max(hi, lo, 0) or n == 0
                        ):
                            checked["bad"] += 1
                    return action

                return act

            def make_opponent_act():
                if opponent == "heuristic":
                    pol = HeuristicPolicy(deck, db, HeuristicConfig())
                else:
                    pol = RandomPolicy(deck, seed=rng.randint(0, 10**6))
                shell = SafetyShell(pol, db, FallbackPolicy(deck), BudgetManager())
                return shell.act

            r = play_match(make_network_act(), make_opponent_act(), deck, deck, seed=seed + i)
            out["games"] += 1
            out["illegal"] += checked["bad"]
            out["reasons"][r.reason] = out["reasons"].get(r.reason, 0) + 1
        q.put(out)
    except BaseException as exc:  # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)})


def fuzz_selfplay_checkpoint(
    checkpoint_path: str,
    games: int = 60,
    seed: int = 0,
    random_decks: bool = True,
    opponent: str = "random",
    batch: int = 10,
    timeout: float = 300.0,
) -> FuzzReport:
    """Play real games with a checkpoint-loaded network policy on one seat,
    recording any native crash, illegal action, or Python exception. Batches
    are smaller than :func:`fuzz_selfplay`'s default (network forward passes
    are slower than the heuristic's, and each child process pays MiniLM/model
    load cost once per batch)."""
    rep = FuzzReport()
    ctx = mp.get_context("spawn")
    done = 0
    batch_i = 0
    while done < games:
        n = min(batch, games - done)
        q = ctx.Queue()
        p = ctx.Process(
            target=_selfplay_checkpoint_worker,
            args=(checkpoint_path, n, seed + 1000 * batch_i, random_decks, opponent, q),
        )
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            rep.crashes.append({"batch": batch_i, "kind": "hang", "games": n})
            rep.ok = False
        elif p.exitcode not in (0, None):
            rep.crashes.append(
                {"batch": batch_i, "kind": "native_crash", "exitcode": p.exitcode, "games": n}
            )
            rep.ok = False
        else:
            try:
                res = q.get_nowait()
            except Exception:  # noqa: BLE001
                res = {"error": "no result"}
            if res.get("error"):
                rep.fail(batch=batch_i, kind="worker_error", **res)
            else:
                rep.cases += res.get("games", 0)
                rep.illegal_actions += res.get("illegal", 0)
                if res.get("illegal"):
                    rep.fail(batch=batch_i, kind="illegal_action", n=res["illegal"])
        done += n
        batch_i += 1
    return rep


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Fuzz the agent and the engine.")
    ap.add_argument("--obs-cases", type=int, default=4000)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--random-decks", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from ..decks.registry import load_deck

    db = get_card_db()
    deck = load_deck("bellibolt_lightning")

    def make_shell() -> SafetyShell:
        from ..agents.heuristic import HeuristicConfig, HeuristicPolicy

        return SafetyShell(
            HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager()
        )

    obs_rep = fuzz_observations(make_shell, cases=args.obs_cases)
    play_rep = fuzz_selfplay(games=args.games, random_decks=args.random_decks)

    blob = {"observation_fuzz": obs_rep.as_dict(), "selfplay_fuzz": play_rep.as_dict()}
    text = json.dumps(blob, indent=2, default=str)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if (obs_rep.ok and play_rep.ok) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
