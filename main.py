"""Kaggle submission entrypoint.

The bundle layout Kaggle expects is::

    submission.tar.gz
    |-- main.py          <- this file, top level
    |-- deck.csv         <- 60 card IDs, one per line
    |-- cg/              <- the engine bindings + shared library
    `-- ptcg/            <- our package

Two properties matter more than anything clever in here:

*Import safety.* The agent process does not necessarily have the bundle root on
``sys.path``, so ``bootstrap_paths()`` runs before any project import. If even
that fails, ``agent()`` still answers with a legal move via
``_emergency_agent`` -- an exception at import time would invalidate the entire
submission, and Kaggle validates by playing us against a copy of ourselves.

*Lazy, once-only construction.* The card database and policy are built on the
first call, not at import, and cached in a module global. Model load has been
observed to cost competitors ~8 s of their 600 s budget; building on demand
keeps that cost inside the first decision where the clock is already running
and where we have the most slack.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

try:
    _THIS_FILE = __file__
except NameError:
    # Kaggle's agent loader does not import this module normally: it reads the
    # source and runs ``exec(compile(raw, path, "exec"), {})`` on a bare dict,
    # so builtins like ``__file__`` are never populated. The compiled code
    # object still carries the real path as its filename, though, so recover
    # it from the current frame instead.
    _THIS_FILE = inspect.currentframe().f_code.co_filename

_HERE = Path(_THIS_FILE).resolve().parent
for _p in (str(_HERE), str(_HERE / "ptcg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_AGENT = None
_BROKEN = ""


def _read_deck() -> list[int]:
    """Load ``deck.csv`` from beside this file. Never raises."""
    for name in ("deck.csv", "decks/deck.csv"):
        p = _HERE / name
        if not p.exists():
            continue
        try:
            out: list[int] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    out.extend(int(tok) for tok in line.replace(",", " ").split())
            if len(out) == 60:
                return out
        except Exception:  # noqa: BLE001
            continue
    return []


def _emergency_agent(obs) -> list[int]:
    """Used only if the real agent could not be constructed at all.

    Always legal: picks the first ``minCount`` options the engine offered.
    """
    try:
        sel = (obs or {}).get("select")
        if not sel:
            deck = _read_deck()
            return deck if len(deck) == 60 else [3] * 60
        n = len(sel.get("option") or ())
        lo = int(sel.get("minCount") or 0)
        hi = int(sel.get("maxCount") or 0)
        k = max(0, min(max(lo, 0), n if n else 0))
        _ = hi
        return list(range(k))
    except Exception:  # noqa: BLE001
        return []


def _build():
    global _AGENT, _BROKEN
    if _AGENT is not None or _BROKEN:
        return

    try:
        from ptcg.core.carddb import get_card_db
        from ptcg.core.clock import BudgetManager
        from ptcg.core.engine import bootstrap_paths
        from ptcg.core.safety import SafetyShell
        from ptcg.agents.fallback import FallbackPolicy
        from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy

        bootstrap_paths()

        deck = _read_deck()
        db = get_card_db()
        if len(deck) != 60:
            from ptcg.decks.registry import load_deck

            deck = load_deck("lucario_fighting")

        # Week 0's proven, ladder-validated heuristic stays the default. The
        # Week-1 BC network is opt-in only -- it has not yet been submitted,
        # so switching the *live* submission to it is a deliberate choice the
        # operator makes, not something that happens silently on import.
        bc_checkpoint = os.environ.get("PTCG_BC_CHECKPOINT")
        policy = None
        if bc_checkpoint:
            try:
                from ptcg.agents.bc_policy import load_bc_policy

                policy = load_bc_policy(bc_checkpoint, deck, db)
            except BaseException as exc:  # noqa: BLE001
                # A bad/missing checkpoint must degrade to the heuristic, not
                # take down the whole submission.
                policy = None
                _ = exc

        if policy is None:
            cfg = HeuristicConfig()
            if os.environ.get("PTCG_PREFER_FIRST") is not None:
                cfg.prefer_first = os.environ["PTCG_PREFER_FIRST"] not in ("0", "false", "")
            policy = HeuristicPolicy(deck, db, cfg)

        fallback = FallbackPolicy(deck)
        budget = BudgetManager(total=600.0)

        recorder = None
        trace_dir = os.environ.get("PTCG_TRACE_DIR")
        if trace_dir:
            from ptcg.core.trace import TraceWriter

            recorder = TraceWriter(trace_dir, tag="ladder")

        _AGENT = SafetyShell(
            policy=policy,
            db=db,
            fallback=fallback,
            budget=budget,
            recorder=recorder,
            debug=bool(os.environ.get("PTCG_DEBUG")),
        )
    except BaseException as exc:  # noqa: BLE001
        _BROKEN = f"{type(exc).__name__}: {exc}"
        _AGENT = None


def shell():
    """Expose the shell for local tooling (arena, tests, profiling)."""
    _build()
    return _AGENT


def agent(obs_dict: dict) -> list[int]:
    """The function the cabt environment calls. Must never raise.

    Kaggle's loader execs this file into a bare namespace and takes the *last*
    callable object defined at module level as "the agent" -- it does not look
    up a function named ``agent`` specifically. So this must stay the last
    top-level ``def`` in the file; anything defined after it (like ``shell``)
    would silently become the submitted agent instead.
    """
    try:
        _build()
        if _AGENT is None:
            return _emergency_agent(obs_dict)
        return _AGENT.act(obs_dict)
    except BaseException:  # noqa: BLE001
        return _emergency_agent(obs_dict)


if globals().get("__name__") == "__main__":  # pragma: no cover
    import json

    _build()
    print(
        json.dumps(
            {
                "built": _AGENT is not None,
                "error": _BROKEN,
                "deck_size": len(_read_deck()),
                "policy": getattr(getattr(_AGENT, "policy", None), "name", None),
            },
            indent=2,
        )
    )
