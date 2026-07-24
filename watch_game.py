"""Play one real game through the engine and print a move-by-move log.

Usage:
    python watch_game.py [deck_name]
"""
from __future__ import annotations

import sys

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.enums import LogType
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match

deck_name = sys.argv[1] if len(sys.argv) > 1 else "lucario_fighting"
db = get_card_db()
deck = load_deck(deck_name)

a = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
b = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())

seen_logs = 0


def on_step(player: int, obs: dict, action: list[int]) -> None:
    global seen_logs
    logs = obs.get("logs") or []
    for entry in logs[seen_logs:]:
        t = entry.get("type")
        name = LogType(t).name if t is not None else "?"
        print(f"  [{name}] {entry}")
    seen_logs = len(logs)


print(f"Playing 1 game on deck {deck_name!r} (heuristic vs heuristic mirror)...\n")
result = play_match(a.act, b.act, deck, deck, seed=1234, on_step=on_step)

print("\n--- result ---")
print("winner:", result.winner, "(-1 = draw)")
print("reason:", result.reason)
print("turns:", result.turns)
print("decisions per player:", result.decisions)
print("time used per player (s):", result.time_used)
print("max decision latency per player (ms):", result.max_decision_ms)

print("\n--- safety report (should be all zero) ---")
for label, shell in (("player 0", a), ("player 1", b)):
    print(f"{label}: {shell.report}")
