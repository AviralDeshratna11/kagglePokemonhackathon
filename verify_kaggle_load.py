"""Reproduce kaggle_environments.agent.get_last_callable exactly, against a
given main.py path, to catch exec-vs-import mismatches before uploading.

Usage:
    python verify_kaggle_load.py path/to/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

target = Path(sys.argv[1] if len(sys.argv) > 1 else "main.py").resolve()
raw = target.read_text(encoding="utf-8")

# Exactly kaggle_environments.agent.get_last_callable's method.
code_object = compile(raw, str(target), "exec")
env: dict = {}
sys.path.append(str(target.parent))
exec(code_object, env)  # noqa: S102

callables = [(k, v) for k, v in env.items() if callable(v)]
print("callables defined at module level, in order:")
for k, v in callables:
    print(" ", k, "->", v)

name, picked = callables[-1]
print(f"\nkaggle_environments would pick: {name!r}")
assert name == "agent", f"expected 'agent' to be picked, got {name!r}"

# Now actually drive a game through it, the way the real ladder would call it.
sys.path.insert(0, str(target.parent))
from ptcg.eval.harness import play_match  # noqa: E402

deck = env["_read_deck"]()
assert len(deck) == 60, f"bad deck: {len(deck)}"

result = play_match(picked, picked, deck, deck, seed=42)
print("\nresult:", result.as_dict())
assert result.reason == "engine_result", f"did not finish cleanly: {result.reason} ({result.error})"

shell_fn = env["shell"]
report = shell_fn().report
print("safety report:", report)
assert report.policy_exceptions == 0
assert report.fallback_exceptions == 0
assert report.sanitised == 0

print("\nKAGGLE-STYLE EXEC LOAD + SELF-PLAY: PASS")
