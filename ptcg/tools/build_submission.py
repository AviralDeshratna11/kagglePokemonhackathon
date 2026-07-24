"""Build and verify ``submission.tar.gz``.

The bundle contract (from the competition rules and ``cabt.py``):

* ``main.py`` at the **top level** of the archive, exposing ``agent(obs)``;
* ``deck.csv`` alongside it;
* the ``cg/`` engine package, because the agent process imports it directly;
* total size at most 197.7 MiB.

Kaggle validates an upload by playing the agent against a copy of itself, so a
bundle that imports cleanly on the dev machine but not in the container fails
*after* consuming one of the five daily submissions. This script therefore does
not just tar a directory: it unpacks the finished archive into a scratch
directory, imports ``main`` from there with the repo removed from ``sys.path``,
and plays a full self-game. If that fails, nothing is written.

Steps, in order:

1. resolve and validate the deck against the engine's own card table;
2. stage only the files that are actually needed;
3. strip caches, tests and artefacts;
4. build the archive and check its size;
5. re-extract, import in isolation, and self-play.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MAX_BYTES = int(197.7 * 1024 * 1024)

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "artifacts", "dist", "tests", "traces", ".venv", "venv", ".idea", ".vscode",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".jsonl", ".gz", ".ipynb"}


def _find_cg_source() -> Path:
    """Locate a ``cg/`` package to vendor into the bundle."""
    local = REPO / "cg"
    if (local / "__init__.py").exists():
        return local
    import kaggle_environments  # noqa: PLC0415

    return Path(kaggle_environments.__file__).parent / "envs" / "cabt" / "cg"


def _copy_tree(src: Path, dst: Path) -> int:
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if Path(f).suffix in EXCLUDE_SUFFIXES:
                continue
            shutil.copy2(Path(root) / f, dst / rel / f)
            n += 1
    return n


def stage(deck_name: str, staging: Path, include_arch: str = "all") -> dict[str, Any]:
    """Assemble the bundle contents in ``staging``."""
    sys.path.insert(0, str(REPO))
    from ptcg.core.carddb import get_card_db
    from ptcg.decks.registry import deck_summary, load_deck, write_deck

    db = get_card_db()
    deck = load_deck(deck_name)
    summary = deck_summary(deck, db)
    if summary["problems"]:
        raise SystemExit(f"deck {deck_name!r} is illegal: {summary['problems']}")

    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "main.py", staging / "main.py")
    write_deck(staging / "deck.csv", deck)

    n_pkg = _copy_tree(REPO / "ptcg", staging / "ptcg")

    cg_src = _find_cg_source()
    cg_dst = staging / "cg"
    cg_dst.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for f in cg_src.iterdir():
        if f.is_dir() or f.suffix in EXCLUDE_SUFFIXES:
            continue
        # The ladder runs linux/amd64; the other binaries are only useful for
        # local development and cost megabytes.
        if include_arch == "linux" and f.name in ("cg.dll", "libcg.dylib", "libcg-arm64.so"):
            continue
        shutil.copy2(f, cg_dst / f.name)
        kept.append(f.name)

    return {
        "deck": deck_name,
        "deck_summary": {k: v for k, v in summary.items() if k != "lines"},
        "package_files": n_pkg,
        "cg_source": str(cg_src),
        "cg_files": sorted(kept),
    }


def verify(bundle: Path, games: int = 1, timeout: int = 900) -> dict[str, Any]:
    """Extract the archive and play a self-game from it, in a clean process.

    ``sys.path`` inside the child is scrubbed of the repo, so an accidental
    dependency on a file that was *not* packaged shows up as an ImportError
    here rather than as an invalid submission.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "extracted"
        root.mkdir()
        with tarfile.open(bundle, "r:gz") as tf:
            tf.extractall(root)

        # Kaggle does not `import main`: kaggle_environments.agent.get_last_callable
        # reads the source, compiles it, and execs it into a *bare* namespace
        # (compile(raw, path, "exec"); env = {}; exec(code_object, env)), then
        # takes the *last callable* defined at module level -- no __file__, no
        # __name__, and no guarantee a function literally named "agent" is what
        # gets called. `import main` would miss all three failure modes, so
        # this reproduces the real loader exactly instead.
        main_path = root / "main.py"
        script = f'''
import json, sys, time
sys.path = [p for p in sys.path if {str(REPO)!r} not in p]
sys.path.insert(0, {str(root)!r})

raw = open({str(main_path)!r}, encoding="utf-8").read()
t0 = time.time()
code_object = compile(raw, {str(main_path)!r}, "exec")
env = {{}}
sys.path.append({str(root)!r})
exec(code_object, env)
load_s = time.time() - t0

callables = [(k, v) for k, v in env.items() if callable(v)]
last_name, agent_fn = callables[-1]
if last_name != "agent":
    print("__RESULT__" + json.dumps({{
        "load_seconds": round(load_s, 3),
        "deck_size": 0,
        "games": [],
        "error": f"kaggle_environments would pick {{last_name!r}} as the agent, not agent()",
    }}))
    raise SystemExit(0)

from ptcg.eval.harness import play_match
deck = env["_read_deck"]()
assert len(deck) == 60, "deck.csv did not yield 60 cards"
results = []
for i in range({games}):
    r = play_match(agent_fn, agent_fn, deck, deck, seed=i)
    results.append(r.as_dict())
print("__RESULT__" + json.dumps({{
    "load_seconds": round(load_s, 3),
    "deck_size": len(deck),
    "games": results,
}}))
'''
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(root),
        )
        out = proc.stdout
        marker = "__RESULT__"
        if proc.returncode != 0 or marker not in out:
            return {
                "ok": False,
                "returncode": proc.returncode,
                "stdout": out[-3000:],
                "stderr": proc.stderr[-3000:],
            }
        payload = json.loads(out.split(marker, 1)[1].splitlines()[0])
        if payload.get("error"):
            payload["ok"] = False
            payload["bad_games"] = []
            return payload
        bad = [
            g for g in payload["games"]
            if g["reason"] in ("agent_exception", "illegal_action", "timeout", "engine_crash")
        ]
        payload["ok"] = not bad
        payload["bad_games"] = bad
        return payload


def build(
    deck_name: str = "bellibolt_lightning",
    out: Path | None = None,
    include_arch: str = "all",
    games: int = 1,
    skip_verify: bool = False,
) -> dict[str, Any]:
    out = out or (REPO / "dist" / "submission.tar.gz")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "bundle"
        info = stage(deck_name, staging, include_arch=include_arch)

        if out.exists():
            out.unlink()
        with tarfile.open(out, "w:gz") as tf:
            for item in sorted(staging.iterdir()):
                tf.add(item, arcname=item.name)

    size = out.stat().st_size
    info.update(
        {
            "bundle": str(out),
            "bytes": size,
            "mib": round(size / 1024 / 1024, 2),
            "limit_mib": round(MAX_BYTES / 1024 / 1024, 2),
            "within_limit": size <= MAX_BYTES,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if not info["within_limit"]:
        raise SystemExit(f"bundle is {info['mib']} MiB, over the {info['limit_mib']} MiB limit")

    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    for required in ("main.py", "deck.csv"):
        if required not in names:
            raise SystemExit(f"{required} is missing from the archive top level")
    info["top_level"] = sorted({n.split("/")[0] for n in names})

    info["verification"] = {"skipped": True} if skip_verify else verify(out, games=games)
    if not skip_verify and not info["verification"].get("ok"):
        raise SystemExit(
            "bundle failed self-play verification:\n"
            + json.dumps(info["verification"], indent=2)[:4000]
        )
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", default="lucario_fighting")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", choices=["all", "linux"], default="all")
    ap.add_argument("--games", type=int, default=1, help="self-play games during verification")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    info = build(
        deck_name=args.deck,
        out=Path(args.out) if args.out else None,
        include_arch=args.arch,
        games=args.games,
        skip_verify=args.skip_verify,
    )
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
