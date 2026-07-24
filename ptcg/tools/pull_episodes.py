"""Pull the official daily Kaggle episode replays and convert them into the
same trace schema ``ptcg/tools/generate_selfplay.py`` produces, so self-play
data and real replay data can be mixed in one BC training run.

**Not runnable on this machine right now**: it needs a Kaggle API token
(``kaggle.json``) that is not configured here. Written and ready so it becomes
a drop-in data source the moment credentials exist -- no other code changes
needed, since ``ptcg/train/dataset.py`` just reads whatever ``.jsonl.gz``
files are in the trace directory.

Usage (once ``kaggle.json`` is in place, per the Kaggle API docs):

    pip install kaggle
    python -m ptcg.tools.pull_episodes --dataset kaggle/pokemon-tcg-ai-battle-episodes-2026-07-19 \
        --out artifacts/traces/replays

Episode JSON format (per the competition's published episode dumps): a list
of steps, each holding the ``obs`` dict handed to an agent and the raw action
it returned, plus final rewards. The parser below is defensive about the
exact key names because the organizers' dump format is not pinned by an API
contract the way the live ``agent(obs)`` interface is -- inspect one real file
and adjust ``_iter_episode_steps`` if the shape has drifted before trusting a
production run of this.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from ..core.actions import build_candidates
from ..core.carddb import CardDB, get_card_db
from ..core.obs import ObsView
from ..core.trace import TraceWriter

__all__ = ["download_dataset", "convert_episode_file", "main"]


def download_dataset(dataset_slug: str, dest: Path) -> Path:
    """Thin wrapper over ``kaggle datasets download``. Requires the ``kaggle``
    package and a configured API token; raises loudly if either is missing
    rather than silently producing an empty corpus."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `kaggle` package is not installed. Run `pip install kaggle` "
            "and place your API token at ~/.kaggle/kaggle.json first."
        ) from exc

    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)
    return dest


def _iter_episode_steps(episode: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any], list[int]]]:
    """Yield ``(player_index, obs_dict, action)`` for every recorded decision
    in one episode dump. Kaggle's generic simulation-episode format nests
    per-step, per-agent observations/actions under ``steps``; adjust here if
    the competition's actual dump differs once a real file is available."""
    steps = episode.get("steps") or []
    for step in steps:
        if not isinstance(step, list):
            continue
        for player_index, agent_step in enumerate(step):
            obs = agent_step.get("observation")
            action = agent_step.get("action")
            if obs is None or action is None:
                continue
            yield player_index, obs, action


def convert_episode_file(path: Path, db: CardDB, out_dir: Path) -> int:
    """Convert one downloaded episode JSON into the trace schema. Returns the
    number of decision records written."""
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data if isinstance(data, list) else [data]

    n_written = 0
    for ep_i, episode in enumerate(episodes):
        rewards = episode.get("rewards") or episode.get("info", {}).get("rewards")
        writers = [TraceWriter(out_dir, tag=f"replay-{path.stem}-{ep_i}-p{p}") for p in range(2)]
        counts = [0, 0]
        try:
            for player_index, obs, action in _iter_episode_steps(episode):
                view = ObsView(obs, db)
                if view.is_deck_selection:
                    continue
                candidates = build_candidates(view, db)
                if not candidates:
                    continue
                writers[player_index](
                    {
                        "player": player_index,
                        "turn": view.turn,
                        "select_type": view.select_type,
                        "context": view.context,
                        "n_options": len(candidates),
                        "chosen": list(action) if isinstance(action, (list, tuple)) else [int(action)],
                        "candidate_card_ids": [c.card_id for c in candidates],
                        "features": [c.features for c in candidates],
                        # Real replays don't give us the opponent's true hand
                        # the way self-play generation privately does, so the
                        # belief head simply isn't trained on this source.
                        "board": {
                            "my_active": [view.me.active.card_id] if view.me.active else [],
                            "my_bench": [p.card_id for p in view.me.bench],
                            "my_hand": [c.card_id if c else None for c in view.me.hand_cards],
                            "my_discard": [d.get("id") for d in view.me.discard],
                            "my_prize": [p.get("id") if p else None for p in view.me.prize],
                            "opp_active": [view.opp.active.card_id] if view.opp.active else [],
                            "opp_bench": [p.card_id for p in view.opp.bench],
                            "opp_discard": [d.get("id") for d in view.opp.discard],
                            "opp_prize": [p.get("id") if p else None for p in view.opp.prize],
                            "opp_hand_count": view.opp.hand_count,
                            "opp_deck_count": view.opp.deck_count,
                            "opp_prizes_remaining": view.opp.prizes_remaining,
                            "my_deck_count": view.me.deck_count,
                        },
                        "globals": {
                            "turn": view.turn,
                            "prize_diff": view.prize_diff,
                            "remaining_time": view.remaining_time,
                            "supporter_played": view.supporter_played,
                            "stadium_played": view.stadium_played,
                            "energy_attached": view.energy_attached,
                            "i_am_first": view.i_am_first,
                        },
                        "logs_tail": list(view.logs[-16:]),
                        "belief_valid": False,
                        "belief_hand_card_ids": [],
                    }
                )
                counts[player_index] += 1
                n_written += 1
        finally:
            for p, w in enumerate(writers):
                outcome = 0.5
                if rewards and len(rewards) == 2:
                    r0, r1 = rewards
                    if r0 != r1:
                        outcome = 1.0 if (r0 if p == 0 else r1) > (r1 if p == 0 else r0) else 0.0
                w.finish({"outcome": outcome, "n_records": counts[p]})
    return n_written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. kaggle/pokemon-tcg-ai-battle-episodes-2026-07-19")
    ap.add_argument("--out", default="artifacts/traces/replays")
    ap.add_argument("--raw-dir", default="artifacts/raw_episodes")
    args = ap.parse_args(argv)

    raw_dir = download_dataset(args.dataset, Path(args.raw_dir))
    db = get_card_db()
    out_dir = Path(args.out)
    total = 0
    for f in sorted(raw_dir.glob("*.json")):
        n = convert_episode_file(f, db, out_dir)
        print(f"{f.name}: {n} records")
        total += n
    print(f"total: {total} records written to {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
