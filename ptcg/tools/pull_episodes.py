"""Pull real Kaggle ladder episode replays from strong players and convert
them into the same trace schema ``ptcg/tools/generate_selfplay.py`` produces,
so real imitation-learning data and self-play data can be mixed in one BC
training run.

Week 4 context: every policy through Week 3 was bootstrapped from our own
heuristic's self-play, so nothing could ever exceed the heuristic's own
ceiling -- confirmed repeatedly (every network policy tied, never beat, the
heuristic in bench). The only way to genuinely raise that ceiling is to
imitate someone better, which means real ladder replays.

Two things this file gets right that a first speculative pass (Week 1, never
run against real data) got wrong, discovered by hand-validating against a
real downloaded episode this session:

1. **Observation/action pairing is offset by one.** Kaggle's replay format
   stores, at ``steps[i][p]``, the observation player ``p`` *received* at
   tick ``i`` and the action they submit *in response to it* -- but that
   action is recorded at ``steps[i][p]['action']``... no wait, it is not:
   empirically, ``steps[i][p]['action']`` is the action player ``p`` took in
   response to the observation from ``steps[i-1][p]``, not
   ``steps[i][p]['observation']``. Verified against a real file: step 0 has
   ``select=None`` (the deck-selection prompt) for both players with empty
   actions; the actual 60-card deck submissions appear as the *action* at
   step 1, still paired with step 0's prompt. Cross-checked against
   ``build_candidates`` for 127 real non-deck decisions across both players
   in one file: 0 mismatches once the action is paired with the *previous*
   tick's observation for that same player.
2. **There is no cheap episode -> team identity index.** The per-day
   datasets are thousands of individual per-episode JSON files with no
   lookup table; identity comes only from each episode's own
   ``info.TeamNames``/``info.Agents[].Name``, which has to be matched
   (loosely -- personal display names, not always the same as the
   leaderboard's team/username columns) against a target set built from the
   public leaderboard CSV.

Usage (once ``kaggle`` is installed and a token is configured, per the
Kaggle API docs)::

    pip install kaggle
    python -m ptcg.tools.pull_episodes \
        --leaderboard-csv artifacts/kaggle_pull/leaderboard.csv \
        --days kaggle/pokemon-tcg-ai-battle-episodes-2026-07-27,kaggle/pokemon-tcg-ai-battle-episodes-2026-07-26 \
        --max-files-per-day 300 \
        --out artifacts/traces/replays
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from ..core.actions import build_candidates
from ..core.carddb import CardDB, get_card_db
from ..core.obs import ObsView
from ..core.trace import TraceWriter

__all__ = [
    "download_dataset",
    "load_target_names",
    "episode_agent_names",
    "matched_players",
    "iter_paired_decisions",
    "convert_episode_file",
    "list_episode_files",
    "pull_and_convert",
    "main",
]


# ---------------------------------------------------------------------------
# Whole-dataset download (kept for the "download everything" path / small
# datasets like the leaderboard/index CSVs).
# ---------------------------------------------------------------------------


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
            "and place your API token at ~/.kaggle/access_token (or the "
            "legacy ~/.kaggle/kaggle.json) first."
        ) from exc

    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    subprocess.run(cmd, check=True)
    return dest


# ---------------------------------------------------------------------------
# Identity: leaderboard target set + per-episode player matching.
# ---------------------------------------------------------------------------


def load_target_names(leaderboard_csv: str | Path, top_n: int = 150) -> set[str]:
    """Lowercased set of team names + individual usernames from the top
    ``top_n`` rows of a downloaded public-leaderboard CSV
    (``kaggle competitions leaderboard <slug> --download``). There is no
    canonical join key between this and an episode's own ``info.TeamNames``
    (personal display names, not always equal to either column here), so
    this is a best-effort target set, not a guaranteed identity lookup."""
    rows: list[dict[str, str]] = []
    with open(leaderboard_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["Rank"]))

    names: set[str] = set()
    for row in rows[:top_n]:
        team = (row.get("TeamName") or "").strip()
        if team:
            names.add(team.lower())
        for u in (row.get("TeamMemberUserNames") or "").split(","):
            u = u.strip()
            if u:
                names.add(u.lower())
    return names


def episode_agent_names(episode: dict[str, Any]) -> list[str]:
    """The two players' display names for one episode, best-effort."""
    info = episode.get("info") or {}
    names = info.get("TeamNames")
    if isinstance(names, list) and len(names) == 2:
        return [str(n) for n in names]
    agents = info.get("Agents") or []
    return [str(a.get("Name", "")) for a in agents[:2]] + [""] * max(0, 2 - len(agents))


def matched_players(names: list[str], target_names: set[str]) -> list[int]:
    """Indices (0, 1, both, or neither) whose display name is in the target
    set, case-insensitive."""
    return [i for i, n in enumerate(names) if n and n.strip().lower() in target_names]


def extract_decks(episode: dict[str, Any]) -> dict[int, list[int]]:
    """Cheap deck-only scan: pulls each player's submitted 60-card deck
    straight from the raw steps, without touching ``CardDB``/
    ``build_candidates`` -- deck-selection needs neither, so this is much
    cheaper than a full ``convert_episode_file`` pass and lets a
    deck-overlap filter run *before* deciding whether an episode is worth
    fully converting."""
    decks: dict[int, list[int]] = {}
    steps = episode.get("steps") or []
    for player_index in (0, 1):
        prev_obs: dict[str, Any] | None = None
        for step in steps:
            if not isinstance(step, list) or len(step) < 2:
                continue
            agent_step = step[player_index]
            action = agent_step.get("action")
            if action and prev_obs is not None and prev_obs.get("select") is None:
                if len(action) == 60 and all(isinstance(c, int) for c in action):
                    decks[player_index] = list(action)
            prev_obs = agent_step.get("observation")
    return decks


def deck_matches_archetype(deck: list[int], anchor_cards: set[int], min_overlap: int = 1) -> bool:
    """Week 5: does this real player's deck share enough of the adopted
    archetype's defining cards to be useful in-distribution training data --
    independent of the player's overall leaderboard rank, since the goal is
    "learn to pilot *this* deck," which needs deck-consistent data more than
    absolute top-rank data (see Week 4's ``bc_replay`` collapse: 0 of 26
    captured decks matched our own archetype at all)."""
    return len(set(deck) & anchor_cards) >= min_overlap


# ---------------------------------------------------------------------------
# Parsing: corrected observation/action pairing (see module docstring).
# ---------------------------------------------------------------------------


def iter_paired_decisions(episode: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any], list[int]]]:
    """Yield ``(player_index, obs_dict, action)`` for every real decision in
    one episode, with the action correctly paired to the *previous* tick's
    observation for that player (see module docstring for why -- this is
    not the naive same-index pairing)."""
    steps = episode.get("steps") or []
    for player_index in (0, 1):
        prev_obs: dict[str, Any] | None = None
        for step in steps:
            if not isinstance(step, list) or len(step) < 2:
                continue
            agent_step = step[player_index]
            action = agent_step.get("action")
            if action and prev_obs is not None:
                yield player_index, prev_obs, list(action)
            prev_obs = agent_step.get("observation")


def convert_episode_file(
    path: Path,
    db: CardDB,
    out_dir: Path,
    only_players: set[int] | None = None,
) -> tuple[int, dict[int, list[int]]]:
    """Convert one downloaded episode JSON into the trace schema.

    ``only_players``, when given, restricts emitted training *decisions* to
    those player indices -- the "imitate the advanced player, not their
    opponent" behavior. When ``None``, both sides are emitted (matches the
    old whole-dataset-download path's behavior).

    Returns ``(n_records_written, decks)`` where ``decks`` maps player index
    -> the 60-card list they submitted, for every player whose deck-selection
    response was observed (independent of ``only_players``, since this is
    cheap and useful even for a non-matched side).
    """
    episode = json.loads(path.read_text(encoding="utf-8"))
    rewards = episode.get("rewards")

    writers: dict[int, TraceWriter] = {}
    counts = {0: 0, 1: 0}
    decks: dict[int, list[int]] = {}
    n_written = 0

    try:
        for player_index, obs, action in iter_paired_decisions(episode):
            view = ObsView(obs, db)
            if view.is_deck_selection:
                if all(isinstance(c, int) for c in action) and len(action) == 60:
                    decks[player_index] = list(action)
                continue
            if only_players is not None and player_index not in only_players:
                continue

            candidates = build_candidates(view, db)
            if not candidates:
                continue

            if player_index not in writers:
                writers[player_index] = TraceWriter(out_dir, tag=f"replay-{path.stem}-p{player_index}")

            writers[player_index](
                {
                    "player": player_index,
                    "turn": view.turn,
                    "select_type": view.select_type,
                    "context": view.context,
                    "n_options": len(candidates),
                    "chosen": [int(a) for a in action],
                    "candidate_card_ids": [c.card_id for c in candidates],
                    "features": [c.features for c in candidates],
                    # Real replays don't give us the opponent's true hand the
                    # way self-play generation privately does, so the belief
                    # head simply isn't trained on this source.
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
        for p, w in writers.items():
            outcome = 0.5
            if rewards and len(rewards) == 2:
                r0, r1 = rewards
                if r0 != r1:
                    outcome = 1.0 if (r0 if p == 0 else r1) > (r1 if p == 0 else r0) else 0.0
            w.finish({"outcome": outcome, "n_records": counts[p]})

    return n_written, decks


# ---------------------------------------------------------------------------
# Selective per-file pull: list files in a day's dataset, download one at a
# time, check identity, convert-and-discard immediately so raw episode JSON
# never accumulates beyond one file on disk.
# ---------------------------------------------------------------------------


def list_episode_files(dataset_slug: str, api: Any = None) -> list[str]:
    """Every ``*.json`` filename in a per-day episode dataset, paginated."""
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

    names: list[str] = []
    token = None
    while True:
        page = api.dataset_list_files(dataset_slug, page_token=token, page_size=200)
        files = getattr(page, "files", None) or []
        for f in files:
            name = getattr(f, "name", None) or (f.get("name") if isinstance(f, dict) else None)
            if name and name.endswith(".json"):
                names.append(name)
        token = getattr(page, "nextPageToken", None) or getattr(page, "next_page_token", None)
        if not token or not files:
            break
    return names


def _download_with_timeout(api: Any, slug: str, fname: str, raw_scratch_dir: Path, timeout: float) -> bool:
    """``api.dataset_download_file`` blocks with no timeout of its own; a
    single hung connection can stall an entire pull indefinitely (observed
    directly this session, hard-killed by hand). Runs the call on a
    *daemon* thread and gives up after ``timeout`` seconds -- deliberately
    not ``concurrent.futures.ThreadPoolExecutor``, which registers an
    atexit hook that would still block process exit waiting for a hung
    worker even after this function "gives up." A daemon thread can be
    abandoned outright; the process exiting simply kills it."""
    import threading

    result: dict[str, BaseException | None] = {"error": None}
    done = threading.Event()

    def _run() -> None:
        try:
            api.dataset_download_file(slug, fname, path=str(raw_scratch_dir), force=True, quiet=True)
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not done.wait(timeout=timeout):
        return False  # timed out; thread is abandoned, not joined
    if result["error"] is not None:
        raise result["error"]
    return True


def pull_and_convert(
    dataset_slugs: list[str],
    target_names: set[str],
    out_dir: Path,
    raw_scratch_dir: Path,
    max_files_per_day: int | None = 400,
    api: Any = None,
    archetype_anchor_cards: set[int] | None = None,
    archetype_min_overlap: int = 1,
) -> dict[str, Any]:
    """The staged, bounded pull: list files per day, download+inspect one at
    a time, keep decisions from players matching ``target_names`` (identity)
    **or** ``archetype_anchor_cards`` (deck overlap), discard the raw JSON
    immediately either way. Reports an honest hit-rate summary rather than
    assuming a target match rate.

    Week 5: identity-only matching (Week 4) pulled decisions from strong
    players regardless of what deck they ran, which produced a corpus with
    zero overlap with our own archetype and a policy that collapsed
    out-of-distribution. ``archetype_anchor_cards`` adds a second, independent
    match path -- keep a player's decisions if their own deck shares enough
    of the adopted archetype's defining cards, regardless of their overall
    leaderboard rank, so the corpus is finally deck-consistent with what gets
    deployed."""
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

    db = get_card_db()
    raw_scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "days": [],
        "files_scanned": 0,
        "files_matched": 0,
        "files_matched_by_identity": 0,
        "files_matched_by_archetype": 0,
        "files_both_matched": 0,
        "records_written": 0,
        "decks_found": [],
        "errors": 0,
    }

    for slug in dataset_slugs:
        day_stat = {"dataset": slug, "files_scanned": 0, "files_matched": 0, "records_written": 0}
        try:
            files = list_episode_files(slug, api=api)
        except Exception as exc:  # noqa: BLE001
            summary["errors"] += 1
            day_stat["error"] = str(exc)
            summary["days"].append(day_stat)
            continue

        if max_files_per_day is not None:
            files = files[:max_files_per_day]

        for fname in files:
            local_path = raw_scratch_dir / fname
            try:
                if not _download_with_timeout(api, slug, fname, raw_scratch_dir, timeout=60.0):
                    # Week 7: a Week-5 pull stalled indefinitely on a hung
                    # network call with no way to notice from the outside.
                    # A bounded per-file timeout means one bad connection
                    # costs 60s, not the rest of the session.
                    summary["errors"] += 1
                    day_stat["timeouts"] = day_stat.get("timeouts", 0) + 1
                    continue
                if not local_path.exists():
                    continue
                episode = json.loads(local_path.read_text(encoding="utf-8"))
                names = episode_agent_names(episode)
                identity_matches = set(matched_players(names, target_names))

                archetype_matches: set[int] = set()
                if archetype_anchor_cards:
                    decks_probe = extract_decks(episode)
                    for p, pdeck in decks_probe.items():
                        if deck_matches_archetype(pdeck, archetype_anchor_cards, archetype_min_overlap):
                            archetype_matches.add(p)

                matches = identity_matches | archetype_matches

                summary["files_scanned"] += 1
                day_stat["files_scanned"] += 1
                if identity_matches:
                    summary["files_matched_by_identity"] += 1
                if archetype_matches:
                    summary["files_matched_by_archetype"] += 1

                if matches:
                    summary["files_matched"] += 1
                    day_stat["files_matched"] += 1
                    if len(matches) == 2:
                        summary["files_both_matched"] += 1
                    n_written, decks = convert_episode_file(local_path, db, out_dir, only_players=matches)
                    summary["records_written"] += n_written
                    day_stat["records_written"] += n_written
                    for p in matches:
                        if p in decks:
                            summary["decks_found"].append(
                                {"episode": episode.get("id"), "player": p, "name": names[p], "deck": decks[p]}
                            )
            except Exception:  # noqa: BLE001 - one bad file must not kill the pull
                summary["errors"] += 1
            finally:
                try:
                    local_path.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

        summary["days"].append(day_stat)

    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leaderboard-csv", default=None, help="omit to disable identity matching entirely")
    ap.add_argument("--days", required=True, help="comma-separated dataset slugs")
    ap.add_argument("--top-n", type=int, default=150)
    ap.add_argument("--max-files-per-day", type=int, default=400)
    ap.add_argument("--out", default="artifacts/traces/replays")
    ap.add_argument("--raw-scratch", default="artifacts/kaggle_raw_scratch")
    ap.add_argument("--report", default="artifacts/week4_pull_report.json")
    ap.add_argument(
        "--archetype-anchor-cards", default=None,
        help="comma-separated card IDs; a player's own deck containing >= --archetype-min-overlap of these "
        "is kept regardless of leaderboard identity (Week 5 deck-matched pull)",
    )
    ap.add_argument("--archetype-min-overlap", type=int, default=1)
    args = ap.parse_args(argv)

    target_names: set[str] = set()
    if args.leaderboard_csv:
        target_names = load_target_names(args.leaderboard_csv, top_n=args.top_n)
        print(f"target identity set: {len(target_names)} names/usernames from top {args.top_n}")

    archetype_anchor_cards = None
    if args.archetype_anchor_cards:
        archetype_anchor_cards = {int(x) for x in args.archetype_anchor_cards.split(",") if x.strip()}
        print(f"archetype anchor cards: {archetype_anchor_cards} (min overlap {args.archetype_min_overlap})")

    t0 = time.time()
    summary = pull_and_convert(
        args.days.split(","),
        target_names,
        Path(args.out),
        Path(args.raw_scratch),
        max_files_per_day=args.max_files_per_day,
        archetype_anchor_cards=archetype_anchor_cards,
        archetype_min_overlap=args.archetype_min_overlap,
    )
    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in summary.items() if k != "decks_found"}, indent=2))
    print(f"decks captured: {len(summary['decks_found'])}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(args.raw_scratch, ignore_errors=True)
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
