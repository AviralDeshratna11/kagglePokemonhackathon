"""Week-4 test suite: the corrected replay obs/action pairing, one-sided
decision emission, deck capture, identity matching, and multi-directory
trace loading.

The pairing/emission tests build their fixture from a real tiny self-play
game reshaped into Kaggle's own steps convention (action at tick i pairs
with the *previous* tick's observation for that player -- see
``ptcg/tools/pull_episodes.py``'s module docstring for how this was
discovered against real downloaded data), so ``build_candidates`` is
exercised against genuinely engine-produced, parseable observations rather
than hand-faked ones.
"""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
from pathlib import Path

import pytest

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match
from ptcg.train.dataset import load_records
from ptcg.core.trace import TraceWriter
from ptcg.tools.pull_episodes import (
    convert_episode_file,
    episode_agent_names,
    iter_paired_decisions,
    load_target_names,
    matched_players,
)


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def deck():
    return load_deck("lucario_fighting")


def _kaggle_shaped_episode(db, deck, seed=1) -> dict:
    """Play one real game and reshape the (player, obs, action) sequence
    into Kaggle's offset steps convention: ``steps[i][p]['action']``
    responds to ``steps[i-1][p]['observation']``, matching what was found
    in a real downloaded episode this session."""
    shell = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
    per_player: dict[int, list[tuple[dict, list]]] = {0: [], 1: []}

    def on_step(player, obs, action):
        per_player[player].append((obs, list(action)))

    result = play_match(shell.act, shell.act, deck, deck, seed=seed, on_step=on_step)

    # Kaggle's convention (confirmed against real downloaded data): the
    # action recorded at tick i is the response to tick (i-1)'s observation
    # for that same player. per_player[p] holds naturally-paired
    # (obs, action) from our own harness, so reshape by offsetting the
    # action one tick forward relative to the observation.
    max_len = max(len(per_player[0]), len(per_player[1])) + 1
    shaped = []
    for i in range(max_len):
        row = []
        for p in (0, 1):
            obs_i = per_player[p][i][0] if i < len(per_player[p]) else None
            action_i = per_player[p][i - 1][1] if 0 < i <= len(per_player[p]) else []
            row.append({"observation": obs_i, "action": action_i})
        shaped.append(row)

    # ``play_match``'s harness submits decks straight into ``battle_start``,
    # bypassing the obs/select round-trip entirely -- unlike the live Kaggle
    # submission path, where the very first ``agent(obs)`` call *is* the
    # deck-selection prompt (``select`` is None) and the agent's return
    # value *is* the 60-card list. Prepend that real shape by hand so the
    # deck-capture path is exercised the same way it would be against an
    # actual replay, with everything after it still genuine engine play.
    deck_prompt = {"select": None, "current": None, "logs": [], "remainingOverageTime": None, "search_begin_input": None}
    deck_tick = [
        {"observation": deck_prompt, "action": []},
        {"observation": deck_prompt, "action": []},
    ]
    deck_submit_tick = [
        {"observation": shaped[0][0]["observation"], "action": list(deck)},
        {"observation": shaped[0][1]["observation"], "action": list(deck)},
    ]
    shaped = [deck_tick, deck_submit_tick] + shaped[1:]

    r0, r1 = (1, -1) if result.winner == 0 else ((-1, 1) if result.winner == 1 else (0, 0))
    return {
        "id": "test-episode",
        "rewards": [r0, r1],
        "info": {"TeamNames": ["StrongPlayer", "randomjoe123"]},
        "steps": shaped,
    }


class TestPairing:
    def test_iter_paired_decisions_offsets_correctly(self, db, deck):
        episode = _kaggle_shaped_episode(db, deck)
        decisions = list(iter_paired_decisions(episode))
        assert decisions, "expected at least one paired decision"
        for player_index, obs, action in decisions:
            assert player_index in (0, 1)
            assert isinstance(obs, dict)
            assert isinstance(action, list)

    def test_pairing_matches_build_candidates(self, db, deck):
        from ptcg.core.obs import ObsView
        from ptcg.core.actions import build_candidates

        episode = _kaggle_shaped_episode(db, deck)
        checked = 0
        for player_index, obs, action in iter_paired_decisions(episode):
            view = ObsView(obs, db)
            if view.is_deck_selection:
                assert len(action) == 60
                continue
            cands = build_candidates(view, db)
            assert len(cands) == len((obs.get("select") or {}).get("option") or [])
            assert all(0 <= a < max(1, len(cands)) for a in action if isinstance(a, int))
            checked += 1
        assert checked > 0


class TestConversion:
    def test_one_sided_emission(self, db, deck):
        episode = _kaggle_shaped_episode(db, deck)
        with tempfile.TemporaryDirectory() as td:
            ep_path = Path(td) / "ep.json"
            ep_path.write_text(json.dumps(episode), encoding="utf-8")
            out_dir = Path(td) / "out"
            n, decks = convert_episode_file(ep_path, db, out_dir, only_players={0})

            written_players = set()
            for f in out_dir.glob("*.jsonl.gz"):
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        if rec.get("type") == "decision":
                            written_players.add(rec["player"])
            assert written_players <= {0}
            assert n > 0
            assert 0 in decks and len(decks[0]) == 60
            assert 1 in decks and len(decks[1]) == 60

    def test_both_sided_emission_when_both_matched(self, db, deck):
        episode = _kaggle_shaped_episode(db, deck)
        with tempfile.TemporaryDirectory() as td:
            ep_path = Path(td) / "ep.json"
            ep_path.write_text(json.dumps(episode), encoding="utf-8")
            out_dir = Path(td) / "out"
            n, decks = convert_episode_file(ep_path, db, out_dir, only_players={0, 1})

            written_players = set()
            for f in out_dir.glob("*.jsonl.gz"):
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        if rec.get("type") == "decision":
                            written_players.add(rec["player"])
            assert written_players == {0, 1}


class TestIdentity:
    def test_episode_agent_names(self):
        ep = {"info": {"TeamNames": ["Alice", "Bob"]}}
        assert episode_agent_names(ep) == ["Alice", "Bob"]

    def test_matched_players_case_insensitive(self):
        names = ["Alice", "bob"]
        assert matched_players(names, {"alice"}) == [0]
        assert matched_players(names, {"bob", "carol"}) == [1]
        assert matched_players(names, {"nobody"}) == []
        assert sorted(matched_players(names, {"alice", "bob"})) == [0, 1]

    def test_load_target_names_top_n(self, tmp_path):
        csv_path = tmp_path / "lb.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Rank", "TeamId", "TeamName", "LastSubmissionDate", "Score", "SubmissionCount", "TeamMemberUserNames"])
            w.writerow(["1", "1", "Top Team", "2026-01-01", "1000", "1", "topuser"])
            w.writerow(["2", "2", "Second", "2026-01-01", "900", "1", "seconduser,partner"])
            w.writerow(["3", "3", "Third", "2026-01-01", "800", "1", "thirduser"])

        names = load_target_names(csv_path, top_n=2)
        assert "top team" in names
        assert "topuser" in names
        assert "seconduser" in names
        assert "partner" in names
        assert "third" not in names
        assert "thirduser" not in names


class TestMultiDirLoadRecords:
    def test_combines_multiple_directories(self):
        with tempfile.TemporaryDirectory() as td:
            dir_a = Path(td) / "a"
            dir_b = Path(td) / "b"
            for d, tag, n in ((dir_a, "a", 3), (dir_b, "b", 2)):
                w = TraceWriter(d, tag=tag)
                for i in range(n):
                    w({"player": 0, "chosen": [0], "features": [[0.0]], "candidate_card_ids": [1]})
                w.finish({"outcome": 1.0})

            records_single = load_records(dir_a)
            assert len(records_single) == 3

            records_combined = load_records([dir_a, dir_b])
            assert len(records_combined) == 5

            records_csv_style = load_records([str(dir_a), str(dir_b)])
            assert len(records_csv_style) == 5
