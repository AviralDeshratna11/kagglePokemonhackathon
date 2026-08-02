"""Week-5 test suite: the archetype pivot -- the new registered deck, and
the deck-overlap matching added to ``pull_episodes.py`` for a deck-matched
(not just identity-matched) real-replay pull.
"""

from __future__ import annotations

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
from ptcg.tools.pull_episodes import (
    convert_episode_file,
    deck_matches_archetype,
    extract_decks,
)

MEWTWO_EX = 431


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def rocket_deck():
    return load_deck("rocket_mewtwo")


class TestRocketMewtwoDeck:
    def test_legal_and_real(self, db, rocket_deck):
        assert len(rocket_deck) == 60
        assert db.validate_deck(rocket_deck) == []
        assert MEWTWO_EX in rocket_deck

    def test_self_play_zero_exceptions(self, db, rocket_deck):
        for seed in range(3):
            shell = SafetyShell(
                HeuristicPolicy(rocket_deck, db, HeuristicConfig()), db, FallbackPolicy(rocket_deck), BudgetManager()
            )
            result = play_match(shell.act, shell.act, rocket_deck, rocket_deck, seed=seed)
            assert result.reason == "engine_result"
            assert shell.report.policy_exceptions == 0
            assert shell.report.fallback_exceptions == 0
            assert shell.report.sanitised == 0


class TestArchetypeMatching:
    def test_deck_matches_archetype_true(self, rocket_deck):
        assert deck_matches_archetype(rocket_deck, {MEWTWO_EX}, min_overlap=1)

    def test_deck_matches_archetype_false(self, db):
        other_deck = load_deck("lucario_fighting")
        assert not deck_matches_archetype(other_deck, {MEWTWO_EX}, min_overlap=1)

    def test_min_overlap_threshold(self):
        deck = [431, 400, 401, 1, 1, 1]
        assert deck_matches_archetype(deck, {431, 400, 401}, min_overlap=1)
        assert deck_matches_archetype(deck, {431, 400, 401}, min_overlap=3)
        assert not deck_matches_archetype(deck, {431, 400, 401, 999}, min_overlap=4)


def _episode_with_decks(deck_a: list[int], deck_b: list[int]) -> dict:
    """A minimal, real-shape episode fixture with only a deck-selection
    exchange (no in-game decisions needed to exercise ``extract_decks``,
    which deliberately doesn't touch ``build_candidates``)."""
    deck_prompt = {"select": None, "current": None, "logs": [], "remainingOverageTime": None, "search_begin_input": None}
    step0 = [
        {"observation": deck_prompt, "action": []},
        {"observation": deck_prompt, "action": []},
    ]
    step1 = [
        {"observation": deck_prompt, "action": list(deck_a)},
        {"observation": deck_prompt, "action": list(deck_b)},
    ]
    return {"id": "fixture", "rewards": [1, -1], "info": {"TeamNames": ["Alice", "Bob"]}, "steps": [step0, step1]}


class TestExtractDecks:
    def test_extract_both_decks(self, rocket_deck):
        other = load_deck("lucario_fighting")
        episode = _episode_with_decks(rocket_deck, other)
        decks = extract_decks(episode)
        assert decks[0] == list(rocket_deck)
        assert decks[1] == list(other)

    def test_convert_episode_file_archetype_gate(self, db, rocket_deck, tmp_path):
        """Simulates the pull loop's own logic: a player whose deck matches
        the archetype gets kept even though nothing here matches by
        identity -- the actual Week-5 fix for the Week-4 collapse."""
        other = load_deck("lucario_fighting")
        episode = _episode_with_decks(rocket_deck, other)
        ep_path = tmp_path / "ep.json"
        ep_path.write_text(json.dumps(episode), encoding="utf-8")

        archetype_matches = {
            p for p, d in extract_decks(episode).items() if deck_matches_archetype(d, {MEWTWO_EX})
        }
        assert archetype_matches == {0}

        out_dir = tmp_path / "out"
        n, decks = convert_episode_file(ep_path, db, out_dir, only_players=archetype_matches)
        # No in-game decisions in this minimal fixture (only deck selection),
        # so n==0 is expected -- what matters is decks captured for both
        # sides and that only_players correctly scoped to player 0.
        assert n == 0
        assert decks[0] == list(rocket_deck)
        assert decks[1] == list(other)
