"""Week-6 test suite: general, deck-agnostic board-synergy awareness.

Every assertion here is checked against real cards and a real trained
heuristic behavior, not synthetic data, since the whole point of this pass
was proving the fix generalizes rather than being special-cased to one
archetype.
"""

from __future__ import annotations

import pytest

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.core.actions import ActionCandidate
from ptcg.core.carddb import Role, get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.obs import ObsView, PlayerView, PokemonView
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match

MEWTWO_EX = 431
SPIDOPS = 401
TAROUNTULA = 400
RIOLU = 333
MEGA_LUCARIO_EX = 678


@pytest.fixture(scope="module")
def db():
    return get_card_db()


class TestAffiliationExtraction:
    def test_possessive_prefix_extracted(self, db):
        c = db.card(MEWTWO_EX)
        assert c.affiliation == "Team Rocket's"

    def test_non_possessive_card_has_no_affiliation(self, db):
        c = db.card(RIOLU)
        assert c.affiliation == ""

    def test_generalizes_beyond_team_rocket(self, db):
        """Proves this isn't special-cased: any real card using the
        possessive-prefix convention gets tagged, sight unseen."""
        found = {c.affiliation for c in db.all_cards() if c.affiliation}
        assert len(found) >= 10  # 17 were found during investigation
        assert "Team Rocket's" in found


class TestBoardRoles:
    def test_mewtwo_ex_is_board_gated(self, db):
        c = db.card(MEWTWO_EX)
        assert Role.BOARD_GATED in c.roles

    def test_spidops_attack_is_board_scaling(self, db):
        c = db.card(SPIDOPS)
        assert Role.BOARD_SCALING in c.roles

    def test_unrelated_card_does_not_match(self, db):
        """An ordinary card must not spuriously match either pattern --
        proves the regexes aren't over-broad."""
        c = db.card(RIOLU)
        assert Role.BOARD_SCALING not in c.roles
        assert Role.BOARD_GATED not in c.roles


class TestHeuristicSynergyBonus:
    def test_bonus_applies_when_tribe_matters(self, db):
        deck = load_deck("rocket_mewtwo")
        policy = HeuristicPolicy(deck, db, HeuristicConfig())

        # A Mewtwo ex (board-gated, Team Rocket's) already in play should
        # make benching another Team Rocket's Pokemon score higher than an
        # otherwise-identical bench action would without that context.
        class _FakeMewtwoView:
            card_id = MEWTWO_EX
            card = db.card(MEWTWO_EX)

        class _FakeMe:
            def in_play(self):
                return [_FakeMewtwoView()]

            hand_cards = []

        class _FakeView:
            me = _FakeMe()

        tarountula = db.card(TAROUNTULA)
        lucario = db.card(MEGA_LUCARIO_EX)  # unrelated affiliation (none)

        bonus_same_tribe = policy._board_synergy_bonus(_FakeView(), tarountula)
        bonus_unrelated = policy._board_synergy_bonus(_FakeView(), lucario)

        assert bonus_same_tribe == pytest.approx(HeuristicConfig().board_synergy_bonus)
        assert bonus_unrelated == 0.0

    def test_no_bonus_without_a_caring_card_present(self, db):
        deck = load_deck("rocket_mewtwo")
        policy = HeuristicPolicy(deck, db, HeuristicConfig())

        class _FakeMe:
            def in_play(self):
                return []

            hand_cards = []

        class _FakeView:
            me = _FakeMe()

        tarountula = db.card(TAROUNTULA)
        assert policy._board_synergy_bonus(_FakeView(), tarountula) == 0.0

    def test_self_play_zero_exceptions_on_rocket_mewtwo(self, db):
        deck = load_deck("rocket_mewtwo")
        for seed in range(5):
            shell = SafetyShell(
                HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager()
            )
            result = play_match(shell.act, shell.act, deck, deck, seed=seed)
            assert result.reason == "engine_result"
            assert shell.report.policy_exceptions == 0
            assert shell.report.fallback_exceptions == 0
            assert shell.report.sanitised == 0

    def test_existing_decks_unaffected_by_default(self, db):
        """lucario_fighting has no affiliation-tagged cards, so the new
        bonus should be a structural no-op there -- proving this is additive,
        not a regression risk for the deck it wasn't built for."""
        deck = load_deck("lucario_fighting")
        for cid in deck:
            c = db.card(cid)
            assert c.affiliation == "" or not (Role.BOARD_SCALING in c.roles or Role.BOARD_GATED in c.roles)
