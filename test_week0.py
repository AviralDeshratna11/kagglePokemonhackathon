"""Week-0 test suite.

Grouped by what would actually cost us a submission if it broke:

* ``TestSanitize`` / ``TestShellInvariants`` -- the "cannot emit an illegal
  action" guarantee, tested directly and through the shell.
* ``TestRulesMath`` -- weakness, resistance and energy payment. Getting energy
  payment wrong is silent: the agent simply never attacks, which is exactly the
  bug that cost ~25 points of win rate during development.
* ``TestCardDB`` / ``TestDecks`` -- the engine-derived database and deck
  legality.
* ``TestEngineContract`` -- pins the facts about the engine that the rest of the
  project assumes, so that a version bump fails here loudly instead of drifting.
"""

from __future__ import annotations

import random

import pytest

from ptcg.core.carddb import Role, get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.enums import CardType, EnergyType, OptionType, SelectType
from ptcg.core.obs import ObsView, damage_after_type, energy_shortfall
from ptcg.core.safety import SafetyShell, sanitize_selection
from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.decks.registry import available_decks, load_deck


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def deck():
    return load_deck("bellibolt_lightning")


@pytest.fixture()
def shell(db, deck):
    return SafetyShell(
        HeuristicPolicy(deck, db, HeuristicConfig()),
        db,
        FallbackPolicy(deck),
        BudgetManager(),
    )


# ---------------------------------------------------------------------------


class TestSanitize:
    """``sanitize_selection`` must be a total function into legal selections."""

    @pytest.mark.parametrize(
        "sel",
        [None, [], "abc", 3, [None, "x"], [-1, 999], [1.5], [[1]], object()],
    )
    def test_never_raises_and_stays_in_range(self, sel):
        out = sanitize_selection(sel, n_options=4, min_count=1, max_count=2)
        assert isinstance(out, list)
        assert all(isinstance(i, int) and 0 <= i < 4 for i in out)
        assert 1 <= len(out) <= 2

    def test_pads_up_to_min_count(self):
        assert len(sanitize_selection([], 5, 3, 3)) == 3

    def test_truncates_to_max_count(self):
        assert len(sanitize_selection([0, 1, 2, 3, 4], 5, 0, 2)) == 2

    def test_deduplicates_by_default(self):
        assert sanitize_selection([2, 2, 2], 5, 1, 3) == [2]

    def test_allows_repeats_when_min_exceeds_options(self):
        # Damage-counter placement legitimately asks for more picks than there
        # are distinct targets.
        out = sanitize_selection([0], n_options=2, min_count=4, max_count=4)
        assert len(out) == 4

    def test_zero_options(self):
        assert sanitize_selection([0], 0, 0, 1) == []

    def test_min_greater_than_max_is_survivable(self):
        out = sanitize_selection([], 3, 2, 1)
        assert 0 <= len(out) <= 3


class TestShellInvariants:
    def test_deck_phase_returns_sixty(self, shell):
        out = shell.act({"select": None, "current": None, "logs": []})
        assert len(out) == 60

    def test_garbage_observations(self, shell):
        for obs in [None, {}, {"select": {}}, {"select": {"option": []}}, "nope", 7, []]:
            out = shell.act(obs)
            assert isinstance(out, list)
            assert all(isinstance(i, int) for i in out)

    def test_broken_policy_falls_back(self, db, deck):
        class Exploding:
            name = "exploding"

            def deck(self):
                return list(deck)

            def score(self, *a, **k):
                raise RuntimeError("boom")

            def desired_count(self, *a, **k):
                raise RuntimeError("boom")

        sh = SafetyShell(Exploding(), db, FallbackPolicy(deck), BudgetManager())
        obs = {
            "select": {
                "type": SelectType.MAIN, "context": 0, "minCount": 1, "maxCount": 1,
                "remainEnergyCost": 0, "remainDamageCounter": 0,
                "option": [{"type": OptionType.END}], "deck": None,
                "contextCard": None, "effect": None,
            },
            "current": None,
            "logs": [],
        }
        assert sh.act(obs) == [0]
        assert sh.report.policy_exceptions >= 1

    def test_loop_guard_masks_repeats(self, db, deck):
        """A policy that always wants the same ability must eventually be stopped."""
        class Obsessive:
            name = "obsessive"

            def deck(self):
                return list(deck)

            def score(self, view, candidates, deadline):
                return [100.0 if c.option_type == OptionType.ABILITY else 0.0 for c in candidates]

            def desired_count(self, view, candidates, scores):
                return 1

        sh = SafetyShell(Obsessive(), db, FallbackPolicy(deck), BudgetManager(), max_repeats=5)
        obs = {
            "select": {
                "type": SelectType.MAIN, "context": 0, "minCount": 1, "maxCount": 1,
                "remainEnergyCost": 0, "remainDamageCounter": 0,
                "option": [
                    {"type": OptionType.ABILITY, "area": 4, "index": 0, "playerIndex": 0},
                    {"type": OptionType.END},
                ],
                "deck": None, "contextCard": None, "effect": None,
            },
            "current": {
                "turn": 3, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                "retreated": False, "result": -1, "stadium": [], "looking": None,
                "players": [_empty_player(), _empty_player()],
            },
            "logs": [],
        }
        picks = [sh.act(obs)[0] for _ in range(40)]
        assert 1 in picks, "loop guard never forced the END escape hatch"
        assert sh.report.loop_breaks > 0

    def test_budget_is_tracked(self, shell):
        shell.act({"select": None, "current": None, "logs": []})
        stats = shell.budget.stats()
        assert stats["remaining"] <= 600.0


def _empty_player():
    return {
        "active": [], "bench": [], "benchMax": 5, "hand": [], "handCount": 0,
        "deckCount": 40, "discard": [], "prize": [None] * 6,
        "poisoned": False, "burned": False, "asleep": False,
        "paralyzed": False, "confused": False,
    }


class TestRulesMath:
    def test_weakness_doubles(self, db):
        bellibolt = db.card(269)   # Lightning, weak to Fighting
        lucario = db.card(678)     # Fighting
        assert damage_after_type(130, lucario, bellibolt) == 260

    def test_resistance_subtracts_thirty(self, db):
        class Fake:
            energy_type = EnergyType.FIGHTING
            weakness = 0
            resistance = EnergyType.FIGHTING
        assert damage_after_type(100, db.card(678), Fake()) == 70

    def test_no_modifier_when_types_differ(self, db):
        assert damage_after_type(90, db.card(678), db.card(678)) == 90

    @pytest.mark.parametrize(
        "have,need,expect",
        [
            ([], [4, 4, 4, 0], 4),
            ([4], [4, 4, 4, 0], 3),
            ([4, 4, 4, 4], [4, 4, 4, 0], 0),
            ([4, 4, 4, 1], [4, 4, 4, 0], 0),      # Grass pays the Colorless
            ([1, 1, 1, 1], [4, 4, 4, 0], 3),      # wrong colour cannot pay colour
            ([10, 10, 10, 10], [4, 4, 4, 0], 0),  # Rainbow pays anything
            ([0], [0], 0),
            ([], [], 0),
        ],
    )
    def test_energy_shortfall(self, have, need, expect):
        assert energy_shortfall(have, need) == expect

    def test_colorless_uses_leftovers_not_matched_colour(self):
        # 3 Lightning must not pay {L}{L}{L} and a Colorless simultaneously.
        assert energy_shortfall([4, 4, 4], [4, 4, 4, 0]) == 1


class TestCardDB:
    def test_loaded(self, db):
        assert len(db) > 1000

    def test_attack_ids_resolve(self, db):
        bolt = db.attacks_of(269)
        assert bolt and bolt[0].damage == 230
        assert bolt[0].energies == (4, 4, 4, 0)

    def test_role_tagging_finds_self_energy_cost(self, db):
        # Iono's Kilowattrel pays for Flashing Draw with its own Energy.
        assert Role.SELF_ENERGY_COST in db.card(271).roles

    def test_role_tagging_finds_acceleration(self, db):
        assert Role.ENERGY_ACCEL in db.card(269).roles

    def test_prizes_scale_with_rule_box(self, db):
        assert db.card(268).prizes_when_koed == 1      # plain Basic
        assert db.card(269).prizes_when_koed == 2      # Pokemon ex
        assert db.card(678).prizes_when_koed == 3      # Mega Evolution ex

    def test_evolution_links_are_by_name(self, db):
        # Several distinct "Riolu" printings all feed Mega Lucario ex.
        assert len(db.by_name("Riolu")) > 1
        assert db.card(678).evolves_from == "Riolu"


class TestDecks:
    def test_all_registry_decks_are_legal(self, db):
        for name in available_decks():
            assert db.validate_deck(load_deck(name)) == [], name

    def test_wrong_size_rejected(self, db, deck):
        assert db.validate_deck(deck[:59])

    def test_five_copies_rejected(self, db):
        bad = [268] * 5 + [4] * 55
        assert any("4-copy" in p for p in db.validate_deck(bad))

    def test_copy_limit_is_per_name_not_per_id(self, db):
        # 333 and 677 are different IDs, both named "Riolu".
        bad = [333] * 3 + [677] * 3 + [4] * 54
        assert any("Riolu" in p for p in db.validate_deck(bad))

    def test_two_ace_specs_rejected(self, db):
        bad = [1125, 1080] + [268] * 4 + [4] * 54
        assert any("ACE SPEC" in p for p in db.validate_deck(bad))

    def test_orphan_evolution_rejected(self, db):
        bad = [269] * 3 + [268] * 0 + [270] * 4 + [4] * 53
        assert any("evolves from" in p for p in db.validate_deck(bad))

    def test_no_basic_pokemon_rejected(self, db):
        assert any("Basic" in p for p in db.validate_deck([4] * 60))


class TestEngineContract:
    """Pins what the rest of the project assumes about the simulator."""

    def test_undocumented_symbols_are_present(self):
        from ptcg.core.engine import get_engine

        exported = set(get_engine().exported)
        assert {"AllCard", "AllAttack"} <= exported

    def test_clock_is_six_hundred_and_there_is_no_per_move_limit(self):
        import json
        from pathlib import Path

        import kaggle_environments

        spec = json.loads(
            (Path(kaggle_environments.__file__).parent / "envs" / "cabt" / "cabt.json").read_text()
        )
        assert spec["observation"]["remainingOverageTime"] == 600
        assert spec["configuration"]["actTimeout"] == 0

    def test_full_game_completes_without_incident(self, db, deck):
        from ptcg.eval.harness import play_match

        a = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db,
                        FallbackPolicy(deck), BudgetManager())
        b = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db,
                        FallbackPolicy(deck), BudgetManager())
        r = play_match(a.act, b.act, deck, deck, seed=1234)
        assert r.reason == "engine_result"
        assert r.winner in (0, 1, -1)
        for rep in (a.report, b.report):
            assert rep.policy_exceptions == 0
            assert rep.fallback_exceptions == 0
            assert rep.sanitised == 0

    def test_latency_stays_far_under_budget(self, db, deck):
        from ptcg.eval.harness import play_match

        a = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db,
                        FallbackPolicy(deck), BudgetManager())
        b = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db,
                        FallbackPolicy(deck), BudgetManager())
        r = play_match(a.act, b.act, deck, deck, seed=7)
        assert max(r.time_used) < 60.0, "a tenth of the clock on one game is a red flag"


class TestFeatureLayout:
    def test_dimension_matches_names(self):
        from ptcg.core.actions import FEATURE_DIM, FEATURE_NAMES

        assert FEATURE_DIM == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_candidates_have_full_width_features(self, db, deck):
        from ptcg.core.actions import FEATURE_DIM, build_candidates

        obs = {
            "select": {
                "type": SelectType.MAIN, "context": 0, "minCount": 1, "maxCount": 1,
                "remainEnergyCost": 0, "remainDamageCounter": 0,
                "option": [{"type": OptionType.END}, {"type": OptionType.ATTACK, "attackId": 368}],
                "deck": None, "contextCard": None, "effect": None,
            },
            "current": {
                "turn": 5, "turnActionCount": 0, "yourIndex": 0, "firstPlayer": 0,
                "supporterPlayed": False, "stadiumPlayed": False, "energyAttached": False,
                "retreated": False, "result": -1, "stadium": [], "looking": None,
                "players": [_empty_player(), _empty_player()],
            },
            "logs": [],
        }
        cands = build_candidates(ObsView(obs, db), db)
        assert len(cands) == 2
        assert all(len(c.features) == FEATURE_DIM for c in cands)


class TestFuzzSmoke:
    def test_observation_fuzz_clean(self, db, deck):
        from ptcg.eval.fuzz import fuzz_observations

        rep = fuzz_observations(
            lambda: SafetyShell(
                HeuristicPolicy(deck, db, HeuristicConfig()), db,
                FallbackPolicy(deck), BudgetManager()
            ),
            cases=800,
            seed=99,
        )
        assert rep.ok, rep.failures[:3]
        assert rep.illegal_actions == 0
