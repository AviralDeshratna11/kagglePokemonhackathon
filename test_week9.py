"""Week-9 test suite: the real-evidence-backed heuristic default, the
symmetric new-deck router in ``deck_bench``, and cross-deck League opponents
for self-play.

Context: the real Kaggle ladder scored the Week-8 heuristic bundle ~18
points *lower* than the build without it (539.6 -> 521.8), and Week 7
diagnosed that self-play refinement never transferred because the league
only ever played mirror matches. This suite locks in the fix for both.
"""

from __future__ import annotations

import torch
import pytest

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.agents.network import BeliefSetDMCLite, CardTable
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match
from ptcg.train.league import League
from ptcg.tools.week3_bench import deck_bench


@pytest.fixture(scope="module")
def db():
    return get_card_db()


class TestRealEvidenceBackedDefault:
    def test_default_config_matches_539_6_scoring_build(self):
        """This is exactly what ``main.py`` constructs with no overrides for
        the live submission -- the class default IS the live agent."""
        cfg = HeuristicConfig()
        assert cfg.search_bonus == 240.0
        assert cfg.enable_expert_strategy_tuning is False

    def test_flag_still_enables_week8_behaviors_when_explicitly_requested(self, db):
        """The Week-8 code wasn't proven wrong, only the untested bundle
        was -- the mechanism must still work for a future isolated A/B."""
        deck = load_deck("kangaskhan_crustle")
        cfg = HeuristicConfig(enable_expert_strategy_tuning=True)
        policy = HeuristicPolicy(deck, db, cfg)

        class _V:
            turn = 20

        card = next(c for cid in deck if (c := db.card(cid)).basic and not c.has_ability)
        # With the flag on, at least one of the four gated behaviors must be
        # able to produce a non-default result (bench-thinning, here).
        if card.name not in policy._evolution_bases:
            assert policy._bench_thinning_bonus(_V(), card) == pytest.approx(cfg.bench_thinning_bonus)


class TestDeckBenchSymmetricRouter:
    def test_router_present_for_new_deck_when_checkpoint_given(self, db):
        info = deck_bench(
            "rocket_mewtwo", "lucario_fighting", games=2, seed=0,
            new_bc_checkpoint="artifacts/bc_rocket_mewtwo.pt",
            baseline_bc_checkpoint="artifacts/bc_text.pt",
            include_crustle=False,
        )
        agents = {row["agent"] for row in info["ladder"]}
        assert "rocket_mewtwo_router" in agents
        assert "lucario_fighting_router" in agents

    def test_router_absent_for_new_deck_without_checkpoint(self, db):
        info = deck_bench(
            "rocket_mewtwo", "lucario_fighting", games=2, seed=0,
            new_bc_checkpoint=None,
            baseline_bc_checkpoint="artifacts/bc_text.pt",
            include_crustle=False,
        )
        agents = {row["agent"] for row in info["ladder"]}
        assert "rocket_mewtwo_router" not in agents
        assert "lucario_fighting_router" in agents


class TestCrossDeckLeague:
    def _tiny_teacher_ckpt(self, db, tmp_path):
        table = CardTable.build(db, with_text=False)
        model = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)
        ckpt_path = tmp_path / "frozen.pt"
        torch.save(
            {
                "state_dict": model.state_dict(), "card_mode": "onehot", "n_cards": table.n_cards,
                "card_ids": table.card_ids, "card_struct": table.struct, "card_text": table.text,
            },
            ckpt_path,
        )
        return ckpt_path

    def test_cross_deck_members_appear_and_pilot_their_own_deck(self, db, tmp_path):
        kang_deck = load_deck("kangaskhan_crustle")
        lucario_deck = load_deck("lucario_fighting")
        ckpt = self._tiny_teacher_ckpt(db, tmp_path)

        league = League(
            db, kang_deck, ckpt,
            cross_deck_opponents={"lucario": (lucario_deck, None)},
        )
        members = league.base_members()
        assert "lucario_heuristic" in members
        assert "lucario_bc" not in members  # no checkpoint given for this entry

        assert league.member_deck("lucario_heuristic") == lucario_deck
        assert league.member_deck("heuristic") == kang_deck  # unaffected same-deck member

    def test_cross_deck_bc_member_appears_when_checkpoint_given(self, db, tmp_path):
        kang_deck = load_deck("kangaskhan_crustle")
        rocket_deck = load_deck("rocket_mewtwo")
        ckpt = self._tiny_teacher_ckpt(db, tmp_path)

        league = League(
            db, kang_deck, ckpt,
            cross_deck_opponents={"rocket": (rocket_deck, "artifacts/bc_rocket_mewtwo.pt")},
        )
        members = league.base_members()
        assert "rocket_heuristic" in members
        assert "rocket_bc" in members
        assert league.member_deck("rocket_bc") == rocket_deck

    def test_cross_deck_game_runs_to_completion(self, db, tmp_path):
        """The real regression this fixes: previously ``generate_round``
        passed the trainee's own deck for *both* seats regardless of which
        deck the sampled opponent actually piloted. A cross-deck opponent's
        policy built from a different deck, played through the engine with
        its own correct deck, must complete a normal game."""
        kang_deck = load_deck("kangaskhan_crustle")
        lucario_deck = load_deck("lucario_fighting")
        ckpt = self._tiny_teacher_ckpt(db, tmp_path)

        league = League(db, kang_deck, ckpt, cross_deck_opponents={"lucario": (lucario_deck, None)})
        opp_name = "lucario_heuristic"
        opp_deck = league.member_deck(opp_name)
        opp_policy = league.make_opponent(opp_name)
        trainee_policy = HeuristicPolicy(kang_deck, db, HeuristicConfig())

        shell0 = SafetyShell(trainee_policy, db, FallbackPolicy(kang_deck), BudgetManager())
        shell1 = SafetyShell(opp_policy, db, FallbackPolicy(opp_deck), BudgetManager())
        result = play_match(shell0.act, shell1.act, kang_deck, opp_deck, seed=0)

        assert result.reason == "engine_result"
        assert shell0.report.policy_exceptions == 0
        assert shell1.report.policy_exceptions == 0

    def test_selfplay_generation_smoke_test_with_cross_deck_roster(self, db, tmp_path):
        """Reuses test_week2.py/test_week7.py's exact self-play smoke
        pattern, now with a non-empty cross-deck roster in the league."""
        from ptcg.agents.network import BeliefSetDMCLite as Net
        from ptcg.tools.generate_league_selfplay import generate_round

        deck = load_deck("kangaskhan_crustle")
        lucario_deck = load_deck("lucario_fighting")
        table = CardTable.build(db, with_text=False)
        model = Net(card_mode="onehot", n_cards=table.n_cards)
        ckpt_path = self._tiny_teacher_ckpt(db, tmp_path)

        league = League(db, deck, ckpt_path, cross_deck_opponents={"lucario": (lucario_deck, None)})
        out_dir = tmp_path / "traces"
        info = generate_round(
            league, model, table, "onehot", out_dir, n_games=6, temperature=1.0, seed_start=1, round_tag="t"
        )
        assert info["total_records"] > 0
        assert info["reasons"].get("engine_result", 0) == 6
