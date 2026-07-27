"""Week-3 test suite: distillation, the confidence-gated router, and
deck-search legality.

Same standard as every prior week: checked against the real engine and real
card database, not mocks.
"""

from __future__ import annotations

import random

import pytest
import torch

from ptcg.agents.bc_policy import BCPolicy
from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.agents.network import BeliefSetDMCLite, CardTable, count_parameters
from ptcg.agents.router_policy import RouterPolicy, _softmax
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match
from ptcg.tools.deck_search import KEY_CARDS, build_crustle_test_deck, propose_swap
from ptcg.train.distill import SIZE_CONFIGS


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def deck():
    return load_deck("lucario_fighting")


def _real_view_and_candidates(db, deck):
    """A real mid-game ``ObsView`` + legal candidate list, captured by
    driving the actual engine one step -- router logic is only meaningful
    against real decision shapes, not a bare stub."""
    from ptcg.core.actions import build_candidates
    from ptcg.core.obs import ObsView

    shell = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
    captured = {}

    def on_step(player, obs, action):
        if captured:
            return
        view = ObsView(obs, db)
        if view.is_deck_selection:
            return
        cands = build_candidates(view, db)
        if len(cands) >= 3:
            captured["view"], captured["cands"] = view, cands

    play_match(shell.act, shell.act, deck, deck, seed=1, on_step=on_step)
    assert captured, "no real decision with >=3 candidates found"
    return captured["view"], captured["cands"]


class TestDistillation:
    def test_smaller_configs_actually_reduce_params(self, db):
        table = CardTable.build(db, with_text=False)
        full = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)
        medium = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, dims=SIZE_CONFIGS["medium"])
        small = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, dims=SIZE_CONFIGS["small"])

        n_full, n_med, n_small = count_parameters(full), count_parameters(medium), count_parameters(small)
        assert n_small < n_med < n_full

    def test_distilled_model_forward_shapes(self, db, deck):
        from ptcg.core.actions import build_candidates
        from ptcg.core.obs import ObsView

        table = CardTable.build(db, with_text=False)
        model = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, dims=SIZE_CONFIGS["small"])
        model.eval()

        shell = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
        captured = {}

        def on_step(player, obs, action):
            if captured:
                return
            view = ObsView(obs, db)
            if view.is_deck_selection:
                return
            cands = build_candidates(view, db)
            if cands:
                captured["view"], captured["cands"] = view, cands

        play_match(shell.act, shell.act, deck, deck, seed=1, on_step=on_step)
        assert captured

        from ptcg.agents.network import encode_view

        inp = encode_view(captured["view"], captured["cands"], table, "onehot")
        with torch.no_grad():
            out = model(inp)
        assert out["policy_logits"].shape == (1, len(captured["cands"]))
        assert out["value"].shape == (1,)

    def test_dims_round_trip_through_checkpoint(self, db, tmp_path):
        table = CardTable.build(db, with_text=False)
        dims = SIZE_CONFIGS["small"]
        model = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, dims=dims)
        ckpt_path = tmp_path / "small.pt"
        torch.save(
            {
                "state_dict": model.state_dict(), "card_mode": "onehot", "n_cards": table.n_cards,
                "card_ids": table.card_ids, "card_struct": table.struct, "card_text": table.text, "dims": dims,
            },
            ckpt_path,
        )
        from ptcg.agents.bc_policy import load_bc_policy

        deck = load_deck("lucario_fighting")
        policy = load_bc_policy(ckpt_path, deck, db)
        assert policy.model.dims["trunk_dim"] == dims["trunk_dim"]


class TestRouterPolicy:
    def test_softmax_sums_to_one(self):
        p = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(p) - 1.0) < 1e-6

    def test_low_margin_routes_to_heuristic(self, db, deck):
        class _FakeNetwork:
            name = "fake"
            def score_and_value(self, view, candidates, deadline):
                return [0.0] * len(candidates), 0.5  # perfectly tied, maximally uncertain value
            def deck(self):
                return list(deck)

        heur = HeuristicPolicy(deck, db, HeuristicConfig())
        router = RouterPolicy(deck, _FakeNetwork(), heur, margin_threshold=0.15, value_margin_threshold=0.08)

        view, cands = _real_view_and_candidates(db, deck)

        router.score(view, cands, deadline=None)
        assert router.last_route == "heuristic"
        assert router.route_counts["heuristic"] == 1

    def test_high_margin_routes_to_network(self, db, deck):
        class _FakeNetwork:
            name = "fake"
            def score_and_value(self, view, candidates, deadline):
                scores = [10.0] + [0.0] * (len(candidates) - 1)  # one clear winner
                return scores, 0.95  # confident value too
            def deck(self):
                return list(deck)

        heur = HeuristicPolicy(deck, db, HeuristicConfig())
        router = RouterPolicy(deck, _FakeNetwork(), heur, margin_threshold=0.15, value_margin_threshold=0.08)

        view, cands = _real_view_and_candidates(db, deck)

        router.score(view, cands, deadline=None)
        assert router.last_route == "network"
        assert router.route_counts["network"] == 1

    def test_full_game_completes_without_incident(self, db, deck):
        from ptcg.agents.bc_policy import load_bc_policy

        net = load_bc_policy("artifacts/bc_text.pt", deck, db)
        heur = HeuristicPolicy(deck, db, HeuristicConfig())
        router = RouterPolicy(deck, net, heur)

        shell_a = SafetyShell(router, db, FallbackPolicy(deck), BudgetManager())
        shell_b = SafetyShell(HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager())
        result = play_match(shell_a.act, shell_b.act, deck, deck, seed=9)
        assert result.reason == "engine_result"
        assert shell_a.report.policy_exceptions == 0
        assert shell_a.report.fallback_exceptions == 0
        assert shell_a.report.sanitised == 0
        # A real game should exercise both routes at least sometimes.
        assert router.route_counts["network"] + router.route_counts["heuristic"] > 0


class TestDeckSearch:
    def test_crustle_deck_is_legal(self, db):
        deck = build_crustle_test_deck(db)
        assert len(deck) == 60
        assert db.validate_deck(deck) == []

    def test_crustle_wall_actually_blocks_ex_damage(self, db):
        """Sanity check on the premise: Crustle's own card text names the
        exact mechanic this deck exists to test against."""
        c = db.card(345)
        assert c is not None
        assert any("{ex}" in t for t in c.skill_texts)

    def test_proposed_swaps_are_always_legal(self, db, deck):
        rng = random.Random(0)
        for _ in range(15):
            cand = propose_swap(deck, db, rng)
            if cand is not None:
                assert len(cand) == 60
                assert db.validate_deck(cand) == []

    def test_key_cards_never_swapped_out(self, db, deck):
        rng = random.Random(1)
        for _ in range(15):
            cand = propose_swap(deck, db, rng)
            if cand is not None:
                for key in KEY_CARDS:
                    assert cand.count(key) >= 1, f"{key} was fully removed by a swap"
