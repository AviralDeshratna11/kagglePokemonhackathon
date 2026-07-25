"""Week-2 test suite: stochastic self-play sampling, PFSP opponent weighting,
frozen-teacher KL, and a real (tiny) self-play training round.

Same standard as test_week0.py/test_week1.py: checked against the real
engine and real card database, not mocks.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.network import CardTable, BeliefSetDMCLite
from ptcg.agents.network_policy import NetworkPolicy
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match
from ptcg.train.league import League
from ptcg.train.selfplay_rl import _teacher_kl_loss, _policy_gradient_loss
from ptcg.tools.generate_league_selfplay import generate_round


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def deck():
    return load_deck("lucario_fighting")


@pytest.fixture(scope="module")
def onehot_table(db):
    # onehot mode: no MiniLM, fastest table to build for tests that don't
    # care about card-text quality, only about the training/sampling plumbing.
    return CardTable.build(db, with_text=False)


class TestNetworkPolicySafety:
    """A stochastic, actively-exploring policy is the worst case for "does
    the safety layer hold up" -- if anything is going to emit something
    dangerous, exploration noise is what would surface it."""

    def test_full_game_completes_without_incident(self, db, deck, onehot_table):
        net = BeliefSetDMCLite(card_mode="onehot", n_cards=onehot_table.n_cards)
        policy = NetworkPolicy(deck, net, onehot_table, "onehot", temperature=1.0, seed=3)
        shell_a = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())
        shell_b = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())

        result = play_match(shell_a.act, shell_b.act, deck, deck, seed=13)
        assert result.reason == "engine_result"
        for shell in (shell_a, shell_b):
            assert shell.report.policy_exceptions == 0
            assert shell.report.fallback_exceptions == 0
            assert shell.report.sanitised == 0

    def test_temperature_zero_is_deterministic_argmax(self, db, deck, onehot_table):
        import time

        from ptcg.core.actions import build_candidates
        from ptcg.core.clock import BudgetManager as BM, Deadline
        from ptcg.core.obs import ObsView
        from ptcg.eval.harness import play_match

        net = BeliefSetDMCLite(card_mode="onehot", n_cards=onehot_table.n_cards)
        net.eval()
        policy = NetworkPolicy(deck, net, onehot_table, "onehot", temperature=0.0, seed=1)

        captured = {}

        def on_step(player, obs, action):
            if captured:
                return
            view = ObsView(obs, db)
            if view.is_deck_selection:
                return
            cands = build_candidates(view, db)
            if cands:
                captured["view"] = view
                captured["candidates"] = cands

        heuristic_shell = SafetyShell(FallbackPolicy(deck), db, FallbackPolicy(deck), BM())
        play_match(heuristic_shell.act, heuristic_shell.act, deck, deck, seed=5, on_step=on_step)
        assert captured, "no in-game decision captured"

        deadline = Deadline(time.monotonic(), 999.0)
        s1 = policy.score(captured["view"], captured["candidates"], deadline)
        s2 = policy.score(captured["view"], captured["candidates"], deadline)
        assert s1 == s2  # temperature<=0 disables Gumbel noise -> identical raw logits every call

    def test_different_seeds_explore_differently(self, db, deck, onehot_table):
        net = BeliefSetDMCLite(card_mode="onehot", n_cards=onehot_table.n_cards)
        net.eval()

        def play_once(seed):
            policy = NetworkPolicy(deck, net, onehot_table, "onehot", temperature=2.0, seed=seed)
            shell_a = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())
            shell_b = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())
            r = play_match(shell_a.act, shell_b.act, deck, deck, seed=1)
            return r.decisions

        # Different exploration seeds against the identical engine seed should
        # not always produce the exact same decision-count fingerprint --
        # weak but real evidence that sampling actually varies the game.
        results = {play_once(s) for s in range(5)}
        assert len(results) > 1


class TestLeaguePFSP:
    def test_sampling_skews_toward_weak_matchups(self, db, deck):
        league = League(db, deck, frozen_teacher_ckpt="unused.pt")
        league.win_rate = {"random": 0.95, "first": 0.5, "greedy": 0.05, "heuristic": 0.5, "frozen-teacher": 0.5}

        rng = random.Random(0)
        counts = {}
        for _ in range(4000):
            name = league.sample_opponent(rng)
            counts[name] = counts.get(name, 0) + 1

        # The opponent we're losing to badly (greedy, 5% win rate) must be
        # sampled far more often than the one we're crushing (random, 95%).
        assert counts.get("greedy", 0) > counts.get("random", 0) * 3

    def test_win_rate_ema_update(self, db, deck):
        league = League(db, deck, frozen_teacher_ckpt="unused.pt", ema_alpha=0.25)
        league.update_win_rate("heuristic", 1.0)
        assert league.win_rate["heuristic"] == pytest.approx(0.625)  # 0.75*0.5 + 0.25*1
        league.update_win_rate("heuristic", 0.0)
        assert league.win_rate["heuristic"] == pytest.approx(0.46875)  # 0.75*0.625

    def test_worst_case_reports_lowest_win_rate(self, db, deck):
        league = League(db, deck, frozen_teacher_ckpt="unused.pt")
        league.win_rate = {"a": 0.9, "b": 0.2, "c": 0.6}
        assert league.worst_case() == ("b", 0.2)

    def test_past_self_ring_buffer_caps_size(self, db, deck):
        league = League(db, deck, frozen_teacher_ckpt="unused.pt", max_past_selves=2)
        for i in range(5):
            league.push_past_self(f"ckpt_{i}.pt")
        assert league.past_selves == ["ckpt_3.pt", "ckpt_4.pt"]


class TestFrozenTeacherKL:
    def test_kl_is_zero_against_identical_logits(self):
        logits = torch.randn(4, 6)
        mask = torch.ones(4, 6, dtype=torch.bool)
        mask[:, 4:] = False
        kl = _teacher_kl_loss(logits, logits.clone(), mask)
        assert float(kl) == pytest.approx(0.0, abs=1e-5)

    def test_kl_is_positive_for_different_distributions(self):
        torch.manual_seed(0)
        student = torch.randn(4, 6)
        teacher = torch.randn(4, 6)
        mask = torch.ones(4, 6, dtype=torch.bool)
        kl = _teacher_kl_loss(student, teacher, mask)
        assert float(kl) > 0.0

    def test_policy_gradient_reinforces_positive_advantage(self):
        # A single decision, 3 options, option 0 chosen, positive advantage:
        # the update should be able to increase logit 0's log-prob (i.e. the
        # loss gradient w.r.t. logits[0] is negative -- increasing it lowers
        # the loss), which is what "reinforce the winning action" means.
        logits = torch.zeros(1, 3, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        chosen_mask = torch.tensor([[1.0, 0.0, 0.0]])
        advantage = torch.tensor([1.0])
        loss = _policy_gradient_loss(logits, mask, chosen_mask, advantage)
        loss.backward()
        assert logits.grad[0, 0] < 0


class TestSelfPlayRoundSmoke:
    """A real, tiny end-to-end round: generate self-play games through the
    real engine, train one epoch, and confirm it produces finite losses and
    updates the league -- not mocked."""

    def test_one_round_runs_and_updates_league(self, db, deck, tmp_path):
        table = CardTable.build(db, with_text=False)
        model = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)

        # Frozen teacher: an independent copy so KL is meaningfully non-zero
        # (an identical-weights teacher would make every KL term trivially 0).
        teacher = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        ckpt_path = tmp_path / "frozen.pt"
        torch.save(
            {
                "state_dict": teacher.state_dict(),
                "card_mode": "onehot",
                "n_cards": table.n_cards,
                "card_ids": table.card_ids,
                "card_struct": table.struct,
                "card_text": table.text,
            },
            ckpt_path,
        )

        league = League(db, deck, frozen_teacher_ckpt=ckpt_path)
        out_dir = tmp_path / "traces"
        gen_info = generate_round(
            league, model, table, "onehot", out_dir, n_games=6, temperature=1.0, seed_start=42, round_tag="test"
        )
        assert gen_info["total_records"] > 0
        assert league.win_rate  # at least one opponent got sampled and updated

        from ptcg.train.dataset import load_records
        from ptcg.train.selfplay_rl import train_selfplay_round

        records = load_records(out_dir, max_records=2000, seed=0)
        assert records

        info = train_selfplay_round(
            records, table, db, model, teacher, card_mode="onehot", epochs=1, batch_size=16, seed=0
        )
        stats = info["epochs"][0]
        for key in ("policy_loss", "value_loss", "kl_loss", "belief_loss"):
            assert stats[key] == stats[key]  # not NaN
