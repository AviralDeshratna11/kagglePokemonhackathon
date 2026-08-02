"""Week-7 test suite: SPR-lite auxiliary loss, MAP-Elites-lite archive,
Double-Oracle-lite meta-game solver, self-play refinement on a new deck, and
the pull-episode download timeout safeguard.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ptcg.agents.network import BeliefSetDMCLite, CardTable
from ptcg.core.carddb import get_card_db
from ptcg.core.trace import TraceWriter
from ptcg.decks.registry import load_deck
from ptcg.train.dataset import load_paired_records, load_records
from ptcg.tools.deck_search import (
    build_crustle_test_deck,
    double_oracle_lite_solve,
    map_elites_search,
    replicator_dynamics,
)
from ptcg.tools.pull_episodes import _download_with_timeout


@pytest.fixture(scope="module")
def db():
    return get_card_db()


class TestSPRPairing:
    def test_pairs_preserve_order_and_drop_last(self, tmp_path):
        d = tmp_path / "traces"
        w = TraceWriter(d, tag="t")
        for i in range(5):
            w({"player": 0, "chosen": [0], "features": [[float(i)]], "candidate_card_ids": [1]})
        w.finish({"outcome": 1.0})

        cur, nxt = load_paired_records(d, seed=0)
        assert len(cur) == 4  # 5 decisions -> 4 adjacent pairs
        # Every pair's "next" step index is exactly "current" step index + 1.
        by_step = {c["step"]: n["step"] for c, n in zip(cur, nxt)}
        for c_step, n_step in by_step.items():
            assert n_step == c_step + 1

    def test_empty_dir_returns_empty(self, tmp_path):
        cur, nxt = load_paired_records(tmp_path / "nothing")
        assert cur == [] and nxt == []


class TestSPRTraining:
    def test_dynamics_head_only_built_when_requested(self, db):
        table = CardTable.build(db, with_text=False)
        plain = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, use_spr=False)
        spr = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, use_spr=True)
        assert plain.dynamics_head is None
        assert spr.dynamics_head is not None
        assert sum(p.numel() for p in spr.parameters()) > sum(p.numel() for p in plain.parameters())

    def test_spr_checkpoint_roundtrips(self, db, tmp_path):
        from ptcg.agents.bc_policy import load_bc_policy

        table = CardTable.build(db, with_text=False)
        model = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards, use_spr=True)
        ckpt_path = tmp_path / "spr.pt"
        torch.save(
            {
                "state_dict": model.state_dict(), "card_mode": "onehot", "n_cards": table.n_cards,
                "card_ids": table.card_ids, "card_struct": table.struct, "card_text": table.text,
                "use_spr": True,
            },
            ckpt_path,
        )
        deck = load_deck("lucario_fighting")
        policy = load_bc_policy(ckpt_path, deck, db)
        assert policy.model.use_spr is True
        assert policy.model.dynamics_head is not None

    def test_train_bc_with_spr_produces_finite_loss(self, db):
        from ptcg.train.bc import train_bc

        table = CardTable.build(db, with_text=False)
        records = load_records("artifacts/traces/selfplay_v1", max_records=200, seed=0)
        cur, nxt = load_paired_records("artifacts/traces/selfplay_v1", max_pairs=200, seed=0)
        assert records and cur

        model, result = train_bc(
            records, table, db, "onehot", epochs=1, batch_size=32, seed=0,
            use_spr=True, spr_pairs=(cur, nxt), spr_weight=0.1,
        )
        stats = result.epochs[0]
        assert stats["train_spr_loss"] == stats["train_spr_loss"]  # not NaN
        assert 0.0 <= stats["train_spr_loss"] <= 2.1  # cosine-similarity loss is bounded in [0, 2]


class TestDeckAgnosticSelfPlay:
    def test_selfplay_machinery_runs_on_a_new_deck(self, db, tmp_path):
        """The exact pattern test_week2.py already proves for
        lucario_fighting, reused verbatim on rocket_mewtwo -- proving the
        self-play machinery was already deck-agnostic, not rebuilt for
        this specific archetype."""
        from ptcg.agents.network import BeliefSetDMCLite as Net
        from ptcg.train.league import League
        from ptcg.tools.generate_league_selfplay import generate_round

        deck = load_deck("rocket_mewtwo")
        table = CardTable.build(db, with_text=False)
        model = Net(card_mode="onehot", n_cards=table.n_cards)

        ckpt_path = tmp_path / "frozen.pt"
        torch.save(
            {
                "state_dict": model.state_dict(), "card_mode": "onehot", "n_cards": table.n_cards,
                "card_ids": table.card_ids, "card_struct": table.struct, "card_text": table.text,
            },
            ckpt_path,
        )

        league = League(db, deck, frozen_teacher_ckpt=ckpt_path)
        out_dir = tmp_path / "traces"
        info = generate_round(league, model, table, "onehot", out_dir, n_games=4, temperature=1.0, seed_start=1, round_tag="t")
        assert info["total_records"] > 0
        assert league.win_rate


class TestMapElites:
    def test_archive_entries_are_all_legal(self, db):
        seed_decks = {
            "lucario_fighting": (load_deck("lucario_fighting"), {333, 678}),
            "rocket_mewtwo": (load_deck("rocket_mewtwo"), {431}),
        }
        info = map_elites_search(seed_decks, db, rounds=1, mutations_per_round=1, screen_games=2, confirm_games=2, seed=0)
        assert info["archive"]
        for entry in info["archive"]:
            assert len(entry["deck"]) == 60
            assert db.validate_deck(entry["deck"]) == []

    def test_descriptor_is_deterministic_bucketing(self):
        from ptcg.tools.deck_search import _descriptor

        key, worst, avg = _descriptor({"a": 0.3, "b": 0.7})
        assert key == (3, 5)
        assert worst == 0.3
        assert avg == pytest.approx(0.5)


class TestReplicatorDynamics:
    def test_dominant_strategy_converges(self):
        payoff = np.array([[0.5, 0.9, 0.9], [0.1, 0.5, 0.5], [0.1, 0.5, 0.5]])
        x = replicator_dynamics(payoff)
        assert x[0] > 0.9
        assert abs(x.sum() - 1.0) < 1e-6

    def test_cyclic_game_converges_near_uniform(self):
        rps = np.array([[0.5, 0.9, 0.1], [0.1, 0.5, 0.9], [0.9, 0.1, 0.5]])
        x = replicator_dynamics(rps)
        assert all(abs(v - 1 / 3) < 0.05 for v in x)

    def test_double_oracle_lite_solve_end_to_end(self, db):
        archive = [
            {"name": "lucario", "deck": load_deck("lucario_fighting"), "worst": 0.5, "avg": 0.5},
            {"name": "rocket", "deck": load_deck("rocket_mewtwo"), "worst": 0.5, "avg": 0.5},
        ]
        info = double_oracle_lite_solve(archive, db, games=2, seed=0, top_n=2)
        assert len(info["payoff_matrix"]) == 2
        assert info["recommended_deck_name"] in {"lucario", "rocket"}
        assert sum(m["weight"] for m in info["equilibrium_mixture"]) == pytest.approx(1.0, abs=1e-3)


class TestDownloadTimeout:
    def test_returns_true_on_fast_call(self, tmp_path):
        class _FastApi:
            def dataset_download_file(self, slug, fname, path, force, quiet):
                pass

        assert _download_with_timeout(_FastApi(), "slug", "f.json", tmp_path, timeout=2.0) is True

    def test_returns_false_on_timeout(self, tmp_path):
        class _SlowApi:
            def dataset_download_file(self, slug, fname, path, force, quiet):
                time.sleep(5)

        assert _download_with_timeout(_SlowApi(), "slug", "f.json", tmp_path, timeout=0.2) is False

    def test_reraises_real_errors(self, tmp_path):
        class _FailingApi:
            def dataset_download_file(self, slug, fname, path, force, quiet):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            _download_with_timeout(_FailingApi(), "slug", "f.json", tmp_path, timeout=2.0)
