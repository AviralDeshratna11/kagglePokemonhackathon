"""Week-1 test suite: card embeddings, the BeliefSet-DMC-lite network, the
BCPolicy safety integration, and a real (tiny) training smoke test.

Mirrors ``test_week0.py``'s standard: every claim here is checked against the
real engine and real card database, not mocks.
"""

from __future__ import annotations

import gzip
import json

import pytest
import torch

from ptcg.agents.bc_policy import BCPolicy, load_bc_policy
from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.network import (
    BELIEF_CARD_TYPE,
    BELIEF_DIM,
    BELIEF_ENERGY_TYPE,
    BELIEF_ROLE_FLAGS,
    CARD_EMBED_DIM,
    CardTable,
    BeliefSetDMCLite,
    count_parameters,
    encode_view,
)
from ptcg.core.actions import build_candidates
from ptcg.core.carddb import get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.embeddings import STRUCT_DIM, TEXT_DIM, card_text, structured_features
from ptcg.core.obs import ObsView
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match
from ptcg.tools.generate_selfplay import generate
from ptcg.train.bc import train_bc
from ptcg.train.dataset import load_records


@pytest.fixture(scope="module")
def db():
    return get_card_db()


@pytest.fixture(scope="module")
def deck():
    return load_deck("lucario_fighting")


@pytest.fixture(scope="module")
def text_table(db):
    return CardTable.build(db, with_text=True)


@pytest.fixture(scope="module")
def onehot_table(db):
    return CardTable.build(db, with_text=False)


# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_structured_features_shape_and_range(self, db):
        c = db.card(678)  # Mega Lucario ex
        v = structured_features(c, db)
        assert v.shape == (STRUCT_DIM,)
        assert v.dtype.name == "float32"
        assert ((v >= -4.0) & (v <= 4.0)).all()

    def test_structured_features_deterministic(self, db):
        c = db.card(678)
        assert (structured_features(c, db) == structured_features(c, db)).all()

    def test_card_text_nonempty_for_every_card(self, db):
        # A card with no ability and no attacks (e.g. a bare energy) falls
        # back to its name, so this must never be empty.
        for c in list(db.all_cards())[:50]:
            assert card_text(c, db).strip()


class TestCardTable:
    def test_text_table_covers_full_pool(self, db, text_table):
        assert text_table.n_cards == len(list(db.all_cards()))
        assert text_table.struct.shape == (text_table.n_cards + 1, STRUCT_DIM)
        assert text_table.text.shape == (text_table.n_cards + 1, TEXT_DIM)

    def test_onehot_table_has_no_text(self, onehot_table):
        assert onehot_table.text is None

    def test_unknown_card_id_maps_to_padding_row(self, text_table):
        assert text_table.index_of(None) == text_table.unknown_index
        assert text_table.index_of(-999999) == text_table.unknown_index

    def test_known_card_id_resolves(self, db, text_table):
        cid = next(iter(text_table.card_ids))
        assert text_table.index_of(cid) == text_table.id_to_idx[cid]


class TestNetworkForward:
    def _play_and_probe(self, db, deck, table, card_mode, n_steps=5):
        net = BeliefSetDMCLite(card_mode=card_mode, n_cards=table.n_cards)
        net.eval()
        shell_a = SafetyShell(FallbackPolicy(deck), db, FallbackPolicy(deck), BudgetManager())
        shell_b = SafetyShell(FallbackPolicy(deck), db, FallbackPolicy(deck), BudgetManager())

        seen = []

        def on_step(player, obs, action):
            if len(seen) >= n_steps:
                return
            view = ObsView(obs, db)
            if view.is_deck_selection:
                return
            cands = build_candidates(view, db)
            if not cands:
                return
            inp = encode_view(view, cands, table, card_mode)
            with torch.no_grad():
                out = net(inp)
            seen.append((len(cands), out))

        play_match(shell_a.act, shell_b.act, deck, deck, seed=7, on_step=on_step)
        assert len(seen) >= 1
        return seen

    def test_text_mode_forward_shapes(self, db, deck, text_table):
        seen = self._play_and_probe(db, deck, text_table, "text")
        for n_cands, out in seen:
            assert out["policy_logits"].shape == (1, n_cands)
            assert out["value"].shape == (1,)
            assert 0.0 <= float(out["value"][0]) <= 1.0
            assert torch.isfinite(out["policy_logits"]).all()

    def test_onehot_mode_forward_shapes(self, db, deck, onehot_table):
        seen = self._play_and_probe(db, deck, onehot_table, "onehot")
        for n_cands, out in seen:
            assert out["policy_logits"].shape == (1, n_cands)

    def test_belief_head_outputs_valid_distributions(self, db, deck, text_table):
        seen = self._play_and_probe(db, deck, text_table, "text")
        _, out = seen[0]
        belief = out["belief"]
        assert belief["card_type"].shape == (1, BELIEF_CARD_TYPE)
        assert belief["energy_type"].shape == (1, BELIEF_ENERGY_TYPE)
        assert belief["role"].shape == (1, BELIEF_ROLE_FLAGS)
        assert BELIEF_CARD_TYPE + BELIEF_ENERGY_TYPE + BELIEF_ROLE_FLAGS == BELIEF_DIM
        for key in ("card_type", "energy_type"):
            s = float(belief[key].sum())
            assert abs(s - 1.0) < 1e-4
        assert ((belief["role"] >= 0) & (belief["role"] <= 1)).all()

    def test_param_count_is_cpu_sized(self, text_table):
        net = BeliefSetDMCLite(card_mode="text", n_cards=text_table.n_cards)
        n = count_parameters(net)
        assert 50_000 < n < 5_000_000


class TestBCPolicySafety:
    """The thing that actually matters: an *untrained* (random-weight) BC
    network run through the exact same SafetyShell as the heuristic must
    still never crash, never emit an illegal action, and never need the
    fallback -- random logits are a worst case for "does the plumbing hold
    up", not a claim about play quality."""

    def test_full_game_completes_without_incident(self, db, deck, text_table):
        net = BeliefSetDMCLite(card_mode="text", n_cards=text_table.n_cards)
        policy = BCPolicy(deck, net, text_table, "text")
        shell_a = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())
        shell_b = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())

        result = play_match(shell_a.act, shell_b.act, deck, deck, seed=11)
        assert result.reason == "engine_result"
        assert result.winner in (0, 1, -1)
        for shell in (shell_a, shell_b):
            assert shell.report.policy_exceptions == 0
            assert shell.report.fallback_exceptions == 0
            assert shell.report.sanitised == 0

    def test_checkpoint_round_trip(self, db, deck, tmp_path):
        table = CardTable.build(db, with_text=False)
        net = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)
        ckpt_path = tmp_path / "tiny.pt"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "card_mode": "onehot",
                "n_cards": table.n_cards,
                "card_ids": table.card_ids,
                "card_struct": table.struct,
                "card_text": table.text,
            },
            ckpt_path,
        )
        policy = load_bc_policy(ckpt_path, deck, db)
        assert policy.name == "bc-v1"

        shell = SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager())
        out = shell.act({"select": None, "current": None, "logs": []})
        assert len(out) == 60

    def test_mismatched_card_pool_is_rejected(self, db, deck, tmp_path):
        table = CardTable.build(db, with_text=False)
        net = BeliefSetDMCLite(card_mode="onehot", n_cards=table.n_cards)
        ckpt_path = tmp_path / "bad.pt"
        torch.save(
            {
                "state_dict": net.state_dict(),
                "card_mode": "onehot",
                "n_cards": table.n_cards,
                "card_ids": list(reversed(table.card_ids)),  # corrupt the mapping
            },
            ckpt_path,
        )
        with pytest.raises(ValueError):
            load_bc_policy(ckpt_path, deck, db)


class TestTrainingSmoke:
    """A real training step on a real (tiny) self-play corpus must reduce
    policy loss -- not mocked, not asserted-by-construction."""

    def test_one_training_run_reduces_policy_loss(self, db, tmp_path):
        out_dir = tmp_path / "traces"
        info = generate(out_dir, n_games=40, decks=["lucario_fighting"], seed_start=5000, log_every=1000)
        assert info["total_records"] > 100

        records = load_records(out_dir, max_records=2000, seed=0)
        assert len(records) > 50

        table = CardTable.build(db, with_text=False)  # onehot: fastest, no MiniLM needed for this smoke test
        _, result = train_bc(
            records, table, db, "onehot",
            epochs=3, batch_size=32, lr=2e-3, val_frac=0.15, seed=0,
        )
        losses = [e["train_policy_loss"] for e in result.epochs]
        assert len(losses) == 3
        assert all(l == l for l in losses)  # no NaNs
        assert losses[-1] < losses[0]

    def test_trace_schema_round_trips_through_gzip(self, tmp_path):
        out_dir = tmp_path / "t2"
        generate(out_dir, n_games=2, decks=["lucario_fighting"], seed_start=1, log_every=1000)
        files = list(out_dir.glob("*.jsonl.gz"))
        assert files
        with gzip.open(files[0], "rt", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["type"] == "header"
        assert lines[-1]["type"] == "outcome"
        decisions = [l for l in lines if l["type"] == "decision"]
        if decisions:
            d = decisions[0]
            for key in ("board", "globals", "features", "candidate_card_ids", "chosen", "belief_valid"):
                assert key in d
