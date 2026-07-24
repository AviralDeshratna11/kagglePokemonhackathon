"""Week-1 evidence report: H1 (card-text vs one-hot generalization), belief
calibration, and a local bench of the trained BC policy against Week 0's
heuristic and the baseline panel.

Run after ``ptcg/train/bc.py`` has produced ``artifacts/bc_text.pt`` and
``artifacts/bc_onehot.pt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy, RandomPolicy
from ..agents.bc_policy import load_bc_policy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..agents.network import BeliefSetDMCLite, CardTable
from ..core.carddb import get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import available_decks, load_deck
from ..eval.arena import round_robin

__all__ = ["h1_generalization_probe", "belief_calibration", "local_bench", "main"]


# ---------------------------------------------------------------------------
# H1: does the text-embedding arm generalize to held-out cards better than
# the one-hot-ID arm?
# ---------------------------------------------------------------------------


def _in_training_card_ids(db) -> set[int]:
    ids: set[int] = set()
    for deck_name in available_decks():
        ids.update(load_deck(deck_name))
    return ids


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression, ``x`` already includes a bias column."""
    d = x.shape[1]
    a = x.T @ x + alpha * np.eye(d)
    b = x.T @ y
    return np.linalg.solve(a, b)


def h1_generalization_probe(text_ckpt: str | Path, onehot_ckpt: str | Path) -> dict[str, Any]:
    db = get_card_db()
    in_deck_ids = _in_training_card_ids(db)

    # Regression target: the card's strongest known attack, in raw damage
    # points -- not itself part of the id/text-only half of the embedding for
    # a fresh network, so a probe that predicts it above baseline is real
    # signal, not a restated input feature.
    targets: dict[int, float] = {}
    for c in db.all_cards():
        atks = db.attacks_of(c.card_id)
        if atks:
            targets[c.card_id] = float(max(a.damage for a in atks))

    results: dict[str, Any] = {}
    for mode, ckpt_path in (("text", text_ckpt), ("onehot", onehot_ckpt)):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        table = CardTable.build(db, with_text=(mode == "text"))
        model = BeliefSetDMCLite(card_mode=mode, n_cards=ckpt["n_cards"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            if mode == "text":
                aux = model.card_embedder.aux_only(table.text)
            else:
                aux = model.card_embedder.aux_only(torch.arange(table.n_cards + 1))
        aux = aux.numpy()

        train_idx, train_y, held_idx, held_y = [], [], [], []
        for cid, y in targets.items():
            idx = table.index_of(cid)
            (train_idx if cid in in_deck_ids else held_idx).append(idx)
            (train_y if cid in in_deck_ids else held_y).append(y)

        x_train = np.concatenate([aux[train_idx], np.ones((len(train_idx), 1), dtype=np.float32)], axis=1)
        x_held = np.concatenate([aux[held_idx], np.ones((len(held_idx), 1), dtype=np.float32)], axis=1)
        y_train = np.array(train_y, dtype=np.float32)
        y_held = np.array(held_y, dtype=np.float32)

        w = _ridge_fit(x_train, y_train, alpha=5.0)
        pred_train = x_train @ w
        pred_held = x_held @ w

        mae_train = float(np.mean(np.abs(pred_train - y_train)))
        mae_held = float(np.mean(np.abs(pred_held - y_held)))
        mae_held_baseline = float(np.mean(np.abs(y_held - y_train.mean())))

        results[mode] = {
            "n_train_cards": len(train_idx),
            "n_held_out_cards": len(held_idx),
            "probe_mae_on_training_cards": round(mae_train, 2),
            "probe_mae_on_held_out_cards": round(mae_held, 2),
            "mean_baseline_mae_on_held_out": round(mae_held_baseline, 2),
            "held_out_improvement_over_mean_baseline_pct": round(
                100.0 * (1.0 - mae_held / mae_held_baseline), 1
            ) if mae_held_baseline > 0 else None,
        }

    return results


# ---------------------------------------------------------------------------
# Belief calibration (already computed during training; surfaced here)
# ---------------------------------------------------------------------------


def belief_calibration(text_ckpt: str | Path, onehot_ckpt: str | Path) -> dict[str, Any]:
    out = {}
    for mode, path in (("text", text_ckpt), ("onehot", onehot_ckpt)):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epochs = ckpt["history"]["epochs"]
        out[mode] = {
            "first_epoch_val_belief_mae": epochs[0]["val_belief_mae"],
            "last_epoch_val_belief_mae": epochs[-1]["val_belief_mae"],
            "last_epoch_val_policy_top1_acc": epochs[-1]["val_policy_top1_acc"],
            "last_epoch_val_value_loss": epochs[-1]["val_value_loss"],
        }
    return out


# ---------------------------------------------------------------------------
# Local bench: BC vs heuristic vs baseline panel
# ---------------------------------------------------------------------------


def local_bench(text_ckpt: str | Path, games: int = 150, deck_name: str = "lucario_fighting") -> dict[str, Any]:
    """``round_robin``/``duel`` call each entry's factory to build an *agent*
    -- a callable ``obs -> list[int]`` -- not a bare ``Policy``. Every policy
    here must go through ``SafetyShell`` first (exactly like the live
    submission and ``ptcg/tools/bench.py``), both because that is what makes
    it callable and because an unshelled policy is not what would ever
    actually play a game."""
    db = get_card_db()
    deck = load_deck(deck_name)

    # Loaded once: ``duel()`` calls its factory fresh *every game*, and
    # rebuilding the card table (frozen MiniLM over the full pool) costs
    # ~24s. The policy object itself carries no cross-game state -- only the
    # SafetyShell/BudgetManager wrapping it needs to be fresh per game.
    bc_policy = load_bc_policy(text_ckpt, deck, db)

    def shell(policy) -> Any:
        return SafetyShell(policy, db, FallbackPolicy(deck), BudgetManager()).act

    entries = {
        "bc-text": (lambda: shell(bc_policy), deck),
        "heuristic": (lambda: shell(HeuristicPolicy(deck, db, HeuristicConfig())), deck),
        "greedy": (lambda: shell(GreedyAttackPolicy(deck)), deck),
        "first": (lambda: shell(FirstPolicy(deck)), deck),
        "random": (lambda: shell(RandomPolicy(deck, seed=7)), deck),
    }
    table, ladder = round_robin(entries, games=games, progress=lambda s: print(f"  {s}"))
    return {"deck": deck_name, "games_per_pair": games, "ladder": ladder.table(), **table.report()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-ckpt", default="artifacts/bc_text.pt")
    ap.add_argument("--onehot-ckpt", default="artifacts/bc_onehot.pt")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--out", default="artifacts/week1_report.json")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {}
    print("=== H1: card-text vs one-hot generalization to held-out cards ===")
    report["h1_generalization"] = h1_generalization_probe(args.text_ckpt, args.onehot_ckpt)
    print(json.dumps(report["h1_generalization"], indent=2))

    print("\n=== belief head calibration ===")
    report["belief_calibration"] = belief_calibration(args.text_ckpt, args.onehot_ckpt)
    print(json.dumps(report["belief_calibration"], indent=2))

    if not args.skip_bench:
        print("\n=== local bench: bc-text vs heuristic vs baselines ===")
        report["local_bench"] = local_bench(args.text_ckpt, games=args.games)
        print(json.dumps(report["local_bench"]["ladder"], indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
