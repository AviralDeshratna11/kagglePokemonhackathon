"""Behavior-cloning training loop for :class:`BeliefSetDMCLite`.

Three losses, summed with modest weights so the policy signal (the thing that
actually plays the game) dominates and value/belief stay auxiliary, matching
the plan:

* **policy** -- multi-positive cross-entropy over the masked option logits
  against every index in ``chosen`` (the engine's multi-select means more than
  one option can be "correct" for a single decision).
* **value** -- BCE of the sigmoid win-probability head against the recorded
  game outcome (1.0 win / 0.0 loss / 0.5 draw).
* **belief** -- cross-entropy (card type, energy type) + BCE (role rates)
  against the privileged self-play label, masked to only the examples where a
  label was actually available (``belief_valid``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..agents.network import BeliefSetDMCLite, CardTable, count_parameters
from ..core.carddb import get_card_db
from .dataset import TraceDataset, collate, load_paired_records, load_records

__all__ = ["train_bc", "evaluate", "BCTrainResult"]


def _policy_loss(logits: torch.Tensor, mask: torch.Tensor, chosen_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.masked_fill(~mask, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    denom = chosen_mask.sum(dim=-1).clamp(min=1.0)
    loss = -(logp * chosen_mask).sum(dim=-1) / denom
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = chosen_mask.gather(1, pred.unsqueeze(1)).squeeze(1)
        acc = correct.mean()
    return loss.mean(), acc


def _belief_loss(belief: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    valid = targets["belief_valid"]
    if valid.sum() == 0:
        return torch.tensor(0.0)
    ct = -(targets["belief_card_type"] * torch.log(belief["card_type"] + eps)).sum(dim=-1)
    et = -(targets["belief_energy_type"] * torch.log(belief["energy_type"] + eps)).sum(dim=-1)
    role = F.binary_cross_entropy(belief["role"], targets["belief_role"], reduction="none").mean(dim=-1)
    per_example = ct + et + role
    return (per_example * valid).sum() / valid.sum().clamp(min=1.0)


def _belief_mae(belief: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> float:
    valid = targets["belief_valid"]
    if valid.sum() == 0:
        return float("nan")
    mae = (belief["card_type"] - targets["belief_card_type"]).abs().sum(-1)
    mae = (mae * valid).sum() / valid.sum().clamp(min=1.0)
    return float(mae)


class BCTrainResult:
    def __init__(self) -> None:
        self.epochs: list[dict[str, float]] = []

    def as_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs}


def evaluate(model: BeliefSetDMCLite, loader: DataLoader, card_mode: str, value_weight: float, belief_weight: float) -> dict[str, float]:
    model.eval()
    tot = {"policy": 0.0, "value": 0.0, "belief": 0.0, "acc": 0.0, "belief_mae": 0.0, "n": 0, "n_belief": 0}
    with torch.no_grad():
        for batch, targets in loader:
            out = model(batch)
            ploss, acc = _policy_loss(out["policy_logits"], batch.option_mask, targets["chosen_mask"])
            vloss = F.binary_cross_entropy(out["value"], targets["value"])
            bloss = _belief_loss(out["belief"], targets)
            bsz = targets["value"].shape[0]
            tot["policy"] += float(ploss) * bsz
            tot["value"] += float(vloss) * bsz
            tot["belief"] += float(bloss) * bsz
            tot["acc"] += float(acc) * bsz
            mae = _belief_mae(out["belief"], targets)
            if mae == mae:  # not NaN
                tot["belief_mae"] += mae * float(targets["belief_valid"].sum())
                tot["n_belief"] += float(targets["belief_valid"].sum())
            tot["n"] += bsz
    n = max(1, tot["n"])
    return {
        "policy_loss": tot["policy"] / n,
        "value_loss": tot["value"] / n,
        "belief_loss": tot["belief"] / n,
        "policy_top1_acc": tot["acc"] / n,
        "belief_mae": tot["belief_mae"] / tot["n_belief"] if tot["n_belief"] else float("nan"),
        "total_loss": tot["policy"] / n + value_weight * tot["value"] / n + belief_weight * tot["belief"] / n,
    }


def _spr_loss(model: BeliefSetDMCLite, cur_batch, cur_targets, next_batch) -> torch.Tensor:
    """Week 7 SPR-lite: predict the next-decision trunk embedding from the
    current trunk + the chosen action's feature vector, trained against a
    stop-gradient target -- see ``DynamicsHead``'s docstring for why this
    shape (BYOL-style, no negative samples needed)."""
    cur_out = model(cur_batch)
    with torch.no_grad():
        next_trunk = model(next_batch)["trunk"].detach()

    chosen_mask = cur_targets["chosen_mask"]
    denom = chosen_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    action_feat = (cur_batch.option_features * chosen_mask.unsqueeze(-1)).sum(dim=1) / denom

    pred_next_trunk = model.dynamics_head(cur_out["trunk"], action_feat)
    return 1.0 - F.cosine_similarity(pred_next_trunk, next_trunk, dim=-1).mean()


def train_bc(
    records: list[dict[str, Any]],
    table: CardTable,
    db,
    card_mode: str,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    value_weight: float = 0.5,
    belief_weight: float = 0.3,
    seed: int = 0,
    use_spr: bool = False,
    spr_pairs: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
    spr_weight: float = 0.1,
) -> tuple[BeliefSetDMCLite, BCTrainResult]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    n_val = max(1, int(len(records) * val_frac))
    perm = torch.randperm(len(records), generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    train_ds = TraceDataset(train_records, table, db, card_mode)
    val_ds = TraceDataset(val_records, table, db, card_mode)

    def _collate(items):
        return collate(items, card_mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = BeliefSetDMCLite(card_mode=card_mode, n_cards=table.n_cards, use_spr=use_spr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    spr_gen = None
    if use_spr and spr_pairs and spr_pairs[0]:
        cur_records, next_records = spr_pairs
        cur_ds = TraceDataset(cur_records, table, db, card_mode)
        next_ds = TraceDataset(next_records, table, db, card_mode)
        # Pairs were shuffled together at the list level (see
        # load_paired_records); both loaders must stay unshuffled here or
        # cur[i]/next[i] would no longer be the same original pair.
        cur_loader = DataLoader(cur_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        next_loader = DataLoader(next_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

        def _spr_batches():
            while True:
                for (cb, ct), (nb, _) in zip(cur_loader, next_loader):
                    yield cb, ct, nb

        spr_gen = _spr_batches()

    result = BCTrainResult()
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = {"policy": 0.0, "value": 0.0, "belief": 0.0, "spr": 0.0, "n": 0}
        for batch, targets in train_loader:
            opt.zero_grad()
            out = model(batch)
            ploss, _ = _policy_loss(out["policy_logits"], batch.option_mask, targets["chosen_mask"])
            vloss = F.binary_cross_entropy(out["value"], targets["value"])
            bloss = _belief_loss(out["belief"], targets)
            loss = ploss + value_weight * vloss + belief_weight * bloss

            sloss_val = 0.0
            if spr_gen is not None:
                cb, ct, nb = next(spr_gen)
                sloss = _spr_loss(model, cb, ct, nb)
                loss = loss + spr_weight * sloss
                sloss_val = sloss.item()

            loss.backward()
            opt.step()

            bsz = targets["value"].shape[0]
            running["policy"] += ploss.item() * bsz
            running["value"] += vloss.item() * bsz
            running["belief"] += bloss.item() * bsz
            running["spr"] += sloss_val * bsz
            running["n"] += bsz

        val_metrics = evaluate(model, val_loader, card_mode, value_weight, belief_weight)
        n = max(1, running["n"])
        epoch_stats = {
            "epoch": epoch,
            "train_policy_loss": running["policy"] / n,
            "train_value_loss": running["value"] / n,
            "train_belief_loss": running["belief"] / n,
            "train_spr_loss": running["spr"] / n,
            "seconds": round(time.time() - t0, 2),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        result.epochs.append(epoch_stats)
        print(
            f"[{card_mode}] epoch {epoch}: train policy={epoch_stats['train_policy_loss']:.4f} "
            f"val policy={val_metrics['policy_loss']:.4f} val_acc={val_metrics['policy_top1_acc']:.3f} "
            f"val_value={val_metrics['value_loss']:.4f} val_belief_mae={val_metrics['belief_mae']:.4f} "
            f"spr={epoch_stats['train_spr_loss']:.4f} ({epoch_stats['seconds']}s)"
        )

    return model, result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default="artifacts/traces/selfplay_v1", help="comma-separated trace directories")
    ap.add_argument("--card-mode", choices=["text", "onehot"], default="text")
    ap.add_argument("--max-records", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-spr", action="store_true", help="Week 7: add the self-predictive auxiliary loss")
    ap.add_argument("--spr-weight", type=float, default=0.1)
    ap.add_argument("--spr-max-pairs", type=int, default=40000)
    args = ap.parse_args(argv)

    db = get_card_db()
    trace_dirs = [d.strip() for d in args.traces.split(",") if d.strip()]
    print(f"loading up to {args.max_records} records from {trace_dirs} ...")
    records = load_records(trace_dirs, max_records=args.max_records, seed=args.seed)
    print(f"loaded {len(records)} records")

    spr_pairs = None
    if args.use_spr:
        cur, nxt = load_paired_records(trace_dirs, max_pairs=args.spr_max_pairs, seed=args.seed)
        print(f"loaded {len(cur)} SPR (current, next) pairs")
        spr_pairs = (cur, nxt)

    table = CardTable.build(db, with_text=(args.card_mode == "text"))
    model, result = train_bc(
        records, table, db, args.card_mode,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        use_spr=args.use_spr, spr_pairs=spr_pairs, spr_weight=args.spr_weight,
    )
    print("params:", count_parameters(model))

    out = Path(args.out) if args.out else Path(f"artifacts/bc_{args.card_mode}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "card_mode": args.card_mode,
            "n_cards": table.n_cards,
            "card_ids": table.card_ids,
            # Baked in so a live agent never has to rebuild the table (and,
            # for text mode, never has to run MiniLM or even import
            # sentence-transformers) at submission runtime.
            "card_struct": table.struct,
            "card_text": table.text,
            "history": result.as_dict(),
            "use_spr": args.use_spr,
        },
        out,
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
