"""Knowledge distillation: train smaller ``BeliefSetDMCLite`` variants
against ``bc_text.pt`` (or any checkpoint) as a frozen teacher, to find the
actual efficiency/quality frontier instead of picking one size blind.

Every head is a *blend* of two losses: a distillation term (the student
matches the teacher's own output distribution -- this is what makes it real
knowledge distillation, not "train smaller from scratch") and a small-weight
ground-truth term (the student also gets pulled toward the recorded label
directly), so distillation can't drift arbitrarily far from what the data
itself says. ``true_weight`` controls the blend; 0.3 by default means
teacher-matching dominates.
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
from .bc import _belief_loss
from .dataset import TraceDataset, collate, load_records

__all__ = ["SIZE_CONFIGS", "distill", "main"]

# Full size (Week 1): card_embed_dim=128, zone_hidden=64, trunk_dim=256,
# option_hidden=96 -- ~436K params in text mode. These targets progressively
# shrink every dimension, keeping zone_hidden divisible by ZONE_HEADS=4.
SIZE_CONFIGS: dict[str, dict[str, int]] = {
    "medium": {
        "card_embed_dim": 64, "card_struct_hidden": 16, "card_text_hidden": 32,
        "zone_hidden": 32, "hist_hidden": 16, "trunk_dim": 128, "option_hidden": 48,
    },
    "small": {
        "card_embed_dim": 32, "card_struct_hidden": 8, "card_text_hidden": 16,
        "zone_hidden": 16, "hist_hidden": 8, "trunk_dim": 64, "option_hidden": 24,
    },
}


def _policy_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    chosen_mask: torch.Tensor,
    temperature: float = 2.0,
    true_weight: float = 0.3,
) -> torch.Tensor:
    s = student_logits.masked_fill(~mask, -1e9)
    t = teacher_logits.masked_fill(~mask, -1e9)

    student_logp_soft = F.log_softmax(s / temperature, dim=-1)
    teacher_p_soft = F.softmax(t / temperature, dim=-1)
    kd = -(teacher_p_soft * student_logp_soft).sum(dim=-1).mean() * (temperature ** 2)

    student_logp = F.log_softmax(s, dim=-1)
    denom = chosen_mask.sum(dim=-1).clamp(min=1.0)
    true_ce = -((student_logp * chosen_mask).sum(dim=-1) / denom).mean()

    return (1 - true_weight) * kd + true_weight * true_ce


def _value_kd_loss(student_value: torch.Tensor, teacher_value: torch.Tensor, true_outcome: torch.Tensor, true_weight: float = 0.3) -> torch.Tensor:
    kd = F.mse_loss(student_value, teacher_value.detach())
    true = F.binary_cross_entropy(student_value, true_outcome)
    return (1 - true_weight) * kd + true_weight * true


def _belief_kd_loss(student_belief: dict[str, torch.Tensor], teacher_belief: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], true_weight: float = 0.3, eps: float = 1e-6) -> torch.Tensor:
    ct_kd = (teacher_belief["card_type"].detach() * (torch.log(teacher_belief["card_type"].detach() + eps) - torch.log(student_belief["card_type"] + eps))).sum(-1).mean()
    et_kd = (teacher_belief["energy_type"].detach() * (torch.log(teacher_belief["energy_type"].detach() + eps) - torch.log(student_belief["energy_type"] + eps))).sum(-1).mean()
    role_kd = F.mse_loss(student_belief["role"], teacher_belief["role"].detach())
    kd = ct_kd + et_kd + role_kd
    true = _belief_loss(student_belief, targets, eps=eps)
    return (1 - true_weight) * kd + true_weight * true


def distill(
    records: list[dict[str, Any]],
    table: CardTable,
    db,
    teacher: BeliefSetDMCLite,
    dims: dict[str, int],
    card_mode: str = "text",
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    value_weight: float = 0.5,
    belief_weight: float = 0.3,
    true_weight: float = 0.3,
    temperature: float = 2.0,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[BeliefSetDMCLite, dict[str, Any]]:
    torch.manual_seed(seed)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

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

    student = BeliefSetDMCLite(card_mode=card_mode, n_cards=table.n_cards, dims=dims)
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    epoch_stats: list[dict[str, float]] = []
    for epoch in range(epochs):
        student.train()
        t0 = time.time()
        running = {"policy": 0.0, "value": 0.0, "belief": 0.0, "n": 0}
        for batch, targets in train_loader:
            opt.zero_grad()
            out = student(batch)
            with torch.no_grad():
                teacher_out = teacher(batch)

            ploss = _policy_kd_loss(out["policy_logits"], teacher_out["policy_logits"], batch.option_mask, targets["chosen_mask"], temperature, true_weight)
            vloss = _value_kd_loss(out["value"], teacher_out["value"], targets["value"], true_weight)
            bloss = _belief_kd_loss(out["belief"], teacher_out["belief"], targets, true_weight)

            loss = ploss + value_weight * vloss + belief_weight * bloss
            loss.backward()
            opt.step()

            bsz = targets["value"].shape[0]
            running["policy"] += ploss.item() * bsz
            running["value"] += vloss.item() * bsz
            running["belief"] += bloss.item() * bsz
            running["n"] += bsz

        # Validation: measure the student against ground truth (not the
        # teacher) so the reported numbers are directly comparable to bc.py's.
        student.eval()
        val_correct = 0.0
        val_n = 0
        with torch.no_grad():
            for batch, targets in val_loader:
                out = student(batch)
                logits = out["policy_logits"].masked_fill(~batch.option_mask, -1e9)
                pred = logits.argmax(dim=-1)
                correct = targets["chosen_mask"].gather(1, pred.unsqueeze(1)).squeeze(1)
                val_correct += float(correct.sum())
                val_n += correct.shape[0]

        n = max(1, running["n"])
        stats = {
            "epoch": epoch,
            "train_policy_kd_loss": running["policy"] / n,
            "train_value_kd_loss": running["value"] / n,
            "train_belief_kd_loss": running["belief"] / n,
            "val_policy_top1_acc": val_correct / max(1, val_n),
            "seconds": round(time.time() - t0, 2),
        }
        epoch_stats.append(stats)
        print(
            f"  [distill] epoch {epoch}: policy_kd={stats['train_policy_kd_loss']:.4f} "
            f"value_kd={stats['train_value_kd_loss']:.4f} val_acc={stats['val_policy_top1_acc']:.3f} "
            f"({stats['seconds']}s)"
        )

    return student, {"epochs": epoch_stats, "dims": dims, "param_count": count_parameters(student)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-ckpt", default="artifacts/bc_text.pt")
    ap.add_argument("--traces", default="artifacts/traces/selfplay_v1")
    ap.add_argument("--size", choices=list(SIZE_CONFIGS), required=True)
    ap.add_argument("--max-records", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--true-weight", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    db = get_card_db()
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    card_mode = teacher_ckpt["card_mode"]

    table = CardTable.from_tensors(teacher_ckpt["card_ids"], teacher_ckpt["card_struct"], teacher_ckpt.get("card_text"))
    teacher = BeliefSetDMCLite(
        card_mode=card_mode, n_cards=teacher_ckpt["n_cards"], dims=teacher_ckpt.get("dims"),
        use_spr=teacher_ckpt.get("use_spr", False),
    )
    teacher.load_state_dict(teacher_ckpt["state_dict"])

    print(f"loading up to {args.max_records} records from {args.traces} ...")
    records = load_records(args.traces, max_records=args.max_records, seed=args.seed)
    print(f"loaded {len(records)} records; teacher params: {count_parameters(teacher)}")

    dims = SIZE_CONFIGS[args.size]
    student, info = distill(
        records, table, db, teacher, dims, card_mode=card_mode,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        true_weight=args.true_weight, temperature=args.temperature, seed=args.seed,
    )
    print(f"student ({args.size}) params: {info['param_count']}")

    out = Path(args.out) if args.out else Path(f"artifacts/distill_{args.size}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "card_mode": card_mode,
            "n_cards": teacher_ckpt["n_cards"],
            "card_ids": teacher_ckpt["card_ids"],
            "card_struct": teacher_ckpt["card_struct"],
            "card_text": teacher_ckpt.get("card_text"),
            "dims": dims,
            "distill_info": info,
        },
        out,
    )
    print(f"saved {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
