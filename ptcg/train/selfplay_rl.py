"""The Week-2 training step: REINFORCE-with-baseline policy gradient +
value regression + frozen-teacher KL, replacing Week 1's cross-entropy
imitation loss. Reuses ``ptcg/train/dataset.py`` entirely unchanged --
``ptcg/tools/generate_league_selfplay.py`` writes the identical trace schema,
so the same loader/collate pipeline applies to self-play data too.

The imitation-vs-self-play distinction lives entirely in the policy loss:
Week 1 pushed probability mass onto whatever the *heuristic* chose. Here the
network reinforces whatever *it itself* chose, weighted by how that game
actually turned out (advantage = realized return minus the value baseline) --
so it can learn from its own outcomes instead of being capped at imitating a
fixed teacher.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..agents.network import BeliefSetDMCLite
from .bc import _belief_loss, _belief_mae
from .dataset import TraceDataset, collate

__all__ = ["train_selfplay_round"]


def _policy_gradient_loss(
    logits: torch.Tensor, mask: torch.Tensor, chosen_mask: torch.Tensor, advantage: torch.Tensor
) -> torch.Tensor:
    """``advantage`` is per-example (b,), broadcast over the chosen options in
    that decision. Multi-select decisions average log-prob over every chosen
    index, matching ``ptcg/train/bc.py``'s multi-positive convention."""
    logits = logits.masked_fill(~mask, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    denom = chosen_mask.sum(dim=-1).clamp(min=1.0)
    mean_logp = (logp * chosen_mask).sum(dim=-1) / denom
    return -(advantage.detach() * mean_logp).mean()


def _teacher_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(student || teacher) over the masked option set, per decision, then
    averaged. Padding columns are masked out of both distributions identically
    (same batch, same option ordering), so the comparison is apples-to-apples."""
    s_logits = student_logits.masked_fill(~mask, -1e9)
    t_logits = teacher_logits.masked_fill(~mask, -1e9)
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    s_p = s_logp.exp()
    kl = (s_p * (s_logp - t_logp)).sum(dim=-1)
    return kl.mean()


def train_selfplay_round(
    records: list[dict[str, Any]],
    table,
    db,
    model: BeliefSetDMCLite,
    teacher_model: BeliefSetDMCLite,
    card_mode: str = "text",
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 2e-4,
    value_weight: float = 0.5,
    kl_weight: float = 1.0,
    belief_weight: float = 0.3,
    grad_clip_norm: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """``grad_clip_norm`` and per-batch advantage normalization are the fix
    for a real failure observed in the first Week-2 run: an unclipped,
    unnormalized REINFORCE update spiked (policy loss -0.78 -> -62.8, KL 0.81
    -> 2.73 in one epoch) and collapsed the deterministic policy to near-zero
    win rate against everything, even though the *stochastic* training-time
    win-rate table looked fine -- high-temperature exploration noise can win
    games through randomness even when the underlying argmax policy has
    degenerated. ``kl_weight`` is also raised (0.3 -> 1.0 default) and ``lr``
    lowered (5e-4 -> 2e-4) as further margin against the same failure mode.
    """
    torch.manual_seed(seed)
    teacher_model.eval()

    ds = TraceDataset(records, table, db, card_mode)

    def _collate(items):
        return collate(items, card_mode)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_stats: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running = {"policy": 0.0, "value": 0.0, "kl": 0.0, "belief": 0.0, "grad_norm": 0.0, "n": 0}
        for batch, targets in loader:
            opt.zero_grad()
            out = model(batch)
            with torch.no_grad():
                teacher_out = teacher_model(batch)

            value = out["value"]
            advantage = targets["value"] - value.detach()
            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            advantage = advantage.clamp(-3.0, 3.0)

            ploss = _policy_gradient_loss(out["policy_logits"], batch.option_mask, targets["chosen_mask"], advantage)
            vloss = F.binary_cross_entropy(value, targets["value"])
            klloss = _teacher_kl_loss(out["policy_logits"], teacher_out["policy_logits"], batch.option_mask)
            bloss = _belief_loss(out["belief"], targets)

            loss = ploss + value_weight * vloss + kl_weight * klloss + belief_weight * bloss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            opt.step()

            bsz = targets["value"].shape[0]
            running["policy"] += ploss.item() * bsz
            running["value"] += vloss.item() * bsz
            running["kl"] += klloss.item() * bsz
            running["belief"] += bloss.item() * bsz
            running["grad_norm"] += float(grad_norm) * bsz
            running["n"] += bsz

        n = max(1, running["n"])
        stats = {
            "epoch": epoch,
            "policy_loss": running["policy"] / n,
            "value_loss": running["value"] / n,
            "kl_loss": running["kl"] / n,
            "belief_loss": running["belief"] / n,
            "mean_grad_norm_pre_clip": running["grad_norm"] / n,
        }
        epoch_stats.append(stats)
        print(
            f"  [selfplay] epoch {epoch}: policy={stats['policy_loss']:.4f} "
            f"value={stats['value_loss']:.4f} kl={stats['kl_loss']:.4f} belief={stats['belief_loss']:.4f} "
            f"grad_norm={stats['mean_grad_norm_pre_clip']:.3f}"
        )

    return {"epochs": epoch_stats}
