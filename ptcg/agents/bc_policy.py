"""The Week-1 brain: :class:`BeliefSetDMCLite` wrapped as a :class:`Policy`.

Deliberately the same shape as :class:`~ptcg.agents.heuristic.HeuristicPolicy`
(``deck()`` / ``score()`` / ``desired_count()``), so it drops into
``SafetyShell`` unchanged -- masking, the loop guard, and ``FallbackPolicy``
all stay exactly as proven in Week 0. This file has no try/except of its own;
``SafetyShell._act`` already wraps ``policy.score()`` and degrades to the
fallback on any exception, same as it does for the heuristic.

"How many to pick" (``desired_count``) is intentionally *not* learned here.
It reuses ``default_desired_count``, the same rule ``HeuristicPolicy`` and
``FallbackPolicy`` use, so the network only has to learn to *rank* options,
not also invent a count head -- a smaller, more sample-efficient job that fits
a self-play-sized corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from ..core.actions import ActionCandidate
from ..core.carddb import CardDB
from ..core.clock import Deadline
from ..core.obs import ObsView
from .base import default_desired_count
from .network import BeliefSetDMCLite, CardTable, encode_view

__all__ = ["BCPolicy", "load_bc_policy"]


class BCPolicy:
    name = "bc-v1"

    def __init__(self, deck_ids: Sequence[int], model: BeliefSetDMCLite, table: CardTable, card_mode: str) -> None:
        self._deck = list(deck_ids)
        self.model = model
        self.model.eval()
        self.table = table
        self.card_mode = card_mode

    def deck(self) -> list[int]:
        return list(self._deck)

    def score(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        deadline: Deadline,
    ) -> list[float]:
        if not candidates:
            return []
        inp = encode_view(view, candidates, self.table, self.card_mode)
        with torch.no_grad():
            out = self.model(inp)
        logits = out["policy_logits"][0]
        return [float(x) for x in logits[: len(candidates)].tolist()]

    def desired_count(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        scores: Sequence[float],
    ) -> int:
        return default_desired_count(view, candidates, scores)


def load_bc_policy(checkpoint_path: str | Path, deck_ids: Sequence[int], db: CardDB) -> BCPolicy:
    """Load a checkpoint saved by ``ptcg/train/bc.py`` into a ready-to-play
    :class:`BCPolicy`. Uses the card embedding table baked into the
    checkpoint (``card_struct``/``card_text``, computed once offline) rather
    than rebuilding it from ``db``/MiniLM at runtime -- for text mode that
    rebuild means running MiniLM over the full ~1,267-card pool, which
    measured ~20s locally. That is harmless against the 600s cumulative
    budget but an unnecessary risk against an undocumented per-move cap, for
    a computation whose result is deterministic and already sitting in the
    checkpoint. Still checked against the live card pool -- a silently
    mismatched card ID -> row mapping would make every score meaningless
    without ever raising, and this check costs nothing (no MiniLM involved).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    card_mode = ckpt["card_mode"]

    live_ids = sorted(c.card_id for c in db.all_cards())
    if live_ids != ckpt["card_ids"]:
        raise ValueError(
            "The live engine's card pool does not match the checkpoint's "
            "training-time card table (card pool changed?). Refusing to load a "
            "policy whose card-ID -> embedding-row mapping would silently be wrong."
        )
    table = CardTable.from_tensors(ckpt["card_ids"], ckpt["card_struct"], ckpt.get("card_text"))

    model = BeliefSetDMCLite(card_mode=card_mode, n_cards=ckpt["n_cards"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return BCPolicy(deck_ids, model, table, card_mode)
