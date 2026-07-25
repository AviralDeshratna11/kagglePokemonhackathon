"""Card representation for the Week-1 network.

Two independent pieces, kept separate on purpose so H1 (card-text embeddings
vs. one-hot ID embeddings) is a clean ablation rather than a re-write:

* :func:`structured_features` -- deterministic, non-trainable numbers already
  sitting in :class:`~ptcg.core.carddb.Card`/``Attack`` (type, stage, HP,
  weakness/resistance, energy costs, role tags). Every network variant gets
  these.
* :class:`TextEncoder` -- a frozen MiniLM sentence-transformer over each
  card's skill + attack text, cached per card ID. Only the *text-embedding*
  H1 arm consumes this; the *one-hot* arm replaces it with a learned
  ``nn.Embedding(card_id)`` instead (see ``ptcg/agents/network.py``), which is
  the whole point of the ablation: MiniLM output is meaningful for a card the
  network never trained on, a learned ID embedding is not.

Nothing here is trainable. The frozen encoder runs in ``torch.no_grad()`` and
its output is treated as a fixed input feature, exactly like the structured
block; the *projection* from 384-d text / raw ID down to the shared
128-d card embedding is a trainable layer that lives in ``network.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from .carddb import Card, CardDB, Role
from .enums import N_CARD_TYPE, N_ENERGY

__all__ = [
    "TEXT_MODEL_NAME",
    "TEXT_DIM",
    "STRUCT_DIM",
    "structured_features",
    "TextEncoder",
    "get_text_encoder",
    "card_text",
]

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384

# Same 16 roles actions.py tags candidates with, kept in the same order so the
# one-hot block lines up if the two are ever compared side by side.
_ROLE_ORDER: tuple[str, ...] = (
    Role.DRAW, Role.SEARCH, Role.BALL, Role.ENERGY_ACCEL, Role.RECOVERY,
    Role.SWITCH_SELF, Role.GUST, Role.HEAL, Role.DISRUPT, Role.COIN_FLIP,
    Role.SELF_DISCARD_COST, Role.DAMAGE_COUNTER, Role.STATUS, Role.SELF_LOCK,
    Role.TURN_END, Role.SELF_ENERGY_COST,
)

STRUCT_DIM = (
    N_CARD_TYPE  # card_type one-hot
    + N_ENERGY  # energy_type one-hot
    + N_ENERGY  # weakness one-hot
    + N_ENERGY  # resistance one-hot
    + 2  # hp_norm, retreat_norm
    + 8  # basic/stage1/stage2/ex/mega_ex/tera/ace_spec/has_ability
    + 1  # prizes_norm
    + len(_ROLE_ORDER)  # role tags (union of ability + attack roles)
    + 3  # max_attack_damage_norm, mean_attack_cost_norm, n_attacks_norm
)


def _norm(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(-4.0, min(4.0, float(x) / scale))


def structured_features(card: Card, db: CardDB) -> np.ndarray:
    """Deterministic feature block for one card. Shape ``(STRUCT_DIM,)``."""
    v = np.zeros(STRUCT_DIM, dtype=np.float32)
    off = 0

    if 0 <= card.card_type < N_CARD_TYPE:
        v[off + card.card_type] = 1.0
    off += N_CARD_TYPE

    if 0 <= card.energy_type < N_ENERGY:
        v[off + card.energy_type] = 1.0
    off += N_ENERGY

    if card.weakness and 0 <= card.weakness < N_ENERGY:
        v[off + card.weakness] = 1.0
    off += N_ENERGY

    if card.resistance and 0 <= card.resistance < N_ENERGY:
        v[off + card.resistance] = 1.0
    off += N_ENERGY

    v[off] = _norm(card.hp, max(1, db.max_hp))
    v[off + 1] = _norm(card.retreat_cost, 4.0)
    off += 2

    v[off] = float(card.basic)
    v[off + 1] = float(card.stage1)
    v[off + 2] = float(card.stage2)
    v[off + 3] = float(card.ex)
    v[off + 4] = float(card.mega_ex)
    v[off + 5] = float(card.tera)
    v[off + 6] = float(card.ace_spec)
    v[off + 7] = float(card.has_ability)
    off += 8

    v[off] = _norm(card.prizes_when_koed, 3.0)
    off += 1

    attacks = db.attacks_of(card.card_id)
    all_roles = set(card.roles)
    for a in attacks:
        all_roles |= a.roles
    for i, r in enumerate(_ROLE_ORDER):
        if r in all_roles:
            v[off + i] = 1.0
    off += len(_ROLE_ORDER)

    if attacks:
        v[off] = _norm(max(a.damage for a in attacks), max(1, db.max_damage))
        v[off + 1] = _norm(sum(a.cost for a in attacks) / len(attacks), 5.0)
        v[off + 2] = _norm(len(attacks), 4.0)
    off += 3

    return v


def card_text(card: Card, db: CardDB) -> str:
    """The text MiniLM sees: ability text plus every attack's name + effect."""
    parts = list(card.skill_texts)
    for a in db.attacks_of(card.card_id):
        parts.append(f"{a.name}. {a.text}" if a.text else a.name)
    text = " ".join(p for p in parts if p).strip()
    return text or card.name


class TextEncoder:
    """Frozen MiniLM, loaded once. Encodes are cached per input string.

    Forces offline mode before touching ``sentence_transformers``: by default
    it phones home on every load to check for adapter-config updates, which
    (a) is pointless once the model is already cached locally, (b) has thrown
    a bare ``RuntimeError`` from inside its own retry/backoff path on a flaky
    connection here rather than falling back to cache, and (c) would hang or
    fail outright in the actual Kaggle submission container, which has no
    network access at all. The model must already be cached (it is, from
    every prior run in this repo) or this raises instead of hanging.
    """

    def __init__(self, model_name: str = TEXT_MODEL_NAME) -> None:
        import os  # noqa: PLC0415

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self._model.eval()

    @lru_cache(maxsize=4096)
    def _encode_one(self, text: str) -> tuple[float, ...]:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            vec = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return tuple(float(x) for x in vec)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Batch entry point; internally cached per-string so repeated cards
        (every ``build_candidates`` call in a game) cost one dict lookup."""
        return np.array([self._encode_one(t) for t in texts], dtype=np.float32)


_ENCODER: TextEncoder | None = None


def get_text_encoder() -> TextEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = TextEncoder()
    return _ENCODER
