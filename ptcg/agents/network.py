"""BeliefSet-DMC-lite: the Week-1 network.

A CPU-sized (~1-3M param) version of the architecture both planning PDFs
describe: permutation-invariant Set-Transformer zone encoders, a history GRU,
a shared trunk, a pointer-style policy head over the engine's variable-length
option list, a value head, and a lightweight opponent-belief head.

Deliberately out of scope here (see the Week-1 plan): the belief head is
trained but not yet *consumed* by search (no ``search_begin``/``search_step``
integration -- that is Week 3), and there is no SPR self-predictive loss or
league training. This module owns representation + policy/value/belief only.

Everything downstream consumes plain Python/numpy via :class:`BoardEncoder`,
so the rest of the codebase (``ObsView``, ``ActionCandidate``) does not need
to know PyTorch exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..core.actions import ActionCandidate, FEATURE_DIM
from ..core.carddb import CardDB
from ..core.embeddings import STRUCT_DIM, TEXT_DIM, card_text, get_text_encoder, structured_features
from ..core.enums import N_CARD_TYPE, N_ENERGY, N_LOG_TYPE
from ..core.obs import ObsView, PlayerView

__all__ = [
    "CARD_EMBED_DIM",
    "TRUNK_DIM",
    "BELIEF_DIM",
    "ZONE_CAPS",
    "HIST_LEN",
    "CardTable",
    "BeliefSetDMCLite",
    "ModelInput",
    "encode_view",
    "zone_cards_from_ids",
    "log_features",
    "count_parameters",
]

CARD_EMBED_DIM = 128
CARD_STRUCT_HIDDEN = 32
CARD_TEXT_HIDDEN = 64
ZONE_HIDDEN = 64
ZONE_HEADS = 4
ZONE_INDUCING = 8
HIST_HIDDEN = 32
HIST_LEN = 16
OPTION_HIDDEN = 96
TRUNK_DIM = 256

# Per-player, per-item zones we actually see card identities for. Bench/hand
# are padded to these caps (5/10 are the engine's real limits; discard/prize
# are truncated to the most recent/relevant N to keep tensors small).
ZONE_CAPS = {"active": 1, "bench": 5, "hand": 10, "discard": 20, "prize": 6}

# Opponent hand + deck remainder + face-down prizes: identity is hidden, only
# cardinality is known -- "count-only attention masks" per the blueprint,
# realized here as normalized scalar counts rather than attending over
# unknown-identity items (which is not mechanically meaningful).
N_HIDDEN_COUNTS = 3  # opp hand_count, opp deck_count, opp prizes_remaining

N_GLOBAL_SCALARS = 9  # turn, prize_diff, remaining_time, my/opp deck_count,
# supporter_played, stadium_played, energy_attached, i_am_first

# Belief head target: a histogram over the opponent's hidden hand + remaining
# deck. Card-type counts + energy-type counts + 3 coarse role rates, not a
# full per-card-ID distribution (too high-dimensional to fit reliably on a
# CPU box with a self-play-sized corpus).
BELIEF_CARD_TYPE = N_CARD_TYPE
BELIEF_ENERGY_TYPE = N_ENERGY
BELIEF_ROLE_FLAGS = 3  # has_ability, basic, ex -- mean rate, not counts
BELIEF_DIM = BELIEF_CARD_TYPE + BELIEF_ENERGY_TYPE + BELIEF_ROLE_FLAGS


# ---------------------------------------------------------------------------
# Card table: precomputed, cached per-card features for a CardDB
# ---------------------------------------------------------------------------


@dataclass
class CardTable:
    """Per-card feature rows for every card in a :class:`CardDB`, plus one
    trailing "unknown card" row (index ``n_cards``) for face-down/unresolved
    card IDs."""

    card_ids: list[int]
    id_to_idx: dict[int, int]
    struct: torch.Tensor  # (n_cards + 1, STRUCT_DIM)
    text: torch.Tensor | None  # (n_cards + 1, TEXT_DIM) or None

    @property
    def n_cards(self) -> int:
        return len(self.card_ids)

    @property
    def unknown_index(self) -> int:
        return self.n_cards

    def index_of(self, card_id: int | None) -> int:
        if card_id is None:
            return self.unknown_index
        return self.id_to_idx.get(card_id, self.unknown_index)

    @classmethod
    def from_tensors(cls, card_ids: list[int], struct: torch.Tensor, text: torch.Tensor | None) -> "CardTable":
        """Reconstruct without touching ``db`` or MiniLM at all -- what a
        submission loads at runtime. ``struct``/``text`` come straight from a
        checkpoint saved by ``ptcg/train/bc.py``, computed once offline. The
        alternative, calling :meth:`build` fresh, runs MiniLM over the full
        card pool on the agent's very first decision (~20s measured locally)
        -- harmless against the 600s cumulative budget in principle, but an
        unnecessary risk against the undocumented per-move cap the community
        reports seeing, for zero benefit since the embeddings are frozen and
        deterministic anyway."""
        return cls(card_ids=list(card_ids), id_to_idx={cid: i for i, cid in enumerate(card_ids)}, struct=struct, text=text)

    @classmethod
    def build(cls, db: CardDB, with_text: bool = True) -> "CardTable":
        ids = sorted(c.card_id for c in db.all_cards())
        id_to_idx = {cid: i for i, cid in enumerate(ids)}

        struct_rows = [structured_features(db.card(cid), db) for cid in ids]
        struct_rows.append(np.zeros(STRUCT_DIM, dtype=np.float32))  # unknown row
        struct = torch.tensor(np.stack(struct_rows), dtype=torch.float32)

        text = None
        if with_text:
            enc = get_text_encoder()
            texts = [card_text(db.card(cid), db) for cid in ids]
            text_rows = enc.encode(texts)
            text_rows = np.concatenate([text_rows, np.zeros((1, TEXT_DIM), dtype=np.float32)], axis=0)
            text = torch.tensor(text_rows, dtype=torch.float32)

        return cls(card_ids=ids, id_to_idx=id_to_idx, struct=struct, text=text)


# ---------------------------------------------------------------------------
# Set Transformer building blocks (Lee et al., ICML 2019)
# ---------------------------------------------------------------------------


class MAB(nn.Module):
    """Multihead Attention Block: Q attends over K/V, with an optional key
    padding mask (``True`` = valid, ``False`` = padding)."""

    def __init__(self, dim_q: int, dim_kv: int, dim_out: int, num_heads: int) -> None:
        super().__init__()
        assert dim_out % num_heads == 0
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_q, dim_out)
        self.fc_k = nn.Linear(dim_kv, dim_out)
        self.fc_v = nn.Linear(dim_kv, dim_out)
        self.fc_o = nn.Linear(dim_out, dim_out)
        self.ln0 = nn.LayerNorm(dim_out)
        self.ln1 = nn.LayerNorm(dim_out)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = q.shape
        m = kv.shape[1]
        h = self.num_heads
        d = self.dim_out // h

        q2 = self.fc_q(q)  # (b,n,dim_out)
        k2 = self.fc_k(kv)  # (b,m,dim_out)
        v2 = self.fc_v(kv)  # (b,m,dim_out)

        q2h = q2.view(b, n, h, d).transpose(1, 2)  # (b,h,n,d)
        k2h = k2.view(b, m, h, d).transpose(1, 2)
        v2h = v2.view(b, m, h, d).transpose(1, 2)

        scores = q2h @ k2h.transpose(-2, -1) / math.sqrt(d)  # (b,h,n,m)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v2h  # (b,h,n,d)
        out = out.transpose(1, 2).reshape(b, n, h * d)  # (b,n,dim_out)

        # Residual is around the attention output using the *projected* query
        # (q2), matching the reference Set Transformer MAB -- q and q2 can
        # have different feature widths (e.g. ISAB's mab1: dim_in -> dim_out).
        o = self.ln0(q2 + out)
        o = self.ln1(o + F.relu(self.fc_o(o)))
        return o


class ISAB(nn.Module):
    """Induced Set Attention Block: O(n) instead of O(n^2) in set size."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inds, dim_out) * 0.02)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        h = self.mab0(self.inducing.expand(b, -1, -1), x, mask=mask)
        return self.mab1(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention: a learned seed query pools the set."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        return self.mab(self.seed.expand(b, -1, -1), x, mask=mask)


class ZoneEncoder(nn.Module):
    """ISAB + PMA over one unordered zone of card embeddings -> one vector."""

    def __init__(self, in_dim: int = CARD_EMBED_DIM, hidden: int = ZONE_HIDDEN) -> None:
        super().__init__()
        self.isab = ISAB(in_dim, hidden, ZONE_HEADS, ZONE_INDUCING)
        self.pma = PMA(hidden, ZONE_HEADS, num_seeds=1)

    def forward(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # cards: (b, n, in_dim), mask: (b, n) bool
        h = self.isab(cards, mask=mask)
        pooled = self.pma(h, mask=mask)  # (b, 1, hidden)
        return pooled.squeeze(1)


# ---------------------------------------------------------------------------
# Card embedder: the H1 ablation lives here
# ---------------------------------------------------------------------------


class CardEmbedder(nn.Module):
    """Projects a card's structured features + (text OR learned ID) to
    ``CARD_EMBED_DIM``. ``mode="text"`` and ``mode="onehot"`` are the two H1
    arms; everything downstream is identical either way."""

    def __init__(
        self,
        mode: str,
        n_cards: int,
        embed_dim: int = CARD_EMBED_DIM,
        struct_hidden: int = CARD_STRUCT_HIDDEN,
        text_hidden: int = CARD_TEXT_HIDDEN,
    ) -> None:
        super().__init__()
        if mode not in ("text", "onehot"):
            raise ValueError(mode)
        self.mode = mode
        self.embed_dim = embed_dim
        self.struct_proj = nn.Linear(STRUCT_DIM, struct_hidden)
        if mode == "text":
            self.text_proj = nn.Linear(TEXT_DIM, text_hidden)
            self.id_embed = None
        else:
            self.text_proj = None
            # +1 for the "unknown card" row.
            self.id_embed = nn.Embedding(n_cards + 1, text_hidden)
        self.out = nn.Linear(struct_hidden + text_hidden, embed_dim)

    def forward(self, struct: torch.Tensor, text: torch.Tensor | None, idx: torch.Tensor | None) -> torch.Tensor:
        s = F.relu(self.struct_proj(struct))
        if self.mode == "text":
            t = F.relu(self.text_proj(text))
        else:
            t = F.relu(self.id_embed(idx))
        return self.out(torch.cat([s, t], dim=-1))

    def embed_table(self, table: CardTable) -> torch.Tensor:
        """Embed every row of a :class:`CardTable` at once: ``(n_cards+1, CARD_EMBED_DIM)``."""
        if self.mode == "text":
            return self.forward(table.struct, table.text, None)
        idx = torch.arange(table.n_cards + 1)
        return self.forward(table.struct, None, idx)

    def aux_only(self, text_or_idx: torch.Tensor) -> torch.Tensor:
        """The text/ID half of the embedding *alone*, without the structured
        block. H1 (does card-text generalize to unseen cards better than a
        learned ID embedding) is a claim about this half specifically -- the
        structured block is identical, deterministic, and available to both
        arms regardless of training, so it would dilute the comparison."""
        if self.mode == "text":
            return F.relu(self.text_proj(text_or_idx))
        return F.relu(self.id_embed(text_or_idx))


# ---------------------------------------------------------------------------
# History encoder
# ---------------------------------------------------------------------------

_LOG_FEAT_DIM = N_LOG_TYPE + 3  # one-hot type + is_mine + value_norm + head_flag


def log_features(entries: Sequence[dict], my_index: int) -> np.ndarray:
    """Feature matrix for up to ``HIST_LEN`` most recent log entries, oldest
    first, padded on the left with zeros. Shape ``(HIST_LEN, _LOG_FEAT_DIM)``."""
    tail = list(entries)[-HIST_LEN:]
    mat = np.zeros((HIST_LEN, _LOG_FEAT_DIM), dtype=np.float32)
    pad = HIST_LEN - len(tail)
    for i, e in enumerate(tail):
        row = pad + i
        t = int(e.get("type", -1))
        if 0 <= t < N_LOG_TYPE:
            mat[row, t] = 1.0
        pi = e.get("playerIndex")
        mat[row, N_LOG_TYPE] = 1.0 if pi is not None and int(pi) == my_index else 0.0
        val = e.get("value")
        mat[row, N_LOG_TYPE + 1] = max(-4.0, min(4.0, float(val) / 100.0)) if val is not None else 0.0
        mat[row, N_LOG_TYPE + 2] = 1.0 if e.get("head") else 0.0
    return mat


class HistoryEncoder(nn.Module):
    def __init__(self, hidden: int = HIST_HIDDEN) -> None:
        super().__init__()
        self.gru = nn.GRU(_LOG_FEAT_DIM, hidden, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (b, HIST_LEN, _LOG_FEAT_DIM)
        _, h_n = self.gru(seq)
        return h_n[-1]  # (b, hidden)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """Pointer-style cross-attention: the trunk state is the query, each
    option's (feature-vector + its referenced card's embedding) is a key/value
    of one "token"; output is one logit per option."""

    def __init__(self, trunk_dim: int = TRUNK_DIM, card_embed_dim: int = CARD_EMBED_DIM, option_hidden: int = OPTION_HIDDEN) -> None:
        super().__init__()
        self.option_hidden = option_hidden
        opt_in = FEATURE_DIM + card_embed_dim
        self.option_proj = nn.Sequential(
            nn.Linear(opt_in, option_hidden), nn.ReLU(), nn.Linear(option_hidden, option_hidden)
        )
        self.query_proj = nn.Linear(trunk_dim, option_hidden)

    def forward(self, trunk: torch.Tensor, option_features: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        # trunk: (b, trunk_dim); option_features: (b, k, opt_in); option_mask: (b, k) bool
        opt = self.option_proj(option_features)  # (b, k, hidden)
        q = self.query_proj(trunk).unsqueeze(1)  # (b, 1, hidden)
        logits = (opt @ q.transpose(-2, -1)).squeeze(-1) / math.sqrt(self.option_hidden)  # (b, k)
        logits = logits.masked_fill(~option_mask, -1e9)
        return logits


class ValueHead(nn.Module):
    def __init__(self, trunk_dim: int = TRUNK_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(trunk_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(trunk)).squeeze(-1)


class BeliefHead(nn.Module):
    """Predicts a histogram over the opponent's hidden hand + remaining deck.
    Input is only ever the trunk (built from legally visible information);
    the supervised target uses privileged self-play information, never the
    model's own input."""

    def __init__(self, trunk_dim: int = TRUNK_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(trunk_dim, 96), nn.ReLU())
        self.card_type_head = nn.Linear(96, BELIEF_CARD_TYPE)
        self.energy_type_head = nn.Linear(96, BELIEF_ENERGY_TYPE)
        self.role_head = nn.Linear(96, BELIEF_ROLE_FLAGS)

    def forward(self, trunk: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(trunk)
        return {
            "card_type": torch.softmax(self.card_type_head(h), dim=-1),
            "energy_type": torch.softmax(self.energy_type_head(h), dim=-1),
            "role": torch.sigmoid(self.role_head(h)),
        }


# ---------------------------------------------------------------------------
# The composite network
# ---------------------------------------------------------------------------


class BeliefSetDMCLite(nn.Module):
    """``dims``, if given, overrides the module-level default widths --
    this is what lets ``ptcg/train/distill.py`` build genuinely smaller
    variants of the exact same architecture (not a different architecture)
    for the efficiency/quality frontier comparison. Omitting it reproduces
    the original Week-1 sizing exactly, so every existing checkpoint still
    loads unchanged."""

    def __init__(self, card_mode: str, n_cards: int, dims: dict[str, int] | None = None) -> None:
        super().__init__()
        d = dims or {}
        card_embed_dim = d.get("card_embed_dim", CARD_EMBED_DIM)
        card_struct_hidden = d.get("card_struct_hidden", CARD_STRUCT_HIDDEN)
        card_text_hidden = d.get("card_text_hidden", CARD_TEXT_HIDDEN)
        zone_hidden = d.get("zone_hidden", ZONE_HIDDEN)
        hist_hidden = d.get("hist_hidden", HIST_HIDDEN)
        trunk_dim = d.get("trunk_dim", TRUNK_DIM)
        option_hidden = d.get("option_hidden", OPTION_HIDDEN)

        self.card_mode = card_mode
        self.dims = {
            "card_embed_dim": card_embed_dim, "card_struct_hidden": card_struct_hidden,
            "card_text_hidden": card_text_hidden, "zone_hidden": zone_hidden,
            "hist_hidden": hist_hidden, "trunk_dim": trunk_dim, "option_hidden": option_hidden,
        }
        self.card_embedder = CardEmbedder(card_mode, n_cards, card_embed_dim, card_struct_hidden, card_text_hidden)

        zone_names = ("my_active", "my_bench", "my_hand", "my_discard", "my_prize", "opp_active", "opp_bench", "opp_discard", "opp_prize")
        self.zone_names = zone_names
        # weights shared across zones (permutation-invariant, zone-agnostic)
        self.zone_encoder = ZoneEncoder(in_dim=card_embed_dim, hidden=zone_hidden)
        self.zone_type_embed = nn.Embedding(len(zone_names), zone_hidden)

        self.history = HistoryEncoder(hidden=hist_hidden)

        trunk_in = len(zone_names) * zone_hidden + hist_hidden + N_HIDDEN_COUNTS + N_GLOBAL_SCALARS
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, trunk_dim), nn.ReLU(), nn.Linear(trunk_dim, trunk_dim), nn.ReLU()
        )

        self.policy_head = PolicyHead(trunk_dim=trunk_dim, card_embed_dim=card_embed_dim, option_hidden=option_hidden)
        self.value_head = ValueHead(trunk_dim=trunk_dim)
        self.belief_head = BeliefHead(trunk_dim=trunk_dim)

    # -- forward --------------------------------------------------------

    def forward(self, batch: "ModelInput") -> dict[str, torch.Tensor]:
        zone_vecs = []
        for i, name in enumerate(self.zone_names):
            cards, mask = batch.zones[name]
            card_emb = self._embed_cards(cards)
            z = self.zone_encoder(card_emb, mask)
            z = z + self.zone_type_embed(torch.full((z.shape[0],), i, dtype=torch.long))
            zone_vecs.append(z)

        hist = self.history(batch.history)
        trunk_in = torch.cat(zone_vecs + [hist, batch.hidden_counts, batch.globals], dim=-1)
        trunk = self.trunk(trunk_in)

        option_cards = self._embed_cards(batch.option_card_idx_or_text)
        option_features = torch.cat([batch.option_features, option_cards], dim=-1)
        policy_logits = self.policy_head(trunk, option_features, batch.option_mask)

        value = self.value_head(trunk)
        belief = self.belief_head(trunk)
        return {"policy_logits": policy_logits, "value": value, "belief": belief, "trunk": trunk}

    def _embed_cards(self, cards) -> torch.Tensor:
        """``cards`` is ``(struct, text)`` for text mode or ``(struct, idx)``
        for onehot mode, both already batched to the same leading shape."""
        struct, aux = cards
        if self.card_mode == "text":
            return self.card_embedder(struct, aux, None)
        return self.card_embedder(struct, None, aux)


# ---------------------------------------------------------------------------
# obs -> tensors
# ---------------------------------------------------------------------------


@dataclass
class ModelInput:
    """Everything :meth:`BeliefSetDMCLite.forward` needs, pre-batched.

    ``zones[name]`` is ``((struct, text_or_idx), mask)`` where ``struct`` has
    shape ``(b, cap, STRUCT_DIM)``, ``text_or_idx`` has shape
    ``(b, cap, TEXT_DIM)`` or ``(b, cap)`` depending on card mode, and
    ``mask`` has shape ``(b, cap)``.
    """

    zones: dict[str, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
    history: torch.Tensor  # (b, HIST_LEN, _LOG_FEAT_DIM)
    hidden_counts: torch.Tensor  # (b, N_HIDDEN_COUNTS)
    globals: torch.Tensor  # (b, N_GLOBAL_SCALARS)
    option_features: torch.Tensor  # (b, k, FEATURE_DIM)
    option_card_idx_or_text: tuple[torch.Tensor, torch.Tensor]  # (struct, text_or_idx) each (b, k, ...)
    option_mask: torch.Tensor  # (b, k) bool


def zone_cards_from_ids(
    card_ids: Sequence[int | None], cap: int, table: CardTable, card_mode: str
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Shared by live inference (:func:`encode_view`) and the offline BC
    dataset loader (``ptcg/train/dataset.py``), so both build tensors the
    network sees identically -- the whole point of recording ``ActionCandidate``
    -shaped data during self-play instead of raw replay JSON."""
    struct = np.zeros((cap, STRUCT_DIM), dtype=np.float32)
    aux = (
        np.zeros((cap, TEXT_DIM), dtype=np.float32)
        if card_mode == "text"
        else np.full((cap,), table.unknown_index, dtype=np.int64)
    )
    mask = np.zeros((cap,), dtype=bool)
    for i, cid in enumerate(card_ids[:cap]):
        # A slot with a card in it (even face-down/unresolved, cid is None but
        # the slot is occupied) still gets mask=True with the "unknown" row;
        # only a genuinely absent slot (fewer cards than cap) is masked out.
        idx = table.index_of(cid)
        struct[i] = table.struct[idx].numpy()
        if card_mode == "text":
            aux[i] = table.text[idx].numpy()
        else:
            aux[i] = idx
        mask[i] = True
    return (struct, aux), mask


def encode_view(
    view: ObsView,
    candidates: Sequence[ActionCandidate],
    table: CardTable,
    card_mode: str,
) -> ModelInput:
    """Build one (batch size 1) :class:`ModelInput` from a live decision."""
    me, opp = view.me, view.opp

    def player_zone_ids(p: PlayerView, area: str) -> list[int | None]:
        """Card IDs per zone. A slot with ``None`` still holds a card (e.g. a
        face-down prize); it is simply unidentified -- matches the recorder in
        ``generate_selfplay.py`` so live inference and offline training build
        identical tensors from identical semantics."""
        if area == "active":
            a = p.active
            return [a.card_id] if a is not None else []
        if area == "bench":
            return [b.card_id for b in p.bench]
        if area == "hand":
            return [c.card_id if c else None for c in p.hand_cards]
        if area == "discard":
            return [d.get("id") for d in p.discard]
        if area == "prize":
            return [pr.get("id") if pr else None for pr in p.prize]
        raise ValueError(area)

    zones: dict[str, tuple] = {}
    zones["my_active"] = zone_cards_from_ids(player_zone_ids(me, "active"), ZONE_CAPS["active"], table, card_mode)
    zones["my_bench"] = zone_cards_from_ids(player_zone_ids(me, "bench"), ZONE_CAPS["bench"], table, card_mode)
    zones["my_hand"] = zone_cards_from_ids(player_zone_ids(me, "hand"), ZONE_CAPS["hand"], table, card_mode)
    zones["my_discard"] = zone_cards_from_ids(player_zone_ids(me, "discard"), ZONE_CAPS["discard"], table, card_mode)
    zones["my_prize"] = zone_cards_from_ids(player_zone_ids(me, "prize"), ZONE_CAPS["prize"], table, card_mode)
    zones["opp_active"] = zone_cards_from_ids(player_zone_ids(opp, "active"), ZONE_CAPS["active"], table, card_mode)
    zones["opp_bench"] = zone_cards_from_ids(player_zone_ids(opp, "bench"), ZONE_CAPS["bench"], table, card_mode)
    zones["opp_discard"] = zone_cards_from_ids(player_zone_ids(opp, "discard"), ZONE_CAPS["discard"], table, card_mode)
    zones["opp_prize"] = zone_cards_from_ids(player_zone_ids(opp, "prize"), ZONE_CAPS["prize"], table, card_mode)

    def _t(pair):
        (struct, aux), mask = pair
        return (torch.tensor(struct)[None], torch.tensor(aux)[None]), torch.tensor(mask)[None]

    zones_t = {k: _t(v) for k, v in zones.items()}

    hist = torch.tensor(log_features(view.logs, view.my_index))[None]

    hidden_counts = np.array(
        [
            math.tanh(opp.hand_count / 7.0),
            math.tanh(opp.deck_count / 30.0),
            math.tanh(opp.prizes_remaining / 6.0),
        ],
        dtype=np.float32,
    )

    globals_ = np.array(
        [
            math.tanh(view.turn / 20.0),
            math.tanh(view.prize_diff / 3.0),
            math.tanh((view.remaining_time or 600.0) / 600.0),
            math.tanh(me.deck_count / 30.0),
            math.tanh(opp.deck_count / 30.0),
            float(view.supporter_played),
            float(view.stadium_played),
            float(view.energy_attached),
            float(view.i_am_first),
        ],
        dtype=np.float32,
    )

    k = len(candidates)
    opt_struct = np.zeros((k, STRUCT_DIM), dtype=np.float32)
    opt_aux = (
        np.zeros((k, TEXT_DIM), dtype=np.float32) if card_mode == "text" else np.full((k,), table.unknown_index, dtype=np.int64)
    )
    opt_feat = np.zeros((k, FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(candidates):
        opt_feat[i] = c.features
        idx = table.index_of(c.card_id)
        opt_struct[i] = table.struct[idx].numpy()
        if card_mode == "text":
            opt_aux[i] = table.text[idx].numpy()
        else:
            opt_aux[i] = idx

    return ModelInput(
        zones=zones_t,
        history=hist,
        hidden_counts=torch.tensor(hidden_counts)[None],
        globals=torch.tensor(globals_)[None],
        option_features=torch.tensor(opt_feat)[None],
        option_card_idx_or_text=(torch.tensor(opt_struct)[None], torch.tensor(opt_aux)[None]),
        option_mask=torch.ones(1, k, dtype=torch.bool) if k > 0 else torch.zeros(1, 0, dtype=torch.bool),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
