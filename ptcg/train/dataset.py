"""Loads the ``.jsonl.gz`` trace corpus (self-play today, real Kaggle daily
replays once credentials exist -- see ``ptcg/tools/pull_episodes.py``) into
training tensors for :class:`~ptcg.agents.network.BeliefSetDMCLite`.

Deliberately not a giant in-memory blob of the full corpus: ``load_records``
subsamples to a bounded record count so this runs comfortably on a CPU box.
That is a real, adjustable knob (``max_records``), not a hidden limitation --
raise it if you have the time/RAM for a bigger training run.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..agents.network import (
    BELIEF_CARD_TYPE,
    BELIEF_ENERGY_TYPE,
    BELIEF_ROLE_FLAGS,
    HIST_LEN,
    STRUCT_DIM,
    TEXT_DIM,
    ZONE_CAPS,
    CardTable,
    ModelInput,
    log_features,
    zone_cards_from_ids,
)
from ..core.actions import FEATURE_DIM
from ..core.carddb import CardDB
from ..core.enums import N_CARD_TYPE, N_ENERGY

__all__ = ["load_records", "TraceDataset", "collate", "belief_target_for_hand"]

ZONE_NAMES = (
    "my_active", "my_bench", "my_hand", "my_discard", "my_prize",
    "opp_active", "opp_bench", "opp_discard", "opp_prize",
)


def load_records(trace_dir: str | Path, max_records: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Flatten every episode's decision records, each tagged with its game's
    outcome. Uniformly subsampled across files (not just the first N) so a
    bounded run still sees the whole corpus's diversity."""
    trace_dir = Path(trace_dir)
    files = sorted(trace_dir.glob("*.jsonl.gz"))
    rng = random.Random(seed)
    rng.shuffle(files)

    out: list[dict[str, Any]] = []
    for fp in files:
        with gzip.open(fp, "rt", encoding="utf-8") as fh:
            lines = fh.readlines()
        outcome = 0.5
        decisions: list[dict[str, Any]] = []
        for line in lines:
            rec = json.loads(line)
            t = rec.get("type")
            if t == "decision":
                decisions.append(rec)
            elif t == "outcome":
                outcome = float(rec.get("outcome", 0.5))
        for d in decisions:
            d["_outcome"] = outcome
        out.extend(decisions)
        if max_records is not None and len(out) >= max_records:
            break

    if max_records is not None and len(out) > max_records:
        rng.shuffle(out)
        out = out[:max_records]
    return out


def belief_target_for_hand(hand_card_ids: Sequence[int], db: CardDB) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth histogram for the belief head from a list of true card
    IDs in a hand. Returns ``(card_type_dist, energy_type_dist, role_rates)``."""
    card_type = np.zeros(BELIEF_CARD_TYPE, dtype=np.float32)
    energy_type = np.zeros(BELIEF_ENERGY_TYPE, dtype=np.float32)
    role = np.zeros(BELIEF_ROLE_FLAGS, dtype=np.float32)
    n = 0
    for cid in hand_card_ids:
        c = db.card(cid)
        if c is None:
            continue
        n += 1
        if 0 <= c.card_type < N_CARD_TYPE:
            card_type[c.card_type] += 1
        if 0 <= c.energy_type < N_ENERGY:
            energy_type[c.energy_type] += 1
        role[0] += float(c.has_ability)
        role[1] += float(c.basic)
        role[2] += float(c.ex)
    if n > 0:
        card_type /= n
        energy_type /= n
        role /= n
    return card_type, energy_type, role


@dataclass
class _Item:
    zones: dict[str, tuple]
    history: np.ndarray
    hidden_counts: np.ndarray
    globals: np.ndarray
    option_struct: np.ndarray
    option_aux: np.ndarray
    option_features: np.ndarray
    n_options: int
    chosen: list[int]
    value: float
    belief_valid: bool
    belief: tuple[np.ndarray, np.ndarray, np.ndarray]


class TraceDataset(Dataset):
    """One item per recorded decision. ``card_mode`` selects the H1 arm."""

    def __init__(self, records: list[dict[str, Any]], table: CardTable, db: CardDB, card_mode: str) -> None:
        self.records = records
        self.table = table
        self.db = db
        self.card_mode = card_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> _Item:
        rec = self.records[i]
        board = rec["board"]
        globals_d = rec["globals"]
        table, mode = self.table, self.card_mode

        zones = {
            "my_active": zone_cards_from_ids(board["my_active"], ZONE_CAPS["active"], table, mode),
            "my_bench": zone_cards_from_ids(board["my_bench"], ZONE_CAPS["bench"], table, mode),
            "my_hand": zone_cards_from_ids(board["my_hand"], ZONE_CAPS["hand"], table, mode),
            "my_discard": zone_cards_from_ids(board["my_discard"], ZONE_CAPS["discard"], table, mode),
            "my_prize": zone_cards_from_ids(board["my_prize"], ZONE_CAPS["prize"], table, mode),
            "opp_active": zone_cards_from_ids(board["opp_active"], ZONE_CAPS["active"], table, mode),
            "opp_bench": zone_cards_from_ids(board["opp_bench"], ZONE_CAPS["bench"], table, mode),
            "opp_discard": zone_cards_from_ids(board["opp_discard"], ZONE_CAPS["discard"], table, mode),
            "opp_prize": zone_cards_from_ids(board["opp_prize"], ZONE_CAPS["prize"], table, mode),
        }

        history = log_features(rec.get("logs_tail") or [], rec["player"])

        hidden_counts = np.array(
            [
                math.tanh(board["opp_hand_count"] / 7.0),
                math.tanh(board["opp_deck_count"] / 30.0),
                math.tanh(board["opp_prizes_remaining"] / 6.0),
            ],
            dtype=np.float32,
        )
        globals_ = np.array(
            [
                math.tanh(globals_d["turn"] / 20.0),
                math.tanh(globals_d["prize_diff"] / 3.0),
                math.tanh((globals_d["remaining_time"] or 600.0) / 600.0),
                math.tanh(board["my_deck_count"] / 30.0),
                math.tanh(board["opp_deck_count"] / 30.0),
                float(globals_d["supporter_played"]),
                float(globals_d["stadium_played"]),
                float(globals_d["energy_attached"]),
                float(globals_d["i_am_first"]),
            ],
            dtype=np.float32,
        )

        cids = rec["candidate_card_ids"]
        k = rec["n_options"]
        opt_struct = np.zeros((k, STRUCT_DIM), dtype=np.float32)
        opt_aux = (
            np.zeros((k, TEXT_DIM), dtype=np.float32) if mode == "text" else np.full((k,), table.unknown_index, dtype=np.int64)
        )
        for j in range(k):
            idx = table.index_of(cids[j] if j < len(cids) else None)
            opt_struct[j] = table.struct[idx].numpy()
            if mode == "text":
                opt_aux[j] = table.text[idx].numpy()
            else:
                opt_aux[j] = idx
        opt_feat = np.array(rec["features"], dtype=np.float32)
        if opt_feat.shape[0] != k:
            opt_feat = np.zeros((k, FEATURE_DIM), dtype=np.float32)

        belief_valid = bool(rec.get("belief_valid"))
        belief = (
            belief_target_for_hand(rec.get("belief_hand_card_ids") or [], self.db)
            if belief_valid
            else (
                np.zeros(BELIEF_CARD_TYPE, dtype=np.float32),
                np.zeros(BELIEF_ENERGY_TYPE, dtype=np.float32),
                np.zeros(BELIEF_ROLE_FLAGS, dtype=np.float32),
            )
        )

        return _Item(
            zones=zones,
            history=history,
            hidden_counts=hidden_counts,
            globals=globals_,
            option_struct=opt_struct,
            option_aux=opt_aux,
            option_features=opt_feat,
            n_options=k,
            chosen=list(rec["chosen"]),
            value=float(rec["_outcome"]),
            belief_valid=belief_valid,
            belief=belief,
        )


def collate(items: list[_Item], card_mode: str) -> tuple[ModelInput, dict[str, torch.Tensor]]:
    b = len(items)
    max_k = max(it.n_options for it in items)

    zones_t: dict[str, tuple] = {}
    for name in ZONE_NAMES:
        structs = np.stack([it.zones[name][0][0] for it in items])
        auxs = np.stack([it.zones[name][0][1] for it in items])
        masks = np.stack([it.zones[name][1] for it in items])
        zones_t[name] = (
            (torch.tensor(structs), torch.tensor(auxs)),
            torch.tensor(masks),
        )

    history = torch.tensor(np.stack([it.history for it in items]))
    hidden_counts = torch.tensor(np.stack([it.hidden_counts for it in items]))
    globals_ = torch.tensor(np.stack([it.globals for it in items]))

    struct_dim = items[0].option_struct.shape[1]
    aux_shape = items[0].option_aux.shape[1:]
    opt_struct = np.zeros((b, max_k, struct_dim), dtype=np.float32)
    opt_aux = (
        np.zeros((b, max_k, *aux_shape), dtype=np.float32)
        if card_mode == "text"
        else np.zeros((b, max_k), dtype=np.int64)
    )
    opt_feat = np.zeros((b, max_k, items[0].option_features.shape[1]), dtype=np.float32)
    opt_mask = np.zeros((b, max_k), dtype=bool)
    chosen_mask = np.zeros((b, max_k), dtype=np.float32)

    for i, it in enumerate(items):
        k = it.n_options
        opt_struct[i, :k] = it.option_struct
        opt_aux[i, :k] = it.option_aux
        opt_feat[i, :k] = it.option_features
        opt_mask[i, :k] = True
        for c in it.chosen:
            if 0 <= c < k:
                chosen_mask[i, c] = 1.0
        if chosen_mask[i].sum() == 0 and k > 0:
            chosen_mask[i, 0] = 1.0  # degenerate guard, should not happen with real traces

    value = torch.tensor(np.array([it.value for it in items], dtype=np.float32))
    belief_valid = torch.tensor(np.array([it.belief_valid for it in items], dtype=np.float32))
    belief_card_type = torch.tensor(np.stack([it.belief[0] for it in items]))
    belief_energy_type = torch.tensor(np.stack([it.belief[1] for it in items]))
    belief_role = torch.tensor(np.stack([it.belief[2] for it in items]))

    model_input = ModelInput(
        zones=zones_t,
        history=history,
        hidden_counts=hidden_counts,
        globals=globals_,
        option_features=torch.tensor(opt_feat),
        option_card_idx_or_text=(torch.tensor(opt_struct), torch.tensor(opt_aux)),
        option_mask=torch.tensor(opt_mask),
    )
    targets = {
        "chosen_mask": torch.tensor(chosen_mask),
        "value": value,
        "belief_valid": belief_valid,
        "belief_card_type": belief_card_type,
        "belief_energy_type": belief_energy_type,
        "belief_role": belief_role,
    }
    return model_input, targets
