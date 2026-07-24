"""Deck registry.

Decks live as plain ``deck.csv`` files -- one card ID per line -- because that
is the exact format the submission bundle requires. Comments (``#``) and blank
lines are stripped on load, so the checked-in lists can be readable while the
shipped artefact stays machine-plain.

Week 3 replaces the hand-written lists here with MAP-Elites output; the loader,
the validator and the registry keys do not change, which is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.carddb import CardDB

__all__ = ["DECK_DIR", "load_deck", "write_deck", "available_decks", "deck_summary"]

DECK_DIR = Path(__file__).resolve().parent


def load_deck(name_or_path: str | Path) -> list[int]:
    """Load a 60-card list by registry name or explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = DECK_DIR / f"{name_or_path}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"No deck {name_or_path!r}; available: {sorted(available_decks())}"
        )
    out: list[int] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            out.append(int(token))
    return out


def write_deck(path: str | Path, deck: Iterable[int]) -> Path:
    """Write the plain form the submission bundle expects."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(int(c)) for c in deck) + "\n", encoding="utf-8")
    return p


def available_decks() -> list[str]:
    return sorted(p.stem for p in DECK_DIR.glob("*.csv"))


def deck_summary(deck: Iterable[int], db: CardDB) -> dict:
    """Human-readable composition, used by the deck report and by tests."""
    deck = list(deck)
    counts: dict[int, int] = {}
    for cid in deck:
        counts[cid] = counts.get(cid, 0) + 1

    lines: list[dict] = []
    n_pokemon = n_trainer = n_energy = 0
    energy_types: dict[int, int] = {}
    for cid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        c = db.card(cid)
        if c is None:
            continue
        if c.is_pokemon:
            n_pokemon += n
        elif c.is_energy:
            n_energy += n
            energy_types[c.energy_type] = energy_types.get(c.energy_type, 0) + n
        else:
            n_trainer += n
        lines.append(
            {
                "card_id": cid,
                "count": n,
                "name": c.name,
                "card_type": int(c.card_type),
                "roles": sorted(c.roles),
            }
        )
    return {
        "size": len(deck),
        "pokemon": n_pokemon,
        "trainer": n_trainer,
        "energy": n_energy,
        "energy_types": energy_types,
        "unique": len(counts),
        "problems": db.validate_deck(deck),
        "lines": lines,
    }
