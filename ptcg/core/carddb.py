"""Card + attack database, derived from the engine's own tables.

Design note (this is the key Week-0 decision that the rest of the project
rests on):

The blueprint proposed parsing ``EN_Card_Data.csv``. That CSV is one row *per
attack*, uses a mix of English and Japanese type glyphs (``竜`` for Dragon),
encodes energy costs as ``{L}{L}{L}●`` strings, and -- critically -- contains
no ``attackId``. But ``Option.attackId`` is exactly what the agent has to
reason about when it is offered an attack. Joining the CSV to the engine by
name is lossy and fragile.

The shared library exports ``AllCard`` and ``AllAttack``, which return the
simulator's authoritative tables including the integer ``attackId`` keys, typed
enums, evolution links and full rules text. We build the database from those
and treat the CSV as an *optional* source of cosmetic metadata (expansion,
collection number) only. Consequences:

* card ids, attack ids and enums cannot drift from the simulator;
* the submission bundle needs no data files at all;
* new card releases are picked up automatically when the engine is updated.

Semantic text (``skills[].text`` and ``attacks[].text``) is retained verbatim so
that Week 1 can attach frozen sentence-transformer embeddings without changing
any interface here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from .enums import CardType, EnergyType
from .engine import get_engine

__all__ = ["Card", "Attack", "CardDB", "get_card_db", "Role"]


# ---------------------------------------------------------------------------
# Role tagging
# ---------------------------------------------------------------------------

class Role:
    """Coarse functional tags derived from rules text.

    These are deliberately *pattern-derived rather than hardcoded per card ID*:
    a card printed next month gets tagged correctly with no code change, and the
    heuristic agent therefore generalises across the whole ~1.3k card pool. The
    same tags become categorical features for the neural policy in Week 1.
    """

    DRAW = "draw"
    SEARCH = "search"
    BALL = "ball"              # search specifically for Pokemon
    ENERGY_ACCEL = "energy_accel"
    RECOVERY = "recovery"      # discard pile -> hand/deck
    SWITCH_SELF = "switch_self"
    GUST = "gust"              # drag opponent's benched Pokemon active
    HEAL = "heal"
    DISRUPT = "disrupt"        # hits opponent's hand / board resources
    COIN_FLIP = "coin_flip"
    SELF_DISCARD_COST = "self_discard_cost"
    DAMAGE_COUNTER = "damage_counter"
    STATUS = "status"
    SELF_LOCK = "self_lock"    # "during your next turn, this Pokemon can't ..."
    TURN_END = "turn_end"      # "Your turn ends."
    #: The effect is paid for by discarding Energy from the Pokemon that owns
    #: it (e.g. Iono's Kilowattrel's *Flashing Draw*). Easy to miss and
    #: expensive: a naive scorer happily strips its own attacker to draw cards.
    SELF_ENERGY_COST = "self_energy_cost"
    #: Week 6: damage/effect scales with a count of same-affiliation Pokemon
    #: in play (e.g. "does 30 damage for each of your Team Rocket's Pokemon
    #: in play"). Deck-agnostic on purpose -- this text pattern recurs across
    #: many trainer-affiliation "tribal" cards (Team Rocket's, Marnie's,
    #: Iono's, ...), not just one archetype.
    BOARD_SCALING = "board_scaling"
    #: Week 6: the Pokemon cannot attack unless a minimum count of
    #: same-affiliation Pokemon are in play (e.g. Team Rocket's Mewtwo ex).
    #: Same deck-agnostic reasoning as ``BOARD_SCALING``.
    BOARD_GATED = "board_gated"
    #: Week 8: "shuffle your hand into your deck" style effects (Judge/Iono-
    #: style Supporters). Anything still in hand when this resolves is gone
    #: for free -- a real reason to spend marginal cards *before* playing it.
    HAND_SHUFFLE = "hand_shuffle"


_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (Role.DRAW, re.compile(r"\bdraw\b", re.I)),
    (Role.SEARCH, re.compile(r"search your deck", re.I)),
    (Role.BALL, re.compile(r"search your deck for[^.]*\bPok[ée]mon\b", re.I)),
    (Role.ENERGY_ACCEL, re.compile(r"attach[^.]*Energy[^.]*(from your (discard pile|hand))", re.I)),
    (Role.RECOVERY, re.compile(r"from your discard pile", re.I)),
    (Role.SWITCH_SELF, re.compile(r"switch your Active Pok[ée]mon", re.I)),
    (Role.GUST, re.compile(r"(switch in 1 of your opponent|switch out your opponent)", re.I)),
    (Role.HEAL, re.compile(r"\bheal\b", re.I)),
    (Role.DISRUPT, re.compile(r"(your opponent (discards|shuffles|reveals)|each player shuffles)", re.I)),
    (Role.COIN_FLIP, re.compile(r"flip \d* ?coins?|flip a coin", re.I)),
    (Role.SELF_DISCARD_COST, re.compile(r"only if you discard", re.I)),
    (Role.DAMAGE_COUNTER, re.compile(r"damage counters?", re.I)),
    (Role.STATUS, re.compile(r"\b(Asleep|Burned|Confused|Paralyzed|Poisoned)\b")),
    (Role.SELF_LOCK, re.compile(r"during your next turn, this Pok[ée]mon can", re.I)),
    (Role.TURN_END, re.compile(r"your turn ends", re.I)),
    (
        Role.SELF_ENERGY_COST,
        re.compile(r"discard (a|an|\d+)[^.]*Energy from this Pok[ée]mon", re.I),
    ),
    (
        Role.BOARD_SCALING,
        re.compile(r"for each of your[^.]*Pok[ée]mon (you have )?in play", re.I),
    ),
    (
        Role.BOARD_GATED,
        re.compile(r"unless you have[^.]*\d+[^.]*or more[^.]*Pok[ée]mon in play", re.I),
    ),
    (
        Role.HAND_SHUFFLE,
        re.compile(r"shuffle[^.]*your hand into your deck", re.I),
    ),
)

#: Possessive trainer-affiliation prefix (``Team Rocket's``, ``Marnie's``,
#: ``Iono's``, ...) -- a real, recurring Pokemon TCG naming convention, not
#: specific to any one archetype. Purely name-derived, so it applies
#: uniformly to every card that uses it without a lookup table.
_AFFILIATION_PATTERN = re.compile(r"^([A-Z][\w' ]*?'s) ")


def _affiliation(name: str) -> str:
    m = _AFFILIATION_PATTERN.match(name)
    return m.group(1) if m else ""


def _tag(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(name for name, pat in _ROLE_PATTERNS if pat.search(text))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Attack:
    attack_id: int
    name: str
    text: str
    damage: int
    energies: tuple[int, ...]
    roles: frozenset[str] = field(default_factory=frozenset)

    @property
    def cost(self) -> int:
        return len(self.energies)

    @property
    def colored_cost(self) -> tuple[int, ...]:
        """Energy requirements excluding colorless."""
        return tuple(e for e in self.energies if e != EnergyType.COLORLESS)

    @property
    def is_variable_damage(self) -> bool:
        """Damage printed as e.g. ``60x`` -- the engine reports the base only."""
        return bool(self.text) and ("×" in self.text or " x " in self.text)


@dataclass(frozen=True)
class Card:
    card_id: int
    name: str
    card_type: int
    energy_type: int
    retreat_cost: int
    hp: int
    weakness: int
    resistance: int
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    tera: bool
    ace_spec: bool
    evolves_from: str | None
    skill_names: tuple[str, ...]
    skill_texts: tuple[str, ...]
    attack_ids: tuple[int, ...]
    roles: frozenset[str] = field(default_factory=frozenset)
    # cosmetic, only present when the CSV is available
    expansion: str = ""
    collection_no: str = ""

    # -- convenience --------------------------------------------------------

    @property
    def is_pokemon(self) -> bool:
        return self.card_type == CardType.POKEMON

    @property
    def is_energy(self) -> bool:
        return self.card_type in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)

    @property
    def is_basic_energy(self) -> bool:
        return self.card_type == CardType.BASIC_ENERGY

    @property
    def is_trainer(self) -> bool:
        return self.card_type in (
            CardType.ITEM,
            CardType.TOOL,
            CardType.SUPPORTER,
            CardType.STADIUM,
        )

    @property
    def is_supporter(self) -> bool:
        return self.card_type == CardType.SUPPORTER

    @property
    def has_ability(self) -> bool:
        return bool(self.skill_names)

    @property
    def rule_box(self) -> bool:
        """ex / Mega ex Pokemon give up 2+ prizes when knocked out."""
        return self.ex or self.mega_ex

    @property
    def prizes_when_koed(self) -> int:
        if self.mega_ex:
            return 3
        if self.ex:
            return 2
        return 1

    @property
    def all_text(self) -> str:
        return " ".join(self.skill_texts)

    @property
    def affiliation(self) -> str:
        """Possessive trainer-affiliation prefix parsed from this card's own
        ``name`` (e.g. ``"Team Rocket's"`` from "Team Rocket's Mewtwo ex"),
        or ``""`` when the card has none. Purely name-derived -- applies to
        any card using this real, recurring naming convention, not
        hardcoded to one archetype."""
        return _affiliation(self.name)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class CardDB:
    """Immutable lookup over the engine's card and attack tables."""

    def __init__(self, cards: dict[int, Card], attacks: dict[int, Attack]) -> None:
        self._cards = cards
        self._attacks = attacks
        self._by_name: dict[str, list[Card]] = {}
        for c in cards.values():
            self._by_name.setdefault(c.name, []).append(c)
        # Name -> cards that evolve from it. The engine links evolution by the
        # *name* string, so multiple printings of "Riolu" all feed Mega Lucario.
        self._evolves_to: dict[str, list[Card]] = {}
        for c in cards.values():
            if c.evolves_from:
                self._evolves_to.setdefault(c.evolves_from, []).append(c)
        self._max_hp = max((c.hp for c in cards.values()), default=1) or 1
        self._max_damage = max((a.damage for a in attacks.values()), default=1) or 1

    # -- lookups ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._cards

    def card(self, card_id: int | None) -> Card | None:
        if card_id is None:
            return None
        return self._cards.get(int(card_id))

    def attack(self, attack_id: int | None) -> Attack | None:
        if attack_id is None:
            return None
        return self._attacks.get(int(attack_id))

    def attacks_of(self, card_id: int | None) -> list[Attack]:
        c = self.card(card_id)
        if c is None:
            return []
        return [a for a in (self.attack(i) for i in c.attack_ids) if a is not None]

    def by_name(self, name: str) -> list[Card]:
        return list(self._by_name.get(name, ()))

    def evolutions_of(self, name: str) -> list[Card]:
        return list(self._evolves_to.get(name, ()))

    def all_cards(self) -> Iterable[Card]:
        return self._cards.values()

    @property
    def max_hp(self) -> int:
        return self._max_hp

    @property
    def max_damage(self) -> int:
        return self._max_damage

    # -- deck legality ------------------------------------------------------

    def validate_deck(self, deck: Sequence[int]) -> list[str]:
        """Return a list of rule violations; empty means the deck is legal.

        Enforces the constraints the engine will reject at ``battle_start``:
        exactly 60 cards, at most 4 copies of any card that is not a Basic
        Energy, at most one ACE SPEC in total, at least one Basic Pokemon, and
        every evolution line rooted in something the deck can actually play.
        """
        problems: list[str] = []
        if len(deck) != 60:
            problems.append(f"deck has {len(deck)} cards, must be exactly 60")

        unknown = sorted({cid for cid in deck if cid not in self._cards})
        if unknown:
            problems.append(f"unknown card ids: {unknown}")

        # The 4-copy limit applies per *card name*, not per card ID: the pool
        # contains several distinct printings of e.g. "Riolu" with different IDs
        # and HP, and four of each would be illegal.
        name_counts: dict[str, int] = {}
        ace_specs = 0
        for cid in deck:
            c = self.card(cid)
            if c is None:
                continue
            if c.ace_spec:
                ace_specs += 1
            if not c.is_basic_energy:
                name_counts[c.name] = name_counts.get(c.name, 0) + 1

        for name, n in name_counts.items():
            if n > 4:
                problems.append(f"{n}x {name} exceeds the 4-copy-per-name limit")
        if ace_specs > 1:
            problems.append(f"{ace_specs} ACE SPEC cards; at most 1 is allowed")

        names = {self.card(cid).name for cid in deck if self.card(cid)}
        basics = [cid for cid in deck if (c := self.card(cid)) and c.is_pokemon and c.basic]
        if not basics:
            problems.append("no Basic Pokemon: every opening hand would be a mulligan")

        for cid in set(deck):
            c = self.card(cid)
            if c is None or not c.is_pokemon or not c.evolves_from:
                continue
            if c.evolves_from not in names:
                problems.append(
                    f"{c.name} evolves from {c.evolves_from!r}, which is not in the deck"
                )
        return problems

    # -- serialisation ------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "cards": [
                {
                    "card_id": c.card_id,
                    "name": c.name,
                    "card_type": c.card_type,
                    "energy_type": c.energy_type,
                    "retreat_cost": c.retreat_cost,
                    "hp": c.hp,
                    "weakness": c.weakness,
                    "resistance": c.resistance,
                    "basic": c.basic,
                    "stage1": c.stage1,
                    "stage2": c.stage2,
                    "ex": c.ex,
                    "mega_ex": c.mega_ex,
                    "tera": c.tera,
                    "ace_spec": c.ace_spec,
                    "evolves_from": c.evolves_from,
                    "skill_names": list(c.skill_names),
                    "skill_texts": list(c.skill_texts),
                    "attack_ids": list(c.attack_ids),
                    "roles": sorted(c.roles),
                    "expansion": c.expansion,
                    "collection_no": c.collection_no,
                }
                for c in sorted(self._cards.values(), key=lambda x: x.card_id)
            ],
            "attacks": [
                {
                    "attack_id": a.attack_id,
                    "name": a.name,
                    "text": a.text,
                    "damage": a.damage,
                    "energies": list(a.energies),
                    "roles": sorted(a.roles),
                }
                for a in sorted(self._attacks.values(), key=lambda x: x.attack_id)
            ],
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _build_from_tables(
    raw_cards: list[dict[str, Any]],
    raw_attacks: list[dict[str, Any]],
    csv_meta: dict[int, tuple[str, str]] | None = None,
) -> CardDB:
    attacks: dict[int, Attack] = {}
    for a in raw_attacks:
        text = a.get("text") or ""
        attacks[a["attackId"]] = Attack(
            attack_id=a["attackId"],
            name=(a.get("name") or "").strip(),
            text=text,
            damage=int(a.get("damage") or 0),
            energies=tuple(int(e) for e in (a.get("energies") or ())),
            roles=_tag(text),
        )

    cards: dict[int, Card] = {}
    for c in raw_cards:
        skills = c.get("skills") or []
        skill_texts = tuple((s.get("text") or "") for s in skills)
        attack_ids = tuple(int(i) for i in (c.get("attacks") or ()))
        combined = " ".join(skill_texts) + " " + " ".join(
            attacks[i].text for i in attack_ids if i in attacks
        )
        exp, coll = (csv_meta or {}).get(c["cardId"], ("", ""))
        cards[c["cardId"]] = Card(
            card_id=c["cardId"],
            name=(c.get("name") or "").strip(),
            card_type=int(c.get("cardType") or 0),
            energy_type=int(c.get("energyType") or 0),
            retreat_cost=int(c.get("retreatCost") or 0),
            hp=int(c.get("hp") or 0),
            weakness=int(c.get("weakness") or 0),
            resistance=int(c.get("resistance") or 0),
            basic=bool(c.get("basic")),
            stage1=bool(c.get("stage1")),
            stage2=bool(c.get("stage2")),
            ex=bool(c.get("ex")),
            mega_ex=bool(c.get("megaEx")),
            tera=bool(c.get("tera")),
            ace_spec=bool(c.get("aceSpec")),
            evolves_from=(c.get("evolvesFrom") or None),
            skill_names=tuple((s.get("name") or "").strip() for s in skills),
            skill_texts=skill_texts,
            attack_ids=attack_ids,
            roles=_tag(combined),
            expansion=exp,
            collection_no=coll,
        )
    return CardDB(cards, attacks)


def _read_csv_meta(csv_path: Path) -> dict[int, tuple[str, str]]:
    """Pull the two fields the engine table lacks: expansion + collection number."""
    import csv as _csv

    out: dict[int, tuple[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            try:
                cid = int(row["Card ID"])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault(cid, (row.get("Expansion", ""), row.get("Collection No.", "")))
    return out


@lru_cache(maxsize=1)
def get_card_db(csv_path: str | None = None, cache_path: str | None = None) -> CardDB:
    """Build (or load) the card database.

    Order of preference:
      1. the live engine (``AllCard`` / ``AllAttack``) -- always correct;
      2. a JSON cache written by ``tools/dump_engine_db.py`` -- used when the
         engine is unavailable (e.g. analysis on an arm64 laptop).
    """
    csv_meta = None
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            csv_meta = _read_csv_meta(p)

    try:
        eng = get_engine()
        return _build_from_tables(eng.all_card_data(), eng.all_attack(), csv_meta)
    except Exception:
        if cache_path and Path(cache_path).exists():
            blob = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            cards = {
                c["card_id"]: Card(
                    card_id=c["card_id"],
                    name=c["name"],
                    card_type=c["card_type"],
                    energy_type=c["energy_type"],
                    retreat_cost=c["retreat_cost"],
                    hp=c["hp"],
                    weakness=c["weakness"],
                    resistance=c["resistance"],
                    basic=c["basic"],
                    stage1=c["stage1"],
                    stage2=c["stage2"],
                    ex=c["ex"],
                    mega_ex=c["mega_ex"],
                    tera=c["tera"],
                    ace_spec=c["ace_spec"],
                    evolves_from=c["evolves_from"],
                    skill_names=tuple(c["skill_names"]),
                    skill_texts=tuple(c["skill_texts"]),
                    attack_ids=tuple(c["attack_ids"]),
                    roles=frozenset(c["roles"]),
                    expansion=c.get("expansion", ""),
                    collection_no=c.get("collection_no", ""),
                )
                for c in blob["cards"]
            }
            attacks = {
                a["attack_id"]: Attack(
                    attack_id=a["attack_id"],
                    name=a["name"],
                    text=a["text"],
                    damage=a["damage"],
                    energies=tuple(a["energies"]),
                    roles=frozenset(a["roles"]),
                )
                for a in blob["attacks"]
            }
            return CardDB(cards, attacks)
        raise
