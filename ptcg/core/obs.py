"""Zero-copy typed views over the raw ``obs_dict``.

The engine hands the agent a nested dict. Every downstream consumer -- the
heuristic scorer today, the set-transformer encoder in Week 1, the ISMCTS
determinizer in Week 3 -- needs the same derived quantities: whose turn is it,
what is on the board, can this attack actually be paid for, how much damage
would it really do after weakness.

Putting that logic here (once, tested) rather than in the agent means the
neural policy inherits it for free and the two brains can never disagree about
the state they are looking at.

Nothing here mutates the observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .carddb import Attack, Card, CardDB
from .enums import AreaType, CardType, EnergyType, SelectContext, SelectType

__all__ = ["PokemonView", "PlayerView", "ObsView", "damage_after_type", "energy_shortfall"]


# ---------------------------------------------------------------------------
# Rules maths
# ---------------------------------------------------------------------------

#: The simulator uses the modern Standard convention: Resistance subtracts a
#: flat 30 damage. Verified against HP-change logs in ``tools/verify_rules.py``.
RESISTANCE_REDUCTION = 30


def damage_after_type(
    base: int,
    attacker: Card | None,
    defender: Card | None,
) -> int:
    """Apply Weakness (x2) and Resistance (-30) to a base damage number.

    Returns a *lower bound estimate*: attack text effects (bonus damage, damage
    modifiers, opponent-side reduction) are not modelled here. Deliberately so
    -- the heuristic only needs a consistent ordering, and Week 1's value head
    learns the residual from data.
    """
    if base <= 0 or attacker is None or defender is None:
        return max(0, base)
    dmg = base
    if defender.weakness and defender.weakness == attacker.energy_type:
        dmg *= 2
    if defender.resistance and defender.resistance == attacker.energy_type:
        dmg = max(0, dmg - RESISTANCE_REDUCTION)
    return dmg


def energy_shortfall(available: Sequence[int], required: Sequence[int]) -> int:
    """How many more Energy are needed to pay ``required`` from ``available``.

    ``available`` and ``required`` are lists of :class:`EnergyType` ints, as the
    engine reports them (``Pokemon.energies`` and ``Attack.energies``).

    Coloured requirements must be met by a matching type or by RAINBOW; leftover
    Energy of any type pays Colorless. Greedy matching is optimal here because
    RAINBOW is the only wildcard and Colorless is the only wildcard requirement.
    """
    pool: dict[int, int] = {}
    for e in available:
        pool[e] = pool.get(e, 0) + 1

    missing = 0
    colorless_needed = 0
    for req in required:
        if req == EnergyType.COLORLESS:
            colorless_needed += 1
            continue
        if pool.get(req):
            pool[req] -= 1
        elif pool.get(EnergyType.RAINBOW):
            pool[EnergyType.RAINBOW] -= 1
        elif req == EnergyType.TEAM_ROCKET and (
            pool.get(EnergyType.PSYCHIC) or pool.get(EnergyType.DARKNESS)
        ):
            key = EnergyType.PSYCHIC if pool.get(EnergyType.PSYCHIC) else EnergyType.DARKNESS
            pool[key] -= 1
        else:
            missing += 1

    remaining = sum(v for v in pool.values() if v > 0)
    if remaining < colorless_needed:
        missing += colorless_needed - remaining
    return missing


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PokemonView:
    raw: dict[str, Any]
    db: CardDB
    area: int
    index: int
    player_index: int

    @property
    def card_id(self) -> int | None:
        return self.raw.get("id")

    @property
    def card(self) -> Card | None:
        return self.db.card(self.card_id)

    @property
    def serial(self) -> int | None:
        return self.raw.get("serial")

    @property
    def hp(self) -> int:
        return int(self.raw.get("hp") or 0)

    @property
    def max_hp(self) -> int:
        return int(self.raw.get("maxHp") or 0)

    @property
    def damage_taken(self) -> int:
        return max(0, self.max_hp - self.hp)

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0

    @property
    def energies(self) -> list[int]:
        return list(self.raw.get("energies") or ())

    @property
    def n_energy(self) -> int:
        return len(self.energies)

    @property
    def appeared_this_turn(self) -> bool:
        return bool(self.raw.get("appearThisTurn"))

    @property
    def tools(self) -> list[dict[str, Any]]:
        return list(self.raw.get("tools") or ())

    @property
    def prizes_given(self) -> int:
        c = self.card
        return c.prizes_when_koed if c else 1

    def usable_attacks(self) -> list[tuple[Attack, int]]:
        """``(attack, shortfall)`` for every attack this Pokemon knows."""
        out: list[tuple[Attack, int]] = []
        for atk in self.db.attacks_of(self.card_id):
            out.append((atk, energy_shortfall(self.energies, atk.energies)))
        return out

    def best_damage_against(self, target: "PokemonView | None") -> int:
        best = 0
        tcard = target.card if target else None
        for atk, short in self.usable_attacks():
            if short > 0:
                continue
            best = max(best, damage_after_type(atk.damage, self.card, tcard))
        return best


@dataclass(frozen=True)
class PlayerView:
    raw: dict[str, Any]
    db: CardDB
    player_index: int

    # -- board --------------------------------------------------------------

    @property
    def active(self) -> PokemonView | None:
        arr = self.raw.get("active") or []
        if not arr or arr[0] is None:
            return None
        return PokemonView(arr[0], self.db, AreaType.ACTIVE, 0, self.player_index)

    @property
    def bench(self) -> list[PokemonView]:
        return [
            PokemonView(p, self.db, AreaType.BENCH, i, self.player_index)
            for i, p in enumerate(self.raw.get("bench") or ())
            if p is not None
        ]

    @property
    def bench_max(self) -> int:
        return int(self.raw.get("benchMax") or 5)

    @property
    def bench_open(self) -> int:
        return max(0, self.bench_max - len(self.bench))

    def in_play(self) -> Iterator[PokemonView]:
        a = self.active
        if a is not None:
            yield a
        yield from self.bench

    # -- zones --------------------------------------------------------------

    @property
    def hand(self) -> list[dict[str, Any]] | None:
        """``None`` for the opponent (we only get ``handCount``)."""
        return self.raw.get("hand")

    @property
    def hand_count(self) -> int:
        return int(self.raw.get("handCount") or 0)

    @property
    def hand_cards(self) -> list[Card | None]:
        h = self.hand or []
        return [self.db.card(c.get("id")) for c in h]

    @property
    def deck_count(self) -> int:
        return int(self.raw.get("deckCount") or 0)

    @property
    def discard(self) -> list[dict[str, Any]]:
        return list(self.raw.get("discard") or ())

    @property
    def prize(self) -> list[dict[str, Any] | None]:
        return list(self.raw.get("prize") or ())

    @property
    def prizes_remaining(self) -> int:
        return len(self.prize)

    # -- status -------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return bool(self.raw.get("poisoned"))

    @property
    def burned(self) -> bool:
        return bool(self.raw.get("burned"))

    @property
    def asleep(self) -> bool:
        return bool(self.raw.get("asleep"))

    @property
    def paralyzed(self) -> bool:
        return bool(self.raw.get("paralyzed"))

    @property
    def confused(self) -> bool:
        return bool(self.raw.get("confused"))

    @property
    def any_status(self) -> bool:
        return self.poisoned or self.burned or self.asleep or self.paralyzed or self.confused

    @property
    def active_immobilised(self) -> bool:
        """Asleep/Paralyzed prevent attacking and retreating."""
        return self.asleep or self.paralyzed

    # -- derived ------------------------------------------------------------

    def count_in_hand(self, predicate) -> int:
        return sum(1 for c in self.hand_cards if c is not None and predicate(c))

    @property
    def basic_energy_in_hand(self) -> int:
        return self.count_in_hand(lambda c: c.is_basic_energy)

    @property
    def pokemon_in_hand(self) -> int:
        return self.count_in_hand(lambda c: c.is_pokemon)


class ObsView:
    """Read-only facade over one ``obs_dict``."""

    __slots__ = ("raw", "db", "_state", "_select")

    def __init__(self, obs: dict[str, Any], db: CardDB) -> None:
        self.raw = obs
        self.db = db
        self._state = obs.get("current")
        self._select = obs.get("select")

    # -- phase --------------------------------------------------------------

    @property
    def is_deck_selection(self) -> bool:
        """The very first call: the engine wants our 60 card IDs."""
        return self._select is None

    @property
    def state(self) -> dict[str, Any] | None:
        return self._state

    @property
    def select(self) -> dict[str, Any] | None:
        return self._select

    @property
    def logs(self) -> list[dict[str, Any]]:
        return list(self.raw.get("logs") or ())

    @property
    def remaining_time(self) -> float | None:
        v = self.raw.get("remainingOverageTime")
        return float(v) if v is not None else None

    @property
    def search_begin_input(self) -> str | None:
        """Opaque blob the engine wants back for ``search_begin`` (Week 3)."""
        return self.raw.get("search_begin_input")

    # -- select -------------------------------------------------------------

    @property
    def select_type(self) -> int:
        return int((self._select or {}).get("type", -1))

    @property
    def context(self) -> int:
        return int((self._select or {}).get("context", -1))

    @property
    def options(self) -> list[dict[str, Any]]:
        return list((self._select or {}).get("option") or ())

    @property
    def min_count(self) -> int:
        return int((self._select or {}).get("minCount", 0) or 0)

    @property
    def max_count(self) -> int:
        return int((self._select or {}).get("maxCount", 0) or 0)

    @property
    def remain_energy_cost(self) -> int:
        return int((self._select or {}).get("remainEnergyCost", 0) or 0)

    @property
    def remain_damage_counter(self) -> int:
        return int((self._select or {}).get("remainDamageCounter", 0) or 0)

    @property
    def select_deck(self) -> list[dict[str, Any]] | None:
        return (self._select or {}).get("deck")

    @property
    def context_card(self) -> Card | None:
        cc = (self._select or {}).get("contextCard")
        return self.db.card(cc.get("id")) if cc else None

    @property
    def effect_card(self) -> Card | None:
        e = (self._select or {}).get("effect")
        return self.db.card(e.get("id")) if e else None

    # -- board --------------------------------------------------------------

    @property
    def my_index(self) -> int:
        return int((self._state or {}).get("yourIndex", 0) or 0)

    @property
    def me(self) -> PlayerView:
        players = (self._state or {}).get("players") or [{}, {}]
        return PlayerView(players[self.my_index], self.db, self.my_index)

    @property
    def opp(self) -> PlayerView:
        players = (self._state or {}).get("players") or [{}, {}]
        return PlayerView(players[1 - self.my_index], self.db, 1 - self.my_index)

    def player(self, index: int) -> PlayerView:
        players = (self._state or {}).get("players") or [{}, {}]
        return PlayerView(players[index], self.db, index)

    # -- turn ---------------------------------------------------------------

    @property
    def turn(self) -> int:
        return int((self._state or {}).get("turn", 0) or 0)

    @property
    def turn_action_count(self) -> int:
        return int((self._state or {}).get("turnActionCount", 0) or 0)

    @property
    def first_player(self) -> int:
        return int((self._state or {}).get("firstPlayer", -1))

    @property
    def i_am_first(self) -> bool:
        return self.first_player == self.my_index

    @property
    def supporter_played(self) -> bool:
        return bool((self._state or {}).get("supporterPlayed"))

    @property
    def stadium_played(self) -> bool:
        return bool((self._state or {}).get("stadiumPlayed"))

    @property
    def energy_attached(self) -> bool:
        return bool((self._state or {}).get("energyAttached"))

    @property
    def retreated(self) -> bool:
        return bool((self._state or {}).get("retreated"))

    @property
    def stadium(self) -> Card | None:
        arr = (self._state or {}).get("stadium") or []
        return self.db.card(arr[0].get("id")) if arr and arr[0] else None

    @property
    def looking(self) -> list[dict[str, Any]] | None:
        return (self._state or {}).get("looking")

    @property
    def result(self) -> int:
        return int((self._state or {}).get("result", -1))

    @property
    def prize_diff(self) -> int:
        """Positive means we are ahead in the prize race."""
        return self.opp.prizes_remaining - self.me.prizes_remaining

    # -- resolution ---------------------------------------------------------

    def pokemon_at(self, player_index: int, area: int | None, index: int | None) -> PokemonView | None:
        if area is None or index is None:
            return None
        p = self.player(player_index)
        if area == AreaType.ACTIVE:
            return p.active
        if area == AreaType.BENCH:
            bench = p.bench
            return bench[index] if 0 <= index < len(bench) else None
        return None

    def resolve_card_id(self, option: dict[str, Any]) -> int | None:
        """Best-effort: what card does this option actually refer to?

        Face-down zones (prize, opponent hand) legitimately resolve to ``None``;
        that absence is itself a feature the belief head will learn from.
        """
        if option.get("cardId") is not None:
            return option["cardId"]

        area = option.get("area")
        index = option.get("index")
        owner = option.get("playerIndex")
        owner = self.my_index if owner is None else int(owner)

        # PLAY carries only `index`, always into our own hand.
        if area is None and index is not None and option.get("type") == 7:
            hand = self.me.hand or []
            return hand[index].get("id") if 0 <= index < len(hand) else None

        if area is None or index is None:
            return None

        p = self.player(owner)
        if area == AreaType.HAND:
            hand = p.hand or []
            return hand[index].get("id") if 0 <= index < len(hand) else None
        if area == AreaType.DISCARD:
            d = p.discard
            return d[index].get("id") if 0 <= index < len(d) else None
        if area == AreaType.DECK:
            deck = self.select_deck or []
            return deck[index].get("id") if 0 <= index < len(deck) else None
        if area == AreaType.LOOKING:
            look = self.looking or []
            c = look[index] if 0 <= index < len(look) else None
            return c.get("id") if c else None
        if area in (AreaType.ACTIVE, AreaType.BENCH):
            pk = self.pokemon_at(owner, area, index)
            return pk.card_id if pk else None
        if area == AreaType.STADIUM:
            s = self.stadium
            return s.card_id if s else None
        if area == AreaType.PRIZE:
            pr = p.prize
            c = pr[index] if 0 <= index < len(pr) else None
            return c.get("id") if c else None
        return None
