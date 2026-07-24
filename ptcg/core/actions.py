"""Canonical action representation -- the contract that outlives Week 0.

The cabt engine violates the usual RL assumption of a fixed action vocabulary:
``obs["select"]["option"]`` is a variable-length list whose entries mean
completely different things depending on ``SelectType``/``SelectContext``. The
standard workaround (a giant fixed action head with masking) does not even type-
check here, because the same index means "discard the 3rd card in my hand" in
one decision and "put a damage counter on the opponent's 3rd benched Pokemon"
in the next.

So the engine's raw options are lifted once, here, into
:class:`ActionCandidate`: a typed record plus a **fixed-width float vector**
describing the option *semantically* (what kind of action, whose board it
touches, which card, which attack, how much damage it would do).

Every policy in the project consumes exactly this:

    scores = policy.score(view, candidates) -> array of shape (len(candidates),)

The Week-0 heuristic computes those scores with hand-written rules. The Week-1
behaviour-cloned network computes them with pointer-style cross-attention over
the very same vectors. Swapping the brain therefore requires changing one line
in ``main.py`` and touches no plumbing -- which is the whole point of building
this in Week 0 rather than Week 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .carddb import Attack, Card, CardDB, Role
from .enums import (
    N_AREA,
    N_CARD_TYPE,
    N_ENERGY,
    N_OPTION_TYPE,
    AreaType,
    EnergyType,
    OptionType,
)
from .obs import ObsView, PokemonView, damage_after_type, energy_shortfall

__all__ = ["ActionCandidate", "build_candidates", "FEATURE_NAMES", "FEATURE_DIM"]


# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------

_ROLE_ORDER: tuple[str, ...] = (
    Role.DRAW,
    Role.SEARCH,
    Role.BALL,
    Role.ENERGY_ACCEL,
    Role.RECOVERY,
    Role.SWITCH_SELF,
    Role.GUST,
    Role.HEAL,
    Role.DISRUPT,
    Role.COIN_FLIP,
    Role.SELF_DISCARD_COST,
    Role.DAMAGE_COUNTER,
    Role.STATUS,
    Role.SELF_LOCK,
    Role.TURN_END,
    Role.SELF_ENERGY_COST,
)


def _build_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names += [f"opt_type::{OptionType(i).name}" for i in range(N_OPTION_TYPE)]
    names += [f"area::{i}" for i in range(N_AREA)]
    names += [f"in_play_area::{i}" for i in range(N_AREA)]
    names += ["is_mine", "targets_opponent"]
    names += ["idx_norm", "number_norm", "count_norm", "tool_idx_norm", "energy_idx_norm"]
    # card block
    names += [f"card_type::{i}" for i in range(N_CARD_TYPE)]
    names += [f"card_energy::{i}" for i in range(N_ENERGY)]
    names += [
        "card_known",
        "card_hp_norm",
        "card_retreat_norm",
        "card_basic",
        "card_stage1",
        "card_stage2",
        "card_ex",
        "card_mega_ex",
        "card_tera",
        "card_ace_spec",
        "card_has_ability",
        "card_prizes_norm",
    ]
    names += [f"card_role::{r}" for r in _ROLE_ORDER]
    # attack block
    names += [
        "attack_known",
        "attack_damage_norm",
        "attack_cost_norm",
        "attack_colored_cost_norm",
        "attack_shortfall_norm",
        "attack_affordable",
        "attack_variable_damage",
    ]
    names += [f"attack_role::{r}" for r in _ROLE_ORDER]
    # tactical block (computed against the live board)
    names += [
        "est_damage_norm",
        "is_knockout",
        "ko_prizes_norm",
        "would_win_game",
        "target_hp_frac",
        "target_is_active",
        "target_energy_norm",
    ]
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _build_feature_names()
FEATURE_DIM: int = len(FEATURE_NAMES)

_OFF = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class ActionCandidate:
    """One legal option, lifted out of the raw dict."""

    index: int                       # index into select["option"] -- the action
    raw: dict[str, Any]
    option_type: int
    area: int | None
    target_index: int | None
    player_index: int | None
    in_play_area: int | None
    in_play_index: int | None
    card_id: int | None
    card: Card | None
    attack: Attack | None
    number: int | None
    count: int | None
    tool_index: int | None
    energy_index: int | None
    special_condition: int | None
    # tactical annotations
    target: PokemonView | None = None
    est_damage: int = 0
    is_knockout: bool = False
    ko_prizes: int = 0
    would_win_game: bool = False
    features: list[float] = field(default_factory=list)

    # -- convenience --------------------------------------------------------

    @property
    def is_mine(self) -> bool:
        return self.player_index is None or self._owner_is_me

    _owner_is_me: bool = True

    def has_role(self, role: str) -> bool:
        return bool(self.card and role in self.card.roles)

    def signature(self) -> tuple:
        """Stable identity for loop detection (index is not stable, this is)."""
        return (
            self.option_type,
            self.area,
            self.target_index,
            self.in_play_area,
            self.in_play_index,
            self.card_id,
            self.attack.attack_id if self.attack else None,
            self.number,
        )

    def describe(self) -> str:
        name = OptionType(self.option_type).name if self.option_type in list(OptionType) else str(self.option_type)
        bits = [name]
        if self.card:
            bits.append(self.card.name)
        if self.attack:
            bits.append(f"[{self.attack.name} {self.attack.damage}]")
        if self.is_knockout:
            bits.append(f"KO(+{self.ko_prizes})")
        elif self.est_damage:
            bits.append(f"dmg={self.est_damage}")
        return " ".join(bits)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _one_hot(vec: list[float], prefix: str, value: int | None, size: int) -> None:
    if value is None:
        return
    v = int(value)
    if 0 <= v < size:
        key = f"{prefix}{v}" if prefix.endswith("::") else prefix
        idx = _OFF.get(f"{prefix}{v}")
        if idx is not None:
            vec[idx] = 1.0


def _norm(x: float | None, scale: float) -> float:
    if x is None or scale <= 0:
        return 0.0
    return max(-4.0, min(4.0, float(x) / scale))


def build_candidates(view: ObsView, db: CardDB) -> list[ActionCandidate]:
    """Lift every legal option into an :class:`ActionCandidate`.

    Cost is O(len(options)) with a handful of dict lookups each; measured at
    well under 0.1 ms for typical option lists, which matters because this runs
    on every one of the ~200 decisions in a game inside a 600 s budget.
    """
    options = view.options
    me = view.my_index
    my_active = view.me.active
    opp_active = view.opp.active
    my_prizes = view.me.prizes_remaining

    out: list[ActionCandidate] = []
    for i, opt in enumerate(options):
        otype = int(opt.get("type", -1))
        area = opt.get("area")
        idx = opt.get("index")
        owner = opt.get("playerIndex")
        in_play_area = opt.get("inPlayArea")
        in_play_index = opt.get("inPlayIndex")

        card_id = view.resolve_card_id(opt)
        card = db.card(card_id)
        attack = db.attack(opt.get("attackId"))

        cand = ActionCandidate(
            index=i,
            raw=opt,
            option_type=otype,
            area=None if area is None else int(area),
            target_index=None if idx is None else int(idx),
            player_index=None if owner is None else int(owner),
            in_play_area=None if in_play_area is None else int(in_play_area),
            in_play_index=None if in_play_index is None else int(in_play_index),
            card_id=card_id,
            card=card,
            attack=attack,
            number=opt.get("number"),
            count=opt.get("count"),
            tool_index=opt.get("toolIndex"),
            energy_index=opt.get("energyIndex"),
            special_condition=opt.get("specialConditionType"),
        )
        cand._owner_is_me = (owner is None) or (int(owner) == me)

        # Which Pokemon does this option point at?
        if cand.area in (AreaType.ACTIVE, AreaType.BENCH):
            cand.target = view.pokemon_at(
                me if cand.player_index is None else cand.player_index,
                cand.area,
                cand.target_index,
            )
        elif cand.in_play_area in (AreaType.ACTIVE, AreaType.BENCH):
            cand.target = view.pokemon_at(me, cand.in_play_area, cand.in_play_index)

        # Attack maths against the current defender.
        if attack is not None and my_active is not None and opp_active is not None:
            cand.est_damage = damage_after_type(attack.damage, my_active.card, opp_active.card)
            if cand.est_damage >= opp_active.hp > 0:
                cand.is_knockout = True
                cand.ko_prizes = opp_active.prizes_given
                cand.would_win_game = cand.ko_prizes >= my_prizes

        cand.features = _featurise(cand, view, db)
        out.append(cand)
    return out


def _featurise(c: ActionCandidate, view: ObsView, db: CardDB) -> list[float]:
    v = [0.0] * FEATURE_DIM

    if 0 <= c.option_type < N_OPTION_TYPE:
        v[_OFF[f"opt_type::{OptionType(c.option_type).name}"]] = 1.0
    _one_hot(v, "area::", c.area, N_AREA)
    _one_hot(v, "in_play_area::", c.in_play_area, N_AREA)

    v[_OFF["is_mine"]] = 1.0 if c._owner_is_me else 0.0
    v[_OFF["targets_opponent"]] = 0.0 if c._owner_is_me else 1.0

    v[_OFF["idx_norm"]] = _norm(c.target_index, 10.0)
    v[_OFF["number_norm"]] = _norm(c.number, 10.0)
    v[_OFF["count_norm"]] = _norm(c.count, 5.0)
    v[_OFF["tool_idx_norm"]] = _norm(c.tool_index, 3.0)
    v[_OFF["energy_idx_norm"]] = _norm(c.energy_index, 8.0)

    card = c.card
    if card is not None:
        v[_OFF["card_known"]] = 1.0
        _one_hot(v, "card_type::", card.card_type, N_CARD_TYPE)
        _one_hot(v, "card_energy::", card.energy_type, N_ENERGY)
        v[_OFF["card_hp_norm"]] = _norm(card.hp, max(1, db.max_hp))
        v[_OFF["card_retreat_norm"]] = _norm(card.retreat_cost, 4.0)
        v[_OFF["card_basic"]] = float(card.basic)
        v[_OFF["card_stage1"]] = float(card.stage1)
        v[_OFF["card_stage2"]] = float(card.stage2)
        v[_OFF["card_ex"]] = float(card.ex)
        v[_OFF["card_mega_ex"]] = float(card.mega_ex)
        v[_OFF["card_tera"]] = float(card.tera)
        v[_OFF["card_ace_spec"]] = float(card.ace_spec)
        v[_OFF["card_has_ability"]] = float(card.has_ability)
        v[_OFF["card_prizes_norm"]] = _norm(card.prizes_when_koed, 3.0)
        for r in _ROLE_ORDER:
            if r in card.roles:
                v[_OFF[f"card_role::{r}"]] = 1.0

    atk = c.attack
    if atk is not None:
        v[_OFF["attack_known"]] = 1.0
        v[_OFF["attack_damage_norm"]] = _norm(atk.damage, max(1, db.max_damage))
        v[_OFF["attack_cost_norm"]] = _norm(atk.cost, 5.0)
        v[_OFF["attack_colored_cost_norm"]] = _norm(len(atk.colored_cost), 5.0)
        my_active = view.me.active
        short = energy_shortfall(my_active.energies, atk.energies) if my_active else len(atk.energies)
        v[_OFF["attack_shortfall_norm"]] = _norm(short, 5.0)
        v[_OFF["attack_affordable"]] = 1.0 if short == 0 else 0.0
        v[_OFF["attack_variable_damage"]] = float(atk.is_variable_damage)
        for r in _ROLE_ORDER:
            if r in atk.roles:
                v[_OFF[f"attack_role::{r}"]] = 1.0

    v[_OFF["est_damage_norm"]] = _norm(c.est_damage, max(1, db.max_damage))
    v[_OFF["is_knockout"]] = float(c.is_knockout)
    v[_OFF["ko_prizes_norm"]] = _norm(c.ko_prizes, 3.0)
    v[_OFF["would_win_game"]] = float(c.would_win_game)

    t = c.target
    if t is not None:
        v[_OFF["target_hp_frac"]] = t.hp_fraction
        v[_OFF["target_is_active"]] = 1.0 if t.area == AreaType.ACTIVE else 0.0
        v[_OFF["target_energy_norm"]] = _norm(t.n_energy, 5.0)
    return v
