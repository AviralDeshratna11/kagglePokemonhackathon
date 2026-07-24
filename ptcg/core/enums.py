"""Engine enumerations for the cabt Pokemon TCG simulator.

Transcribed verbatim from the official API reference
(https://matsuoinstitute.github.io/cabt/api.html) and cross-validated against
the values actually emitted by ``libcg.so`` during self-play
(see ``ptcg/tools/mine_schema.py``).

These are plain ``IntEnum``s so they compare equal to the raw ints in the
observation dict with zero conversion cost.
"""

from __future__ import annotations

from enum import IntEnum


class AreaType(IntEnum):
    """Where a card lives. Used by Option.area / Option.inPlayArea / Log.fromArea."""

    NONE = 0
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class EnergyType(IntEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10       # provides every type
    TEAM_ROCKET = 11   # Psychic or Darkness


class CardType(IntEnum):
    POKEMON = 0
    ITEM = 1
    TOOL = 2
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class SpecialConditionType(IntEnum):
    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class SelectType(IntEnum):
    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class LogType(IntEnum):
    SHUFFLE = 0
    HAS_BASIC_POKEMON = 1
    TURN_START = 2
    TURN_END = 3
    DRAW = 4
    DRAW_REVERSE = 5
    MOVE_CARD = 6
    MOVE_CARD_REVERSE = 7
    SWITCH = 8
    CHANGE = 9
    PLAY = 10
    ATTACH = 11
    EVOLVE = 12
    DEVOLVE = 13
    MOVE_ATTACHED = 14
    ATTACK = 15
    HP_CHANGE = 16
    POISONED = 17
    BURNED = 18
    ASLEEP = 19
    PARALYZED = 20
    CONFUSED = 21
    COIN = 22
    RESULT = 23


class ResultReason(IntEnum):
    """Log.reason on a RESULT log."""

    ZERO_PRIZES = 1
    NO_DECK = 2
    NO_ACTIVE_POKEMON = 3
    CARD_EFFECT = 4


# ---------------------------------------------------------------------------
# Derived constants used all over the codebase.
# ---------------------------------------------------------------------------

#: Areas that hold a Pokemon in play.
IN_PLAY_AREAS = (AreaType.ACTIVE, AreaType.BENCH)

#: EnergyTypes that satisfy any colored requirement.
WILD_ENERGY = (EnergyType.RAINBOW,)

#: Number of distinct values, used to size one-hot feature blocks.
N_AREA = 13
N_ENERGY = 12
N_CARD_TYPE = 7
N_SELECT_TYPE = 11
N_SELECT_CONTEXT = 49
N_OPTION_TYPE = 17
N_LOG_TYPE = 24

#: Contexts in which the engine is asking us to *give something up*.
NEGATIVE_CONTEXTS = frozenset(
    {
        SelectContext.DISCARD,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_PRIZE,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_DECK_ENERGY,
    }
)

#: Contexts in which the engine is asking us to *gain something*.
POSITIVE_CONTEXTS = frozenset(
    {
        SelectContext.TO_HAND,
        SelectContext.TO_FIELD,
        SelectContext.TO_BENCH,
        SelectContext.TO_ACTIVE,
        SelectContext.HEAL,
        SelectContext.REMOVE_DAMAGE_COUNTER,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.ATTACH_TO,
        SelectContext.ATTACH_FROM,
        SelectContext.LOOK,
    }
)

#: Contexts that target the *opponent* and therefore invert card-value logic.
OPPONENT_TARGET_CONTEXTS = frozenset(
    {
        SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.DAMAGE,
    }
)
