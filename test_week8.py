"""Week-8 test suite: expert human strategy grounded in the existing
heuristic -- search-before-draw ordering, recovery-aware discarding, the
Role.HAND_SHUFFLE pre-shuffle-clearing bonus, prized-resource awareness, and
late-game bench thinning.

Follows test_week6.py's style: real cards and real registered decks, plus
small stub views that exercise only the attributes each scorer method
actually reads.

Week 9 update: the real Kaggle ladder scored the whole Week-8 bundle ~18
points *lower* than the build without it (539.6 -> 521.8), so
``HeuristicConfig.enable_expert_strategy_tuning`` now defaults to False and
``search_bonus`` was reverted to its pre-Week-8 value. The behaviors below
are still real and still tested -- just opt-in via that flag rather than
live by default. Tests that exercise the Week-8 behaviors now construct
``HeuristicConfig(enable_expert_strategy_tuning=True)`` explicitly; new
tests confirm the *default* reproduces the flat/zero pre-Week-8 behavior.
"""

from __future__ import annotations

import pytest

from ptcg.agents.fallback import FallbackPolicy
from ptcg.agents.heuristic import HeuristicConfig, HeuristicPolicy
from ptcg.core.carddb import Role, get_card_db
from ptcg.core.clock import BudgetManager
from ptcg.core.safety import SafetyShell
from ptcg.decks.registry import load_deck
from ptcg.eval.harness import play_match

LACEY = 1199                      # Supporter: "Shuffle your hand into your deck..."
LILLIES_DETERMINATION = 1227      # Supporter, same effect; used in kangaskhan_crustle
NIGHT_STRETCHER = 1097            # Item, Role.RECOVERY
HILDA = 1225                      # Supporter, draw-only, no hand-shuffle text
SWITCH = 1123                     # Item, plain, no roles of interest
UNFAIR_STAMP = 1080               # ace_spec Item


@pytest.fixture(scope="module")
def db():
    return get_card_db()


class _FakeCandidate:
    def __init__(self, card):
        self.card = card


class _FakeMe:
    def __init__(self, hand=None, discard=None, prize=None, in_play=None):
        self.hand_cards = hand or []
        self.discard = discard or []
        self.prize = prize or []
        self._in_play = in_play or []

    def in_play(self):
        return self._in_play


class _FakePokemon:
    def __init__(self, card):
        self.card = card


class _FakeView:
    def __init__(self, me):
        self.me = me


class TestSearchBeforeDraw:
    def test_default_matches_real_evidence_backed_value(self):
        """Week 8 raised search_bonus above draw_bonus on theory; Week 9
        reverted it after the real ladder scored that bundle lower. The
        default must match the 539.6-scoring value, not the theory."""
        cfg = HeuristicConfig()
        assert cfg.search_bonus == 240.0
        assert cfg.search_bonus < cfg.draw_bonus

    def test_knob_remains_settable_for_a_future_isolated_ab(self):
        """The mechanism itself wasn't proven wrong -- only the untested
        bundle was. Explicitly requesting the old value must still work,
        for a future, properly isolated real-ladder test of this one knob."""
        cfg = HeuristicConfig(search_bonus=340.0)
        assert cfg.search_bonus > cfg.draw_bonus


class TestHandShuffleRole:
    def test_lacey_is_hand_shuffle(self, db):
        c = db.card(LACEY)
        assert Role.HAND_SHUFFLE in c.roles
        assert c.is_supporter

    def test_lillies_determination_is_hand_shuffle(self, db):
        c = db.card(LILLIES_DETERMINATION)
        assert Role.HAND_SHUFFLE in c.roles

    def test_unrelated_supporter_does_not_match(self, db):
        c = db.card(HILDA)
        assert Role.HAND_SHUFFLE not in c.roles


class TestPreShuffleClearingBonus:
    def test_bonus_when_hand_shuffle_supporter_in_hand(self, db):
        deck = load_deck("kangaskhan_crustle")
        cfg = HeuristicConfig(enable_expert_strategy_tuning=True)
        policy = HeuristicPolicy(deck, db, cfg)
        view = _FakeView(_FakeMe(hand=[db.card(LILLIES_DETERMINATION), db.card(SWITCH)]))
        assert policy._pre_shuffle_clearing_bonus(view) == pytest.approx(cfg.pre_shuffle_clear_bonus)

    def test_no_bonus_without_hand_shuffle_supporter(self, db):
        deck = load_deck("kangaskhan_crustle")
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))
        view = _FakeView(_FakeMe(hand=[db.card(SWITCH), db.card(HILDA)]))
        assert policy._pre_shuffle_clearing_bonus(view) == 0.0

    def test_disabled_by_default_even_with_supporter_in_hand(self, db):
        deck = load_deck("kangaskhan_crustle")
        policy = HeuristicPolicy(deck, db, HeuristicConfig())
        view = _FakeView(_FakeMe(hand=[db.card(LILLIES_DETERMINATION), db.card(SWITCH)]))
        assert policy._pre_shuffle_clearing_bonus(view) == 0.0


class TestRecoveryAwareDiscard:
    def _policy(self, db, deck_ids):
        return HeuristicPolicy(deck_ids, db, HeuristicConfig(enable_expert_strategy_tuning=True))

    def test_redundant_copy_scores_mild_penalty(self, db):
        # 4 copies of SWITCH in the deck; 2 currently visible in hand.
        deck = [SWITCH] * 4 + [1]  # pad with a basic energy so init helpers don't choke
        policy = self._policy(db, deck)
        hand = [db.card(SWITCH), db.card(SWITCH)]
        view = _FakeView(_FakeMe(hand=hand))
        cand = _FakeCandidate(db.card(SWITCH))
        assert policy._score_discard(view, cand) == pytest.approx(-25.0)

    def test_last_copy_no_recovery_scales_with_value(self, db):
        deck = [UNFAIR_STAMP] + [1]  # single ace_spec copy (real rule: max 1)
        policy = self._policy(db, deck)
        hand = [db.card(UNFAIR_STAMP)]
        view = _FakeView(_FakeMe(hand=hand))
        cand = _FakeCandidate(db.card(UNFAIR_STAMP))
        expected = -50.0 - policy._card_value(view, db.card(UNFAIR_STAMP))
        assert policy._score_discard(view, cand) == pytest.approx(expected)
        assert policy._score_discard(view, cand) < -50.0  # strictly worse than the old flat penalty

    def test_last_copy_with_recovery_available_is_cheaper(self, db):
        deck = [UNFAIR_STAMP] + [1]
        policy = self._policy(db, deck)
        hand = [db.card(UNFAIR_STAMP), db.card(NIGHT_STRETCHER)]
        view = _FakeView(_FakeMe(hand=hand))
        cand = _FakeCandidate(db.card(UNFAIR_STAMP))
        assert policy._score_discard(view, cand) == pytest.approx(-40.0)

    def test_none_card_falls_back_to_flat_penalty(self, db):
        policy = self._policy(db, [SWITCH] * 4 + [1])
        view = _FakeView(_FakeMe())
        cand = _FakeCandidate(None)
        assert policy._score_discard(view, cand) == -50.0

    def test_disabled_by_default_uses_flat_penalty_even_for_last_copy(self, db):
        """The real-evidence-backed default (flag off) must reproduce the
        exact 539.6-scoring behavior: every discard is -50.0 flat,
        regardless of recoverability."""
        policy = HeuristicPolicy([UNFAIR_STAMP] + [1], db, HeuristicConfig())
        hand = [db.card(UNFAIR_STAMP)]
        view = _FakeView(_FakeMe(hand=hand))
        cand = _FakeCandidate(db.card(UNFAIR_STAMP))
        assert policy._score_discard(view, cand) == -50.0


class TestPrizedResourceAwareness:
    GUST_SUPPORTER = 1182  # Boss's Orders, Role.GUST

    def test_no_discount_when_prizes_unrevealed(self, db):
        deck = [self.GUST_SUPPORTER] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))
        view = _FakeView(_FakeMe(prize=[None, None]))
        assert policy._all_other_copies_prized(view, db.card(self.GUST_SUPPORTER).name) is False

    def test_disabled_by_default_never_discounts(self, db):
        deck = [self.GUST_SUPPORTER] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig())
        name = db.card(self.GUST_SUPPORTER).name
        view_all_prized = _FakeView(
            _FakeMe(
                prize=[
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                ]
            )
        )
        assert policy._all_other_copies_prized(view_all_prized, name) is False

    def test_discount_only_when_every_other_copy_confirmed_prized(self, db):
        deck = [self.GUST_SUPPORTER] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))
        name = db.card(self.GUST_SUPPORTER).name

        # Only 2 of the other 3 copies revealed in prize -- must NOT trigger.
        view_partial = _FakeView(
            _FakeMe(prize=[{"id": self.GUST_SUPPORTER}, {"id": self.GUST_SUPPORTER}, None])
        )
        assert policy._all_other_copies_prized(view_partial, name) is False

        # All 3 other copies confirmed prized -- must trigger.
        view_full = _FakeView(
            _FakeMe(
                prize=[
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                ]
            )
        )
        assert policy._all_other_copies_prized(view_full, name) is True

    def test_card_value_discounted_when_all_backups_prized(self, db):
        deck = [self.GUST_SUPPORTER] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))
        card = db.card(self.GUST_SUPPORTER)

        view_unrevealed = _FakeView(_FakeMe())
        view_all_prized = _FakeView(
            _FakeMe(
                prize=[
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                    {"id": self.GUST_SUPPORTER},
                ]
            )
        )
        full_value = policy._card_value(view_unrevealed, card)
        discounted_value = policy._card_value(view_all_prized, card)
        assert discounted_value < full_value


class TestBenchThinning:
    # Dwebble: real basic, no ability. Used standalone (without Crustle, its
    # evolution) so it is genuinely NOT an evolution base in this synthetic
    # deck -- unlike in kangaskhan_crustle, where Crustle is also present.
    BASIC_NO_ABILITY = 344

    def test_no_bonus_before_min_turn(self, db):
        deck = [self.BASIC_NO_ABILITY] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))

        class _V:
            turn = 1

        assert policy._bench_thinning_bonus(_V(), db.card(self.BASIC_NO_ABILITY)) == 0.0

    def test_bonus_late_game_for_plain_basic(self, db):
        deck = [self.BASIC_NO_ABILITY] * 4 + [1]
        cfg = HeuristicConfig(enable_expert_strategy_tuning=True)
        policy = HeuristicPolicy(deck, db, cfg)

        class _V:
            turn = 20

        card = db.card(self.BASIC_NO_ABILITY)
        assert card.name not in policy._evolution_bases
        assert policy._bench_thinning_bonus(_V(), card) == pytest.approx(cfg.bench_thinning_bonus)

    def test_disabled_by_default_even_late_game(self, db):
        deck = [self.BASIC_NO_ABILITY] * 4 + [1]
        policy = HeuristicPolicy(deck, db, HeuristicConfig())

        class _V:
            turn = 20

        assert policy._bench_thinning_bonus(_V(), db.card(self.BASIC_NO_ABILITY)) == 0.0

    def test_no_bonus_for_evolution_base(self, db):
        """A Basic that is the base of our own evolution line is never
        "otherwise-unremarkable" -- thinning it away is a real cost."""
        deck = load_deck("lucario_fighting")
        policy = HeuristicPolicy(deck, db, HeuristicConfig(enable_expert_strategy_tuning=True))
        assert policy._evolution_bases  # lucario_fighting does evolve

        class _V:
            turn = 20

        for name in policy._evolution_bases:
            for cid in deck:
                c = db.card(cid)
                if c.name == name:
                    assert policy._bench_thinning_bonus(_V(), c) == 0.0
                    break


class TestSelfPlayZeroExceptions:
    """Same rigor as every prior week: the updated heuristic must not
    introduce a single policy/fallback exception or sanitised action on
    either registered deck."""

    @pytest.mark.parametrize("deck_name", ["kangaskhan_crustle", "lucario_fighting"])
    def test_no_exceptions(self, db, deck_name):
        deck = load_deck(deck_name)
        for seed in range(5):
            shell = SafetyShell(
                HeuristicPolicy(deck, db, HeuristicConfig()), db, FallbackPolicy(deck), BudgetManager()
            )
            result = play_match(shell.act, shell.act, deck, deck, seed=seed)
            assert result.reason == "engine_result"
            assert shell.report.policy_exceptions == 0
            assert shell.report.fallback_exceptions == 0
            assert shell.report.sanitised == 0
