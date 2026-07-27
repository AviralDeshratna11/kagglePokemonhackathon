"""Bounded local deck-robustness search: a real, scoped version of H4/H7.

The plan describes MAP-Elites (a population/behavior-descriptor archive) plus
Double-Oracle (Nash-equilibrium meta-game solving) for deck co-optimization.
That's real additional infrastructure this pass doesn't build (see the
Week-3 plan's "explicitly not attempted"). What this does instead: greedy
local search over legal single-card swaps from the current
``lucario_fighting`` deck, screened against a panel that specifically
includes a **Crustle wall deck** built from the real card pool -- Crustle
(id 345) has "Prevent all damage done to this Pokémon by attacks from your
opponent's Pokémon {ex}", which is exactly the mechanic that plausibly
explains the live ladder score drop this session found (`lucario_fighting`'s
entire game plan is an ex/Mega-ex attacker with no non-ex answer).

Two-stage evaluation per candidate swap: a cheap screen (few games) against
the whole panel to rank several proposals, then a full confirm (many games)
of only the winner, so the search doesn't spend its full game budget on
proposals that were never going to be accepted.

Fitness is the **worst-case** win rate across the panel, not the average --
directly the H4/H7 framing (flatten the matchup spread, don't just win on
average).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from ..agents.baselines import FirstPolicy, GreedyAttackPolicy
from ..agents.fallback import FallbackPolicy
from ..agents.heuristic import HeuristicConfig, HeuristicPolicy
from ..core.carddb import CardDB, get_card_db
from ..core.clock import BudgetManager
from ..core.safety import SafetyShell
from ..decks.registry import available_decks, load_deck, write_deck
from ..eval.arena import duel

__all__ = ["build_crustle_test_deck", "propose_swap", "evaluate_deck", "search", "main"]

DWEBBLE_ID = 344
CRUSTLE_ID = 345
GRASS_ENERGY_ID = 1

# Never swapped away: the deck's whole win condition. Everything else
# (support trainers, the fixed Fighting energy count) is fair game.
KEY_CARDS = {333, 678}  # Riolu, Mega Lucario ex


def build_crustle_test_deck(db: CardDB) -> list[int]:
    """A legal 60-card deck whose entire point is the ex-counter wall, built
    from cards confirmed present in the real pool -- not a stub, a real
    playable (if narrow) deck the engine accepts."""
    trainers_4x = [1121, 1123, 1182, 1224]  # Ultra Ball, Switch, Boss's Orders, Cheren
    trainers_2x = [1122, 1097, 1118]  # Pokegear 3.0, Night Stretcher, Energy Retrieval
    trainers_1x = [1125]  # Master Ball -- an ACE SPEC, at most 1 copy total

    deck = [DWEBBLE_ID] * 4 + [CRUSTLE_ID] * 4
    for cid in trainers_4x:
        deck += [cid] * 4
    for cid in trainers_2x:
        deck += [cid] * 2
    for cid in trainers_1x:
        deck += [cid] * 1
    deck += [GRASS_ENERGY_ID] * (60 - len(deck))

    problems = db.validate_deck(deck)
    if problems:
        raise RuntimeError(f"Crustle test deck is illegal: {problems}")
    return deck


def propose_swap(deck: list[int], db: CardDB, rng: random.Random, tries: int = 80) -> list[int] | None:
    """One random legal single-card swap. Returns ``None`` if no legal swap
    was found in ``tries`` attempts (the search just skips that candidate)."""
    counts: dict[int, int] = {}
    for cid in deck:
        counts[cid] = counts.get(cid, 0) + 1
    removable = [cid for cid in counts if cid not in KEY_CARDS]
    pool = [c.card_id for c in db.all_cards() if not c.ace_spec]

    for _ in range(tries):
        remove_cid = rng.choice(removable)
        add_cid = rng.choice(pool)
        if add_cid == remove_cid:
            continue
        new_deck = list(deck)
        new_deck[new_deck.index(remove_cid)] = add_cid
        if not db.validate_deck(new_deck):
            return new_deck
    return None


def _panel(db: CardDB) -> dict[str, list[int]]:
    others = [d for d in available_decks() if d != "lucario_fighting"]
    panel = {"crustle": build_crustle_test_deck(db)}
    for name in others:
        panel[name] = load_deck(name)
    return panel


def evaluate_deck(deck: list[int], db: CardDB, panel: dict[str, list[int]], games: int, seed: int) -> dict[str, float]:
    """Win rate of ``HeuristicPolicy`` piloting ``deck`` against every panel
    entry (also ``HeuristicPolicy``, piloting its own deck) plus two weaker
    baseline bots mirroring the candidate deck against itself -- the same
    panel shape ``ptcg/tools/bench.py`` uses."""
    results: dict[str, float] = {}

    def shell_for(policy_cls_or_instance, d, **kwargs):
        pol = policy_cls_or_instance(d, db, HeuristicConfig()) if policy_cls_or_instance is HeuristicPolicy else policy_cls_or_instance(d)
        return SafetyShell(pol, db, FallbackPolicy(d), BudgetManager()).act

    for i, (name, opp_deck) in enumerate(panel.items()):
        d = duel(
            "candidate", lambda dd=deck: shell_for(HeuristicPolicy, dd), deck,
            name, lambda od=opp_deck: shell_for(HeuristicPolicy, od), opp_deck,
            games=games, seed0=seed + i * 10_000,
        )
        results[name] = d.summary()["a_winrate"]

    for i, (name, cls) in enumerate((("greedy", GreedyAttackPolicy), ("first", FirstPolicy))):
        d = duel(
            "candidate", lambda dd=deck: shell_for(HeuristicPolicy, dd), deck,
            name, lambda dd=deck, c=cls: shell_for(c, dd), deck,
            games=games, seed0=seed + (i + 10) * 10_000,
        )
        results[name] = d.summary()["a_winrate"]

    return results


def worst_case(results: dict[str, float]) -> tuple[str, float]:
    name = min(results, key=results.get)
    return name, results[name]


def search(
    base_deck_name: str = "lucario_fighting",
    rounds: int = 10,
    candidates_per_round: int = 4,
    screen_games: int = 25,
    confirm_games: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    db = get_card_db()
    panel = _panel(db)

    current = load_deck(base_deck_name)
    baseline_results = evaluate_deck(current, db, panel, confirm_games, seed=seed)
    baseline_worst = worst_case(baseline_results)
    print(f"baseline ({base_deck_name}) results: {baseline_results}")
    print(f"baseline worst case: {baseline_worst}")

    best_deck = current
    best_results = baseline_results
    best_worst = baseline_worst
    history: list[dict[str, Any]] = []
    rng = random.Random(seed)
    t0 = time.time()

    for r in range(rounds):
        proposals = []
        for _ in range(candidates_per_round):
            cand = propose_swap(best_deck, db, rng)
            if cand is not None:
                proposals.append(cand)
        if not proposals:
            continue

        screened = []
        for cand in proposals:
            res = evaluate_deck(cand, db, panel, screen_games, seed=seed + r * 1000)
            screened.append((worst_case(res)[1], cand, res))
        screened.sort(key=lambda t: -t[0])
        top_score, top_deck, _ = screened[0]

        confirm_res = evaluate_deck(top_deck, db, panel, confirm_games, seed=seed + r * 1000 + 500)
        confirm_worst = worst_case(confirm_res)

        accepted = confirm_worst[1] > best_worst[1]
        if accepted:
            best_deck, best_results, best_worst = top_deck, confirm_res, confirm_worst

        history.append(
            {
                "round": r,
                "n_proposals": len(proposals),
                "screen_top_worst": top_score,
                "confirm_results": confirm_res,
                "confirm_worst": confirm_worst,
                "accepted": accepted,
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"  round {r}: confirm_worst={confirm_worst} accepted={accepted} (best so far: {best_worst})")

    return {
        "base_deck": base_deck_name,
        "baseline_results": baseline_results,
        "baseline_worst": baseline_worst,
        "best_results": best_results,
        "best_worst": best_worst,
        "best_deck": best_deck,
        "improved": best_deck != current,
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", default="lucario_fighting")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--candidates-per-round", type=int, default=4)
    ap.add_argument("--screen-games", type=int, default=25)
    ap.add_argument("--confirm-games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/week3_deck_search.json")
    ap.add_argument("--save-best-deck", default=None, help="also write the best deck to this deck.csv path")
    args = ap.parse_args(argv)

    info = search(
        base_deck_name=args.deck, rounds=args.rounds, candidates_per_round=args.candidates_per_round,
        screen_games=args.screen_games, confirm_games=args.confirm_games, seed=args.seed,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.save_best_deck and info["improved"]:
        write_deck(args.save_best_deck, info["best_deck"])
        print(f"wrote best deck to {args.save_best_deck}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
