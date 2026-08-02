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

__all__ = [
    "build_crustle_test_deck", "propose_swap", "evaluate_deck", "search",
    "map_elites_search", "replicator_dynamics", "double_oracle_lite_solve", "main",
]

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


def propose_swap(
    deck: list[int], db: CardDB, rng: random.Random, tries: int = 80, protected_cards: set[int] | None = None
) -> list[int] | None:
    """One random legal single-card swap. Returns ``None`` if no legal swap
    was found in ``tries`` attempts (the search just skips that candidate).

    ``protected_cards`` defaults to ``KEY_CARDS`` (lucario_fighting's own
    win condition) for backward compatibility with the existing single-deck
    search below; Week 7's MAP-Elites pass over multiple archetypes passes
    each deck's own key cards explicitly instead."""
    protected = KEY_CARDS if protected_cards is None else protected_cards
    counts: dict[int, int] = {}
    for cid in deck:
        counts[cid] = counts.get(cid, 0) + 1
    removable = [cid for cid in counts if cid not in protected]
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


# ---------------------------------------------------------------------------
# Week 7: MAP-Elites-lite archive + Double-Oracle-lite meta-game solve.
#
# The Week-3 docstring above said this real infrastructure wasn't built yet.
# It is now, scoped honestly: a real behavior-descriptor archive (not a
# single best-of-N), seeded from the 5 real, evidence-backed decks this
# session actually captured from the live ladder (not invented archetypes),
# mutated via the same legal-swap machinery every deck-search pass this
# session has used, and solved via replicator dynamics over the resulting
# payoff matrix (the standard tractable approximation for an empirical game
# too large for an exact LP solve) rather than a full Double-Oracle
# best-response loop.
# ---------------------------------------------------------------------------


def _fixed_map_elites_panel(db: CardDB) -> dict[str, list[int]]:
    """A small, fixed reference panel used for *every* archive member, so
    archive entries are directly comparable to each other (unlike
    ``_panel``, which excludes whichever deck is being evaluated)."""
    return {
        "crustle": build_crustle_test_deck(db),
        "lucario_fighting": load_deck("lucario_fighting"),
    }


def _descriptor(results: dict[str, float]) -> tuple[tuple[int, int], float, float]:
    """Coarse behavior descriptor: (worst-case decile, average decile)
    against the fixed panel -- the MAP-Elites archive key."""
    worst = min(results.values())
    avg = sum(results.values()) / len(results)
    return (int(worst * 10), int(avg * 10)), worst, avg


def map_elites_search(
    seed_decks: dict[str, tuple[list[int], set[int]]],
    db: CardDB,
    rounds: int = 8,
    mutations_per_round: int = 3,
    screen_games: int = 15,
    confirm_games: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """``seed_decks`` maps a name to ``(deck, protected_cards)``. Archive is
    keyed by the behavior descriptor from :func:`_descriptor`; each cell
    keeps only the highest-worst-case deck found for it (elitist), so the
    archive as a whole traces out the real worst-case/average frontier
    across everything explored, not just one winner."""
    panel = _fixed_map_elites_panel(db)
    rng = random.Random(seed)

    archive: dict[tuple[int, int], dict[str, Any]] = {}
    protections: dict[str, set[int]] = {}

    def insert(name: str, deck: list[int], results: dict[str, float]) -> bool:
        key, worst, avg = _descriptor(results)
        cur = archive.get(key)
        if cur is None or worst > cur["worst"]:
            archive[key] = {"name": name, "deck": deck, "results": results, "worst": worst, "avg": avg}
            return True
        return False

    t0 = time.time()
    for name, (deck, protected) in seed_decks.items():
        res = evaluate_deck(deck, db, panel, confirm_games, seed=seed)
        insert(f"seed:{name}", deck, res)
        protections[name] = protected
        print(f"  seeded {name}: worst={worst_case(res)} avg={sum(res.values())/len(res):.3f}")

    history: list[dict[str, Any]] = []
    seed_names = list(seed_decks.keys())
    for r in range(rounds):
        base_name = rng.choice(seed_names)
        base_deck, base_protected = seed_decks[base_name]
        # mutate off the *current archive member closest to this seed's
        # lineage* when one exists, else the seed itself -- lets the
        # archive actually accumulate improvements round over round.
        lineage = [v for v in archive.values() if v["name"].split(":")[0] == base_name or v["name"] == f"seed:{base_name}"]
        base_for_mutation = rng.choice(lineage)["deck"] if lineage else base_deck

        accepted_this_round = 0
        for _ in range(mutations_per_round):
            cand = propose_swap(base_for_mutation, db, rng, protected_cards=base_protected)
            if cand is None:
                continue
            res = evaluate_deck(cand, db, panel, screen_games, seed=seed + r * 1000)
            if insert(f"{base_name}:r{r}", cand, res):
                accepted_this_round += 1
        history.append({"round": r, "base": base_name, "archive_size": len(archive), "accepted": accepted_this_round, "seconds": round(time.time() - t0, 2)})
        print(f"  round {r} (mutating {base_name}): archive_size={len(archive)} accepted={accepted_this_round}")

    archive_list = sorted(archive.values(), key=lambda v: (-v["worst"], -v["avg"]))
    return {"panel": list(panel.keys()), "archive": archive_list, "history": history, "seconds": round(time.time() - t0, 2)}


def replicator_dynamics(payoff, iterations: int = 500, eta: float = 0.5):
    """Pure, dependency-free (numpy only) replicator-dynamics solve for the
    symmetric-game approximate Nash mixture over a square win-rate payoff
    matrix -- factored out from :func:`double_oracle_lite_solve` so the
    solver itself is testable against a known equilibrium without playing
    any real games."""
    import numpy as np

    payoff = np.asarray(payoff, dtype=float)
    n = payoff.shape[0]
    x = np.full(n, 1.0 / n)
    for _ in range(iterations):
        fitness = payoff @ x
        x = x * np.exp(eta * (fitness - fitness.mean()))
        x = x / x.sum()
    return x


def double_oracle_lite_solve(archive: list[dict[str, Any]], db: CardDB, games: int = 40, seed: int = 0, top_n: int = 8) -> dict[str, Any]:
    """Week 7's Double-Oracle-lite: build the full pairwise payoff matrix
    across the archive's top ``top_n`` decks (by worst-case fitness) and
    solve for an approximate symmetric Nash mixture via replicator
    dynamics -- the standard tractable approximation for an empirical game
    this size, needing no new dependency (pure numpy) and no exact LP
    solver, which would be real added scope not justified here."""
    import numpy as np

    top = archive[:top_n]
    names = [a["name"] for a in top]
    decks = [a["deck"] for a in top]
    n = len(top)

    payoff = np.full((n, n), 0.5)

    def shell_for(policy_cls, d):
        pol = policy_cls(d, db, HeuristicConfig()) if policy_cls is HeuristicPolicy else policy_cls(d)
        return SafetyShell(pol, db, FallbackPolicy(d), BudgetManager()).act

    for i in range(n):
        for j in range(i + 1, n):
            d = duel(
                names[i], lambda dd=decks[i]: shell_for(HeuristicPolicy, dd), decks[i],
                names[j], lambda dd=decks[j]: shell_for(HeuristicPolicy, dd), decks[j],
                games=games, seed0=seed + i * 1000 + j,
            )
            wr = d.summary()["a_winrate"]
            payoff[i, j] = wr
            payoff[j, i] = 1.0 - wr

    x = replicator_dynamics(payoff)

    order = np.argsort(-x)
    ranked = [{"name": names[i], "weight": round(float(x[i]), 4)} for i in order]
    best_idx = int(order[0])

    return {
        "names": names,
        "payoff_matrix": payoff.round(4).tolist(),
        "equilibrium_mixture": ranked,
        "recommended_deck_name": names[best_idx],
        "recommended_deck": decks[best_idx],
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
