# PTCG AI Battle Challenge — Week 0

> **For current project status, see [`PROJECT_STATUS.md`](PROJECT_STATUS.md).**
> This README documents the Week-0 engine architecture in detail and is
> still accurate on that front, but its status line below is stale — the
> project is nine weeks past this point.

Engine harness, safety layer, and a non-crashing rule-based MVP for the Pokémon
TCG AI Battle Challenge (`cabt` engine, Kaggle).

**Status:** Week 0 complete. Zero crashes, zero illegal actions, and zero
timeouts across ~5,000 games and 20,000 adversarial observations. Submission
bundle builds at 0.52 MiB against a 197.7 MiB limit and passes isolated
self-play verification.

---

## Week 0 objectives vs. outcome

| Objective | Outcome |
|---|---|
| Pin `kaggle-environments==1.30.1`, resolve arm64 friction | Pinned; `linux/amd64` Dockerfile provided. Also found `>=1.31` ships `libcg-arm64.so` / `libcg.dylib`, so Docker is now optional on Apple silicon |
| Rule-based heuristic on a simple, low-variance deck | `HeuristicPolicy` on **Mega Lucario ex**; 98.7% / 85.7% / 96.0% vs random / first / greedy on mirror decks |
| Try-catch + action masking, guarantee no engine crashes | 4-layer `SafetyShell`; fuzzed with 20k hostile observations, 0 failures |
| TrueSkill foundation above μ = 600 | Cannot be verified offline. Offline ladder and matchup matrices are in `artifacts/` |

The last row is stated honestly: μ is a property of the live ladder. Everything
here is offline evidence that the agent will not *forfeit* rating, plus a
measured margin over the public example agents that dominate the field.

---

## Findings that changed the plan

Most of these came from reading the shipped engine rather than the docs.

**1. `libcg.so` exports seven symbols the Python bindings never expose.**
`AllCard`, `AllAttack`, `AgentStart`, `SearchBegin`, `SearchStep`, `SearchEnd`,
`SearchRelease`. `AllCard()` / `AllAttack()` return the simulator's own card and
attack tables as JSON — including the integer **`attackId`** keys that
`Option.attackId` refers to and that `EN_Card_Data.csv` does not contain.

The whole feature pipeline was rebuilt on these instead of the CSV. Card IDs,
attack IDs and enums can no longer drift from the simulator, new card releases
are picked up automatically, and the submission bundle needs **no data files at
all**. The CSV is now optional, used only for cosmetic expansion metadata.

**2. The card pool is 1,267 cards, not ~2,000.** The CSV's 2,022 rows are
one-per-attack, not one-per-card. There are 1,556 distinct attacks.

**3. `actTimeout` is 0.** From `cabt.json`: there is no separate per-move
allowance, so every millisecond comes out of the single 600 s pool and
exhausting it is an immediate loss. The engine reports the live figure as
`obs["remainingOverageTime"]`, so `BudgetManager` syncs to the engine's number
rather than trusting local timing.

**4. A latent `SIGABRT` in any local harness.** `dlopen` is refcounted per path,
so `ctypes.cdll.LoadLibrary` on an already-loaded `libcg.so` returns the *same*
library — and a second `GameInitialize()` aborts the process. Both `cg.sim` (in
our bundle) and `kaggle_environments.envs.cabt.cg.sim` (the local harness) call
it at import. `engine.py` adopts an already-initialised handle instead.

**5. The 4-copy deck limit is per card *name*, not per ID.** The pool contains
three distinct printings of "Riolu" with different IDs and HP. A per-ID check
passes decks the engine rejects.

**6. Only 6 of 11 `SelectType`s and 14 of 49 `SelectContext`s ever occur.**
Measured over 60 games (`artifacts/engine_report.json`): `MAIN` is 56.9% of all
decisions, `TO_HAND` 10.7%, `ATTACH_TO`/`ATTACH_FROM` 13.7% combined. 35
documented contexts were never observed. Zero undocumented values appeared, so
the published enums match the implementation.

**7. Decision volume: 55.8 per player per game** (mean; worst observed ~113),
across 16.7 turns. Options per decision: mean 4.9, p95 13, max 37. These
numbers calibrate the clock allocator and size the Week-1 pointer head.

---

## The bug that mattered

The first working heuristic beat a random opponent only 68%. Tracing showed
Bellibolt sitting on one Energy for entire games while filler attacks got
switched on.

The cause was the attachment model. Ranking attachments by *how much they
reduce the energy shortfall* is the obvious rule and is badly wrong: it scores
"switch on a 30-damage filler attack" identically to "advance the deck's actual
win condition", so the agent sprays Energy across the bench and never powers its
attacker.

Replacing it with a damage-weighted power value, `damage / (1 + 2 × shortfall)`,
which is superlinear near completion, plus a designated-attacker concept, moved
win rate against random from 68.3% → 84.5%. Two related defects surfaced from
the same traces: attaching seven Energy to a Pokémon that needed four, and an
ability (`Flashing Draw`) that pays for itself by discarding Energy *off our own
attacker* — now caught by the `SELF_ENERGY_COST` role tag.

---

## Deck choice is empirical, not intuited

The plan named Mega Lucario ex *or* Iono's Bellibolt ex. Both were built and
measured. On **mirror decks**, which isolates agent skill from deck strength:

| Deck | vs random | vs first | vs greedy | mean turns |
|---|---|---|---|---|
| **Mega Lucario ex** | **0.987** | **0.857** | **0.960** | 5.3–10.2 |
| Iono's Bellibolt ex | 0.803 | 0.710 | 0.663 | 16.2–20.6 |

*300 games per cell, seats alternated.*

Head to head, heuristic-on-Lucario beats heuristic-on-Bellibolt 96.3%. Lucario
is both the stronger deck and the easier one to pilot: two attacks, one Energy
type, self-accelerating, and no combo to misplay. Bellibolt is retained as the
slow/complex pole of the Week-4 H4 experiment.

One caveat worth stating: Bellibolt is Lightning and weak to Fighting, so part
of that 96.3% is a type matchup rather than general strength. The mirror-deck
control above is the number to trust.

---

## Measured noise floor

The ablation sweep includes a control: `retreat_to_swap_attacker=False` is
identical to the baseline config, yet scored 0.951 against the baseline's 0.929
over 450 games each. **The noise floor is ±0.02 at 450 games**, which makes most
single-field deltas in `artifacts/ablations.json` non-significant.

This is why Week 4's protocol needs ~2,400 games per matchup for ±2 points
(`arena.games_for_precision`), and why the engine's internal shuffle RNG — which
we cannot seed — is documented in `eval/harness.py` rather than papered over.

---

## Architecture

```
main.py                    Kaggle entrypoint; lazy build, cannot raise
deck.csv                   60 card IDs
ptcg/
  core/
    engine.py              Library loader; path bootstrap, arch detect,
                           single-init guard, extended ctypes bindings
    enums.py               Full engine enums, from the official API reference
    carddb.py              Card/attack DB from AllCard/AllAttack + role tagging
    obs.py                 Typed observation views; weakness/resistance,
                           energy-payment maths
    actions.py             ActionCandidate + fixed-width feature vector
    clock.py               600 s cumulative budget allocator
    safety.py              Masking, loop guard, layered fallbacks
    trace.py               JSONL recorder -> Week-1 BC corpus
  agents/
    base.py                Policy protocol + multi-select count policy
    heuristic.py           The MVP brain
    fallback.py            Cannot-fail floor
    baselines.py           random / first / greedy sparring partners
  decks/                   deck CSVs + registry + legality validator
  eval/
    harness.py             Direct match loop (~10x faster than env.run)
    arena.py               Duels, matchup matrices, Wilson CIs, ladder
    fuzz.py                Adversarial fuzzing
  tools/                   engine_report, bench, ablate, build_submission
tests/                     52 tests
artifacts/                 Generated evidence
```

### The interface that outlives Week 0

The engine's action space is variable-length and context-dependent: the same
index means "discard the 3rd card in my hand" in one decision and "put a damage
counter on the opponent's 3rd benched Pokémon" in the next. A fixed action head
with masking does not type-check here.

So raw options are lifted once into `ActionCandidate`: a typed record plus a
fixed-width feature vector describing the option semantically. Every brain
implements the same contract:

```python
scores = policy.score(view, candidates, deadline)   # one float per candidate
```

The Week-0 heuristic computes those scores with hand-written rules. The Week-1
behaviour-cloned network will compute them with pointer-style cross-attention
over the *same* vectors. Swapping the brain changes one line in `main.py` and
touches no masking, selection, loop-guarding or budgeting code.

`trace.py` records exactly those vectors alongside the chosen index, so the BC
dataloader is ~20 lines and a featurisation bug shows up as a train/inference
mismatch now rather than as an unexplained Elo gap in Week 2.

### Safety, in four layers

1. **Structural validation** — malformed observations short-circuit to the
   trivial legal answer.
2. **Action masking** — selections are only ever produced by indexing into
   `select["option"]`, then clipped, de-duplicated and padded to satisfy
   `minCount`/`maxCount`. `sanitize_selection` is a total function.
3. **Loop guard** — repeatable abilities ("as often as you like during your
   turn") are re-offered immediately after use; a scoring bug there drains the
   clock silently. Repeated identical decisions within a turn get masked, with
   `END` as a guaranteed escape.
4. **Layered fallbacks** — policy exception, deadline overrun or budget panic
   degrade to a rule-based fallback, which degrades to `list(range(minCount))`.

Every rescue is counted in `SafetyShell.report`, not swallowed.

---

## Results

Round robin, 300 games per pair, seats alternated
(`artifacts/bench.json`):

| Matchup | Win rate | 95% CI |
|---|---|---|
| heuristic@Lucario vs random | 1.000 | [0.987, 1.000] |
| heuristic@Lucario vs first | 0.983 | [0.962, 0.993] |
| heuristic@Lucario vs greedy | 0.980 | [0.957, 0.991] |
| heuristic@Bellibolt vs random | 0.807 | [0.758, 0.847] |
| heuristic@Bellibolt vs greedy | 0.667 | [0.612, 0.718] |

Reliability, ~5,000 games plus fuzzing:

| Metric | Value |
|---|---|
| Engine crashes | 0 |
| Illegal actions | 0 |
| Timeouts / forfeits | 0 |
| Policy exceptions | 0 |
| Sanitised selections | 0 |
| Hostile observations survived | 20,000 / 20,000 |
| Max decision latency | 3.3 ms |
| First decision (lazy DB build) | ~148 ms, once per process |
| Clock used per game | ~0.03 s of 600 s |

Latency headroom is roughly four orders of magnitude, which is the budget Week 3
spends on bounded ISMCTS.

---

## Usage

```bash
make install          # pinned deps
make doctor           # platform, engine hash, unbound exports
make test             # 52 tests
make fuzz             # 20k hostile observations + random-deck self-play
make bench            # round robin with Wilson CIs
make ablate           # HeuristicConfig sweep
make build            # build + verify submission.tar.gz
make ci               # test + fuzz + build, the pre-submission gate
```

On Apple silicon with `kaggle-environments==1.30.1` pinned:

```bash
make docker-build && make docker-ci
```

Submit `dist/submission.tar.gz` to the Simulation track. `make build` refuses to
emit a bundle that fails isolated self-play, so a failed build costs nothing
while a failed upload costs one of five daily submissions.

---

## Known limitations

* **Offline ≠ ladder.** Community reports put identical submissions over 100
  rating points apart. Nothing here predicts ladder position; real A/B on the
  live ladder is the only way to settle it.
* **`prefer_first=True` is weakly supported.** −0.029 when disabled, against a
  ±0.02 noise floor. Borderline; needs the larger sweep.
* **The panel is thin.** Three baselines and two decks. `first` and `greedy`
  are not the real ladder. Week 2's scripted league fixes this.
* **Damage estimation ignores attack text.** `damage_after_type` applies only
  weakness and resistance; bonus-damage and reduction effects are not modelled.
  Deliberate — the heuristic needs a consistent ordering, and Week 1's value
  head learns the residual.
* **Search bindings are not wired.** `SearchBegin`/`SearchStep` are exported but
  the C ABI is undocumented, and guessing it segfaults. `search_begin_input` is
  captured in every trace so Week 3 has the data. The official Python signature
  is recorded in `engine.py`.
* **Ablations are near ceiling** (~0.93) on Lucario, compressing effect sizes.
  Re-run against a stronger panel.

## Next: Week 1

1. Attach frozen MiniLM embeddings of `skill_texts` / attack text to the card
   block in `actions.py` — the layout already reserves the slot.
2. Daily Kaggle episode puller; parse replays into the `trace.py` schema.
3. Behaviour-clone the set-transformer + pointer head; drop into `main.py`.
4. Run H1 (card-text embeddings vs one-hot IDs) with held-out teched cards.
