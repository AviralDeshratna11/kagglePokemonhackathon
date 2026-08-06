# Project Status — read this first

This file is the handoff document. If you are a fresh Claude session (or
human) picking this project up, read this before anything else — it tells
you what's actually true *right now*, which `README.md` alone will not:
`README.md` is a detailed, accurate account of the Week-0 engine
architecture, but its own status line still says "Week 0 complete" and
"Next: Week 1." The project is actually nine weeks in. This file is kept
current; `README.md` is not.

Everything claimed here is either committed code, a real Kaggle submission
score, or a report file in `artifacts/`. Where a claim can't be checked
that way, it's flagged as a hypothesis, not fact.

## What this project is

An agent for Kaggle's **Pokémon TCG AI Battle Challenge** — two tracks:
Simulation (`pokemon-tcg-ai-battle`, ranked live ladder, this file's focus)
and Strategy (`pokemon-tcg-ai-battle-challenge-strategy`, a written report —
see `strategy_notebook/ptcg_strategy_report.ipynb`). $240k prize pool
across both. Target discussed with the user: 1100+ on the Simulation
ladder; current confirmed best is 539.6, against a field topping out at
1153.8 across 5,877 teams.

## Current live state (as of the last real submission)

* **Deck:** `kangaskhan_crustle` (`ptcg/decks/kangaskhan_crustle.csv`) —
  Mega Kangaskhan ex + Crustle, a real decklist from leaderboard rank-3
  player "LiamK," found via a MAP-Elites population search over real
  captured archetypes (Week 7).
* **Policy:** `HeuristicPolicy` (`policy_used: "heuristic-v1"`), the
  rule-based scorer — **not** the BC network. See "The open problem" below
  for why.
* `submission.tar.gz` in the repo root matches this exact state, rebuilt
  and self-verified via `ptcg/tools/build_submission.py`. Not yet
  resubmitted to Kaggle as of this writing — that's a manual, user-driven
  action, never done automatically by the assistant.

## Real score history (the ground truth — everything else is a prediction until checked against this)

| Build | Policy | Live score | Note |
|---|---|---|---|
| Week 3 patch | heuristic, `lucario_fighting` | 348.5 → 393.4 | Deck-file fix |
| Week 7 | heuristic, `kangaskhan_crustle` | **539.6** | MAP-Elites-found deck; confirmed real win over the prior lucario build in a 150-games/pair local bench first |
| Week 8 | heuristic + hand-tuned strategy rules, same deck | 521.8 | **Real regression** (−18) despite looking locally neutral-to-favorable. Reverted in Week 9. |
| Week 9a | BC network (`bc_kangaskhan_crustle.pt`), same deck | **420** | **Real regression** (−120 from the 539.6 baseline) despite a *decisive* local bench (9-way panel, every CI excluding 50%, including a first-ever win over the Crustle wall). Reverted immediately. |
| Week 9b (current) | heuristic, same deck (reverted) | *(pending next submission)* | Expected ≈539.6 |

**The pattern across all three regressions:** local validation, however
rigorous, has now twice failed to predict real transfer for a *hand-tuned
heuristic* change and once for a *trained network* change. The common
thread is not "heuristics are bad" or "networks are bad" — it's that this
project's local bench panels (a handful of baselines and its own prior
policies) are not diverse enough to stand in for the real ladder's
thousands of distinct opponents. Treat any future "decisive local win" as
a candidate to validate on the real ladder, not as settled.

## The open problem, as of this file's writing

`bc_kangaskhan_crustle.pt` — a `BeliefSetDMCLite` network behavior-cloned
on ~120K real ladder decisions (deck-matched to Mega Kangaskhan ex, pulled
from six real days 2026-07-28→2026-08-02) — scored 420 live despite
dominating a 9-entry local bench (itself, the heuristic, a confidence-gated
router, the `lucario_fighting` equivalents, `greedy`, `first`, and a
dedicated Crustle-counter wall deck), every duel's 95% CI excluding 50%.

**Ruled out:** a deadline/budget-exhaustion explanation. `BCPolicy.score()`
(`ptcg/agents/bc_policy.py`) does not check its allotted `Deadline` at all
— it always runs a full forward pass regardless of how little time was
budgeted, which is a real, confirmed gap (`SafetyShell` tracks
`deadline_overruns` but no tool in this repo ever surfaces that metric).
This was flagged as the leading hypothesis, then directly tested: 5 local
self-play games with the real checkpoint showed 0 overruns and ~27ms mean
decision time, nowhere near the 600s budget. So this is *not* why the real
score cratered, at least not on hardware resembling this dev machine.

**Leading (unconfirmed) hypothesis:** ordinary imitation-learning
distributional shift. The training corpus, while real, came from one
narrow slice of the ladder (players whose decks overlapped with Mega
Kangaskhan ex). The local bench panel is largely *in-distribution* for
that corpus (the network's own teacher-adjacent policies, plus decks it
was implicitly exposed to). The real ladder's actual diversity is
untested. This has not been directly confirmed — there is no access to
Kaggle's per-game logs — it is the most parsimonious explanation given
what's been ruled out, not a settled fact.

## What's next (recommended, not yet started)

In order of confidence, discussed with the user but not yet executed:

1. **Ship the confidence-gated router next, not raw BC.** Already built
   this session (`kangaskhan_crustle_router` — see
   `ptcg/agents/router_policy.py`, wired into `ptcg/tools/week3_bench.py`'s
   `deck_bench`). It fell back to the heuristic on low-confidence
   decisions; the local bench showed raw BC beating it 68% of the time,
   but that comparison used the same narrow, in-distribution panel — the
   router's fallback is specifically the safety property needed against
   *unfamiliar* opponents, untested here. No new training required to try
   this.
2. **Build a genuinely diverse validation panel from data already on
   disk.** Real decklists/replay corpora already exist for three *other*
   captured archetypes (`rocket_mewtwo`, and decklists for Grimmsnarl/
   Dragapult recorded in `artifacts/week7_seed_decks.json`, not yet fully
   onboarded). Benching a candidate against heuristics piloting *those*
   real decks is a better proxy for ladder diversity than benching against
   policies this project built itself.
3. **Cross-deck self-play refinement**, infra built and tested this
   session but never run: `ptcg/train/league.py`'s
   `League.cross_deck_opponents` + the corresponding fix in
   `ptcg/tools/generate_league_selfplay.py` (previously a real bug: the
   self-play harness passed the trainee's own deck to *both* seats
   regardless of which deck a sampled opponent actually piloted — fixed,
   tested in `test_week9.py`, never exercised at scale). Training against
   varied real-deck opponents, not just imitating one narrow real slice,
   is a different regularizer against the Week 9a failure than more BC
   data would be.
4. **Downgrade confidence in "train BC on one deck-matched real slice"**
   as an established recipe. It worked in every local bench this session
   (`bc_rocket_mewtwo` included) — but Week 9a is the *first* time any BC
   checkpoint got real ladder feedback, and it failed. Treat the recipe as
   locally-validated-only, not proven, until a BC checkpoint clears a real
   ladder submission.

## Standing rules this project operates under

Established the hard way, across three real regressions — worth
preserving even if the exact history above gets summarized away later:

* **Real ladder evidence always outranks local bench confidence**, no
  matter how decisive the local result looks or how many CIs exclude 50%.
  A local win is a candidate, not a result.
* **Never ship on a CI that merely fails to exclude 50%.** (Week 8's exact
  mistake — the A/B bench crossed 50% and got shipped anyway "because it
  looked favorable.")
* **`search_begin`/`search_step`** (the engine's native search API) stay
  permanently excluded from any submitted agent. The C ABI is
  undocumented; a wrong guessed signature segfaults the process with no
  catchable exception. Re-checked for public documentation multiple times
  this session (code + web search); still nothing. Not a one-time no —
  worth re-checking periodically, but never guessed at live.
* **Report failures and regressions as plainly as wins.** Every table in
  this file includes the real regressions, not just the wins.
* **Never touch `submission.tar.gz` without a fuzz pass (20k adversarial
  observations + 150 self-play games, 0 crashes/0 illegal) and a
  self-verified build** (`ptcg/tools/build_submission.py` plays an
  isolated self-game before writing the archive).
* **Actually submitting to Kaggle is always a manual, user-driven action.**
  The assistant builds and self-verifies `submission.tar.gz`; the user
  decides when/whether to upload it.

## Where to look for more detail

* `README.md` — deep, accurate engine/architecture documentation (Week 0
  scope; the design decisions there are all still current, only the
  status line is stale).
* `git log --oneline` — one commit per week, with a genuinely detailed
  body explaining what changed and why; this is the most reliable
  chronological record.
* `artifacts/week*_report.json`, `artifacts/week9_kangaskhan_onboard_report.json`
  — real bench/fuzz/pull numbers behind every claim above.
* `strategy_notebook/ptcg_strategy_report.ipynb` — the Strategy-track
  narrative, validated to execute end-to-end against real artifact files.
* `test_week0.py` … `test_week9.py` — one file per week, run all of them
  (`python -m pytest test_week0.py test_week1.py ... -v`) before trusting
  any further change; 168 tests, all green as of this writing.

## A note on Claude's own memory/plan files

This session also maintains a plan file
(`~/.claude/plans/transient-mapping-thompson.md`) with the full,
week-by-week design rationale, and a small memory system under
`~/.claude/projects/.../memory/`. Both are scoped to this local Claude
Code profile/installation — they are **not** guaranteed to be available
from a different account or machine. This file is the one piece of
continuity guaranteed to travel with the project, because it's committed
to the repo itself. If a new session has access to the plan file too,
it's a richer read; if not, this file plus the commit history above should
be enough to reconstruct everything that matters.
