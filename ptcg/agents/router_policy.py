"""Confidence-gated policy routing -- the safe substitute for the plan's
entropy-gated search.

Both planning PDFs specify: invoke bounded ISMCTS only "when the confidence
margin between the top-1 and top-2 actions falls below a critical threshold,
or when the value head indicates extreme ambiguity" -- i.e. gate an expensive,
risky operation on the network's own uncertainty rather than running it every
decision. This session confirmed `search_begin`/`search_step` cannot safely
be that operation here (undocumented C ABI, segfault-on-guess -- see the
Week-3 plan). But the *trigger* is sound and cheap to compute regardless of
what it gates.

So this gates policy *selection* instead of search: on a confident decision,
trust the network (it should know better than the heuristic by now). On an
uncertain one -- low top-1/top-2 margin, or a value estimate near 0.5 (the
network has no idea who's winning) -- fall back to the heuristic, which is
deterministic, cheap, and already proven never to do anything catastrophic.
Mechanically identical trigger, mechanically safe target.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..core.actions import ActionCandidate
from ..core.clock import Deadline
from ..core.obs import ObsView
from .base import default_desired_count
from .bc_policy import BCPolicy

__all__ = ["RouterPolicy"]


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


class RouterPolicy:
    name = "router-v1"

    def __init__(
        self,
        deck_ids: Sequence[int],
        network_policy: BCPolicy,
        heuristic_policy,
        margin_threshold: float = 0.15,
        value_margin_threshold: float = 0.08,
    ) -> None:
        self._deck = list(deck_ids)
        self.network_policy = network_policy
        self.heuristic_policy = heuristic_policy
        self.margin_threshold = margin_threshold
        self.value_margin_threshold = value_margin_threshold
        # Diagnostics only -- never read by scoring logic, safe to inspect
        # after a game to see how often each path actually fired.
        self.route_counts = {"network": 0, "heuristic": 0}
        self.last_route = ""
        self.last_margin: float | None = None
        self.last_value: float | None = None

    def deck(self) -> list[int]:
        return list(self._deck)

    def score(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        deadline: Deadline,
    ) -> list[float]:
        if not candidates:
            return []
        if len(candidates) == 1:
            self._route("network")
            return self.network_policy.score(view, candidates, deadline)

        net_scores, value = self.network_policy.score_and_value(view, candidates, deadline)
        probs = _softmax(net_scores)
        top2 = sorted(probs, reverse=True)[:2]
        margin = top2[0] - top2[1] if len(top2) == 2 else 1.0
        value_conf = abs((value if value is not None else 0.5) - 0.5) * 2.0

        self.last_margin = margin
        self.last_value = value

        if margin < self.margin_threshold or value_conf < self.value_margin_threshold:
            self._route("heuristic")
            return self.heuristic_policy.score(view, candidates, deadline)

        self._route("network")
        return net_scores

    def desired_count(
        self,
        view: ObsView,
        candidates: Sequence[ActionCandidate],
        scores: Sequence[float],
    ) -> int:
        return default_desired_count(view, candidates, scores)

    def _route(self, which: str) -> None:
        self.last_route = which
        self.route_counts[which] = self.route_counts.get(which, 0) + 1
