"""Decision recorder.

Week 1 needs a behaviour-cloning corpus. The blueprint proposes pulling the
daily Kaggle episode replays, and that remains the primary source -- but those
replays are raw ``obs``/``action`` pairs that still have to be pushed back
through featurisation before they can train anything.

Recording locally costs almost nothing and produces the corpus in its *final*
form: the same ``ActionCandidate`` feature vectors the network will consume at
inference, alongside the index that was chosen. That means:

* the BC dataloader is ~20 lines instead of a replay parser;
* a featurisation bug shows up as a training/inference mismatch here in Week 0
  rather than as an unexplained Elo gap in Week 2;
* self-play data and replay data land in one identical schema, so they can be
  mixed in one buffer.

Writing is buffered and wrapped so that telemetry can never affect play; if the
disk is full the game continues and the trace is simply short.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

__all__ = ["TraceWriter", "NullRecorder"]

SCHEMA_VERSION = 1


class NullRecorder:
    """Recorder that does nothing, for the latency-critical submission path."""

    def __call__(self, record: dict[str, Any]) -> None:  # pragma: no cover
        return

    def close(self) -> None:
        return


class TraceWriter:
    """Append-only, optionally gzipped JSONL writer."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        tag: str = "selfplay",
        compress: bool = True,
        flush_every: int = 64,
        include_features: bool = True,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = f"{tag}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        suffix = ".jsonl.gz" if compress else ".jsonl"
        self.path = self.dir / f"{self.episode_id}{suffix}"
        self._fh = (
            gzip.open(self.path, "wt", encoding="utf-8")
            if compress
            else open(self.path, "w", encoding="utf-8")
        )
        self._n = 0
        self._flush_every = flush_every
        self.include_features = include_features
        self._write({"type": "header", "schema": SCHEMA_VERSION, "episode": self.episode_id})

    # -- recording ----------------------------------------------------------

    def __call__(self, record: dict[str, Any]) -> None:
        if not self.include_features:
            record = {k: v for k, v in record.items() if k != "features"}
        record["type"] = "decision"
        record["step"] = self._n
        self._write(record)

    def finish(self, outcome: dict[str, Any]) -> None:
        """Attach the terminal reward so value targets are available."""
        self._write({"type": "outcome", **outcome})
        self.close()

    # -- io -----------------------------------------------------------------

    def _write(self, obj: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
            self._n += 1
            if self._n % self._flush_every == 0:
                self._fh.flush()
        except BaseException:  # noqa: BLE001 - telemetry must never break play
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except BaseException:  # noqa: BLE001
            pass

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
