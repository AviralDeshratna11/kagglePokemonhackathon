"""Cache the engine's card/attack tables to JSON.

Needed only for analysis on hosts that cannot load the shared library (an arm64
laptop running an older kaggle-environments, say). The agent itself always reads
the tables from the live engine, so this file never ships in the submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/card_db.json")
    ap.add_argument("--csv", default=None, help="EN_Card_Data.csv, for expansion metadata")
    args = ap.parse_args(argv)

    from ..core.carddb import get_card_db

    db = get_card_db(csv_path=args.csv)
    blob = db.to_json()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(blob['cards'])} cards, {len(blob['attacks'])} attacks)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
