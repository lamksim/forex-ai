"""Data splits: the sealed holdout and walk-forward folds.

The most recent `holdout_frac` of each dataset is the holdout. Research code
never receives it. A strategy version may be evaluated on the holdout ONCE,
through HoldoutVault, and every access is logged. If a strategy is changed
after seeing holdout results, it is a new version and the holdout result of
the old one still counts against the lab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_VAULT = Path("research_log/holdout_access.jsonl")


def split_holdout(bars: pd.DataFrame, holdout_frac: float = 0.2) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (research_bars, holdout_start). The holdout boundary is aligned to a day."""
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac must be between 0 and 1")
    span = bars.index[-1] - bars.index[0]
    cut = (bars.index[0] + span * (1 - holdout_frac)).normalize()
    return bars[bars.index < cut], cut


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp   # exclusive
    test_end: pd.Timestamp    # exclusive; test starts at train_end


def walk_forward_folds(index: pd.DatetimeIndex, train_months: int = 24, test_months: int = 6) -> list[Fold]:
    """Rolling folds: fit on `train_months`, test on the following `test_months`, then roll."""
    start = index[0].normalize()
    last = index[-1]
    folds = []
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if train_end >= last:
            break
        folds.append(Fold(train_start, train_end, min(test_end, last + pd.Timedelta(seconds=1))))
        if test_end > last:
            break
        train_start = train_start + pd.DateOffset(months=test_months)
    return folds


class HoldoutViolation(RuntimeError):
    pass


class HoldoutVault:
    def __init__(self, path: Path = DEFAULT_VAULT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def accesses(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def open(self, *, symbol: str, version: str, config_id: str, data_fingerprint: str,
             reason: str) -> None:
        """Record an access, refusing a second look at the same strategy version."""
        for a in self.accesses():
            if a["symbol"] == symbol and (a["version"] == version or a["config_id"] == config_id):
                raise HoldoutViolation(
                    f"Holdout for {symbol} was already used by version {a['version']} "
                    f"(config {a['config_id']}) on {a['ts']}. Each strategy version gets one look."
                )
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": symbol, "version": version, "config_id": config_id,
            "data": data_fingerprint, "reason": reason,
            "prior_looks_on_symbol": sum(1 for a in self.accesses() if a["symbol"] == symbol),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
