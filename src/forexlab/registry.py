"""Experiment log: an append-only record of every test the lab runs.

Data snooping is the main way a research loop fools itself. Testing hundreds
of variants guarantees that some look good by luck. The log makes the number
of tries visible and feeds it into the deflated Sharpe ratio, so a result is
always judged against how hard we looked for it.

The log is a JSON-lines file kept in git. Never edit or delete lines in it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("research_log/experiments.jsonl")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def config_id(family: str, params: dict) -> str:
    blob = json.dumps({"family": family, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _clean(v):
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return str(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


class ExperimentLog:
    def __init__(self, path: Path = DEFAULT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._commit = _git_commit()

    def record(self, *, run_id: str, kind: str, symbol: str, family: str, params: dict,
               data_fingerprint: str, segment: dict, metrics: dict, extra: dict | None = None) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": run_id,
            "kind": kind,                     # "train", "test", "walk_forward", "holdout"
            "symbol": symbol,
            "family": family,
            "config_id": config_id(family, params),
            "params": params,
            "data": data_fingerprint,
            "segment": segment,
            "metrics": _clean(metrics),
            "git_commit": self._commit,
        }
        if extra:
            entry["extra"] = _clean(extra)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def trials(self, symbol: str, family: str | None = None) -> dict:
        """Distinct configurations ever evaluated on in-sample data for this symbol,
        with the mean train Sharpe of each. Counted across ALL runs and families by
        default, because choosing between families is a trial too."""
        per_cfg: dict[str, list[float]] = {}
        for e in self.entries():
            if e["symbol"] != symbol or e["kind"] != "train":
                continue
            if family and e["family"] != family:
                continue
            s = e["metrics"].get("sharpe")
            if isinstance(s, (int, float)):
                per_cfg.setdefault(e["config_id"], []).append(float(s))
        return {cid: sum(v) / len(v) for cid, v in per_cfg.items()}
