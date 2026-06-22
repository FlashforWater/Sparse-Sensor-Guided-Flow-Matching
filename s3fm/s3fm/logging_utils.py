"""Lightweight experiment-result logging.

Spec requirement (M0): record experiment results to CSV or JSONL. We use JSONL
for scalar metric streams (one JSON object per line, append-only) because it is
trivial to parse, robust to interrupted runs, and human-readable. Each run gets
its own output directory holding the resolved config and a ``metrics.jsonl``.

This is intentionally minimal: no external experiment-tracking dependency. The
spec's diagnostic trajectories (J_obs, guidance norm, ...) are just scalar
streams that land here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSONL metric logger.

    Each call to :meth:`log` writes one line. A monotonically increasing ``step``
    is recorded automatically unless provided. A wall-clock timestamp (seconds
    since the logger was created) is attached for runtime accounting.
    """

    def __init__(self, out_dir: str | Path, filename: str = "metrics.jsonl"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._step = 0
        self._t0 = time.monotonic()
        # truncate any previous file so a re-run starts clean
        self.path.write_text("")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = max(self._step, step + 1)
        record: dict[str, Any] = {
            "step": step,
            "wall_time_s": round(time.monotonic() - self._t0, 6),
        }
        record.update(metrics)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Read every logged record back (used by tests and evaluation)."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
