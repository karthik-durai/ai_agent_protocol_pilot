from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: str | Path, default: Any) -> Any:
    """Read JSON from `path` and return `default` on any error.

    Accepts both `str` and `Path` inputs. Uses UTF-8 decoding.
    """
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def summarize_gaps(gap: Dict[str, Any] | None) -> Dict[str, int]:
    """Return counts for missing/ambiguous/conflicts from a gap-report dict."""
    if not isinstance(gap, dict):
        return {"missing": 0, "ambiguous": 0, "conflicts": 0}
    return {
        "missing": len(gap.get("missing", []) or []),
        "ambiguous": len(gap.get("ambiguous", []) or []),
        "conflicts": len(gap.get("conflicts", []) or []),
    }


def summarize_gaps_from_dir(job_dir: str) -> Dict[str, int]:
    """Read `gap_report.json` from `job_dir` and summarize counts."""
    gap = read_json(Path(job_dir) / "gap_report.json", {})
    return summarize_gaps(gap)


__all__ = [
    "read_json",
    "summarize_gaps",
    "summarize_gaps_from_dir",
]

