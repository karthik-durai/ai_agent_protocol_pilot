# agent/tools.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, conint
from langchain.tools import StructuredTool

from storage.paths import write_status
from agent.pipeline import (
    extract_and_build_gaps as _extract_async,
    extract_with_window as _extract_with_window_async,
)
from agent.utils import read_json, summarize_gaps

# -----------------
# Small helpers (centralized in agent.utils)
# -----------------


def _max_span() -> int:
    try:
        return max(0, min(10, int(os.getenv("MAX_SPAN", "4"))))
    except Exception:
        return 4


# -----------------
# Tool 1: extract_and_build_gaps
# -----------------
class ExtractSchema(BaseModel):
    job_dir: str = Field(..., description="Absolute path to the job's artifacts directory.")


async def _extract_tool_async(job_dir: str) -> Dict[str, Any]:
    jdir = Path(job_dir)
    job_id = jdir.name

    write_status(job_id, state="running", step="extract.start")
    out = await _extract_async(job_dir)

    # Ensure we report gaps summary
    gap = read_json(jdir / "gap_report.json", {})
    gaps = summarize_gaps(gap)
    write_status(
        job_id,
        state="running",
        step="extract.done",
        last_action="extract_and_build_gaps",
        gaps_after=gaps,
    )
    return {
        "ok": True,
        "step": "extract",
        "gaps": gaps,
        "missing": int(gaps.get("missing", 0)),
        "conflicts": int(gaps.get("conflicts", 0)),
        "ambiguous": int(gaps.get("ambiguous", 0)),
        "summary": f"GAPS missing={gaps.get('missing',0)} conflicts={gaps.get('conflicts',0)} ambiguous={gaps.get('ambiguous',0)}",
    }


extract_and_build_gaps_tool = StructuredTool.from_function(
    name="extract_and_build_gaps",
    description="Baseline pass: use triage candidates as-is to extract protocol and build the gap report.",
    func=lambda job_dir: {"error": "use async"},  # agent will use coroutine
    coroutine=_extract_tool_async,
    args_schema=ExtractSchema,
)


# -----------------
# Tool 2: extract_with_window
# -----------------
class ExtractWithWindowSchema(BaseModel):
    job_dir: str = Field(..., description="Absolute path to the job's artifacts directory.")
    span: conint(ge=0, le=4) = Field(2, description="Half-span around triaged pages (0..4).")


async def _extract_with_window_tool_async(job_dir: str, span: int = 2) -> Dict[str, Any]:
    jdir = Path(job_dir)
    job_id = jdir.name
    span = max(0, min(_max_span(), int(span)))

    write_status(
        job_id,
        state="running",
        step="reextract_wide.start",
        last_action="extract_with_window",
        span=span,
    )

    out = await _extract_with_window_async(job_dir, span=span)

    # Extract explicit numeric cues for the agent
    before = out.get("before") or {}
    after = out.get("after") or summarize_gaps(read_json(jdir / "gap_report.json", {}))

    before_missing = int((before or {}).get("missing", 0))
    before_conflicts = int((before or {}).get("conflicts", 0))
    before_ambiguous = int((before or {}).get("ambiguous", 0))

    after_missing = int(after.get("missing", 0))
    after_conflicts = int(after.get("conflicts", 0))
    after_ambiguous = int(after.get("ambiguous", 0))

    improved = bool(out.get("improved"))
    pages = None
    try:
        pages = (out.get("args", {}) or {}).get("pages")
    except Exception:
        pages = None

    summary = (
        f"SPAN {span}; BEFORE m={before_missing} c={before_conflicts} a={before_ambiguous} "
        f"→ AFTER m={after_missing} c={after_conflicts} a={after_ambiguous}; improved={improved}"
    )

    # Ensure status reflects outcome
    write_status(
        job_id,
        state="running",
        step="reextract_wide.done",
        last_action="extract_with_window",
        gaps_after=after,
        span=span,
        improved=improved,
        before=before,
        pages=pages,
        summary=summary,
    )

    # Return enriched payload for the LLM's decision
    enriched = {
        "ok": True,
        "step": "extract_with_window",
        "span": span,
        "args": out.get("args", {"span": span, "pages": pages}),
        "pages": pages,
        "before": before,
        "after": after,
        "before_missing": before_missing,
        "before_conflicts": before_conflicts,
        "before_ambiguous": before_ambiguous,
        "after_missing": after_missing,
        "after_conflicts": after_conflicts,
        "after_ambiguous": after_ambiguous,
        "improved": improved,
        "summary": summary,
    }
    return enriched


extract_with_window_tool = StructuredTool.from_function(
    name="extract_with_window",
    description="Expand candidate pages by ±span and re-run extraction, then rebuild the gap report.",
    func=lambda job_dir, span=2: {"error": "use async"},  # agent will use coroutine
    coroutine=_extract_with_window_tool_async,
    args_schema=ExtractWithWindowSchema,
)


# Export tools list for the agent
TOOLS = [extract_and_build_gaps_tool, extract_with_window_tool]
__all__ = [
    "extract_and_build_gaps_tool",
    "extract_with_window_tool",
    "TOOLS",
]
