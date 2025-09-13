# agent/tools.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, conint
from langchain.tools import StructuredTool

from storage.paths import write_status, write_json
from agent.pipeline import (
    extract_and_build_gaps as _extract_async,
    extract_with_window as _extract_with_window_async,
)
from agent.utils import read_json, summarize_gaps
from agent.triage import (
    infer_title as _infer_title_async,
    imaging_verdict as _imaging_verdict_async,
    triage_pages as _triage_pages_async,
)

# -----------------
# Small helpers (centralized in agent.utils)
# -----------------


def _max_span() -> int:
    try:
        return max(0, min(10, int(os.getenv("MAX_SPAN", "4"))))
    except Exception:
        return 4


# -----------------
# Tool A: infer_title (from pages.json)
# -----------------
class InferTitleSchema(BaseModel):
    job_dir: str = Field(..., description="Absolute path to the job's artifacts directory.")
    max_pages: conint(ge=1, le=4) = Field(2, description="Number of early pages to consider.")
    max_chars: conint(ge=500, le=15000) = Field(3000, description="Max characters to pass to the LLM.")


async def _infer_title_tool_async(job_dir: str, max_pages: int = 2, max_chars: int = 3000) -> Dict[str, Any]:
    jdir = Path(job_dir)
    job_id = jdir.name
    pages_obj = read_json(jdir / "pages.json", {})
    pages = pages_obj if isinstance(pages_obj, list) else (pages_obj.get("pages") or [])

    write_status(job_id, state="running", step="title.infer.start", last_action="infer_title")
    resp = await _infer_title_async(pages, max_pages=max_pages, max_chars=max_chars)

    # Merge into meta.json
    meta_path = jdir / "meta.json"
    meta = read_json(meta_path, {}) if meta_path.exists() else {}
    meta.update({
        "title": (resp.get("title") or "")[:300],
        "title_confidence": float(resp.get("confidence", 0) or 0),
    })
    write_json(meta_path.as_posix(), meta)

    write_status(
        job_id,
        state="running",
        step="title.infer.done",
        last_action="infer_title",
        title=meta.get("title", ""),
        title_confidence=meta.get("title_confidence", 0.0),
    )
    return {"ok": True, "title": meta.get("title", ""), "confidence": meta.get("title_confidence", 0.0)}


infer_title_tool = StructuredTool.from_function(
    name="infer_title",
    description="Infer the paper title from early pages (writes meta.json).",
    func=lambda job_dir, max_pages=2, max_chars=3000: {"error": "use async"},
    coroutine=_infer_title_tool_async,
    args_schema=InferTitleSchema,
)


# -----------------
# Tool B: imaging_verdict (from pages.json)
# -----------------
class ImagingVerdictSchema(BaseModel):
    job_dir: str = Field(..., description="Absolute path to the job's artifacts directory.")


async def _imaging_verdict_tool_async(job_dir: str) -> Dict[str, Any]:
    jdir = Path(job_dir)
    job_id = jdir.name
    pages_obj = read_json(jdir / "pages.json", {})
    pages = pages_obj if isinstance(pages_obj, list) else (pages_obj.get("pages") or [])

    write_status(job_id, state="running", step="verdict.start", last_action="imaging_verdict")
    resp = await _imaging_verdict_async(pages)

    payload = {
        "is_imaging": bool(resp.get("is_imaging")),
        "modalities": resp.get("modalities") or [],
        "confidence": float(resp.get("confidence", 0) or 0),
        "reasons": resp.get("reasons") or [],
        "counter_signals": resp.get("counter_signals") or [],
    }
    write_json((jdir / "doc_flags.json").as_posix(), payload)
    write_status(
        job_id,
        state="running",
        step="verdict.done",
        last_action="imaging_verdict",
        **payload,
    )
    return {"ok": True, **payload}


imaging_verdict_tool = StructuredTool.from_function(
    name="imaging_verdict",
    description="Decide if the paper reports MRI acquisition (writes doc_flags.json).",
    func=lambda job_dir: {"error": "use async"},
    coroutine=_imaging_verdict_tool_async,
    args_schema=ImagingVerdictSchema,
)


# -----------------
# Tool C: triage_pages (from pages.json)
# -----------------
class TriagePagesSchema(BaseModel):
    job_dir: str = Field(..., description="Absolute path to the job's artifacts directory.")
    top_k: conint(ge=1, le=12) = Field(6, description="How many candidate pages to keep.")


async def _triage_pages_tool_async(job_dir: str, top_k: int = 6) -> Dict[str, Any]:
    jdir = Path(job_dir)
    job_id = jdir.name
    pages_obj = read_json(jdir / "pages.json", {})
    pages = pages_obj if isinstance(pages_obj, list) else (pages_obj.get("pages") or [])

    write_status(job_id, state="running", step="triage.start", last_action="triage_pages", top_k=top_k)
    sections = await _triage_pages_async(pages, top_k=top_k)
    write_json((jdir / "sections.json").as_posix(), sections)
    write_status(job_id, state="running", step="triage.done", last_action="triage_pages", top_k=top_k, sections=sections)
    return {"ok": True, "sections": sections}


triage_pages_tool = StructuredTool.from_function(
    name="triage_pages",
    description="Classify pages and choose top-K acquisition/method candidates (writes sections.json).",
    func=lambda job_dir, top_k=6: {"error": "use async"},
    coroutine=_triage_pages_tool_async,
    args_schema=TriagePagesSchema,
)


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


# Export tools list for the agent (pre-extraction and extraction tools)
TOOLS = [
    infer_title_tool,
    imaging_verdict_tool,
    triage_pages_tool,
    extract_and_build_gaps_tool,
    extract_with_window_tool,
]
__all__ = [
    "infer_title_tool",
    "imaging_verdict_tool",
    "triage_pages_tool",
    "extract_and_build_gaps_tool",
    "extract_with_window_tool",
    "TOOLS",
]
