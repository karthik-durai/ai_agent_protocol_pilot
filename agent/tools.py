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

    write_status(job_id, state="running", step="mri_verdict.start", last_action="imaging_verdict")
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
        step="mri_verdict.done",
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
        "summary": f"GAPS missing={gaps.get('missing',0)}",
    }


extract_and_build_gaps_tool = StructuredTool.from_function(
    name="extract_and_build_gaps",
    description="Baseline pass: use triage candidates as-is to extract protocol and build the gap report.",
    func=lambda job_dir: {"error": "use async"},  # agent will use coroutine
    coroutine=_extract_tool_async,
    args_schema=ExtractSchema,
)


# Export tools list for the agent (pre-extraction and extraction tools)
TOOLS = [
    infer_title_tool,
    imaging_verdict_tool,
    triage_pages_tool,
    extract_and_build_gaps_tool,
]
__all__ = [
    "infer_title_tool",
    "imaging_verdict_tool",
    "triage_pages_tool",
    "extract_and_build_gaps_tool",
    "TOOLS",
]
