# agent/pipeline.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from storage.paths import write_status
from agent.protocol_card import run_protocol_extraction_async
from agent.gap_report import build_gap_report_async
from agent.utils import read_json, summarize_gaps

# -----------------
# Core primitives (Step 1)
# -----------------

async def extract_and_build_gaps(job_dir: str) -> Dict[str, Any]:
    """
    Baseline pass: use triage candidates as-is → extract + adjudicate → build gaps.
    Expects pages.json and sections.json to already exist in job_dir.
    Writes status before/after and returns a small summary.
    """
    jdir = Path(job_dir)
    job_id = jdir.name

    write_status(job_id, state="running", step="extract.start")

    pages_obj = read_json(jdir / "pages.json", {})
    sections_obj = read_json(jdir / "sections.json", {})

    pages: List[Dict[str, Any]] = (
        pages_obj if isinstance(pages_obj, list) else (pages_obj.get("pages") or [])
    )
    sections: Dict[str, Any] = (
        sections_obj if isinstance(sections_obj, dict) else {"candidates": []}
    )

    # Run extraction + adjudication
    await run_protocol_extraction_async(
        pages=pages,
        sections=sections,
        art_dir=jdir.as_posix(),
    )

    # Build gap report
    await build_gap_report_async(art_dir=jdir.as_posix())

    gap = read_json(jdir / "gap_report.json", {})
    summary = summarize_gaps(gap)

    write_status(
        job_id,
        state="running",
        step="extract.done",
        last_action="extract_and_build_gaps",
        gaps_after=summary,
    )
    return {"ok": True, "gaps": summary, "step": "extract"}
