# agent/pipeline.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List

from storage.paths import write_status
from agent.protocol_card import run_protocol_extraction_async
from agent.gap_report import build_gap_report_llm_async
from agent.utils import read_json, summarize_gaps

# -----------------
# Small utilities (centralized in agent.utils)
# -----------------


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
    await build_gap_report_llm_async(art_dir=jdir.as_posix())

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


async def extract_with_window(job_dir: str, span: int) -> Dict[str, Any]:
    """
    Widen page candidates by ±span around triaged hits, run extraction, then rebuild gaps.
    Returns before/after gap summaries and whether things improved.
    """
    jdir = Path(job_dir)
    job_id = jdir.name

    # Clamp span from env, default 0..4
    try:
        max_span = max(0, min(10, int(os.getenv("MAX_SPAN", "4"))))
    except Exception:
        max_span = 4
    span = int(max(0, min(max_span, int(span))))

    write_status(job_id, state="running", step="reextract_wide.start", span=span, last_action="extract_with_window")

    pages_obj = read_json(jdir / "pages.json", {})
    sections_obj = read_json(jdir / "sections.json", {})

    pages: List[Dict[str, Any]] = (
        pages_obj if isinstance(pages_obj, list) else (pages_obj.get("pages") or [])
    )
    sections: Dict[str, Any] = (
        sections_obj if isinstance(sections_obj, dict) else {"candidates": []}
    )

    # Compute widened candidate pages
    candidates = sections.get("candidates") or []
    seed_pages: List[int] = []
    for c in candidates:
        try:
            p = int(c.get("page"))
            seed_pages.append(p)
        except Exception:
            continue

    max_idx = len(pages) - 1 if pages else 0
    widened_pages: List[int] = []
    for p in seed_pages:
        if p < 0:
            continue
        lo = max(0, p - span)
        hi = min(max_idx, p + span)
        for q in range(lo, hi + 1):
            if q not in widened_pages:
                widened_pages.append(q)
    if not widened_pages and max_idx >= 0:
        # Fallback: first few pages
        widened_pages = list(range(0, min(5, max_idx + 1)))
    widened_pages.sort()

    widened_sections = {"candidates": [{"page": p} for p in widened_pages]}

    # Before/after gaps
    before_gap = read_json(jdir / "gap_report.json", {})
    before = summarize_gaps(before_gap)

    # Run extraction on widened window
    await run_protocol_extraction_async(
        pages=pages,
        sections=widened_sections,
        art_dir=jdir.as_posix(),
    )

    # Rebuild gaps
    await build_gap_report_llm_async(art_dir=jdir.as_posix())

    after_gap = read_json(jdir / "gap_report.json", {})
    after = summarize_gaps(after_gap)
    improved = int(after.get("missing", 0)) < int(before.get("missing", 0))

    write_status(
        job_id,
        state="running",
        step="reextract_wide.done",
        last_action="extract_with_window",
        span=span,
        gaps_after=after,
        improved=improved,
        pages=widened_pages,
    )

    return {
        "ok": True,
        "step": "extract_with_window",
        "args": {"span": span, "pages": widened_pages},
        "before": before,
        "after": after,
        "improved": improved,
    }
