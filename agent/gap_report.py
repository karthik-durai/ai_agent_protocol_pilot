from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from storage.paths import write_json
from agent.utils import read_json

# -----------------------------
# Constants
# -----------------------------
EXTRACTED_FILENAME = "imaging_extracted.json"
CANDIDATES_FILENAME = "imaging_candidates.jsonl"
FLAGS_FILENAME = "doc_flags.json"
OUTPUT_FILENAME = "gap_report.json"

# Required MRI fields (ordered)
REQUIRED_FIELDS: List[str] = [
    "sequence_type",
    "TR_ms",
    "TE_ms",
    "flip_deg",
    "field_strength_T",
    "inplane_res_mm",
    "slice_thickness_mm",
]

def _write_stub_gap_report(
    art_dir: str,
    modalities: List[str],
    extracted_exists: bool,
    candidates_exists: bool,
) -> str:
    """Write an empty/stub gap report and return its path."""
    out_path = os.path.join(art_dir, OUTPUT_FILENAME)
    stub = {
        "schema_version": 1,
        "policy": "llm_gap_v1_stub",
        "modality": modalities or ["MRI"],
        "summary": {"missing": 0, "questions": 0},
        "missing": [],
        "missing_low_conf": [],
        "questions": [],
        "provenance": {
            "from_extracted": extracted_exists,
            "from_candidates": candidates_exists,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(out_path, stub)
    return out_path

# -----------------------------
# Public API
# -----------------------------
async def build_gap_report_async(art_dir: str) -> str:
    "Build gap_report.json deterministically from winners (no LLM)."
    
    extracted_path = os.path.join(art_dir, EXTRACTED_FILENAME)
    candidates_path = os.path.join(art_dir, CANDIDATES_FILENAME)
    flags_path = os.path.join(art_dir, FLAGS_FILENAME)
    out_path = os.path.join(art_dir, OUTPUT_FILENAME)

    extracted = read_json(extracted_path, {})
    fields = extracted.get("fields") or {}
    flags = read_json(flags_path, {})
    modalities = flags.get("modalities") or ["MRI"]

    extracted_exists = os.path.exists(extracted_path)
    candidates_exists = os.path.exists(candidates_path)
    candidates_has_content = (
        os.path.exists(candidates_path) and os.path.getsize(candidates_path) > 0
    )

    # If nothing to work with, write a stub
    if not fields and not candidates_has_content:
        return _write_stub_gap_report(art_dir, modalities, extracted_exists, candidates_exists)

    present = set(fields.keys())
    missing = [f for f in REQUIRED_FIELDS if f not in present]
    missing_low_conf: List[str] = []

    q_templates = {
        "TR_ms": "Please confirm the repetition time (TR) used.",
        "TE_ms": "Please confirm the echo time (TE) used.",
        "flip_deg": "Please confirm the flip angle (degrees).",
        "field_strength_T": "Please confirm the MRI field strength (Tesla).",
        "inplane_res_mm": "Please confirm the in‑plane resolution in mm (e.g., 1.0 × 1.0 or isotropic).",
        "slice_thickness_mm": "Please confirm the slice thickness (mm).",
        "sequence_type": "Please confirm the MRI sequence type used (e.g., T1w, T2w, FLAIR, DWI, EPI).",
    }
    questions: List[Dict[str, Any]] = [
        {
            "field": f,
            "question": q_templates.get(f, f"Please confirm the value for {f}."),
            "rationale": "required MRI parameter missing from winners",
            "evidence_pages": [],
        }
        for f in missing[:3]
    ]

    resp = {
        "schema_version": 1,
        "policy": "code_gap_v1",
        "modality": modalities or ["MRI"],
        "summary": {"missing": len(missing), "questions": len(questions)},
        "missing": missing,
        "missing_low_conf": missing_low_conf,
        "questions": questions,
        "provenance": {
            "from_extracted": extracted_exists,
            "from_candidates": candidates_exists,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    write_json(out_path, resp)
    return out_path
