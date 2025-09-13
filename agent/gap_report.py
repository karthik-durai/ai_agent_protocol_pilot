from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from storage.paths import write_json
from agent.llm_client import llm_json

# ============================================================
# Gap Report v1 — LLM-assisted
# ------------------------------------------------------------
# Inputs:
#   - artifacts/<job>/imaging_extracted.json   (winners)
#   - artifacts/<job>/imaging_candidates.jsonl (all hits)
#   - (optional) artifacts/<job>/doc_flags.json (modalities)
# Output:
#   - artifacts/<job>/gap_report.json
# Behavior:
#   - Summarize winners + grouped candidates
#   - Ask LLM (STRICT JSON) to produce missing values + questions
#   - Validate; on failure write a minimal stub (contract-safe)
# ============================================================

# -----------------------------
# Basic IO helpers
# -----------------------------
def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

# -----------------------------
# Value normalization for grouping representatives
# -----------------------------
def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _norm_units_value(field: str, value: Any, units: str) -> Any:
    """
    Normalize obvious units for comparison (MRI). Canonical forms:
      - TR_ms, TE_ms -> ms (float)
      - flip_deg -> deg (float)
      - field_strength_T -> Tesla (float)
      - inplane_res_mm -> [x,y] in mm (floats)
      - slice_thickness_mm -> mm (float)
      - sequence_type -> lowercase trimmed string
    """
    u = (units or "").strip().lower()

    if field in ("TR_ms", "TE_ms"):
        v = _to_float(value)
        if v is None:
            return None
        # convert seconds to ms if units indicate seconds
        if u in {"s", "sec", "second", "seconds"}:
            v *= 1000.0
        return v

    if field == "flip_deg":
        v = _to_float(value)
        return v

    if field == "field_strength_T":
        v = _to_float(value)
        return v

    if field == "inplane_res_mm":
        # expect [x,y] floats; allow single isotropic number
        if isinstance(value, list):
            out: List[float] = []
            for a in value:
                fv = _to_float(a)
                if fv is None:
                    return None
                out.append(fv)
            if len(out) == 1:
                out = [out[0], out[0]]
            if len(out) != 2:
                return None
            return out
        v = _to_float(value)
        if v is not None:
            return [v, v]
        return None

    if field == "slice_thickness_mm":
        v = _to_float(value)
        if v is None:
            return None
        if u in ("cm", "centimeter", "centimeters"):
            v *= 10.0
        if u in ("µm", "um", "micrometer", "micrometers"):
            v /= 1000.0
        return v

    if field == "sequence_type":
        try:
            return str(value).strip().lower()
        except Exception:
            return None

    # Fallback: return as-is
    return value

def _group_key(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field in ("TR_ms", "TE_ms", "flip_deg", "field_strength_T", "slice_thickness_mm"):
        return round(float(value), 3)
    if field == "inplane_res_mm":
        return tuple(round(float(v), 3) for v in value) if isinstance(value, list) and len(value) == 2 else None
    if field == "sequence_type":
        return str(value).strip().lower()
    return value

def _representatives_by_field(cands: List[Dict[str, Any]]) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    """
    Returns: field -> { group_key -> best_candidate_dict }
    """
    by_field: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for c in cands:
        field = c.get("field")
        if not field:
            continue
        val = c.get("value")
        units = c.get("units") or ""
        norm_val = _norm_units_value(field, val, units)
        key = _group_key(field, norm_val)
        if key is None:
            continue
        repmap = by_field.setdefault(field, {})
        prev = repmap.get(key)
        if prev is None or float(c.get("confidence", 0) or 0) > float(prev.get("confidence", 0) or 0):
            repmap[key] = {
                "value": val,  # keep original formatting
                "norm_value": norm_val,
                "page": int(c.get("page", -1)),
                "confidence": float(c.get("confidence", 0) or 0),
                "evidence": (c.get("evidence") or "")[:200],
                "units": units,
            }
    return by_field

# -----------------------------
# Prompt assembly
# -----------------------------
_FIELDS_ORDER = [
    "sequence_type",
    "TR_ms",
    "TE_ms",
    "flip_deg",
    "field_strength_T",
    "inplane_res_mm",
    "slice_thickness_mm",
]

def _representatives_sorted(cands: List[Dict[str, Any]], per_field_limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Return top-N representative candidates per field, one per distinct value (by confidence)."""
    reps_map = _representatives_by_field(cands)  # field -> key -> rep
    out: Dict[str, List[Dict[str, Any]]] = {}
    for field, keymap in reps_map.items():
        reps = list(keymap.values())
        reps.sort(key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)
        out[field] = reps[:per_field_limit]
    return out

def _group_candidates_for_prompt(cands: List[Dict[str, Any]], per_field_limit: int = 5) -> str:
    """Produce a compact text block grouped by field for the user prompt."""
    grouped = _representatives_sorted(cands, per_field_limit)
    lines: List[str] = []
    for fname in _FIELDS_ORDER:
        reps = grouped.get(fname) or []
        if not reps:
            continue
        lines.append(f"{fname}:")
        for it in reps:
            val = it.get("value")
            units = (it.get("units") or "").replace('"', '\\"')
            page = it.get("page")
            ev = (it.get("evidence") or "").replace('"', '\\"')[:160]
            conf = it.get("confidence", 0)
            lines.append(f"- {{value: {val}, units: \"{units}\", page: {page}, evidence: \"{ev}\", confidence: {conf}}}")
        lines.append("")  # blank
    return "\n".join(lines).strip()

# -----------------------------
# LLM prompts (STRICT JSON)
# -----------------------------
GAP_SYS = (
    "You are a careful gap adjudicator for MRI imaging methods. Using extracted winners "
    "and grouped candidates with evidence, identify ONLY missing fields — parameters that are not extracted "
    "or have very low confidence — and draft 1–3 short author questions to resolve the most impactful missing values.\n\n"
    "Rules:\n"
    "- Low confidence (missing_low_conf): confidence < 0.55.\n"
    "- Focus exclusively on missing fields; do not include 'ambiguous' or 'conflicts' keys.\n"
    "- Required MRI fields: sequence_type, TR_ms, TE_ms, flip_deg, field_strength_T, inplane_res_mm, slice_thickness_mm.\n\n"
    "Output STRICT JSON only in the requested schema. Do not invent values not supported by evidence."
)

GAP_USER_TMPL = """MODALITY: {modality}

WINNERS (from extracted):
{winners_pretty}

CANDIDATES GROUPED BY FIELD (each item: {{value, units, page, evidence, confidence}}):
{grouped}

Return EXACTLY this JSON (omit 'ambiguous' and 'conflicts' entirely):
{{
  "schema_version": 1,
  "policy": "llm_gap_v1",
  "modality": ["MRI"],
  "summary": {{ "missing": <int>, "questions": <int> }},
  "missing": ["TR_ms"],
  "missing_low_conf": ["inplane_res_mm"],
  "questions": [
    {{
      "field": "TR_ms",
      "question": "Please confirm the repetition time (TR) used for the reported scans; it is not explicitly reported.",
      "rationale": "critical parameter missing",
      "evidence_pages": []
    }}
  ],
  "provenance": {{ "from_extracted": true, "from_candidates": true }}
}}
"""

# -----------------------------
# Output validation + stub fallback
# -----------------------------
def _validate_llm_gap(obj: Dict[str, Any]) -> bool:
    """Minimal schema validation for LLM gap output."""
    try:
        if not isinstance(obj, dict):
            return False
        need = {"schema_version","policy","modality","summary","missing","missing_low_conf","questions","provenance"}
        if not need.issubset(set(obj.keys())):
            return False
        if not isinstance(obj["schema_version"], int): return False
        if not isinstance(obj["policy"], str): return False
        if not isinstance(obj["modality"], list): return False
        if not isinstance(obj["summary"], dict): return False
        for k in ("missing","missing_low_conf","questions"):
            if not isinstance(obj[k], list): return False
        for q in obj["questions"]:
            if not isinstance(q, dict) or "field" not in q or "question" not in q: return False
        return True
    except Exception:
        return False

def _write_stub_gap_report(art_dir: str, modalities: List[str], extracted_exists: bool, candidates_exists: bool) -> str:
    out_path = os.path.join(art_dir, "gap_report.json")
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
async def build_gap_report_llm_async(art_dir: str) -> str:
    """
    Build gap_report.json using an LLM (STRICT JSON). On any failure,
    write a minimal stub report to keep the artifact contract stable.
    Returns the written file path.
    """
    extracted_path = os.path.join(art_dir, "imaging_extracted.json")
    candidates_path = os.path.join(art_dir, "imaging_candidates.jsonl")
    flags_path = os.path.join(art_dir, "doc_flags.json")
    out_path = os.path.join(art_dir, "gap_report.json")

    extracted = _read_json(extracted_path)
    fields = extracted.get("fields") or {}
    candidates = _read_jsonl(candidates_path)
    flags = _read_json(flags_path)
    modalities = flags.get("modalities") or ["MRI"]

    extracted_exists = os.path.exists(extracted_path)
    candidates_exists = os.path.exists(candidates_path)

    # If nothing to work with, write a stub
    if not fields and not candidates:
        return _write_stub_gap_report(art_dir, modalities, extracted_exists, candidates_exists)

    winners_pretty = json.dumps(fields, indent=2, ensure_ascii=False)
    grouped = _group_candidates_for_prompt(candidates, per_field_limit=5)
    user = GAP_USER_TMPL.format(
        modality=", ".join(modalities),
        winners_pretty=winners_pretty,
        grouped=grouped or "(no candidates)"
    )

    try:
        resp = await llm_json(GAP_SYS, user)
        if not _validate_llm_gap(resp):
            return _write_stub_gap_report(art_dir, modalities, extracted_exists, candidates_exists)

        # Normalize summary counts to match arrays (missing-only focus)
        sm = resp.get("summary") or {}
        sm["missing"] = len(resp.get("missing", []))
        sm["questions"] = len(resp.get("questions", []))
        resp["summary"] = sm

        # Ensure provenance is complete
        prov = resp.get("provenance") or {}
        prov.update({
            "from_extracted": extracted_exists,
            "from_candidates": candidates_exists,
            "generated_at": datetime.now(timezone.utc).isoformat()
        })
        resp["provenance"] = prov

        write_json(out_path, resp)
        return out_path
    except Exception:
        return _write_stub_gap_report(art_dir, modalities, extracted_exists, candidates_exists)
