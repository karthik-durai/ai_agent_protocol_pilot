# agent/protocol_card.py
from __future__ import annotations
import os
from typing import Any, Dict, List
from storage.paths import write_json
import json
from agent.llm_client import llm_json_typed, llm_json
from agent.schemas import ExtractionCandidates, AdjudicationResponse

CANDIDATES_FILENAME = "imaging_candidates.jsonl"
EXTRACTED_FILENAME = "imaging_extracted.json"

# --- M3a Step 2: strict-JSON extraction prompts (MRI) ---
EXTRACT_MRI_SYS = (
    "You are a precise information extraction model for MRI imaging methods. "
    "Extract ONLY when the text asserts the actual acquisition used in THIS paper. "
    "Anchor every value to an exact substring. Normalize MRI units to TR/TE in ms, flip in degrees, "
    "field strength in Tesla, in‑plane resolution in mm, and slice thickness in mm. "
    "Output STRICT JSON only as specified. Do not include commentary.\n\n"
    "Acceptance cues (extract ONLY when present):\n"
    "- Explicit assertions: 'was', 'were', 'used', 'set to', 'acquired with', 'parameters were', 'TR =', "
    "table cells with concrete values for this study.\n"
    "- Clear table entries of study parameters (not recommendations/examples).\n\n"
    "Rejection cues (NEVER extract from these; omit instead):\n"
    "- Example/generic language: 'e.g.', 'for example', 'such as', 'for instance', 'typically', 'commonly', 'usually', "
    "'may', 'might', 'can', 'recommended', 'default', 'illustrative'.\n"
    "- Guidance or background text not describing this study's acquisition.\n"
    "- In‑plane resolution: NEVER derive from FOV; strings like 'FOV', 'FOV (mm2)', 'mm^2', or dimensions that clearly denote field of view MUST NOT be mapped to in‑plane resolution. "
    "Extract resolution only from phrases like 'resolution', 'in‑plane', 'voxel', or 'isotropic'.\n\n"
    "If uncertain or only examples are present, DO NOT emit a candidate (do not guess)."
)

EXTRACT_MRI_USER_TMPL = """TEXT WINDOW (may include tables/symbols/units):
\"\"\"
{window_text}
\"\"\"

Center page index (zero-based): {center_page}

IMPORTANT: Use only the TEXT WINDOW above as evidence. Ignore any values in these instructions; do NOT copy example numbers. Extract only if supported by the TEXT WINDOW.

FIELDS TO EXTRACT (emit ONLY when supported by explicit text in the TEXT WINDOW):
- sequence_type (string) — explicit sequence name used (e.g., T1w, T2w, FLAIR, DWI, EPI, GRE, SE, MPRAGE)
- TR_ms (number, ms) — explicit TR reported in the text
- TE_ms (number, ms) — explicit TE reported in the text
- flip_deg (number, degrees) — explicit flip angle reported in the text
- field_strength_T (number, Tesla) — explicit field strength reported in the text
- inplane_res_mm ([x,y] numbers, mm) — explicit in‑plane resolution wording (e.g., "resolution", "in‑plane", "voxel", "isotropic"); if isotropic, output [v,v]; NEVER derive from FOV
- slice_thickness_mm (number, mm) — explicit slice thickness reported in the text

Return EXACTLY this JSON:
{{
  "candidates": [
    {{
      "field": "<one of the fields above>",
      "page": <int page index to attribute (use the center page index unless evidence clearly refers to another)>,
      "raw_span": "<short exact substring from the text>",
      "value": <number|string|[number,number]>,
      "units": "<'ms'|'deg'|'T'|'mm'|''>",
      "evidence": "<short exact substring (<=100 chars)>",
      "confidence": <0.0-1.0>,
      "notes": "<optional, short>"
    }}
  ]
}}

Rules:
- Use ONLY the TEXT WINDOW as evidence; DO NOT copy any numbers from these instructions.
- Emit entries only with direct textual support that the parameter was used in THIS study; DO NOT guess.
- Acceptance cues: explicit assertions (e.g., "was used", "were", "set to", "acquired with", table cells of study parameters).
- Rejection cues: example/generic language (e.g., "e.g.", "for example", "such as", "typically", "recommended", "default"). Omit in such cases.
- Normalize TR seconds→ms; accept TR/TE shorthand like "TR/TE = a/b ms" and split accordingly.
- Accept decimal comma and normalize.
- In‑plane resolution: require wording like "resolution", "in‑plane", "voxel", or "isotropic". If isotropic with value v mm → output [v, v]; if text states x × y mm → output [x, y]. NEVER derive from FOV.
- If uncertain, or only examples are present, OMIT the entry entirely (do not output a candidate).
- STRICT JSON only."""


def build_windows(pages: List[Dict[str, Any]], center_page: int, span: int = 1) -> tuple[str, int]:
    """
    Join text from [center_page - span, center_page, center_page + span] (bounds-checked)
    and return (window_text, attributed_center_index).
    """
    if not pages:
        return "", center_page
    start = max(0, center_page - span)
    end = min((pages[-1]["page"] if pages else center_page) + 1, center_page + span + 1)
    chunks: list[str] = []
    for p in pages:
        idx = p.get("page", -1)
        if start <= idx < end:
            t = (p.get("text") or "").strip()
            if t:
                chunks.append(t)
    window_text = "\n\n".join(chunks)
    return window_text, center_page

async def extract_mri_common(window_text: str, center_page: int) -> List[Dict[str, Any]]:
    """
    Call the LLM with strict JSON prompt and return a list of candidate dicts.
    Never raises on JSON issues—returns [] instead.
    """
    if not window_text.strip():
        return []
    user = EXTRACT_MRI_USER_TMPL.format(window_text=window_text, center_page=center_page)
    try:
        parsed = await llm_json_typed(EXTRACT_MRI_SYS, user, ExtractionCandidates)
        items = parsed.candidates
    except Exception:
        # Fallback to untyped parsing if schema validation fails
        try:
            resp = await llm_json(EXTRACT_MRI_SYS, user)
            items = resp.get("candidates") or []
        except Exception:
            return []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            field = it.field
            page = int(getattr(it, 'page', center_page) or center_page)
            raw_span = (it.raw_span or "").strip()
            value = it.value
            units = (it.units or "").strip()
            evidence = (it.evidence or "").strip()
            conf = float(it.confidence or 0)
            notes = it.notes
            if field and raw_span and evidence:
                out.append({
                    "field": field,
                    "page": page,
                    "raw_span": raw_span,
                    "value": value,
                    "units": units,
                    "evidence": evidence[:100],
                    "confidence": conf,
                    "notes": notes if isinstance(notes, str) else ""
                })
        except Exception:
            continue
    return out

async def run_protocol_extraction_async(
    pages: List[Dict[str, Any]] | None,
    sections: Dict[str, Any] | None,
    art_dir: str
) -> Dict[str, str]:
    """
    M3a Step 2: Populate imaging_candidates.jsonl by running extraction
    over windows around triaged pages. Does not aggregate winners yet.
    """
    os.makedirs(art_dir, exist_ok=True)
    cand_path = os.path.join(art_dir, CANDIDATES_FILENAME)
    # Ensure file exists (append mode)
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("")

    # Load sections if not provided
    if sections is None:
        sec_path = os.path.join(art_dir, "sections.json")
        if os.path.exists(sec_path):
            try:
                with open(sec_path, "r", encoding="utf-8") as f:
                    sections = json.load(f)
            except Exception:
                sections = None

    if not pages or not sections or not sections.get("candidates"):
        return {"candidates": cand_path}

    # Iterate candidate pages and append JSONL lines
    with open(cand_path, "a", encoding="utf-8") as outf:
        for c in sections.get("candidates", []):
            try:
                pidx = int(c.get("page", -1))
                if pidx < 0:
                    continue
                window, center = build_windows(pages, pidx, span=1)
                hits = await extract_mri_common(window, center)
                for h in hits:
                    outf.write(json.dumps(h, ensure_ascii=False) + "\n")
            except Exception:
                continue

    # After writing candidates, run adjudication (LLM chooses winners)
    try:
        res = await adjudicate_candidates_async(art_dir)
        out = {"candidates": cand_path}
        out.update(res)
        return out
    except Exception:
        return {"candidates": cand_path}

# --- Step 3: adjudication (LLM chooses winners) ---

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items: List[Dict[str, Any]] = []
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

ADJ_SYS = (
    "You are a careful adjudicator for MRI acquisition parameters. Use ONLY the provided candidates "
    "(with their evidence) to select at most one best value per field. If no acceptable candidate exists, omit the field.\n\n"
    "Acceptance criteria (apply EXACTLY):\n"
    "- Candidates must explicitly describe parameters USED IN THIS STUDY (not examples/generic text).\n"
    "- Confidence threshold: choose a winner only if confidence ≥ 0.55. Otherwise omit the field.\n"
    "- Reject candidates whose evidence contains phrases like 'e.g.', 'for example', 'such as', 'typically', 'commonly', 'usually', 'recommended', 'default', 'illustrative', 'not found', 'not reported'.\n"
    "- In‑plane resolution: NEVER accept evidence that mentions FOV (e.g., 'FOV', 'FOV (mm2)', 'mm^2'); accept only text that uses 'resolution', 'in‑plane', 'voxel', or 'isotropic'.\n"
    "- Prefer explicit labels/tables and higher confidence; break ties with clearer evidence consistency.\n\n"
    "Output contract:\n"
    "- Output STRICT JSON only.\n"
    "- Do NOT invent or infer values not present in the candidate list.\n"
    "- For any field you include, ensure the chosen candidate meets the criteria above; otherwise omit the field."
)

ADJ_USER_TMPL = """CANDIDATES grouped by field. Each item: {{value, units, page, evidence, confidence}}.

{grouped}

Return EXACTLY this JSON:
{{
  "fields": {{
    "sequence_type": {{"value": "<str>", "units": "", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "TR_ms": {{"value": <number>, "units": "ms", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "TE_ms": {{"value": <number>, "units": "ms", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "flip_deg": {{"value": <number>, "units": "deg", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "field_strength_T": {{"value": <number>, "units": "T", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "inplane_res_mm": {{"value": [<number>,<number>], "units": "mm", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "slice_thickness_mm": {{"value": <number>, "units": "mm", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}}
  }}
}}

Rules:
- Choose at most one best entry per field; if NO acceptable candidate exists, OMIT the field.
- Accept a candidate only if its confidence ≥ 0.55 AND its evidence is an explicit assertion for THIS study (not an example).
- Reject candidates whose evidence contains: e.g., for example, such as, typically, commonly, usually, recommended, default, illustrative, not found, not reported.
- For in‑plane resolution, NEVER use FOV evidence (e.g., "FOV", "FOV (mm2)", "mm^2"); only accept text that uses resolution/in‑plane/voxel/isotropic. If isotropic value v → [v,v].
- Prefer explicit labels/tables and higher confidence; break ties with clearer evidence and consistency.
- Units: TR/TE in ms, flip in deg, field strength in T, in‑plane in mm.
- STRICT JSON only; do not invent values that are not present in the candidates list.
"""


_FIELDS = [
    "sequence_type",
    "TR_ms",
    "TE_ms",
    "flip_deg",
    "field_strength_T",
    "inplane_res_mm",
    "slice_thickness_mm",
]

def _group_candidates_for_prompt(cands: List[Dict[str, Any]], per_field_limit: int = 5) -> str:
    by: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _FIELDS}
    for c in cands:
        f = c.get("field")
        if f in by:
            by[f].append(c)
    lines: List[str] = []
    for f in _FIELDS:
        lst = sorted(by[f], key=lambda x: float(x.get("confidence",0) or 0), reverse=True)[:per_field_limit]
        if not lst:
            continue
        lines.append(f"{f}:")
        for it in lst:
            val = it.get("value")
            units = it.get("units") or ""
            page = it.get("page")
            ev = (it.get("evidence") or "")[:120]
            conf = it.get("confidence")
            lines.append(f"- {{value: {val}, units: \"{units}\", page: {page}, evidence: \"{ev}\", confidence: {conf}}}")
        lines.append("")  # blank line
    return "\n".join(lines).strip()

def _coerce_field(fname: str, entry: Dict[str, Any] | Any) -> Dict[str, Any] | None:
    # Accept dicts or Pydantic models from typed parsing
    if hasattr(entry, "model_dump"):
        try:
            entry = entry.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
    if not isinstance(entry, dict):
        return None
    v = entry.get("value")
    u = entry.get("units") or ""
    p = entry.get("page")
    e = (entry.get("evidence") or "").strip()
    c = entry.get("confidence", 0)
    r = (entry.get("reason") or "").strip()
    try:
        if fname in ("TR_ms", "TE_ms", "flip_deg", "field_strength_T", "slice_thickness_mm"):
            v = float(v)
        elif fname == "inplane_res_mm":
            if isinstance(v, list) and len(v) == 2:
                v = [float(v[0]), float(v[1])]
            else:
                return None
            u = "mm"
        elif fname == "sequence_type":
            v = str(v)
            u = ""
        else:
            return None
        return {
            "value": v,
            "units": u,
            "page": int(p) if p is not None else 0,
            "evidence": e[:200],
            "confidence": float(c),
            "reason": r[:120],
        }
    except Exception:
        return None

def _write_extracted(art_dir: str, fields: Dict[str, Any]) -> str:
    out_path = os.path.join(art_dir, EXTRACTED_FILENAME)
    write_json(out_path, {"schema_version": 1, "fields": fields})
    return out_path

async def adjudicate_candidates_async(art_dir: str) -> Dict[str, str]:
    cand_path = os.path.join(art_dir, CANDIDATES_FILENAME)
    cands = _read_jsonl(cand_path)
    if not cands:
        # Write an empty extracted to keep contract stable
        return {"extracted": _write_extracted(art_dir, {})}

    grouped = _group_candidates_for_prompt(cands)
    user = ADJ_USER_TMPL.format(grouped=grouped)
    try:
        parsed = await llm_json_typed(ADJ_SYS, user, AdjudicationResponse)
        raw_fields = parsed.fields or {}
    except Exception:
        # Fallback to untyped if schema validation fails
        try:
            resp = await llm_json(ADJ_SYS, user)
            raw_fields = resp.get("fields") or {}
        except Exception:
            return {"extracted": _write_extracted(art_dir, {})}
    final_fields: Dict[str, Any] = {}
    for fname, entry in raw_fields.items():
        coerced = _coerce_field(fname, entry)
        if coerced is not None:
            final_fields[fname] = coerced

    return {"extracted": _write_extracted(art_dir, final_fields)}
