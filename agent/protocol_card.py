# agent/protocol_card.py
from __future__ import annotations
import os
from typing import Any, Dict, List
from storage.paths import write_json
import json
from agent.llm_client import llm_json

CANDIDATES_FILENAME = "imaging_candidates.jsonl"
EXTRACTED_FILENAME = "imaging_extracted.json"

# --- M3a Step 2: strict-JSON extraction prompts (CT + common) ---
EXTRACT_CT_SYS = (
    "You are a precise information extraction model for CT imaging methods. "
    "Extract only when the text gives explicit evidence. Anchor every value "
    "to an exact substring. Normalize units to mm/kVp/mAs where applicable. "
    "Output STRICT JSON only as specified. Do not include commentary."
)

EXTRACT_CT_USER_TMPL = """TEXT WINDOW (may include tables/symbols/units):
\"\"\"
{window_text}
\"\"\"

Center page index (zero-based): {center_page}

FIELDS TO EXTRACT (emit ONLY when supported by explicit text):
- slice_thickness_mm (mm, number) — e.g., "slice thickness 1.25 mm", "0.5 cm" → 5.0 mm
- kernel (string) — e.g., "B30f", "FC13", "Bone", "Lung", "Standard"
- kernel_family (string) — one of: soft_tissue | bone | lung | standard | detail | smooth | unknown
- kVp (int) — e.g., "120 kVp"
- mAs (number) — e.g., "150 mAs", "ref. mAs 200"
- voxel_size_mm ([x,y,z] floats, mm) — e.g., "1×1×3 mm³", "1 mm isotropic" → [1,1,1]
- matrix ([w,h] ints) — e.g., "512×512", "matrix 256x256"
- fov_mm (mm, number) — e.g., "FOV 25 cm" → 250 mm

Return EXACTLY this JSON:
{{
  "candidates": [
    {{
      "field": "<one of the fields above>",
      "page": <int page index to attribute (use the center page index unless evidence clearly refers to another)>,
      "raw_span": "<short exact substring from the text>",
      "value": <number|string|[number,number,number]|[int,int]>,
      "units": "<'mm'|'kVp'|'mAs'|''>",
      "evidence": "<short exact substring (<=100 chars)>",
      "confidence": <0.0-1.0>,
      "notes": "<optional, short>"
    }}
  ]
}}

Rules:
- Emit entries only with direct textual support; DO NOT guess.
- Normalize cm→mm (×10), μm→mm (÷1000) when applicable.
- voxel_size_mm: accept "1x1x1 mm", "1×1×3 mm³", "1 mm isotropic" → [1,1,1].
- matrix: parse "512x512" or "512 × 512" → [512,512].
- kernel_family: infer from kernel when obvious (e.g., B30f→soft_tissue; B70→bone; “Lung”→lung); else "unknown".
- If uncertain, set confidence ≤ 0.6.
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

async def extract_ct_common(window_text: str, center_page: int) -> List[Dict[str, Any]]:
    """
    Call the LLM with strict JSON prompt and return a list of candidate dicts.
    Never raises on JSON issues—returns [] instead.
    """
    if not window_text.strip():
        return []
    user = EXTRACT_CT_USER_TMPL.format(window_text=window_text, center_page=center_page)
    try:
        resp = await llm_json(EXTRACT_CT_SYS, user)
    except Exception:
        return []
    items = resp.get("candidates") or []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            field = it.get("field")
            page = int(it.get("page", center_page))
            raw_span = (it.get("raw_span") or "").strip()
            value = it.get("value")
            units = (it.get("units") or "").strip()
            evidence = (it.get("evidence") or "").strip()
            conf = float(it.get("confidence", 0) or 0)
            notes = it.get("notes")
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
                hits = await extract_ct_common(window, center)
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
    "You are a careful adjudicator for CT imaging parameters. From multiple candidate "
    "extractions with evidence, choose the best value per field and explain briefly. "
    "Output STRICT JSON only. Do not invent values; omit a field if evidence is insufficient."
)

ADJ_USER_TMPL = """CANDIDATES grouped by field. Each item: {{value, units, page, evidence, confidence}}.

{grouped}

Return EXACTLY this JSON:
{{
  "fields": {{
    "slice_thickness_mm": {{"value": <number>, "units": "mm", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "kernel": {{"value": "<str>", "units": "", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "kernel_family": {{"value": "<soft_tissue|bone|lung|standard|detail|smooth|unknown>", "units": "", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "kVp": {{"value": <int>, "units": "kVp", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "mAs": {{"value": <number>, "units": "mAs", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "voxel_size_mm": {{"value": [<number>,<number>,<number>], "units": "mm", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "matrix": {{"value": [<int>,<int>], "units": "", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}},
    "fov_mm": {{"value": <number>, "units": "mm", "page": <int>, "evidence": "<str>", "confidence": <0.0-1.0>, "reason": "<short>"}}
  }}
}}

Rules:
- Choose at most one best entry per field; omit a field if no solid evidence.
- Prefer explicit labels (e.g., "slice thickness = …", "kernel: …").
- Prefer higher confidence and clearer evidence; break ties with table-like phrasing or consistency with other fields.
- Keep provided units; if you must convert, only cm→mm (×10).
- STRICT JSON only.
"""

_FIELDS = ["slice_thickness_mm","kernel","kernel_family","kVp","mAs","voxel_size_mm","matrix","fov_mm"]

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

def _coerce_field(fname: str, entry: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    v = entry.get("value")
    u = entry.get("units") or ""
    p = entry.get("page")
    e = (entry.get("evidence") or "").strip()
    c = entry.get("confidence", 0)
    r = (entry.get("reason") or "").strip()
    try:
        if fname in ("slice_thickness_mm","fov_mm","mAs"):
            v = float(v)
        elif fname == "kVp":
            v = int(v)
        elif fname == "voxel_size_mm":
            if isinstance(v, list) and len(v) == 3:
                v = [float(v[0]), float(v[1]), float(v[2])]
            else:
                return None
            u = "mm"
        elif fname == "matrix":
            if isinstance(v, list) and len(v) == 2:
                v = [int(v[0]), int(v[1])]
            else:
                return None
            u = ""
        elif fname in ("kernel","kernel_family"):
            v = str(v)
        else:
            return None
        return {"value": v, "units": u, "page": int(p), "evidence": e[:200], "confidence": float(c), "reason": r[:120]}
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
        resp = await llm_json(ADJ_SYS, user)
    except Exception:
        # Fallback: no winners
        return {"extracted": _write_extracted(art_dir, {})}

    raw_fields = resp.get("fields") or {}
    final_fields: Dict[str, Any] = {}
    for fname, entry in raw_fields.items():
        coerced = _coerce_field(fname, entry)
        if coerced is not None:
            final_fields[fname] = coerced

    return {"extracted": _write_extracted(art_dir, final_fields)}