import fitz  # PyMuPDF
import os, hashlib
from typing import List, Dict, Any
from .llm_client import llm_json
from storage.paths import write_json

# ---- MRI-only canonical modalities + normalizer ----
CANONICAL_MODALITIES = ["MRI"]

def normalize_modalities(items):
    """Return ['MRI'] if any entry clearly refers to MRI; else []."""
    if not items:
        return []
    out = []
    for m in items:
        s = (m or "").strip().lower()
        if s in {"mri", "mr"} and "MRI" not in out:
            out.append("MRI")
    return out

IMAGING_VERDICT_SYS = (
    "You are an expert scientific classifier for detecting whether a paper reports MRI acquisition methods. "
    "Output STRICT JSON only. Prioritize precision: classify as imaging ONLY when unambiguous MRI method cues "
    "(parameters or sequence/jargon) appear in the provided text. If cues are weak/ambiguous, classify as non-imaging. "
    "Standardize modality names to: MRI."
)

IMAGING_VERDICT_USER_TMPL = """Decide whether this paper is about MRI methods or clearly includes MRI acquisition details that support reproducibility.

EARLY PAGES TEXT (title + first pages, truncated):
\"\"\"
{early_text}
\"\"\"

MRI CUES (non-exhaustive):
- TR, TE, TI, flip angle
- Field strength (1.5T, 3T, 7T), coil (e.g., 8‑channel head)
- Sequence names: T1‑weighted/T2‑weighted, FLAIR, DWI, EPI, GRE/SE, MPRAGE/MP‑RAGE, TSE/RARE
- Spatial params: in‑plane resolution (e.g., 0.8×0.8 mm), slice thickness (mm)
- Other: acceleration (GRAPPA/SENSE, multiband), echo train length, bandwidth

Return exactly:
{{
  "is_imaging": true|false,
  "modalities": ["MRI"] or [],
  "confidence": 0.0-1.0,
  "reasons": ["short concrete cues you used (≤3)"],
  "counter_signals": ["why not MRI, if applicable (≤2)"]
}}

Rules:
- Require at least ONE hard MRI cue (e.g., TR/TE/flip, 3T, FLAIR/EPI, in‑plane mm) to set "is_imaging": true; otherwise set false.
- Ignore affiliations/references (e.g., “Department of Radiology”) without method cues.
- If uncertain, set confidence ≤ 0.6; when explicit parameters are present, use ≥ 0.7.
- STRICT JSON only."""

TITLE_SYS = "You are a careful bibliographic assistant for scientific PDFs. Extract ONLY the main article title from noisy front-matter. Output STRICT JSON. Do not output section headings, running heads, short titles, or journal/affiliation/footer text. De-hyphenate words split across line breaks, collapse whitespace, and keep the full title (including any subtitle after a colon) as it appears."
TITLE_USER_TMPL = """From the early pages of a scientific PDF, extract the MAIN ARTICLE TITLE.

TEXT (truncated, may include headers/footers/authors/sections):
\"\"\"
{early_text}
\"\"\"

POSITIVE CUES (helpful if present):
- A prominent heading above author names and before Abstract/Introduction
- Title-cased sentence (may span multiple lines), often centered
- May include a subtitle after a colon

NEGATIVE CUES (must NOT be returned as title):
- Section labels: Abstract, Introduction, Methods/Materials and Methods, Results, Discussion, Conclusion(s), Acknowledgments, References, Supplementary, Appendix
- Meta/front-matter: Keywords, Highlights, Graphical abstract, Running head, Short title, Correspondence, Received/Accepted dates, DOI/URL, ORCID
- Journal/Publisher strings: ©, Elsevier, Springer, Wiley, arXiv/bioRxiv/medRxiv banners, “Preprint”
- Affiliations/emails/addresses/departments; author lists

Return exactly:
{{
  "title": "the exact main title string as it appears (linebreaks removed; de-hyphenated)",
  "confidence": 0.0-1.0,
  "reasons": ["why this is the title (≤2, short)"]
}}

Rules:
- Keep the complete title including subtitle after a colon if present.
- Remove surrounding author/affiliation lines, section labels, and headers/footers.
- If multiple plausible candidates exist, choose the one immediately preceding the Abstract/Introduction and with length 6–25 words (40–200 chars) when possible.
- Penalize ALL-CAPS short lines (likely running heads) and strings containing journal/publisher terms.
- If uncertain, set confidence ≤ 0.6.
- STRICT JSON only."""

PAGE_CLASS_SYS = "You are a precise scientific text classifier. Output STRICT JSON only."
PAGE_CLASS_USER_TMPL = """From this page TEXT, decide if it contains MRI acquisition/method details and list modalities.

TEXT:
\"\"\"
{page_text}
\"\"\"

Return exactly:
{{
  "labels": ["methods","acquisition","preprocessing","table","other"],
  "modalities": ["MRI"] or [],
  "score": 0.0-1.0,
  "evidence": ["short substrings from the page (<=3)"]
}}

Rules:
- Only include "MRI" when explicit method cues appear (e.g., TR/TE -> MRI; 1.5T/3T/7T; FLAIR/EPI; in‑plane mm; coil).
- If the page lacks hard cues, set labels ["other"], modalities [], score 0.0.
- STRICT JSON only.
"""

# Stricter acceptance threshold for page-level classifier to reduce false positives
PAGE_SCORE_MIN = 0.65

def pdf_pages_text(
    pdf_path: str,
    *,
    to_json_path: str | None = None
) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    out = []
    total = doc.page_count
    limit = total
    for i in range(limit):
        t = (doc.load_page(i).get_text("text") or "").strip()
        out.append({"page": i, "text": t})

    if to_json_path:
        # build payload and write pages.json side-effect
        pages_payload = {
            "version": 1,
            "pdf_meta": {"filename": os.path.basename(pdf_path), "page_count": total},
            "pages": [
                {
                    "page": p["page"],
                    "chars": len(p["text"]),
                    "sha1": hashlib.sha1(p["text"].encode("utf-8", "ignore")).hexdigest(),
                    "text": p["text"],
                }
                for p in out
            ],
        }
        # call your existing writer (prefer atomic replace inside it)
        write_json(to_json_path, pages_payload)

    return out

async def imaging_verdict(pages: List[Dict[str,Any]]) -> Dict[str,Any]:
    # Use the first 2–3 pages for better recall
    early_blocks = []
    for p in pages[:3]:
        t = (p.get("text") or "").strip()
        if t:
            early_blocks.append(t)
    early_text = ("\n\n".join(early_blocks))[:6000] if early_blocks else ""
    user = IMAGING_VERDICT_USER_TMPL.format(early_text=early_text)
    try:
        resp = await llm_json(IMAGING_VERDICT_SYS, user)
        # Normalize and enforce MRI-only decision
        resp["modalities"] = normalize_modalities(resp.get("modalities") or [])
        conf = float(resp.get("confidence") or 0.0)
        if resp.get("is_imaging") and ("MRI" not in resp["modalities"] or conf < 0.7):
            resp["is_imaging"] = False
        return resp
    except Exception as e:
        # Fail-safe: treat as non-imaging with an explanatory reason
        return {
            "is_imaging": False,
            "modalities": [],
            "confidence": 0.0,
            "reasons": [f"llm error: {str(e)[:120]}"],
            "counter_signals": ["fallback verdict"]
        }

async def infer_title(pages: List[Dict[str,Any]], max_pages: int = 2, max_chars: int = 3000) -> Dict[str,Any]:
    """Infer the paper title using an LLM over the first 1–2 pages of text."""
    early: list[str] = []
    for p in pages[:max_pages]:
        t = (p.get("text") or "").strip()
        if t:
            early.append(t)
    early_text = ("\n\n".join(early))[:max_chars] if early else ""
    if not early_text:
        return {"title": "", "confidence": 0.0, "reasons": ["no text available"]}
    user = TITLE_USER_TMPL.format(early_text=early_text)
    try:
        resp = await llm_json(TITLE_SYS, user)
    except Exception:
        return {"title": "", "confidence": 0.0, "reasons": ["llm error"]}
    title = (resp.get("title") or "").strip()
    conf = float(resp.get("confidence", 0) or 0)
    reasons = resp.get("reasons") or []
    # Basic cleanup: collapse whitespace, guard length
    title = " ".join(title.split())
    if len(title) > 300:
        title = title[:297] + "..."
    # Light filter: strip leading 'Abstract' if model drifted
    if title.lower().startswith("abstract"):
        title = title[8:].strip(" :-")
    return {"title": title, "confidence": conf, "reasons": reasons}

async def triage_pages(pages: List[Dict[str,Any]], top_k: int = 6) -> Dict[str,Any]:
    # classify each page and keep top_k by score where labels/modalities indicate methods/acquisition
    results = []
    for p in pages:
        if not p["text"]: 
            continue
        user = PAGE_CLASS_USER_TMPL.format(page_text=p["text"])
        try:
            resp = await llm_json(PAGE_CLASS_SYS, user)
        except Exception:
            # Skip this page on LLM failure; continue processing others
            continue
        labels = set(resp.get("labels") or [])
        mods = normalize_modalities(resp.get("modalities") or [])
        score = float(resp.get("score", 0) or 0)
        if (("methods" in labels) or ("acquisition" in labels) or ("MRI" in mods)) and score >= PAGE_SCORE_MIN:
            results.append({
                "page": p["page"],
                "score": score,
                "labels": list(labels),
                "modalities": mods,
                "snippets": (resp.get("evidence") or [])[:3]
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"candidates": results[:top_k]}