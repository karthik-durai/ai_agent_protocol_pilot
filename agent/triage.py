import fitz  # PyMuPDF
from typing import List, Dict, Any
from .llm_client import llm_json

IMAGING_VERDICT_SYS = "You are an expert scientific classifier for detecting whether a scientific paper reports medical imaging methods. Output STRICT JSON only. Prioritize recall: if any acquisition parameters, modality indicators, or sequence/reconstruction jargon are present, classify as imaging with appropriate confidence. Only classify as non-imaging when none of these cues are present in the provided text."
IMAGING_VERDICT_USER_TMPL = """Decide whether this paper is about medical imaging methods or clearly includes imaging acquisition details that support reproducibility.

EARLY PAGES TEXT (title + first pages, truncated):
\"\"\"
{early_text}
\"\"\"

CUES TO LOOK FOR (non-exhaustive):
- CT: kVp, mAs, tube current, collimation, pitch, slice thickness, reconstruction kernel/filter (e.g., B30f, FCxx), FBP/IR/MBIR (ASiR/Veo), FOV, voxel size, matrix, convolution kernel
- MRI: TR, TE, TI, flip angle, field strength (e.g., 1.5T/3T/7T), coil (e.g., 8‑channel), EPI/echo‑planar, spin‑echo/gradient‑echo, T1‑weighted/T2‑weighted/FLAIR, DWI with b‑values, DCE
- PET/SPECT: SUV, MBq, acquisition time/bed, OSEM iterations/subsets, TOF, PSF, attenuation correction (CT‑based), sinogram, list‑mode
- Ultrasound: transducer MHz, probe type, focal depth, Doppler, B‑mode, cine
- X‑ray/Radiography/CBCT: kVp, mAs, SID, grid, detector, cone‑beam
- General: voxel size, registration, normalization, resampling, reconstruction algorithm, kernel

Return exactly:
{{
  "is_imaging": true|false,
  "modalities": ["CT","MRI","PET","SPECT","Ultrasound","X-ray","CBCT","PET/CT","PET/MRI"],
  "confidence": 0.0-1.0,
  "reasons": ["short concrete cues you used (≤3)"],
  "counter_signals": ["why not imaging, if applicable (≤2)"]
}}

Rules:
- If any unambiguous method cue appears, set "is_imaging": true (even if the modality is uncertain—leave modalities empty or inferred from jargon).
- Ignore mentions that appear only in references or affiliations (e.g., "Department of Radiology") without method cues.
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
PAGE_CLASS_USER_TMPL = """From this page TEXT, decide if it contains imaging acquisition/method details and list modalities.

TEXT:
\"\"\"
{page_text}
\"\"\"

Return exactly:
{{
  "labels": ["methods","acquisition","preprocessing","table","other"],
  "modalities": ["..."],
  "score": 0.0-1.0,
  "evidence": ["short substrings from the page (<=3)"]
}}

Rules:
- Only include modalities supported by text (e.g., TR/TE->MRI; kVp/mAs->CT; SUV->PET; transducer->Ultrasound).
- STRICT JSON only.
"""

def pdf_pages_text(pdf_path: str, max_pages: int = 40, max_chars: int = 1200) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    out = []
    for i in range(min(max_pages, doc.page_count)):
        t = (doc.load_page(i).get_text("text") or "").strip()
        out.append({"page": i, "text": t[:max_chars]})
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
        return await llm_json(IMAGING_VERDICT_SYS, user)
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
        mods = resp.get("modalities") or []
        score = float(resp.get("score", 0) or 0)
        if (("methods" in labels) or ("acquisition" in labels) or mods) and score > 0:
            results.append({
                "page": p["page"],
                "score": score,
                "labels": list(labels),
                "modalities": mods,
                "snippets": (resp.get("evidence") or [])[:3]
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"candidates": results[:top_k]}