import fitz  # PyMuPDF
from typing import List, Dict, Any
from .llm_client import llm_json

IMAGING_VERDICT_SYS = "You are an expert scientific classifier. Output STRICT JSON only."
IMAGING_VERDICT_USER_TMPL = """Decide if this paper is about medical imaging methods.

TITLE:
{title}

ABSTRACT/INTRO (if any):
{abstract}

Return exactly:
{{
  "is_imaging": true|false,
  "modalities": ["CT","MRI","PET","SPECT","Ultrasound","X-ray","CBCT","PET/CT","PET/MRI"],
  "confidence": 0.0-1.0,
  "reasons": ["..."],
  "counter_signals": ["..."]
}}

Rules:
- Look for acquisition/parameter cues: kVp, mAs, slice thickness, kernel, voxel size, TR/TE, b-values, SUV, transducer, field strength, etc.
- Prefer methods over casual mentions. STRICT JSON only.
"""

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
    title = (pages[0]["text"].splitlines()[0] if pages else "")[:300]
    abstract = "\n".join(pages[0]["text"].splitlines()[1:8])[:1200] if pages else ""
    user = IMAGING_VERDICT_USER_TMPL.format(title=title, abstract=abstract)
    return await llm_json(IMAGING_VERDICT_SYS, user)

async def triage_pages(pages: List[Dict[str,Any]], top_k: int = 6) -> Dict[str,Any]:
    # classify each page and keep top_k by score where labels/modalities indicate methods/acquisition
    results = []
    for p in pages:
        if not p["text"]: 
            continue
        user = PAGE_CLASS_USER_TMPL.format(page_text=p["text"])
        resp = await llm_json(PAGE_CLASS_SYS, user)
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