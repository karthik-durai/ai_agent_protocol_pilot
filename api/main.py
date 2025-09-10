import os, json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from storage.paths import ensure_dirs, new_job_id, write_json
from agent.triage import pdf_pages_text, imaging_verdict, triage_pages

app = FastAPI(title="Protocol Pilot (lite)")

class UploadResp(BaseModel):
    job_id: str
    doc_flags_path: str | None = None
    sections_path: str | None = None

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/upload", response_model=UploadResp)
async def upload(paper: UploadFile = File(...)):
    if paper.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "Please upload a PDF")
    job_id = new_job_id()
    up_dir, art_dir = ensure_dirs(job_id)
    pdf_path = os.path.join(up_dir, "paper.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await paper.read())

    # 1) Pull page texts
    pages = pdf_pages_text(pdf_path)

    # 2) Imaging verdict
    verdict = await imaging_verdict(pages)
    doc_flags = {
        "is_imaging": bool(verdict.get("is_imaging")),
        "modalities": verdict.get("modalities") or [],
        "confidence": float(verdict.get("confidence", 0) or 0),
        "reasons": verdict.get("reasons") or [],
        "counter_signals": verdict.get("counter_signals") or []
    }
    write_json(os.path.join(art_dir, "doc_flags.json"), doc_flags)

    sections_path = None
    # 3) If imaging, do page triage
    if doc_flags["is_imaging"]:
        sections = await triage_pages(pages, top_k=6)
        write_json(os.path.join(art_dir, "sections.json"), sections)
        sections_path = f"/results/{job_id}/sections.json"

    return UploadResp(
        job_id=job_id,
        doc_flags_path=f"/results/{job_id}/doc_flags.json",
        sections_path=sections_path
    )

@app.get("/results/{job_id}/{name}")
def get_artifact(job_id: str, name: str):
    art = os.path.join(os.getenv("DATA_ROOT","./data"), "artifacts", job_id, name)
    if not os.path.exists(art): raise HTTPException(404, "not found")
    with open(art, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)