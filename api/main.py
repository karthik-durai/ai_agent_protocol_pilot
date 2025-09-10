import os, json, asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from pydantic import BaseModel
from storage.paths import ensure_dirs, new_job_id, write_json, artifacts_dir, read_status, write_status, list_jobs
from agent.triage import pdf_pages_text, imaging_verdict, triage_pages

app = FastAPI(title="Protocol Pilot (lite)")

class UploadResp(BaseModel):
    job_id: str
    doc_flags_path: str | None = None
    sections_path: str | None = None

async def _process_job_async(job_id: str, pdf_path: str, art_dir: str):
    try:
        write_status(job_id, "processing", started_at=datetime.now(timezone.utc).isoformat())
        pages = pdf_pages_text(pdf_path)
        verdict = await imaging_verdict(pages)
        doc_flags = {
            "is_imaging": bool(verdict.get("is_imaging")),
            "modalities": verdict.get("modalities") or [],
            "confidence": float(verdict.get("confidence", 0) or 0),
            "reasons": verdict.get("reasons") or [],
            "counter_signals": verdict.get("counter_signals") or []
        }
        write_json(os.path.join(art_dir, "doc_flags.json"), doc_flags)

        if doc_flags["is_imaging"]:
            sections = await triage_pages(pages, top_k=6)
            write_json(os.path.join(art_dir, "sections.json"), sections)

        write_status(job_id, "done", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        write_status(job_id, "error", error=str(e), finished_at=datetime.now(timezone.utc).isoformat())
        raise

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/upload", response_model=UploadResp)
async def upload(background_tasks: BackgroundTasks, paper: UploadFile = File(...)):
    if paper.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "Please upload a PDF")
    job_id = new_job_id()
    up_dir, art_dir = ensure_dirs(job_id)
    pdf_path = os.path.join(up_dir, "paper.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await paper.read())

    # Non-blocking: schedule background processing and return immediately
    write_status(job_id, "queued", created_at=datetime.now(timezone.utc).isoformat())
    # Schedule the async worker directly; FastAPI will await it after sending the response
    background_tasks.add_task(_process_job_async, job_id, pdf_path, art_dir)

    doc_flags_path = f"/results/{job_id}/doc_flags.json"
    sections_path = f"/results/{job_id}/sections.json"  # may not exist yet

    return UploadResp(
        job_id=job_id,
        doc_flags_path=doc_flags_path,
        sections_path=sections_path
    )

@app.post("/ui/upload")
async def ui_upload(background_tasks: BackgroundTasks, paper: UploadFile = File(...)):
    if paper.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "Please upload a PDF")
    job_id = new_job_id()
    up_dir, art_dir = ensure_dirs(job_id)
    pdf_path = os.path.join(up_dir, "paper.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await paper.read())

    write_status(job_id, "queued", created_at=datetime.now(timezone.utc).isoformat())
    background_tasks.add_task(_process_job_async, job_id, pdf_path, art_dir)

    # Redirect to the job viewer page (handled by web.routes)
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)

@app.get("/results/{job_id}/{name}")
def get_artifact(job_id: str, name: str):
    art = os.path.join(os.getenv("DATA_ROOT","./data"), "artifacts", job_id, name)
    if not os.path.exists(art): raise HTTPException(404, "not found")
    with open(art, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(data)

from web.routes import router as viewer_router
app.include_router(viewer_router)