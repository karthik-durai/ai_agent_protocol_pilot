from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os, json
from storage.paths import list_jobs, artifacts_dir, read_status

router = APIRouter()
TEMPLATES = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def home(req: Request):
    jobs = list_jobs(50)
    return TEMPLATES.TemplateResponse("home.html", {"request": req, "jobs": jobs})

@router.get("/job/{job_id}", response_class=HTMLResponse)
async def job(req: Request, job_id: str):
    return TEMPLATES.TemplateResponse("job.html", {"request": req, "job_id": job_id})

# Panels (HTMX fragments)
@router.get("/job/{job_id}/panel/status", response_class=HTMLResponse)
async def status_panel(req: Request, job_id: str):
    st = read_status(job_id)
    art = artifacts_dir(job_id)
    files = [f for f in ("doc_flags.json","sections.json") if os.path.exists(os.path.join(art, f))]
    return TEMPLATES.TemplateResponse("_status.html", {"request": req, "job_id": job_id, "status": st, "files": files})

@router.get("/job/{job_id}/panel/docflags", response_class=HTMLResponse)
async def docflags_panel(req: Request, job_id: str):
    art = artifacts_dir(job_id)
    path = os.path.join(art, "doc_flags.json")
    data = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    return TEMPLATES.TemplateResponse("_docflags.html", {"request": req, "data": data})

@router.get("/job/{job_id}/panel/sections", response_class=HTMLResponse)
async def sections_panel(req: Request, job_id: str):
    art = artifacts_dir(job_id)
    path = os.path.join(art, "sections.json")
    data = None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    return TEMPLATES.TemplateResponse("_sections.html", {"request": req, "data": data})