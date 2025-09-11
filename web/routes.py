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

@router.get("/queue", response_class=HTMLResponse)
async def queue(req: Request):
    jobs = list_jobs(200)
    inflight = [j for j in jobs if j.get("state") in ("queued", "processing", "unknown")]
    # attach title if available
    for j in inflight:
        meta_path = os.path.join(artifacts_dir(j["job_id"]), "meta.json")
        j["title"] = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                j["title"] = (json.load(f) or {}).get("title")
    return TEMPLATES.TemplateResponse("queue.html", {"request": req, "jobs": inflight})

@router.get("/protocols", response_class=HTMLResponse)
async def protocols(req: Request):
    jobs = list_jobs(200)
    have = []
    for j in jobs:
        art = artifacts_dir(j["job_id"])
        if os.path.exists(os.path.join(art, "sections.json")):
            meta_path = os.path.join(art, "meta.json")
            title = None
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    title = (json.load(f) or {}).get("title")
            have.append({**j, "title": title})
    return TEMPLATES.TemplateResponse("protocols.html", {"request": req, "jobs": have})

@router.get("/job/{job_id}", response_class=HTMLResponse)
async def job(req: Request, job_id: str):
    art = artifacts_dir(job_id)
    meta = {}
    mp = os.path.join(art, "meta.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    return TEMPLATES.TemplateResponse("job.html", {"request": req, "job_id": job_id, "meta": meta})

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


# Protocol Card panel (read-only)
@router.get("/job/{job_id}/panel/protocol-card", response_class=HTMLResponse)
async def protocol_card_panel(req: Request, job_id: str):
    """
    Read-only Protocol Card panel.
    Renders winners from artifacts/<job>/imaging_extracted.json via Jinja template.
    Falls back to an empty state if extraction hasn't completed.
    """
    art = artifacts_dir(job_id)
    path = os.path.join(art, "imaging_extracted.json")

    fields = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            fields = data.get("fields") or {}
    except Exception:
        fields = {}

    order = [
        "sequence_type",
        "TR_ms",
        "TE_ms",
        "flip_deg",
        "field_strength_T",
        "inplane_res_mm",
        "slice_thickness_mm",
    ]
    labels = {
        "sequence_type": "Sequence type",
        "TR_ms": "TR (ms)",
        "TE_ms": "TE (ms)",
        "flip_deg": "Flip (°)",
        "field_strength_T": "Field strength (T)",
        "inplane_res_mm": "In‑plane res (mm)",
        "slice_thickness_mm": "Slice thickness (mm)",
    }

    return TEMPLATES.TemplateResponse(
        "_protocol_card.html",
        {
            "request": req,
            "job_id": job_id,
            "fields": fields,
            "order": order,
            "labels": labels
        }
    )

# Gap Report panel (read-only)
@router.get("/job/{job_id}/panel/gap-report", response_class=HTMLResponse)
async def gap_report_panel(req: Request, job_id: str):
    """
    Read-only Gap Report panel.
    Renders artifacts/<job>/gap_report.json via Jinja template.
    Shows a waiting state if the report is not present yet.
    """
    art = artifacts_dir(job_id)
    path = os.path.join(art, "gap_report.json")

    report = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f) or None
        except Exception:
            report = None

    return TEMPLATES.TemplateResponse(
        "_gap_report.html",
        {
            "request": req,
            "job_id": job_id,
            "report": report,
        }
    )

@router.get("/quarantine", response_class=HTMLResponse)
async def quarantine(req: Request):
    jobs = list_jobs(200)
    q = []
    for j in jobs:
        jid = j["job_id"]; art = artifacts_dir(jid)
        # error jobs
        if j.get("state") == "error":
            pass_reason = "pipeline error"
            title = None
            meta_path = os.path.join(art, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    title = (json.load(f) or {}).get("title")
            q.append({"job_id": jid, "state": j["state"], "title": title, "reason": pass_reason})
            continue
        # non-imaging verdict
        flags = os.path.join(art, "doc_flags.json")
        if os.path.exists(flags):
            try:
                with open(flags, "r", encoding="utf-8") as f:
                    df = json.load(f) or {}
                if df.get("is_imaging") is False:
                    title = None
                    meta_path = os.path.join(art, "meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            title = (json.load(mf) or {}).get("title")
                    q.append({"job_id": jid, "state": j["state"], "title": title, "reason": "non-imaging"})
            except Exception:
                pass
    return TEMPLATES.TemplateResponse("quarantine.html", {"request": req, "jobs": q})