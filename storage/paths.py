import os, json, time, secrets, glob
DATA_ROOT = os.getenv("DATA_ROOT", os.path.abspath("./data"))

def ensure_dirs(job_id: str):
    up = os.path.join(DATA_ROOT, "uploads", job_id)
    art = os.path.join(DATA_ROOT, "artifacts", job_id)
    os.makedirs(up, exist_ok=True)
    os.makedirs(art, exist_ok=True)
    return up, art

def new_job_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"job_{ts}_{secrets.token_hex(4)}"

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def artifacts_dir(job_id: str) -> str:
    d = os.path.join(DATA_ROOT, "artifacts", job_id)
    os.makedirs(d, exist_ok=True); return d

def status_path(job_id: str) -> str:
    return os.path.join(artifacts_dir(job_id), "status.json")

def write_status(job_id: str, state: str, **extra):
    """
    Persist job status with richer telemetry.
    - Merges with any existing status.json instead of overwriting unrelated fields.
    - Always updates 'state'.
    - Updates 'step' only if provided in extras.
    - Includes all extra keyword fields (e.g., stop_reason, steps_used, gaps_after, agent_output).
    """
    path = status_path(job_id)
    # Load existing status, if any
    current = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                current = json.load(f) or {}
    except Exception:
        current = {}
    # Merge extras and required fields
    rec = {**current, **extra}
    rec["state"] = state
    if "step" in extra:
        rec["step"] = extra.get("step")
    write_json(path, rec)
    return rec

def read_status(job_id: str):
    p = status_path(job_id)
    if not os.path.exists(p): return {"state": "unknown"}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_jobs(limit: int = 50):
    roots = sorted(
        glob.glob(os.path.join(DATA_ROOT, "artifacts", "*")),
        key=lambda p: os.path.getmtime(p),
        reverse=True
    )[:limit]
    out = []
    for r in roots:
        jid = os.path.basename(r)
        st = read_status(jid)
        out.append({"job_id": jid, "state": st.get("state", "unknown")})
    return out
