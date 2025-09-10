import os, json, time, secrets
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