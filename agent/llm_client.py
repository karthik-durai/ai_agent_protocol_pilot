import os, json, httpx, re
LLM_BASE = os.getenv("LLM_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")

async def llm_json(system: str, user: str, model: str = MODEL, timeout: int = 60):
    """Call Ollama /api/chat and parse STRICT JSON response (best effort)."""
    payload = {
        "model": model,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "options": {"temperature": 0},
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{LLM_BASE}/api/chat", json=payload)
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
    # strict JSON or extract the first {...} blob
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        raise