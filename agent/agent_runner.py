# agent/agent_runner.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict
import json

from agent.utils import summarize_gaps_from_dir

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from storage.paths import write_status
from agent.config import get_llm_config
from agent.tools import TOOLS


def _chat() -> ChatOllama:
    """Construct the chat client using centralized config, with optional env override for temperature."""
    cfg = get_llm_config()
    temp = float(os.getenv("AGENT_TEMPERATURE", str(cfg.temperature)))
    return ChatOllama(base_url=cfg.base_url, model=cfg.model, temperature=temp)


def _max_steps() -> int:
    try:
        return max(1, int(os.getenv("MAX_AGENT_STEPS", "7")))
    except Exception:
        return 7


SYSTEM = """
You are a cautious, goal‑directed tool‑using agent that produces a high‑confidence MRI protocol from a paper’s text.
Act only by calling tools. Do not narrate. Always include job_dir in tool arguments.
Choose tool sequence autonomously; minimize redundant calls; stop if you conclude the paper is not MRI.

Tools:
- imaging_verdict(job_dir): assess whether the paper reports MRI acquisition; writes doc_flags.json; if is_imaging field in doc_flags.json is false, you must STOP.
- infer_title(job_dir): infer the paper title; writes meta.json.
- triage_pages(job_dir, top_k): pick likely acquisition/method pages; writes sections.json. Prefer a small top_k first for speed; if missing fields persist in gap_report after extraction(extract_and_build_gaps), you may call it again with a slightly larger top_k.
- extract_and_build_gaps(job_dir): extract → adjudicate → build imaging_extracted.json and gap_report.json.

Output:
- When acting, respond with a single TOOL CALL only.
- Always pass job_dir in arguments.
"""

HUMAN = """
Job dir: {job_dir}

Goal: produce winners for core MRI fields and a gap report.
Use any tools as needed; be compute‑conscious.
Respond with tool calls only when acting; always include job_dir in arguments.
"""


def build_agent() -> AgentExecutor:
    """Build the tool-calling agent with our prompt and tools."""
    llm = _chat()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", HUMAN),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=_max_steps(),
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


def _finalize(
    job_dir: str,
    *,
    stop_reason: str,
    steps_used: int,
    max_steps: int,
    last_action: str | None,
    gaps_after: Dict[str, int],
    agent_output: str | None = None,
) -> None:
    """Record the final status for this job in storage."""
    job_id = Path(job_dir).name
    write_status(
        job_id,
        state="done",
        step="completed",
        stop_reason=stop_reason,
        steps_used=steps_used,
        max_steps=max_steps,
        last_action=last_action,
        gaps_after=gaps_after,
        agent_output=(agent_output or "")[:2000],
    )


async def agent_run(job_dir: str) -> Dict[str, Any]:
    """Run the agent loop for a given job directory and persist progress."""
    job_id = Path(job_dir).name
    write_status(job_id, state="running", step="agent.start")
    try:
        ex = build_agent()
        result = await ex.ainvoke({
            "job_dir": job_dir,
            "input": ""
        })
        # Derive why the agent stopped
        gaps_after = summarize_gaps_from_dir(job_dir)
        max_steps = _max_steps()
        steps_used = 0
        last_action = None
        agent_output = None
        # Determine if the imaging verdict classified this as non-imaging
        non_imaging = False
        try:
            flags_path = Path(job_dir) / "doc_flags.json"
            if flags_path.exists():
                with flags_path.open("r", encoding="utf-8") as f:
                    flags = json.load(f) or {}
                non_imaging = bool(flags.get("is_imaging") is False)
        except Exception:
            non_imaging = False
        try:
            steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
            steps_used = len(steps)
            if steps:
                # each item is a tuple (AgentAction, observation); try to read tool name
                action = steps[-1][0]
                last_action = getattr(action, "tool", None)
        except Exception:
            pass
        try:
            agent_output = result.get("output") if isinstance(result, dict) else None
        except Exception:
            agent_output = None

        missing = gaps_after.get("missing", 0)
        if non_imaging:
            reason = "non_imaging"
        elif missing == 0:
            reason = "gaps_resolved"
        elif steps_used >= max_steps:
            reason = "budget_exhausted"
        else:
            reason = "model_chose_stop"

        _finalize(job_dir, stop_reason=reason, steps_used=steps_used, max_steps=max_steps, last_action=last_action, gaps_after=gaps_after, agent_output=agent_output)
        return {"ok": True, "result": result}
    except Exception as e:
        write_status(job_id, state="agent error", error=str(e), stop_reason="exception")
        raise
