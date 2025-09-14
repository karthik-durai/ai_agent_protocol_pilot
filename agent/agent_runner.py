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
You are a cautious agent tasked with producing a high-confidence, reproducible MRI protocol.
Act ONLY by calling tools — do not narrate when a tool call is expected.

Available tools (high-level):
• imaging_verdict(job_dir): decide if the paper reports MRI acquisition; writes doc_flags.json (is_imaging, confidence, reasons). If non‑MRI, STOP.
• infer_title(job_dir): infer the paper title; writes meta.json. Prefer to run this AFTER imaging_verdict confirms MRI, to help users identify the paper in the UI.
• triage_pages(job_dir): classify pages to find likely acquisition/methods; writes sections.json.
• extract_and_build_gaps(job_dir): extract → adjudicate → build missing‑only gap report; writes imaging_extracted.json and gap_report.json.

Decision policy:
• Goal: produce winners (protocol card) and a gap report with minimal calls.
• First, call imaging_verdict(job_dir); STOP immediately if it indicates NON‑IMAGING.
• After confirming MRI, if the title is missing or empty, call infer_title(job_dir) to populate the UI.
• Before extraction, ensure sections.json exists (call triage_pages if needed).
• When ready, run extract_and_build_gaps to generate winners + gap report.

Output rules:
• Respond with TOOL CALLS ONLY when taking an action.
• Always include job_dir in tool arguments.
"""

HUMAN = """
Job dir: {job_dir}

Context:
- You decide which tools to call and in what order based on the policy above.
- Start with imaging_verdict(job_dir); STOP immediately if NON‑IMAGING.
- After confirming MRI, if the title is missing or empty, call infer_title(job_dir) to populate meta.json.
- Otherwise, ensure sections.json exists (triage_pages), then run extract_and_build_gaps to produce winners and the gap report.

Constraints:
- Respond with tool calls only when acting; always include job_dir in arguments.

Examples (not prescriptive):
- imaging_verdict(job_dir="{job_dir}") → infer_title(job_dir="{job_dir}") → triage_pages(job_dir="{job_dir}") → extract_and_build_gaps(job_dir="{job_dir}")
- imaging_verdict(job_dir="{job_dir}") → triage_pages(job_dir="{job_dir}") → extract_and_build_gaps(job_dir="{job_dir}")
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
