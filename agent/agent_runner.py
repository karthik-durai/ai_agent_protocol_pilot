# agent/agent_runner.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

import json

def _read_json(p: Path, default: Any) -> Any:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _summarize_gaps(job_dir: str) -> Dict[str, int]:
    gap = _read_json(Path(job_dir) / "gap_report.json", {})
    if not isinstance(gap, dict):
        return {"missing": 0, "ambiguous": 0, "conflicts": 0}
    return {
        "missing": len(gap.get("missing", []) or []),
        "ambiguous": len(gap.get("ambiguous", []) or []),
        "conflicts": len(gap.get("conflicts", []) or []),
    }

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from storage.paths import write_status
from agent.config import get_llm_config
from agent.tools import TOOLS


def _chat() -> ChatOllama:
    cfg = get_llm_config()
    temp = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
    return ChatOllama(
        base_url=(cfg.base_url if hasattr(cfg, "base_url") else cfg.chat_base),
        model=cfg.model,
        temperature=temp,
    )


def _max_steps() -> int:
    try:
        return max(1, int(os.getenv("MAX_AGENT_STEPS", "3")))
    except Exception:
        return 3


SYSTEM = """
You are a cautious agent tasked with producing a high-confidence, reproducible MRI protocol.
Act ONLY by calling tools. Your FIRST tool call must be `extract_and_build_gaps(job_dir)`.

DECISION RULES (MANDATORY):
• After `extract_and_build_gaps`, if (missing + conflicts) > 0 AND steps remain,
  you MUST immediately call `extract_with_window(job_dir, span)` (span in 0..4; prefer 2).
• If your last `extract_with_window` returned `improved=false` AND steps remain,
  you MUST try again with a larger span: span' = min(4, previous_span + 1).
• Stop ONLY when (missing + conflicts) == 0 OR you have zero steps remaining.

OUTPUT RULES:
• When a tool call is required by the rules, RESPOND WITH A TOOL CALL ONLY — do NOT write any narrative text.
• Do NOT claim you are out of steps; the runtime manages step budget.
Prefer the smallest action with the highest expected gain.
"""

HUMAN = """
Job dir: {job_dir}
Steps remaining: {steps_remaining}

Instructions:
1) Call `extract_and_build_gaps(job_dir)` first.
2) If the tool output includes numeric `missing` and `conflicts` whose sum is > 0,
   call `extract_with_window(job_dir, span)` next (explicitly choose span in 0..4; prefer 2).
3) If the previous `extract_with_window` reported `improved=false` and you still have steps remaining,
   call `extract_with_window` again with a larger span (span' = min(4, previous_span + 1)).
4) Always include `job_dir` in tool arguments. Do not narrate when a tool call is required.
5) Stop only when the decision rules say to stop.

Examples:
- Second step: extract_with_window(job_dir="{job_dir}", span=2)
- Third step (if no improvement): extract_with_window(job_dir="{job_dir}", span=3)
"""


def build_agent() -> AgentExecutor:
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


def _finalize(job_dir: str, *, stop_reason: str, steps_used: int, max_steps: int, last_action: str | None, gaps_after: Dict[str, int], agent_output: str | None = None) -> None:
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
    job_id = Path(job_dir).name
    write_status(job_id, state="running", step="agent.start")
    try:
        ex = build_agent()
        result = await ex.ainvoke({
            "job_dir": job_dir,
            "steps_remaining": _max_steps(),
            "input": ""
        })
        # Derive why the agent stopped
        gaps_after = _summarize_gaps(job_dir)
        max_steps = _max_steps()
        steps_used = 0
        last_action = None
        agent_output = None
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
        conflicts = gaps_after.get("conflicts", 0)
        if (missing + conflicts) == 0:
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
