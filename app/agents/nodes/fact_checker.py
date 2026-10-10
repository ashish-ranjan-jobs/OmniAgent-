"""Fact-checker node: per-claim verdicts against evidence ONLY.

Fail-closed design: if the model's verdict JSON cannot be parsed, the whole
draft is treated as one unsupported claim -> triggers reflection (never a
silent pass). Budget caps reflections; finalize then flags the claims.
"""

from __future__ import annotations

from app.agents.feedback import build_reflection_feedback
from app.agents.prompts import FACT_CHECKER_SYSTEM
from app.agents.state import OmniAgentState
from app.core.logging import get_logger
from app.llm.client import LLMClient, parse_json

log = get_logger(__name__)


def _normalize_ids(raw: object) -> list[int]:
    ids: list[int] = []
    for value in raw or []:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids[:5]


async def fact_checker_node(state: OmniAgentState, llm: LLMClient) -> dict:
    if state.get("refused"):
        return {"verdicts": [], "unsupported_claims": [], "feedback": {}}

    evidence = state.get("evidence") or []
    draft = state.get("draft") or ""
    blocks = "\n".join(
        f"[{i}] {ev['content'][:400]}" for i, ev in enumerate(evidence, 1)
    )
    user = (
        f"Draft answer:\n{draft}\n\nEvidence:\n{blocks}\n\n"
        "Extract each checkable claim and give a strict verdict."
    )

    trace_acc: list = []
    result = await llm.complete(
        "fast",
        FACT_CHECKER_SYSTEM,
        user,
        json_mode=True,
        trace=trace_acc,
        node="fact_checker",
    )

    verdicts: list[dict] = []
    data = parse_json(result.text)
    if isinstance(data, dict) and isinstance(data.get("verdicts"), list):
        for v in data["verdicts"]:
            if isinstance(v, dict) and v.get("claim"):
                verdicts.append(
                    {
                        "claim": str(v["claim"])[:300],
                        "supported": bool(v.get("supported")),
                        "evidence_ids": _normalize_ids(v.get("evidence_ids")),
                        "reason": str(v.get("reason", ""))[:300],
                    }
                )

    if not verdicts:
        verdicts = [
            {
                "claim": draft[:300],
                "supported": False,
                "evidence_ids": [],
                "reason": "fact-checker output could not be parsed (fail closed)",
            }
        ]

    unsupported = [v["claim"] for v in verdicts if not v["supported"]]
    feedback = build_reflection_feedback(verdicts) if unsupported else {}
    log.info(
        "fact_checker_done",
        claims=len(verdicts),
        unsupported=len(unsupported),
        will_reflect=bool(feedback),
    )
    return {
        "verdicts": verdicts,
        "unsupported_claims": unsupported,
        "feedback": feedback,
        "trace": (state.get("trace") or []) + trace_acc,
    }
