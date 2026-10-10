"""Evaluation harness (see docs/EVALUATION.md for methodology).

Modes:
  --mode ablation   Retrieval quality per stage (dense -> +bm25 -> +rerank):
                    recall@5 and MRR against labelled golden documents.
                    No LLM required; deterministic and cheap.
  --mode triad      LLM-as-judge RAG Triad over full graph runs:
                    context relevance, groundedness (1 - hallucination rate),
                    answer relevance; plus refusal correctness on unanswerable
                    questions. Requires the LLM backend.
  --mode reflection Groundedness delta from the cyclical reflection loop: the
                    same golden items run twice, MAX_REFLECTIONS=0 (off) vs the
                    configured budget (on), and the harness reports per-arm
                    triad metrics plus the hallucination-rate reduction
                    attributable to reflection. Requires the LLM backend.
  --limit N         Evaluate only the first N golden items (smoke runs).

Results land in evals/results/ as JSON + Markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.agents.graph import build_graph
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import get_engine
from app.llm.client import LLMClient, LLMUnavailableError, parse_json
from app.retrieval.pipeline import RetrievalPipeline, RetrievalStageFlags

log = get_logger(__name__)

ARMS = [
    ("dense", RetrievalStageFlags(dense=True, bm25=False, rerank=False)),
    ("dense+bm25", RetrievalStageFlags(dense=True, bm25=True, rerank=False)),
    ("dense+bm25+rerank", RetrievalStageFlags(dense=True, bm25=True, rerank=True)),
]


def load_golden(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def recall_at_k(expected: set[str], got: list[str], k: int) -> float:
    if not expected:
        return 0.0
    hits = len(expected & set(got[:k]))
    return hits / len(expected)


def mrr(expected: set[str], got: list[str]) -> float:
    for rank, title in enumerate(got, start=1):
        if title in expected:
            return 1.0 / rank
    return 0.0


async def run_ablation(golden: list[dict]) -> dict:
    retriever = RetrievalPipeline()
    results = {}
    for arm_name, flags in ARMS:
        recalls, mrrs, latencies = [], [], []
        for item in golden:
            if item.get("unanswerable"):
                continue
            t0 = time.perf_counter()
            evidence, _ = await retriever.retrieve(item["query"], flags)
            latencies.append((time.perf_counter() - t0) * 1000)
            titles = [ev.title for ev in evidence]
            expected = set(item["expected_docs"])
            recalls.append(recall_at_k(expected, titles, 5))
            mrrs.append(mrr(expected, titles))
        results[arm_name] = {
            "recall@5": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1)
            if latencies
            else 0.0,
            "n": len(recalls),
        }
        log.info("ablation_arm_done", arm=arm_name, **results[arm_name])
    return results


TRIAD_CONTEXT = (
    "You judge retrieval quality. Given a question and numbered passages, "
    "rate each passage 1 if it helps answer the question else 0. "
    'Respond ONLY with JSON: {"relevance": [1, 0, ...]}'
)
TRIAD_GROUNDED = (
    "You are a strict groundedness judge. Given an answer and numbered "
    "evidence, extract each claim in the answer and decide if the evidence "
    "directly supports it. A claim is supported ONLY if the evidence states "
    'it explicitly. Respond ONLY with JSON: '
    '{"claims": [{"claim": "...", "supported": true}]}'
)
TRIAD_ANSWER = (
    "You judge answer quality. Given a question and an answer, decide if the "
    "answer actually addresses the question (not just topic-adjacent text). "
    'Respond ONLY with JSON: {"relevant": true}'
)


async def _triad_item(llm: LLMClient, graph, item: dict, thread_id: str) -> dict:
    """Run one golden item through the graph and score it.

    Unanswerable items score refusal correctness; answerable items score the
    RAG Triad: context relevance and answer relevance via the judge,
    groundedness from the graph's own fact-checker verdicts.
    """
    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"query": item["query"], "trace": [], "reflection_count": 0},
        {"configurable": {"thread_id": thread_id}},
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    if item.get("unanswerable"):
        return {
            "query": item["query"],
            "correct_refusal": bool(result.get("refused")),
            "latency_ms": elapsed,
        }

    evidence = result.get("evidence") or []
    answer = result.get("answer") or ""
    # Context relevance
    ctx_rel = 0.0
    if evidence:
        blocks = "\n".join(f"[{i}] {ev['content'][:300]}" for i, ev in enumerate(evidence, 1))
        r = await llm.complete(
            "fast",
            TRIAD_CONTEXT,
            f"Question: {item['query']}\nPassages:\n{blocks}",
            json_mode=True,
        )
        data = parse_json(r.text) or {}
        rel = data.get("relevance") if isinstance(data, dict) else None
        if isinstance(rel, list) and rel:
            ctx_rel = sum(1 for x in rel if x) / len(rel)
    # Groundedness
    grounded = 0.0
    verdicts = result.get("verdicts") or []
    if verdicts:
        grounded = sum(1 for v in verdicts if v.get("supported")) / len(verdicts)
    elif result.get("refused"):
        grounded = 1.0
    # Answer relevance
    ans_rel = 0.0
    if answer:
        r = await llm.complete(
            "fast",
            TRIAD_ANSWER,
            f"Question: {item['query']}\nAnswer: {answer[:600]}",
            json_mode=True,
        )
        data = parse_json(r.text) or {}
        ans_rel = 1.0 if isinstance(data, dict) and data.get("relevant") else 0.0
    return {
        "query": item["query"],
        "context_relevance": round(ctx_rel, 3),
        "groundedness": round(grounded, 3),
        "answer_relevance": round(ans_rel, 3),
        "reflections": result.get("reflection_count", 0),
        "refused": bool(result.get("refused")),
        "latency_ms": elapsed,
    }


def aggregate_triad(per_item: list[dict]) -> dict:
    """Aggregate per-item triad scores (pure function: no I/O, testable)."""
    scored = [x for x in per_item if "groundedness" in x]
    refusals = [x for x in per_item if "correct_refusal" in x]
    errors = [x for x in per_item if "error" in x]
    return {
        "context_relevance": round(
            sum(x["context_relevance"] for x in scored) / len(scored), 3
        )
        if scored
        else 0.0,
        "groundedness": round(sum(x["groundedness"] for x in scored) / len(scored), 3)
        if scored
        else 0.0,
        "answer_relevance": round(
            sum(x["answer_relevance"] for x in scored) / len(scored), 3
        )
        if scored
        else 0.0,
        "hallucination_rate": round(
            1 - sum(x["groundedness"] for x in scored) / len(scored), 3
        )
        if scored
        else 0.0,
        "refusal_accuracy": round(
            sum(1 for x in refusals if x["correct_refusal"]) / len(refusals), 3
        )
        if refusals
        else None,
        "n_scored": len(scored),
        "n_unanswerable": len(refusals),
        "n_errors": len(errors),
    }


async def run_triad(golden: list[dict], max_reflections: int | None = None) -> dict:
    from langgraph.checkpoint.memory import MemorySaver

    s = get_settings()
    budget = s.max_reflections if max_reflections is None else max_reflections
    llm = LLMClient()
    graph = build_graph(
        llm, RetrievalPipeline(), checkpointer=MemorySaver(), max_reflections=budget
    )
    per_item = []
    for idx, item in enumerate(golden):
        try:
            entry = await _triad_item(llm, graph, item, f"triad-eval-{idx}")
        except LLMUnavailableError as exc:
            # One failed item must not kill a multi-hour run: record and continue.
            entry = {"query": item["query"], "error": str(exc)}
        per_item.append(entry)
        log.info("triad_item_done", query=item["query"][:60])
    return {"aggregate": aggregate_triad(per_item), "per_item": per_item}


def reflection_delta(off: dict, on: dict) -> dict:
    """Delta between the reflection-off and reflection-on aggregates (pure)."""
    h_off, h_on = off["hallucination_rate"], on["hallucination_rate"]
    reduction_pct = round((h_off - h_on) / h_off * 100, 1) if h_off > 0 else None
    refusal_delta = None
    if off["refusal_accuracy"] is not None and on["refusal_accuracy"] is not None:
        refusal_delta = round(on["refusal_accuracy"] - off["refusal_accuracy"], 3)

    def d(key: str) -> float:
        return round(on[key] - off[key], 3)

    return {
        "groundedness": {
            "off": off["groundedness"],
            "on": on["groundedness"],
            "delta": d("groundedness"),
        },
        "hallucination_rate": {
            "off": h_off,
            "on": h_on,
            "reduction_absolute": round(h_off - h_on, 3),
            "reduction_percent": reduction_pct,
        },
        "answer_relevance_delta": d("answer_relevance"),
        "context_relevance_delta": d("context_relevance"),
        "refusal_accuracy_delta": refusal_delta,
        "n_scored": on["n_scored"],
        "n_unanswerable": on["n_unanswerable"],
    }


def avg_reflections(per_item: list[dict]) -> float:
    """Mean reflection count over scored items (pure)."""
    vals = [x.get("reflections", 0) for x in per_item if "reflections" in x]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


async def run_reflection(golden: list[dict]) -> dict:
    """Two-arm harness: same golden items, reflection off (budget 0) vs on."""
    from langgraph.checkpoint.memory import MemorySaver

    s = get_settings()
    llm = LLMClient()
    off_graph = build_graph(
        llm, RetrievalPipeline(), checkpointer=MemorySaver(), max_reflections=0
    )
    on_graph = build_graph(
        llm,
        RetrievalPipeline(),
        checkpointer=MemorySaver(),
        max_reflections=s.max_reflections,
    )
    per_item_off: list[dict] = []
    per_item_on: list[dict] = []
    for idx, item in enumerate(golden):
        try:
            off_entry = await _triad_item(llm, off_graph, item, f"reflect-off-{idx}")
        except LLMUnavailableError as exc:
            off_entry = {"query": item["query"], "error": str(exc)}
        try:
            on_entry = await _triad_item(llm, on_graph, item, f"reflect-on-{idx}")
        except LLMUnavailableError as exc:
            on_entry = {"query": item["query"], "error": str(exc)}
        per_item_off.append(off_entry)
        per_item_on.append(on_entry)
        log.info(
            "reflection_item_done",
            idx=idx,
            query=item["query"][:60],
            groundedness_off=off_entry.get("groundedness"),
            groundedness_on=on_entry.get("groundedness"),
        )
    agg_off = aggregate_triad(per_item_off)
    agg_on = aggregate_triad(per_item_on)
    return {
        "max_reflections_off": 0,
        "max_reflections_on": s.max_reflections,
        "avg_reflections_on": avg_reflections(per_item_on),
        "aggregate_reflection_off": agg_off,
        "aggregate_reflection_on": agg_on,
        "delta": reflection_delta(agg_off, agg_on),
        "per_item_reflection_off": per_item_off,
        "per_item_reflection_on": per_item_on,
    }


def _triad_md(a: dict) -> list[str]:
    return [
        f"- context relevance: {a['context_relevance']}",
        f"- groundedness: {a['groundedness']}",
        f"- answer relevance: {a['answer_relevance']}",
        f"- hallucination rate (1 - groundedness): {a['hallucination_rate']}",
        f"- refusal accuracy (unanswerable): {a['refusal_accuracy']}",
        f"- n scored: {a['n_scored']}, n unanswerable: {a['n_unanswerable']}, "
        f"n errors: {a['n_errors']}",
    ]


def write_reports(
    mode: str, payload: dict, out_dir: str = "evals/results"
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = out / f"{mode}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = out / f"{mode}_{stamp}.md"
    lines = [f"# OmniAgent eval: {mode}", ""]
    if mode == "ablation":
        lines += ["| arm | recall@5 | mrr | p50_latency_ms | n |", "|---|---|---|---|---|"]
        for arm, m in payload["arms"].items():
            lines.append(
                f"| {arm} | {m['recall@5']} | {m['mrr']} | {m['p50_latency_ms']} | {m['n']} |"
            )
    elif mode == "reflection":
        d = payload["delta"]
        lines += [
            f"- arms: MAX_REFLECTIONS={payload['max_reflections_off']} (off) "
            f"vs {payload['max_reflections_on']} (on)",
            f"- avg reflections (on arm): {payload['avg_reflections_on']}",
            "",
            "## Reflection off",
            *_triad_md(payload["aggregate_reflection_off"]),
            "",
            "## Reflection on",
            *_triad_md(payload["aggregate_reflection_on"]),
            "",
            "## Delta (on - off)",
            f"- groundedness: {d['groundedness']['off']} -> {d['groundedness']['on']} "
            f"(delta {d['groundedness']['delta']})",
            f"- hallucination rate: {d['hallucination_rate']['off']} -> "
            f"{d['hallucination_rate']['on']} "
            f"(absolute reduction {d['hallucination_rate']['reduction_absolute']}, "
            f"relative {d['hallucination_rate']['reduction_percent']}%)",
            f"- answer relevance delta: {d['answer_relevance_delta']}",
            f"- context relevance delta: {d['context_relevance_delta']}",
            f"- refusal accuracy delta: {d['refusal_accuracy_delta']}",
            f"- n scored: {d['n_scored']}, n unanswerable: {d['n_unanswerable']}",
        ]
    else:
        lines += _triad_md(payload["aggregate"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


async def main() -> None:
    parser = argparse.ArgumentParser(description="OmniAgent evaluation runner")
    parser.add_argument(
        "--mode", choices=["ablation", "triad", "reflection"], default="ablation"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    s = get_settings()
    golden = load_golden(s.eval_golden_path)
    if args.limit:
        golden = golden[: args.limit]

    if args.mode == "ablation":
        arms = await run_ablation(golden)
        payload = {"mode": "ablation", "golden_items": len(golden), "arms": arms}
    elif args.mode == "reflection":
        reflection = await run_reflection(golden)
        payload = {"mode": "reflection", "golden_items": len(golden), **reflection}
    else:
        triad = await run_triad(golden)
        payload = {"mode": "triad", "golden_items": len(golden), **triad}

    json_path, md_path = write_reports(args.mode, payload)
    print(f"report: {json_path}")
    if args.mode == "reflection":
        print(json.dumps(payload["delta"], indent=2))
    else:
        print(json.dumps(payload.get("arms", payload.get("aggregate")), indent=2))
    await get_engine().dispose()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
