# OmniAgent Evaluation Methodology

Principle: numbers over adjectives. Every quality claim traces to an
artefact in `evals/results/` (JSON + Markdown), reproducible with one command.

## Golden set

`evals/golden/queries.jsonl`: 14 questions over a 24-document corpus
(12 target documents covering RAG/IR topics + 12 vocabulary-overlapping
distractor documents: LoRA, quantization, attention, caching, tracing...).
Each entry has `expected_docs` (document-level labels). Two entries are
**unanswerable** (topics absent from the corpus) to test the refusal path.

## Mode 1: retrieval ablation (deterministic, no LLM)

`make ablation` runs three arms over the same questions:

| arm | dense | bm25 | rerank |
|---|---|---|---|
| dense | x | | |
| dense+bm25 | x | x | |
| dense+bm25+rerank | x | x | x |

Metrics: recall@5 (labelled docs found in top-5), MRR (first relevant doc),
p50 retrieval latency. Stage flags make the arms exact -- the same pipeline
code runs with different toggles (`RetrievalStageFlags`).

## Mode 2: RAG Triad (LLM-as-judge)

`.venv/bin/python -m app.eval.runner --mode triad`

- **Context relevance**: judge rates each retrieved passage 1/0 for answering
  the question; score = mean.
- **Groundedness**: per-claim verdicts from the graph's fact-checker (strict:
  supported only if the evidence states it explicitly); score = supported
  claims / total. **Hallucination rate = 1 - groundedness.**
- **Answer relevance**: judge decides if the answer addresses the question.
- **Refusal accuracy** (unanswerable items): a correct run refuses rather than
  inventing.

Judge = the same local backend (configurable via `EVAL_JUDGE_*` settings);
judged items include the full graph run latency and reflection count.

## Measured results (this deployment, 2 vCPU / 7.8 GB / no GPU)

Retrieval ablation, 24-doc corpus, 12 answerable questions, top-8:

| arm | recall@5 | MRR | p50 latency |
|---|---|---|---|
| dense | 1.000 | 1.000 | 27.0 ms |
| dense+bm25 (RRF) | 1.000 | 0.958 | 24.2 ms |
| dense+bm25+rerank | 1.000 | 1.000 | 412.7 ms |

Tight-budget run (top-3) shows the same shape (rerank p50 240 ms).

### Honest interpretation

On a 24-document corpus of topically distinct documents, **dense retrieval is
at ceiling**: every arm finds every labelled document. Hybrid fusion adds no
recall here, RRF's reshuffle costs MRR (1.000 -> 0.958), and the cross-encoder
restores MRR to 1.000 at +388 ms p50. This mirrors published experience
(Production-RAG's ablation notes the same near-ceiling behavior on small,
well-separated corpora): **hybrid recall gains require a corpus with
adversarial distractors and identifier-heavy queries.**

The value delivered here is the harness: exact arms, labelled metrics, and
committed reports, so any future corpus/model change is measured, not vibes.

## Extending the evidence (next runs)

1. Grow the corpus 10x with near-duplicate distractor topics; rerun
   `make ablation` -- expect the sparse leg to contribute recall on
   identifier-heavy queries (exact model names, parameter names).
2. Run `--mode triad` with reflection off (`MAX_REFLECTIONS=0`) vs on; report
   the groundedness delta attributable to the reflection loop.
3. On a GPU host: vLLM vs Ollama throughput/latency with hardware stated
   (ADR-0005) -- no inference numbers are claimed until measured.

## Limitations

- LLM-as-judge scores inherit judge-model bias; the judge prompts are strict
  and versioned in code, and the judge model is configurable.
- Document-level labels (not chunk-level) bound recall precision; acceptable
  for this corpus size.
- The triad uses the same local 3B model as judge and generator; a stronger
  external judge is the upgrade path.
