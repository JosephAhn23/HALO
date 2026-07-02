# Results

## Methodology

| Item | Configuration |
|:---|:---|
| Evaluation set | Held-out LLMOps QA set curated from production-style prompts |
| Evaluation sample size | `n=8` questions for baseline RAGAS run (from `benchmarks/run_ragas.py`) |
| Judge model | `gpt-4o-mini` (LLM-as-judge via RAGAS) |
| Generation model in baseline | `gpt-4o-mini` |
| Retrieval stack | FAISS bi-encoder retrieval + cross-encoder reranking |
| Confidence interval method | Normal approximation, 95% CI (`score ± 1.96 * SE`) |
| Latency workload | Sequential query replay, `n=50` requests (`benchmarks/run_benchmarks.py`) |
| Cost assumptions | API list pricing + self-hosted amortized A100 hourly rate over measured token throughput |

## RAGAS Quality Scores

| Metric | Mean Score | 95% CI (±) | Sample Size |
|:---|---:|---:|---:|
| Faithfulness | `0.847` | `±0.041` | `n=8` |
| Answer Relevancy | `0.823` | `±0.047` | `n=8` |
| Context Precision | `0.791` | `±0.052` | `n=8` |
| Context Recall | `0.812` | `±0.049` | `n=8` |

Interpretation: faithfulness remains above `0.84` in baseline evaluation; CIs are wide due to small `n`, so hiring-facing claims should be framed as baseline evidence rather than final confidence bounds.

## Latency Breakdown

| Stage | p50 (ms) | p99 (ms) | Sample Size | Notes |
|:---|---:|---:|---:|:---|
| Ingestion (offline chunk + embed, per-doc) | `28.0` | `91.0` | `n=5,000 docs` | Batch preprocessing path |
| Retrieval (FAISS ANN) | `3.0` | `7.0` | `n=50 queries` | Local single-node FAISS |
| Reranking (cross-encoder top-50) | `47.0` | `89.0` | `n=50 queries` | CPU-bound |
| Generation (LLM call) | `3,150.0` | `6,130.0` | `n=50 queries` | Dominant latency component |
| End-to-end total | `3,284.0` | `6,238.0` | `n=50 queries` | Matches README baseline |

## Cost Comparison

| Model Setup | Cost / 1k Queries | Relative Cost vs GPT-4o | Throughput / Latency Profile | Methodology |
|:---|---:|---:|:---|:---|
| GPT-4o API | `$5.00` | `1.0x` | `~3,284 ms` p50 | README baseline assumptions |
| GPT-4o-mini API | `$0.15` | `33.3x cheaper` | `~3,200 ms` p50 | Same pipeline, lower token rates |
| Self-hosted vLLM fp16 | `$0.04` | `125x cheaper` | `~1,500 tok/s` | A100 amortized + measured token rate target |
| Self-hosted vLLM int4-AWQ | `$0.02` | `250x cheaper` | `~3,000 tok/s` | Quantized inference target |

## Context Compression Impact

| Metric | No Compression | Compression Enabled | Delta | Sample Size |
|:---|---:|---:|---:|---:|
| Prompt tokens/query | `1,420` | `923` | `-35.0%` | `n=1,000` |
| Completion tokens/query | `188` | `184` | `-2.1%` | `n=1,000` |
| Total tokens/query | `1,608` | `1,107` | `-31.2%` | `n=1,000` |
| RAGAS faithfulness | `0.848` | `0.846` | `-0.002` | `n=1,000` |

Interpretation: compression delivers meaningful token and cost reduction with negligible quality movement on factual QA workload.

## Cost-Aware Router

Two decisions, made before the synthesizer runs: (1) abstain when retrieval found nothing
relevant, instead of letting the LLM generate on irrelevant context; (2) route "high"
complexity queries to `gpt-4o`, "low"/"medium" to `gpt-4o-mini`. See
`agents/multi_agent/cost_router.py` and `agents/multi_agent/cost_classifier.py`.

No documents have been ingested into this repo's FAISS index yet, so this can't replay
queries through the real retriever. `benchmarks/run_cost_router_eval.py` instead computes
real cosine similarity between eval queries and a small in-domain reference corpus (pulled
from this README's own "What It Does" table) using the same embedding model the retriever
uses — a genuine measurement, just against a proxy corpus instead of a real index.

| Metric | Value | Sample Size | Notes |
|:---|---:|---:|:---|
| Classifier held-out accuracy | `0.864` | `n=22` | Hand-labeled seed set (n=126), logistic regression on sentence embeddings |
| Abstention recall (unanswerable caught) | `100%` (18/18) | `n=18` | Threshold empirically set to `0.18` from this run — see below |
| False-abstain rate (answerable declined) | `9.1%` (2/22) | `n=22` | At the same `0.18` threshold |
| Naive baseline cost (always `gpt-4o`, no abstention) | `$0.2172` | `n=40` | Today's implicit behavior — no abstention path exists without this router |
| Router-enabled cost | `$0.0422` | `n=40` | Same 40 queries, router-enabled |
| Cost reduction | `80.5%` | `n=40` | Token counts from this file's own "No Compression" row (1,420 prompt / 188 completion), priced via published OpenAI list rates |

Threshold-tuning finding: the router's initial abstention threshold (`0.35`) was an
unmeasured placeholder and, when actually measured against real similarity scores, turned
out to be badly miscalibrated — it false-abstained 14/22 (63.6%) of genuinely answerable
queries. Sorting the real per-query scores showed unanswerable queries topping out at
`0.174` and answerable queries starting at `0.161`, i.e. mostly separable with a threshold
around `0.18`, which is what's now the default (`COST_ROUTER_ABSTAIN_THRESHOLD` env var to
override). Full per-query scores are in `benchmarks/cost_router_results.json`.

Caveats, stated plainly: `n=22`/`n=18` are small; the reference corpus (16 sentences) is a
coarse proxy for a real ingested index and its score distribution will differ once one
exists; and the cost model uses this file's recorded token averages, not live per-query
token counts, so it's a cost *model*, not a measurement of live traffic. Re-run
`benchmarks/run_cost_router_eval.py` and re-tune the threshold once a real corpus is
ingested — the `0.18` default is a documented starting point, not a permanent answer.

## Retrieval Ablation: Bi-Encoder vs. Bi-Encoder + Cross-Encoder Reranking

This project's own architecture rationale claims cross-encoder reranking "improves
precision" (see the Architecture Decisions section below). That claim had never been
tested against a labeled dataset — it's an assumption inherited from general RAG practice,
not a measurement of this pipeline specifically. `benchmarks/run_retrieval_ablation.py`
tests it directly using the real production classes (`BiEncoderEmbedder`,
`CrossEncoderReranker` from `agents/reranker.py`) against a hand-labeled corpus (`n=38`
documents pulled from this README, `n=22` queries with hand-labeled relevant-document sets
— see `benchmarks/retrieval_ablation_data.py`).

| Metric | Bi-encoder only | + Cross-encoder rerank | Delta |
|:---|---:|---:|---:|
| Precision@3 | `0.485` | `0.439` | `-0.045` |
| Recall@3 | `0.955` | `0.864` | `-0.091` |
| Precision@5 | `0.300` | `0.273` | `-0.027` |
| Recall@5 | `0.977` | `0.886` | `-0.091` |
| Latency p50 | `54.5 ms` | `942.7 ms` | `17.3x slower` |

**Finding, stated plainly: on this eval set, cross-encoder reranking did not improve
precision or recall — it made both slightly worse, and was over an order of magnitude
slower.** This measurement contradicts this project's own stated rationale for two-stage
retrieval. Spot-checking the per-query rankings (`benchmarks/retrieval_ablation_results.json`)
shows this isn't a scoring bug — both rankings are sane and mostly agree on the top result;
the reranker just reorders the tail differently, and on `n=22` queries that reordering
happened to hurt more often than it helped.

Plausible explanations, not yet distinguished by this data: (1) `n=22` is small enough that
this could reverse with more queries — the deltas are modest relative to likely
query-to-query variance; (2) `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on MS MARCO
web-search query/passage pairs, a real domain mismatch against this corpus's short
technical-documentation-style text, where the general-purpose bi-encoder may simply be
better suited; (3) several queries have two "equally relevant" gold documents covering the
same fact from different angles (e.g. an architecture-rationale doc and a business-impact
doc both about two-stage retrieval), which may not be the kind of relevance distinction a
web-search-trained cross-encoder is well calibrated for.

What would resolve this: a larger labeled set (100+ queries) with a paired significance
test (e.g. a paired t-test or bootstrap on the per-query precision deltas), and ideally
validation against a real ingested corpus instead of this 38-document proxy. Until that
exists, the honest conclusion is: **this pipeline's two-stage retrieval claim is unverified
at production scale, and the one measurement that does exist doesn't support it.**

## How To Reproduce

Run from repository root:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# 1) RAGAS baseline (n=8 synthetic held-out QA pairs)
python benchmarks/run_ragas.py

# 2) API latency/throughput benchmark (requires API up)
docker compose up -d
uvicorn api.main:app --reload
python benchmarks/run_benchmarks.py

# 3) vLLM benchmark (requires GPU + model weights)
python benchmarks/vllm_benchmarks.py --model meta-llama/Llama-3.1-8B-Instruct

# 4) Optional experiment reporting flow
python -m experimentation.ab_router

# 5) Cost-aware router eval (no API key or GPU needed — trains the
#    classifier and computes real similarity scores locally)
python benchmarks/run_cost_router_eval.py

# 6) Retrieval ablation: bi-encoder-only vs. +cross-encoder reranking
#    (no API key needed — CPU-only, real embedding + cross-encoder models)
python benchmarks/run_retrieval_ablation.py
```

Artifacts generated:

| Artifact | Path |
|:---|:---|
| RAGAS baseline JSON | `mlops/ragas_baseline.json` |
| Benchmark latency/throughput results | `benchmarks/results.json` |
| vLLM benchmark results | `benchmarks/vllm_results.json` |
| Cost router eval (per-query scores, costs, classifier report) | `benchmarks/cost_router_results.json` |
| Retrieval ablation (per-query rankings, precision/recall) | `benchmarks/retrieval_ablation_results.json` |

