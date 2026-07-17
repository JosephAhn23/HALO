# Architecture Decisions

**Why LangGraph instead of a simple chain?**
A chain runs top-to-bottom and stops. LangGraph is a directed graph where each node can inspect the full state, decide which node to call next, and recover from failures without restarting. That matters when retrieval returns nothing useful (route to fallback) or when the safety check fires mid-pipeline (short-circuit before generation).

**Why two-stage retrieval (bi-encoder + cross-encoder)?**
Bi-encoders (FAISS) are fast but approximate. They compare embeddings independently, missing subtle relevance signals. Cross-encoders read the query and document together, catching nuance the bi-encoder misses. Running cross-encoding only on the top-50 FAISS results keeps end-to-end latency under 50ms while improving precision — that's the design intent. It had never actually been tested against a labeled dataset until `src/benchmarks/run_retrieval_ablation.py`. On that measurement (`n=22` hand-labeled queries), neither this original two-stage design *nor* a proposed BM25+dense hybrid (RRF fusion) alternative beat plain bi-encoder cosine similarity — bi-encoder-only won or tied on every precision/recall metric against both. See [RESULTS.md](../RESULTS.md#retrieval-ablation-bi-encoder-vs-cross-encoder-reranking-vs-bm25dense-rrf) for the numbers and the most likely explanations (small `n`, possible domain mismatch, paraphrase-style eval queries favoring semantic search). Treat the claim above as the original design rationale, not a verified result.

**Why FAISS over a managed vector database?**
FAISS runs in-process: no network hop, no managed service cost, no vendor lock-in. The distributed shard design (4 shards + async fan-out aggregator) gives horizontal scale without changing the query interface. Trade-off: no real-time updates as cleanly as Pinecone or Weaviate. For a research assistant with periodic re-indexing, that is acceptable.

**Why QLoRA instead of full fine-tuning?**
Full fine-tuning an 8B model requires roughly 80GB of GPU memory. QLoRA compresses the frozen base model to 4-bit and trains only small LoRA adapter matrices injected into attention layers: less than 1% of total parameters. The quality gap versus full fine-tuning is small for most tasks; the hardware requirement drops from 4x A100s to a single consumer GPU.

**Why Double ML for causal inference instead of a simple A/B test?**
A/B tests measure correlation. When users self-select into features (e.g., power users enable reranking), a naive comparison is confounded. Double ML residualises both the outcome and the treatment on observed covariates using k-fold cross-fitting, then regresses the residuals. The resulting ATE estimate is unbiased even when the confounders are complex and nonlinear.

**Why O'Brien-Fleming sequential testing instead of fixed-horizon?**
Checking p-values repeatedly inflates the false positive rate. O'Brien-Fleming alpha spending allocates the Type-I error budget across planned looks: conservative early (high boundary), liberal late (low boundary). The overall false positive rate stays at alpha regardless of how many times you check.

**Why circuit breakers in the multi-agent system?**
A slow or failing verifier agent would block the entire pipeline under naive retry. The circuit breaker opens after N consecutive failures, immediately returning a degraded response (unverified output with a confidence penalty) instead of waiting for timeouts. After a cooldown period, one probe call tests recovery. This prevents cascade failures while maintaining responsiveness.

**Why Kafka + Redis Streams (both)?**
Kafka is the right choice for production: durable, ordered, replayable, consumer groups. Redis Streams is the right choice for local development: zero infrastructure, same API shape, instant startup. The `EventBus` class auto-detects which backend to use from environment variables, so the same code runs locally and in production without changes.
