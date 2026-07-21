# HALO RAGAS Quality Gate (GitHub Action)

A composite action that fails your CI job when RAG quality scores drop below
a hard threshold or regress vs. a saved baseline. It's a thin wrapper around
[`src/cicd/ragas_gate.py`](../../../src/cicd/ragas_gate.py) — pure Python
standard library, no extra dependencies to install, no LLM calls made by the
action itself. You bring the scores (from `ragas`, your own eval harness,
whatever produces a `{"metric": score}` JSON), this step decides whether the
build passes.

This is the same action HALO's own CI ([`.github/workflows/ci.yml`](../../workflows/ci.yml))
uses to gate itself — it's not a demo wrapper published separately from what's
actually running in production here.

## Usage

From another repository:

```yaml
- name: RAGAS quality gate
  id: gate
  uses: JosephAhn23/HALO/.github/actions/gate@main
  with:
    scores: reports/ragas_scores.json
    baseline: eval/ragas_baseline.json   # optional — omit to skip regression checking
    tolerance: '0.03'                    # optional — default shown

- name: Block merge on failure
  if: steps.gate.outputs.passed != 'true'
  run: exit 1
```

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `scores` | yes | — | Path to current scores JSON (`{"faithfulness": 0.85, ...}` or `{"scores": {...}}`) |
| `baseline` | no | _(none)_ | Path to a baseline scores JSON. Omitted → regression check is skipped, only hard thresholds apply. |
| `tolerance` | no | `0.03` | Max allowed absolute drop vs. baseline before the gate fails |
| `thresholds` | no | _(built-in)_ | Path to a JSON file of custom per-metric minimums. Default floor: `faithfulness 0.80`, `answer_relevancy 0.78`, `context_precision 0.75`, `context_recall 0.75` |
| `output-file` | no | _(none)_ | Optional path to write the full structured gate result as JSON |

## Outputs

| Output | Description |
|---|---|
| `passed` | `"true"` or `"false"` |
| `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` | Echoed input scores, for use in PR comments or downstream steps |

## What it doesn't do

It doesn't run `ragas` for you, and it doesn't call any LLM. If you need a
real, non-hand-written eval to feed it, see
[`src/benchmarks/run_real_corpus_eval.py`](../../../src/benchmarks/run_real_corpus_eval.py)
for a working reference implementation (ingest → retrieve → rerank → RAGAS).
