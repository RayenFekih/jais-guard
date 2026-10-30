# Reviewed benchmark evidence

The explicit public allowlist is [benchmarks/allowlist.json](benchmarks/allowlist.json).
Only its 17 enumerated files are reviewed evidence; adding a directory does not
clear arbitrary future contents. Originals remain unchanged under ignored local
`results/`. Destination hashes, source hashes and copy/projection rules are recorded.

| Evidence | Public files | Verification and use |
| --- | --- | --- |
| Native classification throughput | [comparison](benchmarks/vllm/comparison.md), [CSV](benchmarks/vllm/comparison.csv), [metrics](benchmarks/vllm/metrics.json) | Completed v4, 16 HTTP workloads and four standalone measurements; FP16 HF versus FP16 vLLM |
| HTTP numerical evidence | [timings](benchmarks/vllm/http-timings.json) | 400 durations per workload; aggregate mean, linear percentiles and throughput checked; no texts, tokens or candidate IDs |
| Classifier equivalence | [parity summaries](benchmarks/vllm/parity-summary.json) | Both trained-head gates passed; zero FP16 default/locked mismatches; max/mean tolerances 0.02/0.005; BF16 disagreements preserved |
| Historical quality and sequential latency | [comparison](benchmarks/quality-latency/comparison.md), [quality CSV](benchmarks/quality-latency/comparison.csv), [latency CSV](benchmarks/quality-latency/latency.csv), [aggregates](benchmarks/quality-latency/aggregates.json) | Three-seed quality, validation-selected representative checkpoints, local timings and remote Jev clearly separated |

## Presentation shortlist

- [Latency and throughput](benchmarks/vllm/latency-throughput.svg): matched HTTP concurrency 1/4/8/16, both selected classifiers. PNG copy included.
- [Startup and memory](benchmarks/vllm/startup-memory.svg): load/ready times and CUDA allocator memory. PNG copy included.
- [Historical quality](benchmarks/quality-latency/quality.svg): previously examined constructed benchmark; no independent-test claim. PNG copy included.
- [Sequential latency](benchmarks/quality-latency/latency.svg): local sequential timings; Jev remains a separate remote scope. PNG copy included.

The v4 report copies and figures are byte-identical to originals. The combined
quality Markdown replaces only the absolute checkout prefix; all numbers are
unchanged. Aggregate JSON projects selected sections. Parity summaries omit
private paths and mismatch IDs but retain diagnostic counts and numerical values.
HTTP timing projections omit all identifiers, token IDs and probabilities.

The observed GPU was an RTX 4050. HF serializes predictions with a GPU lock;
vLLM continuously batches concurrent requests. Memory is allocated/reserved CUDA
memory, not total board memory. FP16 parity does not imply BF16 equivalence.
Results remain measurements on these checkpoints and workload, not universal claims.

Private prediction caches, source-family ledgers, training exports, weights,
reviewer notes and author hashes are not allowlisted. The prior tracked v4 report
under `results/` remains untouched. Hash checking detects accidental corruption;
the allowlist is not a cryptographic signature by an independent reviewer.

Recreate reviewed copies from authentic local originals into a new directory:

```bash
.venv/bin/python scripts/prepare_public_evidence.py --output /absolute/path/new-public-evidence
jaisguard verify-evidence --directory /absolute/path/new-public-evidence
```

The preparation script refuses an existing destination. Its output was reproduced
and compared to the exact allowlist and all 17 public destination hashes.
