# JaisGuard: vLLM versus original Hugging Face/FastAPI

Same validation-selected trained checkpoints, exact saved thresholds, label order VALID=0 / GIBBERISH=1 and final nonpadding token. Historical quality results are unchanged. No Jev API requests. A CPU export alone is not successful conversion: GPU parity is required.

Acceptance: HF FP16 versus vLLM FP16, verified trained head and bias, zero default and locked-threshold mismatches, max probability difference ≤0.02 and mean ≤0.005. Original BF16 differences are preserved as nonblocking diagnostics by user authorization.

## Parity

### v2-frozen-43

Locked threshold: 0.9563547372817993.

- hf-original_vs_hf-fp16: max absolute probability difference 0.05112457, mean 0.00422282, locked mismatches 1, default mismatches 3; within_strict_tolerance=False (BF16 rows are diagnostic).
- hf-fp16_vs_vllm: max absolute probability difference 0.00748599, mean 0.00048215, locked mismatches 0, default mismatches 0; within_strict_tolerance=True (BF16 rows are diagnostic).
- hf-original_vs_vllm: max absolute probability difference 0.04363859, mean 0.00416221, locked mismatches 1, default mismatches 3; within_strict_tolerance=False (BF16 rows are diagnostic).
- benchmark: 400 inputs, max/mean probability difference 0.00606002/0.00049197, locked mismatches=0.
- clean: 2 inputs, max/mean probability difference 0.00000467/0.00000266, locked mismatches=0.
- dialect: 3 inputs, max/mean probability difference 0.00014949/0.00006020, locked mismatches=0.
- gibberish: 2 inputs, max/mean probability difference 0.00098772/0.00054103, locked mismatches=0.
- recoverable_corruption: 1 inputs, max/mean probability difference 0.00019326/0.00019322, locked mismatches=0.
- validation: 250 inputs, max/mean probability difference 0.00748599/0.00047603, locked mismatches=0.
### v2-lora-44

Locked threshold: 0.942931056022644.

- hf-original_vs_hf-fp16: max absolute probability difference 0.12901187, mean 0.00352034, locked mismatches 1, default mismatches 1; within_strict_tolerance=False (BF16 rows are diagnostic).
- hf-fp16_vs_vllm: max absolute probability difference 0.01287434, mean 0.00037334, locked mismatches 0, default mismatches 0; within_strict_tolerance=True (BF16 rows are diagnostic).
- hf-original_vs_vllm: max absolute probability difference 0.13155121, mean 0.00353919, locked mismatches 1, default mismatches 1; within_strict_tolerance=False (BF16 rows are diagnostic).
- benchmark: 400 inputs, max/mean probability difference 0.01287434/0.00040227, locked mismatches=0.
- clean: 2 inputs, max/mean probability difference 0.00000104/0.00000050, locked mismatches=0.
- dialect: 3 inputs, max/mean probability difference 0.00000824/0.00000274, locked mismatches=0.
- gibberish: 2 inputs, max/mean probability difference 0.00009133/0.00004566, locked mismatches=0.
- recoverable_corruption: 1 inputs, max/mean probability difference 0.00000012/0.00000009, locked mismatches=0.
- validation: 250 inputs, max/mean probability difference 0.01149392/0.00033861, locked mismatches=0.

## HTTP latency

| Checkpoint | Backend | Concurrent requests | Mean ms | p50 | p95 | p99 | requests/s | Load s | Ready s | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2-frozen-43 | hf | 1 | 29.554 | 28.252 | 40.612 | 47.578 | 33.810 | 1.277 | 12.554 | 1996.0 | 2168.0 |
| v2-frozen-43 | hf | 4 | 98.069 | 97.919 | 103.809 | 105.784 | 40.613 | 1.277 | 12.554 | 2092.0 | 2168.0 |
| v2-frozen-43 | hf | 8 | 196.825 | 196.917 | 210.967 | 223.151 | 40.301 | 1.277 | 12.554 | 2220.0 | 2264.0 |
| v2-frozen-43 | hf | 16 | 396.417 | 396.233 | 450.224 | 473.877 | 39.639 | 1.277 | 12.554 | 2476.0 | 2520.0 |
| v2-frozen-43 | vllm | 1 | 34.330 | 32.544 | 44.251 | 50.270 | 29.107 | 12.273 | 27.063 | 4505.7 | 4524.0 |
| v2-frozen-43 | vllm | 4 | 58.303 | 58.031 | 62.650 | 65.278 | 68.480 | 12.273 | 27.063 | 4507.1 | 4524.0 |
| v2-frozen-43 | vllm | 8 | 62.435 | 60.316 | 68.574 | 157.931 | 127.847 | 12.273 | 27.063 | 4510.4 | 4526.0 |
| v2-frozen-43 | vllm | 16 | 67.091 | 66.942 | 75.902 | 81.233 | 234.866 | 12.273 | 27.063 | 4514.1 | 4528.0 |
| v2-lora-44 | hf | 1 | 40.563 | 37.775 | 57.116 | 63.175 | 24.637 | 1.404 | 13.553 | 1998.4 | 2170.0 |
| v2-lora-44 | hf | 4 | 154.491 | 157.196 | 193.687 | 204.209 | 25.811 | 1.404 | 13.553 | 2094.4 | 2170.0 |
| v2-lora-44 | hf | 8 | 254.602 | 254.568 | 275.214 | 279.425 | 31.162 | 1.404 | 13.553 | 2222.4 | 2266.0 |
| v2-lora-44 | hf | 16 | 600.782 | 612.810 | 685.099 | 701.389 | 26.132 | 1.404 | 13.553 | 2478.4 | 2522.0 |
| v2-lora-44 | vllm | 1 | 32.176 | 31.651 | 36.079 | 41.447 | 31.052 | 11.491 | 26.541 | 4505.7 | 4524.0 |
| v2-lora-44 | vllm | 4 | 58.170 | 57.553 | 64.335 | 74.450 | 68.651 | 11.491 | 26.541 | 4507.1 | 4524.0 |
| v2-lora-44 | vllm | 8 | 60.400 | 59.710 | 69.810 | 81.462 | 132.153 | 11.491 | 26.541 | 4510.4 | 4526.0 |
| v2-lora-44 | vllm | 16 | 67.340 | 66.331 | 87.013 | 96.433 | 236.544 | 11.491 | 26.541 | 4515.0 | 4528.0 |

Completed HTTP workloads: 16/16. Missing workloads remain pending; partial results are not a completed suite.

## Standalone batch-1 local inference

Includes original input tokenization and transfers; no HTTP. vLLM uses LLM.classify, HF uses the existing Predictor. Both must preserve the same locked classifications.

| Model | Backend | Mean ms | p50 | p95 | p99 | requests/s | Load s |
|---|---|---:|---:|---:|---:|---:|---:|
| v2-frozen-43 | hf-fp16 | 24.687 | 23.212 | 36.607 | 47.194 | 40.506 | 1.138 |
| v2-frozen-43 | vllm | 29.957 | 28.997 | 36.386 | 38.391 | 33.379 | 15.452 |
| v2-lora-44 | hf-fp16 | 30.789 | 29.480 | 41.097 | 57.409 | 32.477 | 1.472 |
| v2-lora-44 | vllm | 29.007 | 28.834 | 30.804 | 32.008 | 34.472 | 12.741 |

## Measurement scope and limitations

Each concurrency setting sends the same 400 frozen inputs once, one input per request, using a reused HTTP connection pool and three warmup inputs. Throughput is completed requests / wall time; mean and linear p50/p95/p99 summarize request durations. vLLM can batch up to 16 concurrent requests; the original HF GPU lock serializes inference. Every HTTP result must pass probability/class parity against the HF FP16 reference. The existing FastAPI app and GPU lock are reused with an FP16-only predictor adaptation; the original BF16 implementation remains unchanged.

Loading is separate from warm inference; ready time includes process startup, integrity checks, tokenizer and engine initialization. Peak memory is CUDA allocated/reserved memory in the actual GPU worker, with resident weights included; not total board usage. Reserved memory can retain loading/warmup allocations. vLLM reserves more memory by design. Only one model is resident at a time. These local HTTP results exclude remote networking. Existing Jev results include remote network overhead and are not rerun or relabeled.

FP16 serving equivalence uses the existing tolerances: maximum absolute difference ≤0.02 and mean ≤0.005 across both probabilities, with zero classification mismatches at default and saved thresholds. Exceeding either limit blocks performance claims against the HF FP16 reference. BF16 differences do not block this authorized engineering experiment and remain reported above. Thresholds and tolerances are unchanged.

vLLM 0.32.0 uses its supported Transformers backend with vLLM attention/linear layers, LAST pooler and the verified original FP32 head weight and bias. `classifier.bias` avoids the installed LegacyMixin rule that drops `score.bias`. LoRA uses a verified CPU FP32 static merge into a new export; runtime LoRA classification support is not assumed. Merging may change floating-point rounding; parity remains mandatory.

This is the previously examined, constructed Arabic benchmark, not new quality evaluation or production traffic. Hardware, clocks, thermal state, dtype, kernel/backend versions and caches affect latency. No test-based model/threshold selection or silent HF fallback.
