# JaisGuard quality and latency

Exploratory head-to-head comparison on a previously examined, human-reviewed, constructed Arabic-input benchmark (400 inputs; 188 VALID, 212 GIBBERISH). This is not production traffic or an unbiased final evaluation. Jev and untuned Jais are zero-shot; JaisGuard was fine-tuned specifically for this task. No test-based threshold or model selection is performed.

Quality reuses all archived predictions and locked operating points. Three-seed means ± sample SD describe repeated models on the same 400 inputs, not independent test samples. INVALID stays -1 and counts as wrong; VALID FRR counts explicit GIBBERISH only, so acceptance need not equal 1 − FRR.

Latency in this quality table uses one representative checkpoint per architecture and iteration, selected by minimum validation loss (ties by seed). All measured seed timings appear below. Default and locked thresholds share one forward pass; quality is never recomputed from the latency runs.

| Model / operating point | Accuracy | Macro F1 | G precision | G recall | VALID FRR | VALID acceptance | INVALID / 400 | GPU mean ms | HTTP mean ms | Remote mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1/frozen/default | 0.8950 ± 0.0043 | 0.8942 ± 0.0045 | 0.8831 ± 0.0122 | 0.9245 ± 0.0142 | 0.1383 ± 0.0184 | 0.8617 ± 0.0184 | 0.0000 ± 0.0000 | 28.35 | 27.11 | pending |
| v1/frozen/validation_locked | 0.8383 ± 0.0364 | 0.8367 ± 0.0386 | 0.9687 ± 0.0389 | 0.7217 ± 0.0994 | 0.0301 ± 0.0392 | 0.9699 ± 0.0392 | 0.0000 ± 0.0000 | 28.35 | 27.11 | pending |
| v1/lora/default | 0.9017 ± 0.0072 | 0.9010 ± 0.0078 | 0.8993 ± 0.0393 | 0.9198 ± 0.0330 | 0.1188 ± 0.0525 | 0.8812 ± 0.0525 | 0.0000 ± 0.0000 | 34.69 | 37.23 | pending |
| v1/lora/validation_locked | 0.8950 ± 0.0195 | 0.8949 ± 0.0194 | 0.9286 ± 0.0212 | 0.8695 ± 0.0472 | 0.0762 ± 0.0273 | 0.9238 ± 0.0273 | 0.0000 ± 0.0000 | 34.69 | 37.23 | pending |
| untuned/untuned_prompt/zero_shot | 0.7500 | 0.7705 | 0.9915 | 0.5519 | 0.0053 | 0.9734 | 30.0000 | 160.84 | 153.81 | pending |
| v2/frozen/default | 0.8792 ± 0.0038 | 0.8781 ± 0.0036 | 0.8638 ± 0.0059 | 0.9167 ± 0.0166 | 0.1631 ± 0.0111 | 0.8369 ± 0.0111 | 0.0000 ± 0.0000 | 23.50 | 25.43 | pending |
| v2/frozen/validation_locked | 0.8275 ± 0.0195 | 0.8256 ± 0.0203 | 0.9843 ± 0.0103 | 0.6855 ± 0.0366 | 0.0124 ± 0.0081 | 0.9876 ± 0.0081 | 0.0000 ± 0.0000 | 23.50 | 25.43 | pending |
| v2/lora/default | 0.8967 ± 0.0177 | 0.8955 ± 0.0187 | 0.8757 ± 0.0369 | 0.9403 ± 0.0152 | 0.1525 ± 0.0546 | 0.8475 ± 0.0546 | 0.0000 ± 0.0000 | 34.24 | 35.65 | pending |
| v2/lora/validation_locked | 0.9100 ± 0.0109 | 0.9100 ± 0.0109 | 0.9687 ± 0.0178 | 0.8585 ± 0.0340 | 0.0319 ± 0.0192 | 0.9681 ± 0.0192 | 0.0000 ± 0.0000 | 34.24 | 35.65 | pending |
| jev/hosted/zero_shot | 0.6225 | 0.5802 | 1.0000 | 0.2877 | 0.0000 | 1.0000 | 0.0000 | pending | pending | 217.52 |

## Local latency measurements

Batch size 1; three warmup inputs; 400 sequential measured inputs in frozen order. GPU latency includes preprocessing, transfers and classification/generation with CUDA synchronization. HTTP is a separate run and includes local transport and server overhead. Loading is excluded from both. Throughput uses summed prediction time for GPU and full request-loop time for HTTP. Raw samples and server durations are saved separately. Memory is process CUDA allocator memory, not total board VRAM.

| Scope / model | Mean ms | p50 | p95 | p99 | inputs/s | Load s | Inference peak allocated MiB | Inference peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_gpu/v1-frozen-42 | 28.35 | 24.36 | 50.38 | 62.89 | 35.27 | 4.452 | 1997.45 | 2010.00 |
| local_gpu/v1-frozen-43 | 24.20 | 23.64 | 28.03 | 30.37 | 41.31 | 2.506 | 1997.45 | 2010.00 |
| local_gpu/v1-frozen-44 | 32.18 | 31.07 | 45.54 | 47.44 | 31.08 | 2.015 | 1997.45 | 2010.00 |
| local_gpu/v1-lora-42 | 35.67 | 33.98 | 48.85 | 65.77 | 28.03 | 2.340 | 2002.26 | 2014.00 |
| local_gpu/v1-lora-43 | 34.69 | 33.73 | 40.72 | 46.63 | 28.83 | 2.307 | 2002.26 | 2014.00 |
| local_gpu/v1-lora-44 | 36.36 | 34.44 | 47.08 | 59.65 | 27.50 | 2.288 | 2002.26 | 2014.00 |
| local_gpu/v2-frozen-42 | 23.62 | 23.28 | 26.21 | 28.77 | 42.33 | 1.950 | 1997.45 | 2010.00 |
| local_gpu/v2-frozen-43 | 23.50 | 23.20 | 25.80 | 28.53 | 42.56 | 2.022 | 1997.45 | 2010.00 |
| local_gpu/v2-frozen-44 | 24.42 | 23.61 | 29.42 | 39.17 | 40.94 | 2.097 | 1997.45 | 2010.00 |
| local_gpu/v2-lora-42 | 35.40 | 34.83 | 41.17 | 43.01 | 28.25 | 2.152 | 2002.26 | 2014.00 |
| local_gpu/v2-lora-43 | 35.91 | 34.64 | 45.45 | 59.15 | 27.85 | 2.282 | 2002.26 | 2014.00 |
| local_gpu/v2-lora-44 | 34.24 | 33.57 | 40.14 | 44.21 | 29.21 | 2.248 | 2002.26 | 2014.00 |
| local_gpu/untuned | 160.84 | 134.22 | 306.99 | 497.37 | 6.22 | 3.174 | 2214.23 | 2222.00 |
| local_http/v1-frozen-42 | 27.11 | 25.89 | 33.79 | 45.61 | 36.83 | 3.010 | see GPU run | see GPU run |
| local_http/v1-frozen-43 | 29.90 | 28.36 | 39.68 | 48.13 | 33.40 | 2.257 | see GPU run | see GPU run |
| local_http/v1-frozen-44 | 26.40 | 25.49 | 32.53 | 43.64 | 37.82 | 2.018 | see GPU run | see GPU run |
| local_http/v1-lora-42 | 35.57 | 35.11 | 39.38 | 43.86 | 28.08 | 2.198 | see GPU run | see GPU run |
| local_http/v1-lora-43 | 37.23 | 36.32 | 42.88 | 55.38 | 26.83 | 2.217 | see GPU run | see GPU run |
| local_http/v1-lora-44 | 37.10 | 36.49 | 41.52 | 50.97 | 26.92 | 2.207 | see GPU run | see GPU run |
| local_http/v2-frozen-42 | 26.20 | 25.33 | 31.30 | 37.57 | 38.11 | 1.994 | see GPU run | see GPU run |
| local_http/v2-frozen-43 | 25.43 | 25.02 | 28.67 | 31.52 | 39.26 | 2.032 | see GPU run | see GPU run |
| local_http/v2-frozen-44 | 25.89 | 25.36 | 28.82 | 32.11 | 38.56 | 1.948 | see GPU run | see GPU run |
| local_http/v2-lora-42 | 37.48 | 36.32 | 43.84 | 58.21 | 26.65 | 2.153 | see GPU run | see GPU run |
| local_http/v2-lora-43 | 49.58 | 49.19 | 70.54 | 72.88 | 20.15 | 2.206 | see GPU run | see GPU run |
| local_http/v2-lora-44 | 35.65 | 35.25 | 39.00 | 43.90 | 28.02 | 2.209 | see GPU run | see GPU run |
| local_http/untuned | 153.81 | 124.61 | 295.82 | 445.36 | 6.50 | 5.976 | see GPU run | see GPU run |

## Model-only timing

Classifier forward only; untuned model.generate includes the decoding loop and CPU scheduling. These exclude prompt/tokenization, transfers outside the model call, softmax and parsing. They are not CUDA-kernel-only durations.

| Model | Mean ms | p50 | p95 | p99 | inputs/s |
|---|---:|---:|---:|---:|---:|
| local_gpu/v1-frozen-42 | 26.22 | 22.52 | 44.63 | 59.08 | 38.14 |
| local_gpu/v1-frozen-43 | 22.51 | 21.97 | 26.11 | 28.19 | 44.42 |
| local_gpu/v1-frozen-44 | 30.34 | 29.14 | 43.59 | 45.32 | 32.96 |
| local_gpu/v1-lora-42 | 33.60 | 31.96 | 46.62 | 63.60 | 29.76 |
| local_gpu/v1-lora-43 | 32.57 | 31.64 | 38.17 | 44.44 | 30.70 |
| local_gpu/v1-lora-44 | 34.33 | 32.39 | 44.74 | 57.51 | 29.13 |
| local_gpu/v2-frozen-42 | 21.92 | 21.58 | 24.58 | 26.97 | 45.63 |
| local_gpu/v2-frozen-43 | 21.77 | 21.47 | 24.09 | 26.68 | 45.93 |
| local_gpu/v2-frozen-44 | 22.74 | 21.94 | 27.65 | 37.41 | 43.97 |
| local_gpu/v2-lora-42 | 33.30 | 32.81 | 38.69 | 40.90 | 30.03 |
| local_gpu/v2-lora-43 | 33.86 | 32.54 | 43.29 | 56.96 | 29.54 |
| local_gpu/v2-lora-44 | 32.13 | 31.45 | 37.88 | 41.84 | 31.13 |
| local_gpu/untuned | 159.41 | 132.82 | 305.43 | 495.98 | 6.27 |

Historical source drift is recorded in manifest.json. The old data loader source differs from its historical lock; all non-code historical artifacts were hash-verified and current metrics reproduced cached prediction quality. The new manifest pins the current inference source. No historical locks were edited.

Jev 1.14.0 existing remote HTTP: mean 217.52 ms, p50 203.41, p95 321.96, p99 358.48.

Jev includes remote network overhead; Jais is served locally on the RTX 4050. These are different deployment paths, not a controlled model-speed comparison. Jev had no warmup; local timings use warmup and connection reuse. No Jev requests were rerun. Untuned Jais uses its original zero-shot generative prompt; classifiers use a trained head. Generation has variable output length and can produce INVALID. Cache state, precision, GPU clock/temperature and other processes affect latency. Model loading may benefit from the filesystem cache.

Missing local measurements are **pending**, never inferred from historical batch timings.

## Validation-only representative checkpoints

- v1/frozen: v1-frozen-42, epoch 5, validation loss 0.14952733, locked threshold 0.7881332635879517; `[original-checkout]/results/fine_tuning/09a987d05f6a4db2a7da985c6a3eb4fa/frozen-42/best`
- v1/lora: v1-lora-43, epoch 6, validation loss 0.10865296, locked threshold 0.8631153106689453; `[original-checkout]/results/fine_tuning/c87bd70a25ae40f4a7f301bc2d2b8709/lora-43/best`
- v2/frozen: v2-frozen-43, epoch 5, validation loss 0.22949336, locked threshold 0.9563547372817993; `[original-checkout]/results/fine_tuning_iteration2/51ce252a2cbb4e6e98d487bf2b9f0847/frozen-43/best`
- v2/lora: v2-lora-44, epoch 4, validation loss 0.17111792, locked threshold 0.942931056022644; `[original-checkout]/results/fine_tuning_iteration2/0d227aa2e3954a27837687aef10ddd31/lora-44/best`
