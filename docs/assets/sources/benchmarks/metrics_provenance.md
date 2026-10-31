# JaisGuard benchmark provenance

Source: `jaisguard_readme_metrics.json`. SHA-256: `e20cb852f360af90abf94da138914e54bfddc6c72e11ce2402e0e0a839f2ff39`.

All bar heights use JSON values directly. F1 labels retain all supplied digits; latency and throughput retain two decimal places. All axes start at zero. No uncertainty estimates were supplied, so no error bars are plotted.

## Classification quality

400 historical constructed and previously examined inputs (188 VALID, 212 GIBBERISH); same inputs across approaches, different inference protocols

- Untuned Jais 3 0.9B: Macro F1 77.04545454545455%; zero-shot generative prompt; 30 INVALID generated outputs counted as incorrect.

- Frozen Backbone + Head: Macro F1 81.2297077922078%; v2 seed 43; BF16 batch 2; validation-locked threshold 0.9563547372817993.

- LoRA + Head: Macro F1 92.24413462681152%; v2 seed 44; BF16 batch 2; validation-locked threshold 0.942931056022644.

- Jev 1.14.0: Macro F1 58.0179466327474%; remote hosted zero-shot API.

## Inference latency

400 inputs; local RTX 4050 Laptop GPU 6GB; local batch 1 with tokenization and transfers, 3 warmups, no HTTP; remote Jev API includes network, no warmup; loading excluded

- Untuned Jais 3 0.9B: 160.84 ms; local GPU generative baseline.

- Frozen Backbone + Head: 23.50 ms; local GPU classifier; v2 seed 43.

- LoRA + Head: 34.24 ms; local GPU classifier; v2 seed 44.

- Jev 1.14.0: 217.52 ms; hosted remote API.

## Optimized serving throughput

Historical FP16 LoRA seed 44 full-backbone exports; 400 warm local HTTP requests per backend; concurrency 16; one input per request; RTX 4050 Laptop GPU 6GB; HF serialized vs vLLM batched

- HF / FastAPI: 26.13 requests/s; peak allocated GPU memory 2478.36 MiB.

- vLLM 0.32.0: 236.54 requests/s; peak allocated GPU memory 4515.01 MiB.

Reported throughput improvement: 9.05×, as supplied in JSON. The ratio of the displayed rounded rates is approximately 9.05243×.

Native vLLM used historical full-backbone exports, not the compact published HF packages. Peak allocated GPU memory is CUDA allocator peak, not total board usage.

Quality/latency and serving experiments differ in precision and batching. Serving results do not describe native vLLM compatibility of the compact published HF packages.

## Reproduction

Python 3.12; Matplotlib 3.10.8. Install: `python -m pip install matplotlib==3.10.8`.

Run: `python plot_benchmarks.py jaisguard_readme_metrics.json --out figures`.

PNG: 3840 × 2220 pixels, 300 dpi. SVG: vector paths, including text, for portable typography.
