# Figure sources and regeneration

The final README uses the owner's supplied assets from `data/plots`, copied without
redrawing or editing SVG content. Descriptive filenames below distinguish them from
the earlier figures. SVG is the primary format; adjacent PNG files are optional
alternatives. All supplied archive members, including original figures, scripts,
metrics and provenance notes, are preserved under `sources/`.

| Integrated figure | Original source | Placement |
| --- | --- | --- |
| [architecture-simple.svg](architecture-simple.svg) | `data/plots/jaisguard_simple.svg` | README architecture |
| [classification-quality.svg](classification-quality.svg) | `jaisguard_benchmark_charts.zip` → `classification_quality.svg` | README quality |
| [inference-latency.svg](inference-latency.svg) | Same archive → `inference_latency.svg` | README latency |
| [transformer-architecture.svg](transformer-architecture.svg) | `jaisguard_architecture_bundle.zip` → `jaisguard_architecture.svg` | [Detailed architecture](../architecture.md) |
| [serving-throughput.svg](serving-throughput.svg) | Benchmark archive → `serving_throughput.svg` | [Serving overview](../vllm-serving.md#throughput-overview) |

The supplied benchmark charts are vertical bar charts and retain their original
colors and full-precision quality labels. This integration follows the owner's
latest asset selection; no chart orientation, values or labels were changed.

## Benchmark evidence

[Supplied metrics](sources/benchmarks/jaisguard_readme_metrics.json) and
[original provenance notes](sources/benchmarks/metrics_provenance.md) accompany the
charts. These are presentation inputs; the experimental reports remain authoritative.

| Figure / table | Verified evidence | Metric and conditions |
| --- | --- | --- |
| Quality | [Public aggregates](../benchmarks/quality-latency/aggregates.json), [checkpoint report](../model-publication-preparation.md#metrics-individual-checkpoints-versus-experiments) | Macro F1: untuned 77.04545454545455%, frozen 81.2297077922078%, LoRA 92.24413462681152%, Jev 58.0179466327474%; individual v2 checkpoints, locked thresholds, BF16 batch 2; original zero-shot baseline protocols |
| Latency | [Comparison CSV](../benchmarks/quality-latency/comparison.csv), [protocol](../benchmarks/quality-latency/comparison.md) | Mean ms/input: untuned 160.84, frozen 23.50, LoRA 34.24, Jev 217.52; local GPU batch 1 versus hosted API |
| Standalone memory table | [Timing/memory CSV](../benchmarks/quality-latency/latency.csv) | CUDA allocated peaks: untuned 2,214.23 MiB, frozen 1,997.45 MiB, LoRA 2,002.26 MiB; Jev unavailable |
| Throughput | [Serving CSV](../benchmarks/vllm/comparison.csv), [protocol/parity](../benchmarks/vllm/comparison.md) | FP16 LoRA seed 44, concurrency 16: HF 26.13 vs vLLM 236.54 requests/s; peak allocated 2,478.36 vs 4,515.01 MiB |
| Architecture | [Implementation](../../src/jais_guard/finetuning/model.py), [configuration](../../configs/finetuning-iteration2.json), [supplied notes](sources/architecture/architecture_notes.md) | 28 layers, hidden 1,536, last non-padding pooling, FP32 head, rank-8 q/v LoRA |

Quality uses the same 400 historical, previously examined constructed inputs.
The frozen and LoRA checkpoints are validation-selected v2 seeds 43 and 44, not
three-seed means. Local latency includes tokenization/transfers, three warmups,
no HTTP and excludes loading; Jev includes networking with no warmup. Local GPU:
RTX 4050 Laptop, 6 GB. These are different deployment paths, not a controlled
same-hardware speed comparison. Standalone quality/latency and FP16 HTTP serving
have separate precision and batching conditions.

Values in the supplied JSON were checked against original local checkpoint reports
and tracked benchmark CSVs. Memory converts bytes to MiB using 1,048,576 bytes/MiB.
The 9.05× ratio comes from original unrounded throughput; rounding displayed rates
before dividing gives a slightly different ratio. No model, threshold or benchmark
artifact was modified.

## Reproduction

The supplied scripts specify **Python 3.12 and Matplotlib 3.10.8**. Use a separate
plotting environment; these are not the repository's Python 3.11 runtime pins.
Scripts are preserved verbatim and were inspected, not executed during integration.
Rendering versions and SVG metadata can change bytes on regeneration.

From the repository root, after installing the plotting dependencies:

```bash
python docs/assets/sources/benchmarks/plot_benchmarks.py \
  docs/assets/sources/benchmarks/jaisguard_readme_metrics.json \
  --out /tmp/jaisguard-benchmark-figures
python docs/assets/sources/architecture/draw_architecture.py
```

The benchmark command writes the three original underscore-named SVG/PNG pairs and
provenance notes into the requested directory. The architecture command writes
`jaisguard_architecture.svg` and PNG beside its script. Compare outputs before
copying them to the descriptive integrated filenames.

**Simplified architecture:** no separate generation script was supplied for
`jaisguard_simple.svg`. The editable SVG is its source; the accompanying PNG is
preserved. The detailed architecture script does not generate the simplified figure.

The prior [standard-library generator](generate_readme.py), its
[metric extract](readme-metrics.json), and earlier [architecture](architecture.svg),
[quality](quality.svg), and [latency](latency.svg) figures remain preserved.
That generator produces only those earlier filenames, **not the current supplied
README figures**.

## Integration checks

All copied SVGs and PNGs match their supplied source bytes. The SVGs were parsed,
rasterized with librsvg and visually inspected for clipping and readable labels.
Exactly three content figures are embedded in the README; the detailed architecture
and throughput figures appear in dedicated documentation. Local Markdown links,
README heading anchors, GFM tables and benchmark allowlist hashes were checked.
The README retains the model revisions and previously tested inference command;
this visual/documentation update does not rerun GPU inference or benchmark models.

Anonymous Hub API verification on 2026-10-31 confirmed both public model revisions:
LoRA `74977da942572cf15e0cbf6c815a343b1bd61af1`, frozen
`169fe462f83667ddc9af547ebeea538d0a0b9602`. The dataset returned HTTP 401;
the README retains the authorized-access disclosure. The local link checker passed
138 targets. Ruff passed with `--exclude docs/assets/sources` to preserve supplied
third-party plotting scripts verbatim; repository runtime code is unchanged.
