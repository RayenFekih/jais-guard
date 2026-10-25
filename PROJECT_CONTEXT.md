# JaisGuard project context

JaisGuard investigates whether the Arabic-centric `IFM/Jais-3-0.9B` model can
be adapted into an accurate, efficient Arabic gibberish classifier. Preserve
meaningful Arabic-script inputs, including Modern Standard Arabic, regional
dialects, spelling mistakes, missing spaces, short phrases, and incomplete
sentences. Arabizi and mixed-script inputs are out of scope for v1. Unsupported
scripts must not be assigned GIBBERISH by default; report them as unsupported
or exclude them explicitly from an in-scope experiment.

## Authoritative task definition

Use the exact Arabic labeling policy in `docs/dataset.md` (Authoritative labeling
policy), shared by the previous Jev evaluation, for baseline prompts, dataset
review, and future fine-tuning. Its verbatim prompt constant is `DEFINITION` in
`src/jais_guard/prompts.py`. Do not replace it with a broader or stricter notion
of grammatical correctness, application relevance, or linguistic fluency.
Meaningful Arabic inputs remain VALID despite spelling errors, missing spaces,
incomplete sentences, or brevity. Preserve existing labels; flag questionable
examples for manual review without automatically relabeling them. Retain
selectable prompt variants and the exact output labels VALID and GIBBERISH.

## Development strategy

Use Hugging Face Transformers and PyTorch in the main `.venv` environment for:

- Model and tokenizer inspection.
- Baseline evaluation of the original model.
- Later fine-tuning experiments.
- Later custom classification heads and direct-logit inference.

Reserve the separate `.venv-vllm` environment for later inference optimization,
throughput benchmarking, and production serving experiments. Preserve the
existing working vLLM scripts and environment. Their presence does not make vLLM
the backend for the current baseline work. Use the main environment for the current smoke test; existing standalone vLLM
scripts remain available for later experiments.

Use the official model card, configuration, and modeling implementation as the
source of truth for Jais's custom K2 Horizon architecture. Do not assume standard
Llama or Qwen internals. Inspect installed dependencies and existing code before
changing them. Account for the local GPU's 6 GB VRAM limit.

## Learning progress and current milestone

The introductory study of tokenization, chat templates, prefill, decoding, and
KV caching is complete.

The current milestone is baseline Arabic gibberish classification using the
original Jais-3-0.9B model, without fine-tuning. Use the labels `VALID` and
`GIBBERISH`. Establish and inspect the untuned model's behavior before moving to
training, architecture changes, or inference optimization. Keep invalid or
unparsable outputs visible and count them as evaluation failures.

For the current prompt experiment, compare system-only classification instructions
with an explicit classification request in the user message, using identical
in-scope examples and unchanged model, reasoning, and decoding settings. Preserve
raw outputs, parsed predictions, token counts, and per-example results. Report
invalid outputs separately from incorrect classification labels.

Reuse `data/gibberish.jsonl` for Phase 1, preserving the existing 40-example dev
and 160-example test splits and source annotations. Use dev for current prompt
experiments; do not run test inference until explicitly requested. Review hints
must not automatically change labels. See `docs/dataset.md` for the schema and
the historical Jev benchmark's methodological limits.

Keep model inference, prompting, and evaluation metrics separate so later
experiments can reuse the evaluation framework. Preserve configurations, raw
model outputs, and relevant intermediate representations or measurements to make
results inspectable and reproducible. Do not claim accuracy, latency, or cost
improvements without supporting measurements.

## Learning-oriented workflow

This project is also a hands-on LLM engineering learning exercise.

- Make small, focused changes, one understandable step at a time.
- Start with the smallest useful implementation. Avoid extra scripts, elaborate
  command-line interfaces, and abstractions until a concrete need arises.
- Explain non-obvious implementation decisions and their trade-offs.
- Keep intermediate outputs inspectable so model behavior can be studied.
- Reuse existing code where appropriate; avoid unnecessary abstractions and
  dependencies.
- Prioritize correctness and reproducibility before performance optimization.
- When preparing commits, provide separate logical commits as needed, using only
  `git add` and `git commit -m` commands. Use subtle engineering humour in commit
  messages, and leave committing to the user unless explicitly asked to do it.
- Do not implement future milestones prematurely. Fine-tuning, custom heads,
  optimization, serving, and deployment require subsequent scoped work.

Implement only the work requested for the current step; this context does not
authorize future milestones on its own.
