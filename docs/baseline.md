# Initial baseline smoke test

For the existing 200-example dataset, schema, prior Jev methodology, and the
development-only runner option, see [dataset inspection](dataset.md).

Run from the repository root using the main Transformers/PyTorch environment:

```bash
.venv/bin/python scripts/smoke_baseline.py
```

The script loads the original Jais-3-0.9B once and compares two prompt variants on eight hand-authored Arabic-script
examples: MSA, Tunisian and Egyptian Arabic, a short meaningful input, spelling
mistakes, missing spaces, an incomplete phrase, and obvious nonsense. Arabizi
and mixed-script inputs are outside v1 scope and excluded, not labeled gibberish. It prints each input, raw response, parsed label, stop
reason, and token count. It also saves these results and the configuration to
`results/baseline/prompt_comparison.json`, after every example. The previous
`smoke.json` is preserved. This is a diagnostic, not an accuracy benchmark.

Edit `configs/baseline.json` to change reasoning effort or the output token limit.
The model revision is pinned for reproducibility. The model is loaded in FP16 on
CUDA, or FP32 on CPU. CPU inference will be slower. First use needs the model
files and executes its official custom modeling code (`trust_remote_code=True`).
No fine-tuning is performed.

The official Jais chat template opens a reasoning block: `high` uses `think`,
`medium` uses `think_fast`, and `low` uses `think_faster`. We start with low effort
and ordinary greedy generation. This still allows reasoning; we do not manually
close the block or constrain the possible tokens. Only an exact `VALID` or
`GIBBERISH` after the matching closing reasoning tag is accepted. The official
EOS IDs end generation. Truncated or malformed answers count as failures and
remain visible in the output. Inputs exceeding the context budget raise an error
rather than being silently truncated.

Prompt formatting and parsing live in `src/jais_guard/prompts.py`, model loading
and generation in `src/jais_guard/inference.py`, and the small result summary in
`src/jais_guard/metrics.py`. The smoke runner has no additional command-line
options or benchmark machinery. Existing vLLM experiments and `.venv-vllm` are
preserved for later optimization work.

CPU checks, including the official template when cached:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_baseline.py -q
```

The comparison was executed using the existing CPU fallback because CUDA was
unavailable. Results below are smoke-test observations, not a benchmark claim.

## Prompt comparison

`system_only` places classification instructions in the system message and only
raw text in the user message. `explicit_user` retains that same system message
and adds an explicit classification request plus `<text>` delimiters in the user
message. Both variants use the same v1 definition. This tests the combined effect
of the user instruction and delimiters; it does not isolate those two changes.
The earlier run used a broader scope and is not a controlled comparison with
this run.

Model revision, Transformers loading, low reasoning effort, greedy decoding,
256-token output limit, and 1024-token context budget remain unchanged. One model
is loaded for both variants. Each saved result includes the rendered prompt,
raw response, parsed label, input/output token counts, and stop reason. Summaries
separate `incorrect` (a wrong valid label) from `invalid` (no valid label or
truncation); accuracy counts both as failures. An unsupported alphabetic script
raises an explicit error without assigning a classification. This simple scope
check is not a language detector.

## Observed comparison

| Prompt variant | Correct | Invalid output | Incorrect label |
| --- | --- | --- | --- |
| System only | 6/8 | 2 | 0 |
| Explicit user request | 8/8 | 0 | 0 |

Both runs ended normally for every example. System-only prompting answered the
capital question and responded conversationally to the missing-spaces input;
explicit user prompting returned correct labels for both. Only one gibberish
example is included, so these results do not establish general detection quality.
The run used CPU FP32 via the unchanged device fallback; do not attribute
comparisons with earlier GPU runs solely to the new prompt. The two variants
within this run used identical inference settings.
