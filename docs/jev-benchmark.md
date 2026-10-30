# Jev versus JaisGuard: opt-in API benchmark

This comparison uses the immutable, human-reviewed, constructed Arabic-input benchmark
`data/exports/test-test-400-v1` (400 inputs: 188 VALID, 212 GIBBERISH). It was already
examined during JaisGuard development. Results are exploratory regression/head-to-head
evidence, **not an unbiased final evaluation or a sample of production traffic**.
Jev and untuned Jais are zero-shot; JaisGuard was fine-tuned specifically for this task.
No prompts, operating thresholds, datasets or models are selected using these API results.

## Official contract and fixed protocol

The [official System One API](https://docs.typesafe.ai/api/system-one) accepts
`POST https://api.typesafe.ai/v1/systemone`, bearer authentication, `model`, string
`state`, and a `questions` object. This runner uses one `noul` question named
`recoverable`; the corresponding response is `answers.recoverable.noul`, a yes
probability in [0, 1]. The response also supplies the model ID and input/output
usage. Only the raw input is sent as state: benchmark IDs, labels and provenance
are never included in API requests.

`configs/jev_benchmark.json` contains the complete fixed question,
`arabic-recoverability-noul-v1`. It treats short, incomplete, informal, dialectal and
misspelled Arabic as meaningful whenever reasonable interpretation is recoverable.
The fixed rule is **yes probability >= 0.5 → VALID**, including an exact tie.
No threshold tuning or sweep is implemented.

As verified on 2026-10-30, the [official model list and pricing](https://docs.typesafe.ai/models)
document `jev-latest` as an alias for `jev-1.14.0`, accept pinned version IDs even
when omitted from the models-list API, and list $0.042 per million input tokens
with free output tokens. Smoke the alias first and inspect its returned version.
Use the documented pinned version for the full benchmark. If the alias changes,
verify the new version's availability and pricing before a new benchmark; do not
silently replace an established run's model or pricing snapshot.

The response schema, probability, usage integer types and model ID are validated.
Unexpected schema/model changes stop execution. Returned alias versions are
recorded and any detected drift stops further work. Alias use for a full run
requires `--allow-alias` and remains less reproducible than a pinned version.

## WSL commands

Run from the repository root using the existing uv environment. No CUDA is needed.
httpx is already declared in the project dependencies. `--no-sync` preserves the
installed environment. Enter credentials without putting their value in shell history:

```bash
read -rs -p 'TypeSafe API key: ' TYPESAFE_API_KEY
export TYPESAFE_API_KEY
```

Smoke uses only your manually supplied inputs, never benchmark labels. This is a
small **paid** API call batch; the flag is deliberately explicit:

```bash
uv run --no-sync python scripts/benchmark_jev.py smoke \
  --run-dir results/jev/smoke-v1 \
  --model jev-latest \
  --text 'كيف نبدل كلمة السر متاع الحساب؟' \
  --text 'عايز اعرف مواعيد القطار بكرة' \
  --text 'ضصث ضضض صثصث' \
  --execute-paid
```

Inspect the printed `model` in the stored responses. The following commands use
`jev-1.14.0`, the pinned version documented when this integration was implemented.
Dry-run verifies frozen dataset integrity, identical cached inputs/labels, previous
parser/metric compatibility and validation-locked classifier thresholds. It does
not read the API key, create a result directory, or make API calls:

```bash
uv run --no-sync python scripts/benchmark_jev.py evaluate \
  --run-dir results/jev/test400-jev1140-v1 \
  --model jev-1.14.0 --dry-run
```

Explicitly execute all 400 requests, sequentially by default:

```bash
uv run --no-sync python scripts/benchmark_jev.py evaluate \
  --run-dir results/jev/test400-jev1140-v1 \
  --model jev-1.14.0 --execute-paid
```

Repeat **the identical command** to resume. Already committed successful responses
are never sent again. Add `--retry-failed` only to explicitly retry terminal failures.
After a process interruption, requests left in flight are marked uncertain and
are not automatically resent. `--retry-uncertain` explicitly accepts that their
server-side outcome/billing may be unknown and they could be repeated. There is
no documented idempotency contract guaranteeing exactly-once billing across a
crash between the server response and the local commit.

Optional concurrency: add `--concurrency 2` **on the first execution** and on every
resume. Execution settings, question, model, data hashes and cached result hashes
are locked in the run manifest. Changing them requires a new versioned directory.
A process lock prevents simultaneous writers. Do not delete the checkpoint database.

Generate the comparison offline, with **no API calls or credential requirement**:

```bash
uv run --no-sync python scripts/benchmark_jev.py report \
  --run-dir results/jev/test400-jev1140-v1
```

Unfinished/uncertain inputs block reporting by default; `--allow-incomplete` emits
an explicitly incomplete report retaining those inputs as INVALID. Completed
terminal failures are also INVALID, and completion/reliability are always reported.
`evaluate` exits 2 unless all 400 succeed; this does not delete checkpoints.

## Outputs, retry policy and interpretation

Each run lives separately under `results/jev/`:

- `manifest.json`: fixed question, model, threshold policy, execution settings,
  dataset and historical comparison hashes, verified pricing snapshot.
- `requests.sqlite3`: durable WAL/FULL checkpoints, inputs, statuses, safe response
  fields and per-attempt timing/status/usage. Credentials, headers, raw error
  bodies and arbitrary response fields are not persisted. This database contains
  benchmark text and should be treated as evaluation data.
- `reports/<new timestamp-id>/report.md`, `metrics.json`, `predictions.csv`,
  `predictions.jsonl`, `errors.jsonl`: each report gets a fresh directory.

Default retries: three additional attempts after an initial request, exponential
backoff starting at one second, bounded at 30 seconds. HTTP 408/429/500/502/503/504/529
and transport failures are retryable; permanent authorization/request failures
stop the queue. `Retry-After` and `retry-after-ms` are respected. If a server asks
for a delay beyond the bound, execution pauses instead of retrying too early.
See [official errors](https://docs.typesafe.ai/api/errors) and
[documented retry guidance](https://docs.typesafe.ai/sdk/python/api/retries).
Timeout/backoff/retry counts and concurrency are CLI settings locked per run.
Transport timeouts can have unknown server-side billing even when retried.
No proxies, redirects, SDK autologging or HTTP debug logging are enabled.

All 400 inputs remain in the metric denominator. Failed, missing or malformed
responses are INVALID (-1), wrong for accuracy and true-class recall/F1. Confusion
rows are VALID/GIBBERISH; columns are VALID/GIBBERISH/INVALID. VALID false rejection
counts explicit GIBBERISH predictions only. VALID acceptance counts explicit VALID
predictions only; therefore they need not sum to one when INVALID outputs exist.
The invalid VALID count and overall response success rate expose this gap.
This is compatible with the existing untuned Jais invalid-output metric convention.

Latency includes client-observed HTTP request time, network, service queueing and
TLS/connection overhead, without hidden warm-up calls. Mean, p50/median, p95 and
p99 for successful requests, all completed attempts, and last-invocation per-input
time are recorded separately. The latter includes retries/backoff during that
invocation; it does not include time spent between resume invocations. Cached
JaisGuard GPU timings have different boundaries; they are displayed separately.

HTTP error rate has explicit denominators: HTTP responses and journaled attempts.
Journaled attempts include in-flight requests whose delivery/outcome is uncertain.
Retry totals include resumes; error responses and transport failures are separate.
Token usage sums every attempt with reported usage, including retries. Unknown
usage is counted, not guessed. Cost is a documentation-based estimate, not a bill:
known subtotal is reported; total is unknown if any attempt lacks verified model
pricing or usage. Smoke costs are separate from the 400-example run. Taxes,
account-specific terms and unreported failure billing are not inferred.

The existing completed comparison at
`results/version_comparison/jaisguard-v1-v2-v1/benchmark` is reused read-only.
All twelve classifiers' two operating points and the untuned baseline are checked
against stored predictions and the same frozen text, IDs, labels and provenance.
Fine-tuned summaries use three seeds and sample standard deviations; baseline
and Jev are single runs. Confusion aggregates distinguish means from repeated-seed
sums. Existing checkpoints, dataset versions, MLflow experiments and historical
results are never overwritten or rerun. No new MLflow infrastructure is required.

## CPU verification

```bash
uv run --no-sync pytest -q tests/test_jev.py
uv run --no-sync ruff check --target-version py311 \
  src/jais_guard/jev.py src/jais_guard/jev_reporting.py \
  scripts/benchmark_jev.py tests/test_jev.py
```

All HTTP behavior in tests uses `httpx.MockTransport`; no paid calls are made.
The API contract has been checked against documentation, not live service access.
Real permissions, quota, service behavior and billing require your explicit smoke.
