# Existing Arabic gibberish evaluation data

Reuse `data/gibberish.jsonl`; do not regenerate, shuffle, or reassign its splits.
The file contains 200 Arabic-script examples: 100 meaningful and 100 gibberish.
All IDs and exact texts are unique; the loader also checks duplicates after
collapsing whitespace. No non-Arabic alphabetic characters were found.

## Authoritative labeling policy

The following Arabic definition, also used for the previous Jev evaluation, is
binding for dataset annotation, baseline classification, manual review, and
future fine-tuning:

الكلام غير المفهوم هو نص لا يمكن أن يُستخلص منه أي سؤال أو موضوع أو طلب ذي معنى، مثل: حروف عشوائية أو مكررة، أو ضغط عشوائي على لوحة المفاتيح، أو كلمات مقلوبة أو مبعثرة الحروف، أو كلمات مخترعة تشبه العربية في شكلها فقط، أو سلسلة من كلمات حقيقية لا تحمل معنى مترابطًا.

أما ما يلي فليس كلامًا غير مفهوم: المواضيع القصيرة جدًا أو الكلمة المفتاحية الواحدة، وأي لهجة عربية (مصرية أو شامية أو خليجية أو غيرها)، والأخطاء الإملائية والكتابة غير الرسمية، والطلب الواضح الذي يتخلله قليل من التشويش، والطلبات ذات المعنى حتى لو كانت خارج نطاق هذا التطبيق أو لا يُتوقع أن تُرجع نتائج.

Map meaningful text to `VALID` and gibberish to `GIBBERISH`. v1 supports only
Arabic-script inputs, including regional dialects. Arabizi and mixed-script
inputs are unsupported, not automatically gibberish. Spelling mistakes, missing
spaces, incomplete sentences, or short phrases do not alone justify GIBBERISH
when a meaningful question, topic, or request can be recovered.

Preserve the existing labels and splits. Review questionable examples against
this policy and their annotation rationale; record concerns without automatic
relabeling. Review hints are prompts for human inspection, not alternative rules
that override this definition.

`src/jais_guard/prompts.py` stores this definition verbatim as `DEFINITION`.
Both selectable templates (`system_only` and `explicit_user`) share it through
the system instruction. The explicit-user template additionally requests
classification around the supplied text. Both request only `VALID` or
`GIBBERISH`; unexpected output remains an evaluation failure. Ordinary generation
cannot guarantee compliance with the output instruction. Reasoning and decoding
settings are unchanged. Earlier saved results used the previous prompt and are
not results for this definition update; evaluation has not been rerun.

## JSONL schema

Each line is an object with these required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable ID, e.g. `FTW-N-001` |
| `input_text` | string | Original input, preserved verbatim |
| `is_gibberish` | boolean | `false` maps to `VALID`; `true` to `GIBBERISH` |
| `split` | string | Existing `dev` or `test` assignment |
| `subtype` | string | Input category below |
| `dialect` | string | MSA, Egyptian, Levantine, Gulf, or `none` |
| `query_type` | string | `fatwa_search` or `khutbah_request` |
| `difficulty` | string | easy (68), medium (82), hard (50) overall |
| `label_rationale` | string | Original annotation explanation |

The loader retains all source fields and adds `text` (an alias for `input_text`),
`label`, and `review_reason`. It rejects missing/invalid required fields,
non-boolean labels, unknown splits, duplicate IDs/texts, and split counts other
than dev=40/test=160. It does not enforce a closed subtype/dialect vocabulary or
correct labels automatically. Validation applies to the entire file before
selecting a split, so cross-split duplicates cannot be missed.

## Splits and coverage

| Property | Dev | Test |
| --- | --- | --- |
| Examples | 40 | 160 |
| VALID / GIBBERISH | 20 / 20 | 80 / 80 |
| Fatwa / khutbah query types | 20 / 20 | 80 / 80 |
| MSA | 18 | 61 |
| Egyptian | 2 | 14 |
| Levantine | 2 | 9 |
| Gulf | 2 | 11 |
| Dialect `none` | 16 | 65 |

`none` is a source annotation, not a missing value. There are no explicitly
annotated Tunisian/Maghrebi examples, so this dataset cannot establish coverage
of those dialects. It is concentrated on two religious application contexts,
not general Arabic traffic. `out_of_scope` below means outside those applications,
not outside JaisGuard's supported scripts; these meaningful inputs remain VALID.

| Subtype | Label | Dev | Test |
| --- | --- | --- | --- |
| direct_question | VALID | 5 | 19 |
| keyword_search | VALID | 2 | 12 |
| khutbah_request | VALID | 5 | 27 |
| informal_spelling | VALID | 1 | 9 |
| typo | VALID | 1 | 3 |
| short_topic | VALID | 2 | 4 |
| noisy_but_meaningful | VALID | 2 | 2 |
| out_of_scope | VALID | 2 | 4 |
| keyboard_pattern | GIBBERISH | 5 | 18 |
| random_chars | GIBBERISH | 5 | 16 |
| scrambled_words | GIBBERISH | 2 | 12 |
| arabic_like_nonsense | GIBBERISH | 4 | 19 |
| incoherent_sentence | GIBBERISH | 4 | 15 |

## Manual review

The loader attaches conservative subtype-based hints for scrambled words,
Arabic-like nonsense, incoherent sentences, short topics, and meaningful noisy
inputs. These are review candidates, not proven annotation mistakes. Other
examples may also need review. No candidates are removed or relabeled.

For example, dev `FTW-G-003` contains reversed words whose intended meaning may
be recoverable. Dev `FTW-G-005` has real words in an implausible question: reviewers
should distinguish unintelligibility from creative or unusual meaning. Invented
word judgments need dialect knowledge. Short and noisy VALID examples protect
against rejecting legitimate queries merely for their form. Compare each input
with its original rationale and record any future adjudication separately.

Dataset authorship, generation process, and licensing are not recorded in the
provided artifacts. Keep the originals; establish provenance before publication.

## Previous Jev evaluation archive

`data/jev_results.zip` contains 200 detailed JSONL records, a flattened latest CSV,
JSON/Markdown summaries, threshold selection/sweep CSVs, and Plotly HTML charts.
The JSONL adds app/prompt identifiers, prompt hash/language, requested and returned
model, threshold, true/predicted booleans, gibberish score, correctness, request ID,
input/output tokens, latency, error fields, and UTC timestamp. It does not retain
split or annotation rationale; join by ID to the source dataset. Every detailed
record matches this dataset's ID, text, and truth label.

The September 30, 2026 report used `jev-1.13.0` and two Arabic, domain-specific
prompts (`fatwa_ar` and `khateeb_ar`), one per application. Prompts define
recoverable meaning and explicitly preserve dialects, typos, short topics,
noisy meaningful requests, and meaningful requests outside application scope.
The decision rule was score >= 0.5, with gibberish as the positive class.
All 200 calls returned results, with no recorded request errors.

The original summary explicitly ignored dev/test separation and pooled all 200:
accuracy 82%, gibberish recall 64%, precision 100%, valid-query false rejection
0% (TP=64, FN=36, FP=0, TN=100). Easy character/keyboard noise was detected well;
Arabic-like nonsense and incoherent sentences had only 30.4% and 31.6% accuracy.
These are historical results, not Jais measurements.

A later `threshold_selection.csv` labels shared 0.163 and per-app thresholds
(fatwa 0.163, khateeb 0.158) as dev-selected. On 160 test examples, it reports:

| Threshold | Accuracy | Gibberish recall | Valid false rejection |
| --- | --- | --- | --- |
| Default 0.5 | 82.5% | 65% | 0% |
| Shared dev-selected 0.163 | 97.5% | 96.25% | 1.25% |

No evaluator source or precise threshold-selection objective/tie rule is included,
so the dev-selection procedure cannot be independently reconstructed from code.
The archive also contains threshold sweeps and charts using all splits. Test
results have therefore already been inspected historically; preserve the split
now, but do not describe it as a never-observed holdout or tune Jais on it.
The archive's reported p50/p95 latency is 270.7/319.8 ms and total cost $0.005907.
Hardware, network timing boundaries, pricing calculation, and concurrency are not
specified; these are not directly comparable to local Transformers latency/cost.
Jev's domain-specific prompts and adjustable score threshold also differ from
Jais's current generic label-generation prompt.

## Inspect and validate without loading a model

From the repository root:

```bash
.venv/bin/python -m src.jais_guard.data
```

This prints split/category counts and development review candidates with their
original labels and rationales. It validates test records but runs no inference
and does not print individual test inputs.

The current runner can use only the existing 40 dev examples:

```bash
.venv/bin/python scripts/smoke_baseline.py --dev
```

This runs the existing two prompt variants unchanged and saves metadata and
per-example outputs to `results/baseline/dev_prompt_comparison.json`. Without
`--dev`, it still uses the small smoke set. There is no test-set inference option.
Neither dev nor test inference was run as part of the loader implementation.
