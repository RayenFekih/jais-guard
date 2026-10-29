# JaisGuard: qualitative audit of the 76-candidate draft

**29 October 2026 — desk audit and proposal only. No annotations or protocol changes.**

## Conclusion

The current draft is a useful **constructed diagnostic reservoir**, but is not yet
a credible representative sample for the general-purpose 400-example test. Its
lexical separation from training/validation does not establish naturalness,
mechanism realism, independent authorship or representative dialect coverage.

The most important correction to the earlier collection report is that **human-origin
text is not equivalent to observed natural input**. Twelve rows are verbatim
curated dictionary examples, 32 are assistant-authored and 32 are Python-generated
or transformed. **Zero rows are documented as observed user interactions or actual
input failures.** That is a provenance finding, not a claim that none could occur
in practice. All 32 proposed GIBBERISH cases are constructed.

A broad, explicitly subjective surface rubric identifies:

| Audit bucket | Count | Interpretation |
|---|---:|---|
| Plausible ordinary-looking prototypes | 31 | 16 assistant-written VALID, 7 less-marked dictionary VALID, 8 lightly altered dictionary VALID |
| Dictionary/editorially marked surface | 9 | 5 public and 4 altered VALID rows with at least 10 combining marks |
| Deliberately constructed stress cases | 36 | All 32 proposed GIBBERISH plus 4 severely scrambled proposed VALID |
| **Total** | **76** | No bucket constitutes verified production representativeness |

The 31 plausible prototypes can guide future acquisition, but are not 31 authenticated
natural messages. The nine marked examples can be perfectly meaningful Arabic;
their typography does not justify a GIBBERISH label or automatic exclusion.

**Do not simply add another 324 similarly constructed rows.** Before collecting,
review the general-test eligibility criteria and separate the diagnostic purpose.
No existing candidate is moved, excluded, relabeled or allocated by this report.

## Inputs and method

Read: the complete `draft.jsonl`, collection report, blind preview, semantic-review
flags and selected source-parent records; inspected the construction recipes.
The preview's ID→text mapping exactly matches the draft. Current proposals remain
44 VALID / 32 GIBBERISH, with all 76 pending human review.

This is not a blind annotation experiment: this audit has seen proposed labels,
source metadata and parents. It cannot measure annotator agreement or establish
input-only recoverability. Recommendations below need human review. No Jais or
other classifier was invoked, fitted or used to choose cases. Computed figures are
descriptive string/Unicode counts only. No model performance is estimated.

Per-candidate recommendations for **every ID** are in
[`candidate_audit.csv`](../results/dataset_audits/test400-qualitative-20261029/candidate_audit.csv)
and the matching JSON. Computed counts and ID lists are in
[`audit_statistics.json`](../results/dataset_audits/test400-qualitative-20261029/audit_statistics.json).
Those files are advisory audit outputs, not annotation imports.

## 1. Naturalness and VALID diversity

### What works

There are plausible service requests, short reports, colloquial constructions,
Arabic-script loanwords and recoverable imperfections. Examples include:

| ID | Concrete evidence | Value and qualification |
|---|---|---|
| `T-1d463afe8d34` | `هو ينفع أستلم النظارة من فرع تاني؟` | Naturalistic Egyptian request; still assistant-authored |
| `T-c31b66c60b8e` | `الكونترا بعثوهالي مصوّرة ومقصوص منها آخر سطر` | Plausible Tunisian report and loanword; not an observed message |
| `T-c727f791e425` | `عم جرّب أرفع المخطط بس الأبعاد عم تنقص كل مرة` | Progressive colloquial report, distinct from a complete formal question |
| `T-8f0321b03132` | `وش أسوي إذا السجل مربوط برقم جوال قديم؟` | Gulf-compatible request, plausible administrative problem |
| `T-d2d05aad047c` | `تعديل اسم الراكب بعد إصدار التذكرة` | Meaningful incomplete topic; incompleteness is not gibberish |
| `T-7360d8b16950` | `لم استلم رمز التاكيد رغم ان الرقم مسجل باسمي` | Common omitted hamzas; meaningful without editorial correction |
| `T-2a43acb62f3d` | Unusual `نسهلك`, followed by `كيفاش نمشي للمرسى` | The route-seeking intent survives the transcription oddity |

### Coverage is too narrow

- **All 44 VALID rows have at least five whitespace-separated words** (median 7,
  maximum 13). There are no one-word topics, greetings, very short fragments or
  one-to-four-word inputs. Those are important false-rejection cases under the task definition.
- Normalized character lengths are 18–60. No rows cover the 1–15 or 101+ bins.
  Similar class medians do not make the tails representative.
- The 16 assistant VALID rows are predominantly neat service/administrative
  requests and reports. They contain multiple topics, but mostly one interaction genre.
  Social chat, non-request assertions, humor, personal narrative, informal follow-ups,
  regional voice and organically incomplete writing are underrepresented.
- Twenty-eight VALID rows carry the generic `everyday` domain from one dictionary
  source or its derivatives. This metadata count is not evidence of 28 independent topics,
  speakers or message types. Other domain labels often have only one or two cases.
- The 12 key-repeat rows all inject **one character at the midpoint of the alphabetic
  character sequence**. These are not 12 distinct naturally occurring typo mechanisms.
  They omit transpositions, neighboring-key substitutions, organically variable spacing,
  missing punctuation and realistic combinations of errors.
- Orthography mixes polished MSA, fully marked dialect dictionary spellings and
  controlled errors. This is useful for diagnostics but not evidence of a natural spelling distribution.
- The recorded difficulty categories are authoring hints. For example, the clear
  telecom fragment `T-8dc286083a30` and omitted-hamza sentence `T-7360d8b16950`
  need not be difficult for an ordinary Arabic reader. Do not treat 15 VALID cases
  marked “hard” as validated difficulty coverage.

**Collection action:** obtain genuine input snippets and preserve their original
writing, including meaningful short topics and unfinished requests. Use current
fluent prototypes to define needed situations, not to regenerate template variants.
Latin-script code-switching remains intentionally outside this benchmark; its
absence is not an in-scope coverage defect. Arabic-script borrowed vocabulary stays in scope.

## 2. GIBBERISH mechanisms and realism

The six subtype names reduce to approximately **four construction families**, not
six independently evidenced real failure mechanisms:

| Construction family | Count | Evidence and issue |
|---|---:|---|
| Whole-message keyboard substitution | 8 | One fixed mapping; no recorded physical key trace or observed incident |
| Seeded chunked noise | 8 | Four “pseudoword” and four “random character” rows use exactly the same generator |
| Assistant-created Arabic-like words/fragments | 12 | Eight invented-morphology and four “garbled fragment” cases; the latter lack an observed corruption parent |
| Function-word sequences | 4 | Authored particle lists, not demonstrated accidental inputs; semantic policy is ambiguous |

There are useful Arabic-shaped challenges, but **none of these 32 is documented
as a naturally observed failure**. That does not make every proposed label wrong;
it makes representativeness and mechanism-frequency claims unsupported.

Specific problems:

- `T-3a6cef709830` (“pseudoword”) and `T-78400b952fa6` (“random”) both come from
  `test400-chunked-noise-v1`. The builder cycles chunk lengths **5, 6, 7**, then crops
  the string. Different seeds or subtype names do not create a new mechanism.
  The easy/medium distinction here also follows the generation branch rather than reader validation.
- The keyboard mapper rotates each artificial key row with wraparound and special
  mappings for alif variants. It is not established as the result of a common real
  keyboard/layout failure. Keep it described as a simulation unless independently validated.
- `T-0585c1e67ff0` contains `تٕ`: hamza-below remains attached after the base letter
  changes. `T-23e96cef13c0`, `T-87108a279a4b` and `T-e87f5cf6ef36` retain 13, 13 and
  17 combining marks respectively through substitution. Those patterns can reveal
  the implementation rather than ordinary input behavior.
- The fragments in `T-a64ca5f1e1c9` and `T-7ee7b123cb57` resemble corruption, but
  there is no recorded real error process establishing them as examples of that process.
- The draft has no observed OCR/copy-paste corruption, autocomplete/ASR fragmentation,
  or verified real key-burst/long-press incidents. Some such inputs remain meaningful;
  a failure mechanism itself must never determine the binary label.

**Collection action:** record actual error mechanisms where possible. Use calibrated
reconstructions only when the source incident, mapping and parameters are documented.
Retain a small number of chunked-noise controls in a diagnostic pool; do not expand
one generator into a large nominally diverse general test.

## 3. Shortcut audit

The classifier implementation takes text, not provenance fields. Thus metadata
association is **not direct input leakage** under the current interface. It is a
warning that source style can carry the same association. IDs, dialect hints,
intended labels and source fields must remain outside model inputs.

Counts below use existing provisional labels; they are not a fitted shortcut model.
No single count proves that Jais actually uses the feature.

| Observable property | VALID, n=44 | GIBBERISH, n=32 | Concern |
|---|---:|---:|---|
| Public source_method | 12 | 0 | Public style is VALID-only |
| Arabic question mark | 4 | 0 | Interrogative punctuation identifies a one-sided subset |
| Arabic comma | 5 | 0 | Editorial/phrase punctuation differs |
| Question mark or Arabic comma | 8 | 0 | Combined punctuation cue is label-associated |
| Any punctuation | 9 | 1 | Most constructed negatives lack normal message punctuation |
| Any combining mark | 23 | 5 | Source typography differs by class |
| At least 10 combining marks | 9 | 3 | Small curated-format pockets |
| Text not already NFC | 9 | 3 | Mixed encoding histories reflect source/recipe |
| Alif maqsura `ى` anywhere | 5 | 14 | Artificial mapping can distort letter frequencies |
| Normalized length >40 | 17 | 6 | Longer tail disproportionately VALID |
| Adjacent repeated letter after comparison normalization | 20 | 10 | Repetition is not specific to GIBBERISH; some is natural morphology |

Normalized character medians overlap (36 VALID / 33 GIBBERISH), and the earlier
report gives tokenizer medians 16 / 17. Nevertheless, word medians differ (7 / 5),
synthetic word shapes differ and the tails do not match. Length matching alone is insufficient.

Even the balanced `python` and `codex` groups conceal recipe-level coupling:
Python VALID is midpoint repetition or selected shuffling; Python GIBBERISH is fixed
substitution or chunked noise. Assistant VALID is fluent service prose; assistant
GIBBERISH is invented vocabulary/particles. This is not equivalent to matching
acquisition conditions across labels.

**Do not “fix” these counts by corrupting genuine punctuation, fabricating public
GIBBERISH, stripping all diacritics, or relabeling to quotas.** Obtain better-matched
natural material, preserve raw source text and isolate artificial typography tests.
Arabic `ى`, diacritics, repetition and Unicode normalization differences can all
occur in meaningful text; none is a ground-truth rule.

## 4. Dialect support

| Existing VALID hint | Count | Assessment |
|---|---:|---|
| Tunisian | 30 | 12 dictionary examples + 16 derivatives + 2 assistant examples; meaningful source support, but concentrated genre and unknown speakers |
| MSA | 8 | Consistent register in mostly service-style assistant prose; not eight naturally sourced observations |
| Egyptian | 2 | Plausible markers in two authored strings; far too little variation to establish coverage |
| Levantine | 2 | One strongly compatible construction, one shared colloquial utterance needing region review |
| Gulf | 2 | Gulf-compatible vocabulary/constructions; cannot establish country or speaker identity |

Useful evidence includes `هو ينفع` / `فرع تاني` in `T-1d463afe8d34`, `مش عارف` /
`دراعي` in `T-268f8cc6a30f`, `عم` plus imperfective in `T-c727f791e425`, and `وش أسوي`
/ `جوال` and `أبي` / `للحين` in `T-8f0321b03132` and `T-583d0f354d3f`.

In contrast, `T-a92ecf86852e` (`حطّولي سكر بالقهوة مع إني طالبها سادة`) is meaningful
colloquial Arabic but does not uniquely support a Levantine label. Do not change its
binary judgment just because the regional hint is weak. The other authored dialect
cases are plausible, not authenticated native-speaker samples.

Tunisian forms/loanwords support regional compatibility, but unknown contributor
identities and dictionary editorial spelling limit population coverage. Idiomatic
items such as `T-956e5ff5b3f2` need dialect competence rather than literal MSA reading.
**Do not regard the 30-example Tunisian quota as qualitatively fulfilled yet.**
All 32 GIBBERISH cases correctly avoid invented dialect labels with `not_applicable`.
A corrupted parent's origin can be recorded separately without assigning a dialect
to an uninterpretable resulting string.

## 5. Ambiguity and priority manual review

### P0: 16 highest-priority cases

**Four scrambled proposed VALID cases:**

| ID | Evidence | Review requirement |
|---|---|---|
| `T-798d58640a53` | `دش اتوح كيربا ييخا عقوطتل طخي ٕصاةنار` | Can a native reader identify meaning without the fishing source? A combining mark begins the final token. |
| `T-43465a020054` | `تطيح شةبر باكابري يف اةلمالك تجا حرةرحم شربة` | Some food-related fragments may be recognizable; source-free paraphrase is essential. |
| `T-d7dc586f6f3d` | `دش اطلٕف نم ودي قوش يهب سالٕكيا` | Several short distorted words admit competing readings; do not reveal the child/street source initially. |
| `T-e4d962088237` | `يفلاكر فهاي شراب نويدما كنل فياه باشر رقف` | A coherent interpretation may be inferable, but knowing the Africa/diamonds source primes it. |

Their earlier correction to VALID is a proposal, **not proof** that an unprimed
reader can recover that interpretation. Nor does this audit establish GIBBERISH.
Keep all four unresolved until input-only human review; nominate them for diagnostic
recoverability analysis rather than automatic general-test inclusion.

**Four proposed GIBBERISH particle lists:**
`T-1679cb1c170d`, `T-425f420f9c15`, `T-d66f6361e2c6`, `T-f3f4f0e73346`.
A recognizable word list may be deliberate grammar practice; alternatively, the
sequence may have no recoverable communicative intent. The stored definition does
not resolve every such boundary without judgment. Do not force a consensus or
quietly broaden “gibberish” to mean “not a complete sentence.” Consider an unscored
ambiguity ledger if the reviewers cannot resolve the interpretation.

**Eight Arabic-like invented cases:**
`T-150086583df0`, `T-372bd589c43d`, `T-7da9b814d38f`, `T-86b35aeda912`,
`T-d0fd8c6f749e`, `T-dcd69a55ae85`, `T-dcdc71351b88`, `T-dd65cbfcfae2`.
Unknown dialect words, technical terms, names or playful language can resemble
nonce forms. Their creator's intention is not ground truth. Seek native/domain-aware
interpretation, not “I do not know these words, therefore GIBBERISH.” They may be
useful diagnostic negatives after adjudication, with legitimate unfamiliar-vocabulary
counterexamples collected independently.

### P1: 31 further quality/representation reviews

The CSV lists each ID and rationale. Priority groups are:

- Eight simulated keyboard substitutions: check causal plausibility and Unicode
  side effects; default proposal is diagnostic-only, not observed general-input evidence.
- Eight chunked-noise rows: consolidate the mechanism in future reporting; nominate
  redundant examples for exclusion from the general-purpose allocation, not deletion.
- Four assistant “garbled fragments”: require mechanism/provenance qualification.
- Dictionary/idiom and heavily marked typo cases: check intent with native competence,
  separate typography from meaning, and review whether each belongs in format/idiom diagnostics.
- `T-a92ecf86852e`: prioritize regional-hint review while preserving its clear meaningfulness.

### P2: 29 lower-ambiguity prototypes

These have relatively plausible recoverable intent or mild errors. They still need
human annotation and source/genre eligibility review. P2 does not mean a final
VALID label or automatic eligibility for the representative test.

### Recommended review procedure — not executed

Show ID/text only first, as in the existing preview. Ask whether a reasonable meaning
or intent is recoverable and, for uncertain cases, request a short paraphrase.
Do not demand a unique intended reconstruction when another reasonable interpretation
already supports VALID. Do not supply the original parent, English gloss, recipe,
intended label or model output before the initial judgment.

Then inspect provenance and diagnostics separately. A second independent native
judgment is recommended for P0 and dialect-dependent cases **if the user approves
and an appropriate reviewer is available**; no additional reviewer is appointed here.
With a single reviewer, preserve unresolved flags rather than pretending agreement
was established. Binary labels remain binary; unresolved is review metadata, not a
third scored class. No current label or flag is altered by these recommendations.

## 6. Proposed revised collection strategy — pending review

**The current `test-400-v1` plan, lock and databases remain unchanged.** The following
is a proposal, not authorization to move cases, amend the plan or start collection.
Do not overwrite or reinitialize the existing lock to implement it.

### A. General-purpose 400-example test

1. Retain the proposed approximately 200 VALID / 200 GIBBERISH and, for VALID,
   80 MSA / 30 Tunisian / 30 Egyptian / 30 Levantine / 30 Gulf **subject to approval**.
   Treat this as a balanced, stratified case-control benchmark, not an estimate of
   deployment prevalence. Reassess all coverage slots; the current Tunisian count
   is not 30 confirmed representative observations.
2. For the 200 VALID, target at least **160 naturally written, preserved input
   snippets**, up to 20 native-user elicited inputs and at most 20 documented benign
   reconstructions. Assistant-written prototypes should not silently fill the natural
   quota. Prefer multiple independent sources, speakers and situations; separate
   authentic writing from curated dictionary examples.
3. For the 200 GIBBERISH, target at least **100 independently observed failures**
   and at most 100 reconstructions grounded in documented actual mechanisms. The
   mixture should reflect the acquisition evidence, not equal recipe quotas invented
   in advance. If that natural material cannot be obtained reliably, stop short and
   return a gap report; do not manufacture production representativeness.
4. Include meaningful one-to-four-word inputs, informal statements, social exchanges,
   unfinished follow-ups, organic spelling variation and short topics. Cover lengths
   in **both** classes with no recipe-only tail. Remain within the existing declared
   tokenizer budget unless a separate scope change is approved; do not silently truncate.
5. Use authentic acquisition artifacts rather than adding artificial punctuation or
   erasing diacritics to make histograms equal. Audit class-by-source, class-by-recipe,
   Unicode, word-length and register tables. A particular corpus need not supply both
   labels: do not mislabel meaningful dictionary entries to fill a negative quota.
6. Reserve whole authors/conversations/source records and generation families where
   available. Multiple paraphrases, corruption levels or normalization forms of one
   base belong to one family, not independent slots across train/validation/general/diagnostic.
7. Complete content review before final dataset freeze. Existing checkpoint choices
   and validation thresholds remain fixed; never use test predictions to curate cases,
   tune thresholds or choose a favorable seed.

These acquisition targets are deliberately stricter than the earlier source-method
allocation, which could yield a mostly constructed benchmark. They require a source
access/rights feasibility review before approval. No promise of 400 authentic cases
is made based on the current sources.

### B. Separate explicitly challenging diagnostic set

Propose an independent `diagnostic-v1` with an initial **budget of up to 120 adjudicated
cases**, around 20 per phenomenon, rather than forcing balanced labels or exact
counts before review:

| Phenomenon | Purpose |
|---|---|
| Regional idioms, loanwords and unfamiliar legitimate vocabulary | False rejection under unfamiliar yet meaningful Arabic |
| Diacritics, NFC/NFD, spacing and benign Unicode variation | Invariance to legitimate formatting |
| Recoverability gradients | Where damage stops allowing reasonable interpretation |
| Verified keyboard/OCR/copy-paste mechanisms | Causal robustness, with mechanism parameters documented |
| Arabic-like nonce forms with meaningful rare-word controls | Prevent unknown-vocabulary shortcuts |
| Lists, fragments and instruction-like text | Clarify boundaries of communicative intent without requiring full sentences |

Do not set labels by phenomenon or by corruption strength. Adjudicate each input
under the same recoverable-meaning definition. Store unresolved cases in an
**unscored ambiguity ledger** until a defensible binary judgment is available.

The current 36 stress constructions, nine editorially marked cases and ambiguous
idioms provide diagnostic proposals, not automatic accepted members. The 31
naturalistic prototypes help design acquisition prompts but are not automatically
eligible general-test observations. Whole source/parent families should be assigned
to one evaluation pool; normalization pairs and corruption gradients remain together
inside diagnostics, not duplicated into the general test.

Report diagnostic results by phenomenon, with label counts and denominators. Do
not pool diagnostic accuracy with general-test accuracy or market it as deployment
performance. Do not use either held-out pool for tuning. If diagnostics later become
a development resource, explicitly retire that version from held-out status rather
than reusing it as evidence of generalization.

## 7. Read-only outcome and next action

New files are this report and advisory audit outputs under
`results/dataset_audits/test400-qualitative-20261029/`. Existing examples, proposed
labels, blind preview, annotations, protocol locks and experiment artifacts are not
changed. No more candidates were collected and no evaluation was run.

**Recommended next action:** review the 16 P0 cases and approve or revise the
separation/eligibility proposal before further acquisition. Do not import or freeze
the draft, and do not treat its current counts as fulfilled general-test quotas.
