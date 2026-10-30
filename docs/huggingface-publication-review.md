# JaisGuard publication review — 2026-10-30

The collection is traceable enough to prepare a release, but **the exact 2,150-record
candidate is not cleared yet**. The concrete remaining issues are two biography-style
Aya prompts, project-owner authority to release project contributions, and assembling
the scoped license/attribution notice. No original records, labels or experiments
were changed, and no upload was performed.

The original automated scan's zero findings were a narrow result, not a privacy
clearance. This investigation found additional issues through text inspection and
broader screening; it does not overwrite that historical audit.

## Where the records came from

All counts below are exclusive. `source_method=python` alone is not an origin claim.

| Actual recorded origin | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| External dataset input copied verbatim | 250 | 0 | 0 | 250 |
| Python adaptation of external dataset input | 65 | 0 | 0 | 65 |
| Direct assistant/Codex authoring | 227 | 0 | 0 | 227 |
| Python transformation of legacy assistant seeds | 20 | 0 | 0 | 20 |
| Assistant scenarios processed by Python, including unchanged inputs | 680 | 212 | 340 | 1,232 |
| Standalone Python keyboard/character/word-bank recipes, without request parents | 258 | 38 | 60 | 356 |
| **Total** | **1,500** | **250** | **400** | **2,150** |

Thus 315 are externally sourced, 1,479 have recorded assistant-authored text/seed
ancestry, and 356 are standalone algorithmic outputs. No records are identified as
original project inputs written manually by a human. Human review is annotation,
not evidence of human authorship. Public upstream inputs include human-authored
questions and localized requests; those are counted under external sources.
Of the 1,232 later scenario records, 290 have the `meaningful_request` passthrough
recipe and 942 have corruption recipes. This distinction describes the recorded
construction, not current labels or observed user traffic.

| External source | Verbatim | Adapted | Total | Recorded and publisher-declared terms |
| --- | ---: | ---: | ---: | --- |
| Amazon MASSIVE 1.1, ar-SA | 110 | 20 | 130 | CC BY 4.0 |
| Cohere Labs Aya Dataset, original-annotations | 123 | 27 | 150 | Apache 2.0 |
| Google Research TyDi QA, question_text only | 17 | 18 | 35 | Apache 2.0 |
| **Total** | **250** | **65** | **315** | |

Exact revisions and hashes are in the staged `metadata.json` and local investigation
ledger. Frozen records cite MASSIVE `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`,
Aya `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` and TyDi
`da78f23f9119363459acbaf46bf89426ff26c259`. The cached repository-info snapshots
match those revisions. The cached source cards state the same licenses as the
current publisher pages linked below.

Evidence checks performed against available local source material:

- All 250 public records match their cached upstream input/question exactly.
- All 65 public-source adaptations match the declared edit operation or character
  permutation relationship to the cached source input. This is an edit-relation
  check, not a full replay of every original RNG stream.
- All 150 Aya origins, including parents of adaptations, are recorded as
  `original-annotations`, corroborated against the cached Aya Parquet.
- All 227 direct assistant texts appear verbatim in their saved authoring JSON.
- All 20 legacy assistant seed texts appear in saved authoring evidence.
- All 1,050 later train/validation rows have matching corruption-sidecar IDs.
  Their 892 non-null original request parents match saved parent files and hashes;
  passthrough or the pure corruption helper reproduces all 892 corresponding texts.
- The original generation scripts were inspected rather than executed. For example,
  the saved realism recipe script writes candidates on execution; running it would
  violate the preservation requirement.

No provenance names a copied book, news article, private customer log or production
conversation as an additional source. TyDi document URLs point to Wikipedia, but
cached-source checks establish that the selected field is `question_text`, not
Wikipedia article/answer passages. Do not add a Wikipedia passage license merely
because a question was associated with a Wikipedia document. Earlier draft collection
materials and unused corpus proposals are not evidence that their sources entered
this frozen export. No Derja/Naous/ArBanking77 source occurs in its recorded origins.

These findings establish the documented acquisition path; they do not prove that
an upstream contributor never copied a phrase, or that AI output is unique.

## Content findings and scope

All 2,150 staged rows were checked against the frozen source projection and subjected
to expanded screening. Text inspection focused on the 250 public inputs, meaningful
inputs and longer text, with targeted inspection of risk terms in the remaining rows.
This was an AI-assisted content review, not an independent human sign-off of every
character. No external plagiarism detector or identity investigation was performed.

Use zero-based row positions in the staged Parquet files:

| Reference | Finding | Evidence and disposition |
| --- | --- | --- |
| `train[486]`, Aya upstream `train:row-46955` | First-person biography: two-part name, age 38, university/degree and employment lead-in | A completion prompt; fictional status or authority to disclose is not established. Privacy decision pending. Do not reproduce the biography in public review notes. |
| `train[277]`, Aya upstream `train:row-37408` | First-person family narrative naming a spouse and child, with study details | Also a completion prompt. Could be fictional; that is not verified. Privacy decision pending. |
| 14 Aya completion prompts, listed in evidence.json | Narrative/dialogue fragments rather than solely standalone questions | Publisher marks these original annotations and offers Apache 2.0. No specific copied publication was identified. This is not proof of infringement and does not justify demanding individual permissions for all 14. Two are the privacy cases above. |
| Generic medical, family, payment, religion, political/geographic and public-person references | Ordinary requests/topics; some personal-sounding scenarios | No concrete medical record, account value, contact number, harmful operational instruction or private person's profile was identified beyond the biography questions. Public figures and place names alone are not private data. |

The expanded screen found no email, phone/long numeric identifier, recognizable API
credential/private key or targeted harmful-instruction match. Source texts are short
Arabic-script inputs; no recognizable long book passage or lyrics was identified.
A request to write a poem/play a song is not itself the poem/song text. Licensed
third-party dataset text is present in 315 rows; the correct review outcome is
licensed reuse with attribution, not a claim that the dataset contains no copyright.
No obvious release-blocking violence/sexual instruction was found in the inspected
content; this is not a toxicity certification or a claim that every phrase is inoffensive.

The staged schema excludes internal candidate IDs, generation payloads, reviewer
notes, Aya author hashes and demographics. The staged metadata scan found no
workspace path, private metadata key or credential. The source-group attribution
`Codex / OpenAI; human annotation pending` is a stale historical provenance string;
it does not override the current human/provisional label counts. Use accurate
attribution wording in the release notice without rewriting the source record.

## Overlaps: what zero does and does not mean

The original audit still has zero exact, normalized, recorded cross-split family
and cross-split heuristic near-duplicate findings. Its near-duplicate test requires
similar lengths and only compares different splits. Family checks rely on IDs,
which are not proof of independent underlying requests.

A supplementary normalized substring check (shorter normalized text at least 20
characters, all split pairs including within-split pairs) finds **one pair**:
`test[319]` is contained in `test[204]`. Both ask about storage duration of cooked
lentils; the longer prompt adds handling details and a follow-up. Both labels are
VALID and their recorded family IDs differ. This is within-test dependency, not
train/test leakage. Preserve both records and disclose the dependence; do not
claim 400 fully independent scenarios. The original length-ratio/cross-split
heuristic could not detect this case.

Known-parent normalized comparisons also find zero cross-split equality among
700 available training parents, 212 validation parents and 80 unchanged test
requests. This is narrower than a semantic similarity guarantee.

`data/test400_collection_v2/private/parents.jsonl` and `corruptions.jsonl` are absent
from this checkout. Frozen test rows and the read-only annotation DB retain the
assistant-scenario/run/seed provenance, but not those private parent texts. Sidecars
are therefore unavailable for 340 scenario rows; the 80 meaningful_request originals
are inferable from their passthrough recipe, while parents for the 260 transformed
requests cannot be replayed here. Restore saved evidence if available; otherwise
explicitly disclose this reproducibility limit. Do not invent replacement parents.
This does not identify a new restricted source, nor does it erase the existing
benchmark construction evidence. It prevents a complete parent-level independence
or deterministic reconstruction claim for this checkout.

The benchmark was already examined and informed iteration 2. Disclosure of that
fact remains mandatory regardless of overlap counts.

## Practical licensing recommendation

**Conditional recommendation:** CC BY 4.0 for project-owned contributions and
compilation/annotation rights, with MASSIVE retained under CC BY 4.0 and Aya/TyDi
components explicitly retained under Apache 2.0. Expose both applicable license
identifiers in Hugging Face YAML, with clear per-component scope in the card.
This is not a dual-license offer letting users choose either license for every row.
A blanket CC0 release is not supported: the project cannot waive third-party
attribution/license conditions. A blanket claim that every row is newly owned
CC BY material would also be misleading.

Ordinary open-source reuse does not require obtaining fresh bespoke permission
from every dataset publisher. Their published terms supply the route, provided the
actual distribution satisfies those terms and the privacy cases are resolved:

- MASSIVE: give its contributor attribution, source/revision, CC BY 4.0 link,
  modification notice for the 20 adaptations, and retain applicable supplied notices.
- Aya/TyDi: provide a copy of Apache 2.0, preserve applicable attribution notices
  and clearly identify selection/format changes and the 45 adapted texts. Cached
  pinned repository-info snapshots show no separate NOTICE file; do not invent one
  or claim an exhaustive inspection of all upstream distributions. Include source
  card attribution/citation context and check for additional notices if packaging changes.
- Project additions: the owner must authorize the grant and confirm authority over
  the project-authored/assistant-output material, including any employer/account rights.
  The local-review-only/publication-review-pending strings are workflow placeholders,
  not identified third-party nonredistribution contracts. They can be superseded in
  release documentation after real approval; they are not sufficient evidence of it.

OpenAI's published terms generally assign its output rights to the user/customer,
subject to law and the applicable agreement; they do not establish this project's
account owner, employer rights, third-party content rights, or an unconditional
right to train/distribute any model. The recorded model/session metadata is missing.
Account-contract applicability remains an owner confirmation, not something inferred
from the string `source_method=codex`. Do not claim that OpenAI supplied a bespoke
publication permission. See the consumer and business terms below, including any
applicable use restrictions; the classifier is not being uploaded in this task.

## Prepared files and the minimum owner decisions

Local, Git-ignored investigation files:

- `artifacts/huggingface-review/investigation/evidence.json`: reproducible checks,
  risk references, counts, missing evidence and source-file SHA-256 hashes.
- `record-provenance.json` in that directory: all 2,150 rows mapped to public/source
  origin, upstream source ID/revision, recipe and text hash; no reviewer notes.
- `ATTRIBUTION.draft.md`: source/revision/license/modification table for all 315
  external records, without internal IDs or annotator hashes.
- `licensing-statement.draft.md`: proposed scoped statement, source attributions and
  Apache 2.0 copy, explicitly draft and not yet a project license grant.
- `artifacts/huggingface-review/approved-review.json`: evidence-populated draft;
  license and actual approval statuses remain unresolved.

`configs/publication.proposed.json` prepares the suggested license identifiers and
statement location. The active `configs/publication.json` remains unlicensed.
The staged five files and their metadata hash remain unchanged. No approval is
implied by preparing these drafts.

Only these substantive decisions remain for the owner:

1. Confirm authority to license the project contributions under the proposed scoped
   CC BY terms, and confirm which account/organization terms covered the AI authoring.
2. Resolve the two biography prompts: establish fictional/authorized status or give
   an evidence-based privacy release rationale. If they cannot be released unchanged,
   stop this exact version; a separately authorized derived publication must be
   versioned and must not silently replace the experiment data.
3. Accept the scoped license/attribution package and the disclosed benchmark/lineage
   limitations. Restore test parent sidecars if available; otherwise retain the
   explicit limitation instead of claiming complete parent-level independence.

No need to repeat all annotation or invent a new agreement statistic. After those
items are settled, integrate the reviewed statement/notices and the containment/
lineage disclosure into the card, rebuild, then bind the approval to the **new**
metadata hash. The existing publisher permits only five staging files: full license
text and attributions can be included in README.md, or the exporter/manifest/allowlist
must be extended deliberately if separate notice files are preferred. The draft
currently includes the license text for the README route. No bypass flag is needed.

Reproduce the evidence without regenerating or changing source datasets:

```bash
cd /home/rayen/projects/fine-tuning
.venv/bin/python scripts/investigate_publication.py
.venv/bin/python -m src.jais_guard.publication --dry-run \
  --review artifacts/huggingface-review/approved-review.json
```

The second command is expected to remain blocked while the license/approvals are null.
The first command writes only local investigation reports. It does not manufacture
approval or rewrite the approval draft.

## Primary references checked

- [MASSIVE pinned source card](https://huggingface.co/datasets/AmazonScience/massive/blob/ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740/README.md)
- [Aya publisher card and license](https://huggingface.co/datasets/CohereLabs/aya_dataset)
- [TyDi publisher card and license](https://huggingface.co/datasets/google-research-datasets/tydiqa)
- [TyDi collection methodology](https://github.com/google-research-datasets/tydiqa)
- [Apache 2.0 redistribution conditions, section 4](https://www.apache.org/licenses/LICENSE-2.0)
- [CC BY 4.0 scope and conditions](https://creativecommons.org/licenses/by/4.0/legalcode.en)
- [OpenAI consumer terms: content and use restrictions](https://openai.com/policies/row-terms-of-use/)
- [OpenAI Services Agreement: content and use restrictions](https://openai.com/policies/services-agreement/)

Web checks corroborate publisher policies; the pinned cached evidence and its hashes
are the basis for the record-specific investigation. No dataset texts were sent to
external plagiarism/identity search services.
