# JaisGuard evaluation-source acquisition feasibility

> Historical checkpoint: preserved as evidence, not current operating instructions.
> See [current public workflows](public-workflows.md).

**29 October 2026 — documentation research only; no new candidates collected.**

## 1. Decision and evidence standard

There are plausible sources of naturally written Arabic across the requested
varieties. There is **not yet an evidenced, rights-cleared reservoir of 200 authentic
GIBBERISH inputs**, or even a verified yield of 100 observed failures. Completion
of a representative 400-example benchmark cannot responsibly be promised now.

Recommended direction: permission-first, small human-reviewed acquisition pilots;
keep the existing 76 as an unchanged diagnostic-review reservoir. Do not fill the
general test with more pseudo-Arabic or arbitrary character transformations.

This study inspected publisher/project documentation and terms, not corpus rows.
No corpus files, posts, OCR outputs or new candidate records were acquired. No
registration form was submitted, no publisher/contributor was contacted, and no
model was executed. A link or publisher count establishes a research lead, not a
cleared or adjudicated sample. All quantities below are either **published inventory
claims** or **future screening caps**, explicitly distinguished from accepted yield.

The existing test protocol verifies: six locked checkpoint entries and 1,008 stored
train/validation references. The test DB remains empty. Approval in principle of
two evaluation pools does not silently rewrite that lock or the current schema.

## 2. Current requirements and gaps

The current implementation requires Arabic-script text, a binary *intended*
proposal distinct from human truth, stable IDs, source/family provenance, license
and rights evidence. Public records require a pinned revision and verified rights.
Only Apache-2.0, MIT, CC-BY-4.0 and CC-BY-SA-4.0 are currently admitted by the public
validator. NC/ND/custom terms and even CC0 or LGPL require an explicit policy/code
review rather than a false license tag. **LGPL/GPL are not non-commercial licenses**;
they are simply outside this validator's existing list, with additional data-rights
questions for third-party posts.

`source_method` currently permits only public/codex/python; partitions are only
train/validation/test. Test import/freeze/export assume exactly 400. These constraints
cannot faithfully represent all consented observations, actual OCR outputs or an
up-to-120 diagnostic pool without a reviewed migration. See section 8.

Distinguish these acquisition origins:

| Origin | What it establishes | What it does not establish |
|---|---|---|
| Naturally occurring discussion/message | Human text entered in a real interaction | Binary class, dialect or representativeness without review |
| Donated pre-existing own input | Potentially authentic writing, subject to consent and provenance | A guaranteed supply or independent authors unless documented |
| Crowdsourced/elicited response | Human production under a task | Naturally occurring application traffic |
| Dictionary example / fictional dialogue | Curated human language | Actual input distribution or observed typing mistakes |
| Recorded typing mistake | Actual human error under its recorded conditions | Unrecoverable meaning; many mistakes remain VALID |
| Actual OCR output from a real scan | Observed machine failure with an identifiable engine/source | Observed user traffic, or an automatic GIBBERISH label |
| Rendered/degraded image or character simulation | A controlled synthetic condition | An observed real-world incident |

A participant can make a real error during an elicited task: record both facts,
without calling the underlying task ordinary traffic. A bad OCR character score,
spelling correction, sentiment class, moderation action or country label is never
JaisGuard ground truth.

## 3. Source-by-source findings

### S1 — Wikimedia discussion and revision history: best low-friction public pilot

[Wikimedia terms, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)
provide CC-BY-SA-4.0 reuse routes for covered contributor text, with attribution,
change notices and exceptions for imported material. Inspect page-specific terms,
quotes and histories. Discussion posts are observed volunteer writing; encyclopedia
prose is a different, curated genre. Reverted text is not automatically gibberish.

**Coverage:** Arabic discussions primarily offer MSA; Egyptian-project discussions
may offer Egyptian. [The Tunisian Incubator project exists](https://incubator.wikimedia.org/wiki/Wp/aeb),
but its existence does not establish sufficient Tunisian messages or authors.
No source-supported Gulf quota is established. **Cap:** 40 logical posts initially
(25 MSA-oriented, 10 Egyptian-project, up to 5 Tunisian-project, only if present).
This allocation is a review budget, not a verified yield.

**Retrieval later:** use [MediaWiki revision metadata](https://www.mediawiki.org/wiki/API:Revisions)
and complete discussion units; retain project/page/revision/section and author/thread
families. [API etiquette](https://www.mediawiki.org/wiki/API:Etiquette) requires an
identifiable client and considerate requests. Do not run a scraper in this study.

**Privacy/diversity:** public usernames, signatures and IPs can identify people;
never promise anonymity while retaining identifying attribution links. Exclude
sensitive units when privacy and required attribution cannot both be respected.
Avoid repeatedly sampling the same editor or technical discussion.

**Overlap:** Wikipedia ancestry overlaps with TyDi and potentially other corpora.
Group article, talk page and revision descendants; compare original TyDi document
URLs, record IDs and text, not only our short exported questions. Copied quotations,
public benchmark texts and vandalism need separate inspection. Never resurrect
suppressed/private revision material. The expected GIBBERISH yield is unknown.

### S2 — TSAC: strong Tunisian natural-writing lead, permission blocked

[Original TSAC repository](https://github.com/fbougares/TSAC) describes about 17,000
Facebook comments from Tunisian radio/TV pages. Its [license file](https://github.com/fbougares/TSAC/blob/master/LICENSE)
is LGPL-3.0; a mirror mixes an LGPL header with an Apache description. Do not use
that mirror description as permission for underlying posts. **Cap after clearance:**
60 comments, not 60 accepted cases. No negative-class yield is established.

Inspect original `train_pos.txt`, `train_neg.txt`, `test_pos.txt`, `test_neg.txt` only
after a source-specific rights decision. Sentiment labels are irrelevant to binary
meaning. Confirm author/thread metadata, original spelling and underlying post reuse.
A page's Tunisian origin is not a verified dialect label for every comment. Personal
mentions and copied text need review. Track originals across all aggregate mirrors;
if authors are unknown, do not count rows as independent authors. Current admission: **0**.

### S3 — ASTD: natural Arabic tweets, cleaned-text and rights caveats

[ASTD's original repository](https://github.com/mahmoudnabil/ASTD) advertises over
10,000 cleaned tweets and a GPL-2.0 repository license. It is a useful Arabic/Egyptian
lead, not a verified Egyptian allocation. **Cap after rights review:** 40 records.
Examine the documented `data/Tweets.txt` and cleaning provenance; do not treat cleaned
text as a raw keystroke trace. Verify underlying post reuse and original IDs/author
grouping. Handles, quoted text and deleted content require care. Cross-corpus tweet
IDs/text must be deduplicated. Current admitted yield and GIBBERISH yield: **unverified**.

### S4 — ArSenTD-LEV: documented Levantine tweets, custom terms unresolved

[Publisher's dataset card](https://huggingface.co/datasets/ramybaly/arsentd_lev/raw/main/README.md)
describes 4,000 tweets with Jordan/Lebanon/Syria/Palestine country hints and links to
custom license terms. The linked original terms endpoint was inaccessible in this
review; `license: other` is not an open-data grant. **Cap after permission:** 40 tweets.
Native review must establish variety and recoverability independently of country and
sentiment. Obtain author/source IDs and privacy/deletion conditions. It shares possible
original tweets with other social corpora; deduplicate globally. No current cleared yield.

### S5 — QADI: all requested regional leads, but IDs are not available text

[QCRI's QADI documentation](https://github.com/qcri/QADI) reports 540,590 tweet IDs
across 18 countries, including relevant Tunisian, Egyptian, Levantine and Gulf groups.
Its retrieval instructions require hydration. No current API entitlement, surviving
text count or suitable redistribution grant was established here. **Conditional cap:**
80 IDs, 20 per regional group, after access/terms review; not an estimate of recovered
or accepted posts. Do not execute the README's historical command without verifying
current authorized access.

Country is not dialect truth. Source location/author metadata is linkable personal
information. Keep one tweet/user family across mirrors and respect current source
conditions; do not retain deleted/suppressed material through alternate copies.
This is the main broad Gulf social-writing lead found, but not a ready Gulf supply.

### S6 — QALB: genuine written mistakes, not an established gibberish pool

The [organizer FAQ](https://sites.google.com/site/arabicnlpnetwork/wanlp/qalbfaq)
shows registration-based distribution. The associated
[author's resource page](https://www.cs.columbia.edu/~noura/Resources.html) describes
raw Aljazeera comments retaining spelling/grammar errors. Current authorization for
our release and access must be obtained; do not follow an unverified mirror license.
**Conditional cap:** 30 original comment units. Register/contact the organizers only
with explicit approval, not as part of this study.

Expect useful MSA imperfections, not proven unrecoverable inputs. Original and corrected
versions, comment IDs, annotators and source pages must remain one family. Commenter
privacy and genre bias remain. The known corpus description does not establish our
required short-input yield or dialect diversity. Current clearance/yield: **not established**.

### S7 — ZAEBUC*: authentic human task production, explicitly restricted data

[The project](https://www.zaebuc.org/home) currently reports 318 Arabic essays and
1,835 Arabic utterances; writing/speech includes task-elicited student production,
not ordinary typed requests. Gulf/Egyptian speech hints need native verification.
Crucially, the [data registration form](https://docs.google.com/forms/d/e/1FAIpQLScVJeTjlwQz28wisObZVh5s6oR9tEL6xVUK9JYQl7gCSlZnPQ/viewform)
specifies **CC-BY-NC-SA-4.0**. The accompanying MIT code license does not relicense
that data. **Cap only after an explicit exception or additional grant:** 20 units.
No form was submitted. Check writer/session grouping across editions, consent scope,
student identifiers and mixed-script exclusions. Primary value is imperfect meaningful
writing or elicited conversational language; do not assume a negative-class reservoir.

### S8 — Gumar: Gulf human prose, unsuitable as a natural-input substitute

[NYU Abu Dhabi's Gumar page](https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/computational-approaches-to-modeling-language-lab/research/gumar-corpus.html)
describes more than 112 million words in over 1,200 online conversational novels.
That is human-written fiction, not naturally occurring application input. A usable
bulk redistribution grant was not established from the reviewed page. **Initial cap: 0**
for the general pool. With separately cleared terms it could inform dialect diagnostics.
Unknown authorship and many lines from one novel do not provide author diversity;
group whole novels/authors. Do not turn its large word count into a promised Gulf quota.

## 4. Authentic malformed-input and failure sources

### F1 — Recorded human spelling mistakes: promising permission-first lead

[ISLRN 922-673-450-479-2](https://www.islrn.org/resources/922-673-450-479-2/)
records an Arabic corpus by Ahmed Satti, Karim Bouzoubaa and Si Lhoussain Aouragh,
describing errors actually made while people edited documents, with TEI person/document/
error information. A current usable download and reuse grant were not established.
**Cap after access/permission:** 20 error-bearing units, constrained by actual typists.
The registry identifies a rights holder and a source portal, providing a concrete
permission/retrieval route. Do not copy demographic profiles or assume every error
is gibberish. Documents/typists, not individual errors, are families. Accepted yield unknown.

### F2 — ArabScribe: human noisy spelling under elicitation, restricted

[CAMeL's ArabScribe page](https://camel.abudhabi.nyu.edu/arabscribe/)
describes 10,000 word transcriptions using Arabic and Roman keyboards from audio
impressions. Its terms permit internal research/evaluation but do not grant ordinary
redistribution or modification. **Cap after an additional grant:** 10 Arabic-script
items for diagnostics, not general traffic. Word/typist grouping and single-word
meaning need review. These are elicited spellings, not evidence of 10,000 spontaneous
mistyped messages or unrecoverable strings. Current admission: **0**.

### F3 — A misleading typing-data lead to reject: BKSD

[The BKSD paper](https://www.mdpi.com/2076-3417/13/20/11478)
describes controlled password/phrase typing and says mistaken attempts were removed.
Thus keystroke-dynamics terminology does not establish a useful error-text corpus.
**Acquisition cap: 0**. Repeated fixed strings and authentication timing profiles do
not meet this task's content/diversity needs. Other typing studies likewise need
verification of retained erroneous text, not merely timings or statistical features.

### F4 — Real OCR failure material: source images and error strings are different

| Source and evidence | What is actually advertised | Conditional first pilot | Gate and main risk |
|---|---|---:|---|
| [NOD/Yarmouk record](https://zenodo.org/records/5068735) | 100 original Arabic scanned Wikipedia pages, reproduced in 44 image versions | 20 distinct originals, real color scans only | Rights field did not render its value during this review; verify authoritative license metadata and upstream text terms before ingestion |
| [Misraj-DocOCR card](https://huggingface.co/datasets/Misraj/Misraj-DocOCR/raw/main/README.md) | 400 real+synthetic pages, image/UUID/ground-truth Markdown, Apache-2.0 card | 10 verified real pages | Establish original document rights, lineage and real-versus-synthetic status; no erroneous engine text established from these fields |
| [Humans in the Loop](https://humansintheloop.org/resources/datasets/arabic-documents-ocr-dataset/) | 10,000 real document images, boxes and title transcriptions; CC0 dedication; access form | 10 cleared nonpersonal pages | CC0 requires a narrow current-validator update; invoices/forms/business cards require item-level privacy screening |

These figures are image/page inventories, **not counts of malformed Arabic strings**.
Do not download the entire NOD archive: its advertised archive is large, and 4,400
Arabic image variants are only 100 source-page families. Artificial-noise variants
belong to controlled diagnostics; use actual scans when assessing real scan failures.
NOD's review budget is an alternative small route, not an additional independent
pool to sum with all derivative Yarmouk releases.

Published ground truth is the intended transcription, not a gibberish judgment.
Existing raw outputs may be usable only if they have engine/version, configuration,
source page/region and permission evidence. Otherwise a future, separately approved
OCR run could observe engine errors on real scans. That is **machine-observed OCR
output**, not observed human user input; no such run is performed or authorized by
this report. Do not describe fresh rendering/degradation as a naturally occurring scan.

CER/WER, layout failures and missing words do not determine VALID/GIBBERISH. First
judge the output text alone; inspect scan/ground truth only afterward for mechanism
confirmation, avoiding the same source-priming problem found in the 76-row draft.
A blank output or wholly unsupported script is an extraction/coverage failure, not
a forced binary candidate. Retain meaningful and unrecoverable outputs from the
same acquisition conditions; do not select only the worst-looking substrings.

**Author/privacy/overlap:** documents are not speakers. Books, page variants, crops,
correct transcriptions and engine outputs share a family. Personal documents can
carry sensitive details even under an open license. NOD's Wikipedia ancestry raises
TyDi document overlap risk; track original page/article identifiers. Misraj's mixed
published document origins need independent checking rather than reliance on a card tag.

### F5 — Consent-based own-input donations and actual workflow failures

No authorized product-log source or recruitable contributor pool was identified in
this repository. This is a proposed acquisition channel, not a present inventory.
After a release/consent design is approved, a feasible *pilot request* is up to ten
contributors, at most five pre-existing **own** snippets each: **at most 50 raw units**,
with no guarantee of recruiting anyone or obtaining any GIBBERISH. Cover the five
varieties through actual contributor language, not assigned writing prompts.

For real typing failures, accept voluntary records of the original input and, if
available, a later correction plus device/layout context. Do not ask participants
to manufacture nonsense and call it observed. Limit any future logging to an expressly
consented input field in a real workflow; no global keylogger or unrelated application
capture. Cap a later pilot at 30 episodes, not 30 assumed negatives. Ordinary organic
errors often remain VALID. If the task itself is elicited, record that origin separately.

Use a contributor release covering intended private use/public release, preserve
consent evidence separately from review data, and decline third-party messages or
sensitive snippets that cannot be safely reused. Original, correction, session and
contributor are one family; anonymization does not manufacture independence.

## 5. What quantities are supported?

**Verified new accepted examples in this study: 0.** This follows the no-collection
instruction and is not an assertion that the sources contain no usable text.
No reviewed source supports an accepted-label yield forecast, especially for GIBBERISH.

The source caps above are practical, small screening budgets supported by available
inventory descriptions or an explicit future recruitment design—not promised returns.
They are not additive: social corpora can share tweets, OCR releases share pages,
and institutional corpus editions share writers. For any source, the defensible
accepted range before a pilot is **0 to its screening cap**, with no estimated pass
rate. This uncertainty should remain visible rather than hidden behind corpus size.

Recommended order after the required decisions:

1. **Migration/rights decisions first.** Name the logical dataset roles and approve
   the consent/OCR provenance representation before importing anything. Keep old locks intact.
2. **Small Wikimedia discussion pilot:** at most 40 units as allocated above. This
   is the most straightforward reviewed public-license route, primarily for meaningful
   Arabic, not a credible supply of 200 negatives. Establish original-author diversity.
3. **Permission triage for natural regional writing:** TSAC for Tunisian; ASTD/QADI
   leads for Egyptian; ArSenTD-LEV for Levantine; QADI Gulf country groups. Obtain
   current terms/access evidence before the 60/40/40/80-unit caps. Prioritize Gulf and
   Levantine access gaps rather than another Tunisian dictionary sample.
4. **Approved voluntary own-input pilot:** up to 50 units, conditional on actual
   recruitment. This is the clearest path to controlled consent and author diversity,
   not a guaranteed negative-class source. No outreach has occurred.
5. **Error-mechanism pilot:** seek the recorded human-error corpus; consider a small
   real-scan OCR route only after rights and future engine-execution approval. Use
   restricted ArabScribe/ZAEBUC only if an explicit new grant/policy decision clears them.
6. **Measure feasibility before scaling:** publish counts actually inspected, rights
   rejections, scope failures, independent authors/families, human-adjudicated VALID/GIBBERISH
   and unresolved cases. Record why a source yields no negatives; do not reinterpret
   mistakes, learner language or moderation flags as gibberish to meet quotas.

A proposed go/no-go check before planning the full negative allocation is at least
10 adjudicated authentic failures from at least two independent families **plus an
identified larger permitted reservoir**. This is an operational proposal, not a
classifier threshold or proof that 200 exist. If the pilot does not demonstrate
such access, retain an incomplete general test and report the gap. Do not silently
reduce the natural-origin requirement, replace negatives with diagnostic generators,
change class quotas or claim a representative 400. Any revised benchmark scope
needs an explicit versioned decision.

## 6. Handling the existing 76 candidates

**Retain files, IDs, proposals and audit history exactly as they are.** Recommend
zero automatic credit toward the general-test natural-input quota. Treat the 76
as diagnostic-review material pending migration and adjudication, not 76 accepted
members of a 120-item challenge set. No transfer is made by this report.

Review the 16 P0 flags first, without source parents or proposed labels:

- **Four shuffled proposed VALID:** `T-798d58640a53`, `T-43465a020054`,
  `T-d7dc586f6f3d`, `T-e4d962088237`. Require input-only recoverability; later inspect
  originals to explain the mechanism, not to supply a missing interpretation.
- **Four particle lists:** `T-1679cb1c170d`, `T-425f420f9c15`, `T-d66f6361e2c6`,
  `T-f3f4f0e73346`. Adjudicate meaningful list/educational intent versus incoherence;
  incomplete grammar alone is not a negative criterion.
- **Eight nonce/morphology cases:** `T-150086583df0`, `T-372bd589c43d`,
  `T-7da9b814d38f`, `T-86b35aeda912`, `T-d0fd8c6f749e`, `T-dcd69a55ae85`,
  `T-dcdc71351b88`, `T-dd65cbfcfae2`. Native/domain familiarity matters; author intention
  or unfamiliar vocabulary does not settle the class.

Unresolved cases belong in a separate unscored ambiguity ledger, not a third scored
class. No new reviewers are appointed; the sole reviewer can retain flags, with an
additional independent native judgment only if separately arranged and approved.

**Family constraints:** retain the entire Derja.Ninja mirror lineage conservatively
in the diagnostic reservation pool unless better contributor/entry metadata becomes
available. Do not split a parent, crop, spelling variant or normalization pair across
pools. Keep the assistant authoring run and its unused proposals reserved together;
per-item IDs do not create independent authors or construction families. The eight
chunked-noise cases share one generator: propose a small nonredundant control subset
rather than filling diagnostic slots with seeds. Keep any excluded proposals in
history; no deletion or automatic relabeling is recommended.

Diagnostic-v1 remains held out, up to 120 *adjudicated* cases, and is never a tuning
set for the current experiments. A source family assigned there cannot also supply
near variants for general-test scoring or later training.

## 7. Automation versus human work

| Can be automated after source clearance | Requires human judgment/permission |
|---|---|
| Fetch approved immutable revisions; verify hashes and notices | Establish dataset-text rights, consent/release conditions and publication compatibility |
| Record source IDs, ancestor/parent/conversation groups and pseudonymous author keys | Confirm actual author diversity when IDs are absent or sources are mirrors |
| Exact/NFKC/diacritic/tatweel comparisons and lexical near-duplicate flags | Decide semantic overlap, recoverable intent, unfamiliar dialect and ambiguous cases |
| Arabic-script detection, length/token-budget reports, Unicode anomaly flags | Decide whether scope exclusion or redaction changes communicative meaning |
| Source-by-label/genre/length/recipe descriptive tables | Confirm a regional hint, mechanism realism and representative coverage |
| Schema validation and immutable read-only audit reports | Final binary human labels, adjudication and freeze approval |

Do not automate collection by querying a classifier for likely negatives. Laughter,
hesitation, spelling errors, offensive but meaningful content, educational lists and
learner Arabic may all be VALID. Do not cherry-pick a nonsensical substring from a
recoverable complete message to create a negative. Use an agreed input unit and
record any extraction bounds before judgment. Never use test/diagnostic predictions
for source selection, seed selection or threshold tuning.

The current comparison checks remain useful, but new source aliases must resolve
to canonical original IDs (tweet, page, book, author/session) across mirrors. Extend
reference reservations to all existing train/validation rows, retired originals,
the 76 draft and its archived versions, and reserved unused authoring/parent material.
Existing religious data remains only a read-only contamination reference.
Public foundation-model pretraining exposure cannot be ruled out by new source names.

## 8. Required migration and consequences — identified, not executed

1. **Append-only protocol adoption receipt.** Record the old plan/lock/draft hashes,
   approved general-versus-diagnostic eligibility, source admission decisions and role
   allocation rules. Point to the same six checkpoint/threshold identities and pinned
   baseline model/prompt; include its completed run reference where needed. Do not edit
   an old lock or rerun selection. New decisions are independently versioned.
2. **New isolated collection workspaces/DBs.** Preserve `data/test400_v1/` as historical
   staging. Create separate new workspaces only after approval. Since no final general
   export exists in the inspected workflow, logical name `test-400-v1` can be retained
   while collection-protocol version advances; diagnostics gets its own logical role.
   If a finalized export appears before adoption, use a new dataset version instead.
3. **Explicit role-aware import/freeze/reader.** Current partitions do not include
   diagnostic and the freeze/export helpers assume 400. Prefer separate held-out DBs
   with explicit dataset-role manifests rather than retrofitting old DBs in place.
   New readers must distinguish exactly-400 general from up-to-120 adjudicated diagnostic
   and keep the ambiguity ledger unscored. No directory-glob training ingestion.
4. **Honest provenance schema.** Add approved representations for direct consented
   observations, human elicitation and engine output. Preserve original production
   mode separately from noise origin. Do not tag observed human input as `codex`, or
   invent a Python seed to satisfy the old generator rule. Record an OCR engine/version,
   image/region and settings rather than implying its output was a human typo.
5. **Narrow source-policy updates only where justified.** CC0 support may be appropriate
   for the named document source after review. LGPL/GPL/custom/NC cases need individual
   source decisions; no blanket whitelist relaxation and no automatic admission from
   repository tags. Restricted and public-release permissions must remain distinguishable.
6. **Lineage-preserving future allocation.** A relocation/copy receipt must map old IDs
   to one logical held-out role and preserve original snapshots/proposals. Do not count
   historical copies as distinct inputs. No model/threshold artifact changes are required;
   future performance reports must name the new protocol and separate the two datasets.

This migration is the consequence of adopting the split, not something to work
around by deleting the existing folder or changing its stored plan hash. No migration,
source-rule change, schema update, candidate acquisition or annotation is implemented
in this feasibility study.

## 9. Deliverables and recommendation

The new machine-readable catalogue is
`results/dataset_audits/source-feasibility-20261029/source_catalog.json`.
It records 16 leads, documentation links, origin/variety qualifications, access and
permission gates, privacy/author concerns, overlap constraints and screening caps.
No candidate text is included. A separate integrity receipt records unchanged
pre-existing files.

**Recommended next decision:** adopt the explicit migration/eligibility design and
approve a small rights-cleared pilot, starting with public discussions and source
permission clarification—not another generation batch. Keep both evaluation pools
held out and the existing 76 untouched while feasibility is measured. On current
evidence, 400 authentic examples are an unresolved acquisition goal, not an assured
next deliverable.
