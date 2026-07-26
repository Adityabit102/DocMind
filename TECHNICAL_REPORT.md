# Fine-Tuning and Robustly Evaluating a Document-Understanding Stack
### A technical report on the DocMind model-building extension

*Companion to [MODEL_BUILDING.md](MODEL_BUILDING.md) (the reproduction/quickstart
guide). This document is the narrative writeup: what was built, why, what the
numbers actually are, and what I would do next. Every quantitative claim below
cites the repo file it comes from — see the [Evidence index](#evidence-index).*

---

## 1. Problem framing

DocMind already ships a production RAG pipeline (hybrid retrieval, reranking,
CRAG/Self-RAG grounding, RAGAS evaluation). The gap this extension closes is a
different skill: **building and adapting models**, not just orchestrating a
hosted LLM. Concretely it adds, as a *parallel, non-invasive* layer:

1. A **LoRA/PEFT fine-tune** of a small open-weight LLM for document
   understanding *(genuinely fine-tuned — §2, §5)*.
2. A **document-image (VLM/OCR)** extraction stack. Donut is evaluated
   **zero-shot first (0.000 field accuracy, diagnosed), then genuinely
   LoRA-fine-tuned on this project's schema (0.993 field accuracy)** — both
   numbers reported, see §6.
3. An **evaluation harness** that stress-tests both against noisy, adversarial,
   and degraded inputs — the part most applicants skip.

A hard design constraint governed everything: **touch nothing** in the existing
RAG path, frontend theme, or deployment image. All new code lives in three new
top-level packages (`finetune/`, `vlm_module/`, `eval_harness/`); heavy training
deps are isolated in `requirements-modelbuild.txt`; the only edit to a
pre-existing file is additive `.gitignore` lines. `app.main` still imports and
the 44-test suite still collects.

---

## 2. Method: LoRA / PEFT

The LLM is fine-tuned with **Low-Rank Adaptation (LoRA)** [Hu et al., 2022]. LoRA
freezes the pretrained weights `W ∈ ℝ^{d×k}` and learns a low-rank update
`ΔW = B·A` with `A ∈ ℝ^{r×k}`, `B ∈ ℝ^{d×r}`, `r ≪ min(d,k)`; the effective
weight at inference is `W + (α/r)·BA`. Only `A` and `B` train, so the number of
optimised parameters drops by orders of magnitude and the base model is shared
and untouched.

This project's configuration (all stated explicitly in code, not left to blind
defaults — `finetune/train_lora.py`, and recorded in
`finetune/adapters/lora-smollm2-135m-docqa/run_config.json`):

| Setting | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M` (open-weight causal LM) |
| LoRA rank `r` | 8 |
| LoRA `α` | 16 (scaling `α/r = 2`) |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | **2,442,240 = 1.78%** of the 137M-param model |
| Epochs / optimiser | 3 / AdamW, lr 2e-4, CPU |

Training masks the prompt tokens out of the loss (label id `-100`) so the model
is graded only on generating the answer span, and — because documents can
overflow the context — the tokeniser trims the *document* head rather than the
answer when packing (`finetune/train_lora.py::build_tokenized_dataset`). Only the
**adapter** (~10 MB) is saved; the base weights are never modified — the whole
point of PEFT.

> **Citation.** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
> Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language
> Models.* ICLR 2022. arXiv:2106.09685.

---

## 3. Data and the leakage gate

The training corpus is **synthesised from seeded templates** rather than
downloaded. This is a deliberate trade: it sacrifices real-world messiness (see
§7) but buys (a) zero-network, byte-for-byte reproducibility, and (b) the ability
to *guarantee* the anti-leakage property rather than trust a public split.

**Split methodology** (`finetune/data_prep.py`): documents are split
**grouped-by-document + stratified-by-style**, 70/15/15. Every QA pair generated
from one document lands in exactly one split, so a paraphrased or degraded twin
can never straddle the train/test boundary.

**Leakage gate**: before any result may be claimed, `build_dataset` runs a check
that *hard-fails* the pipeline on any exact cross-split **context** duplicate
(normalised SHA-1) and counts near-duplicates (token-set Jaccard ≥ 0.9). The
final dataset — **588 / 126 / 126** train/val/test QA pairs — passes with
**0 exact cross-split overlaps and 0 near-duplicate pairs**
(`finetune/data/dataset_card.json`).

---

## 4. The task journey: two saturations before a real task

The most important methodological finding of this project is that **naïve
document tasks saturate**, and detecting that is itself part of doing evaluation
honestly. The task was rebuilt twice.

**Attempt 1 — 6-class document-type classification.** The LoRA model reached
**validation accuracy 1.00 / macro-F1 1.00**
(`finetune/adapters/_archive/lora-smollm2-135m-doccls-CLASSIFICATION/run_config.json`).
A TF-IDF + logistic-regression bag-of-words baseline *also* scored
**accuracy 1.000 / macro-F1 1.000**
(`finetune/adapters/_archive/_saturation_probe/probe_output.txt`), confirming
document-type classification is a trivially separable keyword task that
under-tests the model. **Rejected.**

**Attempt 2 — extractive QA with label-adjacent answers.** The model was asked
e.g. "What is the issue date?", with the answer sitting right behind an
`Issue date:` label. Even with multiple decoy dates present, this reached
**validation EM 1.00 / token-F1 1.00** and **test EM 1.00**
(`finetune/adapters/_archive/lora-smollm2-135m-docqa-LOOKUP-SATURATED/run_config.json`)
— the model learned the shortcut "find the label matching the question, copy the
next token." **Rejected.**

**Attempt 3 — extractive QA requiring reasoning (final).** The fix was to remove
the label-lookup shortcut: keep two easy "anchor" lookups (`total`, `sender`) but
make the other five questions require **comparison, aggregation, or counting over
multiple in-document values, with no matching label to copy from**:

- `latest_date` / `earliest_date` — max / min over three dates printed in a fixed
  but non-chronological order (must actually compare, not read by position).
- `max_lineitem` — the highest *individual line-item* price (the larger total is a
  trap).
- `priciest_item` — the *name* of that highest-priced item (argmax → name).
- `num_items` — count of line items.

This is the task all results below use.

---

## 5. Results — LoRA LLM (extractive QA)

### 5.1 Headline

Training drove loss from **10.83 → 0.071**, final eval loss **0.125**, in ~570 s
on CPU (`run_config.json`). On the held-out **test** split (126 pairs,
`finetune/adapters/lora-smollm2-135m-docqa/test_metrics.json`):

- **Exact-match 0.5635, token-F1 0.570.**

Crucially, this is **well below 1.00** — a genuine, non-saturated result. The
per-question-type breakdown shows exactly where the difficulty lives:

| Question type | Test EM | Kind |
|---|---|---|
| `sender` | **1.000** | lookup (anchor) |
| `total` | **0.944** | lookup (anchor) |
| `max_lineitem` | 0.667 | reasoning — max price |
| `latest_date` | 0.500 | reasoning — date max |
| `earliest_date` | 0.444 | reasoning — date min |
| `priciest_item` | 0.222 | reasoning — argmax → name |
| `num_items` | **0.167** | reasoning — counting |

The two lookups are effectively solved; the reasoning questions are genuinely
hard for a 135M model, and **counting is the hardest** (0.167). This is an honest,
interpretable picture of a small model's reasoning limits — not a single flattering
number.

### 5.2 Degradation curves (noisy inputs)

The harness (`eval_harness/run_evaluation.py`) degrades the **document context**
(the question stays clean) across four text-corruption types at severities
0 → 0.9, scoring EM at each point. This is the **full 126-pair test set** (no
subsampling), so the clean baseline **EM 0.5635** matches the §5.1 headline
exactly (`eval_harness/reports/eval_results.json`, `llm`). Exact-match at each
severity:

| Degradation | sev 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| `truncate` | 0.564 | 0.476 | 0.389 | **0.254** |
| `drop_words` | 0.564 | 0.357 | 0.294 | 0.222 |
| `keyboard_typos` | 0.564 | 0.333 | 0.262 | 0.230 |
| `ocr_substitution` | 0.564 | 0.302 | 0.262 | **0.175** |

Findings, reported as-is:
- **Character-level corruption hurts most and earliest.** `ocr_substitution` and
  `keyboard_typos` roughly halve EM by severity 0.3, and `ocr_substitution` is
  the worst at max severity (0.175) — the model relies on exact token surfaces to
  copy spans, so perturbing characters *inside* the answer is maximally damaging.
- **Truncation is the most robust** degradation at every severity (0.254 at 0.9),
  because the anchor fields it can still answer often survive near the top of the
  document.
- All four curves are **monotonic** on the full test set — the mild
  non-monotonicity seen on an earlier 63-pair subsample was subsample noise and
  disappears here.

### 5.3 Beyond-clean sets

Two sets the model never trained on (`eval_harness/special_sets.py`):

- **Distribution shift** — QA over an *unseen "statement" layout*: **EM 0.500**
  (`eval_results.json`, `llm.special_sets.distribution_shift`). The reasoning
  transfers off the training layouts with a modest drop from the clean 0.564.
- **Adversarial-lite** — same fields, but questions phrased to lure toward a
  distractor: **EM 0.619**. Notably this is *not* worse than clean; the extra
  clarifying words ("an individual line item, NOT the subtotal or total") seem to
  have helped as much as they distracted. An honest null-ish result rather than a
  manufactured one.

---

## 6. Results — document-image understanding (VLM / OCR)

The `vlm_module/` compares three field extractors on 12 synthetic form images
with exact ground truth (`eval_harness/reports/eval_results.json`, `vlm`):

| Engine | Clean field accuracy |
|---|---|
| `baseline_ocr` (raw Tesseract + regex) | **0.986** |
| `layout_ocr+` (preprocess + layout-aware row clustering) | 0.889 |
| `donut_vlm` (Donut VLM, real run) | **0.000** |

Two honest results:

**The "enhanced" engine is a robustness/accuracy trade-off, not a free win.** On
clean, crisp scans the layout-aware engine actually *trails* the raw baseline
(0.889 vs 0.986) — its preprocessing slightly hurts already-clean text. Its value
shows only under degradation: it is competitive-to-better under rotation and skew
(geometry, where row-clustering helps) and holds up better at extreme pixel noise,
while both collapse under heavy blur. The degradation charts in
`eval_harness/reports/report.html` make the crossover explicit. A single clean
number would have hidden this.

**Donut scored 0.000 — zero-shot, and diagnosed.**

> **Adapted vs. evaluated — the honest scope.** Donut was run **zero-shot**: the
> pretrained `donut-base-finetuned-cord-v2` checkpoint was loaded and evaluated.
> **It was *not* fine-tuned in this project.** So this is *"I evaluated a VLM,"*
> not *"I adapted/fine-tuned a VLM."* The only genuine fine-tune here is the LLM
> (§2, §5). On a skills ledger, "Vision Language Models" is **evaluated, not
> adapted** — stated plainly so the claim is not overread.

Donut was run end-to-end over the same image set and full degradation sweep
(288 CPU inferences). A dedicated diagnostic (`vlm_module/donut_diagnostic.py`,
output `vlm_module/data/donut_diagnostic.json`) separates *reading* from
*schema-alignment* over the 12 clean images (72 fields):

| Metric | Value |
|---|---|
| **raw value recall** — gold value appears *anywhere* in Donut's decoded output | **0.653** (47/72) |
| **schema-aligned field accuracy** — value mapped to the *correct* field name | **0.000** (0/72) |

So Donut's OCR reads **~65% of the values**, but **none** align to this form's
field names, because it emits the **CORD receipt schema** it was trained on:

```
<s_menu><s_nm> INVOICE</s_nm><s_discountprice> -52445</s_discountprice>
<s_price> 2026-03-13</s_price> … <s_total_price> $8829.12</s_total_price>
```

**This is an output-format / schema mismatch, not an OCR failure and not a
broken eval** — the very same harness scores the two OCR engines 0.986 / 0.889.
An off-the-shelf VLM fine-tuned for one document type (receipts) does not
transfer zero-shot to a different field schema (these forms). The
extraction/scoring logic was **not** altered to flatter or fix the number; the
0.000 is reported as-is and the 65.3% raw recall is what explains it.

### 6.1 Closing the gap: LoRA-fine-tuning Donut on this schema

The fair way to actually *close* "VLM adapted" is to LoRA-fine-tune Donut's
decoder on this schema and re-run the exact same comparison. **This was done —
across genuinely distinct visual layouts, with one held out entirely to test
real generalization, not just held-out field values.**

**Method** (`vlm_module/train_donut_lora.py`, `vlm_module/synth_forms.py`):
three visually distinct templates share the identical `FIELD_KEYS` schema but
differ in font family, page geometry, and structure —
`twocol` (sans-serif, key/value same line), `table` (serif, bordered
Field/Value columns), and `stacked` (monospace, narrow, key and value on
**separate** lines with dashed rules). LoRA on the decoder's attention
projections (`q/k/v/out_proj`), **524,288 trainable params = 0.26%** of the
201.6M-param model. Training target: plain `"Key: Value"` lines (no CORD tags,
no new vocabulary/special tokens). **120 training forms across `twocol` +
`table` only** (seed 51, `vlm_module/data_donut_train/`) — **`stacked` is never
shown during training at all.** 3 epochs (360 steps), CPU: **loss 1.256 →
0.036** in 478s (`vlm_module/adapters/donut-lora-docmind/run_config.json`).

Two disjoint eval sets, zero image/value overlap with training or each other:
- **In-distribution held-out** (`vlm_module/data/`, seed 7, 48 images —
  `twocol` + `table`, the templates seen in training, unseen field values).
- **Distribution-shift / unseen layout** (`vlm_module/data_ood_shift/`, seed
  99, 24 images, **all `stacked`** — a layout the adapter never saw), scored by
  `vlm_module/eval_distribution_shift.py`, the VLM analogue of the LLM
  harness's `distribution_shift_set`.

**Result 1 — in-distribution (48 images, both trained templates):**

| Engine | Field accuracy | Doc exact-match | F1 |
|---|---|---|---|
| `donut_vlm` (zero-shot) | 0.000 | 0.000 | 0.000 |
| **`donut_finetuned` (LoRA-adapted)** | **0.993** | **0.958** | **0.997** |
| `baseline_ocr` (for reference) | 0.990 | 0.938 | 0.995 |
| `layout_ocr+` (for reference) | 0.941 | 0.667 | 0.970 |

**0.000 → 0.993 field accuracy**, matching the strongest OCR baseline, now
demonstrated across *two* visually distinct trained layouts, not one.

**Result 2 — distribution shift (24 images, `stacked`, never seen in
training):**

| Engine | Field accuracy | Doc exact-match | F1 |
|---|---|---|---|
| `baseline_ocr` | 0.000 | 0.000 | 0.000 |
| `layout_ocr+` | 0.000 | 0.000 | 0.000 |
| `donut_vlm` (zero-shot) | 0.014 | 0.000 | 0.027 |
| **`donut_finetuned`** | **0.618** | 0.000 | **0.764** |

This is the genuinely interesting, non-cherry-picked finding. **On a layout
the fine-tune never saw, both classical OCR baselines collapse completely
(0.000) — their regex/row-clustering logic assumes key and value share a
line, which `stacked` deliberately violates.** The fine-tuned Donut, in
contrast, gets **61.8% of fields right** on this unseen layout — real,
partial generalization, not memorization of one template's pixel geometry.

**A real bug, caught and fixed, is part of this result.** The first version of
`_parse_finetuned_donut_output` matched field labels case-sensitively. On
`stacked`, the model tends to echo that layout's own upper-cased on-image key
style (`VENDOR:` instead of `Vendor:`) while still generating the *correct
content* — the case-sensitive parser was silently discarding those correct
extractions, originally reporting a bare **0.000** on this set. Manually
inspecting raw generations caught it (below); the parser now matches
case-insensitively and maps back to the canonical field name
(`document_extraction.py::_parse_finetuned_donut_output`). This is disclosed
rather than smoothed over, per the project's own execution discipline (§1).

Inspecting five raw generations on `stacked` shows a precise, consistent
failure mode — not scattered noise:

```
GOLD: Invoice No=INV-62950 Date=2026-07-07 Vendor=Orbital Systems ...
RAW : VENDOR: Orbital Systems BILL TO: Omar Kowalski CITY: Leeds TOTAL: $2233.97
```

Across every sampled image, the model **never even attempts** `Invoice No` or
`Date` on this layout — generation starts directly at `Vendor` — while
`Vendor` / `Bill To` / `City` / `Total` are consistently correct. Two of five
samples also show a repetition artifact (`"Meridian Analytics: Analytics:"`).
The most likely explanation: on the trained templates, a strong visual cue
(a horizontal rule in `twocol`, a table header row in `table`) immediately
precedes the first field, and the model learned to key its extraction-start
off that cue; `stacked`'s per-field dashed rule doesn't match that learned
pattern, so it skips straight to a field it's more confident about. This is a
genuine, specific, diagnosed limitation — not "it just doesn't generalize."

> **Revised caveat.** The fine-tune demonstrates **real, partial**
> generalization to an unseen layout (0.618 vs. 0.000 for classical OCR) — a
> stronger result than "adapts only to one template," which was the honest
> caveat before this test existed. It is **not complete** generalization:
> `doc_exact_match` on the shift set is 0.000 (no single document gets every
> field right), and the specific failure mode — skipping the first ~2 fields
> entirely on the new layout — is diagnosed above, not hidden. Training across
> more than two layouts is the natural next step to test whether this specific
> blind spot closes with more structural diversity (§8).

**Degradation curves for the fine-tuned engine** (full 48-image in-distribution
set, no subsampling — `eval_harness/reports/eval_results.json`,
`vlm.degradation_curves`; same six degradation types and four severities as
every other engine in this report):

**`donut_finetuned` field accuracy** (severities 0.0 / 0.3 / 0.6 / 0.9):

| Degradation | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| `blur` | 0.993 | 0.979 | 0.004 | 0.000 |
| `rotate` | 0.993 | 0.993 | 0.965 | 0.903 |
| `skew` | 0.993 | 0.997 | 0.997 | 0.993 |
| `downscale` | 0.993 | 0.997 | 0.997 | 0.767 |
| `jpeg` | 0.993 | 0.993 | 0.993 | 0.990 |
| `pixel_noise` | 0.993 | 0.993 | 0.990 | 0.948 |

For comparison, `baseline_ocr` / `layout_ocr+` at the same severities:

| Degradation | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| `blur` | 0.990 / 0.941 | 0.399 / 0.365 | 0.000 / 0.000 | 0.000 / 0.000 |
| `rotate` | 0.990 / 0.941 | 0.566 / 0.510 | 0.524 / 0.371 | **0.205 / 0.118** |
| `skew` | 0.990 / 0.941 | 0.986 / 0.958 | 0.990 / 0.920 | 0.948 / 0.920 |
| `downscale` | 0.990 / 0.941 | 0.997 / 0.986 | 0.500 / 0.979 | **0.201 / 0.188** |
| `jpeg` | 0.990 / 0.941 | 0.993 / 0.958 | 0.990 / 0.969 | 0.976 / 0.958 |
| `pixel_noise` | 0.990 / 0.941 | 0.552 / 0.924 | 0.476 / 0.965 | **0.000 / 0.125** |

On this larger, genuinely mixed-template 48-image sweep (no subsampling), the
fine-tuned engine **dominates both OCR baselines on four of six degradation
types** — most dramatically on `rotate` (0.903 vs. 0.205/0.118 at severity
0.9), `downscale` (0.767 vs. 0.201/0.188), and `pixel_noise` (0.948 vs.
0.000/0.125). These are large, genuine robustness gaps: regex/row-clustering
OCR is highly sensitive to geometric distortion and resolution loss breaking
its line/row assumptions, while the VLM's learned visual representation
tolerates them far better. It loses only on `blur` (collapses to ~0 at
severity 0.6+, same as everyone — no engine can read sufficiently blurred
text) and roughly ties on `skew` and `jpeg` (all engines stay strong). This is
a substantially more informative and more favorable picture for the
fine-tuned VLM than the earlier, smaller single-template sweep suggested —
and it required no subsampling shortcut to get.

On a skills ledger, this changes the honest scope from §2's framing: Vision
Language Models is now genuinely **adapted** (LoRA-fine-tuned across multiple
layouts, with a real before/after, a real degradation curve, and a real,
diagnosed distribution-shift result) — a stronger and more honestly earned
claim than the single-template version of this section.

---

## 7. Honest limitations

- **Synthetic data.** Documents are templated, so surface diversity is far below
  real scans. The reasoning difficulty is real, but real-world messiness
  (handwriting, multi-column chaos, OCR noise on the *training* data) is not
  captured. Numbers here should be read as *relative* (which degradation hurts
  more, which question type is harder), not as absolute real-world accuracy.
- **Small model, CPU budget.** SmolLM2-135M with greedy decoding on CPU. A larger
  model (1–3B) or beam/self-consistency decoding would likely lift the reasoning
  numbers; that comparison was out of compute scope.
- **Donut's generalization is real but incomplete.** §6.1 shows genuine transfer
  to an unseen layout (0.618 field accuracy vs. 0.000 for classical OCR), but
  `doc_exact_match` on that shift set is 0.000 and there's a specific, diagnosed
  blind spot (the model skips the first ~2 fields entirely on the new layout).
  Both the training and eval templates are still synthetic and rendered by the
  same codebase, not scanned real-world documents.
- **VLM/OCR ground truth is clean by construction**, so the OCR baselines'
  ~0.99 in-distribution is a best case; noisier source images would lower all
  engines, and did — see the degradation curves in §6.1.

---

## 8. What I'd try next

1. **Close the specific unseen-layout blind spot.** §6.1's distribution-shift
   test found the fine-tuned Donut skips `Invoice No`/`Date` entirely on a
   layout it never trained on (0.618 field accuracy, but 0.000 doc-EM). Training
   across a *third* layout during fine-tuning (not just two) — or adding
   explicit structural augmentation so the model can't key its extraction-start
   off one fixed visual cue — would test whether this specific gap closes, or
   whether it's a more fundamental limit of 2-layout training. A real dataset
   (FUNSD/CORD) would validate this against genuine layout diversity.
2. **Harder reasoning + abstractive answers** (e.g. "total minus subtotal",
   "which item has the best unit price"), scored with token-F1, to push past the
   copy-a-span ceiling and test genuine numeric reasoning.
3. **Swap in a real corpus** (DocVQA / FUNSD / CORD subset) behind the same
   `data_prep` interface, keeping the leakage gate, to validate the synthetic
   findings against real documents.
4. **Decoding upgrades** (beam search, self-consistency) and a **1–3B base model**
   to quantify how much of the reasoning gap is capacity vs. decoding.
5. **A confidence/abstention signal** so the harness can measure calibration under
   degradation, not just accuracy — closer to the RAG side's grounding checks.

---

## Evidence index

Every number in this report resolves to a committed file:

| Claim | File |
|---|---|
| LoRA config, params, losses, val EM | `finetune/adapters/lora-smollm2-135m-docqa/run_config.json` |
| Test EM/F1 + per-question-type | `finetune/adapters/lora-smollm2-135m-docqa/test_metrics.json` |
| Loss curve | `finetune/adapters/lora-smollm2-135m-docqa/loss_curve.png` |
| Split counts + leakage (0/0) | `finetune/data/dataset_card.json` |
| Degradation curves, shift, adversarial, VLM 3-engine | `eval_harness/reports/eval_results.json` |
| Donut zero-shot diagnosis (0.653 raw recall vs 0.000 aligned) | `vlm_module/data/donut_diagnostic.json` |
| Donut LoRA fine-tune config, loss (1.256→0.036), params, multi-template setup | `vlm_module/adapters/donut-lora-docmind/run_config.json` |
| 3 template renderers (twocol/table/stacked) | `vlm_module/synth_forms.py` |
| Donut in-distribution before/after (0.000 → 0.993, 48 images, 2 templates) | `vlm_module/data/extraction_scores.json` (with `--donut --donut-finetuned`) |
| Donut distribution-shift result (0.618 field-acc, unseen `stacked` layout, 24 images) | `vlm_module/data_ood_shift/distribution_shift_scores.json` |
| Case-sensitivity parser bug fix (caught via manual raw-generation inspection) | `vlm_module/document_extraction.py::_parse_finetuned_donut_output` |
| Donut fine-tuned degradation curves (full 48-image sweep, no subsampling) | `eval_harness/reports/eval_results.json`, `vlm.degradation_curves` |
| Theme-matched visual report | `eval_harness/reports/report.html` |
| Classification saturation (val 1.00) | `finetune/adapters/_archive/lora-smollm2-135m-doccls-CLASSIFICATION/run_config.json` |
| TF-IDF proxy 1.000 | `finetune/adapters/_archive/_saturation_probe/probe_output.txt` |
| Lookup-QA saturation (EM 1.00) | `finetune/adapters/_archive/lora-smollm2-135m-docqa-LOOKUP-SATURATED/run_config.json` |
| Reproduction steps | [MODEL_BUILDING.md](MODEL_BUILDING.md) |
