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
2. A **document-image (VLM/OCR)** extraction stack *(the VLM, Donut, is run
   **zero-shot / evaluated, not fine-tuned** — see §6)*.
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

The fair way to actually *close* "VLM adapted" would be to **LoRA-fine-tune
Donut's decoder on this schema** and re-run this exact diagnostic + harness —
see §8. That was not done here.

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
- **Donut not fine-tuned.** Its 0.000 reflects **zero-shot** schema transfer, not
  a ceiling; its OCR reads ~65% of values (§6). Fine-tuning Donut on this schema
  is the fair next step to actually *close* "VLM adapted".
- **VLM/OCR ground truth is clean by construction**, so the baseline's 0.986 is a
  best case; noisier source images would lower all three engines.

---

## 8. What I'd try next

1. **Fine-tune Donut (or a small Qwen2-VL) on this field schema** and re-run the
   same harness — turning the 0.000 schema-mismatch result into a real
   adapt-a-VLM story with a before/after number.
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
| Theme-matched visual report | `eval_harness/reports/report.html` |
| Classification saturation (val 1.00) | `finetune/adapters/_archive/lora-smollm2-135m-doccls-CLASSIFICATION/run_config.json` |
| TF-IDF proxy 1.000 | `finetune/adapters/_archive/_saturation_probe/probe_output.txt` |
| Lookup-QA saturation (EM 1.00) | `finetune/adapters/_archive/lora-smollm2-135m-docqa-LOOKUP-SATURATED/run_config.json` |
| Reproduction steps | [MODEL_BUILDING.md](MODEL_BUILDING.md) |
