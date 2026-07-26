# DocMind — Model-Building Extension

A **model-building layer** that sits *alongside* the existing RAG pipeline
without touching it. It adds three new, parallel components:

1. **`finetune/`** — LoRA / PEFT fine-tuning of a small open-weight LLM for
   document **extractive QA** (locate & copy a field value from a document that
   contains distractors).
2. **`vlm_module/`** — document-image understanding: OCR baseline vs. a
   layout-aware extractor (and an optional Donut VLM engine).
3. **`eval_harness/`** — a stress-test harness that scores both components on
   **noisy / adversarial / degraded** inputs, not just clean validation data.

The two stacks are **parallel, not sequential** — nothing here feeds into or
depends on the RAG pipeline, and vice versa. See
`DocMind_Model_Building_Extension_Design_Doc.md` for the full design.

> **Scope discipline.** Nothing in the existing RAG path, frontend theme, or
> deployment image was modified. All new code lives in the three new top-level
> packages. Heavy training deps are isolated in `requirements-modelbuild.txt`
> and are **not** added to the deployment `requirements.txt`.

---

## What actually ran (honest status)

Per the design doc's execution discipline — *"fine-tuned" means a training run
actually happened; "evaluated" means the report exists with real numbers* —
here is exactly what has been executed on this machine (CPU-only):

| Component | Status | Evidence |
|---|---|---|
| LoRA fine-tune (SmolLM2-135M, extractive QA) | **Run** | `finetune/adapters/lora-smollm2-135m-docqa/` — saved adapter, `loss_curve.png`, `run_config.json` |
| Data prep + leakage check | **Run** | `finetune/data/dataset_card.json` (0 cross-split context duplicates) |
| OCR baseline vs. layout-aware extraction | **Run** | `vlm_module/data/extraction_scores.json` |
| Donut VLM — zero-shot | **Run for real** | scored **0.00** field accuracy — see honesty note below |
| Donut VLM — LoRA fine-tuned across 2 layouts, tested on a 3rd (unseen) | **Run for real** | `vlm_module/adapters/donut-lora-docmind/run_config.json`; **0.00 → 0.993** in-distribution (48 images), **0.00 → 0.618** on a never-seen layout (24 images) |
| Degradation harness (LLM + OCR/VLM, 4 engines) | **Run** | `eval_harness/reports/eval_results.json` + `report.html` |

**Why the task changed (classification → extractive QA).** The first version of
this module fine-tuned a **6-class document-type classifier**. It reached
**val accuracy 1.00 / macro-F1 1.00** — and a TF-IDF + logistic-regression
baseline *also* scored 1.000/1.000, confirming document-type classification is a
trivially-separable keyword task that under-tests the model. So the task was
replaced with **DocVQA-style extractive QA**, where every document embeds
**distractors** (multiple dates, amounts, and names) and the model must
disambiguate which value the question targets. The archived classification run
(adapter, `run_config.json`, `loss_curve.png`, report) lives under
`finetune/adapters/_archive/`, `finetune/data/_archive_classification/`, and
`eval_harness/reports/report_classification.html` for before/after comparison.

**Headline numbers from the QA fine-tune** (SmolLM2-135M, LoRA r=8 α=16 on the
attention + MLP projections, 3 epochs, CPU). See
`finetune/adapters/lora-smollm2-135m-docqa/run_config.json` for the source of
these numbers:

- LoRA: **2,442,240 trainable params = 1.78%** of the 137M model; targets the
  attention (`q/k/v/o`) + MLP (`gate/up/down`) projections.
- Training loss **10.83 → 0.071**; final eval loss **0.125**.
- **Test exact-match 0.5635 / token-F1 0.570** (126 QA pairs) — see
  `finetune/adapters/lora-smollm2-135m-docqa/test_metrics.json`.
- Per-question-type test EM shows *why* it's non-trivial — the two lookup
  anchors are near-solved while the reasoning questions are genuinely hard:

  | question type | test EM | kind |
  |---|---|---|
  | `sender` | 1.000 | lookup |
  | `total` | 0.944 | lookup |
  | `max_lineitem` | 0.667 | reasoning (max price) |
  | `latest_date` | 0.500 | reasoning (date max) |
  | `earliest_date` | 0.444 | reasoning (date min) |
  | `priciest_item` | 0.222 | reasoning (argmax name) |
  | `num_items` | 0.167 | reasoning (count) |

Clean EM is **0.56 — well below 1.00 by design**. A first attempt at this QA
task that used *label-adjacent* lookups (ask "issue date", answer sits behind an
"Issue date:" label) **saturated at EM 1.00**; the fix was to make the questions
require comparison/aggregation with **no matching label to copy from**. Both the
saturated lookup-QA run and the earlier classification run are archived (see
`finetune/adapters/_archive/`).

---

## Quickstart

```bash
# 1) Install the extension-only deps (kept out of the deployment image)
pip install -r requirements-modelbuild.txt
#    OCR baseline also needs the Tesseract binary:
#      macOS:  brew install tesseract
#      Debian: apt-get install tesseract-ocr

# 2) Build + leakage-check the dataset (idempotent; refuses to proceed on leak)
python -m finetune.data_prep

# 3) Fine-tune (CPU-friendly default). Add --quick for a 20-step smoke test.
python -m finetune.train_lora --base-model HuggingFaceTB/SmolLM2-135M --epochs 3

# 4) Score the saved adapter on the test split
python -m finetune.eval_task_metrics \
    --base-model HuggingFaceTB/SmolLM2-135M \
    --adapter-dir finetune/adapters/lora-smollm2-135m-docqa

# 5) Build the multi-template form-image datasets + compare OCR engines
#    (in-distribution eval: twocol+table; training: twocol+table; OOD: stacked)
python -m vlm_module.synth_forms -n 48 --templates twocol table            # -> vlm_module/data/
python -c "from vlm_module.synth_forms import build_image_dataset; \
    build_image_dataset(n=120, seed=51, out_dir='vlm_module/data_donut_train', templates=['twocol','table']); \
    build_image_dataset(n=24, seed=99, out_dir='vlm_module/data_ood_shift', templates=['stacked'])"
python -m vlm_module.eval_extraction            # baseline vs. layout-aware
#    To include the real Donut VLM: fetch weights once (streams ~806 MB), then:
python -m vlm_module.fetch_donut
python -m vlm_module.eval_extraction --donut               # zero-shot: ~0.00
#    To reproduce the LoRA fine-tune (0.00 -> 0.99 in-distribution;
#    0.00 -> 0.62 on the never-seen "stacked" layout):
python -m vlm_module.train_donut_lora --epochs 3
python -m vlm_module.eval_extraction --donut --donut-finetuned
python -m vlm_module.eval_distribution_shift --donut --donut-finetuned

# 6) Run the degradation harness and render the theme-matched report
python -m eval_harness.run_evaluation \
    --adapter finetune/adapters/lora-smollm2-135m-docqa
python -m eval_harness.report          # -> eval_harness/reports/report.html
```

---

## Component detail

### `finetune/` — LoRA document extractive-QA model
- **`data_prep.py`** — synthesises a seeded, offline corpus of documents
  (invoice / purchase-order / quote layouts), each carrying **decoy** dates,
  amounts, and names, and builds **7 question types** per document. Splits are
  **grouped-by-document + stratified-by-style** (70/15/15) and pass a
  **leakage check** (exact SHA-1 + near-duplicate Jaccard on the *context*) that
  *hard-fails* before any result can be claimed.
- **`train_lora.py`** — applies LoRA via `peft` with **explicit, documented**
  `target_modules` / `r` / `alpha` (not blind defaults), masks the prompt out of
  the loss so only the answer span is graded, trains with the HF `Trainer`, logs
  train + val loss, and saves **only the adapter** (base weights untouched — the
  point of PEFT). Emits `loss_curve.png` + `run_config.json`.
- **`eval_task_metrics.py`** — reusable `QAModel` wrapper + SQuAD-style
  **exact-match (EM)** and **token-F1**, overall and per question type.

### `vlm_module/` — document-image understanding
- **`synth_forms.py`** — renders form images (PIL) with exact field ground truth
  so field-level accuracy is unambiguous and offline-reproducible.
- **`baseline_ocr.py`** — the naive point of comparison: raw Tesseract + regex.
- **`document_extraction.py`** — the *enhanced* engine: non-destructive image
  preprocessing + **layout-aware row clustering** (reunites keys with values by
  geometry, robust to two-column forms and skew). Also ships an **optional Donut
  VLM engine** (`--engine donut`) — a genuine OCR-free vision-language document
  parser, opt-in because of its size.
- **`eval_extraction.py`** — field-level P/R/F1, engine-vs-engine.

> **Honesty note (design doc §5.2) — Donut: zero-shot FIRST, then genuinely
> fine-tuned.** The Donut VLM (`donut-base-finetuned-cord-v2`) was first loaded
> pretrained and evaluated **zero-shot** — this scored **0.00 field accuracy**,
> diagnosed (`vlm_module/donut_diagnostic.py` → `vlm_module/data/donut_diagnostic.json`)
> as an output-format mismatch: **raw value recall 0.653** (Donut's OCR reads
> ~65% of the gold values) vs. **schema-aligned accuracy 0.000** (none land under
> the right field name, because it emits the CORD receipt schema
> `<s_menu>`, `<s_total_price>`, … it was pretrained on, not this form's keys).
> Not an OCR failure or a broken eval — the same harness scores the OCR engines
> 0.986 / 0.889.
>
> **Then it was actually LoRA-fine-tuned — across genuinely distinct layouts,
> with one held out entirely.** `vlm_module/synth_forms.py` defines three
> visually distinct templates on the same field schema: `twocol` (sans-serif,
> same-line), `table` (serif, bordered columns), `stacked` (monospace, key and
> value on *separate* lines). `vlm_module/train_donut_lora.py` trains LoRA on
> the decoder's attention projections (524,288 params = 0.26%) using **120
> forms across `twocol`+`table` only** (seed 51) — **`stacked` is never shown
> during training.** 3 epochs, loss 1.256→0.036 in 478s CPU.
>
> **In-distribution** (48 held-out `twocol`/`table` images, seed 7 — templates
> seen in training, values not):
>
> | | zero-shot | LoRA fine-tuned |
> |---|---|---|
> | Field accuracy | 0.000 | **0.993** |
>
> **Distribution shift** (24 `stacked` images, seed 99 — a layout *never* seen
> in training, scored by `vlm_module/eval_distribution_shift.py`):
>
> | | baseline_ocr | layout_ocr+ | donut zero-shot | **donut fine-tuned** |
> |---|---|---|---|---|
> | Field accuracy | 0.000 | 0.000 | 0.014 | **0.618** |
>
> On a layout the fine-tune never saw, **both classical OCR baselines collapse
> to 0.000** (their line-based logic assumes key and value share a row —
> `stacked` breaks that assumption on purpose), while the fine-tuned Donut
> still gets **61.8% of fields right** — real, partial generalization to an
> unseen layout, not memorization of one template.
>
> **A real bug was caught and fixed as part of getting this number honestly.**
> The parser matching the fine-tuned model's output was originally
> case-sensitive; on `stacked` the model tends to echo that layout's own
> upper-cased on-image style (`VENDOR:`) while still generating correct
> *content* — the old parser silently zeroed those out, initially reporting a
> bare 0.000 on this set. Manually inspecting raw generations caught it; the
> parser (`document_extraction.py::_parse_finetuned_donut_output`) now matches
> case-insensitively. Also diagnosed: the model **never attempts** `Invoice No`
> or `Date` at all on `stacked` (starts generation at `Vendor`) while the other
> four fields are consistently correct — a specific, identified blind spot
> (likely tied to a training-layout visual cue that doesn't transfer), not
> vague "it doesn't generalize." Full detail: `TECHNICAL_REPORT.md` §6.1.
>
> This is a real, reproducible, and honestly-caveated before/after
> (`vlm_module/adapters/donut-lora-docmind/run_config.json`,
> `vlm_module/data/extraction_scores.json`,
> `vlm_module/data_ood_shift/distribution_shift_scores.json`) — "Vision
> Language Models" is genuinely **adapted**, with real (if partial and
> diagnosed) generalization evidence, not just an in-distribution number.
>
> The weights were fetched manually (`models/donut-cord/`, git-ignored) because
> the HF hub downloader stalls on the 806 MB blob over anonymous connections;
> `document_extraction._default_donut_model()` loads from that local dir if
> present. To reproduce: `python -m eval_harness.run_evaluation --skip-llm --donut`.

### `eval_harness/` — noisy / adversarial / degraded evaluation
- **`degrade_inputs.py`** — seeded degradations. *Text:* OCR character
  substitution, word dropout, truncation, keyboard typos. *Image:* blur,
  rotation, skew, downscale (low-res), JPEG artifacts, pixel noise.
- **`special_sets.py`** — a **distribution-shift** set (QA over an *unseen*
  "statement" document layout the model was never trained on) and an
  **adversarial-lite** set (the same answerable fields, but questions phrased to
  *lure the model toward an in-document distractor*).
- **`run_evaluation.py`** — sweeps severities to produce degradation *curves*
  (clean baseline vs. each degradation) for both components.
- **`report.py`** — renders `report.html`, styled with DocMind's existing design
  tokens (warm taupe→cream, editorial serif, flat ink-on-paper surfaces). It
  **reuses the theme values, not the theme files** — the Next.js app and its
  build are untouched.

---

## Reproducibility & artifacts
Everything is seeded. Transient training checkpoints are git-ignored. Re-running
the Quickstart regenerates identical splits, images, and metrics.

**LoRA weight binaries (`adapter_model.safetensors`) are git-ignored, not
committed** — this is the original design doc's own guidance
("not committed if large; document how to reproduce", §5.1). In practice this
was forced by Hugging Face's git remote, which rejects plain-git binary files
outright (LFS/Xet required) regardless of size; the `hf-space.yml` deploy
workflow mirrors the *entire* repo on every push, so any tracked binary here
would break every future deploy. Everything else that proves the run
happened — `run_config.json` (LoRA config, losses, val metrics), `test_metrics.json`,
`loss_curve.png`, `loss_history.json`, tokenizer files — **is** committed.
Reproduce the adapter with:
```bash
python -m finetune.train_lora --base-model HuggingFaceTB/SmolLM2-135M \
    --epochs 3 --run-name lora-smollm2-135m-docqa
```
(seed and full config are recorded in the committed `run_config.json`).

## What this does **not** cover
RLHF / preference alignment, CUDA kernel optimization, distributed training, and
formal publication — all explicitly out of scope (design doc §9).
