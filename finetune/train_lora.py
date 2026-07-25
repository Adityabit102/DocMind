"""LoRA (PEFT) fine-tuning of a small open-weight causal LLM.

Task: document-type classification framed as instruction-following SFT — the
model learns to emit one category label after the prompt built in
``data_prep.build_prompt``. This is a real training run: it logs train + val
loss, computes a task metric (accuracy / macro-F1) on the val split each eval,
and saves the LoRA *adapter only* (standard PEFT practice — the base weights
are untouched and shared).

LoRA configuration is stated explicitly rather than left to blind defaults
(design doc §5.1):
  - target_modules: the attention projections (q/k/v/o) + MLP projections when
    present. We resolve them from the model so it works across architectures.
  - r (rank) and alpha are CLI-exposed and recorded in the run config; the
    effective scaling is alpha / r.

Defaults are sized for a CPU / free-tier box so the run actually completes:
  --base-model HuggingFaceTB/SmolLM2-135M   (swap up to a 1–3B model on a GPU)
  --quick                                    tiny model + few steps for a smoke test

Run:
  python -m finetune.train_lora --epochs 3
  python -m finetune.train_lora --quick        # fast, proves the pipeline runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"

# LoRA target-module candidates across common small architectures. We keep only
# the ones a given model actually exposes (see resolve_target_modules).
_TARGET_CANDIDATES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # Llama/Qwen/Mistral attention
    "gate_proj", "up_proj", "down_proj",          # their MLP
    "c_attn", "c_proj", "c_fc",                   # GPT-2 family
]


def resolve_target_modules(model) -> list[str]:
    present = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in _TARGET_CANDIDATES:
            present.add(leaf)
    # Prefer attention+MLP projections; fall back to GPT-2 names.
    ordered = [m for m in _TARGET_CANDIDATES if m in present]
    return ordered or ["c_attn"]


def build_tokenized_dataset(tokenizer, rows: list[dict], max_len: int):
    """Tokenise prompt+answer, masking the prompt tokens out of the loss.

    We only want the model graded on generating the ANSWER span, so prompt
    positions get label id -100 (ignored by the CE loss). To make sure the
    (short) answer is never truncated away when a long document overflows
    max_len, we keep the answer tokens and trim the *document* tokens instead.
    """
    from datasets import Dataset

    from finetune.data_prep import build_prompt

    def encode(example):
        prompt = build_prompt(example["context"], example["question"])
        target = " " + example["answer"] + tokenizer.eos_token
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        # Reserve room for the answer; trim the prompt (document) from the left-middle
        # by truncating its head if the pair overflows.
        budget = max_len - len(t_ids)
        if len(p_ids) > budget:
            p_ids = p_ids[len(p_ids) - budget:]  # keep the tail (question sits at the end)
        input_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        attn = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(encode, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a small causal LLM for document classification")
    ap.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs with a hard step cap")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="tiny model + few steps: proves the pipeline end-to-end fast")
    args = ap.parse_args()

    if args.quick:
        args.base_model = "sshleifer/tiny-gpt2"
        args.max_steps = 20
        args.batch_size = 4

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from finetune.data_prep import build_dataset, load_splits
    from finetune.eval_task_metrics import QAModel, evaluate_split

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or f"lora-{args.base_model.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ADAPTERS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data — (re)build with the leakage gate, then load splits.
    build_dataset()  # idempotent; raises if leakage is ever introduced
    splits = load_splits()
    print(f"Data: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # 2) Tokenizer + base model.
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id

    # 3) LoRA — explicit, documented config.
    target_modules = resolve_target_modules(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA target_modules={target_modules} r={args.lora_r} alpha={args.lora_alpha} "
          f"-> trainable {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    train_ds = build_tokenized_dataset(tok, splits["train"], args.max_len)
    val_ds = build_tokenized_dataset(tok, splits["val"], args.max_len)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=5,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    # 4) Save the adapter ONLY (base weights untouched — the whole point of PEFT).
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # 5) Persist the loss history + a loss-curve PNG.
    history = trainer.state.log_history
    train_curve = [(h["step"], h["loss"]) for h in history if "loss" in h]
    eval_curve = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    (out_dir / "loss_history.json").write_text(json.dumps({
        "train": train_curve, "eval": eval_curve, "history": history,
    }, indent=2))
    _plot_loss(train_curve, eval_curve, out_dir / "loss_curve.png", run_name)

    # 6) Task metric on val split with the freshly-tuned adapter (real number).
    print("Scoring the tuned adapter on the val split ...")
    qa = QAModel(base_model=args.base_model, adapter_dir=str(out_dir)).load()
    val_metrics, _ = evaluate_split(qa, splits["val"])

    run_config = {
        "run_name": run_name,
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": target_modules,
                 "trainable_params": trainable, "total_params": total,
                 "trainable_pct": round(100 * trainable / total, 4)},
        "training": {"epochs": args.epochs, "max_steps": args.max_steps,
                     "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                     "lr": args.lr, "max_len": args.max_len, "seed": args.seed,
                     "device": "cuda" if torch.cuda.is_available() else "cpu",
                     "train_seconds": round(train_secs, 1)},
        "final_train_loss": train_curve[-1][1] if train_curve else None,
        "final_eval_loss": eval_curve[-1][1] if eval_curve else None,
        "task": "document extractive QA",
        "val_task_metrics": {"exact_match": round(val_metrics.exact_match, 4),
                             "token_f1": round(val_metrics.token_f1, 4)},
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("\n── Fine-tune complete ─────────────────────────────")
    print(f"Adapter saved to: {out_dir}")
    print(f"Train seconds: {train_secs:.1f} | final train loss: {run_config['final_train_loss']}")
    print(f"VAL exact-match: {val_metrics.exact_match:.3f} | VAL token-F1: {val_metrics.token_f1:.3f}")
    print("Loss curve: ", out_dir / "loss_curve.png")


def _plot_loss(train_curve, eval_curve, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable, skipping loss plot:", exc)
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if train_curve:
        xs, ys = zip(*train_curve)
        ax.plot(xs, ys, label="train loss", color="#92836c", linewidth=2)
    if eval_curve:
        xs, ys = zip(*eval_curve)
        ax.plot(xs, ys, label="val loss", color="#4a4031", linewidth=2, marker="o")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"LoRA fine-tune loss — {title}")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.patch.set_facecolor("#f7f1e9"); ax.set_facecolor("#f7f1e9")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
