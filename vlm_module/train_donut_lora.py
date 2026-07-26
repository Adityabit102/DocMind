"""LoRA fine-tune of Donut's decoder on THIS project's field schema.

The zero-shot Donut engine (`document_extraction.extract_fields_donut`) scores
0.00 field accuracy because it emits the CORD *receipt* schema it was
pretrained on, not this form's `Invoice No / Vendor / Bill To / ...` keys
(diagnosed in `donut_diagnostic.py`: 0.65 raw OCR recall, 0.00 schema-aligned).
This script closes that gap for real: LoRA-adapts the decoder's attention
projections so the model learns to emit OUR schema directly, then the same
diagnostic + degradation harness can be re-run for an honest 0.00 -> X
before/after.

Data discipline (mirrors finetune/data_prep.py's leakage gate): training
images are rendered from a DIFFERENT seed into a DIFFERENT directory
(vlm_module/data_donut_train/) than the eval set (vlm_module/data/, seed=7,
used by every other VLM comparison in this project) — zero image or value
overlap between train and eval.

Target format: plain "Key: Value" lines (one per field, FIELD_KEYS order),
terminated with EOS. No special task-prefix token is introduced (avoids
resizing Donut's vocab/embeddings) — the model's default decoder start token
is enough since, after fine-tuning, this model only ever performs one task.

Run:
    python -m vlm_module.train_donut_lora --epochs 3
    python -m vlm_module.train_donut_lora --quick   # 10-step smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent / "adapters" / "donut-lora-docmind"
TRAIN_DIR = Path(__file__).resolve().parent / "data_donut_train"


def _target_text(fields: dict, field_keys: list[str]) -> str:
    return "\n".join(f"{k}: {fields[k]}" for k in field_keys)


def build_examples(train_dir: Path, field_keys: list[str]) -> list[dict]:
    manifest = json.loads((train_dir / "ground_truth.json").read_text())
    return [{"image": str(train_dir / "images" / fname), "target": _target_text(fields, field_keys)}
            for fname, fields in manifest.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune Donut's decoder on this project's field schema")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="10-step smoke test")
    ap.add_argument("--out-dir", default=str(ADAPTER_DIR))
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from PIL import Image

    from vlm_module import document_extraction as de
    from vlm_module.synth_forms import FIELD_KEYS

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (TRAIN_DIR / "ground_truth.json").exists():
        raise RuntimeError(f"Training images not found at {TRAIN_DIR}. "
                           "Generate with: python -c \"from vlm_module.synth_forms import "
                           "build_image_dataset; build_image_dataset(n=90, seed=51, "
                           "out_dir='vlm_module/data_donut_train')\"")

    examples = build_examples(TRAIN_DIR, FIELD_KEYS)
    if args.quick:
        examples = examples[:4]
    print(f"Training examples: {len(examples)}  (held out from the seed=7 eval set entirely)")

    proc, model, _ = de._load_donut(de._HUB_DONUT)  # always start from the zero-shot base
    tok = proc.tokenizer
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids("<s>")

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          bias="none", task_type="SEQ_2_SEQ_LM", target_modules=target_modules)
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA target_modules={target_modules} r={args.lora_r} alpha={args.lora_alpha} "
          f"-> trainable {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    n_steps = int(len(examples) * args.epochs) if not args.quick else 10
    loss_history: list[dict] = []
    t0 = time.time()
    step = 0
    epoch = 0.0
    while step < n_steps:
        for ex in examples:
            if step >= n_steps:
                break
            img = Image.open(ex["image"]).convert("RGB")
            pv = proc(img, return_tensors="pt").pixel_values
            target_ids = tok(ex["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            labels = torch.tensor([target_ids])

            out = model(pixel_values=pv, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

            step += 1
            epoch = step / len(examples)
            if step % 5 == 0 or step == n_steps:
                loss_history.append({"step": step, "loss": out.loss.item(), "epoch": round(epoch, 3)})
                print(f"step {step}/{n_steps}  epoch={epoch:.2f}  loss={out.loss.item():.4f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    train_secs = time.time() - t0
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    proc.image_processor.save_pretrained(out_dir)

    (out_dir / "loss_history.json").write_text(json.dumps(loss_history, indent=2))
    _plot_loss(loss_history, out_dir / "loss_curve.png")

    run_config = {
        "base_model": "naver-clova-ix/donut-base-finetuned-cord-v2 (started from ZERO-SHOT weights)",
        "task": "document field extraction, fine-tuned to this project's schema",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": target_modules,
                 "trainable_params": trainable, "total_params": total,
                 "trainable_pct": round(100 * trainable / total, 4)},
        "training": {"n_train_examples": len(examples), "epochs_equiv": round(n_steps / len(examples), 2),
                     "steps": n_steps, "lr": args.lr, "seed": args.seed,
                     "device": "cpu", "train_seconds": round(train_secs, 1)},
        "final_train_loss": loss_history[-1]["loss"] if loss_history else None,
        "data_leakage_note": ("training images rendered from seed=51 into vlm_module/data_donut_train/; "
                              "the eval set (vlm_module/data/, seed=7) used by every VLM comparison in "
                              "this project is entirely disjoint — zero image or field-value overlap."),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("\n── Donut LoRA fine-tune complete ──────────────────")
    print(f"Adapter saved to: {out_dir}")
    print(f"Train seconds: {train_secs:.1f} | final loss: {run_config['final_train_loss']}")
    print("Loss curve:", out_dir / "loss_curve.png")


def _plot_loss(history: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable, skipping loss plot:", exc)
        return
    if not history:
        return
    xs = [h["step"] for h in history]
    ys = [h["loss"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, color="#92836c", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("training step"); ax.set_ylabel("loss")
    ax.set_title("Donut LoRA fine-tune loss — document field extraction")
    ax.grid(True, alpha=0.25)
    fig.patch.set_facecolor("#f7f1e9"); ax.set_facecolor("#f7f1e9")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
