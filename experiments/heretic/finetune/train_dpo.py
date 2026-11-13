#!/usr/bin/env python3
"""LoRA DPO fine-tune Qwen2.5-7B-Instruct на uncensored-датасете (снятие отказов).

Подход: DPO на unalignment/toxic-dpo-v0.1 (chosen = «отвечает», rejected = «отказ»).
Unsloth + TRL. На 4090 (24 ГБ) 7B в 4-bit + LoRA rank 64 влезает.

ВАЖНО перед запуском:
  - Датасет GATED: сначала войти на huggingface.co, принять условия на странице
    unalignment/toxic-dpo-v0.1, и выставить HF_TOKEN (huggingface-cli login).
  - Проверить имена колонок датасета (ожидаются prompt/chosen/rejected).

Запуск (на GPU):
  pip install unsloth trl
  python finetune/train_dpo.py --max-samples 3000 --epochs 1 --output models/lora_uncensored

Результат: lora_uncensored_merged (полная 16-bit модель), затем прогнать eval:
  python eval/run_eval.py --model models/lora_uncensored_merged --tag S2_finetuned
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="unalignment/toxic-dpo-v0.1")
    ap.add_argument("--max-samples", type=int, default=3000, help="0 = весь датасет")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--output", default="models/lora_uncensored")
    args = ap.parse_args()

    t_start = time.time()

    from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
    PatchDPOTrainer()
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset

    print(f"[load] {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    print(f"[data] {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    # оставляем только нужные колонки (имена сверить: обычно prompt/chosen/rejected)
    keep = [c for c in ("prompt", "chosen", "rejected") if c in ds.column_names]
    ds = ds.select_columns(keep)
    print(f"[data] колонки={keep}, строк={len(ds)}")

    print("[train] DPO...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=args.epochs,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            seed=42,
            output_dir=args.output,
            save_strategy="no",
        ),
        beta=0.1,
        train_dataset=ds,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=512,
    )
    trainer.train()

    merged = args.output + "_merged"
    print(f"[save] merged -> {merged}")
    model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
    print(f"[done] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
