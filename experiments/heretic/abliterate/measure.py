#!/usr/bin/env python3
"""Измеряет refusal direction = mean(harmful) - mean(harmless) по слоям.

Без TransformerLens — через transformers output_hidden_states. Работает на любой
causal LM (в т.ч. Qwen2.5). Направление считается на последнем токене (после
инструкции, т.е. на позиции начала ответа).

Запуск (на GPU):
  python abliterate/measure.py --model Qwen/Qwen2.5-7B-Instruct \
      --out abliterate/refusal_dirs.pt --max-samples 128 --batch-size 4

Результат: refusal_dirs.pt = { dirs: [(layer, unit_vector)], scores: [(layer, L2_norm)] }
Выбор слоя — по максимальной L2-норме разности средних (эвристика, можно
переопределить вручную через --layer в ablate.py).
"""
import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--harmful", default="mlabonne/harmful_behaviors")
    ap.add_argument("--harmless", default="mlabonne/harmless_alpaca")
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="abliterate/refusal_dirs.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    t_start = time.time()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[load] {args.model} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, output_hidden_states=True
    )
    model.to(device)
    model.eval()

    def format_prompt(text: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )

    def mean_hidden_by_layer(texts):
        """Средний hidden-вектор последнего токена по слоям (без эмбеддингов)."""
        sums = None
        count = 0
        for i in range(0, len(texts), args.batch_size):
            batch = [format_prompt(t) for t in texts[i:i + args.batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple len n_layers+1, [B,T,H]
            last = enc["attention_mask"].sum(dim=1) - 1  # [B] last real token
            b = torch.arange(enc["input_ids"].shape[0], device=device)
            per_layer = [layer[b, last].sum(dim=0) for layer in hs[1:]]
            if sums is None:
                sums = [torch.zeros_like(v) for v in per_layer]
            for s, v in zip(sums, per_layer):
                s += v
            count += enc["input_ids"].shape[0]
        return [s / count for s in sums]

    harmful = load_dataset(args.harmful, split="train")["text"][: args.max_samples]
    harmless = load_dataset(args.harmless, split="train")["text"][: args.max_samples]
    print(f"[data] harmful={len(harmful)} harmless={len(harmless)}")

    print("[run] harmful mean hidden...")
    mean_h = mean_hidden_by_layer(harmful)
    print("[run] harmless mean hidden...")
    mean_l = mean_hidden_by_layer(harmless)

    dirs = []
    for idx, (h, l) in enumerate(zip(mean_h, mean_l)):
        layer_num = idx + 1  # hidden_states[1] = выход слоя 0
        diff = (h - l).float()
        norm = diff.norm().item()
        unit = diff / diff.norm()
        dirs.append((layer_num, norm, unit.cpu()))

    ranked = sorted(dirs, key=lambda x: x[1], reverse=True)
    print("\n[result] слои по L2-норме разности (top 12):")
    for layer_num, norm, _ in ranked[:12]:
        print(f"  layer {layer_num:>2}: {norm:.4f}")

    torch.save(
        {
            "model": args.model,
            "dirs": [(ln, u) for ln, _, u in dirs],
            "scores": [(ln, n) for ln, n, _ in dirs],
            "elapsed_sec": round(time.time() - t_start, 1),
        },
        args.out,
    )
    print(f"\n[save] {args.out}  ({time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
