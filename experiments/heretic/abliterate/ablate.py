#!/usr/bin/env python3
"""Ортогонализует веса относительно refusal direction и сохраняет модель.

Вычитает проекцию на направление отказа у матриц, пишущих в residual stream:
для Qwen2.5 это attention o_proj и MLP down_proj выбранных слоёв.

  W' = W - r (r^T W)

Запуск (на GPU):
  # один слой (по умолчанию — слой с макс. L2-нормой из measure.py):
  python abliterate/ablate.py --model Qwen/Qwen2.5-7B-Instruct \
      --dirs abliterate/refusal_dirs.pt --out models/abliterated
  # вручную, диапазон средних слоёв:
  python abliterate/ablate.py ... --layer-start 12 --layer-end 16

После этого прогнать eval: python eval/run_eval.py --model models/abliterated --tag S1_abliterated
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def orthogonalize(W: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """W: [d_out, d_in], r: [d_model] единичной нормы. Возвращает W - r(r^T W)."""
    r = r.to(W.device, W.dtype)
    proj = r.unsqueeze(1) @ (r.unsqueeze(0) @ W)
    return W - proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dirs", default="abliterate/refusal_dirs.pt")
    ap.add_argument("--layer", type=int, default=None, help="конкретный слой (1-based)")
    ap.add_argument("--layer-start", type=int, default=None)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--out", default="models/abliterated")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    t_start = time.time()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    data = torch.load(args.dirs, map_location="cpu")
    dirs = dict(data["dirs"])  # layer -> unit vector

    if args.layer is not None:
        layers = [args.layer]
    elif args.layer_start is not None and args.layer_end is not None:
        layers = list(range(args.layer_start, args.layer_end + 1))
    else:
        layers = [max(data["scores"], key=lambda x: x[1])[0]]
    print(f"[ablate] слои: {layers}")

    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    n_layers = len(model.model.layers)
    with torch.no_grad():
        for ln in layers:
            if ln < 1 or ln > n_layers:
                print(f"  ! слой {ln} вне диапазона 1..{n_layers}, пропускаю")
                continue
            r = dirs[ln]
            block = model.model.layers[ln - 1]
            block.self_attn.o_proj.weight.data = orthogonalize(block.self_attn.o_proj.weight.data, r)
            block.mlp.down_proj.weight.data = orthogonalize(block.mlp.down_proj.weight.data, r)
            print(f"  layer {ln}: o_proj + down_proj ортогонализованы")

    print(f"[save] {args.out}")
    model.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(args.out)
    print(f"[done] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
