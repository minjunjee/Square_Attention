#!/usr/bin/env python3
"""Fastest available forward implementation of each baseline, so that every model is timed at its best.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python best_baselines.py --out results/cost_best.jsonl

Softmax attention: every SDPA backend torch offers (FlashAttention, memory-efficient, cuDNN), causal, bf16.
DeltaNet: every delta-rule kernel in flash-linear-attention (chunk, fused_chunk, fused_recurrent), bf16.
The proposed layer is timed the same way for reference (fla write + fused read).
"""
import argparse
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import gpu_guard
from fused_layer import inputs, layer_forward, timeit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 65536, 131072])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    from fla.ops.delta_rule import chunk_delta_rule, fused_chunk_delta_rule, fused_recurrent_delta_rule
    H, d = 4, 64
    for T in args.lengths:
        B = max(1, 16384 // T)
        steps, warm = (10, 3) if T <= 131072 else (3, 1)

        def time_mem(fn):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ms = timeit(fn, steps=steps, warm=warm)
            return round(ms, 3), round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        q, k, v, beta = inputs(B, T, H, d, torch.bfloat16)
        qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
        res = dict(T=T, B=B, tokens=B * T)
        with torch.no_grad():
            for name, be in (("sdpa_flash", SDPBackend.FLASH_ATTENTION), ("sdpa_efficient", SDPBackend.EFFICIENT_ATTENTION),
                             ("sdpa_cudnn", SDPBackend.CUDNN_ATTENTION)):
                if name == "sdpa_efficient" and T > 131072:
                    continue                                         # always the slowest backend up to 131k
                try:
                    with sdpa_kernel([be]):
                        res[name], res[name + "_gb"] = time_mem(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True))
                except Exception as e:                               # noqa: BLE001
                    res[name] = f"unavailable: {type(e).__name__}"
            for name, fn in (("delta_chunk", chunk_delta_rule), ("delta_fused_chunk", fused_chunk_delta_rule),
                             ("delta_fused_recurrent", fused_recurrent_delta_rule)):
                if name == "delta_fused_recurrent" and T > 16384:
                    continue                                         # sequential over time: slow at long lengths
                try:
                    res[name], res[name + "_gb"] = time_mem(lambda: fn(q, k, v, beta)[0])
                except Exception as e:                               # noqa: BLE001
                    res[name] = f"unavailable: {type(e).__name__}"
            res["ours_layer_bf16"], res["ours_layer_bf16_gb"] = time_mem(lambda: layer_forward(q, k, v, beta, 5, "bf16"))
        print(json.dumps(res), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
