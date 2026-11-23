#!/usr/bin/env python3
"""How well does the squared read map onto a GPU when every position reads, as in a language model?

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python cost_read.py --out results/cost_read.jsonl

One token-mixing layer, forward and backward, with the same number of tokens per batch at every
sequence length T. Heads H=4, head dimension d=64.
  sdpa            softmax attention, causal, bf16 (FlashAttention through torch SDPA)
  delta_fla       DeltaNet, delta rule with the standard read, bf16 (flash-linear-attention chunk kernel)
  squared/<prec>  delta rule written chunk by chunk, then every position builds its own state S_t and
                  squares it r times before reading (this project's read), in fp32, tf32 or bf16
  iter/<prec>     the same states read by 2^r matrix-vector products instead of r squarings
Needs a GPU on which nothing else runs, otherwise the times mean little.
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import gpu_guard


def chunk_states(q, k, v, beta, C):
    """Delta rule S_t = S_(t-1) + beta_t (v_t - S_(t-1) k_t) k_t^T with S_0 = I, computed chunk by chunk.
    Yields, for each chunk, the per-position states (B, H, C, d, d) and the read vectors (B, H, C, d)."""
    B, H, T, d = k.shape
    S = torch.eye(d, device=k.device, dtype=k.dtype).expand(B, H, d, d)
    I_C = torch.eye(C, device=k.device, dtype=k.dtype)
    for c in range(T // C):
        sl = slice(c * C, (c + 1) * C)
        Kc, Vc, Qc, bc = k[:, :, sl], v[:, :, sl], q[:, :, sl], beta[:, :, sl]
        A = I_C + bc[..., None] * torch.tril(Kc @ Kc.transpose(-1, -2), diagonal=-1)
        rhs = bc[..., None] * (Vc - Kc @ S.transpose(-1, -2))
        U = torch.linalg.solve_triangular(A.float(), rhs.float(), upper=False, unitriangular=True).to(k.dtype)
        St = S[:, :, None] + torch.cumsum(U[..., :, None] * Kc[..., None, :], dim=2)
        yield St, Qc
        S = St[:, :, -1]


def squared_read(q, k, v, beta, r, C=64):
    d = k.shape[-1]
    outs = []
    for St, Qc in chunk_states(q, k, v, beta, C):
        M = St
        for _ in range(r):
            M = M @ M
            M = M / (M.flatten(-2).norm(dim=-1)[..., None, None] / math.sqrt(d) + 1e-6)
        outs.append(F.normalize((M @ Qc[..., None]).squeeze(-1), dim=-1))
    return torch.cat(outs, dim=2)


def iter_read(q, k, v, beta, r, C=64):
    outs = []
    for St, Qc in chunk_states(q, k, v, beta, C):
        y = Qc[..., None]
        for _ in range(2 ** r):
            y = F.normalize(St @ y, dim=-2)
        outs.append(y.squeeze(-1))
    return torch.cat(outs, dim=2)


def bench(name, fn, make_inputs, steps=10, warm=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for i in range(warm + steps):
            inputs = make_inputs()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*inputs)
            out.float().sum().backward()
            torch.cuda.synchronize()
            if i >= warm:
                times.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        return dict(op=name, error="out of memory")
    times.sort()
    return dict(op=name, ms=round(1000 * times[len(times) // 2], 2), ms_min=round(1000 * times[0], 2),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384, help="tokens per batch at every length")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    from fla.ops.delta_rule import chunk_delta_rule
    dev, H, d = "cuda", args.H, args.d
    for T in args.lengths:
        B = max(1, args.tokens // T)

        def inputs(dtype, layout="bhtd"):
            def make():
                shape = (B, H, T, d) if layout == "bhtd" else (B, T, H, d)
                q = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
                k = F.normalize(torch.randn(shape, device=dev, dtype=torch.float32), dim=-1).to(dtype).requires_grad_()
                v = torch.randn(shape, device=dev, dtype=dtype, requires_grad=True)
                bshape = (B, H, T) if layout == "bhtd" else (B, T, H)
                beta = torch.rand(bshape, device=dev, dtype=dtype).requires_grad_()
                return q, k, v, beta
            return make

        cases = [
            ("sdpa/bf16", lambda q, k, v, b: F.scaled_dot_product_attention(q, k, v, is_causal=True), inputs(torch.bfloat16), None),
            ("delta_fla/bf16", lambda q, k, v, b: chunk_delta_rule(q, k, v, b)[0], inputs(torch.bfloat16, "bthd"), None),
            ("squared/fp32", lambda q, k, v, b: squared_read(q, k, v, b, args.r), inputs(torch.float32), False),
            ("squared/tf32", lambda q, k, v, b: squared_read(q, k, v, b, args.r), inputs(torch.float32), True),
            ("squared/bf16", lambda q, k, v, b: squared_read(q, k, v, b, args.r), inputs(torch.bfloat16), None),
            ("iter/tf32", lambda q, k, v, b: iter_read(q, k, v, b, args.r), inputs(torch.float32), True),
        ]
        for name, fn, make, tf32 in cases:
            if args.only and args.only not in name:
                continue
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            res = bench(name, fn, make)
            res.update(T=T, B=B, tokens=B * T, H=H, d=d, r=args.r, gpu=torch.cuda.get_device_name())
            if "ms" in res:
                res["tokens_per_s"] = round(B * T / (res["ms"] / 1000))
            print(json.dumps(res), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
        torch.backends.cuda.matmul.allow_tf32 = False


if __name__ == "__main__":
    main()
