#!/usr/bin/env python3
"""A fused GPU kernel for the squared read, forward only.

For N states M (N, d, d) and read vectors q (N, d): r times M <- M @ M rescaled by its Frobenius norm,
then y = unit(M q). The PyTorch version writes every intermediate M to GPU memory; the kernel keeps
each 64x64 matrix on chip for all r squarings.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python squared_read_triton.py --out results/cost_kernel.jsonl
"""
import argparse
import json
import math
import time

import torch
import triton
import triton.language as tl

import gpu_guard


@triton.jit
def _squared_read_kernel(M_ptr, q_ptr, y_ptr, R: tl.constexpr, D: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, D)
    offs = rows[:, None] * D + rows[None, :]
    M = tl.load(M_ptr + pid * D * D + offs).to(tl.float32)
    scale = math.sqrt(D)
    for _ in tl.static_range(R):
        if PRECISION == "bf16":
            Mb = M.to(tl.bfloat16)
            M = tl.dot(Mb, Mb, out_dtype=tl.float32)
        else:
            M = tl.dot(M, M, input_precision=PRECISION)
        M = M / (tl.sqrt(tl.sum(M * M)) / scale + 1e-6)
    q = tl.load(q_ptr + pid * D + rows).to(tl.float32)
    y = tl.sum(M * q[None, :], axis=1)
    y = y / (tl.sqrt(tl.sum(y * y)) + 1e-6)
    tl.store(y_ptr + pid * D + rows, y)


def squared_read_fused(M, q, r, precision="tf32"):
    N, D, _ = M.shape
    y = torch.empty(N, D, device=M.device, dtype=torch.float32)
    _squared_read_kernel[(N,)](M.contiguous(), q.contiguous(), y, R=r, D=D, PRECISION=precision, num_warps=4)
    return y


def squared_read_torch(M, q, r):
    d = M.shape[-1]
    for _ in range(r):
        M = M @ M
        M = M / (M.flatten(-2).norm(dim=-1)[..., None, None] / math.sqrt(d) + 1e-6)
    y = (M @ q[..., None]).squeeze(-1)
    return y / (y.norm(dim=-1, keepdim=True) + 1e-6)


def timeit(fn, steps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return 1000 * ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=65536, help="number of (position, head) pairs, e.g. 16,384 tokens x 4 heads")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--ckpt", default=None, help="take the states from this trained model (attention choice) on real programs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    torch.manual_seed(0)
    N, d, r = args.N, args.d, args.r
    if args.ckpt:                                                  # states a trained model builds on real programs
        import numpy as np
        import poc_fair as P
        m = P.Tied(16, 4, read="square", gate_ctx=True, tap_attn=8).cuda()
        m.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
        m.tap_scale = 10.0
        tok, _, _, _ = P.make_batch(np.random.default_rng(0), N // 4, twin_depth=8)
        with torch.no_grad():
            e = m.emb(torch.from_numpy(tok).cuda())
            k, v, qv = m.roles(e)
            beta = torch.sigmoid(m.gate(e.transpose(1, 2))[:, 0, :e.shape[1]])
            Ufull = P.delta_updates(k, v, beta, 1.0)
            Ms, qs = [], []
            for pos in P.read_positions(16, 4):
                Ms.append(torch.eye(d, device="cuda") + Ufull[:, :pos + 1].transpose(1, 2) @ k[:, :pos + 1])
                qs.append(qv[:, pos])
        M, q = torch.cat(Ms).contiguous(), torch.cat(qs).contiguous()
        N = M.shape[0]
    else:                                                          # random states: identity plus a few rank-one updates
        U = torch.randn(N, 8, d, device="cuda") * 0.3
        K = torch.nn.functional.normalize(torch.randn(N, 8, d, device="cuda"), dim=-1)
        M = torch.eye(d, device="cuda") + U.transpose(1, 2) @ K
        q = torch.nn.functional.normalize(torch.randn(N, d, device="cuda"), dim=-1)
    ref = squared_read_torch(M.double(), q.double(), r).float()
    flops = N * r * 2 * d ** 3
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    rows.append(("torch fp32", timeit(lambda: squared_read_torch(M, q, r)), squared_read_torch(M, q, r)))
    torch.backends.cuda.matmul.allow_tf32 = True
    rows.append(("torch tf32", timeit(lambda: squared_read_torch(M, q, r)), squared_read_torch(M, q, r)))
    torch.backends.cuda.matmul.allow_tf32 = False
    Mb, qb = M.bfloat16(), q.bfloat16()
    rows.append(("torch bf16", timeit(lambda: squared_read_torch(Mb, qb, r)), squared_read_torch(Mb, qb, r).float()))
    for prec in ("ieee", "tf32", "bf16"):
        rows.append((f"fused {prec}", timeit(lambda: squared_read_fused(M, q, r, prec)), squared_read_fused(M, q, r, prec)))
    for name, ms, y in rows:
        agree = float(((y @ ref.T).diagonal() > 0.99).float().mean()) if args.ckpt else None
        res = dict(impl=name, states="trained" if args.ckpt else "random", N=N, d=d, r=r, frac_cos_above_0_99=agree, ms=round(ms, 3), tflops=round(flops / (ms / 1000) / 1e12, 1),
                   max_abs_err=float((y - ref).abs().max()), gpu=torch.cuda.get_device_name())
        print(json.dumps(res), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
