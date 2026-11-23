#!/usr/bin/env python3
"""Squared read at every position with the per-position states kept on chip. Forward only.

The write (delta rule) is done chunk by chunk in PyTorch: it gives the pseudo-values u_t and the state
at the start of every chunk. One Triton program then takes one chunk of one head, keeps the 64x64 state
on chip, and for every position applies the rank-one update S <- S + u_t k_t^T, squares a copy r times
(rescaled by its Frobenius norm) and reads it with q_t. No per-position state is written to GPU memory.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python fused_read.py --out results/cost_fused.jsonl
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import gpu_guard


def write_states(k, v, beta, C):
    """Delta rule with S_0 = I, chunk by chunk. Returns U (B, H, T, d) and the chunk-start states (B, H, NC, d, d)."""
    B, H, T, d = k.shape
    S = torch.eye(d, device=k.device, dtype=torch.float32).expand(B, H, d, d)
    I_C = torch.eye(C, device=k.device, dtype=torch.float32)
    Us, Ss = [], []
    for c in range(T // C):
        sl = slice(c * C, (c + 1) * C)
        Kc, Vc, bc = k[:, :, sl].float(), v[:, :, sl].float(), beta[:, :, sl].float()
        A = I_C + bc[..., None] * torch.tril(Kc @ Kc.transpose(-1, -2), diagonal=-1)
        U = torch.linalg.solve_triangular(A, bc[..., None] * (Vc - Kc @ S.transpose(-1, -2)), upper=False, unitriangular=True)
        Ss.append(S)
        Us.append(U)
        S = S + U.transpose(-1, -2) @ Kc
    return torch.cat(Us, dim=2).contiguous(), torch.stack(Ss, dim=2).contiguous()


@triton.jit
def _fused_chunk_read(S0_ptr, U_ptr, K_ptr, Q_ptr, Y_ptr, T, NC,
                      R: tl.constexpr, D: tl.constexpr, C: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)                                           # one program per (batch x head, chunk)
    bh = pid // NC
    c = pid % NC
    rows = tl.arange(0, D)
    S = tl.load(S0_ptr + pid * D * D + rows[:, None] * D + rows[None, :])
    base = (bh * T + c * C) * D
    scale = 8.0                                                      # sqrt(D) for D = 64
    for t in range(C):
        u = tl.load(U_ptr + base + t * D + rows)
        k = tl.load(K_ptr + base + t * D + rows).to(tl.float32)
        q = tl.load(Q_ptr + base + t * D + rows).to(tl.float32)
        S = S + u[:, None] * k[None, :]
        M = S
        for _ in tl.static_range(R):
            if PRECISION == "bf16":
                Mb = M.to(tl.bfloat16)
                M = tl.dot(Mb, Mb, out_dtype=tl.float32)
            else:
                M = tl.dot(M, M, input_precision=PRECISION)
            M = M / (tl.sqrt(tl.sum(M * M)) / scale + 1e-6)
        y = tl.sum(M * q[None, :], axis=1)
        y = y / (tl.sqrt(tl.sum(y * y)) + 1e-6)
        tl.store(Y_ptr + base + t * D + rows, y)


def fused_read(q, k, U, S0, r, C=64, precision="tf32"):
    B, H, T, d = k.shape
    assert d == 64, "the kernel assumes d = 64"
    NC = T // C
    y = torch.empty(B, H, T, d, device=k.device, dtype=torch.float32)
    _fused_chunk_read[(B * H * NC,)](S0, U, k.contiguous(), q.contiguous(), y, T, NC, R=r, D=d, C=C, PRECISION=precision,
                                     num_warps=4)
    return y


def squared_read_reference(q, k, v, beta, r, C=64):
    """Per-position states written to memory, as in cost_read.py."""
    from cost_read import squared_read
    return squared_read(q.float(), k.float(), v.float(), beta.float(), r, C)


def timeit(fn, steps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return 1000 * ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="only compare with the reference on a small input")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    torch.manual_seed(0)
    d, H, r = 64, args.H, args.r
    if args.check:
        B, T = 2, 256
        q = torch.randn(B, H, T, d, device="cuda")
        k = F.normalize(torch.randn(B, H, T, d, device="cuda"), dim=-1)
        v = torch.randn(B, H, T, d, device="cuda") * 0.1
        beta = torch.rand(B, H, T, device="cuda") * 0.5
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = squared_read_reference(q, k, v, beta, r)
        U, S0 = write_states(k, v, beta, 64)
        for prec in ("ieee", "tf32", "bf16"):
            y = fused_read(q, k, U, S0, r, precision=prec)
            cos = (y * ref).sum(-1)
            print(f"{prec}: max |fused - reference| {float((y - ref).abs().max()):.2e}, share with cosine > 0.99: {float((cos > 0.99).float().mean()):.4f}")
        return
    from fla.ops.delta_rule import chunk_delta_rule
    for T in args.lengths:
        B = args.tokens // T
        q = torch.randn(B, H, T, d, device="cuda")
        k = F.normalize(torch.randn(B, H, T, d, device="cuda"), dim=-1)
        v = torch.randn(B, H, T, d, device="cuda") * 0.1
        beta = torch.rand(B, H, T, device="cuda") * 0.5
        qb, kb, vb = (x.bfloat16() for x in (q, k, v))
        rows = []
        with torch.no_grad():
            rows.append(("sdpa bf16, forward", timeit(lambda: F.scaled_dot_product_attention(qb, kb, vb, is_causal=True))))
            qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (qb, kb, vb))       # fla takes (B, T, H, d)
            bt = beta.bfloat16().transpose(1, 2).contiguous()
            rows.append(("delta_fla bf16, forward", timeit(lambda: chunk_delta_rule(qt, kt, vt, bt)[0])))
            rows.append(("write: delta rule chunks in PyTorch", timeit(lambda: write_states(k, v, beta, 64))))
            U, S0 = write_states(k, v, beta, 64)
            for prec in ("tf32", "bf16"):
                rows.append((f"read: fused squared read {prec}", timeit(lambda: fused_read(q, k, U, S0, r, precision=prec))))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats()
            rows.append(("unfused squared read tf32 (write + states in memory + read)", timeit(lambda: squared_read_reference(q, k, v, beta, r), steps=3, warm=1)))
            unfused_mem = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            fused_read(q, k, U, S0, r, precision="tf32")
            torch.cuda.synchronize()
            fused_mem = torch.cuda.max_memory_allocated() / 2 ** 30
        for name, ms in rows:
            res = dict(T=T, B=B, H=H, d=d, r=r, tokens=B * T, op=name, ms=round(ms, 3))
            print(json.dumps(res), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
        res = dict(T=T, B=B, op="peak memory, GB", unfused=round(unfused_mem, 2), fused_read_with_inputs=round(fused_mem, 2))
        print(json.dumps(res), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
