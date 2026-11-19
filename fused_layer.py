#!/usr/bin/env python3
"""The whole squared-read layer on the GPU: write with the DeltaNet kernels of flash-linear-attention,
read with a fused Triton kernel that keeps each 64x64 state on chip. Inputs in (B, T, H, d) layout.

Forward: fla computes the chunk-start states and the delta-rule pseudo-values; one Triton program per
(sequence, head, chunk) rebuilds every per-position state on chip, squares it r times (rescaled by its
Frobenius norm) and reads it with q. No per-position state is written to GPU memory.

Backward (training): memory-saving, not fused. The write is recomputed in PyTorch with autograd; the
read is a custom function whose forward is the Triton kernel and whose backward recomputes one chunk
at a time in PyTorch. Memory stays at one chunk of per-position states.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python fused_layer.py --check
       CUDA_VISIBLE_DEVICES=<allowed GPU UUID> .venv_tc/bin/python fused_layer.py --bench --out results/cost_layer.jsonl
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

C = 64


@triton.jit
def _read_bthd(H_ptr, U_ptr, K_ptr, Q_ptr, Y_ptr, T, NT, NH,
               R: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)                                           # pid = (b * NT + c) * NH + h
    hh = pid % NH
    c = (pid // NH) % NT
    b = pid // (NH * NT)
    rows = tl.arange(0, D)
    # chunk-start state from fla, stored as h = S^T (K x V); load it transposed into S (V x K)
    S = tl.load(H_ptr + pid * D * D + rows[None, :] * D + rows[:, None]).to(tl.float32)
    scale = 8.0                                                      # sqrt(D) for D = 64
    for t in range(CH):
        off = ((b * T + c * CH + t) * NH + hh) * D
        u = tl.load(U_ptr + off + rows).to(tl.float32)
        k = tl.load(K_ptr + off + rows).to(tl.float32)
        q = tl.load(Q_ptr + off + rows).to(tl.float32)
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
        tl.store(Y_ptr + off + rows, y)


def fla_write(k, v, beta):
    """Delta rule with S_0 = I through fla's kernels. Returns chunk-start states h (B, NT, H, d, d), stored
    as S^T, and pseudo-values u (B, T, H, d) measured from each chunk's start state."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd
    B, T, H, d = k.shape
    h0 = torch.eye(d, device=k.device, dtype=torch.float32).expand(B, H, d, d).contiguous()
    w, u, _ = prepare_wy_repr_fwd(k=k, v=v, beta=beta, cu_seqlens=None, chunk_size=C)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None, initial_state=h0, output_final_state=False, chunk_size=C)
    return h, v_new


def fused_read(q, k, h, u, r, precision="bf16"):
    B, T, H, d = k.shape
    assert d == 64 and T % C == 0
    NT = T // C
    y = torch.empty(B, T, H, d, device=k.device, dtype=torch.float32)
    _read_bthd[(B * NT * H,)](h.contiguous(), u.contiguous(), k.contiguous(), q.contiguous(), y, T, NT, H,
                              R=r, D=d, CH=C, PRECISION=precision, num_warps=4)
    return y


def layer_forward(q, k, v, beta, r, precision="bf16"):
    h, u = fla_write(k, v, beta)
    return fused_read(q, k, h, u, r, precision)


# ----------------------------------------------------------------------------- training path
def read_chunk_torch(S0, U, K, Q, r):
    """PyTorch read of one chunk. S0 (B, H, d, d) maps k to v; U, K, Q (B, H, C, d)."""
    d = K.shape[-1]
    M = S0[:, :, None] + torch.cumsum(U[..., :, None] * K[..., None, :], dim=2)
    for _ in range(r):
        M = M @ M
        M = M / (M.flatten(-2).norm(dim=-1)[..., None, None] / math.sqrt(d) + 1e-6)
    return F.normalize((M @ Q[..., None]).squeeze(-1), dim=-1, eps=1e-6)


class FusedChunkRead(torch.autograd.Function):
    """Forward: Triton kernel on one chunk. Backward: recompute that chunk in PyTorch."""

    @staticmethod
    def forward(ctx, S0, U, K, Q, r, precision):
        ctx.save_for_backward(S0, U, K, Q)
        ctx.r = r
        B, H, Cc, d = K.shape
        h = S0.transpose(-1, -2)[:, None].contiguous()               # (B, 1, H, d, d) as S^T, one chunk
        return fused_read(Q.transpose(1, 2), K.transpose(1, 2), h, U.transpose(1, 2), r, precision).transpose(1, 2)

    @staticmethod
    def backward(ctx, gy):
        S0, U, K, Q = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [x.detach().requires_grad_() for x in (S0, U, K, Q)]
            y = read_chunk_torch(*leaves, ctx.r)
            grads = torch.autograd.grad(y, leaves, gy)
        return (*grads, None, None)


def write_torch(k, v, beta):
    """Differentiable chunked delta rule in PyTorch (B, H, T, d). Returns per-chunk (S_start, U)."""
    B, H, T, d = k.shape
    S = torch.eye(d, device=k.device, dtype=k.dtype).expand(B, H, d, d)
    I_C = torch.eye(C, device=k.device, dtype=k.dtype)
    out = []
    for c in range(T // C):
        sl = slice(c * C, (c + 1) * C)
        Kc, Vc, bc = k[:, :, sl], v[:, :, sl], beta[:, :, sl]
        A = I_C + bc[..., None] * torch.tril(Kc @ Kc.transpose(-1, -2), diagonal=-1)
        U = torch.linalg.solve_triangular(A, bc[..., None] * (Vc - Kc @ S.transpose(-1, -2)), upper=False, unitriangular=True)
        out.append((S, U))
        S = S + U.transpose(-1, -2) @ Kc
    return out


def layer_train(q, k, v, beta, r, precision="bf16"):
    """Training path, (B, H, T, d) layout: PyTorch write with autograd, fused read per chunk with recompute."""
    ys = []
    for c, (S, U) in enumerate(write_torch(k, v, beta)):
        sl = slice(c * C, (c + 1) * C)
        ys.append(FusedChunkRead.apply(S, U, k[:, :, sl], q[:, :, sl], r, precision))
    return torch.cat(ys, dim=2)


def layer_reference(q, k, v, beta, r):
    """Plain PyTorch with autograd everywhere, (B, H, T, d)."""
    ys = []
    for c, (S, U) in enumerate(write_torch(k, v, beta)):
        sl = slice(c * C, (c + 1) * C)
        ys.append(read_chunk_torch(S, U, k[:, :, sl], q[:, :, sl], r))
    return torch.cat(ys, dim=2)


# ----------------------------------------------------------------------------- checks and timing
def inputs(B, T, H, d, dtype, layout="bthd", grad=False):
    shape = (B, T, H, d) if layout == "bthd" else (B, H, T, d)
    q = torch.randn(shape, device="cuda")
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1)
    v = torch.randn(shape, device="cuda") * 0.1
    beta = torch.rand(shape[:3], device="cuda") * 0.5
    out = [x.to(dtype) for x in (q, k, v, beta)]
    return [x.requires_grad_() for x in out] if grad else out


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


def check(r):
    torch.manual_seed(0)
    B, T, H, d = 2, 256, 4, 64
    q, k, v, beta = inputs(B, T, H, d, torch.float32, "bhtd")
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = layer_reference(q, k, v, beta, r)
    for prec in ("ieee", "tf32", "bf16"):
        y = layer_forward(*(x.transpose(1, 2).contiguous() for x in (q, k, v)), beta.transpose(1, 2).contiguous(), r, prec).transpose(1, 2)
        cos = (y * ref).sum(-1)
        print(f"forward {prec}: max |layer - reference| {float((y - ref).abs().max()):.2e}, cosine > 0.99: {float((cos > 0.99).float().mean()):.4f}")
    qg, kg, vg, bg = inputs(B, T, H, d, torch.float32, "bhtd", grad=True)
    g = torch.randn(B, H, T, d, device="cuda")
    (layer_reference(qg, kg, vg, bg, r) * g).sum().backward()
    ref_grads = [x.grad.clone() for x in (qg, kg, vg, bg)]
    for x in (qg, kg, vg, bg):
        x.grad = None
    (layer_train(qg, kg, vg, bg, r, "ieee") * g).sum().backward()
    for name, a, b in zip(("q", "k", "v", "beta"), ref_grads, (qg.grad, kg.grad, vg.grad, bg.grad)):
        rel = float((a - b).norm() / (a.norm() + 1e-12))
        print(f"gradient {name}: relative difference to plain PyTorch {rel:.2e}")


def bench(r, out_path):
    from fla.ops.delta_rule import chunk_delta_rule
    H, d = 4, 64
    rows = []
    for T in (1024, 4096, 16384, 65536, 131072):
        B = max(1, 16384 // T)
        q, k, v, beta = inputs(B, T, H, d, torch.bfloat16)
        qh = q.transpose(1, 2)
        with torch.no_grad():
            res = dict(T=T, B=B, tokens=B * T, H=H, d=d, r=r)
            res["flash_attention_fwd"] = timeit(lambda: F.scaled_dot_product_attention(qh, k.transpose(1, 2), v.transpose(1, 2), is_causal=True))
            res["deltanet_fla_fwd"] = timeit(lambda: chunk_delta_rule(q, k, v, beta)[0])
            res["ours_write_fla"] = timeit(lambda: fla_write(k, v, beta))
            h, u = fla_write(k, v, beta)
            res["ours_read_bf16"] = timeit(lambda: fused_read(q, k, h, u, r, "bf16"))
            res["ours_read_tf32"] = timeit(lambda: fused_read(q, k, h, u, r, "tf32"))
            res["ours_layer_bf16"] = timeit(lambda: layer_forward(q, k, v, beta, r, "bf16"))
            torch.cuda.reset_peak_memory_stats()
            layer_forward(q, k, v, beta, r, "bf16")
            torch.cuda.synchronize()
            res["ours_layer_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        if T <= 16384:                                               # training step: forward and backward
            qf, kf, vf, bf = inputs(B, T, H, d, torch.bfloat16, grad=True)
            res["flash_attention_fwd_bwd"] = timeit(lambda: F.scaled_dot_product_attention(qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2), is_causal=True).sum().backward(), steps=5)
            res["deltanet_fla_fwd_bwd"] = timeit(lambda: chunk_delta_rule(qf, kf, vf, bf)[0].float().sum().backward(), steps=5)
            qt, kt, vt, btt = inputs(B, T, H, d, torch.float32, "bhtd", grad=True)
            torch.cuda.reset_peak_memory_stats()
            res["ours_train_fwd_bwd"] = timeit(lambda: layer_train(qt, kt, vt, btt, r, "bf16").sum().backward(), steps=3, warm=1)
            res["ours_train_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        res = {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in res.items()}
        print(json.dumps(res), flush=True)
        rows.append(res)
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--r", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    if args.check:
        check(args.r)
    if args.bench:
        bench(args.r, args.out)


if __name__ == "__main__":
    main()
