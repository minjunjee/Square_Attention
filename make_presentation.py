#!/usr/bin/env python3
"""Presentation tables, computed directly from the result files.

usage: python3 make_presentation.py            -> writes PRESENTATION.md and prints it

Every learning rate is chosen by mean in-distribution accuracy (plan section 5, rule 3).
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

RUNS = [json.loads(l) for l in open("results/res_server.jsonl")]
NOW = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")


def ok(r):
    return all(v >= 0.99 for v in r["acc_in_dist"].values())


def ind(r):
    return float(np.mean(list(r["acc_in_dist"].values())))


def deep(r, lo, hi=None, field="acc_deep"):
    hi = hi or lo
    return float(np.mean([r[field][str(d)] for d in range(lo, hi + 1)]))


def first_ok_step(r):
    for step, i, _, _ in r.get("curve", []):
        if all(v >= 0.99 for v in i.values()):
            return step
    return None


def chosen(pred):
    by = defaultdict(list)
    for r in RUNS:
        if pred(r):
            by[r["lr"]].append(r)
    if not by:
        return None, []
    lr = max(by, key=lambda x: np.mean([ind(r) for r in by[x]]))
    return lr, by[lr]


def p(x, digits=1):
    return "" if x is None else f"{100 * x:.{digits}f}"


tied = lambda tag, read, attn: (lambda r: r["tag"] == tag and r["model"] == "tied" and r["read"] == read
                                and bool(r.get("tap_attn")) == attn and not r.get("pre"))
out = [f"# PoC 발표용 숫자", "",
       f"생성: {NOW}. `make_presentation.py`가 `results/res_server.jsonl`에서 직접 계산한다. 정확도는 %이고, "
       "learning rate는 설정마다 3e-4, 1e-3, 3e-3을 돌려 학습 분포 평균이 가장 높은 것을 골랐다. "
       "\"99% seed\"는 학습 분포의 깊이 1에서 4를 모두 99% 이상 맞힌 seed이다. 두 체인 시험의 깊이 5에서 8은 학습에 나오지 않는다.", ""]

# 1. effect
out += ["## 표 1. 효과: 같은 layer에서 읽는 방법만 바꿨을 때", "",
        "변수 binding 문제(Wu et al., ICML 2025), 깊이 1에서 4로 학습, 30,000 step, batch 64. 실험 전에 정한 효과 기준은 "
        "두 체인 깊이 2에서 8 평균이 한 번 읽기보다 30%p 이상 높고, seed 10개 중 7개 이상이 99% seed인 것이다.", "",
        "| 설정 | 읽는 방법 | parameter | lr | 99% seed | 학습 분포 | 두 체인 2에서 8 | 두 체인 5에서 8 |", "|---|---|---|---|---|---|---|---|"]
for label, tag, attn in (("고친 설정, attention으로 고르기", "e1fix", True), ("처음 설정, 고정 비율", "e1", False)):
    for read, rname in (("square", "**제곱해서 읽기, 제안**"), ("once", "한 번 읽기"), ("iter", "순서대로 2^r번 곱하기")):
        lr, rs = chosen(tied(tag, read, attn))
        if rs:
            out.append(f"| {label} | {rname} | {rs[0]['params']:,} | {lr:g} | {sum(map(ok, rs))}/{len(rs)} | {p(np.mean([ind(r) for r in rs]))} | "
                       f"{p(np.mean([deep(r, 2, 8) for r in rs]))} | {p(np.mean([deep(r, 5, 8) for r in rs]))} |")
lr, rs = chosen(tied("e1fix", "square", True))
out += [f"| 찍기 수준 | | | | | | {p(np.mean([deep(r, 2, 8, 'guess_deep') for r in rs]))} | {p(np.mean([deep(r, 5, 8, 'guess_deep') for r in rs]))} |", "",
        "고친 설정은 효과 기준 두 조건을 모두 만족한다. 설정을 고를 때 쓰지 않은 seed 5에서 9만 세면 99% seed는 "
        f"{sum(ok(r) for r in rs if r['seed'] >= 5)}/{sum(1 for r in rs if r['seed'] >= 5)}이다.", ""]

# 2. mechanism
lr, rs = chosen(tied("e1", "square", False))
out += ["## 표 2. 작동 원리: 시험 때 제곱 횟수 r을 바꾸면 따라가는 깊이가 2^r로 바뀐다", "",
        f"처음 설정의 제곱해서 읽는 layer, lr {lr:g}, seed {len(rs)}개 평균, 두 체인 정확도. 학습 때 r은 2에서 5, 표 1의 모든 숫자는 r=6.", "",
        "| 제곱 횟수 r | S를 곱하는 횟수 | 깊이 1 | 깊이 2 | 깊이 3 | 깊이 4 | 깊이 5 | 깊이 6 | 깊이 7 | 깊이 8 |", "|---|---|---|---|---|---|---|---|---|---|"]
for k in ("0", "1", "2", "3", "4", "5", "6"):
    out.append(f"| {k} | {2 ** int(k)} | " + " | ".join(p(np.mean([r['acc_deep_by_squarings'][k][str(d)] for r in rs]), 0) for d in range(1, 9)) + " |")
out += ["", "제곱 1번이면 깊이 2까지, 2번이면 깊이 4까지, 3번이면 깊이 8까지 풀리기 시작한다.", ""]

# 3. benefit
out += ["## 표 3. 이익: 같은 학습량의 표준 모델과 비교", "",
        "100,000 step, batch 64. 진행 중이라 일부 비교군의 seed가 아직 다 끝나지 않았다. 실험 전에 정한 이익 기준은 99% seed에 닿은 표준 모델보다 "
        "두 체인 깊이 5에서 8이 20%p 이상 높거나, 정확도가 비슷하면서 같은 정확도에 닿는 step이나 시간이 절반 이하인 것이다.", "",
        "| 모델 | parameter | lr | 99% seed / 끝난 seed | 두 체인 5에서 8, 99% seed | 두 체인 깊이 8, 99% seed | 두 체인 5에서 8, 모든 seed | 99%에 처음 닿은 step 중앙값 |", "|---|---|---|---|---|---|---|---|"]
rows = [("**제안 layer, 고친 설정**", lambda r: r["tag"] == "e2fix"),
        ("제안 layer, 처음 설정", lambda r: r["tag"] == "e2" and r["model"] == "tied")]
rows += [(f"DeltaNet {L}층", (lambda L: lambda r: r["tag"] == "e2" and r["model"] == "delta" and r["L"] == L)(L)) for L in (2, 4, 8)]
rows += [("Transformer 12층, 논문 크기", lambda r: r["tag"] == "e2" and r["model"] == "tf" and r["L"] == 12),
         ("Transformer 4층", lambda r: r["tag"] == "e2" and r["model"] == "tf" and r["L"] == 4),
         ("LSTM 2층", lambda r: r["tag"] == "e2" and r["model"] == "lstm")]
for label, pred in rows:
    lr, rs = chosen(pred)
    if not rs:
        continue
    good = [r for r in rs if ok(r)]
    steps = [s for s in map(first_ok_step, good) if s is not None]
    out.append(f"| {label} | {rs[0]['params']:,} | {lr:g} | {len(good)}/{len(rs)} | {p(np.mean([deep(r, 5, 8) for r in good])) if good else '학습 부족'} | "
               f"{p(np.mean([deep(r, 8) for r in good])) if good else ''} | {p(np.mean([deep(r, 5, 8) for r in rs]))} | "
               f"{f'{int(np.median(steps)):,}' if steps else ''} |")
out += ["", "깊이 8의 찍기 수준은 50%이다. 99% seed가 없는 모델은 계획서 5절의 규칙대로 학습 부족으로 보고 비교에 쓰지 않는다.", ""]

# 4. looped vs squared
lr_l, loop_rs = chosen(lambda r: r["tag"] == "e3" and r["model"] == "delta")
lr_s, sq_rs = chosen(lambda r: r["tag"] == "e2fix")
out += ["## 표 4. 같은 block을 반복하는 모델과의 비교", "",
        f"둘 다 100,000 step, batch 64. 반복 DeltaNet: DeltaNet block 하나를 학습 때 4번 적용, lr {lr_l:g}, seed {len(loop_rs)}개 평균. "
        f"제곱해서 읽기: 고친 설정, lr {lr_s:g}, seed {len(sq_rs)}개 평균. 두 체인 정확도.", "",
        "| 모델 | 시험 때 | 깊이 2 | 깊이 4 | 깊이 6 | 깊이 8 |", "|---|---|---|---|---|---|"]
for k in ("1", "2", "4", "8", "16"):
    out.append(f"| 반복 DeltaNet | block {k}번 적용 | " + " | ".join(p(np.mean([r['acc_deep_by_loops'][k][str(d)] for r in loop_rs]), 0) for d in (2, 4, 6, 8)) + " |")
for k in ("1", "2", "3", "4", "5"):
    out.append(f"| 제곱해서 읽기 | 제곱 {k}번, S를 {2 ** int(k)}번 곱한 것 | " + " | ".join(p(np.mean([r['acc_deep_by_squarings'][k][str(d)] for r in sq_rs]), 0) for d in (2, 4, 6, 8)) + " |")
out += ["", "반복 모델은 깊이 k에 block k번이 필요하고 반복을 늘려도 깊이 8에서 멈춘다. 제곱은 깊이의 로그만큼의 단계로 가고, "
        "block 전체를 다시 계산하는 대신 64×64 행렬만 곱한다.", ""]

# 5. variable tracking format
def vt_pred(cond, model, read=None, pre=None):
    def f(r):
        if r["tag"] != "vt" or r["model"] != model:
            return False
        v = r["vt"]
        c = {(1, 0): "A", (1, 256): "B", (3, 0): "C", (3, 256): "D"}[(v["name_max"], v["prog_len"])]
        if c != cond or (read and r.get("read") != read):
            return False
        return (r.get("pre_kind") or "tf") == pre if pre else not r.get("pre")
    return f


out += ["## 표 5. 덜 규칙적인 형식: 여러 token 이름과 방해 token", "",
        "`VAR <이름> = <숫자 또는 이름> ;` 형식. A 이름 1 token, B 이름 1 token과 방해 token, C 이름 1에서 3 token, D 둘 다. "
        "lr 1e-3, 30,000 step으로 줄인 실험이다. 마지막 열은 seed별 두 체인 깊이 5에서 8이다.", "",
        "| 조건 | 모델 | 99% seed | 두 체인 2에서 4 평균 | 두 체인 5에서 8 평균 | seed별 두 체인 5에서 8 |", "|---|---|---|---|---|---|"]
for cond, label, pred in (("A", "제안 layer", vt_pred("A", "tied", "square")), ("A", "DeltaNet 4층", vt_pred("A", "delta")),
                          ("B", "제안 layer", vt_pred("B", "tied", "square")), ("C", "제안 layer", vt_pred("C", "tied", "square")),
                          ("D", "제안 layer", vt_pred("D", "tied", "square")), ("D", "DeltaNet 4층", vt_pred("D", "delta")),
                          ("B", "DeltaNet 앞 layer + 제곱해서 읽기", vt_pred("B", "tied", "square", "delta")),
                          ("C", "DeltaNet 앞 layer + 제곱해서 읽기", vt_pred("C", "tied", "square", "delta")),
                          ("D", "DeltaNet 앞 layer + 제곱해서 읽기", vt_pred("D", "tied", "square", "delta")),
                          ("D", "DeltaNet 앞 layer + 한 번 읽기", vt_pred("D", "tied", "once", "delta"))):
    rs = sorted([r for r in RUNS if pred(r)], key=lambda r: r["seed"])
    if rs:
        out.append(f"| {cond} | {label} | {sum(map(ok, rs))}/{len(rs)} | {p(np.mean([deep(r, 2, 4) for r in rs]))} | {p(np.mean([deep(r, 5, 8) for r in rs]))} | "
                   + ", ".join(p(deep(r, 5, 8), 0) for r in rs) + " |")
out += ["", "깨끗한 형식에서는 완벽하다. 여러 token 이름에서 무너지는 원인은 이름을 key 하나로 만드는 부분이고, 앞 layer 하나로 해결된다(C). "
        "방해 token까지 있으면 제곱해서 읽을 때만 일부 seed가 회복한다(D). 방해 token 문제는 아직 풀지 못했다(B).", ""]

# 6. GPU
kern = {r["impl"]: r for r in map(json.loads, open("results/cost_kernel.jsonl"))}
layer = {}
for line in open("results/cost_layer.jsonl"):
    r = json.loads(line)
    layer[r["T"]] = r                                                # the last measurement of each length
unfused = {r["T"]: r["ms"] for r in map(json.loads, open("results/cost_fused.jsonl")) if r.get("op", "").startswith("unfused")}
best = {}
for line in open("results/cost_best.jsonl"):
    r = json.loads(line)
    best[r["T"]] = r                                                 # the last measurement of each length


def fastest(r, keys):
    vals = [(r[k], k) for k in keys if isinstance(r.get(k), (int, float))]
    return min(vals) if vals else (None, None)


def ratio(a, b):
    return f"{a / b:.1f}배 느림" if a >= b else f"**{b / a:.1f}배 빠름**"


names = {"sdpa_flash": "FlashAttention", "sdpa_cudnn": "cuDNN", "sdpa_efficient": "memory-efficient",
         "delta_chunk": "chunk", "delta_fused_recurrent": "fused_recurrent", "delta_fused_chunk": "fused_chunk"}
out += ["## 표 6. GPU 가속: 모든 위치에서 읽는 layer 전체의 속도", "",
        "RTX PRO 6000 Blackwell 한 장을 다른 실행 없이 썼다. head 4개, d=64, r=5, bf16, 순전파. 길이 16,384까지는 한 번에 16,384 token, 그보다 길면 sequence 하나이다. "
        "비교군은 각자 가장 빠른 구현을 썼다. softmax attention은 PyTorch SDPA의 FlashAttention, cuDNN, memory-efficient 백엔드 중 가장 빠른 것이고, "
        "DeltaNet은 flash-linear-attention의 chunk와 fused_recurrent 커널 중 가장 빠른 것이다. fla의 fused_chunk 커널은 이 버전에서 호출 형식이 맞지 않아 재지 못했다. "
        "제안 layer는 쓰기(delta rule)를 DeltaNet과 같은 fla 커널로, 읽기를 상태를 on-chip에 둔 채 제곱하는 Triton 커널로 한다. "
        "쓰기가 DeltaNet과 같으므로 제안 layer는 구조상 DeltaNet보다 빠를 수 없다.", "",
        "| 길이 T | softmax attention, 가장 빠른 구현 | DeltaNet, 가장 빠른 구현 | **제안 layer** | 제안 대 DeltaNet | 제안 대 attention | 합치기 전 PyTorch | 커널로 빨라진 배수 |",
        "|---|---|---|---|---|---|---|---|"]
for T in sorted(best):
    r = best[T]
    att, att_k = fastest(r, ("sdpa_flash", "sdpa_cudnn", "sdpa_efficient"))
    dn, dn_k = fastest(r, ("delta_chunk", "delta_fused_chunk", "delta_fused_recurrent"))
    ours = r["ours_layer_bf16"]
    un = unfused.get(T)
    out.append(f"| {T:,} | {att:.2f} ms, {names[att_k]} | {dn:.2f} ms, {names[dn_k]} | **{ours:.2f} ms** | {ratio(ours, dn)} | {ratio(ours, att)} | "
               f"{f'{un:.1f} ms' if un else ''} | {f'{un / ours:.0f}배' if un else ''} |")
slow = [best[T]["ours_layer_bf16"] / fastest(best[T], ("delta_chunk", "delta_fused_chunk", "delta_fused_recurrent"))[0] for T in best]
out += ["", f"제안 layer는 모든 길이에서 DeltaNet보다 {min(slow):.1f}배에서 {max(slow):.1f}배 느리다. 시간이 token 수에 비례하는 반면 softmax attention은 길이의 제곱으로 늘어서, "
        "약 3만 token부터 제안 layer가 가장 빠른 attention 구현보다 빠르다. 13만 token에서 제안 layer의 메모리는 "
        + f"{layer[max(layer)]['ours_layer_peak_gb']:.2f} GB이다.", "",
        "| 읽기 커널만, 학습된 상태 65,536개 | 시간 | 처리량, 추정 | float64 기준과 cosine 0.99 이상 |", "|---|---|---|---|"]
for k, nm in (("torch fp32", "PyTorch fp32"), ("torch bf16", "PyTorch bf16"), ("fused tf32", "합친 Triton 커널 TF32"), ("fused bf16", "**합친 Triton 커널 bf16**")):
    r = kern[k]
    out.append(f"| {nm} | {r['ms']:.2f} ms | {r['tflops']:.0f} TFLOP/s | {p(r['frac_cos_above_0_99'], 0)}% |")


text = "\n".join(out) + "\n"
open("PRESENTATION.md", "w").write(text)
print(text)
