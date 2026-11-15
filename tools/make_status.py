#!/usr/bin/env python3
"""Regenerates the status tables from results/res_server.jsonl and the pool logs.

usage: python3 tools/make_status.py [--splice]

Writes results/STATUS.md. With --splice, also replaces the text between the markers
<!-- AUTO:name --> and <!-- /AUTO:name --> in SERVER_RESULTS.md with the table of that name.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KST = timezone(timedelta(hours=9))


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def runs():
    return [json.loads(line) for line in open(os.path.join(ROOT, "results/res_server.jsonl"))]


def ok(r):
    return all(v >= 0.99 for v in r["acc_in_dist"].values())


def deep(r, lo, hi):
    return float(np.mean([r["acc_deep"][str(d)] for d in range(lo, hi + 1)]))


def first_ok_step(r):
    for step, ind, _, _ in r.get("curve", []):
        if all(v >= 0.99 for v in ind.values()):
            return step
    return None


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def e2_table(rs):
    from summarize import name, setting
    groups = defaultdict(lambda: defaultdict(list))
    for r in rs:
        if r["tag"] in ("e2", "e2fix"):
            groups[(r["tag"],) + setting(r)][r["lr"]].append(r)
    rows = []
    for key, by_lr in groups.items():
        lr = max(by_lr, key=lambda x: np.mean([np.mean(list(r["acc_in_dist"].values())) for r in by_lr[x]]))
        g = by_lr[lr]
        good = [r for r in g if ok(r)]
        steps = [s for s in (first_ok_step(r) for r in good) if s is not None]
        nm = name(g[0]) + (", 고친 설정" if key[0] == "e2fix" else "")
        rows.append((g[0]["model"] != "tied", nm, g[0]["params"], lr, len(g), len(good),
                     np.mean([deep(r, 5, 8) for r in good]) if good else None, np.mean([deep(r, 5, 8) for r in g]),
                     int(np.median(steps)) if steps else None, sorted(by_lr)))
    rows.sort()
    out = ["| 설정 | parameter | 끝난 lr | 고른 lr | 끝난 seed | 99%에 닿은 seed | 두 체인 5에서 8, 99% seed만 | 두 체인 5에서 8, 모든 seed | 99%에 처음 닿은 step 중앙값 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, nm, p, lr, n, nok, d_ok, d_all, st, lrs in rows:
        out.append(f"| {nm} | {p:,} | {', '.join(f'{x:g}' for x in lrs)} | {lr:g} | {n} | {nok} | {pct(d_ok)} | {pct(d_all)} | {'' if st is None else f'{st:,}'} |")
    std = [x for x in rows if x[0] and x[5] > 0]
    notes = []
    if std:
        best = max(std, key=lambda x: x[6])
        for x in rows:
            if not x[0] and x[6] is not None:
                notes.append(f"{x[1]}: 99%에 닿은 표준 모델 중 가장 나은 {best[1]}({pct(best[6])})보다 {100 * (x[6] - best[6]):+.1f}%p.")
    return "\n".join(out) + ("\n\n" + " ".join(notes) if notes else "")


def vt_label(r):
    v = r["vt"]
    return {(1, 0): "A", (1, 256): "B", (3, 0): "C", (3, 256): "D"}[(v["name_max"], v["prog_len"])]


def vt_tables(rs):
    base, fix = defaultdict(list), defaultdict(list)
    for r in rs:
        if r["tag"] != "vt":
            continue
        model = {"tied": "제안 layer" if r["read"] == "square" else "같은 layer, 한 번 읽기",
                 "delta": f"DeltaNet {r['L']}층", "tf": f"Transformer {r['L']}층", "lstm": "LSTM"}[r["model"]]
        if r.get("pre"):
            kind = {"tf": "Transformer", "delta": "DeltaNet", "conv": "conv"}[r.get("pre_kind") or "tf"]
            fix[(vt_label(r), kind, "제곱" if r["read"] == "square" else "한 번")].append(r)
        else:
            base[(vt_label(r), model)].append(r)

    def row(rs_):
        rs_ = sorted(rs_, key=lambda r: r["seed"])
        return (", ".join(str(r["seed"]) for r in rs_), pct(np.mean([np.mean(list(r["acc_in_dist"].values())) for r in rs_])),
                f"{sum(map(ok, rs_))}/{len(rs_)}", ", ".join(f"{100 * deep(r, 2, 4):.0f}" for r in rs_),
                ", ".join(f"{100 * deep(r, 5, 8):.0f}" for r in rs_))

    b = ["| 조건 | 모델 | 끝난 seed | 학습 분포 평균 | 99%에 닿은 seed | 두 체인 2에서 4, seed별 | 두 체인 5에서 8, seed별 |", "|---|---|---|---|---|---|---|"]
    order = {"제안 layer": 0, "같은 layer, 한 번 읽기": 1}
    for (c, m), rs_ in sorted(base.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 2), kv[0][1])):
        b.append(f"| {c} | {m} | " + " | ".join(row(rs_)) + " |")
    f = ["| 조건 | 앞 layer | 읽는 방법 | 끝난 seed | 학습 분포 평균 | 99%에 닿은 seed | 두 체인 2에서 4, seed별 | 두 체인 5에서 8, seed별 |", "|---|---|---|---|---|---|---|---|"]
    for (c, k, rd), rs_ in sorted(fix.items()):
        f.append(f"| {c} | {k} | {rd} | " + " | ".join(row(rs_)) + " |")
    return "\n".join(b), "\n".join(f)


def e3_table(rs):
    g = defaultdict(list)
    for r in rs:
        if r["tag"] == "e3":
            nm = ("DeltaNet block" if r["model"] == "delta" else "Transformer block") + \
                 (f", 학습 때 {r['loops_train'][0]}에서 {r['loops_train'][1]}번" if r.get("loops_train") else f", 학습 때 {r['loops']}번")
            g[(nm, r["lr"])].append(r)
    out = ["| 모델 | lr | 끝난 seed | 학습 분포 평균 | 99%에 닿은 seed | 두 체인 5에서 8, 기본 반복 횟수 | 반복 1, 2, 4, 8, 16번일 때 두 체인 5에서 8 평균 |",
           "|---|---|---|---|---|---|---|"]
    for (nm, lr), rs_ in sorted(g.items()):
        by = defaultdict(list)
        for r in rs_:
            for k, v in r.get("acc_deep_by_loops", {}).items():
                by[int(k)].append(np.mean([v[str(d)] for d in range(5, 9)]))
        sweep = ", ".join(f"{100 * np.mean(by[k]):.0f}" for k in (1, 2, 4, 8, 16) if k in by)
        out.append(f"| {nm} | {lr:g} | {', '.join(str(r['seed']) for r in sorted(rs_, key=lambda r: r['seed']))} | "
                   f"{pct(np.mean([np.mean(list(r['acc_in_dist'].values())) for r in rs_]))} | {sum(map(ok, rs_))}/{len(rs_)} | "
                   f"{pct(np.mean([deep(r, 5, 8) for r in rs_]))} | {sweep} |")
    return "\n".join(out)


def e8_table(rs):
    """Interference test: one row per (model, d, n), with the verdict rule fixed in SERVER_RESULTS.md 4.8."""
    def deep_or_none(r, lo, hi):
        return deep(r, lo, hi) if str(hi) in r["acc_deep"] else None

    def mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    g = defaultdict(list)
    for r in rs:
        if r["tag"] == "e8":
            g[(r["model"] != "tied", r["d"] if r["model"] == "tied" else 0, r["n"])].append(r)
    ref = [r for r in rs if r["tag"] == "e1fix" and r["read"] == "square" and r["lr"] == 1e-3]
    out = ["| 모델 | 상태 | 대입문 n | 끝난 seed | 학습 분포 평균 | 99%에 닿은 seed | 두 체인 5에서 8, seed별 | 두 체인 5에서 8, 99% seed | "
           "9에서 16, 99% seed | 17에서 32, 99% seed | 99%에 처음 닿은 step 중앙값 | 판정 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    rows = [((False, 64, 16), ref, "제안 layer, 실험 1 고친 설정, 이름 26개")] + \
           [(k, v, "DeltaNet 2층" if k[0] else "제안 layer, 이름 512개") for k, v in sorted(g.items())]
    for (is_std, d, n), rs_, label in rows:
        rs_ = sorted(rs_, key=lambda r: r["seed"])
        good = [r for r in rs_ if ok(r)]
        steps = [s for s in (first_ok_step(r) for r in good) if s is not None]
        d58 = mean_or_none([deep(r, 5, 8) for r in good])
        v = ""
        if label.endswith("512개") and len(rs_) >= 3:
            v = "유지" if len(good) >= 2 and d58 is not None and d58 >= 0.90 else "깨짐"
            verdict[(d, n)] = v
        state = f"{d}×{d}" if not is_std else "층마다 head 4개, 16×16"
        out.append(f"| {label} | {state} | {n} | {len(rs_)} | {pct(np.mean([np.mean(list(r['acc_in_dist'].values())) for r in rs_]))} | "
                   f"{len(good)}/{len(rs_)} | {', '.join(f'{100 * deep(r, 5, 8):.0f}' for r in rs_[:10])} | {pct(d58)} | "
                   f"{pct(mean_or_none([deep_or_none(r, 9, 16) for r in good]))} | {pct(mean_or_none([deep_or_none(r, 17, 32) for r in good]))} | "
                   f"{f'{int(np.median(steps)):,}' if steps else ''} | {v} |")
    notes = []
    for d in sorted({d for d, _ in verdict}):
        ns = sorted(n for dd, n in verdict if dd == d)
        broke = [n for n in ns if verdict[(d, n)] == "깨짐"]
        held = [n for n in ns if verdict[(d, n)] == "유지"]
        notes.append(f"d={d}: 판정이 난 n {', '.join(map(str, ns))}. 유지 {', '.join(map(str, held)) or '없음'}, "
                     f"깨짐 {', '.join(map(str, broke)) or '없음'}" + (f", 처음 깨진 n은 {min(broke)}." if broke else "."))
    return "\n".join(out) + ("\n\n" + "  \n".join(notes) if notes else "")


def e8b_table(rs):
    """Interference follow-up at n = 16: number of names and state size, with the same verdict rule."""
    g = defaultdict(list)
    for r in rs:
        if r["tag"] == "e8b" or (r["tag"] == "e8" and r["model"] == "tied" and r["n"] == 16):
            g[(r["d"], r["n_names"])].append(r)
    g[(64, 26)] = [r for r in rs if r["tag"] == "e1fix" and r["read"] == "square" and r["lr"] == 1e-3]
    out = ["| 상태 | 이름 수 | 끝난 seed | 학습 분포 평균 | 99%에 닿은 seed | 두 체인 5에서 8, seed별 | 두 체인 5에서 8, 99% seed | 판정 |",
           "|---|---|---|---|---|---|---|---|"]
    for (d, k), rs_ in sorted(g.items()):
        rs_ = sorted(rs_, key=lambda r: r["seed"])
        good = [r for r in rs_ if ok(r)]
        d58 = float(np.mean([deep(r, 5, 8) for r in good])) if good else None
        v = ("유지" if len(good) >= 2 and d58 is not None and d58 >= 0.90 else "깨짐") if len(rs_) >= 3 and k != 26 else ""
        out.append(f"| {d}×{d} | {k}{', 실험 1' if k == 26 else ''} | {len(rs_)} | {pct(np.mean([np.mean(list(r['acc_in_dist'].values())) for r in rs_]))} | "
                   f"{len(good)}/{len(rs_)} | {', '.join(f'{100 * deep(r, 5, 8):.0f}' for r in rs_[:10])} | {pct(d58)} | {v} |")
    return "\n".join(out)


def running_table():
    out = ["| 풀 | 작업 | 모델 | 진행 | step당 | 남은 시간, 추정 |", "|---|---|---|---|---|---|"]
    queued = []
    for d in ("e2_tf12", "poolB", "e2fix", "vt", "vt_fix3", "vt_fix4", "e3_now", "e8", "e8b"):
        plog = os.path.join(ROOT, f"logs/{d}/pool.log")
        if not os.path.exists(plog):
            continue
        cmds = {}
        for line in open(plog):
            m = re.search(r" start (\d+) slot .*?: (.*)$", line)
            if m:
                cmds[int(m.group(1))] = m.group(2)
        jobs = [l for l in open(os.path.join(ROOT, f"jobs/{d}.txt")).read().splitlines() if l.strip() and not l.startswith("#")]
        n_done = 0
        for f in glob.glob(os.path.join(ROOT, f"logs/{d}/[0-9]*.log")):
            txt = open(f).read()
            if "RESULT" in txt or "Traceback" in txt:
                n_done += 1
                continue
            n = int(re.search(r"/(\d+)\.log", f).group(1))
            steps = re.findall(r"^step (\d+) .* t (\d+)s", txt, re.M)
            if not steps or n not in cmds or int(steps[-1][0]) < 100:
                continue
            s, t = map(int, steps[-1])
            total = int(re.search(r"--steps (\d+)", cmds[n]).group(1))
            m = re.search(r"--model (\S+)(?: --L (\d+))?", cmds[n])
            out.append(f"| {d} | {n} | {m.group(1)}{' L' + m.group(2) if m.group(2) else ''} | {s:,}/{total:,} | {1000 * t / s:.0f} ms | {(total - s) * t / s / 3600:.1f} h |")
        alive = subprocess.run(["pgrep", "-f", f"[r]un_gpu_pool[a-z_]*.sh jobs/{d}.txt"], capture_output=True).returncode == 0
        queued.append(f"{d}: 시작 {len(cmds)}, 끝남 {n_done}, 목록 {len(jobs)}, 풀 {'살아 있음' if alive else '끝남'}")
    return "\n".join(out) + "\n\n" + "  \n".join(queued)


def gpu_check():
    try:
        idx = dict(l.split(", ")[::-1] for l in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                                               capture_output=True, text=True).stdout.strip().splitlines())
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    except Exception as e:                                           # noqa: BLE001
        return f"nvidia-smi 실패: {e}"
    mine, others = Counter(), Counter()
    for line in apps.strip().splitlines():
        pid, uuid = line.split(", ")
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            cwd = "?"
        (mine if cwd.startswith(ROOT) else others)[idx.get(uuid, "MIG")] += 1
    return (f"이 프로젝트의 GPU 프로세스: " + ", ".join(f"GPU {k} {v}개" for k, v in sorted(mine.items())) +
            ". 다른 세션: " + (", ".join(f"GPU {k} {v}개" for k, v in sorted(others.items())) or "없음") + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splice", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    import sys
    sys.path.insert(0, ROOT)
    rs = runs()
    counts = Counter(r["tag"] for r in rs)
    vt_base, vt_fix = vt_tables(rs)
    parts = dict(e2=e2_table(rs), vt_base=vt_base, vt_fix=vt_fix, e3=e3_table(rs), e8=e8_table(rs), e8b=e8b_table(rs), running=running_table())
    text = (f"# 자동 상태 보고\n\n생성: {now()}. `tools/make_status.py`가 결과 파일과 풀 로그에서 만든다.\n\n"
            f"끝난 실행 수: {dict(sorted(counts.items()))}\n\n{gpu_check()}\n\n"
            f"## 실험 2\n\n{parts['e2']}\n\n## (b) 기본 실행\n\n{parts['vt_base']}\n\n## (b) 보완\n\n{parts['vt_fix']}\n\n"
            f"## 실험 3\n\n{parts['e3']}\n\n## 실험 8, 간섭 시험\n\n{parts['e8']}\n\n## 실험 8 보완, n=16\n\n{parts['e8b']}\n\n## 돌고 있는 작업\n\n{parts['running']}\n")
    open("results/STATUS.md", "w").write(text)
    if args.splice:
        for p in ("SERVER_RESULTS.md", "draft/DRAFT.md"):
            s = open(p).read()
            for k, v in parts.items():
                s = re.sub(rf"(<!-- AUTO:{k} -->\n).*?(<!-- /AUTO:{k} -->)", lambda m: m.group(1) + f"_자동 갱신 {now()}_\n\n" + v + "\n" + m.group(2), s, flags=re.S)
            open(p, "w").write(s)
    print(text)


if __name__ == "__main__":
    main()
