#!/bin/bash
# One-screen status of the server runs.
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
echo "== $(TZ=Asia/Seoul date '+%F %T KST')"
python3 - <<'PY'
import json, collections
c = collections.Counter()
for l in open("results/res_server.jsonl"):
    c[json.loads(l)["tag"]] += 1
print("results by tag:", dict(sorted(c.items())))
PY
for d in e2_tf12 poolB e1fix e2fix e5e6; do
  [ -f logs/$d/pool.log ] || continue
  started=$(grep -c " start " logs/$d/pool.log); total=$(grep -vc '^\s*$\|^#' jobs/$d.txt 2>/dev/null)
  running=$(pgrep -fc "run_gpu_pool.sh jobs/$d.txt")
  echo "pool $d: started $started/$total, pool alive: $([ "$running" -gt 0 ] && echo yes || echo no)"
done
echo "chain (last 3):"; tail -3 logs/chain.log | sed 's/^/   /'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/   gpu /'
echo "my processes per GPU: $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read p g; do case "$(readlink /proc/$p/cwd 2>/dev/null)" in "$ROOT"*) echo $g;; esac; done | sort | uniq -c | tr '\n' ' ')"
echo "other users' or sessions' processes on GPUs 4-7: $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read p g; do c=$(readlink /proc/$p/cwd 2>/dev/null); case "$c" in "$ROOT"*) ;; *) echo "$g:$p"; esac; done | tr '\n' ' ')"
errs=$(grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log 2>/dev/null | wc -l); echo "logs with Traceback/Killed/OOM: $errs"
