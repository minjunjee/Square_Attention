#!/bin/bash
# Waits for experiment 4 (the first jobs of poolB), picks the fix, then reruns experiment 1 with it and,
# after that, the proposed layer of experiment 2. Extra pools take 3 slots per GPU on GPUs 4 and 5.
cd "$(dirname "$0")"
until grep -q "starting poolA" logs/chain.log 2>/dev/null; do sleep 60; done
N4=$(wc -l < jobs/e4.txt)
if [ "$N4" -eq 0 ]; then echo "$(date '+%F %T') experiment 4 was not needed" | tee -a logs/chain.log; exit 0; fi
until grep -q "start $((N4 + 1)) " logs/poolB/pool.log 2>/dev/null; do sleep 60; done
while pgrep -f -- "--tag e4 " > /dev/null; do sleep 60; done
read E1LR OK N <<< "$(.venv/bin/python choose_lr.py results/res_server.jsonl --tag e1 --model tied --read square)"
.venv/bin/python choose_fix.py results/res_server.jsonl --e1_lr "$E1LR" > logs/choose_fix.txt
FIX=$(tail -1 logs/choose_fix.txt)
echo "$(date '+%F %T') e4 done; choice: $FIX (details in logs/choose_fix.txt)" | tee -a logs/chain.log
[ "$FIX" = "none" ] && exit 0
P=".venv/bin/python poc_fair.py"
C1="--steps 30000 --tap_sharp 10 --gate_ctx 1 --r_hi 5 --r_eval 6 --eval_every 2000 --out results/res_server.jsonl $FIX"
: > jobs/e1fix.txt
for rd in square once iter; do
  if [ $rd = square ]; then seeds="0 1 2 3 4 5 6 7 8 9"; else seeds="0 1 2 3 4"; fi
  for s in $seeds; do for lr in 3e-4 1e-3 3e-3; do
    echo "$P --model tied --read $rd --lr $lr --seed $s --tag e1fix $C1 --save ckpt/e1fix_${rd}_lr${lr}_s$s.pt" >> jobs/e1fix.txt
  done; done
done
echo "$(date '+%F %T') starting e1fix ($(wc -l < jobs/e1fix.txt) jobs, 3 per GPU)" | tee -a logs/chain.log
./run_gpu_pool.sh jobs/e1fix.txt "4 5" 3
FIXARGS=$(echo "$FIX" | sed 's/^--//; s/ /=/')
read LR OK N <<< "$(.venv/bin/python - <<PY
import json, numpy as np
from collections import defaultdict
by = defaultdict(list)
for l in open("results/res_server.jsonl"):
    r = json.loads(l)
    if r["tag"] == "e1fix" and r["model"] == "tied" and r["read"] == "square":
        by[r["lr"]].append(r)
s = {lr: np.mean([np.mean(list(r["acc_in_dist"].values())) for r in rs]) for lr, rs in by.items()}
b = max(s, key=s.get)
print(f"{b:g}", sum(all(v >= 0.99 for v in r["acc_in_dist"].values()) for r in by[b]), len(by[b]))
PY
)"
echo "$(date '+%F %T') e1fix done: chosen lr $LR, $OK of $N seeds reach 99%" | tee -a logs/chain.log
: > jobs/e2fix.txt
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$P --model tied --read square --lr $LR --seed $s --tap_sharp 10 --gate_ctx 1 --r_hi 5 --r_eval 6 $FIX --tag e2fix --steps 100000 --bs 64 --eval_every 2000 --out results/res_server.jsonl --save ckpt/e2fix_tied_s$s.pt" >> jobs/e2fix.txt
done
echo "$(date '+%F %T') starting e2fix (10 jobs, 3 per GPU)" | tee -a logs/chain.log
./run_gpu_pool.sh jobs/e2fix.txt "4 5" 3
echo "$(date '+%F %T') e2fix done" | tee -a logs/chain.log
