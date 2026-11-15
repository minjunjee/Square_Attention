#!/bin/bash
# Waits for the experiment 1 and 2 reruns with the fix from experiment 4 (or for the decision that there
# is no fix), then runs experiments 5 and 6 with the same fix, 3 slots per GPU on GPUs 4 and 5.
# When every other pool has finished, runs experiment 7 alone on GPU 4.
cd "$(dirname "$0")"
until grep -q "e2fix done\|choice: none\|experiment 4 was not needed" logs/chain.log 2>/dev/null; do sleep 60; done
FIX=$(tail -1 logs/choose_fix.txt 2>/dev/null)
[ "$FIX" = "none" ] && FIX=""
echo "$(date '+%F %T') experiments 5 and 6 use the flags '${FIX}' for the proposed layer" | tee -a logs/chain.log
sed "s|--model tied \(.*\) --tag e5|--model tied \1 $FIX --tag e5|" jobs/e5_draft.txt > jobs/e5.txt
sed "s|--model tied \(.*\) --lr|--model tied \1 $FIX --lr|" jobs/e6_draft.txt > jobs/e6.txt
cat jobs/e5.txt jobs/e6.txt > jobs/e5e6.txt
echo "$(date '+%F %T') starting e5 and e6 ($(wc -l < jobs/e5e6.txt) jobs, 3 per GPU)" | tee -a logs/chain.log
./run_gpu_pool.sh jobs/e5e6.txt "4 5" 3
echo "$(date '+%F %T') e5 and e6 done" | tee -a logs/chain.log
while pgrep -f "run_gpu_pool.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') all pools finished; starting experiment 7 alone on GPU 4" | tee -a logs/chain.log
CUDA_VISIBLE_DEVICES=GPU-6c3df329-6760-6320-4613-e6a25973be69 .venv/bin/python cost_bench.py --lines 16 64 256 --out results/cost.jsonl > logs/cost_bench.log 2>&1
echo "$(date '+%F %T') experiment 7 done" | tee -a logs/chain.log
