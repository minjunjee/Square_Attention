#!/bin/bash
# Run ONLY after the user allows physical GPUs 6 and 7.
# Moves the not-yet-started jobs of tf12 (lr 1e-3, 3e-3) and experiment 3 to GPUs 6 and 7, and runs
# experiment 7 first, alone, on GPU 7.
set -u
cd "$(dirname "$0")/.."
A=$(grep -c " start " logs/e2_tf12/pool.log); B=$(grep -c " start " logs/poolB/pool.log)
[ "$A" -le 12 ] || { echo "poolA already started job 13; stop"; exit 1; }
[ "$B" -lt 111 ] || { echo "poolB already reached experiment 3; stop"; exit 1; }
grep -q "^GPU-23d6f53a" gpu_allowlist.txt || echo "GPU-23d6f53a-13cb-5e0e-0db3-e9eeb686e419 physical GPU 6" >> gpu_allowlist.txt
grep -q "^GPU-cff86157" gpu_allowlist.txt || echo "GPU-cff86157-308b-5bfb-7f38-3a9801628cf9 physical GPU 7" >> gpu_allowlist.txt
python3 tools/neutralize_lines.py jobs/e2_tf12.txt 13 20 jobs/tf12_on67.txt
python3 tools/neutralize_lines.py jobs/poolB.txt 111 140 jobs/e3_on67.txt
echo "$(TZ=Asia/Seoul date '+%F %T KST') GPUs 6 and 7 allowed: tf12 jobs 13-20 and experiment 3 moved there; experiment 7 first on GPU 7" | tee -a logs/chain.log
./run_gpu_pool_claim.sh jobs/tf12_on67.txt "6" 4 > /dev/null 2>&1 &
./run_gpu_pool_claim.sh jobs/e3_on67.txt "6" 4 > /dev/null 2>&1 &
(
  CUDA_VISIBLE_DEVICES=GPU-cff86157-308b-5bfb-7f38-3a9801628cf9 .venv/bin/python cost_bench.py --lines 16 64 256 --out results/cost.jsonl > logs/cost_bench.log 2>&1
  echo "$(TZ=Asia/Seoul date '+%F %T KST') experiment 7 done on GPU 7" | tee -a logs/chain.log
  ./run_gpu_pool_claim.sh jobs/tf12_on67.txt "7" 4 > /dev/null 2>&1 &
  ./run_gpu_pool_claim.sh jobs/e3_on67.txt "7" 4 > /dev/null 2>&1 &
  wait
) &
wait
echo "$(TZ=Asia/Seoul date '+%F %T KST') GPU 6/7 pools finished" | tee -a logs/chain.log
