#!/bin/bash
# Waits for experiment 1, then queues experiments 2 and 3 (and 4 first if needed) on GPUs 4 and 5.
cd "$(dirname "$0")"
while pgrep -f "run_gpu_pool.sh jobs/e1.txt" > /dev/null; do sleep 60; done
read LR OK N <<< "$(.venv/bin/python choose_lr.py results/res_server.jsonl --tag e1 --model tied --read square)"
echo "$(date '+%F %T') e1 done: chosen lr $LR, $OK of $N seeds reach 99% at every depth" | tee -a logs/chain.log
P=".venv/bin/python poc_fair.py"
C1="--steps 30000 --tap_sharp 10 --gate_ctx 1 --r_hi 5 --r_eval 6 --eval_every 2000 --out results/res_server.jsonl"
: > jobs/e2_tied.txt
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "$P --model tied --read square --lr $LR --seed $s --tap_sharp 10 --gate_ctx 1 --r_hi 5 --r_eval 6 --tag e2 --steps 100000 --bs 64 --eval_every 2000 --out results/res_server.jsonl --save ckpt/e2_tied_s$s.pt" >> jobs/e2_tied.txt
done
: > jobs/e4.txt
if [ $((2 * OK)) -lt "$N" ]; then
  for s in 0 1 2 3 4; do
    for lr in 1e-3 3e-4; do echo "$P --model tied --read square --hard_taps 2 --lr $lr --seed $s --tag e4 $C1 --save ckpt/e4_pair_lr${lr}_s$s.pt" >> jobs/e4.txt; done
    for x in 0.01 0.1; do echo "$P --model tied --read square --tap_entropy $x --lr $LR --seed $s --tag e4 $C1 --save ckpt/e4_ent${x}_s$s.pt" >> jobs/e4.txt; done
    echo "$P --model tied --read square --tap_attn 8 --lr $LR --seed $s --tag e4 $C1 --save ckpt/e4_attn_s$s.pt" >> jobs/e4.txt
  done
  echo "$(date '+%F %T') fewer than half the seeds reach 99%: experiment 4 runs first ($(wc -l < jobs/e4.txt) jobs)" | tee -a logs/chain.log
fi
cat jobs/e4.txt jobs/e2_tied.txt jobs/e2_rest.txt jobs/e3.txt > jobs/poolB.txt
echo "$(date '+%F %T') starting poolA (e2_tf12, 3 per GPU) and poolB ($(wc -l < jobs/poolB.txt) jobs, 5 per GPU)" | tee -a logs/chain.log
./run_gpu_pool.sh jobs/e2_tf12.txt "4 5" 3 &
./run_gpu_pool.sh jobs/poolB.txt "4 5" 5 &
wait
echo "$(date '+%F %T') poolA and poolB finished" | tee -a logs/chain.log
