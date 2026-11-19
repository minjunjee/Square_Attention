#!/bin/bash
# Runs the commands in a file on GPUs, several commands per GPU at a time.
# usage: ./run_gpu_pool.sh jobs.txt "4 5" [jobs per GPU, default 4]
# GPUs are given by nvidia-smi index and handed to each job by UUID: on the course server CUDA
# numbers devices differently from nvidia-smi (MIG instances on GPUs 0-3 take CUDA ids 4-7).
# Only GPUs listed in gpu_allowlist.txt are accepted.
# Each line of jobs.txt is one command. Output goes to logs/<jobs file name>/<line number>.log.
set -u
cd "$(dirname "$0")"
JOBS=$1
GPUS=(${2:-4 5})
PER=${3:-4}
SLOT_UUID=()
for g in "${GPUS[@]}"; do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$g" '$1 == g {print $2}')
  if [ -z "$uuid" ] || ! grep -q "^$uuid " gpu_allowlist.txt; then
    echo "GPU $g (${uuid:-unknown}) is not in gpu_allowlist.txt; refusing to run" >&2
    exit 1
  fi
  for ((s = 0; s < PER; s++)); do SLOT_UUID+=("$uuid"); done
done
logdir=logs/$(basename "$JOBS" .txt)
mkdir -p "$logdir"
declare -A pid
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  [[ "$cmd" == \#* ]] && continue
  n=$((n + 1))
  while true; do
    for s in "${!SLOT_UUID[@]}"; do
      if [ -z "${pid[$s]:-}" ] || ! kill -0 "${pid[$s]}" 2>/dev/null; then
        CUDA_VISIBLE_DEVICES=${SLOT_UUID[$s]} bash -c "$cmd" > "$logdir/$n.log" 2>&1 < /dev/null &
        pid[$s]=$!
        echo "$(date '+%F %T') start $n slot $s ${SLOT_UUID[$s]}: $cmd" >> "$logdir/pool.log"
        continue 3
      fi
    done
    sleep 5
  done
done < "$JOBS"
wait
echo "$(date '+%F %T') all $n jobs finished" >> "$logdir/pool.log"
