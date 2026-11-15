#!/bin/bash
# Like run_gpu_pool.sh, but several pools may work on the same jobs file: each job is claimed by
# creating locks/<jobs file name>/<line number> first, and a claimed job is skipped.
# usage: ./run_gpu_pool_claim.sh jobs.txt "4 5" [jobs per GPU, default 4]
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
name=$(basename "$JOBS" .txt)
logdir=logs/$name
lockdir=locks/$name
mkdir -p "$logdir" "$lockdir"
declare -A pid
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  [[ "$cmd" == \#* ]] && continue
  n=$((n + 1))
  while true; do
    for s in "${!SLOT_UUID[@]}"; do
      if [ -z "${pid[$s]:-}" ] || ! kill -0 "${pid[$s]}" 2>/dev/null; then
        mkdir "$lockdir/$n" 2> /dev/null || continue 3          # another pool has this job
        CUDA_VISIBLE_DEVICES=${SLOT_UUID[$s]} bash -c "$cmd" > "$logdir/$n.log" 2>&1 < /dev/null &
        pid[$s]=$!
        echo "$(date '+%F %T') start $n slot $s ${SLOT_UUID[$s]} (pool $$): $cmd" >> "$logdir/pool.log"
        continue 3
      fi
    done
    sleep 5
  done
done < "$JOBS"
wait
echo "$(date '+%F %T') pool $$ reached the end of $JOBS" >> "$logdir/pool.log"
