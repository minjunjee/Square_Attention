#!/bin/bash
# Pauses this user's processes on physical GPU 5 (SIGSTOP), runs a command alone on that GPU, and resumes
# them (SIGCONT) on any exit. Only this project's processes (working directory here) are touched.
cd "$(dirname "$0")/.."
G5=GPU-446f0d79-347a-62cb-f394-be4a9f8b410e
PIDS=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v g=$G5 '$2 == g {print $1}' \
       | while read p; do case "$(readlink /proc/$p/cwd 2>/dev/null)" in /home/minjun_dev/poc_squared_read*) echo $p;; esac; done)
resume() { [ -n "$PIDS" ] && kill -CONT $PIDS 2>/dev/null; echo "$(TZ=Asia/Seoul date '+%T KST') resumed: $PIDS"; }
trap resume EXIT
echo "$(TZ=Asia/Seoul date '+%T KST') pausing $(echo $PIDS | wc -w) processes on GPU 5: $PIDS"
kill -STOP $PIDS
sleep 5
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n 6p
CUDA_VISIBLE_DEVICES=$G5 "$@"
