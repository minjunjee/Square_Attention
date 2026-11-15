#!/bin/bash
# Blocks until logs/chain.log grows, a new crash appears in a job log, or $1 seconds pass.
cd "$(dirname "$0")/.."
limit=${1:-540}
c0=$(wc -l < logs/chain.log); e0=$(grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log 2>/dev/null | wc -l)
for ((t = 0; t < limit; t += 15)); do
  sleep 15
  c=$(wc -l < logs/chain.log); e=$(grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log 2>/dev/null | wc -l)
  if [ "$c" -gt "$c0" ]; then echo "EVENT chain.log:"; tail -n $((c - c0)) logs/chain.log; exit 0; fi
  if [ "$e" -gt "$e0" ]; then echo "EVENT new crash log(s):"; grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log; exit 0; fi
done
echo "no event in ${limit}s"
