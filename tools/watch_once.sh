#!/bin/bash
# One monitoring cycle: wait until a new result line, a crash, or $1 seconds; then refresh the status
# tables in the documents and print the runs that finished during the cycle.
cd "$(dirname "$0")/.."
limit=${1:-540}
n0=$(wc -l < results/res_server.jsonl)
crash0=$(grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log 2>/dev/null | wc -l)
for ((t = 0; t < limit; t += 15)); do
  : # new results no longer end the wait early
  [ "$(grep -l 'Traceback\|Killed\|out of memory' logs/*/[0-9]*.log 2>/dev/null | wc -l)" -gt "$crash0" ] && break
  sleep 15
done
.venv/bin/python tools/make_status.py --splice > /dev/null 2>&1 || echo "make_status failed"
.venv/bin/python make_presentation.py > /dev/null 2>&1 || echo "make_presentation failed"
echo "== $(TZ=Asia/Seoul date '+%F %T KST')"
crash=$(grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log 2>/dev/null | wc -l)
[ "$crash" -gt "$crash0" ] && echo "NEW CRASH LOGS:" && grep -l "Traceback\|Killed\|out of memory" logs/*/[0-9]*.log
tail -n +$((n0 + 1)) results/res_server.jsonl | .venv/bin/python -c "
import json, sys
for l in sys.stdin:
    r = json.loads(l); ok = all(v >= 0.99 for v in r['acc_in_dist'].values())
    nm = r['model'] + ('/' + r['read'] if r['model'] == 'tied' else ' L%d' % r['L'])
    if r.get('pre'): nm += ' +' + (r.get('pre_kind') or 'tf')
    if r['tag'] == 'e8': nm += ' d%s n%d' % (r['d'] if r['model'] == 'tied' else '-', r['n'])
    if r.get('vt'):
        v = r['vt']; nm += ' [' + {(1, 0): 'A', (1, 256): 'B', (3, 0): 'C', (3, 256): 'D'}[(v['name_max'], v['prog_len'])] + ']'
    d = [r['acc_deep'][str(k)] for k in range(1, 9)]
    print('new %-6s %-26s lr %-6g s%d in-dist %5.1f %-3s two 2-4 %5.1f two 5-8 %5.1f' % (r['tag'], nm, r['lr'], r['seed'],
          100 * sum(r['acc_in_dist'].values()) / len(r['acc_in_dist']), 'ok' if ok else '', 100 * sum(d[1:4]) / 3, 100 * sum(d[4:]) / 4))
"
grep "이 프로젝트의 GPU" results/STATUS.md
