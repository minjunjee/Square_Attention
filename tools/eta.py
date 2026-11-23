"""Rough per-job speed and remaining time of running jobs. The command of job N is read from the
'start N' line of the pool log, the progress from the last 'step S ... t Ts' line of logs/<pool>/N.log."""
import glob, re, time
for d in ("e2_tf12", "poolB", "e2fix", "vt", "vt_fix3"):
    cmds = {}
    for line in open(f"logs/{d}/pool.log"):
        m = re.search(r" start (\d+) slot .*?: (.*)$", line)
        if m:
            cmds[int(m.group(1))] = m.group(2)
    for f in sorted(glob.glob(f"logs/{d}/[0-9]*.log"), key=lambda x: int(re.search(r"/(\d+)\.log", x).group(1))):
        txt = open(f).read()
        if "RESULT" in txt or "Traceback" in txt:
            continue
        n = int(re.search(r"/(\d+)\.log", f).group(1))
        steps = re.findall(r"^step (\d+) .* t (\d+)s", txt, re.M)
        if not steps or n not in cmds:
            continue
        s, t = map(int, steps[-1])
        total = int(re.search(r"--steps (\d+)", cmds[n]).group(1))
        model = re.search(r"--model (\S+)", cmds[n]).group(1)
        rate = t / s
        print(f"{d:8s} job {n:3d} {model:5s} step {s:6d}/{total} {1000 * rate:5.0f} ms/step, about {(total - s) * rate / 3600:.1f} h left")
