"""Rough per-job speed and remaining time for running jobs, from the last 'step N ... t Ts' line of each log."""
import glob, os, re, subprocess, time
running = subprocess.run(["pgrep", "-af", "poc_fair.py"], capture_output=True, text=True).stdout
for d in ("e2_tf12", "poolB", "e1fix", "e2fix", "e5e6"):
    rows = []
    for f in glob.glob(f"logs/{d}/[0-9]*.log"):
        txt = open(f).read()
        if "RESULT" in txt:
            continue
        m = re.findall(r"^step (\d+) .* t (\d+)s", txt, re.M)
        n = int(re.search(r"/(\d+)\.log", f).group(1))
        cmd = open(f"jobs/{d}.txt").read().splitlines()
        cmd = [c for c in cmd if c.strip() and not c.startswith("#")][n - 1]
        total = int(re.search(r"--steps (\d+)", cmd).group(1))
        if m:
            s, t = map(int, m[-1])
            age = time.time() - os.path.getmtime(f)
            rate = t / s if s > 1 else float("nan")
            rows.append((d, n, re.search(r"--model (\S+)", cmd).group(1), s, total, rate, (total - s) * rate / 3600))
    for r in sorted(rows, key=lambda x: x[1]):
        print(f"{r[0]:8s} job {r[1]:3d} {r[2]:6s} step {r[3]:6d}/{r[4]} {1000 * r[5]:6.0f} ms/step, about {r[6]:.1f} h left")
