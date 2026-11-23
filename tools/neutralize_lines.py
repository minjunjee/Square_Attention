"""Turns lines first..last of a jobs file into comments of the same byte length, in place, and writes
the original lines to another file. A running pool reads its jobs file line by line, so lines it has
not reached yet are skipped and the file offsets do not move.
usage: python3 tools/neutralize_lines.py jobs/poolB.txt 111 140 jobs/e3_on67.txt"""
import sys
path, first, last, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with open(path, "r+b") as f:
    lines = f.read().split(b"\n")
    moved = []
    for i in range(first - 1, last):
        moved.append(lines[i])
        tag = b"# moved to GPU 6/7 "
        lines[i] = (tag + b"x" * max(0, len(lines[i]) - len(tag)))[:len(lines[i])]
    data = b"\n".join(lines)
    f.seek(0); f.write(data); f.truncate()
open(out, "wb").write(b"\n".join(moved) + b"\n")
print(f"moved {len(moved)} lines from {path} to {out}")
