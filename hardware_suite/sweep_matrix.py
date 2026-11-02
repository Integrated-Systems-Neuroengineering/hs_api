#!/usr/bin/env python3
"""sweep_matrix.py DIR cores... -- every test x every core from each core's summary.txt: P = pass, F = fail, H = hang,
N = not run, - = recorded informationally (core-1-only groups, counts), ? = no summary for that core."""
import sys, os, re, collections
d, cores = sys.argv[1], sys.argv[2:]
rows, res = [], collections.defaultdict(dict)
for c in cores:
    p = os.path.join(d, "summary_core%s.txt" % c)
    if not os.path.exists(p): continue
    for l in open(p).read().splitlines()[1:]:
        if not l.strip(): continue
        name, val = l[:34].strip(), l[34:].strip()
        if name not in rows: rows.append(name)
        v = val.upper()
        res[name][c] = ("P" if v.startswith("PASS") else "F" if v.startswith("FAIL") else "H" if v.startswith("HANG")
                        else "N" if v.startswith("NOT RUN") else "P" if re.match(r"^(\d+) OF \1 PASSED", v) else "F" if re.match(r"^\d+ OF \d+ PASSED", v) else "-")
print("%-34s %s" % ("test \\ core", " ".join("%3s" % c for c in cores)))
for r in rows: print("%-34s %s" % (r[:34], " ".join("%3s" % res[r].get(c, "?") for c in cores)))
bad = [(r, c) for r in rows for c in cores if res[r].get(c, "?") in ("F", "H", "N", "?")]
print("ALL PASS ON EVERY CORE" if not bad else "NOT ALL PASS: %d cells (%s)" % (len(bad), ", ".join("%s@core%s" % b for b in bad[:8])))
