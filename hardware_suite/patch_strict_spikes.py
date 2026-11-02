#!/usr/bin/env python3
"""patch_strict_spikes.py HS_API_DIR -- test_max_number_axons: replace 'at least one spike at step 1' with an exact check:
every one of the numberAxons neurons spikes at step 1, exactly once (a lost row, e.g. row 512, can no longer pass).
Applied only inside test_max_number_axons, in tests/test_bitstream_hardware_fast.py and tests/test_bitstream_hardware.py."""
import sys, os
root = sys.argv[1]
OLD = "        assert len(currSpikes2[0]) != 0\n"
NEW = '''        names = [s[1] if isinstance(s, (tuple, list)) else s for s in currSpikes2[0]]
        missing = sorted(set(f"N1.{i}" for i in range(numberAxons)) - set(names), key=lambda k: int(k.split(".")[1]))
        assert len(names) == numberAxons and not missing, (
            "step 1: %d spikes (%d distinct), expected %d; first missing: %s" % (len(names), len(set(names)), numberAxons,
            [(m, "row %d grp %d" % (int(m.split(".")[1]) // 16, int(m.split(".")[1]) % 16)) for m in missing[:6]]))
'''
done = 0
for f in ("tests/test_bitstream_hardware_fast.py", "tests/test_bitstream_hardware.py"):
    p = os.path.join(root, f); s = open(p).read()
    a = s.find("    def test_max_number_axons(")
    if a < 0: print("%s: no test_max_number_axons, skipped" % f); continue
    b = s.find("\n    def ", a + 10); b = len(s) if b < 0 else b
    body = s[a:b]
    if "first missing" in body: print("%s: already strict" % f); continue
    if body.count(OLD) != 1: sys.exit("%s: expected exactly one weak spike assertion in test_max_number_axons, found %d" % (f, body.count(OLD)))
    if not os.path.exists(p + ".before_strict"): open(p + ".before_strict", "w").write(s)
    open(p, "w").write(s[:a] + body.replace(OLD, NEW) + s[b:]); done += 1
    print("%s: test_max_number_axons now checks every spike" % f)
print("%d file(s) patched" % done)
