import os, struct
import numpy as np

MAGIC, PRE, FRAME = 0xCA97CAFE, 64, 1088

# Folder is found relative to this script, so it works from any directory.
import sys
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captures',
                    sys.argv[1] if len(sys.argv) > 1 else 'run2')

def rd(name):
    path = os.path.join(BASE, name)
    out = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(13)
            if len(h) < 13: break
            m, _r, pid, n = struct.unpack('<IBII', h)
            assert m == MAGIC
            out.append(np.frombuffer(f.read(n * 8), dtype=np.complex64).copy())
    print(f"{name}: {len(out)} frames")
    return out

X1 = rd('steep_x1.iq')
S2 = rd('steep_s2.iq')
A  = rd('steep_alice_probe.iq')
R  = rd('steep_bob_rx.iq')
E  = rd('steep_bob_echo.iq')
U1 = rd('steep_x1_unit.iq')
U2 = rd('steep_s2_unit.iq')

print()
print("x1 real part all zero :", all(np.all(x.real == 0) for x in X1))
print("s2 real part all zero :", all(np.all(s.real == 0) for s in S2))
print("x1 == Alice TX Q      :", all(np.array_equal(x.imag, a.imag) for x, a in zip(X1, A)))
print("preambles identical   :", all(np.array_equal(X1[0][:PRE], s[:PRE]) for s in S2))

bad = sum(int(np.sum(np.sign(s[PRE:].imag) != np.sign(e[PRE:FRAME].imag * r.imag)))
          for s, r, e in zip(S2, R, E))
print("s2 vs sign(x2*y1n) mismatches:", bad, " (should be 0)")
print("s2 fraction of +1     :", np.mean([np.mean(s[PRE:].imag > 0) for s in S2]))

print("x1_unit == x1/0.5     :", all(np.allclose(u.imag, x.imag / 0.5) for u, x in zip(U1, X1)))
print("s2_unit == s2/0.5     :", all(np.allclose(u.imag, s.imag / 0.5) for u, s in zip(U2, S2)))
print("unit levels           :", np.unique(np.concatenate([u.imag for u in U1 + U2])))

from collections import Counter
idx = []
for r in R:
    sc = [abs(np.dot(x.imag[PRE:], r.imag)) for x in X1]
    idx.append(int(np.argmax(sc)))
dups    = {k: v for k, v in Counter(idx).items() if v > 1}
missing = sorted(set(range(len(X1))) - set(idx))
print("probe indices echoed more than once:", dups)
print("probes with no echo                :", missing)

F1  = rd('steep_x1_full.iq');       F1U = rd('steep_x1_full_unit.iq')
F2  = rd('steep_s2_full.iq');       F2U = rd('steep_s2_full_unit.iq')

print("full x1 Q == x1 Q          :", all(np.array_equal(f.imag, x.imag) for f, x in zip(F1, X1)))
print("full s2 Q == s2 Q          :", all(np.array_equal(f.imag, s.imag) for f, s in zip(F2, S2)))
print("preamble I == Q            :", all(np.array_equal(f.real[:PRE], f.imag[:PRE]) for f in F1 + F2))
print("I levels (x1, s2)          :", np.unique(np.concatenate([f.real[PRE:] for f in F1 + F2])))
print("I fraction of +            :", np.mean([np.mean(f.real[PRE:] > 0) for f in F1 + F2]))
iv = np.concatenate([f.real[PRE:] for f in F1]); qv = np.concatenate([f.imag[PRE:] for f in F1])
print("corr(I, Q) of x1           :", np.corrcoef(iv, qv)[0, 1], " (should be near 0, within ~0.01)")
print("unit == full/0.5           :", all(np.allclose(u.real, f.real / 0.5) and np.allclose(u.imag, f.imag / 0.5)
                                            for u, f in zip(F1U + F2U, F1 + F2)))