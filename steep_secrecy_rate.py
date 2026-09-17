#!/usr/bin/env python3
"""
steep_secrecy_rate.py
Computes digital secrecy rate from STEEP Eve sweep output.

Two models:
  Model A — Erasure wiretap (CRC = detected erasure):
    Cs = max(0, alice_rate - eve_rate)

  Model B — BSC mapping to paper Eq.134 (arXiv:2309.14529 §VII):
    PA|B = (1 - alice_rate) * 0.5
    PE|B = (1 - eve_rate)   * 0.5
    xi   = max(0, H(PE|B) - H(PA|B))

Usage:
    python3 steep_secrecy_rate.py results_fine.tsv
    cat results_fine.tsv | python3 steep_secrecy_rate.py
"""

import sys, math

PAYLOAD_BITS = 256  # 32 bytes

def H(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

def parse(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('p1') or line.startswith('P1') or line.startswith('#'):
            continue
        parts = line.split('\t') if '\t' in line else line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[5])))
        except ValueError:
            continue
    return rows

def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    rows = parse(src.readlines())
    if not rows:
        print("No data rows found.")
        sys.exit(1)

    print(f"\n{'='*88}")
    print(f"  STEEP Secrecy Rate  |  N=200 pairs  |  Payload={PAYLOAD_BITS} bits")
    print(f"  Model A (erasure):  Cs = max(0, alice% - eve%)  [bits per bit of payload]")
    print(f"  Model B (BSC §VII): xi = max(0, H(PE|B) - H(PA|B))")
    print(f"  Rs_bits = Model A x {PAYLOAD_BITS}  [bits of secrecy per frame]")
    print(f"{'='*88}")

    for p1 in sorted(set(r[0] for r in rows)):
        subset = sorted((r[1], r[2], r[3]) for r in rows if r[0] == p1)
        print(f"\n  Eve P1 = {p1:+.0f} dB")
        print(f"  {'P2(dB)':>7}  {'Alice%':>7}  {'Eve%':>7}  "
              f"{'PA|B':>7}  {'PE|B':>7}  {'Cs(A)':>8}  {'xi(B)':>8}  {'Rs_bits':>9}")
        print(f"  {'-'*82}")
        for p2, alice_pct, eve_pct in subset:
            a, e = alice_pct / 100.0, eve_pct / 100.0
            Cs  = max(0.0, a - e)
            PA  = (1.0 - a) * 0.5
            PE  = (1.0 - e) * 0.5
            xi  = max(0.0, H(PE) - H(PA))
            print(f"  {p2:>7.0f}  {alice_pct:>7.1f}  {eve_pct:>7.1f}  "
                  f"{PA:>7.4f}  {PE:>7.4f}  {Cs:>8.4f}  {xi:>8.4f}  {Cs*PAYLOAD_BITS:>9.1f}")

    print(f"\n{'='*88}\n")

if __name__ == '__main__':
    main()