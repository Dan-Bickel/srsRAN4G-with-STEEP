#!/usr/bin/env python3
"""Standalone clean-chain check for P-STEEP captures. Does not import the Eve script."""
import os
import struct
import numpy as np

# Must match steep_manager.h
PREAMBLE_LEN  = 64
PROBE_LEN     = 1024
FRAME_LEN     = PREAMBLE_LEN + PROBE_LEN
C1            = 0.5
STEEP_AMP     = 0.5
CAPTURE_MAGIC = 0xCA97CAFE


def read_capture(path):
    frames = []
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return frames
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(13)
            if len(hdr) < 13:
                break
            magic, _role, probe_id, n = struct.unpack('<IBII', hdr)
            if magic != CAPTURE_MAGIC:
                print(f"WARNING: bad magic 0x{magic:08X}, stopping")
                break
            raw = f.read(n * 8)
            if len(raw) < n * 8:
                break
            frames.append((probe_id, np.frombuffer(raw, dtype=np.complex64).copy()))
    print(f"{os.path.basename(path)}: {len(frames)} frames")
    return frames


P = read_capture('/tmp/steep_alice_probe.iq')   # Alice TX: preamble + probe
R = read_capture('/tmp/steep_bob_rx.iq')        # Bob RX:   y1_norm (probe only)
E = read_capture('/tmp/steep_bob_echo.iq')      # Bob TX:   preamble + echo

if not (P and R and E):
    raise SystemExit("Missing capture data. Run srsRAN first.")

alice_x1 = [p[PREAMBLE_LEN:FRAME_LEN].imag for _, p in P]
n_check  = min(len(R), len(E))
print(f"\nChecking {n_check} echoes against {len(alice_x1)} Alice probes\n")

used = []
for k in range(n_check):
    y1n = R[k][1].imag                      # Bob's received probe, gain-normalised
    x2  = E[k][1][PREAMBLE_LEN:FRAME_LEN].imag

    # Pair with Alice's probe by correlation (works because the probe is random +/-A)
    sc = np.array([abs(np.dot(x1, y1n)) for x1 in alice_x1]) / (STEEP_AMP**2 * PROBE_LEN)
    order = np.argsort(sc)[::-1]
    j, margin = int(order[0]), float(sc[order[0]] - (sc[order[1]] if len(sc) > 1 else 0))
    x1 = alice_x1[j]
    used.append(j)

    rms_err = np.sqrt(np.mean((y1n - x1) ** 2)) / STEEP_AMP    # chain noise vs probe amplitude
    nz      = np.abs(y1n) > 1e-9
    ratio   = np.mean(np.abs(x2[nz]) / (C1 * np.abs(y1n[nz])))
    frac_p  = np.mean(np.sign(x2 * y1n) > 0)
    levels  = np.unique(np.round(x1 / STEEP_AMP, 3))

    print(f"echo {k:3d} -> Alice idx {j:3d} (match {sc[j]:.2f}, margin {margin:.2f}) | "
          f"probe levels {levels} | rms(y1n-x1)/A = {rms_err:.2e} | "
          f"|x2|/(C1|y1n|) = {ratio:.4f} | frac(b=+1) = {frac_p:.3f}")

dupes = len(used) - len(set(used))
print(f"\nDuplicate Alice indices used: {dupes}  (should be 0)")
