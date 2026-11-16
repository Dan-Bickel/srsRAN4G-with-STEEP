#!/usr/bin/env python3
"""Replay Node-1's decoder on saved captures.  Usage: python3 check_decode.py <run_folder> <mult|add>
Works because ZMQ is lossless: what Node-1 receives is (gain) * (Node-2's transmitted echo).
Keeps only the FIRST echo per probe (later echoes of the same probe are dropped and counted)."""
import os, struct, sys
import numpy as np

MAGIC, PRE, FRAME = 0xCA97CAFE, 64, 1088
C1, C2 = 0.5, 0.3            # must match steep_manager.h
DECODE_LEN = 256             # what the C++ decoder actually decodes (payload_size * 8)

run  = sys.argv[1] if len(sys.argv) > 1 else 'run2'
mode = sys.argv[2] if len(sys.argv) > 2 else 'mult'
if mode not in ('mult', 'add'):
    raise SystemExit("mode must be 'mult' or 'add'")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captures', run)

def rd(name):
    out = []
    with open(os.path.join(BASE, name), 'rb') as f:
        while True:
            h = f.read(13)
            if len(h) < 13: break
            m, _r, _pid, n = struct.unpack('<IBII', h)
            assert m == MAGIC
            out.append(np.frombuffer(f.read(n * 8), dtype=np.complex64).copy())
    return out

X1 = rd('steep_x1.iq')            # Node-1's probes (+/-0.5, Q)
S2 = rd('steep_s2.iq')            # Node-2's secret signs (x0.5), same order as echoes
R  = rd('steep_node2_rx.iq')      # Node-2's soft received probe (gain-normalised)
E  = rd('steep_node2_echo.iq')    # Node-2's transmitted echo
n = min(len(S2), len(R), len(E))
print(f"mode={mode}: {len(X1)} probes, {n} echoes")

seen, dropped = set(), 0
err_dec = tot_dec = err_all = tot_all = 0
for k in range(n):
    sc = [abs(np.dot(x.imag[PRE:], R[k].imag)) for x in X1]
    j = int(np.argmax(sc))
    if j in seen:
        dropped += 1
        continue
    seen.add(j)
    x1 = X1[j].imag[PRE:]
    x2 = E[k].imag[PRE:FRAME]
    if mode == 'mult':
        resid = x2 * x1                 # Node-1: sign(rx * x1)
    else:
        resid = x2 - C1 * x1            # Node-1: rx/G - C1*x1
    got  = resid >= 0
    true = S2[k].imag[PRE:] > 0
    e = got != true
    err_dec += int(e[:DECODE_LEN].sum()); tot_dec += DECODE_LEN
    err_all += int(e.sum());              tot_all += len(e)
print(f"echoes kept (first per probe): {len(seen)}   later duplicate echoes dropped: {dropped}")
print(f"bit errors, first {DECODE_LEN} symbols (what the C++ decodes): {err_dec} / {tot_dec}")
print(f"bit errors, all 1024 symbols                              : {err_all} / {tot_all}")
