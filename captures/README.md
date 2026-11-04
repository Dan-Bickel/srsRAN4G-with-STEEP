# STEEP I/Q Captures

Samples collected from srsRAN 4G (ZMQ virtual radio). Node-1 is the UE and Node-2 is the eNB.

## Which run do I use?

| Run | STEEP on Q only | STEEP on I and Q | x1 frames | s2 frames |
|---|---|---|---|---|
| `run1/` | Yes | No | 126 | 128 |
| `run2/` | Yes | Yes | 162 | 164 |

Use `run2/` if you need both versions. Both runs also contain the srsRAN logs (`ue.log`, `enb.log`).

## Which file do I want?

### STEEP on the imaginary (Q) component only
The real (I) component is 0, so there is no LTE in these files.

| File | Signal | Symbol values |
|---|---|---|
| `steep_x1_unit.iq` | Node-1 probe x1 | ±1 |
| `steep_s2_unit.iq` | Node-2 secret s2 | ±1 |
| `steep_x1.iq` | Node-1 probe x1 | ±0.5 |
| `steep_s2.iq` | Node-2 secret s2 | ±0.5 |

### STEEP on both real (I) and imaginary (Q) components (`run2/` only)

| File | Signal | Symbol values |
|---|---|---|
| `steep_x1_full_unit.iq` | Node-1 probe x1 | ±1 on I and on Q |
| `steep_s2_full_unit.iq` | Node-2 secret s2 | ±1 on I and on Q |
| `steep_x1_full.iq` | Node-1 probe x1 | ±0.5 on I and on Q |
| `steep_s2_full.iq` | Node-2 secret s2 | ±0.5 on I and on Q |

### Notes that apply to all of the above
- Each frame is **1088 samples**: a 64-symbol preamble followed by 1024 data symbols.
- The preamble is the same public sequence in x1 and s2.
- The ±1 and ±0.5 versions carry **identical bits**. ±0.5 is the amplitude actually transmitted over the air.
- x1 and s2 are independent random sequences. Frame k of x1 is not related to frame k of s2.
- In the full I/Q files, the **Q component is exactly what was transmitted**. The **I component was generated and saved only. It was never transmitted**, so LTE was unaffected and those samples did not pass through ZMQ.
- In the full I/Q files, the preamble symbols are c·(1+j) (the same chip on I and Q). The data symbols on I and Q are independent, so each sample is one of four points.

## Frame counts

`x1` and `s2` have slightly different frame counts because Node-2 occasionally echoed a probe more than once, or skipped one (an artifact of the test setup). Since x1 and s2 are independent random sequences, use the first N frames of each, where N is the x1 count, if you need equal counts.

## Validation files (not needed for analysis)

| File | Contents |
|---|---|
| `steep_alice_probe.iq` | What Node-1 actually transmitted (Q channel) |
| `steep_bob_rx.iq` | What Node-2 received (the probe, gain-normalised) |
| `steep_bob_echo.iq` | Node-2's echo: the probe with each symbol sign-flipped by the secret bit |

These echo files do **not** pair one-to-one with the probes:
- `run1`: the last probe was echoed three times.
- `run2`: probes 29 and 31 (0-indexed) were echoed twice each, probes 30 and 32 got no echo, and probe 161 was echoed three times.

Do not reuse repeated echoes of the same probe. Two echoes of one probe with different secrets reveal the XOR of the two secrets, which breaks the security assumption that each probe is used once.

## File format

Each file is a sequence of frames:

```
[ header: 13 bytes ][ samples: n_samples x 8 bytes ]
```

Header (little-endian): `uint32 magic = 0xCA97CAFE`, `uint8 role`, `uint32 probe_id`, `uint32 n_samples`.
Samples are `complex64` (float32 I, float32 Q).

## Loading in Python

```python
import struct
import numpy as np

def read_frames(path):
    frames = []
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(13)
            if len(hdr) < 13:
                break
            magic, role, probe_id, n = struct.unpack('<IBII', hdr)
            assert magic == 0xCA97CAFE
            frames.append(np.frombuffer(f.read(n * 8), dtype=np.complex64))
    return frames

x1 = read_frames('run2/steep_x1_full_unit.iq')
preamble = x1[0][:64]    # complex, public
probe    = x1[0][64:]    # 1024 complex symbols; .real and .imag are each +1 or -1
```

## How to collect a new run

Requires the repo built with the P-STEEP capture code (`srsRAN_4G/build`).

1. Clear old captures: `rm -f /tmp/steep_*.iq`
2. Open three terminals and start them in this order:
   ```
   sudo srsepc ~/.config/srsran/epc.conf
   sudo <repo>/srsRAN_4G/build/srsenb/src/srsenb ~/.config/srsran/enb.conf
   sudo <repo>/srsRAN_4G/build/srsue/src/srsue ~/.config/srsran/ue.conf
   ```
3. Let it run about 3 minutes (about 150 probes). Check progress with `grep -c "transmitted, N=" /tmp/ue.log`. Node-1 stops capturing at 200 probes.
4. Stop with Ctrl+C: UE first, then eNB, then EPC.
5. Confirm the 11 files exist: `ls -l /tmp/steep_*.iq`
6. Copy them (from the repo root). `/tmp` is cleared on reboot and overwritten by the next run, so do this right away:
   ```
   mkdir -p captures
   mkdir captures/runN          # pick a new name; this fails if it already exists
   cp /tmp/steep_*.iq captures/runN/
   cp /tmp/ue.log /tmp/enb.log captures/runN/
   ```
7. Validate: `python3 check_x1_s2.py runN`