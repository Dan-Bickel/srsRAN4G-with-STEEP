#!/usr/bin/env python3
"""
steep_summarize.py — compact summary of a STEEP stress-test log session.

Usage:
    python3 steep_summarize.py /tmp/ue.log [/tmp/enb.log] [--rs FLOAT]

Options:
    --rs FLOAT    Rs at Eve's natural position (bits), from steep_eve_offline.py.
                  If omitted, the script also checks /tmp/steep_eve_rs.txt
                  (written automatically by steep_eve_offline.py if you add the
                  two-line snippet described in the handoff notes).

New in this version:
  - Parses HEX field from every recovery line and compares against the
    deterministically expected secret "Probe #XXXX from Bob!" for that probe ID.
  - Reports exact / near-correct (1–2 bit errors) / garbled breakdown.
  - Reports BER across all recoveries (21-byte message region only;
    null padding bytes are ignored per handoff spec).
  - Reports mean MSE_A for correct vs garbled subsets.
  - Reports Rs from the most recent Eve offline run (via --rs or auto-file).
  - success_rate label now annotated as the flawed raw metric so it is not
    mistaken for the correct-recovery rate.
"""

import re
import sys
import argparse
from datetime import datetime
from statistics import mean, median, stdev


# ── Payload / message constants (must match steep_manager.h) ──────────────────
PAYLOAD_SIZE   = 32          # bytes total (32 B = 256 bits)
MESSAGE_LEN    = 21          # len("Probe #XXXX from Bob!") — compared region
MESSAGE_BITS   = MESSAGE_LEN * 8   # 168 bits used for BER

# Recovery quality thresholds (message region only)
NEAR_MAX_ERRORS = 2          # 0 = exact; 1–2 = near-correct; >2 = garbled


# ── regex patterns ─────────────────────────────────────────────────────────────

TS_RE = re.compile(
    r'^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)'
)

INJECTED_RE = re.compile(
    r'STEEP\s*\[ALICE\]:\s*probe\s+#(\d+)\s+transmitted\b',
    re.IGNORECASE
)

# Full current-format recovery line — captures probe_id and hex bytes.
# Format: ... *** SECRET RECOVERED probe=#N ... | HEX: XX XX ... | MSE_A= ...
RECOVERED_HEX_RE = re.compile(
    r'STEEP\s*\[ALICE\]:\s*\*+\s*SECRET\s+RECOVERED\s+probe=#(\d+)'
    r'.*?\|\s*HEX:\s*((?:[0-9a-fA-F]{2}\s*)+)',
    re.IGNORECASE
)

# Fallback: recovery line without HEX field (future-proofing / older builds).
RECOVERED_NOHEX_RE = re.compile(
    r'STEEP\s*\[ALICE\]:\s*\*+\s*SECRET\s+RECOVERED\s+probe=#(\d+)\b',
    re.IGNORECASE
)

TIMEOUT_RE = re.compile(
    r'STEEP\s*\[ALICE\].*?\bprobe(?:\s+id=|#)(\d+)\b.*?\btimed\s*out\b',
    re.IGNORECASE
)

CRC_RE = re.compile(
    r'STEEP\s+decode:\s*CRC\s+mismatch\s+#(\d+)',
    re.IGNORECASE
)

ALICE_STATS_RE = re.compile(
    r'STEEP\s*\[ALICE\]:\s*stats:\s*'
    r'sent=(\d+)\s+recovered=(\d+)\s+timed_out=(\d+)\s+success_rate=([\d.]+)%',
    re.IGNORECASE
)

MSE_A_RE = re.compile(
    r'\bMSE_A=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',
    re.IGNORECASE
)

BOB_ENCODING_RE = re.compile(
    r'STEEP\s*\[BOB\]:\s*encoding\s+secret:\s*"Probe\s+#(\d+)\s+from\s+Bob!',
    re.IGNORECASE
)

BOB_ECHO_TX_RE = re.compile(
    r'STEEP\s*\[BOB\]:\s*echo\s+transmitted\b.*?\btotal_echoes=(\d+)',
    re.IGNORECASE
)

BOB_CAPTURE_RE = re.compile(
    r'STEEP\s+capture:\s+echo\s+#(\d+)\s+saved',
    re.IGNORECASE
)

BOB_PREAMBLE_RE = re.compile(
    r'STEEP\s*\[BOB\]:\s*preamble\s+detected\b',
    re.IGNORECASE
)

RESET_STATS_RE = re.compile(
    r'STEEP\s+manager\s+reset\s*[—-]\s*final\s+stats:\s*'
    r'sent=(\d+)\s+recovered=(\d+)\s+echoed=(\d+)\s+timed_out=(\d+)',
    re.IGNORECASE
)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_ts(line):
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1))
    except ValueError:
        return None


def safe_read_lines(path):
    try:
        with open(path, 'r', errors='replace') as f:
            yield from f
    except FileNotFoundError:
        print(f"  [warn] Log not found: {path}")
    except OSError as exc:
        print(f"  [warn] Could not read {path}: {exc}")


def expected_secret_bytes(probe_id):
    """
    Reconstruct the 32-byte secret Bob encodes for the given probe ID.
    Format: "Probe #XXXX from Bob!" (21 ASCII bytes) + 11 null bytes.
    Must match handle_tx_bob() in steep_manager.cpp exactly.
    """
    msg = f"Probe #{probe_id:04d} from Bob!"
    result = bytearray(PAYLOAD_SIZE)
    for i, ch in enumerate(msg[:PAYLOAD_SIZE]):
        result[i] = ord(ch)
    return bytes(result)


def count_bit_errors_region(recovered, expected, n_bytes=MESSAGE_LEN):
    """
    Hamming distance in bits between the first n_bytes of two sequences.
    Only the message region is compared; null padding is ignored.
    """
    a = recovered[:n_bytes]
    b = expected[:n_bytes]
    length = min(len(a), len(b))
    return sum(bin(x ^ y).count('1') for x, y in zip(a[:length], b[:length]))


def classify_recovery(bit_errors):
    if bit_errors == 0:
        return 'exact'
    elif bit_errors <= NEAR_MAX_ERRORS:
        return 'near'
    else:
        return 'garbled'


def as_printable(raw_bytes, n=MESSAGE_LEN):
    return ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw_bytes[:n])


def load_eve_rs(path='/tmp/steep_eve_rs.txt'):
    """Read Rs at Eve's natural position written by steep_eve_offline.py."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


# ── log parsers ───────────────────────────────────────────────────────────────

def parse_ue_log(path):
    """
    Returns:
      inject_ts        — {probe_id: datetime}
      recover_ts       — {probe_id: datetime}
      timeouts         — {probe_id}
      ue_crc_max       — int
      final_stats      — (sent, recovered, timed_out, rate) or None
      mse_by_pid       — {probe_id: float}
      recovery_quality — {probe_id: {'bit_errors', 'category',
                                      'recovered_msg', 'expected_msg'}}
    """
    inject_ts        = {}
    recover_ts       = {}
    timeouts         = set()
    ue_crc_max       = 0
    final_stats      = None
    mse_by_pid       = {}
    recovery_quality = {}

    for line in safe_read_lines(path):
        ts = parse_ts(line)

        # ── probe transmitted ─────────────────────────────────────────────────
        m = INJECTED_RE.search(line)
        if m:
            pid = int(m.group(1))
            if ts:
                inject_ts[pid] = ts
            continue

        # ── secret recovered (with HEX field — current format) ───────────────
        m = RECOVERED_HEX_RE.search(line)
        if m:
            pid     = int(m.group(1))
            hex_str = m.group(2).strip()
            if ts:
                recover_ts[pid] = ts

            # Parse hex → bytes
            try:
                rec_bytes = bytes(int(h, 16) for h in hex_str.split())
            except ValueError:
                rec_bytes = None

            # Compute quality vs expected (message region only)
            if rec_bytes is not None and len(rec_bytes) >= MESSAGE_LEN:
                exp_bytes  = expected_secret_bytes(pid)
                bit_errors = count_bit_errors_region(rec_bytes, exp_bytes)
                recovery_quality[pid] = {
                    'bit_errors':    bit_errors,
                    'category':      classify_recovery(bit_errors),
                    'recovered_msg': as_printable(rec_bytes),
                    'expected_msg':  as_printable(exp_bytes),
                }

            # MSE_A
            mse_m = MSE_A_RE.search(line)
            if mse_m:
                try:
                    mse_by_pid[pid] = float(mse_m.group(1))
                except (TypeError, ValueError):
                    pass
            continue

        # ── secret recovered (fallback — no HEX field) ───────────────────────
        m = RECOVERED_NOHEX_RE.search(line)
        if m:
            pid = int(m.group(1))
            if pid not in recover_ts and ts:
                recover_ts[pid] = ts
            mse_m = MSE_A_RE.search(line)
            if mse_m:
                try:
                    mse_by_pid[pid] = float(mse_m.group(1))
                except (TypeError, ValueError):
                    pass
            continue

        # ── timeout ───────────────────────────────────────────────────────────
        m = TIMEOUT_RE.search(line)
        if m:
            timeouts.add(int(m.group(1)))
            continue

        # ── CRC ───────────────────────────────────────────────────────────────
        m = CRC_RE.search(line)
        if m:
            ue_crc_max = max(ue_crc_max, int(m.group(1)))
            continue

        # ── embedded stats ────────────────────────────────────────────────────
        m = ALICE_STATS_RE.search(line)
        if m:
            final_stats = (int(m.group(1)), int(m.group(2)),
                           int(m.group(3)), float(m.group(4)))
            continue

    return (inject_ts, recover_ts, timeouts, ue_crc_max,
            final_stats, mse_by_pid, recovery_quality)


def parse_enb_log(path):
    encoding_ids      = set()
    capture_ids       = set()
    preamble_count    = 0
    enb_crc_max       = 0
    reset_stats       = None
    last_total_echoes = None

    for line in safe_read_lines(path):
        m = BOB_ENCODING_RE.search(line)
        if m:
            encoding_ids.add(int(m.group(1)))
            continue

        m = BOB_ECHO_TX_RE.search(line)
        if m:
            last_total_echoes = int(m.group(1))
            continue

        m = BOB_CAPTURE_RE.search(line)
        if m:
            capture_ids.add(int(m.group(1)))
            continue

        m = BOB_PREAMBLE_RE.search(line)
        if m:
            preamble_count += 1
            continue

        m = CRC_RE.search(line)
        if m:
            enb_crc_max = max(enb_crc_max, int(m.group(1)))
            continue

        m = RESET_STATS_RE.search(line)
        if m:
            reset_stats = (int(m.group(1)), int(m.group(2)),
                           int(m.group(3)), int(m.group(4)))
            continue

    return (encoding_ids, capture_ids, preamble_count,
            enb_crc_max, reset_stats, last_total_echoes)


# ── main summary ──────────────────────────────────────────────────────────────

def summarize(ue_log, enb_log=None, rs_override=None):
    SEP = '=' * 64

    print(f"\n{SEP}")
    print("  STEEP Stress-Test Summary")
    print(f"  UE log : {ue_log}")
    if enb_log:
        print(f"  ENB log: {enb_log}")
    print(SEP)

    # ── parse ─────────────────────────────────────────────────────────────────
    (inject_ts, recover_ts, timeouts, ue_crc_max,
     final_stats, mse_by_pid, recovery_quality) = parse_ue_log(ue_log)

    if enb_log:
        (enb_encoding_ids, enb_capture_ids, enb_preamble_count,
         enb_crc_max, enb_reset_stats, enb_total_echoes) = parse_enb_log(enb_log)
    else:
        enb_encoding_ids  = set()
        enb_capture_ids   = set()
        enb_preamble_count = 0
        enb_crc_max       = 0
        enb_reset_stats   = None
        enb_total_echoes  = None

    # ── headline counts ───────────────────────────────────────────────────────
    all_sent    = set(inject_ts.keys()) | timeouts
    n_sent      = len(all_sent)
    n_recovered = len(recover_ts)
    n_timeouts  = len(timeouts)
    n_missing   = max(0, n_sent - n_recovered - n_timeouts)
    raw_rate    = 100.0 * n_recovered / n_sent if n_sent else 0.0

    # Fallback to embedded stats when TX lines were filtered out of the log.
    if n_sent == 0 and final_stats:
        n_sent      = final_stats[0]
        n_recovered = final_stats[1]
        n_timeouts  = final_stats[2]
        n_missing   = max(0, n_sent - n_recovered - n_timeouts)
        raw_rate    = final_stats[3]

    # ── RTTs ──────────────────────────────────────────────────────────────────
    rtts_ms = []
    for pid, t_inj in inject_ts.items():
        t_rec = recover_ts.get(pid)
        if t_rec:
            dt_ms = (t_rec - t_inj).total_seconds() * 1000.0
            if 0 < dt_ms < 60_000:
                rtts_ms.append(dt_ms)

    # ── session duration ──────────────────────────────────────────────────────
    all_ts    = list(inject_ts.values()) + list(recover_ts.values())
    session_s = (max(all_ts) - min(all_ts)).total_seconds() if len(all_ts) >= 2 else 0.0
    pid_min   = min(all_sent) if all_sent else 0
    pid_max   = max(all_sent) if all_sent else 0

    # ── recovery quality breakdown ────────────────────────────────────────────
    n_qual    = len(recovery_quality)
    n_exact   = sum(1 for q in recovery_quality.values() if q['category'] == 'exact')
    n_near    = sum(1 for q in recovery_quality.values() if q['category'] == 'near')
    n_garbled = sum(1 for q in recovery_quality.values() if q['category'] == 'garbled')

    total_bit_errors   = sum(q['bit_errors'] for q in recovery_quality.values())
    total_bits_checked = n_qual * MESSAGE_BITS
    ber = total_bit_errors / total_bits_checked if total_bits_checked > 0 else 0.0

    # MSE_A by quality category
    mse_exact   = [mse_by_pid[p] for p in recovery_quality
                   if recovery_quality[p]['category'] == 'exact'   and p in mse_by_pid]
    mse_near    = [mse_by_pid[p] for p in recovery_quality
                   if recovery_quality[p]['category'] == 'near'    and p in mse_by_pid]
    mse_garbled = [mse_by_pid[p] for p in recovery_quality
                   if recovery_quality[p]['category'] == 'garbled' and p in mse_by_pid]
    mse_all     = list(mse_by_pid.values())

    # ── print ─────────────────────────────────────────────────────────────────
    print(f"\n  Session duration   : {session_s/60:.1f} min  ({session_s:.0f} s)")
    print(f"  Probe ID range     : {pid_min} – {pid_max}")

    # ─ Alice headline ─────────────────────────────────────────────────────────
    print("\n  ── Alice (UE log) ──────────────────────────────────────────")
    print(f"  Probes sent         : {n_sent}")
    print(f"  Secrets recovered   : {n_recovered}")
    print(f"  Timed out           : {n_timeouts}")
    print(f"  Unaccounted         : {n_missing}")
    print(f"  Raw recovery rate   : {raw_rate:.2f}%  ← counts garbled; see below")

    if final_stats:
        fs = final_stats
        match = (fs[0] == n_sent and fs[1] == n_recovered and fs[2] == n_timeouts)
        print(f"  In-log stats check  : sent={fs[0]} rec={fs[1]} to={fs[2]} "
              f"rate={fs[3]:.1f}% [{'OK' if match else 'MISMATCH'}]")

    # ─ Recovery accuracy (new) ────────────────────────────────────────────────
    print(f"\n  ── Recovery Accuracy (21-byte message region vs expected) ──")
    if n_qual > 0:
        pct_e = 100.0 * n_exact   / n_qual
        pct_n = 100.0 * n_near    / n_qual
        pct_g = 100.0 * n_garbled / n_qual
        print(f"  Compared            : {n_qual} / {n_recovered}")
        print(f"  Exact match (0 err) : {n_exact:4d}  ({pct_e:5.1f}%)")
        print(f"  Near-correct (1–{NEAR_MAX_ERRORS} b): {n_near:4d}  ({pct_n:5.1f}%)")
        print(f"  Garbled (>{NEAR_MAX_ERRORS} bit err) : {n_garbled:4d}  ({pct_g:5.1f}%)")
        print(f"  BER (message region): {ber:.4%}  "
              f"({total_bit_errors} errors / {total_bits_checked} bits)")

        if mse_exact:
            print(f"  MSE_A mean — exact  : {mean(mse_exact):.4e}")
        if mse_near:
            print(f"  MSE_A mean — near   : {mean(mse_near):.4e}")
        if mse_garbled:
            print(f"  MSE_A mean — garbled: {mean(mse_garbled):.4e}")

        # Show worst garbled recoveries (up to 5)
        garbled_pids = sorted(
            (p for p, q in recovery_quality.items() if q['category'] == 'garbled'),
            key=lambda p: recovery_quality[p]['bit_errors'],
            reverse=True
        )
        if garbled_pids:
            print(f"\n  Worst garbled (top {min(5, len(garbled_pids))}):")
            for p in garbled_pids[:5]:
                q = recovery_quality[p]
                print(f"    probe #{p:04d}: {q['bit_errors']:3d} bit err  "
                      f"got=\"{q['recovered_msg']}\"  "
                      f"exp=\"{q['expected_msg']}\"")
    else:
        print("  No hex-parseable recoveries found.")
        print("  (Log must be from the current AF-model binary to contain HEX fields.)")

    # ─ Secrecy rate ───────────────────────────────────────────────────────────
    print("\n  ── Secrecy Rate Rs (from Eve offline analysis) ─────────────")
    rs_val = rs_override if rs_override is not None else load_eve_rs()
    if rs_val is not None:
        print(f"  Rs at Eve natural pos : {rs_val:.4f} bits/channel use")
        if rs_val > 0:
            print(f"  STEEP claim           : CONFIRMED  (Rs > 0 despite Eve advantage)")
        else:
            print(f"  Rs ≤ 0 — check Bob SNR setting or re-run Eve analysis.")
    else:
        print("  Rs not available. Either:")
        print("    • pass  --rs FLOAT  on the command line, or")
        print("    • add the two-line Rs writer to steep_eve_offline.py so it")
        print("      auto-saves to /tmp/steep_eve_rs.txt on each run.")

    # ─ RTT ───────────────────────────────────────────────────────────────────
    print("\n  ── RTT (probe TX → echo decoded) ───────────────────────────")
    if rtts_ms:
        print(f"  Mean     : {mean(rtts_ms):.1f} ms")
        print(f"  Min/Max  : {min(rtts_ms):.1f} / {max(rtts_ms):.1f} ms")
        print(f"  Samples  : {len(rtts_ms)}")
        if len(rtts_ms) >= 2:
            print(f"  Std dev  : {stdev(rtts_ms):.1f} ms")
    else:
        print("  No matched TX/recover timestamp pairs found.")

    # ─ MSE_A overall ─────────────────────────────────────────────────────────
    print("\n  ── Alice MSE_A (all recoveries) ─────────────────────────────")
    if mse_all:
        print(f"  Samples  : {len(mse_all)}")
        print(f"  Mean     : {mean(mse_all):.4e}")
        print(f"  Median   : {median(mse_all):.4e}")
        print(f"  Min/Max  : {min(mse_all):.4e} / {max(mse_all):.4e}")
    else:
        print("  No MSE_A values found.")

    # ─ CRC ───────────────────────────────────────────────────────────────────
    print("\n  ── CRC mismatches ──────────────────────────────────────────")
    print(f"  UE  : {'ZERO (clean)' if ue_crc_max == 0 else f'#{ue_crc_max} (!)'}")
    if enb_log:
        print(f"  ENB : {'ZERO (clean)' if enb_crc_max == 0 else f'#{enb_crc_max} (!)'}")

    # ─ Bob ───────────────────────────────────────────────────────────────────
    if enb_log:
        print("\n  ── Bob (ENB log) ────────────────────────────────────────────")
        print(f"  Preambles detected : {enb_preamble_count}")
        print(f"  Secrets encoded    : {len(enb_encoding_ids)}")
        print(f"  Echo captures      : {len(enb_capture_ids)}")
        if enb_total_echoes is not None:
            print(f"  Last total_echoes  : {enb_total_echoes}")
        if enb_reset_stats:
            rs = enb_reset_stats
            print(f"  Reset stats        : sent={rs[0]} rec={rs[1]} "
                  f"echoed={rs[2]} to={rs[3]}")
        if enb_encoding_ids and recover_ts:
            overlap = len(enb_encoding_ids & set(recover_ts.keys()))
            print(f"  Bob/Alice ID match : {overlap}")

    # ─ cross-checks ──────────────────────────────────────────────────────────
    print("\n  ── Cross-checks ────────────────────────────────────────────")
    if final_stats and inject_ts:
        print(f"  UE parsed TX lines  : {len(inject_ts)} / embedded sent={final_stats[0]}")
    if final_stats and recover_ts:
        print(f"  UE parsed recoveries: {len(recover_ts)} / embedded recovered={final_stats[1]}")
    if enb_log and enb_encoding_ids:
        exp_n = final_stats[0] if final_stats else n_sent
        print(f"  Bob encoded secrets : {len(enb_encoding_ids)} / Alice sent={exp_n}")

    # ─ unaccounted ───────────────────────────────────────────────────────────
    unaccounted = all_sent - set(recover_ts.keys()) - timeouts
    if unaccounted:
        ids     = sorted(unaccounted)
        preview = ', '.join(str(i) for i in ids[:15])
        suffix  = f'  (+{len(ids)-15} more)' if len(ids) > 15 else ''
        print(f"\n  [!] Unaccounted probe IDs: {preview}{suffix}")
    else:
        if all_sent:
            print("\n  All parsed sent probes accounted for (recovered or timed out).")
        elif final_stats:
            print("\n  Embedded stats found; no individual TX lines parsed.")

    print(f"\n{SEP}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='STEEP stress-test log summarizer (AF analog model)'
    )
    ap.add_argument('ue_log',            help='/tmp/ue.log')
    ap.add_argument('enb_log', nargs='?', help='/tmp/enb.log (optional)')
    ap.add_argument('--rs', type=float, metavar='FLOAT',
                    help='Rs at Eve natural position (bits), from steep_eve_offline.py')
    args = ap.parse_args()
    summarize(args.ue_log, args.enb_log, rs_override=args.rs)