# srsRAN4G-with-STEEP
Implement STEEP on srsRAN4G

---

## Running the STEEP Experiment

This experiment assumes you have already installed srsRAN 4G and all the needed libraries. https://docs.srsran.com/projects/4g/en/latest/app_notes/source/zeromq/source/index.html

**Before you start:** Replace `/path/to/repo/` in all commands below with the path to your local copy of this repository. Full paths are used throughout to avoid conflicts with any system-installed srsRAN binaries.

---

## Build

```bash
cmake --build /path/to/repo/srsRAN_4G/build -- -j$(nproc)
```

---

## Open 5 terminals and run in this order

**Terminal 1 — Core Network**
```bash
sudo srsepc ~/.config/srsran/epc.conf 2>&1 | tee /tmp/epc.log
```
You will see: network startup messages, then silence. Leave it running.

**Terminal 2 — Base Station (Bob)**
```bash
sudo /path/to/repo/srsRAN_4G/build/srsenb/src/srsenb ~/.config/srsran/enb.conf 2>&1 | tee /tmp/enb.log
```
You will see: standard srsRAN base station output. Leave it running.

**Terminal 3 — User Equipment (Alice)**
```bash
sudo /path/to/repo/srsRAN_4G/build/srsue/src/srsue ~/.config/srsran/ue.conf 2>&1 | tee /tmp/ue.log
```
You will see: standard srsRAN UE output. Wait until you see `RRC Connected` before starting Terminal 4.

**Terminal 4 — Ping from network side**
```bash
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward
sudo iptables -t nat -A POSTROUTING -s 172.16.0.0/24 -o srs_spgw_sgi -j MASQUERADE
ping 172.16.0.2
```
You will see: continuous ping replies. This confirms LTE data traffic is running alongside STEEP.

**Terminal 5 — Ping from UE side**
```bash
ping 172.16.0.1
```
You will see: continuous ping replies in the other direction.

---

## How long to run

Let the experiment run for **2–3 minutes** (roughly 120–180 probes). Longer runs give more data but diminishing returns.

---

## Stop order

Stop in reverse order. Press Ctrl+C in:

1. Terminal 5 (UE ping)
2. Terminal 4 (network ping)
3. Terminal 3 (UE / Alice)
4. Terminal 2 (ENB / Bob)
5. Terminal 1 (EPC)

---

## View results (run in any free terminal)

**Session summary — recovery accuracy, BER, RTT:**
```bash
python3 /path/to/repo/steep_summarize.py /tmp/ue.log /tmp/enb.log
```

**Eve secrecy rate analysis — Rs sweep and plot:**
```bash
python3 /path/to/repo/eve/steep_eve_offline.py
```
The Rs plot is saved to `/tmp/steep_eve_analysis.png`.

**Optional — view raw STEEP log messages:**
```bash
grep "STEEP" /tmp/ue.log
grep "STEEP" /tmp/enb.log
```

---

## Before running again

```bash
rm /tmp/steep_alice_probe.iq /tmp/steep_bob_echo.iq
```
This clears the IQ capture files. Without this step the Eve script will read data mixed from two different runs.