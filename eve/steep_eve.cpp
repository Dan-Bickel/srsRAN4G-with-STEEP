// steep_eve.cpp — Eve adversary simulation with Alice legitimate-receiver baseline.
//
// For each (P1, P2) grid point this binary computes TWO things in parallel:
//
//   Alice (legitimate receiver):
//     - Probe payload is known exactly (Alice transmitted it; clean capture → always decodable).
//     - Echo received with Phase-2 AWGN at P2 dB SNR.
//     - Secret recovered if echo CRC passes (XOR with known probe payload).
//     → alice_secret%: the legitimate-receiver baseline.
//
//   Eve (adversary):
//     - Probe received with Phase-1 AWGN at P1 dB SNR.
//     - Echo received with Phase-2 AWGN at P2 dB SNR (independent of Alice's).
//     - Secret recovered only if BOTH CRCs pass (she must decode probe AND echo).
//     → eve_secret%: the adversary recovery rate.
//
//   Secrecy advantage = alice_secret% − eve_secret%.
//   A positive advantage means a channel region where Alice succeeds but Eve does not.
//
// NOTE: byte_acc (always 100% when CRC passes by construction) is omitted — it is
// a CRC tautology, not a security metric.
//
// Build:  cd eve && mkdir -p build && cd build && cmake .. && make
// Run:    ./steep_eve --p1 40,0,-10 --p2 -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-15,-20
//
// Self-contained — no srsRAN dependency.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using cf_t = std::complex<float>;

// ── STEEP constants (must match steep_manager.h) ──────────────────────────────
static constexpr uint32_t STEEP_MAGIC         = 0xDEADBEEFu;
static constexpr uint32_t CAPTURE_FRAME_MAGIC = 0xCA97CAFEu;
static constexpr uint32_t SAMPLES_PER_BIT     = 32u;
static constexpr uint32_t CRC_SIZE_BYTES      = 4u;

static const int8_t STEEP_CHIP[32] = {
  +1,-1,+1,-1,+1,+1,-1,-1,+1,+1,+1,-1,+1,-1,-1,+1,
  -1,+1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,+1,-1
};

// ── Utilities ─────────────────────────────────────────────────────────────────
static uint32_t frame_size_samples(uint32_t payload_size)
{
  return (8u + payload_size + CRC_SIZE_BYTES) * 8u * SAMPLES_PER_BIT;
}

static uint32_t compute_crc32(const uint8_t* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++)
      crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1u));
  }
  return crc ^ 0xFFFFFFFFu;
}

// Add AWGN to a sample buffer at the specified SNR (dB).
// Signal power is measured from the buffer itself. Uses the provided RNG.
static void add_awgn(cf_t* samples, uint32_t n, float snr_db, std::mt19937& rng)
{
  float sig_pwr = 0.0f;
  for (uint32_t i = 0; i < n; i++) sig_pwr += std::norm(samples[i]);
  sig_pwr /= static_cast<float>(n);
  if (sig_pwr <= 0.0f) return;

  const float noise_std = std::sqrt(sig_pwr / (2.0f * std::pow(10.0f, snr_db / 10.0f)));
  std::normal_distribution<float> dist(0.0f, noise_std);
  for (uint32_t i = 0; i < n; i++)
    samples[i] += cf_t(dist(rng), dist(rng));
}

// Correlation decoder. Returns true if the frame passes CRC.
// When it returns true, pid_out and payload_out are bit-exact by CRC guarantee.
static bool decode_frame(const cf_t* samples, uint32_t nof_samples,
                         uint32_t payload_size,
                         uint32_t& pid_out, std::vector<uint8_t>& payload_out)
{
  if (nof_samples < frame_size_samples(payload_size)) return false;

  uint32_t idx = 0;

  // Magic word detection (fast early-out)
  uint32_t magic = 0;
  for (int b = 0; b < 4; b++) {
    uint8_t dec = 0;
    for (int bit = 7; bit >= 0; bit--) {
      float corr = 0.0f;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++)
        corr += samples[idx++].imag() * static_cast<float>(STEEP_CHIP[s]);
      if (corr > 0.0f) dec |= (1u << bit);
    }
    magic = (magic << 8) | dec;
  }
  if (magic != STEEP_MAGIC) return false;

  const uint32_t total_bytes = 8u + payload_size + CRC_SIZE_BYTES;
  std::vector<uint8_t> frame(total_bytes, 0u);
  frame[0] = (STEEP_MAGIC >> 24) & 0xFFu;
  frame[1] = (STEEP_MAGIC >> 16) & 0xFFu;
  frame[2] = (STEEP_MAGIC >>  8) & 0xFFu;
  frame[3] = (STEEP_MAGIC >>  0) & 0xFFu;

  for (uint32_t b = 4; b < total_bytes; b++) {
    for (int bit = 7; bit >= 0; bit--) {
      float corr = 0.0f;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++)
        corr += samples[idx++].imag() * static_cast<float>(STEEP_CHIP[s]);
      if (corr > 0.0f) frame[b] |= (1u << bit);
    }
  }

  const size_t   data_len = 8u + payload_size;
  const uint32_t exp_crc  = compute_crc32(frame.data(), data_len);
  const uint32_t dec_crc  = ((uint32_t)frame[data_len+0] << 24) |
                            ((uint32_t)frame[data_len+1] << 16) |
                            ((uint32_t)frame[data_len+2] <<  8) |
                            ((uint32_t)frame[data_len+3]      );
  if (exp_crc != dec_crc) return false;

  const uint32_t pid = ((uint32_t)frame[4] << 24) | ((uint32_t)frame[5] << 16) |
                       ((uint32_t)frame[6] <<  8) | ((uint32_t)frame[7]      );
  if (pid == 0) return false;

  pid_out = pid;
  payload_out.assign(frame.begin() + 8, frame.begin() + 8 + payload_size);
  return true;
}

// ── Capture file I/O ──────────────────────────────────────────────────────────
struct CaptureFrame {
  uint8_t           role;
  uint32_t          probe_id;
  std::vector<cf_t> samples;
};

static bool read_capture_file(const char* path, std::vector<CaptureFrame>& frames)
{
  FILE* fp = fopen(path, "rb");
  if (!fp) { fprintf(stderr, "[Eve] ERROR: cannot open %s\n", path); return false; }

  while (true) {
    uint32_t magic = 0;
    if (fread(&magic, sizeof(magic), 1, fp) != 1) break;
    if (magic != CAPTURE_FRAME_MAGIC) {
      fprintf(stderr, "[Eve] bad frame magic 0x%08x — stopping read of %s\n", magic, path);
      break;
    }
    CaptureFrame fr;
    if (fread(&fr.role,     sizeof(fr.role),     1, fp) != 1) break;
    if (fread(&fr.probe_id, sizeof(fr.probe_id), 1, fp) != 1) break;
    uint32_t ns = 0;
    if (fread(&ns, sizeof(ns), 1, fp) != 1) break;
    if (ns == 0 || ns > 500000u) {
      fprintf(stderr, "[Eve] implausible n_samples=%u — stopping\n", ns);
      break;
    }
    fr.samples.resize(ns);
    if (fread(fr.samples.data(), sizeof(cf_t), ns, fp) != ns) {
      fprintf(stderr, "[Eve] short read at probe_id=%u\n", fr.probe_id);
      break;
    }
    frames.push_back(std::move(fr));
  }
  fclose(fp);
  return !frames.empty();
}

// ── Argument parsing ──────────────────────────────────────────────────────────
static std::vector<float> parse_csv(const char* s)
{
  std::vector<float> v;
  char* buf = strdup(s);
  for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ","))
    v.push_back(static_cast<float>(atof(tok)));
  free(buf);
  return v;
}

static void usage(const char* prog)
{
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  --probe FILE    probe IQ capture (default: /tmp/steep_alice_probe.iq)\n"
    "  --echo  FILE    echo  IQ capture (default: /tmp/steep_bob_echo.iq)\n"
    "  --seed  N       RNG seed (0=random, default: 42)\n"
    "  --p1    CSV     Phase-1 SNR values for Eve in dB  (default: 40,30,20,10,0,-10)\n"
    "  --p2    CSV     Phase-2 SNR values for both Alice and Eve in dB\n"
    "                  (default: 40,30,20,10,0,-10)\n"
    "  --payload N     payload size in bytes (default: 32)\n"
    "\n"
    "TSV output columns:\n"
    "  p1_db  p2_db  alice_secret%%  eve_probe%%  eve_echo%%  eve_secret%%  adv%%\n"
    "\n"
    "alice_secret: legitimate receiver — known exact probe, noisy echo at P2 SNR.\n"
    "eve_secret:   adversary — noisy probe at P1, noisy echo at P2 (independent AWGN).\n"
    "adv:          alice_secret − eve_secret (positive = secrecy advantage for Alice).\n"
    "\n"
    "IMPORTANT: when P1 is high (Eve's probe decodes perfectly), alice_secret ≈ eve_secret\n"
    "at the same P2 SNR. This is the expected honest result of a DF implementation.\n"
    "The secrecy advantage in practice is argued from Alice having a better Phase-2\n"
    "physical path than Eve — not from this binary directly.\n",
    prog);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  const char* probe_path = "/tmp/steep_alice_probe.iq";
  const char* echo_path  = "/tmp/steep_bob_echo.iq";
  uint32_t    seed       = 42u;
  uint32_t    payload_sz = 32u;
  std::vector<float> p1_list = {40, 30, 20, 10, 0, -10};
  std::vector<float> p2_list = {40, 30, 20, 10, 0, -10};

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "--probe")   && i+1<argc) probe_path = argv[++i];
    else if (!strcmp(argv[i], "--echo")    && i+1<argc) echo_path  = argv[++i];
    else if (!strcmp(argv[i], "--seed")    && i+1<argc) seed       = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--payload") && i+1<argc) payload_sz = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--p1")      && i+1<argc) p1_list    = parse_csv(argv[++i]);
    else if (!strcmp(argv[i], "--p2")      && i+1<argc) p2_list    = parse_csv(argv[++i]);
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
    else { fprintf(stderr, "[Eve] unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
  }

  const uint32_t base_seed = seed ? seed : (uint32_t)std::random_device{}();
  fprintf(stderr, "[Eve] seed=%u  payload=%u\n", base_seed, payload_sz);
  fprintf(stderr, "[Eve] probe=%s\n      echo=%s\n", probe_path, echo_path);

  std::vector<CaptureFrame> probe_frames, echo_frames;
  if (!read_capture_file(probe_path, probe_frames)) return 1;
  if (!read_capture_file(echo_path,  echo_frames))  return 1;
  fprintf(stderr, "[Eve] %zu frames read from probe file\n", probe_frames.size());
  fprintf(stderr, "[Eve] %zu frames read from echo  file\n", echo_frames.size());

  // Build probe_id → frame index maps and match pairs
  std::map<uint32_t, size_t> probe_map, echo_map;
  for (size_t i = 0; i < probe_frames.size(); i++) probe_map[probe_frames[i].probe_id] = i;
  for (size_t i = 0; i < echo_frames.size();  i++) echo_map [echo_frames[i].probe_id]  = i;

  struct Pair { size_t pi, ei; };
  std::vector<Pair> pairs;
  for (auto& kv : probe_map) {
    auto it = echo_map.find(kv.first);
    if (it != echo_map.end())
      pairs.push_back({kv.second, it->second});
  }
  fprintf(stderr, "[Eve] %zu matched probe-echo pairs\n", pairs.size());

  if (pairs.empty()) { fprintf(stderr, "[Eve] No pairs — check capture files.\n"); return 1; }

  const uint32_t fss = frame_size_samples(payload_sz);

  // Pre-decode all probes from the CLEAN capture to establish Alice's exact probe knowledge.
  // These should decode at 100% (no noise has been added). Any failures indicate a
  // problem with the capture file itself, not with the experiment.
  std::map<uint32_t, std::vector<uint8_t>> alice_probe_store;
  uint32_t clean_decode_failures = 0;
  for (auto& fr : probe_frames) {
    if (fr.samples.size() < fss) { ++clean_decode_failures; continue; }
    uint32_t pid = 0;
    std::vector<uint8_t> pl;
    if (decode_frame(fr.samples.data(), (uint32_t)fr.samples.size(), payload_sz, pid, pl))
      alice_probe_store[fr.probe_id] = pl;
    else
      ++clean_decode_failures;
  }
  fprintf(stderr,
    "[Eve] Alice probe store: %zu probes decoded clean, %u failures\n",
    alice_probe_store.size(), clean_decode_failures);

  // Filter pairs to only those where Alice's probe is known
  std::vector<Pair> valid_pairs;
  for (auto& pr : pairs) {
    if (alice_probe_store.count(probe_frames[pr.pi].probe_id) &&
        echo_frames[pr.ei].samples.size() >= fss)
      valid_pairs.push_back(pr);
  }
  fprintf(stderr, "[Eve] %zu valid pairs for sweep\n\n", valid_pairs.size());

  const uint32_t N = (uint32_t)valid_pairs.size();
  if (N == 0) { fprintf(stderr, "[Eve] No valid pairs — abort.\n"); return 1; }

  // TSV header (stdout) and stderr column header
  printf("p1_db\tp2_db\talice_secret%%\teve_probe%%\teve_echo%%\teve_secret%%\tadv%%\n");
  fprintf(stderr, "%-8s %-8s  %-15s  %-12s  %-12s  %-13s  %s\n",
          "P1(dB)", "P2(dB)", "alice_secret%", "eve_probe%",
          "eve_echo%", "eve_secret%", "adv%");

  std::mt19937 rng;

  for (float p1 : p1_list) {
    for (float p2 : p2_list) {

      // Re-seed per grid point so each cell is independently reproducible.
      // Encoding (p1, p2) into the seed offset keeps cells truly independent.
      {
        uint32_t p1_bits, p2_bits;
        float p1f = p1, p2f = p2;
        memcpy(&p1_bits, &p1f, 4);
        memcpy(&p2_bits, &p2f, 4);
        rng.seed(base_seed ^ (p1_bits * 2654435761u) ^ (p2_bits * 2246822519u));
      }

      uint32_t alice_ok    = 0;
      uint32_t eve_p_ok    = 0;  // Eve probe decode
      uint32_t eve_e_ok    = 0;  // Eve echo decode
      uint32_t eve_ok      = 0;  // Eve both decode → secret recovered

      for (auto& pr : valid_pairs) {
        const CaptureFrame& pf = probe_frames[pr.pi];
        const CaptureFrame& ef = echo_frames[pr.ei];

        // ── Alice: clean probe knowledge + Phase-2 noisy echo ──────────────
        // Alice's probe payload is known from alice_probe_store (she transmitted it).
        // She receives Bob's echo with Phase-2 channel noise.
        // If her echo decoder passes CRC, she XORs with her stored probe → secret.
        // (The XOR step cannot fail once CRC passes — both payloads are bit-exact.)
        {
          std::vector<cf_t> echo_buf(ef.samples.begin(),
                                      ef.samples.begin() + fss);
          add_awgn(echo_buf.data(), fss, p2, rng);

          uint32_t pid = 0;
          std::vector<uint8_t> pl;
          if (decode_frame(echo_buf.data(), fss, payload_sz, pid, pl))
            alice_ok++;
          // Note: if pid ≠ pf.probe_id after CRC pass that would be a serious bug;
          // in practice this cannot happen with correct probe_id matching.
        }

        // ── Eve: Phase-1 noisy probe + Phase-2 noisy echo (independent) ───
        // Eve receives the probe and echo through her own channel — separate AWGN
        // instances from Alice's, as would occur in a real physical scenario.
        {
          std::vector<cf_t> probe_buf(pf.samples.begin(),
                                       pf.samples.begin() + fss);
          add_awgn(probe_buf.data(), fss, p1, rng);

          std::vector<cf_t> echo_buf(ef.samples.begin(),
                                      ef.samples.begin() + fss);
          add_awgn(echo_buf.data(), fss, p2, rng);

          uint32_t p_pid = 0, e_eid = 0;
          std::vector<uint8_t> p_pl, e_pl;
          const bool p_ok = decode_frame(probe_buf.data(), fss, payload_sz, p_pid, p_pl);
          const bool e_ok = decode_frame(echo_buf.data(),  fss, payload_sz, e_eid, e_pl);

          if (p_ok) eve_p_ok++;
          if (e_ok) eve_e_ok++;

          // Eve recovers the secret iff both CRCs pass and probe IDs match.
          // When both pass: p_pl = probe_payload (bit-exact), e_pl = probe_payload XOR secret.
          // Eve computes: p_pl XOR e_pl = secret. This is always correct when both CRCs pass.
          // There is no middle ground — CRC is all-or-nothing.
          if (p_ok && e_ok && p_pid == e_eid) eve_ok++;
        }
      }

      const float f          = 100.0f / static_cast<float>(N);
      const float alice_pct  = alice_ok  * f;
      const float ep_pct     = eve_p_ok  * f;
      const float ee_pct     = eve_e_ok  * f;
      const float eve_pct    = eve_ok    * f;
      const float adv        = alice_pct - eve_pct;

      fprintf(stderr, "%-8.0f %-8.0f  %-15.1f  %-12.1f  %-12.1f  %-13.1f  %.1f\n",
              p1, p2, alice_pct, ep_pct, ee_pct, eve_pct, adv);
      printf("%.0f\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
             p1, p2, alice_pct, ep_pct, ee_pct, eve_pct, adv);
      fflush(stdout);
    }
    fprintf(stderr, "\n");
  }

  fprintf(stderr, "[Eve] Done. N=%u valid pairs.\n", N);
  return 0;
}