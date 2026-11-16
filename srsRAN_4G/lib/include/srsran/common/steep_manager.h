#ifndef SRSRAN_STEEP_MANAGER_H
#define SRSRAN_STEEP_MANAGER_H

#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>
#include <vector>
#include "srsran/srslog/srslog.h"

namespace srsran {

// Three states: Node-1 sits in PROBING while awaiting an echo;
// Node-2 sits in ECHOING while the echo is queued for the next TX callback.
enum class steep_state_t { IDLE, PROBING, ECHOING };
enum class steep_role_t  { NODE1, NODE2 };

// Which STEEP scheme a run uses.  BOTH nodes must be started in the same mode.
//   MULTIPLICATIVE  P-STEEP (arXiv 2410.03515 Sec. IV): Node-2 echoes the secret sign times
//                   its soft estimate of the probe:   x2 = C1 * s * y1_norm,  s = +/-1.
//                   Node-1 decodes with sign(rx * x1).
//   ADDITIVE        G-STEEP-style (arXiv 2410.03515 Sec. III; arXiv 2309.14529 Sec. III):
//                   Node-2 echoes its soft probe estimate plus the secret:
//                                                     x2 = C1 * y1_norm + C2 * s.
//                   Node-1 removes C1 * x1 (its own known probe) and decodes the leftover.
//                   NOTE: the paper's G-STEEP uses a Gaussian secret; here s is +/-1,
//                   so this is an approximation of the analyzed scheme.
// Selected at startup from the STEEP_MODE environment variable (see steep_manager.cpp).
enum class steep_mode_t  { MULTIPLICATIVE, ADDITIVE };

class steep_manager
{
public:
  using cf_t = std::complex<float>;

  // ── AF model parameters ─────────────────────────────────────────────────────
  // Preamble: 64 chips of a known PN pattern, injected at STEEP_AMP on the
  // imaginary (Q) channel.  Used for sync detection and gain estimation.
  static constexpr uint32_t PREAMBLE_LEN = 64u;

  // Probe body: PROBE_LEN uniform-random real samples following the preamble,
  // also on the Q channel.  Node-2 buffers these as y1 and forwards them as x2.
  static constexpr uint32_t PROBE_LEN   = 1024u;

  // Total samples consumed / produced per STEEP frame.
  static constexpr uint32_t FRAME_LEN   = PREAMBLE_LEN + PROBE_LEN;   // 1088

  // Amplitudes (pre-ZMQ; ZMQ gain ~1 M× is estimated and corrected for).
  static constexpr float    STEEP_AMP   = 0.5f;   // preamble chip amplitude
  static constexpr float    BPSK_AMP    = 0.1f;   // BPSK secret chip amplitude

  // Echo scaling (see steep_mode_t above).
  //   C1: scale applied to Node-2's soft probe estimate (both modes).
  //   C2: amplitude of the +/-1 secret in ADDITIVE mode only (unused in MULTIPLICATIVE).
  // Both are compile-time constants; tune them once a round trip works.
  // BPSK_AMP is a leftover from the old scheme; it is only used by the placeholder
  // MSE log metric in MULTIPLICATIVE mode (offline analysis is the authority).
  static constexpr float    C1          = 0.5f;
  static constexpr float    C2          = 0.3f;

  // Normalised cross-correlation threshold for preamble detection.
  // Range [0, 1].  0.70 gives strong rejection of noise while accepting a real
  // preamble even at moderate ZMQ gain variation.
  static constexpr float PREAMBLE_THRESHOLD = 0.85f;

  // ── IQ capture ──────────────────────────────────────────────────────────────
  // Frame format: MAGIC(4B) role(1B) probe_id(4B) n_samples(4B) samples(n*8B)
  static constexpr uint32_t CAPTURE_FRAME_MAGIC = 0xCA97CAFEu;
  static constexpr uint32_t CAPTURE_MAX_FRAMES  = 200u;

  steep_manager();
  ~steep_manager();

  // init() must be called before on_tx / on_rx.
  bool init(const std::array<uint8_t, 32>& session_key);
  void reset();

  void set_role(steep_role_t role) {
    role_ = role;
    // Fresh randomness every run. Each node draws its own seed, so Node-1's probe
    // and Node-2's secret are independent. Seed is logged to reproduce a run.
    std::random_device rd;
    rng_seed_ = rd();
    rng_.seed(rng_seed_);
    logger_.warning("STEEP: role=%s mode=%s RNG seed=%u",
                    role == steep_role_t::NODE2 ? "node-2" : "node-1",
                    mode_name(mode_), rng_seed_);
  }

  // Scheme selection.  Normally set from the STEEP_MODE environment variable in the
  // constructor; call set_mode() before traffic starts to override from code.
  void set_mode(steep_mode_t m);
  steep_mode_t get_mode() const { return mode_; }
  static const char* mode_name(steep_mode_t m) {
    return m == steep_mode_t::ADDITIVE ? "additive" : "multiplicative";
  }
  void set_secret(const std::vector<uint8_t>& secret) { secret_ = secret; }
  void set_payload_size(uint32_t n)                   { payload_size_ = n; }

  // IQ capture (call in ue.cc to record Node-1's probe and echo for Eve).
  void set_capture_enabled(bool en) { capture_enabled_ = en; }

  // RNG seed (drives Node-1's random probe).  seed=0 → non-deterministic; seed≠0 → fixed (default 42).
  void set_rng_seed(uint32_t seed);

  steep_state_t get_state() const;

  // Called from radio.cc hooks (already wired; no changes needed there).
  void on_rx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);
  void on_tx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);

  const std::array<uint8_t, 32>& session_key() const { return session_key_; }

private:
  // ── Type alias (must precede any clock_t::time_point members) ───────────────
  using clock_t = std::chrono::steady_clock;

  // ── Preamble signal processing ──────────────────────────────────────────────
  // generate_preamble: write PREAMBLE_LEN cf_t samples to *out.
  //   Real part = 0; imaginary part = chip[i] * STEEP_AMP.
  void  generate_preamble(cf_t* out) const;

  // detect_preamble: linear scan of buf[search_start .. buf.size()-FRAME_LEN].
  //   Returns true and sets offset_out to the first position where the
  //   normalised cross-correlation of the imaginary channel with the known chip
  //   pattern exceeds PREAMBLE_THRESHOLD.
  //   Requires at least FRAME_LEN samples following the candidate position.
  bool  detect_preamble(const std::vector<cf_t>& buf,
                        uint32_t                  search_start,
                        uint32_t&                 offset_out) const;

  // estimate_gain: matched-filter estimate of channel gain from the received
  //   preamble region (PREAMBLE_LEN samples starting at rx_preamble).
  //   G = MF / (STEEP_AMP * PREAMBLE_LEN)
  float estimate_gain(const cf_t* rx_preamble) const;

  // ── Role handlers (called under mtx_) ───────────────────────────────────────
  void handle_tx_node1(cf_t* samples, uint32_t nof_samples);
  void handle_rx_node1(const cf_t* samples, uint32_t nof_samples);
  void handle_rx_node2  (const cf_t* samples, uint32_t nof_samples);
  void handle_tx_node2  (cf_t* samples, uint32_t nof_samples);

  // ── IQ capture ──────────────────────────────────────────────────────────────
  void write_capture_frame(FILE*& fp, const char* path,
                           uint8_t role, uint32_t probe_id,
                           const cf_t* samples, uint32_t n_samples);
  void close_capture_files();

  // ── Core state ───────────────────────────────────────────────────────────────
  mutable std::mutex       mtx_;
  steep_state_t            state_        = steep_state_t::IDLE;
  steep_role_t             role_         = steep_role_t::NODE1;
  steep_mode_t             mode_         = steep_mode_t::MULTIPLICATIVE;
  bool                     enabled_      = true;   // false when STEEP_ENABLE=0 (hooks do nothing)
  std::array<uint8_t, 32>  session_key_{};
  std::vector<uint8_t>     secret_;
  uint32_t                 payload_size_ = 32u;   // secret length in bytes (32 B = 256 bits)
  uint32_t                 next_probe_id_ = 1u;
  uint32_t                 tx_call_count_ = 0u;

  // ── Node-1-specific state ─────────────────────────────────────────────────────
  // node1_probe_buf_: the PROBE_LEN random samples Node-1 transmitted (x1).
  // Stored as cf_t with real=0 for uniform access via .imag().
  std::vector<cf_t>  node1_probe_buf_;
  uint32_t           node1_probe_id_  = 0u;
  float              node1_gain_est_  = 1.0f;  // estimated Node-2→Node-1 channel gain
  float              mse_node1_       = 0.0f;  // MSE_A from most-recent echo decode

  // ── Node-2-specific state ───────────────────────────────────────────────────────
  // node2_probe_buf_: y1_norm = y1 / Ĝ (gain-normalised received probe).
  std::vector<cf_t>  node2_probe_buf_;
  uint32_t           node2_probe_id_        = 0u;
  float              gain_estimate_       = 1.0f;  // estimated Node-1→Node-2 channel gain
  bool               echo_ready_          = false;
  bool               echo_cooldown_active_ = false;  // true for 200 ms after echo TX
  clock_t::time_point last_echo_time_{};             // timestamp of most-recent echo TX

  // ── Timing (primarily Node-1-side; Node-2 uses echo_cooldown only) ──────────────
  clock_t::time_point start_time_{};
  clock_t::time_point last_probe_time_{};
  clock_t::time_point probing_since_time_{};
  clock_t::time_point last_heartbeat_time_{};
  bool                timing_initialized_ = false;

  static constexpr uint32_t STARTUP_DELAY_SECS  = 5u;
  static constexpr uint32_t PROBE_INTERVAL_SECS = 1u;
  static constexpr uint32_t PROBE_TIMEOUT_SECS  = 8u;

  // ── Stats ────────────────────────────────────────────────────────────────────
  uint32_t probes_sent_      = 0u;
  uint32_t probes_recovered_ = 0u;
  uint32_t echoes_sent_      = 0u;
  uint32_t probes_timed_out_ = 0u;
  uint32_t dup_probes_dropped_ = 0u;   // Node-2: repeat detections of the same probe, not echoed

  // ── Accumulation buffers ─────────────────────────────────────────────────────
  // Incremental search positions allow each on_rx call to scan only newly added
  // samples, bounding the correlation work to O(nof_samples × PREAMBLE_LEN).
  std::vector<cf_t>  rx_accum_buf_;         // Node-2 RX accumulator
  std::vector<cf_t>  rx_node1_accum_buf_;   // Node-1 RX accumulator
  uint32_t           rx_node2_search_pos_   = 0u;
  uint32_t           rx_node1_search_pos_ = 0u;
  static constexpr uint32_t RX_ACCUM_MAX  = 65536u;

  // ── IQ capture ──────────────────────────────────────────────────────────────
  bool     capture_enabled_        = false;
  uint32_t capture_frames_written_ = 0u;
  FILE*    probe_capture_fp_       = nullptr;   // /tmp/steep_node1_probe.iq
  FILE*    echo_capture_fp_        = nullptr;   // /tmp/steep_node2_echo.iq
  FILE*    node2_rx_capture_fp_      = nullptr;   // /tmp/steep_node2_rx.iq
  FILE*    x1_capture_fp_          = nullptr;   // /tmp/steep_x1.iq
  FILE*    s2_capture_fp_          = nullptr;   // /tmp/steep_s2.iq
  FILE*    x1_unit_capture_fp_     = nullptr;   // /tmp/steep_x1_unit.iq
  FILE*    s2_unit_capture_fp_     = nullptr;   // /tmp/steep_s2_unit.iq
  FILE*    x1_full_capture_fp_      = nullptr;   // /tmp/steep_x1_full.iq
  FILE*    x1_full_unit_capture_fp_ = nullptr;   // /tmp/steep_x1_full_unit.iq
  FILE*    s2_full_capture_fp_      = nullptr;   // /tmp/steep_s2_full.iq
  FILE*    s2_full_unit_capture_fp_ = nullptr;   // /tmp/steep_s2_full_unit.iq

  // ── RNG ──────────────────────────────────────────────────────────────────────
  uint32_t     rng_seed_ = 42u;
  std::mt19937 rng_;

  srslog::basic_logger& logger_;
};

} // namespace srsran
#endif // SRSRAN_STEEP_MANAGER_H