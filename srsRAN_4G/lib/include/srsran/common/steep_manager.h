#ifndef SRSRAN_STEEP_MANAGER_H
#define SRSRAN_STEEP_MANAGER_H

#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>   // FILE*
#include <mutex>
#include <random>   // std::mt19937
#include <vector>
#include "srsran/srslog/srslog.h"

namespace srsran {

// Three states: Alice sits in PROBING while awaiting an echo;
// Bob sits in ECHOING while the echo is queued for the next TX callback.
enum class steep_state_t { IDLE, PROBING, ECHOING };
enum class steep_role_t  { ALICE, BOB };

class steep_manager
{
public:
  using cf_t = std::complex<float>;

  // ── AF model parameters ─────────────────────────────────────────────────────
  // Preamble: 64 chips of a known PN pattern, injected at STEEP_AMP on the
  // imaginary (Q) channel.  Used for sync detection and gain estimation.
  static constexpr uint32_t PREAMBLE_LEN = 64u;

  // Probe body: PROBE_LEN uniform-random real samples following the preamble,
  // also on the Q channel.  Bob buffers these as y1 and forwards them as x2.
  static constexpr uint32_t PROBE_LEN   = 1024u;

  // Total samples consumed / produced per STEEP frame.
  static constexpr uint32_t FRAME_LEN   = PREAMBLE_LEN + PROBE_LEN;   // 1088

  // Amplitudes (pre-ZMQ; ZMQ gain ~1 M× is estimated and corrected for).
  static constexpr float    STEEP_AMP   = 0.5f;   // preamble chip amplitude
  static constexpr float    BPSK_AMP    = 0.1f;   // BPSK secret chip amplitude

  // AF mixing: x2 = C1 * y1_norm + C2 * s2
  // C1 forward-scales the received probe; C2 scales the secret contribution.
  // Both are compile-time constants for the initial implementation; tune them
  // in Step 4 once a round-trip is confirmed working.
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

  // Role and optional secret override (Bob generates its secret dynamically,
  // so set_secret is not required but kept for API compatibility).
  void set_role(steep_role_t role)                    { role_ = role; }
  void set_secret(const std::vector<uint8_t>& secret) { secret_ = secret; }
  void set_payload_size(uint32_t n)                   { payload_size_ = n; }

  // IQ capture (call in ue.cc to record Alice's probe and echo for Eve).
  void set_capture_enabled(bool en) { capture_enabled_ = en; }

  // Phase-1 noise (Bob): simulate a noisier Alice→Bob leg by adding AWGN
  // to a local copy of received samples before preamble detection.
  // The original buffer passed to the LTE stack is never modified.
  void set_phase1_snr_db(float snr_db) {
    phase1_snr_db_        = snr_db;
    phase1_noise_enabled_ = true;
  }
  void disable_phase1_noise() { phase1_noise_enabled_ = false; }

  // AWGN RNG seed.  seed=0 → non-deterministic; seed≠0 → fixed (default 42).
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
  void handle_tx_alice(cf_t* samples, uint32_t nof_samples);
  void handle_rx_alice(const cf_t* samples, uint32_t nof_samples);
  void handle_rx_bob  (const cf_t* samples, uint32_t nof_samples);
  void handle_tx_bob  (cf_t* samples, uint32_t nof_samples);

  // ── AWGN helper (must be called under mtx_; rng_ is not thread-safe) ────────
  void add_awgn(cf_t* samples, uint32_t n, float snr_db);

  // ── IQ capture ──────────────────────────────────────────────────────────────
  void write_capture_frame(FILE*& fp, const char* path,
                           uint8_t role, uint32_t probe_id,
                           const cf_t* samples, uint32_t n_samples);
  void close_capture_files();

  // ── Core state ───────────────────────────────────────────────────────────────
  mutable std::mutex       mtx_;
  steep_state_t            state_        = steep_state_t::IDLE;
  steep_role_t             role_         = steep_role_t::ALICE;
  std::array<uint8_t, 32>  session_key_{};
  std::vector<uint8_t>     secret_;
  uint32_t                 payload_size_ = 32u;   // secret length in bytes (32 B = 256 bits)
  uint32_t                 next_probe_id_ = 1u;
  uint32_t                 tx_call_count_ = 0u;

  // ── Alice-specific state ─────────────────────────────────────────────────────
  // alice_probe_buf_: the PROBE_LEN random samples Alice transmitted (x1).
  // Stored as cf_t with real=0 for uniform access via .imag().
  std::vector<cf_t>  alice_probe_buf_;
  uint32_t           alice_probe_id_  = 0u;
  float              alice_gain_est_  = 1.0f;  // estimated Bob→Alice channel gain
  float              mse_alice_       = 0.0f;  // MSE_A from most-recent echo decode

  // ── Bob-specific state ───────────────────────────────────────────────────────
  // bob_probe_buf_: y1_norm = y1 / Ĝ (gain-normalised received probe).
  std::vector<cf_t>  bob_probe_buf_;
  uint32_t           bob_probe_id_        = 0u;
  float              gain_estimate_       = 1.0f;  // estimated Alice→Bob channel gain
  bool               echo_ready_          = false;
  bool               echo_cooldown_active_ = false;  // true for 200 ms after echo TX
  clock_t::time_point last_echo_time_{};             // timestamp of most-recent echo TX

  // ── Timing (primarily Alice-side; Bob uses echo_cooldown only) ──────────────
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

  // ── Accumulation buffers ─────────────────────────────────────────────────────
  // Incremental search positions allow each on_rx call to scan only newly added
  // samples, bounding the correlation work to O(nof_samples × PREAMBLE_LEN).
  std::vector<cf_t>  rx_accum_buf_;         // Bob RX accumulator
  std::vector<cf_t>  rx_alice_accum_buf_;   // Alice RX accumulator
  uint32_t           rx_bob_search_pos_   = 0u;
  uint32_t           rx_alice_search_pos_ = 0u;
  static constexpr uint32_t RX_ACCUM_MAX  = 65536u;

  // ── IQ capture ──────────────────────────────────────────────────────────────
  bool     capture_enabled_        = false;
  uint32_t capture_frames_written_ = 0u;
  FILE*    probe_capture_fp_       = nullptr;   // /tmp/steep_alice_probe.iq
  FILE*    echo_capture_fp_        = nullptr;   // /tmp/steep_bob_echo.iq

  // ── Phase-1 noise (Bob) ──────────────────────────────────────────────────────
  bool  phase1_noise_enabled_ = false;
  float phase1_snr_db_        = 20.0f;

  // ── RNG ──────────────────────────────────────────────────────────────────────
  uint32_t     rng_seed_ = 42u;
  std::mt19937 rng_;

  srslog::basic_logger& logger_;
};

} // namespace srsran
#endif // SRSRAN_STEEP_MANAGER_H