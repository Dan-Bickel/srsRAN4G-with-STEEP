#include "srsran/common/steep_manager.h"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ── Out-of-class definitions (C++14 ODR requirement for static constexpr) ────
// C++17 makes these implicit; C++14 requires one definition per TU that ODR-uses
// the constant (i.e. passes it by reference or takes its address).
constexpr uint32_t srsran::steep_manager::PREAMBLE_LEN;
constexpr uint32_t srsran::steep_manager::PROBE_LEN;
constexpr uint32_t srsran::steep_manager::FRAME_LEN;
constexpr float    srsran::steep_manager::STEEP_AMP;
constexpr float    srsran::steep_manager::BPSK_AMP;
constexpr float    srsran::steep_manager::C1;
constexpr float    srsran::steep_manager::C2;
constexpr float    srsran::steep_manager::PREAMBLE_THRESHOLD;
constexpr uint32_t srsran::steep_manager::CAPTURE_FRAME_MAGIC;
constexpr uint32_t srsran::steep_manager::CAPTURE_MAX_FRAMES;
constexpr uint32_t srsran::steep_manager::STARTUP_DELAY_SECS;
constexpr uint32_t srsran::steep_manager::PROBE_INTERVAL_SECS;
constexpr uint32_t srsran::steep_manager::PROBE_TIMEOUT_SECS;
constexpr uint32_t srsran::steep_manager::RX_ACCUM_MAX;

namespace srsran {

// ── Preamble chip pattern ─────────────────────────────────────────────────────
// The 32-chip PN sequence from the original design repeated twice to give
// PREAMBLE_LEN = 64 bipolar (±1) chips.  Injected on the imaginary channel at
// amplitude STEEP_AMP.  The autocorrelation peak at zero lag makes it easy to
// detect and gives a clean matched-filter gain estimate.
static constexpr int8_t PREAMBLE_CHIP[steep_manager::PREAMBLE_LEN] = {
  +1,-1,+1,-1,+1,+1,-1,-1,+1,+1,+1,-1,+1,-1,-1,+1,
  -1,+1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,+1,-1,
  +1,-1,+1,-1,+1,+1,-1,-1,+1,+1,+1,-1,+1,-1,-1,+1,
  -1,+1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,+1,-1
};

// ── Link measurement (for hardware bring-up) ─────────────────────────────────
// Compares the received preamble with the known pattern on BOTH channels.
//   phase_deg : rotation of the received signal.  Near 0 means the Q-only design is
//               fine; large values mean energy is leaking from Q into I (a rotated
//               channel), which would make the sign decode fail.
//   i_over_q  : strength on the I channel relative to Q (0 = clean).
//   peak      : largest sample magnitude in the frame (check against ADC clipping).
static void link_measure(const std::complex<float>* pre, const std::complex<float>* frame,
                         uint32_t frame_len, float& phase_deg, float& i_over_q, float& peak)
{
  double ci = 0.0, cq = 0.0;
  for (uint32_t k = 0; k < steep_manager::PREAMBLE_LEN; k++) {
    const double chip = static_cast<double>(PREAMBLE_CHIP[k]);
    ci += chip * pre[k].real();
    cq += chip * pre[k].imag();
  }
  phase_deg = static_cast<float>(std::atan2(ci, cq) * 180.0 / 3.14159265358979323846);
  i_over_q  = (std::fabs(cq) > 0.0) ? static_cast<float>(std::fabs(ci) / std::fabs(cq)) : 0.0f;
  peak = 0.0f;
  for (uint32_t k = 0; k < frame_len; k++) peak = std::max(peak, std::abs(frame[k]));
}

// ── Constructor / Destructor ──────────────────────────────────────────────────

steep_manager::steep_manager()
  : logger_(srslog::fetch_basic_logger("STEEP")),
    rng_(42u)
{
  logger_.set_level(srslog::basic_levels::info);

  // Mode selection from the STEEP_MODE environment variable:
  //   STEEP_MODE=mult | multiplicative   -> multiplicative (P-STEEP)   [default]
  //   STEEP_MODE=add  | additive         -> additive (G-STEEP-style)
  // NOTE: sudo discards environment variables placed before it, so run
  //   sudo STEEP_MODE=add <path>/srsenb ...      (variable AFTER sudo).
  // BOTH nodes must be started with the same value.
  // STEEP on/off from the STEEP_ENABLE environment variable (default: on).
  //   STEEP_ENABLE=0 | off | false | no  -> the hooks do nothing, so the radio runs as
  //   plain LTE.  Use this for the baseline run in the lab, to check the link before STEEP.
  // (Like STEEP_MODE, put it AFTER sudo:  sudo STEEP_ENABLE=0 <path>/srsenb ...)
  const char* en = std::getenv("STEEP_ENABLE");
  if (en != nullptr && en[0] != '\0') {
    std::string e(en);
    for (auto& ch : e) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (e == "0" || e == "off" || e == "false" || e == "no") enabled_ = false;
  }
  if (!enabled_) logger_.warning("STEEP: DISABLED by STEEP_ENABLE; radio runs as plain LTE");

  const char* env = std::getenv("STEEP_MODE");
  if (env != nullptr && env[0] != '\0') {
    std::string v(env);
    for (auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (v == "add" || v == "additive") {
      mode_ = steep_mode_t::ADDITIVE;
    } else if (v == "mult" || v == "multiplicative") {
      mode_ = steep_mode_t::MULTIPLICATIVE;
    } else {
      logger_.warning("STEEP: unrecognised STEEP_MODE='%s' (use 'mult' or 'add'); "
                      "defaulting to multiplicative", env);
      mode_ = steep_mode_t::MULTIPLICATIVE;
    }
  }
  logger_.warning("STEEP manager constructed, mode=%s", mode_name(mode_));
}

void steep_manager::set_mode(steep_mode_t m)
{
  std::lock_guard<std::mutex> lock(mtx_);
  mode_ = m;
  logger_.warning("STEEP: mode set to %s", mode_name(mode_));
}

steep_manager::~steep_manager()
{
  close_capture_files();
}

// ── Init / Reset ──────────────────────────────────────────────────────────────

bool steep_manager::init(const std::array<uint8_t, 32>& key)
{
  std::lock_guard<std::mutex> lock(mtx_);
  session_key_             = key;
  state_                   = steep_state_t::IDLE;
  echo_ready_              = false;
  echo_cooldown_active_    = false;
  next_probe_id_           = 1u;
  tx_call_count_           = 0u;
  timing_initialized_      = false;
  probes_sent_             = 0u;
  probes_recovered_        = 0u;
  echoes_sent_             = 0u;
  probes_timed_out_        = 0u;
  capture_frames_written_  = 0u;
  node1_probe_id_          = 0u;
  node1_gain_est_          = 1.0f;
  mse_node1_               = 0.0f;
  node2_probe_id_            = 0u;
  gain_estimate_           = 1.0f;
  rx_node2_search_pos_       = 0u;
  rx_node1_search_pos_     = 0u;
  node1_probe_buf_.clear();
  node2_probe_buf_.clear();
  rx_accum_buf_.clear();
  rx_node1_accum_buf_.clear();
  close_capture_files();
  logger_.info("STEEP manager initialized (AF model, FRAME_LEN=%u, payload_size=%u B)",
               FRAME_LEN, payload_size_);
  return true;
}

void steep_manager::reset()
{
  std::lock_guard<std::mutex> lock(mtx_);
  state_                = steep_state_t::IDLE;
  echo_ready_           = false;
  echo_cooldown_active_ = false;
  tx_call_count_        = 0u;
  timing_initialized_   = false;
  capture_frames_written_ = 0u;
  rx_node2_search_pos_    = 0u;
  rx_node1_search_pos_  = 0u;
  node1_probe_buf_.clear();
  node2_probe_buf_.clear();
  rx_accum_buf_.clear();
  rx_node1_accum_buf_.clear();
  close_capture_files();
  logger_.warning("STEEP manager reset — final stats: sent=%u recovered=%u echoed=%u timed_out=%u",
                  probes_sent_, probes_recovered_, echoes_sent_, probes_timed_out_);
}

// ── RNG seed ──────────────────────────────────────────────────────────────────

void steep_manager::set_rng_seed(uint32_t seed)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (seed == 0) {
    std::random_device rd;
    rng_seed_ = rd();
  } else {
    rng_seed_ = seed;
  }
  rng_.seed(rng_seed_);
  logger_.warning("STEEP: RNG seeded with %u (%s)",
                  rng_seed_, seed == 0 ? "random" : "fixed");
}

// ── State ─────────────────────────────────────────────────────────────────────

steep_state_t steep_manager::get_state() const
{
  std::lock_guard<std::mutex> lock(mtx_);
  return state_;
}

// ── Preamble generation ───────────────────────────────────────────────────────

void steep_manager::generate_preamble(cf_t* out) const
{
  for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
    out[i] = cf_t(0.0f, static_cast<float>(PREAMBLE_CHIP[i]) * STEEP_AMP);
}

// ── Preamble detection ────────────────────────────────────────────────────────
// Linear scan from search_start.  For each candidate offset, computes the
// normalised cross-correlation of the buffer's imaginary channel with the known
// chip pattern.  Returns the first offset that exceeds PREAMBLE_THRESHOLD.
//
// Only positions where at least FRAME_LEN samples follow the preamble start are
// considered, so a successful detection always guarantees the full probe body
// is already in the buffer.
//
// Cost per call: O((new_samples + PREAMBLE_LEN) × PREAMBLE_LEN) when the caller
// passes the previous high-water mark as search_start.

bool steep_manager::detect_preamble(const std::vector<cf_t>& buf,
                                     uint32_t                  search_start,
                                     uint32_t&                 offset_out) const
{
  if (buf.size() < FRAME_LEN) return false;
  const uint32_t search_end = static_cast<uint32_t>(buf.size()) - FRAME_LEN;
  if (search_start > search_end) return false;

  float    best_corr   = 0.0f;
  uint32_t best_offset = 0u;
  bool     found       = false;

  for (uint32_t off = search_start; off <= search_end; off++) {
    float corr      = 0.0f;
    float rx_energy = 0.0f;
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++) {
      const float v  = buf[off + i].imag();
      corr      += v * static_cast<float>(PREAMBLE_CHIP[i]);
      rx_energy += v * v;
    }
    if (rx_energy < 1e-12f) continue;
    const float norm_corr = corr / std::sqrt(rx_energy * static_cast<float>(PREAMBLE_LEN));
    if (norm_corr >= PREAMBLE_THRESHOLD && norm_corr > best_corr) {
      best_corr   = norm_corr;
      best_offset = off;
      found       = true;
    }
  }

  if (found) { offset_out = best_offset; }
  return found;
}

// ── Gain estimation ───────────────────────────────────────────────────────────
// Matched-filter estimate of the scalar channel gain G from PREAMBLE_LEN
// received samples starting at rx_preamble.
//
// Derivation:
//   Received preamble (noiseless): rx[i].imag = G * chip[i] * STEEP_AMP
//   MF output:                     sum(rx[i].imag * chip[i])
//                                = G * STEEP_AMP * sum(chip[i]^2)
//                                = G * STEEP_AMP * PREAMBLE_LEN   (chips are ±1)
//   Therefore:                     G = MF / (STEEP_AMP * PREAMBLE_LEN)

float steep_manager::estimate_gain(const cf_t* rx_preamble) const
{
  float mf = 0.0f;
  for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
    mf += rx_preamble[i].imag() * static_cast<float>(PREAMBLE_CHIP[i]);
  const float gain = mf / (STEEP_AMP * static_cast<float>(PREAMBLE_LEN));
  // Floor at a small positive value to guard against numerical issues.
  // A real preamble at any plausible ZMQ gain produces gain >> 1.
  return (gain > 0.01f) ? gain : 0.01f;
}

// ── IQ capture ────────────────────────────────────────────────────────────────

void steep_manager::write_capture_frame(FILE*& fp, const char* path,
                                         uint8_t role, uint32_t probe_id,
                                         const cf_t* samples, uint32_t n_samples)
{
  if (!fp) {
    fp = fopen(path, "wb");
    if (!fp) { logger_.error("STEEP capture: cannot open '%s'", path); return; }
    logger_.warning("STEEP capture: opened '%s'", path);
  }
  const uint32_t magic = CAPTURE_FRAME_MAGIC;
  fwrite(&magic,     sizeof(uint32_t), 1,        fp);
  fwrite(&role,      sizeof(uint8_t),  1,        fp);
  fwrite(&probe_id,  sizeof(uint32_t), 1,        fp);
  fwrite(&n_samples, sizeof(uint32_t), 1,        fp);
  fwrite(samples,    sizeof(cf_t),     n_samples, fp);
  fflush(fp);
}

void steep_manager::close_capture_files()
{
  if (probe_capture_fp_) { fclose(probe_capture_fp_); probe_capture_fp_ = nullptr; }
  if (echo_capture_fp_)  { fclose(echo_capture_fp_);  echo_capture_fp_  = nullptr; }
  if (node2_rx_capture_fp_) { fclose(node2_rx_capture_fp_); node2_rx_capture_fp_ = nullptr; }
  if (x1_capture_fp_) { fclose(x1_capture_fp_); x1_capture_fp_ = nullptr; }
  if (s2_capture_fp_) { fclose(s2_capture_fp_); s2_capture_fp_ = nullptr; }
  if (x1_unit_capture_fp_) { fclose(x1_unit_capture_fp_); x1_unit_capture_fp_ = nullptr; }
  if (s2_unit_capture_fp_) { fclose(s2_unit_capture_fp_); s2_unit_capture_fp_ = nullptr; }
  if (x1_full_capture_fp_)      { fclose(x1_full_capture_fp_);      x1_full_capture_fp_      = nullptr; }
  if (x1_full_unit_capture_fp_) { fclose(x1_full_unit_capture_fp_); x1_full_unit_capture_fp_ = nullptr; }
  if (s2_full_capture_fp_)      { fclose(s2_full_capture_fp_);      s2_full_capture_fp_      = nullptr; }
  if (s2_full_unit_capture_fp_) { fclose(s2_full_unit_capture_fp_); s2_full_unit_capture_fp_ = nullptr; }
}

// ── Node-1 TX ─────────────────────────────────────────────────────────────────
// Generates one STEEP frame per PROBE_INTERVAL_SECS after startup:
//   [PREAMBLE_LEN chips at STEEP_AMP] [PROBE_LEN random uniform samples]
// Both are injected into the imaginary (Q) channel; the real (I) LTE signal
// is untouched.  The probe body is stored in node1_probe_buf_ for Phase-2
// subtraction when the echo arrives.

void steep_manager::handle_tx_node1(cf_t* samples, uint32_t nof_samples)
{
  ++tx_call_count_;
  const auto now = clock_t::now();

  if (!timing_initialized_) {
    start_time_          = now;
    last_probe_time_     = now;
    last_heartbeat_time_ = now;
    timing_initialized_  = true;
    logger_.warning("STEEP [NODE-1]: timing initialized, FRAME_LEN=%u nof_samples=%u",
                    FRAME_LEN, nof_samples);
  }

  auto elapsed_secs = [&](clock_t::time_point from) -> long long {
    return std::chrono::duration_cast<std::chrono::seconds>(now - from).count();
  };

  if (elapsed_secs(last_heartbeat_time_) >= 5) {
    last_heartbeat_time_ = now;
    logger_.warning(
        "STEEP [NODE-1] heartbeat: mode=%s tx_calls=%u state=%d sent=%u recovered=%u timed_out=%u",
        mode_name(mode_), tx_call_count_, static_cast<int>(state_),
        probes_sent_, probes_recovered_, probes_timed_out_);
  }

  if (elapsed_secs(start_time_) < static_cast<long long>(STARTUP_DELAY_SECS)) return;

  // ── Timeout: probe outstanding too long ───────────────────────────────────
  if (state_ == steep_state_t::PROBING) {
    if (elapsed_secs(probing_since_time_) >= static_cast<long long>(PROBE_TIMEOUT_SECS)) {
      logger_.warning("STEEP [NODE-1]: probe #%u timed out — returning to IDLE", node1_probe_id_);
      ++probes_timed_out_;
      state_              = steep_state_t::IDLE;
      last_probe_time_    = now;
      node1_probe_buf_.clear();
      rx_node1_accum_buf_.clear();
      rx_node1_search_pos_ = 0u;
    }
    return;   // stay in PROBING (or just reset to IDLE above)
  }

  if (state_ != steep_state_t::IDLE)                                     return;
  if (elapsed_secs(last_probe_time_) < static_cast<long long>(PROBE_INTERVAL_SECS)) return;
  if (nof_samples < FRAME_LEN)                                           return;

  // ── Inject frame into imaginary channel ───────────────────────────────────
  // Clear Q channel for the whole frame region first.
  for (uint32_t i = 0; i < FRAME_LEN; i++)
    samples[i] = cf_t(samples[i].real(), 0.0f);

  // Preamble: known chip pattern at STEEP_AMP.
  for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
    samples[i] = cf_t(samples[i].real(),
                      static_cast<float>(PREAMBLE_CHIP[i]) * STEEP_AMP);

  // Probe body: random BPSK, each sample is +STEEP_AMP or -STEEP_AMP (P-STEEP, M=2).
  // Stored as (0, v) complex so .imag() gives the scalar probe value.
  std::bernoulli_distribution probe_bit(0.5);
  node1_probe_buf_.resize(PROBE_LEN);
  for (uint32_t i = 0; i < PROBE_LEN; i++) {
    const float v     = probe_bit(rng_) ? STEEP_AMP : -STEEP_AMP;
    node1_probe_buf_[i]         = cf_t(0.0f, v);
    samples[PREAMBLE_LEN + i]   = cf_t(samples[PREAMBLE_LEN + i].real(), v);
  }

  node1_probe_id_     = next_probe_id_++;
  state_              = steep_state_t::PROBING;
  probing_since_time_ = now;
  last_probe_time_    = now;
  rx_node1_accum_buf_.clear();
  rx_node1_search_pos_ = 0u;
  ++probes_sent_;

  logger_.warning("STEEP [NODE-1]: probe #%u transmitted, N=%u samples (sent=%u recovered=%u)",
                  node1_probe_id_, PROBE_LEN, probes_sent_, probes_recovered_);

  if (capture_enabled_ && capture_frames_written_ < CAPTURE_MAX_FRAMES) {
    write_capture_frame(probe_capture_fp_, "/tmp/steep_node1_probe.iq",
                        0u, node1_probe_id_, samples, FRAME_LEN);
    ++capture_frames_written_;

    // x1 alone: preamble + probe, pure STEEP samples (I = 0, no LTE).
    std::vector<cf_t> x1_clean(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
      x1_clean[i] = cf_t(0.0f, static_cast<float>(PREAMBLE_CHIP[i]) * STEEP_AMP);
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      x1_clean[PREAMBLE_LEN + i] = node1_probe_buf_[i];
    write_capture_frame(x1_capture_fp_, "/tmp/steep_x1.iq",
                        3u, node1_probe_id_, x1_clean.data(), FRAME_LEN);
    
    // Same signal, unit amplitude (+/-1).
    std::vector<cf_t> x1_unit(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
      x1_unit[i] = cf_t(0.0f, static_cast<float>(PREAMBLE_CHIP[i]));
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      x1_unit[PREAMBLE_LEN + i] = cf_t(0.0f, node1_probe_buf_[i].imag() > 0.0f ? 1.0f : -1.0f);
    write_capture_frame(x1_unit_capture_fp_, "/tmp/steep_x1_unit.iq",
                        5u, node1_probe_id_, x1_unit.data(), FRAME_LEN);


    // Full I/Q version (saved only, NOT transmitted): the transmitted probe stays
    // on Q; an independent +/-A probe is generated for I.
    std::bernoulli_distribution i_bit(0.5);
    std::vector<cf_t> x1_full(FRAME_LEN), x1_full_unit(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++) {
      const float c = static_cast<float>(PREAMBLE_CHIP[i]);
      x1_full[i]      = cf_t(c * STEEP_AMP, c * STEEP_AMP);
      x1_full_unit[i] = cf_t(c, c);
    }
    for (uint32_t i = 0; i < PROBE_LEN; i++) {
      const float q  = node1_probe_buf_[i].imag();
      const float iv = i_bit(rng_) ? STEEP_AMP : -STEEP_AMP;
      x1_full[PREAMBLE_LEN + i]      = cf_t(iv, q);
      x1_full_unit[PREAMBLE_LEN + i] = cf_t(iv > 0.0f ? 1.0f : -1.0f,
                                            q  > 0.0f ? 1.0f : -1.0f);
    }
    write_capture_frame(x1_full_capture_fp_, "/tmp/steep_x1_full.iq",
                        7u, node1_probe_id_, x1_full.data(), FRAME_LEN);
    write_capture_frame(x1_full_unit_capture_fp_, "/tmp/steep_x1_full_unit.iq",
                        8u, node1_probe_id_, x1_full_unit.data(), FRAME_LEN);
  }
}

// ── Node-1 RX ─────────────────────────────────────────────────────────────────
// Waits for Node-2's echo, then:
//   residual[i] = y2[i].imag() − C1 * x1[i].imag() * node1_gain_est_
//               ≈ C2 * s2[i] * G_BA  (noise terms omitted from this sketch)
// BPSK threshold-decode the first (payload_size_ * 8) residual samples to
// recover the secret bits.  Compute MSE_A as the mean squared deviation of
// the residual from the decoded ideal amplitude; used later for Rs.

void steep_manager::handle_rx_node1(const cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::PROBING) return;

  // Always accumulate regardless of wait status — we must not lose samples
  // that may contain the preamble.
  rx_node1_accum_buf_.insert(rx_node1_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_node1_accum_buf_.size() > RX_ACCUM_MAX) {
    const uint32_t trim =
        static_cast<uint32_t>(rx_node1_accum_buf_.size()) - RX_ACCUM_MAX;
    rx_node1_accum_buf_.erase(rx_node1_accum_buf_.begin(),
                               rx_node1_accum_buf_.begin() + trim);
    rx_node1_search_pos_ = (rx_node1_search_pos_ >= trim)
                           ? rx_node1_search_pos_ - trim : 0u;
  }

  // 20 ms anti-loopback gate: any ZMQ loopback of Node-1's own probe arrives
  // in < 1 ms; Node-2's legitimate echo takes at least 10–15 ms (detection +
  // TX scheduling).  20 ms clears both.
  const auto now        = clock_t::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now - probing_since_time_).count();
  if (elapsed_ms < 20) return;

  if (rx_node1_accum_buf_.size() < FRAME_LEN) return;

  uint32_t offset = 0;
  if (!detect_preamble(rx_node1_accum_buf_, rx_node1_search_pos_, offset)) {
    // Advance the resume position so we never re-scan settled territory.
    if (rx_node1_accum_buf_.size() >= FRAME_LEN)
      rx_node1_search_pos_ =
          static_cast<uint32_t>(rx_node1_accum_buf_.size()) - FRAME_LEN + 1u;
    return;
  }

  // Ensure the full probe body is already in the buffer.
  if (offset + FRAME_LEN > rx_node1_accum_buf_.size()) return;

  // ── Estimate Node-2→Node-1 channel gain from the echo preamble ────────────────
  node1_gain_est_ = estimate_gain(rx_node1_accum_buf_.data() + offset);
  logger_.info("STEEP [NODE-1]: echo preamble at offset=%u, gain=%.2e", offset, node1_gain_est_);
  {
    float ph = 0.0f, ioq = 0.0f, pk = 0.0f;
    link_measure(rx_node1_accum_buf_.data() + offset, rx_node1_accum_buf_.data() + offset,
                 FRAME_LEN, ph, ioq, pk);
    logger_.warning("STEEP [NODE-1] LINK: gain=%.3e phase_deg=%.1f i_over_q=%.3f peak=%.3e",
                    node1_gain_est_, ph, ioq, pk);
  }

  const cf_t* y2 = rx_node1_accum_buf_.data() + offset + PREAMBLE_LEN;
  const uint32_t n_secret_bits = payload_size_ * 8u;

  // Decoding depends on the mode (both rely on Node-1 knowing its own probe x1 exactly,
  // which is the security property of STEEP):
  //   MULTIPLICATIVE:  rx = G * C1 * b * (x1 + noise)   ->  sign(rx * x1) = b
  //   ADDITIVE:        rx = G * (C1 * (x1 + noise) + C2 * b)
  //                    -> r = rx / G - C1 * x1  ~=  C2 * b + noise,  sign(r) = b
  //                    (G = node1_gain_est_, measured from the echo preamble)
  //
  // Clamp to PROBE_LEN to guard against a probe body shorter than
  // payload_size_ * 8 (should never happen with default sizes but let's
  // be defensive).
  const uint32_t decode_len = std::min(n_secret_bits, PROBE_LEN);
  if (node1_probe_buf_.size() < decode_len) {
    logger_.error("STEEP [NODE-1]: node1_probe_buf_ too short (%zu < %u) — discarding echo",
                  node1_probe_buf_.size(), decode_len);
    rx_node1_accum_buf_.erase(rx_node1_accum_buf_.begin(),
                               rx_node1_accum_buf_.begin() + offset + FRAME_LEN);
    rx_node1_search_pos_ = 0u;
    return;
  }

  std::vector<uint8_t> recovered(payload_size_, 0u);
  float mse_accum = 0.0f;
  for (uint32_t i = 0; i < decode_len; i++) {
    const float rx_q   = y2[i].imag();
    const float x1_q   = node1_probe_buf_[i].imag();         // Node-1's known probe
    float resid;
    float ideal;
    if (mode_ == steep_mode_t::ADDITIVE) {
      resid = rx_q / node1_gain_est_ - C1 * x1_q;   // ~= C2 * b
      ideal = ((resid >= 0.0f) ? 1.0f : -1.0f) * C2;
    } else {
      resid = rx_q * x1_q;   // = G*C1*b*x1^2, so sign(resid) = secret bit b
      ideal = ((resid >= 0.0f) ? 1.0f : -1.0f) * C2 * BPSK_AMP;   // placeholder metric
    }
    // BPSK threshold: positive residual → bit 1, negative → bit 0.
    const int   bit_val = (resid >= 0.0f) ? 1 : 0;
    const uint32_t byte_idx = i / 8u;
    const uint32_t bit_pos  = 7u - (i % 8u);
    if (bit_val) recovered[byte_idx] |= (1u << bit_pos);

    const float err   = resid - ideal;
    mse_accum += err * err;
  }
  mse_node1_ = mse_accum / static_cast<float>(decode_len);

  // ── Recover Rs ────────────────────────────────────────────────────────────
  // MSE_A is known; MSE_E is not yet available (Eve is deferred).
  // Log MSE_A so it can be compared manually or scripted against Eve's output.
  // Rs = log2(MSE_E / MSE_A) — noted here as a formula; will be computed
  // properly once Eve is implemented.

  // ── Build ASCII and hex representations ──────────────────────────────────
  std::string ascii, hex;
  char buf[4];
  for (auto b : recovered) {
    snprintf(buf, sizeof(buf), "%02x ", b);
    hex += buf;
    ascii += (b >= 32u && b < 127u) ? static_cast<char>(b) : '.';
  }

  ++probes_recovered_;
  const float rate = probes_sent_ > 0u
                     ? (100.0f * probes_recovered_ / probes_sent_) : 0.0f;

  logger_.warning(
      "STEEP [NODE-1]: *** SECRET RECOVERED probe=#%u offset=%u | "
      "ASCII: \"%s\" | HEX: %s| MSE_A=%.4e gain=%.2e Rs=log2(MSE_E/%.4e) bits ***",
      node1_probe_id_, offset, ascii.c_str(), hex.c_str(),
      mse_node1_, node1_gain_est_, mse_node1_);
  logger_.warning(
      "STEEP [NODE-1]: stats: sent=%u recovered=%u timed_out=%u success_rate=%.1f%%",
      probes_sent_, probes_recovered_, probes_timed_out_, rate);

  // ── Advance and reset ─────────────────────────────────────────────────────
  rx_node1_accum_buf_.erase(rx_node1_accum_buf_.begin(),
                             rx_node1_accum_buf_.begin() + offset + FRAME_LEN);
  rx_node1_search_pos_ = 0u;
  node1_probe_buf_.clear();
  state_ = steep_state_t::IDLE;
}

// ── Node-2 RX ────────────────────────────────────────────────────────────────────
// Detects Node-1's preamble, estimates the Node-1→Node-2 channel gain Ĝ, then
// stores the gain-normalised probe body:
//   y1_norm[i] = y1[i].imag() / Ĝ
// as node2_probe_buf_ (imaginary part only; stored as cf_t for uniformity).
// Transitions to ECHOING so that handle_tx_node2 fires on the next TX callback.

void steep_manager::handle_rx_node2(const cf_t* samples, uint32_t nof_samples)
{
  // During the echo cooldown window (200 ms after a TX), ignore incoming
  // samples to avoid re-detecting our own echo as a new probe.
  if (echo_cooldown_active_) {
    const auto now = clock_t::now();
    const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - last_echo_time_).count();
    if (ms < 1000) return;
    echo_cooldown_active_ = false;
    rx_accum_buf_.clear();
    rx_node2_search_pos_ = 0u;
    logger_.info("STEEP [NODE-2]: echo cooldown expired — listening resumed");
  }

  if (state_ != steep_state_t::IDLE) return;

  rx_accum_buf_.insert(rx_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_accum_buf_.size() > RX_ACCUM_MAX) {
    const uint32_t trim =
        static_cast<uint32_t>(rx_accum_buf_.size()) - RX_ACCUM_MAX;
    rx_accum_buf_.erase(rx_accum_buf_.begin(),
                         rx_accum_buf_.begin() + trim);
    rx_node2_search_pos_ = (rx_node2_search_pos_ >= trim)
                         ? rx_node2_search_pos_ - trim : 0u;
  }

  if (rx_accum_buf_.size() < FRAME_LEN) return;

  uint32_t offset = 0;
  if (!detect_preamble(rx_accum_buf_, rx_node2_search_pos_, offset)) {
    rx_node2_search_pos_ =
        static_cast<uint32_t>(rx_accum_buf_.size()) - FRAME_LEN + 1u;
    return;
  }

  if (offset + FRAME_LEN > rx_accum_buf_.size()) return;

  // ── Gain estimation ───────────────────────────────────────────────────────
  gain_estimate_ = estimate_gain(rx_accum_buf_.data() + offset);
  logger_.info("STEEP [NODE-2]: preamble at offset=%u, gain=%.2e", offset, gain_estimate_);

  // ── Duplicate-probe guard ─────────────────────────────────────────────────
  // Node-1 draws a fresh random probe every time, so two genuine probes are
  // essentially uncorrelated (|corr| ~ 0.03 for 1024 samples).  If the newly
  // detected probe body matches the previously buffered one almost exactly, it is
  // the SAME probe seen a second time (stale/replayed samples), and echoing it
  // again would use a second secret on the same probe, which leaks information.
  const cf_t* y1_start = rx_accum_buf_.data() + offset + PREAMBLE_LEN;
  float dup_corr = 0.0f;
  if (node2_probe_buf_.size() == PROBE_LEN) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t i = 0; i < PROBE_LEN; i++) {
      const double a = node2_probe_buf_[i].imag();
      const double b = y1_start[i].imag() / gain_estimate_;
      dot += a * b;  na += a * a;  nb += b * b;
    }
    if (na > 0.0 && nb > 0.0) dup_corr = static_cast<float>(dot / std::sqrt(na * nb));
  }
  const long long ms_since_echo = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      clock_t::now() - last_echo_time_).count();
  float ph = 0.0f, ioq = 0.0f, pk = 0.0f;
  link_measure(rx_accum_buf_.data() + offset, rx_accum_buf_.data() + offset, FRAME_LEN, ph, ioq, pk);
  logger_.warning("STEEP [NODE-2] DIAG: detection offset=%u this_call_samples=%u accum=%zu "
                  "ms_since_last_echo=%lld corr_with_previous_probe=%.3f",
                  offset, nof_samples, rx_accum_buf_.size(), ms_since_echo, dup_corr);
  logger_.warning("STEEP [NODE-2] LINK: gain=%.3e phase_deg=%.1f i_over_q=%.3f peak=%.3e",
                  gain_estimate_, ph, ioq, pk);
  if (std::fabs(dup_corr) > 0.9f) {
    ++dup_probes_dropped_;
    logger_.warning("STEEP [NODE-2]: repeat of the previous probe detected, NOT echoing "
                    "(dropped so far: %u)", dup_probes_dropped_);
    rx_accum_buf_.erase(rx_accum_buf_.begin(),
                         rx_accum_buf_.begin() + offset + FRAME_LEN);
    rx_node2_search_pos_ = 0u;
    return;
  }

  // ── Store gain-normalised probe body: y1_norm = y1 / G_hat ────────────────
  node2_probe_buf_.resize(PROBE_LEN);
  for (uint32_t i = 0; i < PROBE_LEN; i++) {
    const float y1_i = y1_start[i].imag() / gain_estimate_;
    node2_probe_buf_[i] = cf_t(0.0f, y1_i);
  }
  logger_.warning("STEEP [NODE-2]: preamble detected, gain=%.2e, "
                  "probe buffered (N=%u) — echo queued",
                  gain_estimate_, PROBE_LEN);

  node2_probe_id_ = echoes_sent_ + 1u;  // log label; increment happens at TX

  echo_ready_ = true;
  state_      = steep_state_t::ECHOING;

  // Consume the detected frame from the accumulator.
  rx_accum_buf_.erase(rx_accum_buf_.begin(),
                       rx_accum_buf_.begin() + offset + FRAME_LEN);
  rx_node2_search_pos_ = 0u;
}

// ── Node-2 TX ────────────────────────────────────────────────────────────────────
// Constructs and injects the AF echo frame:
//   echo frame = [preamble] [x2_body]
//   x2[i]      = C1 * y1_norm[i] + C2 * s2_bpsk[i]
//
// Secret encoding: each bit of the ASCII message maps to one BPSK sample on
// the imaginary channel.  Bit 1 → +BPSK_AMP, bit 0 → −BPSK_AMP.  The
// message is regenerated from the dynamic "Probe #NNNN from Node-2!" string,
// zero-padded to payload_size_ bytes, then bit-serialised to PROBE_LEN
// BPSK chips (one bit per sample).  If the message is shorter than PROBE_LEN
// bits, the remaining samples carry only the forwarded probe (C2 term = 0).

void steep_manager::handle_tx_node2(cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::ECHOING || !echo_ready_) return;
  if (nof_samples < FRAME_LEN) return;
  if (node2_probe_buf_.size() < PROBE_LEN) return;

  // P-STEEP secret: one random BPSK sign (+1 / -1) per probe slot.
  std::bernoulli_distribution secret_bit(0.5);
  std::vector<float> s_sign(PROBE_LEN);
  for (uint32_t i = 0; i < PROBE_LEN; i++)
    s_sign[i] = secret_bit(rng_) ? 1.0f : -1.0f;

  // Clear imaginary channel for the echo frame region.
  for (uint32_t i = 0; i < FRAME_LEN; i++)
    samples[i] = cf_t(samples[i].real(), 0.0f);

  // Preamble (same known pattern so Node-1 can detect the echo).
  generate_preamble(samples);

  // Probe body, by mode (soft probe estimate, no hard decision in either):
  //   MULTIPLICATIVE: x2[i] = C1 * s[i] * y1_norm[i]    (secret sign flips the probe)
  //   ADDITIVE:       x2[i] = C1 * y1_norm[i] + C2 * s[i]  (secret added on top of the probe)
  for (uint32_t i = 0; i < PROBE_LEN; i++) {
    const float y1_norm = node2_probe_buf_[i].imag();
    const float x2      = (mode_ == steep_mode_t::ADDITIVE)
                              ? (C1 * y1_norm + C2 * s_sign[i])
                              : (C1 * s_sign[i] * y1_norm);
    samples[PREAMBLE_LEN + i] = cf_t(samples[PREAMBLE_LEN + i].real(), x2);
  }

  if (capture_enabled_) {
    // What Node-2 received (gain-normalised probe y1_norm): used offline to pair
    // echoes with Node-1's probes and to validate the clean chain.
    write_capture_frame(node2_rx_capture_fp_, "/tmp/steep_node2_rx.iq",
                        2u, node2_probe_id_, node2_probe_buf_.data(), PROBE_LEN);
    // What Node-2 transmitted (preamble + echo body).
    write_capture_frame(echo_capture_fp_, "/tmp/steep_node2_echo.iq",
                        1u, node2_probe_id_, samples, FRAME_LEN);

    // s2 alone: preamble + secret signs, pure STEEP samples (I = 0).
    // (Amplitude is STEEP_AMP; only the sign carries the secret bit.)
    std::vector<cf_t> s2_clean(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
      s2_clean[i] = cf_t(0.0f, static_cast<float>(PREAMBLE_CHIP[i]) * STEEP_AMP);
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      s2_clean[PREAMBLE_LEN + i] = cf_t(0.0f, s_sign[i] * STEEP_AMP);
    write_capture_frame(s2_capture_fp_, "/tmp/steep_s2.iq",
                        4u, node2_probe_id_, s2_clean.data(), FRAME_LEN);
    
    // Same signal, unit amplitude (+/-1).
    std::vector<cf_t> s2_unit(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++)
      s2_unit[i] = cf_t(0.0f, static_cast<float>(PREAMBLE_CHIP[i]));
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      s2_unit[PREAMBLE_LEN + i] = cf_t(0.0f, s_sign[i]);
    write_capture_frame(s2_unit_capture_fp_, "/tmp/steep_s2_unit.iq",
                        6u, node2_probe_id_, s2_unit.data(), FRAME_LEN);

                        
    // Full I/Q version (saved only, NOT transmitted): the secret stays on Q;
    // an independent +/-A secret is generated for I.
    std::bernoulli_distribution i_bit(0.5);
    std::vector<cf_t> s2_full(FRAME_LEN), s2_full_unit(FRAME_LEN);
    for (uint32_t i = 0; i < PREAMBLE_LEN; i++) {
      const float c = static_cast<float>(PREAMBLE_CHIP[i]);
      s2_full[i]      = cf_t(c * STEEP_AMP, c * STEEP_AMP);
      s2_full_unit[i] = cf_t(c, c);
    }
    for (uint32_t i = 0; i < PROBE_LEN; i++) {
      const float iv = i_bit(rng_) ? STEEP_AMP : -STEEP_AMP;
      s2_full[PREAMBLE_LEN + i]      = cf_t(iv, s_sign[i] * STEEP_AMP);
      s2_full_unit[PREAMBLE_LEN + i] = cf_t(iv > 0.0f ? 1.0f : -1.0f, s_sign[i]);
    }
    write_capture_frame(s2_full_capture_fp_, "/tmp/steep_s2_full.iq",
                        9u, node2_probe_id_, s2_full.data(), FRAME_LEN);
    write_capture_frame(s2_full_unit_capture_fp_, "/tmp/steep_s2_full_unit.iq",
                        10u, node2_probe_id_, s2_full_unit.data(), FRAME_LEN);

    logger_.info("STEEP capture: echo #%u saved", node2_probe_id_);
  }

  echo_ready_           = false;
  echo_cooldown_active_ = true;
  last_echo_time_       = clock_t::now();
  rx_accum_buf_.clear();
  rx_node2_search_pos_    = 0u;
  ++echoes_sent_;
  state_ = steep_state_t::IDLE;

  logger_.warning("STEEP [NODE-2]: echo transmitted (mode=%s C1=%.2f gain=%.2e total_echoes=%u)",
                  mode_name(mode_), C1, gain_estimate_, echoes_sent_);
}

// ── on_tx / on_rx ─────────────────────────────────────────────────────────────

void steep_manager::on_tx(cf_t* samples, uint32_t nof_samples,
                           time_t /*full_secs*/, double /*frac_secs*/)
{
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::NODE1)
    handle_tx_node1(samples, nof_samples);
  else
    handle_tx_node2(samples, nof_samples);
}

void steep_manager::on_rx(cf_t* samples, uint32_t nof_samples,
                           time_t /*full_secs*/, double /*frac_secs*/)
{
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::NODE1)
    handle_rx_node1(samples, nof_samples);
  else
    handle_rx_node2(samples, nof_samples);
}

} // namespace srsran