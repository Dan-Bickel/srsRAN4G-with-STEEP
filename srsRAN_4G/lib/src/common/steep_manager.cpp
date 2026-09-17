#include "srsran/common/steep_manager.h"
#include <cmath>
#include <cstdio>
#include <cstring>

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

// ── Constructor / Destructor ──────────────────────────────────────────────────

steep_manager::steep_manager()
  : logger_(srslog::fetch_basic_logger("STEEP")),
    rng_(42u)
{
  logger_.set_level(srslog::basic_levels::info);
  logger_.info("STEEP manager constructed (AF analog model)");
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
  alice_probe_id_          = 0u;
  alice_gain_est_          = 1.0f;
  mse_alice_               = 0.0f;
  bob_probe_id_            = 0u;
  gain_estimate_           = 1.0f;
  rx_bob_search_pos_       = 0u;
  rx_alice_search_pos_     = 0u;
  alice_probe_buf_.clear();
  bob_probe_buf_.clear();
  rx_accum_buf_.clear();
  rx_alice_accum_buf_.clear();
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
  rx_bob_search_pos_    = 0u;
  rx_alice_search_pos_  = 0u;
  alice_probe_buf_.clear();
  bob_probe_buf_.clear();
  rx_accum_buf_.clear();
  rx_alice_accum_buf_.clear();
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
  logger_.warning("STEEP: AWGN RNG seeded with %u (%s)",
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

// ── AWGN injection (must be called under mtx_) ────────────────────────────────

void steep_manager::add_awgn(cf_t* samples, uint32_t n, float snr_db)
{
  float signal_power = 0.0f;
  for (uint32_t i = 0; i < n; i++)
    signal_power += std::norm(samples[i]);
  signal_power /= static_cast<float>(n);
  if (signal_power == 0.0f) return;

  const float noise_power = signal_power / std::pow(10.0f, snr_db / 10.0f);
  const float noise_std   = std::sqrt(noise_power / 2.0f);
  std::normal_distribution<float> dist(0.0f, noise_std);
  for (uint32_t i = 0; i < n; i++)
    samples[i] += cf_t(dist(rng_), dist(rng_));
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
}

// ── Alice TX ─────────────────────────────────────────────────────────────────
// Generates one STEEP frame per PROBE_INTERVAL_SECS after startup:
//   [PREAMBLE_LEN chips at STEEP_AMP] [PROBE_LEN random uniform samples]
// Both are injected into the imaginary (Q) channel; the real (I) LTE signal
// is untouched.  The probe body is stored in alice_probe_buf_ for Phase-2
// subtraction when the echo arrives.

void steep_manager::handle_tx_alice(cf_t* samples, uint32_t nof_samples)
{
  ++tx_call_count_;
  const auto now = clock_t::now();

  if (!timing_initialized_) {
    start_time_          = now;
    last_probe_time_     = now;
    last_heartbeat_time_ = now;
    timing_initialized_  = true;
    logger_.warning("STEEP [ALICE]: timing initialized, FRAME_LEN=%u nof_samples=%u",
                    FRAME_LEN, nof_samples);
  }

  auto elapsed_secs = [&](clock_t::time_point from) -> long long {
    return std::chrono::duration_cast<std::chrono::seconds>(now - from).count();
  };

  if (elapsed_secs(last_heartbeat_time_) >= 5) {
    last_heartbeat_time_ = now;
    logger_.warning(
        "STEEP [ALICE] heartbeat: tx_calls=%u state=%d sent=%u recovered=%u timed_out=%u",
        tx_call_count_, static_cast<int>(state_),
        probes_sent_, probes_recovered_, probes_timed_out_);
  }

  if (elapsed_secs(start_time_) < static_cast<long long>(STARTUP_DELAY_SECS)) return;

  // ── Timeout: probe outstanding too long ───────────────────────────────────
  if (state_ == steep_state_t::PROBING) {
    if (elapsed_secs(probing_since_time_) >= static_cast<long long>(PROBE_TIMEOUT_SECS)) {
      logger_.warning("STEEP [ALICE]: probe #%u timed out — returning to IDLE", alice_probe_id_);
      ++probes_timed_out_;
      state_              = steep_state_t::IDLE;
      last_probe_time_    = now;
      alice_probe_buf_.clear();
      rx_alice_accum_buf_.clear();
      rx_alice_search_pos_ = 0u;
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

  // Probe body: uniform random values in [−STEEP_AMP, +STEEP_AMP].
  // Stored as (0, v) complex so .imag() gives the scalar probe value.
  std::uniform_real_distribution<float> probe_dist(-STEEP_AMP, STEEP_AMP);
  alice_probe_buf_.resize(PROBE_LEN);
  for (uint32_t i = 0; i < PROBE_LEN; i++) {
    const float v     = probe_dist(rng_);
    alice_probe_buf_[i]         = cf_t(0.0f, v);
    samples[PREAMBLE_LEN + i]   = cf_t(samples[PREAMBLE_LEN + i].real(), v);
  }

  alice_probe_id_     = next_probe_id_++;
  state_              = steep_state_t::PROBING;
  probing_since_time_ = now;
  last_probe_time_    = now;
  rx_alice_accum_buf_.clear();
  rx_alice_search_pos_ = 0u;
  ++probes_sent_;

  logger_.warning("STEEP [ALICE]: probe #%u transmitted, N=%u samples (sent=%u recovered=%u)",
                  alice_probe_id_, PROBE_LEN, probes_sent_, probes_recovered_);

  if (capture_enabled_ && capture_frames_written_ < CAPTURE_MAX_FRAMES) {
    write_capture_frame(probe_capture_fp_, "/tmp/steep_alice_probe.iq",
                        0u, alice_probe_id_, samples, FRAME_LEN);
    ++capture_frames_written_;
  }
}

// ── Alice RX ─────────────────────────────────────────────────────────────────
// Waits for Bob's echo, then:
//   residual[i] = y2[i].imag() − C1 * x1[i].imag() * alice_gain_est_
//               ≈ C2 * s2[i] * G_BA  (noise terms omitted from this sketch)
// BPSK threshold-decode the first (payload_size_ * 8) residual samples to
// recover the secret bits.  Compute MSE_A as the mean squared deviation of
// the residual from the decoded ideal amplitude; used later for Rs.

void steep_manager::handle_rx_alice(const cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::PROBING) return;

  // Always accumulate regardless of wait status — we must not lose samples
  // that may contain the preamble.
  rx_alice_accum_buf_.insert(rx_alice_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_alice_accum_buf_.size() > RX_ACCUM_MAX) {
    const uint32_t trim =
        static_cast<uint32_t>(rx_alice_accum_buf_.size()) - RX_ACCUM_MAX;
    rx_alice_accum_buf_.erase(rx_alice_accum_buf_.begin(),
                               rx_alice_accum_buf_.begin() + trim);
    rx_alice_search_pos_ = (rx_alice_search_pos_ >= trim)
                           ? rx_alice_search_pos_ - trim : 0u;
  }

  // 20 ms anti-loopback gate: any ZMQ loopback of Alice's own probe arrives
  // in < 1 ms; Bob's legitimate echo takes at least 10–15 ms (detection +
  // TX scheduling).  20 ms clears both.
  const auto now        = clock_t::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now - probing_since_time_).count();
  if (elapsed_ms < 20) return;

  if (rx_alice_accum_buf_.size() < FRAME_LEN) return;

  uint32_t offset = 0;
  if (!detect_preamble(rx_alice_accum_buf_, rx_alice_search_pos_, offset)) {
    // Advance the resume position so we never re-scan settled territory.
    if (rx_alice_accum_buf_.size() >= FRAME_LEN)
      rx_alice_search_pos_ =
          static_cast<uint32_t>(rx_alice_accum_buf_.size()) - FRAME_LEN + 1u;
    return;
  }

  // Ensure the full probe body is already in the buffer.
  if (offset + FRAME_LEN > rx_alice_accum_buf_.size()) return;

  // ── Estimate Bob→Alice channel gain from the echo preamble ────────────────
  alice_gain_est_ = estimate_gain(rx_alice_accum_buf_.data() + offset);
  logger_.info("STEEP [ALICE]: echo preamble at offset=%u, gain=%.2e", offset, alice_gain_est_);

  const cf_t* y2 = rx_alice_accum_buf_.data() + offset + PREAMBLE_LEN;
  const uint32_t n_secret_bits = payload_size_ * 8u;

  // Residual after probe subtraction:
  //   r[i] = y2[i].imag() − C1 * x1_norm[i]
  // where x1_norm[i] = alice_probe_buf_[i].imag() (already in the same
  // amplitude domain as what was transmitted; gain cancels because Alice
  // subtracts the probe SHE sent, not what Bob received — this is the key
  // security property of the analog AF model).
  //
  // Clamp to PROBE_LEN to guard against a probe body shorter than
  // payload_size_ * 8 (should never happen with default sizes but let's
  // be defensive).
  const uint32_t decode_len = std::min(n_secret_bits, PROBE_LEN);
  if (alice_probe_buf_.size() < decode_len) {
    logger_.error("STEEP [ALICE]: alice_probe_buf_ too short (%zu < %u) — discarding echo",
                  alice_probe_buf_.size(), decode_len);
    rx_alice_accum_buf_.erase(rx_alice_accum_buf_.begin(),
                               rx_alice_accum_buf_.begin() + offset + FRAME_LEN);
    rx_alice_search_pos_ = 0u;
    return;
  }

  std::vector<uint8_t> recovered(payload_size_, 0u);
  float mse_accum = 0.0f;
  for (uint32_t i = 0; i < decode_len; i++) {
    const float rx_q   = y2[i].imag();
    const float x1_q   = alice_probe_buf_[i].imag();         // Alice's known probe
    const float resid  = rx_q - alice_gain_est_ * C1 * x1_q;
    // BPSK threshold: positive residual → bit 1, negative → bit 0.
    const int   bit_val = (resid >= 0.0f) ? 1 : 0;
    const uint32_t byte_idx = i / 8u;
    const uint32_t bit_pos  = 7u - (i % 8u);
    if (bit_val) recovered[byte_idx] |= (1u << bit_pos);

    // MSE relative to ideal BPSK amplitude at the expected C2 * BPSK_AMP level.
    const float ideal = (bit_val ? 1.0f : -1.0f) * C2 * BPSK_AMP;
    const float err   = resid - ideal;
    mse_accum += err * err;
  }
  mse_alice_ = mse_accum / static_cast<float>(decode_len);

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
      "STEEP [ALICE]: *** SECRET RECOVERED probe=#%u offset=%u | "
      "ASCII: \"%s\" | HEX: %s| MSE_A=%.4e gain=%.2e Rs=log2(MSE_E/%.4e) bits ***",
      alice_probe_id_, offset, ascii.c_str(), hex.c_str(),
      mse_alice_, alice_gain_est_, mse_alice_);
  logger_.warning(
      "STEEP [ALICE]: stats: sent=%u recovered=%u timed_out=%u success_rate=%.1f%%",
      probes_sent_, probes_recovered_, probes_timed_out_, rate);

  // ── Advance and reset ─────────────────────────────────────────────────────
  rx_alice_accum_buf_.erase(rx_alice_accum_buf_.begin(),
                             rx_alice_accum_buf_.begin() + offset + FRAME_LEN);
  rx_alice_search_pos_ = 0u;
  alice_probe_buf_.clear();
  state_ = steep_state_t::IDLE;
}

// ── Bob RX ────────────────────────────────────────────────────────────────────
// Detects Alice's preamble, estimates the Alice→Bob channel gain Ĝ, then
// stores the gain-normalised probe body:
//   y1_norm[i] = y1[i].imag() / Ĝ
// as bob_probe_buf_ (imaginary part only; stored as cf_t for uniformity).
// Transitions to ECHOING so that handle_tx_bob fires on the next TX callback.

void steep_manager::handle_rx_bob(const cf_t* samples, uint32_t nof_samples)
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
    rx_bob_search_pos_ = 0u;
    logger_.info("STEEP [BOB]: echo cooldown expired — listening resumed");
  }

  if (state_ != steep_state_t::IDLE) return;

  // Append to accumulator (using the potentially-noisy copy for detection,
  // but the original samples for gain estimation and probe buffering).
  // We keep two views: rx_accum_buf_ always holds the source-of-truth received
  // samples; noisy scanning happens against a per-call-local view.
  rx_accum_buf_.insert(rx_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_accum_buf_.size() > RX_ACCUM_MAX) {
    const uint32_t trim =
        static_cast<uint32_t>(rx_accum_buf_.size()) - RX_ACCUM_MAX;
    rx_accum_buf_.erase(rx_accum_buf_.begin(),
                         rx_accum_buf_.begin() + trim);
    rx_bob_search_pos_ = (rx_bob_search_pos_ >= trim)
                         ? rx_bob_search_pos_ - trim : 0u;
  }

  // Build the noisy accumulator for scanning.
  // Rather than maintaining a separate noisy ring buffer (complex and memory-
  // heavy), apply noise only when enabled by re-running on the clean accumulator.
  // This is affordable: phase-1 noise is a research dial, not a hot path.
  std::vector<cf_t> scan_buf;
  const std::vector<cf_t>* scan_ptr;
  if (phase1_noise_enabled_) {
    scan_buf = rx_accum_buf_;
    add_awgn(scan_buf.data(), static_cast<uint32_t>(scan_buf.size()), phase1_snr_db_);
    scan_ptr = &scan_buf;
  } else {
    scan_ptr = &rx_accum_buf_;
  }

  if (scan_ptr->size() < FRAME_LEN) return;

  uint32_t offset = 0;
  if (!detect_preamble(*scan_ptr, rx_bob_search_pos_, offset)) {
    if (scan_ptr->size() >= FRAME_LEN)
      rx_bob_search_pos_ =
          static_cast<uint32_t>(scan_ptr->size()) - FRAME_LEN + 1u;
    return;
  }

  if (offset + FRAME_LEN > rx_accum_buf_.size()) return;

  // ── Gain estimation (from the CLEAN accumulator) ──────────────────────────
  gain_estimate_ = estimate_gain(rx_accum_buf_.data() + offset);
  logger_.info("STEEP [BOB]: preamble at offset=%u, gain=%.2e", offset, gain_estimate_);

  // ── Store gain-normalised probe body ──────────────────────────────────
  // When phase-1 noise is enabled, Bob's received probe y₁ = x₁ + w₁.
  // The noise w₁ must be forwarded in the echo (AF model requirement).
  // This is what creates the irreducible MSE floor that protects Alice
  // even when Eve has a perfect Phase-1 channel — Eve receives w₁ in
  // the echo but cannot separate it from the secret.
  bob_probe_buf_.resize(PROBE_LEN);
  if (phase1_noise_enabled_) {
    // Extract probe body from clean accumulator, then add AWGN calibrated to
    // Q-channel power only. The I channel carries LTE uplink which is much
    // stronger than the STEEP Q signal — using total I²+Q² power via add_awgn
    // would flood Q with noise calibrated to LTE level, not STEEP level.
    std::vector<cf_t> noisy_probe(PROBE_LEN);
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      noisy_probe[i] = rx_accum_buf_[offset + PREAMBLE_LEN + i];

    float q_power = 0.0f;
    for (uint32_t i = 0; i < PROBE_LEN; i++)
      q_power += noisy_probe[i].imag() * noisy_probe[i].imag();
    q_power /= static_cast<float>(PROBE_LEN);

    if (q_power > 0.0f) {
      const float noise_power = q_power / std::pow(10.0f, phase1_snr_db_ / 10.0f);
      const float noise_std   = std::sqrt(noise_power);
      std::normal_distribution<float> dist_q(0.0f, noise_std);
      for (uint32_t i = 0; i < PROBE_LEN; i++)
        noisy_probe[i] = cf_t(noisy_probe[i].real(),
                              noisy_probe[i].imag() + dist_q(rng_));
    }

    for (uint32_t i = 0; i < PROBE_LEN; i++) {
      const float y1_i = noisy_probe[i].imag() / gain_estimate_;
      bob_probe_buf_[i] = cf_t(0.0f, y1_i);
    }
  } else {
    const cf_t* y1_start = rx_accum_buf_.data() + offset + PREAMBLE_LEN;
    for (uint32_t i = 0; i < PROBE_LEN; i++) {
      const float y1_i = y1_start[i].imag() / gain_estimate_;
      bob_probe_buf_[i] = cf_t(0.0f, y1_i);
    }
  }
  logger_.warning("STEEP [BOB]: preamble detected, gain=%.2e, "
                  "probe buffered (N=%u, w1_noise=%s) — echo queued",
                  gain_estimate_, PROBE_LEN,
                  phase1_noise_enabled_ ? "YES" : "no");

  bob_probe_id_ = echoes_sent_ + 1u;  // log label; increment happens at TX

  echo_ready_ = true;
  state_      = steep_state_t::ECHOING;

  // Consume the detected frame from the accumulator.
  rx_accum_buf_.erase(rx_accum_buf_.begin(),
                       rx_accum_buf_.begin() + offset + FRAME_LEN);
  rx_bob_search_pos_ = 0u;
}

// ── Bob TX ────────────────────────────────────────────────────────────────────
// Constructs and injects the AF echo frame:
//   echo frame = [preamble] [x2_body]
//   x2[i]      = C1 * y1_norm[i] + C2 * s2_bpsk[i]
//
// Secret encoding: each bit of the ASCII message maps to one BPSK sample on
// the imaginary channel.  Bit 1 → +BPSK_AMP, bit 0 → −BPSK_AMP.  The
// message is regenerated from the dynamic "Probe #NNNN from Bob!" string,
// zero-padded to payload_size_ bytes, then bit-serialised to PROBE_LEN
// BPSK chips (one bit per sample).  If the message is shorter than PROBE_LEN
// bits, the remaining samples carry only the forwarded probe (C2 term = 0).

void steep_manager::handle_tx_bob(cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::ECHOING || !echo_ready_) return;
  if (nof_samples < FRAME_LEN) return;
  if (bob_probe_buf_.size() < PROBE_LEN) return;

  // Build dynamic secret (same as the original DF implementation).
  char msg[64] = {};
  snprintf(msg, sizeof(msg), "Probe #%04u from Bob!", bob_probe_id_);
  std::vector<uint8_t> secret_bytes(payload_size_, 0u);
  const size_t msg_len = strlen(msg);
  for (size_t i = 0; i < payload_size_ && i < msg_len; i++)
    secret_bytes[i] = static_cast<uint8_t>(msg[i]);

  logger_.warning("STEEP [BOB]: encoding secret: \"%s\"", msg);

  // Serialise secret bytes to per-sample BPSK chips (MSB first).
  const uint32_t n_secret_samples = payload_size_ * 8u;   // one sample per bit
  std::vector<float> s2_bpsk(PROBE_LEN, 0.0f);
  for (uint32_t i = 0; i < n_secret_samples && i < PROBE_LEN; i++) {
    const uint32_t byte_idx = i / 8u;
    const uint32_t bit_pos  = 7u - (i % 8u);
    const int      bit      = (secret_bytes[byte_idx] >> bit_pos) & 1u;
    s2_bpsk[i] = bit ? BPSK_AMP : -BPSK_AMP;
  }

  // Clear imaginary channel for the echo frame region.
  for (uint32_t i = 0; i < FRAME_LEN; i++)
    samples[i] = cf_t(samples[i].real(), 0.0f);

  // Preamble (Bob uses the same known pattern so Alice can detect the echo).
  generate_preamble(samples);

  // Probe body: x2[i] = C1 * y1_norm[i] + C2 * s2[i]
  for (uint32_t i = 0; i < PROBE_LEN; i++) {
    const float y1_norm = bob_probe_buf_[i].imag();
    const float x2      = C1 * y1_norm + C2 * s2_bpsk[i];
    samples[PREAMBLE_LEN + i] = cf_t(samples[PREAMBLE_LEN + i].real(), x2);
  }

  if (capture_enabled_) {
    write_capture_frame(echo_capture_fp_, "/tmp/steep_bob_echo.iq",
                        1u, bob_probe_id_, samples, FRAME_LEN);
    logger_.info("STEEP capture: echo #%u saved", bob_probe_id_);
  }

  echo_ready_          = false;
  echo_cooldown_active_ = true;
  last_echo_time_      = clock_t::now();
  rx_accum_buf_.clear();
  rx_bob_search_pos_   = 0u;
  ++echoes_sent_;
  state_ = steep_state_t::IDLE;

  logger_.warning("STEEP [BOB]: echo transmitted (C1=%.2f C2=%.2f gain=%.2e total_echoes=%u)",
                  C1, C2, gain_estimate_, echoes_sent_);
}

// ── on_tx / on_rx ─────────────────────────────────────────────────────────────

void steep_manager::on_tx(cf_t* samples, uint32_t nof_samples,
                           time_t /*full_secs*/, double /*frac_secs*/)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::ALICE)
    handle_tx_alice(samples, nof_samples);
  else
    handle_tx_bob(samples, nof_samples);
}

void steep_manager::on_rx(cf_t* samples, uint32_t nof_samples,
                           time_t /*full_secs*/, double /*frac_secs*/)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::ALICE)
    handle_rx_alice(samples, nof_samples);
  else
    handle_rx_bob(samples, nof_samples);
}

} // namespace srsran