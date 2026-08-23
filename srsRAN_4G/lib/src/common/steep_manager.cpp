#include "srsran/common/steep_manager.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace srsran {

const int8_t steep_manager::STEEP_CHIP[32] = {
  +1,-1,+1,-1,+1,+1,-1,-1,+1,+1,+1,-1,+1,-1,-1,+1,
  -1,+1,-1,-1,-1,+1,+1,-1,-1,-1,-1,+1,-1,+1,+1,-1
};

steep_manager::steep_manager() : logger_(srslog::fetch_basic_logger("STEEP"))
{
  srand(42);
  logger_.set_level(srslog::basic_levels::info);
  logger_.info("STEEP manager constructed");
}

bool steep_manager::init(const std::array<uint8_t, 32>& key)
{
  std::lock_guard<std::mutex> lock(mtx_);
  session_key_        = key;
  state_              = steep_state_t::IDLE;
  echo_ready_         = false;
  next_probe_id_      = 1;
  tx_call_count_      = 0;
  timing_initialized_ = false;
  probes_sent_        = 0;
  probes_recovered_   = 0;
  echoes_sent_        = 0;
  probes_timed_out_   = 0;
  echoed_probe_ids_.clear();
  rx_accum_buf_.clear();
  rx_alice_accum_buf_.clear();
  probe_buffer_.clear();
  logger_.info("STEEP manager initialized");
  return true;
}

void steep_manager::reset()
{
  std::lock_guard<std::mutex> lock(mtx_);
  state_              = steep_state_t::IDLE;
  echo_ready_         = false;
  tx_call_count_      = 0;
  timing_initialized_ = false;
  echoed_probe_ids_.clear();
  rx_accum_buf_.clear();
  rx_alice_accum_buf_.clear();
  probe_buffer_.clear();
  logger_.warning("STEEP manager reset — final stats: sent=%u recovered=%u echoed=%u timed_out=%u",
                  probes_sent_, probes_recovered_, echoes_sent_, probes_timed_out_);
}

steep_state_t steep_manager::get_state() const
{
  std::lock_guard<std::mutex> lock(mtx_);
  return state_;
}

bool steep_manager::valid_transition(steep_state_t from, steep_state_t to) const
{
  switch (from) {
    case steep_state_t::IDLE:
      return to == steep_state_t::PROBING || to == steep_state_t::ECHOING;
    case steep_state_t::PROBING:
      return to == steep_state_t::DATA || to == steep_state_t::IDLE;
    case steep_state_t::ECHOING:
      return to == steep_state_t::IDLE;
    case steep_state_t::DATA:
      return to == steep_state_t::IDLE;
    default:
      return false;
  }
}

bool steep_manager::transition(steep_state_t next)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (!valid_transition(state_, next)) {
    logger_.error("Invalid STEEP state transition attempted");
    return false;
  }
  state_ = next;
  return true;
}

bool steep_manager::store_probe(const steep_probe_t& probe)
{
  std::lock_guard<std::mutex> lock(mtx_);
  return store_probe_locked(probe);
}

bool steep_manager::consume_probe(uint32_t probe_id, steep_probe_t& out)
{
  std::lock_guard<std::mutex> lock(mtx_);
  return consume_probe_locked(probe_id, out);
}

bool steep_manager::has_pending_probes() const
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (const auto& p : probe_buffer_)
    if (!p.consumed) return true;
  return false;
}

bool steep_manager::mix_secret(steep_probe_t& probe, const std::vector<uint8_t>& secret)
{
  std::lock_guard<std::mutex> lock(mtx_);
  return mix_secret_locked(probe, secret);
}

bool steep_manager::store_probe_locked(const steep_probe_t& probe)
{
  for (const auto& p : probe_buffer_) {
    if (p.probe_id == probe.probe_id) {
      logger_.warning("Rejected duplicate probe id=%u", probe.probe_id);
      return false;
    }
  }
  probe_buffer_.push_back(probe);
  return true;
}

bool steep_manager::consume_probe_locked(uint32_t probe_id, steep_probe_t& out)
{
  for (auto& p : probe_buffer_) {
    if (p.probe_id == probe_id && !p.consumed) {
      p.consumed = true;
      out        = p;
      return true;
    }
  }
  return false;
}

bool steep_manager::mix_secret_locked(steep_probe_t& probe,
                                       const std::vector<uint8_t>& secret)
{
  if (secret.size() != probe.payload.size()) {
    logger_.error("mix_secret: size mismatch %zu vs %zu",
                  secret.size(), probe.payload.size());
    return false;
  }
  for (size_t i = 0; i < probe.payload.size(); i++)
    probe.payload[i] ^= secret[i];
  return true;
}

uint32_t steep_manager::compute_crc32(const uint8_t* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int j = 0; j < 8; j++)
      crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1u));
  }
  return crc ^ 0xFFFFFFFFu;
}

void steep_manager::encode_frame(cf_t* samples, uint32_t probe_id,
                                  const std::vector<uint8_t>& payload)
{
  std::vector<uint8_t> frame(8 + payload.size());
  frame[0] = (STEEP_MAGIC >> 24) & 0xFF;
  frame[1] = (STEEP_MAGIC >> 16) & 0xFF;
  frame[2] = (STEEP_MAGIC >>  8) & 0xFF;
  frame[3] = (STEEP_MAGIC >>  0) & 0xFF;
  frame[4] = (probe_id >> 24) & 0xFF;
  frame[5] = (probe_id >> 16) & 0xFF;
  frame[6] = (probe_id >>  8) & 0xFF;
  frame[7] = (probe_id >>  0) & 0xFF;
  std::copy(payload.begin(), payload.end(), frame.begin() + 8);

  uint32_t crc = compute_crc32(frame.data(), frame.size());
  frame.push_back((crc >> 24) & 0xFF);
  frame.push_back((crc >> 16) & 0xFF);
  frame.push_back((crc >>  8) & 0xFF);
  frame.push_back((crc >>  0) & 0xFF);

  uint32_t idx = 0;
  for (uint8_t byte : frame) {
    for (int bit = 7; bit >= 0; bit--) {
      float polarity = ((byte >> bit) & 1) ? 1.0f : -1.0f;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++)
        samples[idx++] += cf_t(0.0f, polarity * STEEP_AMP * static_cast<float>(STEEP_CHIP[s]));
    }
  }
}

bool steep_manager::decode_frame(const cf_t* samples, uint32_t nof_samples,
                                  uint32_t& probe_id_out,
                                  std::vector<uint8_t>& payload_out)
{
  if (nof_samples < frame_size_samples()) return false;

  uint32_t idx = 0;

  // ---- Magic detection (cheap early rejection) ----
  uint32_t magic = 0;
  for (int b = 0; b < 4; b++) {
    uint8_t decoded = 0;                          // renamed from 'byte' to avoid C++17 conflict
    for (int bit = 7; bit >= 0; bit--) {
      float corr = 0.0f;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++)
        corr += samples[idx++].imag() * static_cast<float>(STEEP_CHIP[s]);
      if (corr > 0.0f) decoded |= (1u << bit);
    }
    magic = (magic << 8) | decoded;
  }
  if (magic != STEEP_MAGIC) return false;

  // ---- Full frame decode ----
  uint32_t total_bytes = 8u + payload_size_ + CRC_SIZE_BYTES;
  std::vector<uint8_t> frame(total_bytes, 0u);

  // Magic is confirmed; fill frame[0..3] with the known magic bytes for CRC.
  frame[0] = (STEEP_MAGIC >> 24) & 0xFFu;
  frame[1] = (STEEP_MAGIC >> 16) & 0xFFu;
  frame[2] = (STEEP_MAGIC >>  8) & 0xFFu;
  frame[3] = (STEEP_MAGIC >>  0) & 0xFFu;

  // Decode probe_id, payload, and CRC (bytes 4 .. total_bytes-1).
  for (uint32_t b = 4; b < total_bytes; b++) {
    for (int bit = 7; bit >= 0; bit--) {
      float corr = 0.0f;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++)
        corr += samples[idx++].imag() * static_cast<float>(STEEP_CHIP[s]);
      if (corr > 0.0f) frame[b] |= (1u << bit);
    }
  }

  // ---- CRC check ----
  size_t   data_len     = 8u + payload_size_;
  uint32_t expected_crc = compute_crc32(frame.data(), data_len);
  uint32_t decoded_crc  = ((uint32_t)frame[data_len + 0] << 24) |
                          ((uint32_t)frame[data_len + 1] << 16) |
                          ((uint32_t)frame[data_len + 2] <<  8) |
                          ((uint32_t)frame[data_len + 3] <<  0);
  if (expected_crc != decoded_crc) {
    static uint32_t crc_miss_count = 0;
    ++crc_miss_count;
    if (crc_miss_count == 1 || crc_miss_count % 1000 == 0)
      logger_.warning("STEEP decode: CRC mismatch #%u (expected %08x got %08x)",
                      crc_miss_count, expected_crc, decoded_crc);
    return false;
  }

  // ---- Extract results ----
  uint32_t probe_id = ((uint32_t)frame[4] << 24) | ((uint32_t)frame[5] << 16) |
                      ((uint32_t)frame[6] <<  8) | ((uint32_t)frame[7] <<  0);
  if (probe_id == 0) {
    logger_.warning("STEEP decode: probe_id=0 after CRC pass - discarding");
    return false;
  }

  probe_id_out = probe_id;
  payload_out.assign(frame.begin() + 8, frame.begin() + 8 + payload_size_);
  return true;
}

void steep_manager::handle_tx_alice(cf_t* samples, uint32_t nof_samples)
{
  ++tx_call_count_;

  auto now = clock_t::now();
  if (!timing_initialized_) {
    start_time_         = now;
    last_probe_time_    = now;
    last_heartbeat_time_ = now;
    timing_initialized_ = true;
    logger_.warning("STEEP [ALICE]: timing initialized, nof_samples=%u frame_size=%u",
                    nof_samples, frame_size_samples());
  }

  auto elapsed_secs = [&](clock_t::time_point from) {
    return std::chrono::duration_cast<std::chrono::seconds>(now - from).count();
  };

  if (elapsed_secs(last_heartbeat_time_) >= 5) {
    last_heartbeat_time_ = now;
    logger_.warning("STEEP [ALICE] heartbeat: tx_calls=%u state=%d sent=%u recovered=%u timed_out=%u",
                    tx_call_count_, (int)state_,
                    probes_sent_, probes_recovered_, probes_timed_out_);
  }

  if (elapsed_secs(start_time_) < (long long)STARTUP_DELAY_SECS) return;

  if (state_ == steep_state_t::PROBING) {
    if (elapsed_secs(probing_since_time_) >= (long long)PROBE_TIMEOUT_SECS) {
      logger_.warning("STEEP [ALICE]: probe id=%u timed out after %lld s — returning to IDLE",
                      next_probe_id_ - 1, elapsed_secs(probing_since_time_));
      ++probes_timed_out_;
      state_           = steep_state_t::IDLE;
      last_probe_time_ = now;
    }
    return;
  }

  if (state_ != steep_state_t::IDLE)                                   return;
  if (elapsed_secs(last_probe_time_) < (long long)PROBE_INTERVAL_SECS) return;
  if (nof_samples < frame_size_samples())                               return;

  steep_probe_t probe;
  probe.probe_id = next_probe_id_++;
  probe.payload.resize(payload_size_);
  for (auto& b : probe.payload) b = rand() & 0xFF;

  // NEW
  // Zero the imaginary (STEEP) channel before encoding to prevent accumulation.
  // The LTE imaginary component is negligible at this hook point (confirmed by TX verify).
  for (uint32_t i = 0; i < frame_size_samples(); i++)
    samples[i] = cf_t(samples[i].real(), 0.0f);

  encode_frame(samples, probe.probe_id, probe.payload);

  /* Diagnostic tool
  {
  float max_imag = 0.0f;
  float min_imag = 0.0f;
  for (uint32_t i = 0; i < frame_size_samples(); i++) {
    max_imag = std::max(max_imag, samples[i].imag());
    min_imag = std::min(min_imag, samples[i].imag());
  }
  logger_.warning("STEEP [ALICE]: TX verify id=%u max_imag=%.1f min_imag=%.1f (expected ±%.1f)",
                  probe.probe_id, max_imag, min_imag, STEEP_AMP);
  }
  */

  store_probe_locked(probe);
  state_              = steep_state_t::PROBING;
  probing_since_time_ = now;
  last_probe_time_    = now;
  ++probes_sent_;
  logger_.warning("STEEP [ALICE]: probe injected id=%u (sent=%u recovered=%u timed_out=%u)",
                  probe.probe_id, probes_sent_, probes_recovered_, probes_timed_out_);
}

void steep_manager::handle_rx_alice(const cf_t* samples, uint32_t nof_samples)
{
  // Diagnostic: confirm RX is running and check for STEEP signal energy
  static uint32_t alice_rx_count = 0;
  ++alice_rx_count;
  if (alice_rx_count == 1 || alice_rx_count % 500 == 0) {
    float max_imag = 0.0f;
    for (uint32_t i = 0; i < std::min(nof_samples, 1000u); i++)
      max_imag = std::max(max_imag, std::abs(samples[i].imag()));
    logger_.warning("STEEP [ALICE] RX DIAG: call #%u state=%d accum=%zu nof_samples=%u max_imag=%.4f (need>%.4f)",
                    alice_rx_count, (int)state_,
                    rx_alice_accum_buf_.size(), nof_samples,
                    max_imag, STEEP_AMP * 0.5f);
  }

  if (state_ != steep_state_t::PROBING) return;

  rx_alice_accum_buf_.insert(rx_alice_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_alice_accum_buf_.size() > RX_ACCUM_MAX) {
    rx_alice_accum_buf_.erase(rx_alice_accum_buf_.begin(),
                              rx_alice_accum_buf_.begin() +
                              (rx_alice_accum_buf_.size() - RX_ACCUM_MAX));
  }

  uint32_t frame_size = frame_size_samples();
  if (rx_alice_accum_buf_.size() < frame_size) return;

  for (uint32_t offset = 0; offset + frame_size <= rx_alice_accum_buf_.size();
       offset += SAMPLES_PER_BIT) {

    uint32_t probe_id = 0;
    std::vector<uint8_t> payload;
    if (!decode_frame(rx_alice_accum_buf_.data() + offset,
                      rx_alice_accum_buf_.size() - offset,
                      probe_id, payload))
      continue;

    steep_probe_t original;
    if (!consume_probe_locked(probe_id, original)) {
      logger_.warning("STEEP [ALICE]: echo id=%u not in probe buffer — ignoring (loopback?)",
                      probe_id);
      rx_alice_accum_buf_.erase(rx_alice_accum_buf_.begin(),
                                rx_alice_accum_buf_.begin() + offset + frame_size);
      return;
    }

    std::vector<uint8_t> recovered(payload_size_);
    for (uint32_t i = 0; i < payload_size_; i++)
      recovered[i] = payload[i] ^ original.payload[i];

    std::string hex;
    char buf[4];
    for (auto b : recovered) { snprintf(buf, sizeof(buf), "%02x ", b); hex += buf; }

    ++probes_recovered_;
    float rate = probes_sent_ > 0 ? (100.0f * probes_recovered_ / probes_sent_) : 0.0f;

    logger_.warning("STEEP [ALICE]: *** SECRET RECOVERED id=%u offset=%u accum=%zu: %s***",
                    probe_id, offset, rx_alice_accum_buf_.size(), hex.c_str());
    logger_.warning("STEEP [ALICE]: stats: sent=%u recovered=%u timed_out=%u success_rate=%.1f%%",
                    probes_sent_, probes_recovered_, probes_timed_out_, rate);

    state_ = steep_state_t::IDLE;
    rx_alice_accum_buf_.clear();
    prune_probe_buffer_locked();
    return;
  }
}

void steep_manager::handle_rx_bob(const cf_t* samples, uint32_t nof_samples)
{
  static uint32_t bob_rx_count = 0;
  ++bob_rx_count;
  if (bob_rx_count == 1 || bob_rx_count % 500 == 0) {
    logger_.warning("STEEP [BOB] DIAGNOSTIC: on_rx call #%u nof_samples=%u accum=%zu state=%d echoes=%u",
                    bob_rx_count, nof_samples, rx_accum_buf_.size(), (int)state_, echoes_sent_);
  }

  if (state_ != steep_state_t::IDLE) return;

  rx_accum_buf_.insert(rx_accum_buf_.end(), samples, samples + nof_samples);
  if (rx_accum_buf_.size() > RX_ACCUM_MAX) {
    rx_accum_buf_.erase(rx_accum_buf_.begin(),
                        rx_accum_buf_.begin() + (rx_accum_buf_.size() - RX_ACCUM_MAX));
  }

  uint32_t frame_size = frame_size_samples();
  if (rx_accum_buf_.size() < frame_size) return;

  float max_imag = 0;
  for (uint32_t i = 0; i < std::min((uint32_t)rx_accum_buf_.size(), 1000u); i++)
    max_imag = std::max(max_imag, std::abs(rx_accum_buf_[i].imag()));
  logger_.debug("STEEP [BOB]: accum=%zu max_imag=%f", rx_accum_buf_.size(), max_imag);

  for (uint32_t offset = 0; offset + frame_size <= rx_accum_buf_.size();
       offset += SAMPLES_PER_BIT) {

    uint32_t probe_id = 0;
    std::vector<uint8_t> payload;
    if (!decode_frame(rx_accum_buf_.data() + offset,
                      rx_accum_buf_.size() - offset,
                      probe_id, payload))
      continue;

    if (echoed_probe_ids_.count(probe_id)) {
      logger_.debug("STEEP [BOB]: id=%u already echoed — ignoring (loopback?)", probe_id);
      rx_accum_buf_.erase(rx_accum_buf_.begin(),
                          rx_accum_buf_.begin() + offset + frame_size);
      return;
    }

    steep_probe_t probe;
    probe.probe_id = probe_id;
    probe.payload  = payload;

    if (secret_.size() == payload_size_)
      mix_secret_locked(probe, secret_);
    else
      logger_.warning("STEEP [BOB]: no secret set or wrong size — echoing unmodified");

    pending_echo_ = probe;
    echo_ready_   = true;
    state_        = steep_state_t::ECHOING;

    logger_.info("STEEP [BOB]: probe detected id=%u offset=%u accum=%zu echo queued (echoed=%u)",
                 probe_id, offset, rx_accum_buf_.size(), echoes_sent_);

    rx_accum_buf_.erase(rx_accum_buf_.begin(),
                        rx_accum_buf_.begin() + offset + frame_size);
    return;
  }

  /* Diagnostic tool
  if (bob_rx_count % 500 == 0 && rx_accum_buf_.size() >= frame_size) {
    float max_imag = 0.0f;
    for (const auto& s : rx_accum_buf_)
      max_imag = std::max(max_imag, std::abs(s.imag()));

    logger_.warning("STEEP [BOB]: scan complete, no probe found. accum=%zu max_imag=%.1f need>%.1f",
                    rx_accum_buf_.size(), max_imag, STEEP_AMP * 0.5f);
  }
  */
}

void steep_manager::handle_tx_bob(cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::ECHOING || !echo_ready_) return;
  if (nof_samples < frame_size_samples()) return;

  encode_frame(samples, pending_echo_.probe_id, pending_echo_.payload);
  echo_ready_ = false;
  echoed_probe_ids_.insert(pending_echo_.probe_id);
  rx_accum_buf_.clear();
  ++echoes_sent_;
  state_ = steep_state_t::IDLE;
  logger_.info("STEEP [BOB]: echo transmitted id=%u (total echoes=%u)",
               pending_echo_.probe_id, echoes_sent_);
}

void steep_manager::on_tx(cf_t* samples, uint32_t nof_samples,
                           time_t full_secs, double frac_secs)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::ALICE)
    handle_tx_alice(samples, nof_samples);
  else
    handle_tx_bob(samples, nof_samples);
}

void steep_manager::on_rx(cf_t* samples, uint32_t nof_samples,
                           time_t full_secs, double frac_secs)
{
  std::lock_guard<std::mutex> lock(mtx_);
  if (role_ == steep_role_t::ALICE)
    handle_rx_alice(samples, nof_samples);
  else
    handle_rx_bob(samples, nof_samples);
}

void steep_manager::prune_probe_buffer_locked()
{
  auto before = probe_buffer_.size();
  probe_buffer_.erase(
    std::remove_if(probe_buffer_.begin(), probe_buffer_.end(),
                   [](const steep_probe_t& p) { return p.consumed; }),
    probe_buffer_.end()
  );
  logger_.debug("STEEP: probe buffer pruned %zu → %zu entries", before, probe_buffer_.size());
}

} // namespace srsran