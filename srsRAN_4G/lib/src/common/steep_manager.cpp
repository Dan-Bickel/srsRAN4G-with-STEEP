#include "srsran/common/steep_manager.h"
#include <cstdio>
#include <cstdlib>

namespace srsran {

steep_manager::steep_manager() : logger_(srslog::fetch_basic_logger("STEEP"))
{
  srand(42); // fixed seed for reproducible PoC; replace with proper RNG later
  logger_.set_level(srslog::basic_levels::info);
  logger_.info("STEEP manager constructed");
}

bool steep_manager::init(const std::array<uint8_t, 32>& key)
{
  std::lock_guard<std::mutex> lock(mtx_);
  session_key_    = key;
  state_          = steep_state_t::IDLE;
  echo_ready_     = false;
  handshake_done_ = false;
  next_probe_id_  = 1;
  probe_buffer_.clear();
  logger_.info("STEEP manager initialized");
  return true;
}

void steep_manager::reset()
{
  std::lock_guard<std::mutex> lock(mtx_);
  state_          = steep_state_t::IDLE;
  echo_ready_     = false;
  handshake_done_ = false;
  probe_buffer_.clear();
  logger_.warning("STEEP manager reset to IDLE");
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

// ---- Public probe buffer (lock themselves) ----

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

// ---- Private locked versions (called from on_rx/on_tx which already hold lock) ----

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

// ---- Encoding: write bytes into imaginary component of IQ samples ----
// Format: [MAGIC 4B][PROBE_ID 4B][PAYLOAD NB] — each byte = 8 samples
// Bit 1 = +STEEP_AMP on imaginary axis, Bit 0 = -STEEP_AMP

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

  uint32_t idx = 0;
  for (uint8_t byte : frame) {

    for (int bit = 7; bit >= 0; bit--) {
      float v = ((byte >> bit) & 1) ? STEEP_AMP : -STEEP_AMP;

      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++) {  // spread across SAMPLES_PER_BIT
        samples[idx++] += cf_t(0.0f, v);
      }
    }

    /* OLD - updated for resampling
    for (int bit = 7; bit >= 0; bit--) {
      float v    = ((byte >> bit) & 1) ? STEEP_AMP : -STEEP_AMP;
      samples[idx] += cf_t(0.0f, v);
      idx++;
    }
    */
  }
}

bool steep_manager::decode_frame(const cf_t* samples, uint32_t nof_samples,
                                  uint32_t& probe_id_out,
                                  std::vector<uint8_t>& payload_out)
{
  if (nof_samples < frame_size_samples()) {

    logger_.info("STEEP decode: not enough samples %u < %u", 
                 nof_samples, frame_size_samples());

    return false;}

  uint32_t total_bytes = 8 + payload_size_;
  std::vector<uint8_t> frame(total_bytes);
  uint32_t idx = 0;
  for (uint32_t b = 0; b < total_bytes; b++) {
    frame[b] = 0;

    for (int bit = 7; bit >= 0; bit--) {
      int votes = 0;
      for (uint32_t s = 0; s < SAMPLES_PER_BIT; s++) {
        votes += (samples[idx++].imag() > 0.0f) ? 1 : -1;
      }
      if (votes > 0) frame[b] |= (1 << bit);
    }

    /* OLD - updated for resampling
    for (int bit = 7; bit >= 0; bit--) {
      if (samples[idx].imag() > 0.0f)
        frame[b] |= (1 << bit);
      idx++;
    }
    */
  }

  uint32_t magic = ((uint32_t)frame[0] << 24) | ((uint32_t)frame[1] << 16) |
                   ((uint32_t)frame[2] <<  8) | ((uint32_t)frame[3] <<  0);

  if (magic != STEEP_MAGIC){
    //logger_.info("STEEP decode: magic mismatch got=0x%08x expected=0x%08x", 
    //             magic, STEEP_MAGIC);
    return false;
  }

  probe_id_out = ((uint32_t)frame[4] << 24) | ((uint32_t)frame[5] << 16) |
                 ((uint32_t)frame[6] <<  8) | ((uint32_t)frame[7] <<  0);
  payload_out.assign(frame.begin() + 8, frame.end());
  return true;
}

// ---- Alice TX: generate and inject probe once ----

void steep_manager::handle_tx_alice(cf_t* samples, uint32_t nof_samples)
{
  if (handshake_done_)               return; // one-shot: call reset() to try again
  if (state_ != steep_state_t::IDLE) return;
  if (nof_samples < frame_size_samples()) return;

  steep_probe_t probe;
  probe.probe_id = next_probe_id_++;
  probe.payload.resize(payload_size_);
  for (auto& b : probe.payload) b = rand() & 0xFF;

  encode_frame(samples, probe.probe_id, probe.payload);
  logger_.info("STEEP [ALICE]: sample[0] imag after encode = %f", samples[0].imag());
  store_probe_locked(probe);
  state_ = steep_state_t::PROBING;
  logger_.info("STEEP [ALICE]: probe injected, id=%u", probe.probe_id);
}

// ---- Alice RX: detect echo, XOR out original, recover secret ----

void steep_manager::handle_rx_alice(const cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::PROBING) return;

  uint32_t probe_id = 0;
  std::vector<uint8_t> payload;
  if (!decode_frame(samples, nof_samples, probe_id, payload)) return;

  steep_probe_t original;
  if (!consume_probe_locked(probe_id, original)) {
    logger_.warning("STEEP [ALICE]: echo id=%u not in probe buffer", probe_id);
    return;
  }

  // XOR out Alice's original random payload to get Bob's secret
  std::vector<uint8_t> recovered(payload_size_);
  for (uint32_t i = 0; i < payload_size_; i++)
    recovered[i] = payload[i] ^ original.payload[i];

  // Log the recovered secret as hex
  std::string hex;
  char buf[4];
  for (auto b : recovered) {
    snprintf(buf, sizeof(buf), "%02x ", b);
    hex += buf;
  }
  logger_.info("STEEP [ALICE]: *** SECRET RECOVERED: %s ***", hex.c_str());

  state_          = steep_state_t::IDLE;
  handshake_done_ = true; // stop here; call reset() to run again
}

// ---- Bob RX: detect probe, mix secret, queue echo ----

void steep_manager::handle_rx_bob(const cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::IDLE) return;

  uint32_t frame_size = frame_size_samples();
  if (nof_samples < frame_size) return;

  float max_real = 0, max_imag = 0;
  for (uint32_t i = 0; i < std::min(nof_samples, 1000u); i++) {
    max_real = std::max(max_real, std::abs(samples[i].real()));
    max_imag = std::max(max_imag, std::abs(samples[i].imag()));
  }
  logger_.info("STEEP [BOB]: max real=%f max imag=%f", max_real, max_imag);

  // Slide through the buffer one bit-width at a time looking for the magic header
  for (uint32_t offset = 0; offset + frame_size <= nof_samples; offset += SAMPLES_PER_BIT) {
    uint32_t             probe_id = 0;
    std::vector<uint8_t> payload;
    if (!decode_frame(samples + offset, nof_samples - offset, probe_id, payload)) {
      continue;
    }

    // Found a valid probe
    steep_probe_t probe;
    probe.probe_id = probe_id;
    probe.payload  = payload;

    if (secret_.size() == payload_size_) {
      mix_secret_locked(probe, secret_);
    } else {
      logger_.warning("STEEP [BOB]: no secret set or wrong size — echoing unmodified");
    }

    pending_echo_ = probe;
    echo_ready_   = true;
    state_        = steep_state_t::ECHOING;
    logger_.info("STEEP [BOB]: probe detected id=%u at offset=%u, echo queued",
                 probe_id, offset);
    return;
  }
}

// ---- Bob TX: inject echo ----

void steep_manager::handle_tx_bob(cf_t* samples, uint32_t nof_samples)
{
  if (state_ != steep_state_t::ECHOING || !echo_ready_) return;
  if (nof_samples < frame_size_samples()) return;

  encode_frame(samples, pending_echo_.probe_id, pending_echo_.payload);
  echo_ready_ = false;
  state_      = steep_state_t::IDLE;
  logger_.info("STEEP [BOB]: echo transmitted, id=%u", pending_echo_.probe_id);
}

// ---- Main hooks called by radio class ----

void steep_manager::on_rx(cf_t* samples, uint32_t nof_samples,
                           time_t full_secs, double frac_secs)
{
  std::lock_guard<std::mutex> lock(mtx_);
  //logger_.info("STEEP RX hook fired: %u samples", nof_samples); // was debug
  logger_.debug("STEEP RX: %u samples at %lld + %.9f s",
                nof_samples, (long long)full_secs, frac_secs);
  if (role_ == steep_role_t::ALICE)
    handle_rx_alice(samples, nof_samples);
  else
    handle_rx_bob(samples, nof_samples);
}

void steep_manager::on_tx(cf_t* samples, uint32_t nof_samples,
                           time_t full_secs, double frac_secs)
{
  std::lock_guard<std::mutex> lock(mtx_);
  logger_.debug("STEEP TX: %u samples at %lld + %.9f s",
                nof_samples, (long long)full_secs, frac_secs);
  if (role_ == steep_role_t::ALICE)
    handle_tx_alice(samples, nof_samples);
  else
    handle_tx_bob(samples, nof_samples);
}

} // namespace srsran