#include "srsran/common/steep_manager.h"

namespace srsran {

steep_manager::steep_manager() :
  logger_(srslog::fetch_basic_logger("STEEP"))
{
  logger_.info("STEEP manager constructed");
}

bool steep_manager::init(const std::array<uint8_t, 32>& session_key)
{
  std::lock_guard<std::mutex> lock(mtx_);
  session_key_ = session_key;
  state_       = steep_state_t::IDLE;
  probe_buffer_.clear();
  logger_.info("STEEP manager initialized");
  return true;
}

void steep_manager::reset()
{
  std::lock_guard<std::mutex> lock(mtx_);
  state_ = steep_state_t::IDLE;
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
  // Alice:  IDLE -> PROBING -> DATA -> IDLE
  // Bob:    IDLE -> ECHOING -> IDLE
  // Anyone: any state -> IDLE (error recovery)
  switch (from) {
    case steep_state_t::IDLE:
      return to == steep_state_t::PROBING ||
             to == steep_state_t::ECHOING;
    case steep_state_t::PROBING:
      return to == steep_state_t::DATA ||
             to == steep_state_t::IDLE;
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
  logger_.info("STEEP state transitioned");
  return true;
}

bool steep_manager::store_probe(const steep_probe_t& probe)
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (const auto& p : probe_buffer_) {
    if (p.probe_id == probe.probe_id) {
      logger_.warning("Rejected duplicate probe_id — probe reuse prevented");
      return false;
    }
  }
  probe_buffer_.push_back(probe);
  logger_.info("Probe stored, id=%u", probe.probe_id);
  return true;
}

bool steep_manager::consume_probe(uint32_t probe_id, steep_probe_t& out)
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto& p : probe_buffer_) {
    if (p.probe_id == probe_id && !p.consumed) {
      p.consumed = true;
      out        = p;
      logger_.info("Probe consumed, id=%u", probe_id);
      return true;
    }
  }
  logger_.warning("Probe id=%u not found or already consumed", probe_id);
  return false;
}

bool steep_manager::has_pending_probes() const
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (const auto& p : probe_buffer_) {
    if (!p.consumed) return true;
  }
  return false;
}

bool steep_manager::mix_secret(steep_probe_t&              probe,
                               const std::vector<uint8_t>& secret)
{
  if (secret.size() != probe.payload.size()) {
    logger_.error("mix_secret: secret size %zu != probe payload size %zu",
                  secret.size(), probe.payload.size());
    return false;
  }
  // XOR mixes Bob's secret into the probe payload byte by byte.
  // Alice can reverse this later because she knows the original payload.
  for (size_t i = 0; i < probe.payload.size(); ++i) {
    probe.payload[i] ^= secret[i];
  }
  logger_.info("Secret mixed into probe id=%u", probe.probe_id);
  return true;
}

void steep_manager::on_rx(uint32_t nof_samples, time_t full_secs, double frac_secs)
{
  // printf(">>> STEEP RX hook fired: %u samples\n", nof_samples);
  // fflush(stdout);
  logger_.info("STEEP RX: %u samples at %lld + %.9f s",
               nof_samples, (long long)full_secs, frac_secs);
}

void steep_manager::on_tx(uint32_t nof_samples, time_t full_secs, double frac_secs)
{
  // printf(">>> STEEP TX hook fired: %u samples\n", nof_samples);
  // fflush(stdout);
  logger_.info("STEEP TX: %u samples at %lld + %.9f s",
               nof_samples, (long long)full_secs, frac_secs);
}

/* OLD implementation
void steep_manager::on_rx(uint32_t nof_samples, const srsran_timestamp_t& ts)
{
  // Week 3: observation only - log that samples arrived, touch nothing
  logger_.debug("STEEP RX: %u samples at %lld + %.9f s",
                nof_samples,
                (long long)ts.full_secs,
                ts.frac_secs);
}

void steep_manager::on_tx(uint32_t nof_samples, const srsran_timestamp_t& ts)
{
  // Week 3: observation only - log that samples are going out, touch nothing
  logger_.debug("STEEP TX: %u samples at %lld + %.9f s",
                nof_samples,
                (long long)ts.full_secs,
                ts.frac_secs);
}
*/

} // namespace srsran