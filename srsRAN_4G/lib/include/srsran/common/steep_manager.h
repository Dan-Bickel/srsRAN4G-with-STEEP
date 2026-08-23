#ifndef SRSRAN_STEEP_MANAGER_H
#define SRSRAN_STEEP_MANAGER_H

#include <array>
#include <complex>
#include <cstdint>
#include <mutex>
#include <set>
#include <chrono>
#include <vector>
#include "srsran/srslog/srslog.h"

namespace srsran {

enum class steep_state_t { IDLE, PROBING, ECHOING, DATA };
enum class steep_role_t  { ALICE, BOB };

struct steep_probe_t {
  uint32_t             probe_id = 0;
  std::vector<uint8_t> payload;
  bool                 consumed = false;
};

class steep_manager
{
public:
  using cf_t = std::complex<float>;

  static constexpr uint32_t STEEP_MAGIC     = 0xDEADBEEF;
  static constexpr float    STEEP_AMP       = 1.0f;
  static constexpr uint32_t SAMPLES_PER_BIT = 32u;
  static constexpr uint32_t CRC_SIZE_BYTES  = 4u;

  steep_manager();
  ~steep_manager() = default;

  bool init(const std::array<uint8_t, 32>& session_key);
  void reset();

  void set_role(steep_role_t role)                    { role_ = role; }
  void set_secret(const std::vector<uint8_t>& secret) { secret_ = secret; }
  void set_payload_size(uint32_t n)                   { payload_size_ = n; }

  steep_state_t get_state() const;
  bool          transition(steep_state_t next);

  bool store_probe(const steep_probe_t& probe);
  bool consume_probe(uint32_t probe_id, steep_probe_t& out);
  bool has_pending_probes() const;
  bool mix_secret(steep_probe_t& probe, const std::vector<uint8_t>& secret);

  void on_rx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);
  void on_tx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);

  const std::array<uint8_t, 32>& session_key() const { return session_key_; }

private:
  bool valid_transition(steep_state_t from, steep_state_t to) const;

  bool store_probe_locked(const steep_probe_t& probe);
  bool consume_probe_locked(uint32_t probe_id, steep_probe_t& out);
  bool mix_secret_locked(steep_probe_t& probe, const std::vector<uint8_t>& secret);

  uint32_t frame_size_samples() const {
    return (8u + payload_size_ + CRC_SIZE_BYTES) * 8u * SAMPLES_PER_BIT;
  }

  void encode_frame(cf_t* samples, uint32_t probe_id,
                    const std::vector<uint8_t>& payload);
  bool decode_frame(const cf_t* samples, uint32_t nof_samples,
                    uint32_t& probe_id_out, std::vector<uint8_t>& payload_out);

  void handle_tx_alice(cf_t* samples, uint32_t nof_samples);
  void handle_rx_alice(const cf_t* samples, uint32_t nof_samples);
  void handle_rx_bob  (const cf_t* samples, uint32_t nof_samples);
  void handle_tx_bob  (cf_t* samples, uint32_t nof_samples);

  static uint32_t compute_crc32(const uint8_t* data, size_t len);
  void prune_probe_buffer_locked();

  mutable std::mutex   mtx_;
  steep_state_t        state_        = steep_state_t::IDLE;
  steep_role_t         role_         = steep_role_t::ALICE;
  std::array<uint8_t, 32> session_key_{};
  std::vector<steep_probe_t> probe_buffer_;

  std::vector<uint8_t> secret_;
  uint32_t             payload_size_  = 32;
  uint32_t             next_probe_id_ = 1;
  bool                 echo_ready_    = false;
  steep_probe_t        pending_echo_;

  uint32_t             tx_call_count_ = 0;

  using clock_t = std::chrono::steady_clock;
  clock_t::time_point  start_time_{};
  clock_t::time_point  last_probe_time_{};
  clock_t::time_point  probing_since_time_{};
  clock_t::time_point  last_heartbeat_time_{};
  bool                 timing_initialized_ = false;

  static constexpr uint32_t STARTUP_DELAY_SECS  = 5u;
  static constexpr uint32_t PROBE_INTERVAL_SECS = 1u;   // was 5
  static constexpr uint32_t PROBE_TIMEOUT_SECS  = 8u;   // was 15 - retry faster

  // Zero-mean bipolar chip sequence (16 positive, 16 negative chips).
  // Length must equal SAMPLES_PER_BIT. Used for correlation detection.
  static const int8_t STEEP_CHIP[32];

  uint32_t probes_sent_      = 0u;
  uint32_t probes_recovered_ = 0u;
  uint32_t echoes_sent_      = 0u;
  uint32_t probes_timed_out_ = 0u;

  std::set<uint32_t> echoed_probe_ids_;
  std::vector<cf_t>  rx_accum_buf_;
  std::vector<cf_t>  rx_alice_accum_buf_;
  static constexpr uint32_t RX_ACCUM_MAX = 65536u;

  srslog::basic_logger& logger_;
};

} // namespace srsran
#endif