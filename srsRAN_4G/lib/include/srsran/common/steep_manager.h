#ifndef SRSRAN_STEEP_MANAGER_H
#define SRSRAN_STEEP_MANAGER_H

#include <array>
#include <complex>
#include <cstdint>
#include <mutex>
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

  static constexpr uint32_t STEEP_MAGIC = 0xDEADBEEF;
  //static constexpr float    STEEP_AMP   = 0.001f;  // ~-60 dB, invisible to LTE

  static constexpr float    STEEP_AMP        = 0.05f;
  static constexpr uint32_t SAMPLES_PER_BIT  = 32u;

  steep_manager();
  ~steep_manager() = default;

  bool init(const std::array<uint8_t, 32>& session_key);
  void reset();

  // Configuration - call these before enabling
  void set_role(steep_role_t role)                    { role_ = role; }
  void set_secret(const std::vector<uint8_t>& secret) { secret_ = secret; }
  void set_payload_size(uint32_t n)                   { payload_size_ = n; }

  // State machine
  steep_state_t get_state() const;
  bool          transition(steep_state_t next);

  // Probe buffer (public API)
  bool store_probe(const steep_probe_t& probe);
  bool consume_probe(uint32_t probe_id, steep_probe_t& out);
  bool has_pending_probes() const;

  // Secret mixing
  bool mix_secret(steep_probe_t& probe, const std::vector<uint8_t>& secret);

  // Sample hooks - called by radio class every TX/RX
  void on_rx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);
  void on_tx(cf_t* samples, uint32_t nof_samples, time_t full_secs, double frac_secs);

  const std::array<uint8_t, 32>& session_key() const { return session_key_; }

private:
  // State machine helpers
  bool valid_transition(steep_state_t from, steep_state_t to) const;

  // Mutex-free versions called from inside on_rx/on_tx (which already hold the lock)
  bool store_probe_locked(const steep_probe_t& probe);
  bool consume_probe_locked(uint32_t probe_id, steep_probe_t& out);
  bool mix_secret_locked(steep_probe_t& probe, const std::vector<uint8_t>& secret);

  // Encoding/decoding into IQ samples
  // uint32_t frame_size_samples() const { return (8u + payload_size_) * 8u; }
  uint32_t frame_size_samples() const { return (8u + payload_size_) * 8u * SAMPLES_PER_BIT; }
// Now correctly: 40 * 8 * 32 = 10,240
  void     encode_frame(cf_t* samples, uint32_t probe_id,
                        const std::vector<uint8_t>& payload);
  bool     decode_frame(const cf_t* samples, uint32_t nof_samples,
                        uint32_t& probe_id_out, std::vector<uint8_t>& payload_out);

  // Role-specific logic
  void handle_tx_alice(cf_t* samples, uint32_t nof_samples);
  void handle_rx_alice(const cf_t* samples, uint32_t nof_samples);
  void handle_rx_bob  (const cf_t* samples, uint32_t nof_samples);
  void handle_tx_bob  (cf_t* samples, uint32_t nof_samples);

  mutable std::mutex         mtx_;
  steep_state_t              state_         = steep_state_t::IDLE;
  steep_role_t               role_          = steep_role_t::ALICE;
  std::array<uint8_t, 32>   session_key_{};
  std::vector<steep_probe_t> probe_buffer_;

  std::vector<uint8_t> secret_;
  uint32_t             payload_size_   = 32;
  uint32_t             next_probe_id_  = 1;
  bool                 echo_ready_     = false;
  bool                 handshake_done_ = false; // one-shot: stops after first success
  steep_probe_t        pending_echo_;

  srslog::basic_logger& logger_;
};

} // namespace srsran
#endif