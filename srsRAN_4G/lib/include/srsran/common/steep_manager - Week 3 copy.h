#ifndef SRSRAN_STEEP_MANAGER_H
#define SRSRAN_STEEP_MANAGER_H

#include <array>
#include <cstdint>
#include <mutex>
#include <vector>
#include "srsran/srslog/srslog.h"

// OLD - #include "srsran/phy/rf/rf.h"

namespace srsran {

// The four states STEEP can be in at any moment
enum class steep_state_t {
  IDLE,      // doing nothing
  PROBING,   // Alice has sent a probe, waiting for echo
  ECHOING,   // Bob has received a probe, mixing in secret and sending back
  DATA       // Alice has received the echo, ready to extract the message
};

// One probe packet: a header tag, a random payload, and a "used" flag
struct steep_probe_t {
  uint32_t             probe_id;            // unique ID so we never reuse a probe
  std::vector<uint8_t> payload;             // random data Alice sends
  bool                 consumed = false;    // marks probe as used after echo
};

class steep_manager
{
public:
  steep_manager();
  ~steep_manager() = default;

  // Start up the manager with a shared secret key (32 bytes)
  bool init(const std::array<uint8_t, 32>& session_key);

  // Reset everything back to IDLE
  void reset();

  // --- State machine ---
  steep_state_t get_state() const;
  bool          transition(steep_state_t next);

// --- Probe buffer ---
  // Alice stores a probe here after sending it
  bool store_probe(const steep_probe_t& probe);
  // Bob (or Alice on return) pulls a probe out to work with
  bool consume_probe(uint32_t probe_id, steep_probe_t& out);
  // Quick check: is there anything waiting?
  bool has_pending_probes() const;

  // Called every time samples arrive from the radio (RX path)
  // OLD - void on_rx(uint32_t nof_samples, const srsran_timestamp_t& ts);
  void on_rx(uint32_t nof_samples, time_t full_secs, double frac_secs);

  // Called every time samples are about to be sent to the radio (TX path)
  // OLD - void on_tx(uint32_t nof_samples, const srsran_timestamp_t& ts);
  void on_tx(uint32_t nof_samples, time_t full_secs, double frac_secs);

  // --- Secret mixing (Bob's side) ---
  // XORs the secret message into the probe payload
  // Returns false if sizes don't match
  bool mix_secret(steep_probe_t& probe, const std::vector<uint8_t>& secret);

  // --- Read-only key access ---
  const std::array<uint8_t, 32>& session_key() const { return session_key_; }

private:
  bool valid_transition(steep_state_t from, steep_state_t to) const;

  mutable std::mutex         mtx_;
  steep_state_t              state_   = steep_state_t::IDLE;
  std::array<uint8_t, 32>   session_key_{};
  std::vector<steep_probe_t> probe_buffer_;
  srslog::basic_logger&      logger_;
};

} // namespace srsran

#endif // SRSRAN_STEEP_MANAGER_H