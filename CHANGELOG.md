# Changelog

## 0.3.0

### Added

- Runtime physical-layer updates (network aging): `SpanUpdate`,
  `TopologyModel.with_span_updates`, `QoTEngine.set_topology` and
  `Simulator.update_spans` / `OpticalEnv.update_spans` change the fibre loss
  and amplifier noise figure of individual spans during a simulation while
  keeping the traffic state. The QoT after any sequence of updates is
  bit-identical to that of a freshly built engine on the updated topology.
  `Simulator.base_topology` and `Simulator.physical_layer_version` track the
  physical layer; `reset(options={"keep_physical_layer": True})` keeps it
  across a full reset. See "Runtime physical-layer updates (aging)" in
  `docs/docs/physical_layer.md`.

### Fixed

- `reset(options={"only_episode_counters": True})` after a terminated episode
  replayed the last processed request of the previous episode (an accepting
  action then raised `ValueError("service_id ... is already active")`, and a
  reject was counted twice). The counter-only reset now prepares the next
  request; a mid-episode counter-only reset is unchanged.
