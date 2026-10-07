# Changelog

## 0.4.0 (unreleased)

### Changed

- The request-analysis cache is bounded: new `ScenarioConfig.request_buffer_limit`
  (default `8`; `-1` = unlimited, the behaviour of 0.3.0 and earlier; `0`
  disables the cache), accepted by `build_scenario` overrides and `make_env`.
  The `RequestAnalysisEngine` evicts the least recently used analysis first and
  reports `cache_size` and `cache_evictions`. The cache key holds the
  allocation version, so a normal step loop never hits it, and each entry takes
  about 0.2–0.3 MB on nobel-eu with 320 slots: the unbounded cache grew to
  about 0.9 GB after 3,000 arrivals and 8.7 GB after 120,000. With the default
  the memory stays flat (about 75 MB over 12,000 arrivals with the
  `jocn_benchmark` defaults). This changes memory use only: results are
  bit-identical for every limit, and the field is not part of
  `runtime_structure_key()`. The opt-in capture buffers
  (`capture_traffic_table`, `capture_step_trace`) remain unbounded, as
  documented.

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
