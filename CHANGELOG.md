# Changelog

## 0.4.0

### Added

- `ScenarioConfig.analysis_detail` (`"full"` by default, or `"resources"`,
  also a `make_env` argument). `"resources"` builds a lean request analysis:
  paths, formats, required slots, resource-valid starts and, with
  `RESOURCE_AND_QOT`, the QoT arrays, while the fragmentation damage, link
  metrics, route cuts/RSS and free-run statistics are zero-filled (same shapes
  and dtypes). It requires `enable_observation=False`; the fragmentation terms
  of `StepTransition` and of the reward are then 0. A build that asks for
  inspection (`Observation.build_snapshot`) stays full. On nobel-eu
  (`jocn_benchmark`, `RESOURCE_ONLY`, no observation or mask, a heuristic
  that evaluates the QoT itself) a step takes 0.47 ms instead of 0.82 ms
  (−43%), with the same valid starts, slot counts, decisions and GSNRs.
- `QoTEngine.summarize_candidates_at(state=, service_id=, path=, candidates=,
  launch_power=None)` evaluates several `(modulation, service_slot_start,
  service_num_slots)` candidates of one route with a single preparation of
  the route's interferers. Each result is bit-identical to
  `summarize_candidate_at`. On a loaded nobel-eu network a candidate costs
  15 µs instead of 26 µs when the 6 formats of a route are evaluated together.

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
- The slot width has one source of truth, `frequency_slot_bandwidth` (Hz).
  `ScenarioConfig.channel_width` now defaults to `None` and is derived as
  `frequency_slot_bandwidth / 1e9`; an explicit value must match it
  (`rel_tol=1e-9`), else `ValueError`. Before, setting only
  `frequency_slot_bandwidth=6.25e9` silently computed slot counts for 12.5 GHz
  slots (a 100 Gb/s 16QAM request got 2 slots instead of 4) while the grid and
  the QoT used 6.25 GHz. `bandwidth` must likewise equal
  `num_spectrum_resources × frequency_slot_bandwidth` when given.
  `build_scenario` re-derives both when an override changes the grid (so
  `num_spectrum_resources` overrides of the presets stay valid), and the
  presets, `build_nobel_eu_graph_load_scenario` (which hard-coded
  `bandwidth=4e12` for any slot count) and `make_env` no longer pass them.
  A `dataclasses.replace` that changes the grid carries the resolved values
  over and now raises; pass `channel_width=None, bandwidth=None` with it.
  Results of every consistent configuration (all presets) are unchanged.
  New "Spectral grid and channel width" section in
  `docs/docs/physical_layer.md` documents the width semantics of the slot
  count, the QoT signal bandwidth, the per-channel launch power and the
  `nli_interferer_psd="cut"` approximation.
- The request analysis selects the format window directly from `k_paths`-row
  work arrays instead of padding eight arrays per request (bit-identical
  output; about 4% faster in the lean loop above, 1% with the preset
  defaults).

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
