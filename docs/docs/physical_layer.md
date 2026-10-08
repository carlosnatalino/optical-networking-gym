# Heterogeneous physical layer

The QoT engine computes the generalised SNR (GSNR) of a lightpath as a sum of
noise-to-signal ratios (NSR). Nonlinear interference (NLI) uses the incoherent
closed-form GN model with a modulation-format correction (see
[below](#modulation-format-correction)). All QoT arithmetic runs in the Cython
kernel `optical/kernels/qot_kernel.pyx`, and its pure-Python twin
`qot_kernel.py` has the same API.

## Undirected model

Lightpaths are bidirectional: a lightpath reserves its slots on every link of
its route for both directions. The physical model is undirected accordingly.
The GSNR of a route is computed once, in its *canonical direction* (from the
endpoint with the lower node index to the other), which is the direction of
the topology's k-shortest path records. A route and its reverse get the same
GSNR, bit for bit, whatever the source of the request, and every per-link or
per-span quantity (gain ripple, span order, CFM2 distances) follows that
direction.

This is a deliberate simplification. Only the CFM2 correction would depend on
the direction, through the dispersion accumulated since the transmitter. The
reverse direction differs by about 1e-3 dB on nobel-eu (at most 0.015 dB over
the multi-hop paths, with or without heterogeneous span lengths), and that
difference is disregarded. Modelling each direction on its own would also
require a per-direction inventory (span order, amplifiers, connectors, ripple)
and XCI from the co-directional halves of the interferers only.

## Spectral grid and channel width

The spectrum is a grid of `num_spectrum_resources` slots of
`frequency_slot_bandwidth` Hz starting at `frequency_start`. The slot width is
set **only** by `frequency_slot_bandwidth`; the other two width fields of
`ScenarioConfig` are derived from it and validated against it:

| Field | Unit | Value |
|---|---|---|
| `frequency_slot_bandwidth` | Hz | The slot width (source of truth). |
| `channel_width` | GHz | `frequency_slot_bandwidth / 1e9` when `None` (the default); a different value raises `ValueError`. |
| `bandwidth` | Hz | `num_spectrum_resources × frequency_slot_bandwidth` when `None` (the default); a different value raises `ValueError`. |

`build_scenario` re-derives `channel_width` and `bandwidth` when an override
changes `frequency_slot_bandwidth` or `num_spectrum_resources`. A plain
`dataclasses.replace` carries the resolved values over, so set them to `None`
when changing the grid that way.

**Slots of a request.** A request of `bit_rate` Gb/s with a format of
`spectral_efficiency` b/s/Hz needs
`ceil(bit_rate / (spectral_efficiency × slot width in GHz))` slots. The guard
slot is not part of the lightpath: it is reserved after it
(`occupied_slot_end_exclusive` is one past the last service slot, unless the
lightpath ends at the last slot of the grid).

**Signal bandwidth in the QoT.** A lightpath of `n` slots starting at slot `s`
is modelled as a signal of bandwidth `n × frequency_slot_bandwidth`, centred
at `frequency_start + frequency_slot_bandwidth × (s + n/2)`. The occupied
bandwidth stands in for the symbol rate: for example, a 40 Gb/s 64QAM
lightpath uses one 12.5 GHz slot and is modelled as a 12.5 GHz signal.

**Launch power.** The launch power is per channel (`launch_power_dbm`, or the
request's own value), so the power spectral density `P/B` of a lightpath
decreases as its width grows.

**Interferer PSD.** The NLI noise-to-signal ratio of a span is
`(P/B_cut)² × [asinh(·) of the SCI + Σ φ_j]`, where `φ_j` is the
cross-channel interference (XCI) of interferer `j`. With
`nli_interferer_psd="cut"` (the default, kept for historical comparability),
every interferer is assumed to have the PSD of the channel under test, so the
width of the channel under test also changes the interference attributed to
its neighbours. `nli_interferer_psd="actual"` weights each `φ_j` by
`(G_j/G_cut)²`, with `G_j` the interferer's own PSD, and is the physically
consistent choice.

> Observation (TNSM study, nobel-eu, 140 Erlang, 0 dBm, `cut`). Evaluating the same
> lightpath as a channel two slots wider changes its GSNR by −0.02 to −1.36 dB (−0.81 dB
> on average) on an empty network (more ASE), but by +1.75 dB on average on a loaded one
> (lower PSD, hence less XCI from every interferer).

These modelling choices (occupied bandwidth instead of the symbol rate, a
fixed power per channel instead of a fixed PSD, and `"cut"` as the default)
are kept because changing any of them changes every result.

## Scenario options

| `ScenarioConfig` field | Default | Effect |
|---|---|---|
| `launch_power_dbm_choices`, `launch_power_seed` | `None` | Each dynamic request draws its launch power from the choices, on its own RNG stream (the traffic sequence is unchanged). |
| `nli_interferer_psd` | `"cut"` | `"actual"` weights each interferer's XCI by its own power spectral density; `"cut"` assumes the channel-under-test PSD (historical). |
| `nli_include_interferers` | `None` | Include the cross-channel interference (XCI) of established services. `None` follows `measure_disruptions`, so **by default the NLI is self-channel only and the GSNR does not depend on the load**; set it to `True` for a GN/EGN model with XCI (the `jocn_benchmark` preset does). |
| `nli_coherence_epsilon` | `0.0` | Coherent NLI accumulation: path NLI × N_spans^ε. |
| `nli_modulation_correction` | `"egn_xci"` | Modulation-format correction of the GN model: `"egn_xci"`, `"cfm2"` or `"gn"` (see below). |
| `roadm_add_osnr_db`, `roadm_drop_osnr_db`, `roadm_express_osnr_db`, `transceiver_osnr_db` | `None` | Constant-OSNR node and transceiver noise terms. |

With every option at its default, the results are bit-identical to the
historical engine. The default keeps `nli_include_interferers` off because
XCI makes every QoT evaluation depend on the lightpaths already established,
which changes the results of existing studies and costs time (on nobel-eu a
request takes about 2.5–3.5x longer to process with XCI).

## Modulation-format correction

The GN model assumes Gaussian-distributed signals, so it overestimates the NLI
of real constellations, most for low-order formats and short reaches.
`nli_modulation_correction` selects the correction:

- `"egn_xci"` (default, historical): the closed-form EGN correction of
  Poggiolini et al. (JLT 33(2), 2015). Each span subtracts
  `Φ_j · (B_j/|Δf_j|) · (5/3) · L_eff/L_s` from the XCI term of every
  interferer `j`. The correction depends only on the interferers' formats: a
  lightpath's own format does not change its GSNR, and nothing is corrected
  when no interferers are included.
- `"cfm2"`: the machine-learning correction factors of CFM2 (Ranjbar Zefreh et
  al., JLT 38(18), 2020, Eqs. (2) and (11)). In every span `n`, the SCI term
  is multiplied by `ρ_CUT(Φ_CUT, R_CUT, β2,acc)` and the XCI term of each
  interferer by `ρ_nch(Φ_nch, β2,acc)`. The factors were fitted to the full EGN
  model over 8500 C-band systems, so the lightpath's own format changes its
  GSNR, also without interferers.
- `"gn"`: no correction (Gaussian signals).

`Φ` is the EGN constant of the format (1 for BPSK/QPSK, 2/3 for 8QAM, 17/25 for
16QAM, 69/100 for 32QAM, 13/21 for 64QAM, and 0 for Gaussian signals), looked
up by spectral efficiency. For CFM2 in a mesh network:

- `β2,acc` is the dispersion a channel has accumulated from the start of its
  route to the input of span `n`, in the route's canonical direction (see
  [Undirected model](#undirected-model)), using the kernel's constant `|β2|` =
  21.3 ps²/km. Interferers accumulate it along their own route, likewise.
- The occupied bandwidth stands in for the symbol rate `R_CUT`.
- The factors were fitted for 32–128 GBaud channels and 80–120 km spans.
- The paper's coherence (CFM3) and roll-off (CFM4) refinements are not modelled.

The CFM2 factors do not depend on the candidate frequency. `ρ_CUT` is folded
into the per-span SCI term once per kernel call, and `ρ_nch` is computed once
per link state and cached, so CFM2 costs the same as the default correction.
The coefficients live in `optical/cfm2.py`.

The XCI term of an interferer, `asinh(·) − asinh(·)`, depends on the span only
through the fibre attenuation. When all spans of a link share it, the kernel
evaluates it once per link and candidate instead of once per span, with the
same expression, so the results are bit-identical and XCI costs roughly a
constant per link instead of per span.

## Per-span inventory

`network.equipment.EquipmentLibrary.from_json` reads a GNPy `eqpt_config.json`.
The EDFA noise-figure models `fixed_gain`, `variable_gain` (two-coil) and
`advanced_model` are ported from GNPy, gain-ripple profiles are read from
`advanced_config_from_json`, and the ROADM and transceiver OSNR defaults are
read as well.

`network.inventory.NetworkInventory` is a JSON list of the spans of every link:
fibre length and loss, connector losses `con_in_db`/`con_out_db`, and the
amplifier's type, NF and ripple scale/shift. `apply_inventory(topology,
inventory, equipment)` returns a `TopologyModel` with the same routes and a
heterogeneous physical layer. The engine then accounts for:

- lumped losses: the amplifier gain includes them, and the input loss lowers
  the power launched into the fibre;
- per-amplifier NF;
- EDFA gain ripple, as a channel-dependent power offset that accumulates
  within a link and is reset by per-channel equalisation at each ROADM.

## Runtime physical-layer updates (aging)

Aging studies change the fibre loss and the amplifier noise figure of some
spans *during* a simulation, while the traffic state (established lightpaths,
spectrum, time) is kept. `Simulator.update_spans` (and `OpticalEnv.update_spans`,
which delegates to it) does this through the public API:

```python
from optical_networking_gym import SpanUpdate

env.reset(seed=0)
...  # run part of an episode
aged = env.update_spans(
    [
        SpanUpdate(link_id=3, span_index=0, attenuation_db_per_km=0.23),
        SpanUpdate(link_id=3, span_index=1, noise_figure_db=6.0),
        SpanUpdate(link_id=7, span_index=2, attenuation_db_per_km=0.21, noise_figure_db=5.5),
    ],
    refresh_active_services=False,
)
assert env.simulator.topology is aged
env.simulator.physical_layer_version  # 1
```

A `SpanUpdate` field left at `None` keeps the current value.
`TopologyModel.with_span_updates(updates)` returns the updated model without
changing the original. Updates are applied in order, so a later update of the
same span wins, and everything else (nodes, links, span lengths, lumped
losses, gain ripple, path records) is shared with the original. Span length,
lumped losses and gain ripple cannot be updated.

Semantics of `update_spans(updates, *, refresh_active_services=False)`:

- **Validation.** Every update is validated before anything changes: the link
  id and span index must be in range, and the values must be finite and
  positive. Otherwise `ValueError` is raised and nothing changes. An empty
  list is a no-op.
- **Propagation.** `simulator.topology` becomes the new model, for every
  helper: the QoT engine, request analysis, action mask, observation, reward,
  runtime state and traffic model. `simulator.physical_layer_version` (0 at
  start) is incremented on every non-empty call.
- **Pending request.** If a request is waiting for an action, its analysis,
  action mask and observation are rebuilt, so the policy and the QoT check of
  the next step (`mask_mode="resource_and_qot"`) see the new physics.
  `request_analysed_callback` / `on_request_analysed` is not called again.
- **Established lightpaths.** Their stored QoT is unchanged by default. With
  `refresh_active_services=True`, the QoT of the services on the changed links
  is recomputed. If `measure_disruptions` is set, those that no longer meet
  their threshold are disrupted (or dropped with `drop_on_disruption`) by the
  same logic as for any other QoT change, and counted in the statistics.
- **Lifetime.** Updates persist across `reset(options={"only_episode_counters":
  True})` and `reset(options={"keep_physical_layer": True})`. Any other
  `reset()` restores the topology passed to the constructor
  (`simulator.base_topology`) and sets the version back to 0. An update made
  before the first `reset()` is therefore lost unless that reset keeps the
  physical layer.
- **Exactness.** After any sequence of updates, the engine computes exactly
  (bit for bit) what a freshly built engine on
  `base_topology.with_span_updates(all updates)` computes in the same runtime
  state, with either kernel, every `nli_modulation_correction` and with or
  without interferers.
- **Cost.** Only the changed links are re-read
  (`QoTEngine.set_topology(topology, changed_link_ids=...)`), and only the
  cached routes through them are invalidated; they are rebuilt lazily when
  they are next evaluated. Updating two spans on nobel-eu takes about 0.07 ms,
  and at most about 0.4 ms when every k-shortest path is cached. Rebuilding
  the analysis of the pending request costs as much as preparing a new
  request (a few ms on nobel-eu), so pass the updates that happen at the
  same time in one call.

The aging process itself (which spans age, by how much and when) is up to the
caller. A typical loop updates the spans every `N` requests (`aging_model` and
`policy` stand for your own code):

```python
for step in range(episode_length):
    if step and step % 500 == 0:
        env.update_spans(aging_model.updates_at(env.simulator.state.current_time))
    action = policy(env)
    env.step(action)
```

The request-analysis cache key holds `QoTEngine.topology_version`, so an
analysis computed under an older physical layer is never returned, even when
`QoTEngine.set_topology` is called directly.

`examples/heuristics/network_aging.py` runs such a loop: every 250 requests it
raises the loss and noise figure of every span and refreshes the established
lightpaths, which are then disrupted when they fall below their threshold.

## Noise breakdown and environment hooks

`QoTEngine.noise_breakdown(...)` and `service_noise_breakdown(state,
service_id)` return a `LightpathNoiseBreakdown`. It holds the ASE and NLI NSR
of each link, the NLI split into its self-channel (`link_sci_nsr`) and
cross-channel (`link_xci_nsr`) parts, the coherent excess, the ROADM
add/express/drop and transceiver terms, and the total. SCI scales with the
square of the launch power; with `nli_interferer_psd="actual"` the XCI NSR
does not depend on it. NLI is clipped at 0 per span, the remainder being kept
in `nli_correction_nsr`; the clip can only trigger with the default
`"egn_xci"` correction, so with `"cfm2"` or `"gn"` the per-link sums are exact
(up to rounding).

`QoTEngine.summarize_candidate_at(state=, service_id=, path=, modulation=,
service_slot_start=, service_num_slots=, launch_power=None)` returns the
`QoTCandidateSummary` (OSNR, ASE, NLI, margin, threshold check, NLI shares) of
one candidate in the current runtime state. `summarize_candidates_at(state=,
service_id=, path=, candidates=, launch_power=None)` does the same for a list
of `(modulation, service_slot_start, service_num_slots)` candidates of one
route, preparing the route's interferers once; each result is bit-identical to
the single call. It pays off when the candidates are evaluated anyway (on a
loaded nobel-eu network, 15 µs instead of 26 µs per candidate for the 6
formats of a route), not for a first fit that stops after the first feasible
format.

The route is either a `PathRecord` (`path=`) or any sequence of links in order
(`link_ids=`, in either direction), e.g. a sub-path or the output of an
external planner. The per-link arrays follow the route's canonical direction.
`TopologyModel.path_from_link_ids(link_ids)` builds the matching record. The
engine and the runtime state identify a path by its links, not by its id, so
records that reuse an id are never confused.

`OpticalEnv.observation(obs)` maps the returned observation, like gymnasium's
`ObservationWrapper`. `OpticalEnv.on_action_applied(transition)` runs right
after each action, before time advances, so subclasses can record the network
state a new lightpath sees at establishment.
`OpticalEnv.on_request_analysed(analysis)` runs for every new request before
the policy acts, with the `RequestAnalysis` the environment built: candidate
paths, formats (`modulation_indices`), slots (`required_slots_by_path_mod`,
`resource_valid_starts`), `launch_power_dbm` and the GSNR of every evaluated
candidate (`gsnr_db_by_start`, float32 precision). These are the QoT queries of
the RMSA. The hook is only wired when a subclass overrides it.
`Simulator.last_transition` holds the most recent step outcome.

`heuristics.select_heuristic_action(name, env, info)` selects an action with a
heuristic given by name (`HEURISTIC_NAMES` lists them).
