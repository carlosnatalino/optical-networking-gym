# Heterogeneous physical layer

The QoT engine computes the generalised SNR (GSNR) of a lightpath as a sum of
noise-to-signal ratios (NSR). Nonlinear interference (NLI) uses the incoherent
closed-form GN model with a modulation-format correction (see
[below](#modulation-format-correction)). All QoT arithmetic runs in the Cython
kernel `optical/kernels/qot_kernel.pyx`, and its pure-Python twin
`qot_kernel.py` has the same API.

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

- `β2,acc` is the dispersion a channel has accumulated from its own
  transmitter to the input of span `n`, using the kernel's constant `|β2|` =
  21.3 ps²/km. It is measured in the direction of travel: the topology keeps
  one path record for both directions of a node pair, and a lightpath whose
  source is the record's last node travels its links backwards
  (`QoTEngine.travels_reversed(path, source_id)`). Within a link, spans are
  taken in their stored order, as for every other per-span quantity.
  CFM2 is the only direction-dependent part of the model, and the effect is
  small (on nobel-eu about 1e-3 dB, at most 0.015 dB).
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

The route is either a `PathRecord` (`path=`, with `reverse=True` for a channel
that travels it backwards) or any sequence of links in the order of travel
(`link_ids=`), e.g. a sub-path or the output of an external planner.
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
