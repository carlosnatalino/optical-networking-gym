# Heterogeneous physical layer

The QoT engine computes the generalised SNR (GSNR) of a lightpath as a sum of
noise-to-signal ratios (NSR), with the incoherent GN model and the closed-form
EGN modulation-format correction for nonlinear interference (NLI). All QoT
arithmetic runs in the Cython kernel `optical/kernels/qot_kernel.pyx`, and its
pure-Python twin `qot_kernel.py` has the same API.

## Scenario options

| `ScenarioConfig` field | Default | Effect |
|---|---|---|
| `launch_power_dbm_choices`, `launch_power_seed` | `None` | Each dynamic request draws its launch power from the choices, on its own RNG stream (the traffic sequence is unchanged). |
| `nli_interferer_psd` | `"cut"` | `"actual"` weights each interferer's XCI by its own power spectral density; `"cut"` assumes the channel-under-test PSD (historical). |
| `nli_include_interferers` | `None` | Include XCI from established services without enabling disruption tracking (`None` follows `measure_disruptions`). |
| `nli_coherence_epsilon` | `0.0` | Coherent NLI accumulation: path NLI × N_spans^ε. |
| `roadm_add_osnr_db`, `roadm_drop_osnr_db`, `roadm_express_osnr_db`, `transceiver_osnr_db` | `None` | Constant-OSNR node and transceiver noise terms. |

With every option at its default, the results are bit-identical to the
historical engine.

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
of each link, the coherent excess, the ROADM add/express/drop and transceiver
terms, and the total.

`OpticalEnv.observation(obs)` maps the returned observation, like gymnasium's
`ObservationWrapper`. `OpticalEnv.on_action_applied(transition)` runs right
after each action, before time advances, so subclasses can record the network
state a new lightpath sees at establishment. `Simulator.last_transition` holds
the most recent step outcome.
