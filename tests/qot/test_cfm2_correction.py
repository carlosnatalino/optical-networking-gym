"""CFM2 modulation-format correction factors [Ranjbar Zefreh et al., JLT 2020].

Covers the factor formulas (Eq. (11), Table II), the compiled kernel against its
pure-Python twin, and the engine: with CFM2 the lightpath's own format changes
its GSNR, while the default correction is left untouched.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym import (
    Modulation,
    QoTEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
    make_env,
    select_first_fit_action,
)
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import PathRecord
from optical_networking_gym.optical import cfm2
from optical_networking_gym.optical.kernels import qot_kernel as compiled_kernel

_TWIN_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "optical_networking_gym" / "optical" / "kernels" / "qot_kernel.py"
)


def _load_python_twin():
    spec = importlib.util.spec_from_file_location("qot_kernel_python_twin_cfm2", _TWIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


python_kernel = _load_python_twin()

# Table II of the paper, typed independently of ``optical/cfm2.py``.
TABLE_II = (
    0.93143, -0.77122, 0.91090, -14.555, 0.85816, -0.99415, 1.0812, 0.0052247, 0.99313,
    -1.8838, 0.62974, -11.421, 0.67368, -1.1759, 0.0064482, 187380.0, 1952.7, -2.0016,
)

QPSK = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)
QAM16 = Modulation("16QAM", 100_000.0, 4, minimum_osnr=13.24)
QAM64 = Modulation("64QAM", 100_000.0, 6, minimum_osnr=18.6)


# --------------------------------------------------------------------------
# Factor formulas
# --------------------------------------------------------------------------


def _reference_rho_nch(phi: float, dispersion_ps2: float) -> float:
    a = TABLE_II
    return a[0] + a[1] * phi ** a[2] + a[3] * phi ** a[4] * (1 + a[5] * (abs(dispersion_ps2) + a[6]) ** a[7])


def _reference_rho_cut(phi: float, rate_tbaud: float, dispersion_ps2: float) -> float:
    a = TABLE_II
    return (
        a[8]
        + a[9] * phi ** a[10]
        + a[11] * phi ** a[12] * (1 + a[13] * rate_tbaud ** a[14] + a[15] * (abs(dispersion_ps2) + a[16]) ** a[17])
    )


def test_coefficients_match_table_ii() -> None:
    assert cfm2.CFM2_COEFFICIENTS == pytest.approx(TABLE_II, rel=1e-12)


def test_dispersion_unit_matches_the_kernel_beta2() -> None:
    assert python_kernel._ABS_BETA_2 * 1e27 == pytest.approx(cfm2.ABS_BETA2_PS2_PER_KM, rel=1e-12)


def test_gaussian_signals_reduce_to_the_constant_terms() -> None:
    # Phi = 0: every format-dependent term vanishes, for any dispersion.
    for dispersion in (0.0, 1e3, 5e4):
        assert float(cfm2.rho_interferer(0.0, dispersion)) == TABLE_II[0]
        assert cfm2.rho_cut(0.0, 0.032, dispersion) == TABLE_II[8]


@pytest.mark.parametrize("phi", [1.0, 17 / 25, 13 / 21])
@pytest.mark.parametrize("distance_km", [0.0, 80.0, 1000.0, 4000.0])
def test_factors_match_independent_reference(phi: float, distance_km: float) -> None:
    dispersion = 21.3 * distance_km
    assert float(cfm2.rho_interferer(phi, dispersion)) == pytest.approx(_reference_rho_nch(phi, dispersion), rel=1e-12)
    assert cfm2.rho_cut(phi, 0.05, dispersion) == pytest.approx(_reference_rho_cut(phi, 0.05, dispersion), rel=1e-12)


def test_factor_values_are_physically_sensible() -> None:
    # Hand-evaluated Eq. (11): the GN model overestimates QPSK XCI strongly in
    # the first span, much less after 1000 km of SMF.
    assert float(cfm2.rho_interferer(1.0, 0.0)) == pytest.approx(0.081, abs=2e-3)
    assert float(cfm2.rho_interferer(1.0, 21.3 * 1000)) == pytest.approx(0.85, abs=0.01)
    # SCI: strong correction in the first span (a16 term), relaxing towards
    # a9 + a10 + a12 * (1 + a14 * R^a15) ~ 0.823 as dispersion accumulates.
    assert cfm2.rho_cut(1.0, 0.032, 0.0) == pytest.approx(0.269, abs=2e-3)
    assert cfm2.rho_cut(1.0, 0.032, 21.3 * 80) == pytest.approx(0.665, abs=2e-3)
    assert cfm2.rho_cut(1.0, 0.032, 21.3 * 1e5) == pytest.approx(0.823, abs=2e-3)
    sci = [cfm2.rho_cut(13 / 21, 0.05, 21.3 * km) for km in (0.0, 80.0, 400.0, 2000.0)]
    assert np.all(np.diff(sci) > 0)
    # XCI factors grow towards the Gaussian limit with accumulated dispersion.
    rho = cfm2.rho_interferer(np.array([1.0, 17 / 25, 13 / 21])[:, None], 21.3 * np.array([0.0, 500.0, 2000.0]))
    assert np.all(np.diff(rho, axis=1) > 0)
    assert np.all((rho > 0.0) & (rho < 1.0))


def test_phi_table_matches_table_i() -> None:
    assert cfm2.PHI_BY_SPECTRAL_EFFICIENCY == {
        1: 1.0,
        2: 1.0,
        3: 2 / 3,
        4: 17 / 25,
        5: 69 / 100,
        6: 13 / 21,
        7: 1105 / 1681,
        8: 257 / 425,
    }


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------

SLOT = 12.5e9
F_START = 3e8 / 1565e-9
ALPHA = 0.2 / (2 * 10 * math.log10(math.e) * 1e3)
SPAN_OFFSETS = np.array([0, 3, 5], dtype=np.int32)
RUNNING_OFFSETS = np.array([0, 4, 6], dtype=np.int32)


def _kernel_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    lengths = rng.uniform(60.0, 100.0, 5)
    slots = rng.choice(np.arange(10, 300, 6), size=6, replace=False)
    widths = rng.integers(2, 6, size=6)
    # One block per link: (3 spans x 4 interferers) + (2 spans x 2 interferers).
    return {
        "lengths": lengths,
        "ids": np.arange(100, 106, dtype=np.int32),
        "freqs": F_START + SLOT * slots + SLOT * widths / 2.0,
        "bw": SLOT * widths.astype(np.float64),
        "phi": rng.choice([1.0, 2 / 3, 17 / 25], size=6),
        "powers": 10 ** ((rng.uniform(-6, 0, size=6) - 30) / 10),
        "rho": rng.uniform(0.05, 0.95, size=3 * 4 + 2 * 2),
    }


def _path_noise(kernel, inputs: dict[str, np.ndarray], **extra):
    n = inputs["lengths"].shape[0]
    return kernel.path_noise(
        SPAN_OFFSETS,
        inputs["lengths"],
        np.full(n, ALPHA),
        np.full(n, 10**0.55),
        np.full(n, 10**0.05),
        np.ones(n),
        np.zeros((n, 0)),
        RUNNING_OFFSETS,
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        inputs["powers"],
        current_service_id=0,
        center_frequency=F_START + SLOT * 152,
        bandwidth=SLOT * 4,
        launch_power=5e-4,
        include_nli=True,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        interferer_psd_actual=True,
        nli_scale=1.1,
        extra_nsr=1e-4,
        **extra,
    )


def _summarize(kernel, inputs: dict[str, np.ndarray], **extra):
    n = inputs["lengths"].shape[0]
    return kernel.summarize_candidate_starts(
        SPAN_OFFSETS,
        inputs["lengths"],
        np.full(n, ALPHA),
        np.full(n, 10**0.55),
        RUNNING_OFFSETS,
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        np.array([0, 40, 120, 200, 290], dtype=np.int32),
        current_service_id=0,
        frequency_start=F_START,
        frequency_slot_bandwidth=SLOT,
        service_num_slots=4,
        launch_power=1e-3,
        threshold=10.0,
        include_nli=True,
        running_launch_powers=inputs["powers"],
        interferer_psd_actual=True,
        **extra,
    )


def test_cfm2_kernel_matches_python_twin() -> None:
    inputs = _kernel_inputs(np.random.default_rng(11))
    extra = {"cfm2": True, "cut_phi": 17 / 25, "running_rho": inputs["rho"]}
    for got, expected in zip(_path_noise(compiled_kernel, inputs, **extra), _path_noise(python_kernel, inputs, **extra)):
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    for got, expected in zip(_summarize(compiled_kernel, inputs, **extra), _summarize(python_kernel, inputs, **extra)):
        np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_cfm2_single_link_matches_python_twin() -> None:
    inputs = _kernel_inputs(np.random.default_rng(12))
    results = []
    for kernel in (compiled_kernel, python_kernel):
        results.append(
            kernel.accumulate_link_noise(
                inputs["lengths"][:3],
                np.full(3, ALPHA),
                np.full(3, 10**0.55),
                inputs["ids"][:4],
                inputs["freqs"][:4],
                inputs["bw"][:4],
                inputs["phi"][:4],
                current_service_id=0,
                center_frequency=F_START + SLOT * 152,
                bandwidth=SLOT * 4,
                launch_power=1e-3,
                include_nli=True,
                cfm2=True,
                cut_phi=1.0,
                running_rho=inputs["rho"][:12],
                cut_start_distance_km=400.0,
            )
        )
    np.testing.assert_allclose(results[0], results[1], rtol=1e-12)


def test_legacy_mode_ignores_the_cfm2_arguments() -> None:
    inputs = _kernel_inputs(np.random.default_rng(13))
    ignored = {"cfm2": False, "cut_phi": 0.5, "running_rho": inputs["rho"]}
    for kernel in (compiled_kernel, python_kernel):
        for got, expected in zip(_path_noise(kernel, inputs, **ignored), _path_noise(kernel, inputs)):
            np.testing.assert_array_equal(got, expected)
        for got, expected in zip(_summarize(kernel, inputs, **ignored), _summarize(kernel, inputs)):
            np.testing.assert_array_equal(got, expected)


def test_cfm2_scales_the_gaussian_sci_by_a9() -> None:
    # No interferers: CFM2 with Phi_CUT = 0 is the GN model times a9 exactly.
    inputs = _kernel_inputs(np.random.default_rng(14))
    empty = {
        "ids": np.empty(0, dtype=np.int32),
        "freqs": np.empty(0),
        "bw": np.empty(0),
        "phi": np.empty(0),
        "powers": np.empty(0),
    }
    lone = {**inputs, **empty}
    for kernel in (compiled_kernel, python_kernel):
        gn = kernel.path_noise(*_lone_args(lone), **_lone_kwargs())
        corrected = kernel.path_noise(*_lone_args(lone), **_lone_kwargs(), cfm2=True, cut_phi=0.0)
        np.testing.assert_allclose(corrected[2], TABLE_II[8] * gn[2], rtol=1e-12)
        np.testing.assert_array_equal(corrected[1], gn[1])  # ASE unchanged


def _lone_args(inputs: dict[str, np.ndarray]) -> tuple[object, ...]:
    n = inputs["lengths"].shape[0]
    return (
        SPAN_OFFSETS,
        inputs["lengths"],
        np.full(n, ALPHA),
        np.full(n, 10**0.55),
        np.ones(n),
        np.ones(n),
        np.zeros((n, 0)),
        np.zeros(3, dtype=np.int32),
        inputs["ids"],
        inputs["freqs"],
        inputs["bw"],
        inputs["phi"],
        inputs["powers"],
    )


def _lone_kwargs() -> dict[str, object]:
    return {
        "current_service_id": 0,
        "center_frequency": F_START + SLOT * 152,
        "bandwidth": SLOT * 4,
        "launch_power": 1e-3,
        "include_nli": True,
        "frequency_start": F_START,
        "frequency_slot_bandwidth": SLOT,
        "interferer_psd_actual": False,
    }


@pytest.mark.parametrize("kernel", [compiled_kernel, python_kernel], ids=["compiled", "python"])
def test_cfm2_validates_the_interferer_factors(kernel) -> None:
    inputs = _kernel_inputs(np.random.default_rng(15))
    with pytest.raises(ValueError, match="running_rho is required"):
        _path_noise(kernel, inputs, cfm2=True, cut_phi=1.0)
    with pytest.raises(ValueError, match="expected 16"):
        _summarize(kernel, inputs, cfm2=True, cut_phi=1.0, running_rho=inputs["rho"][:-1])


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

NOBEL_EU = BUILTIN_TOPOLOGY_DIR / "nobel-eu.xml"


@pytest.fixture(scope="module")
def topology() -> TopologyModel:
    return TopologyModel.from_file(NOBEL_EU, k_paths=2, max_span_length_km=80.0)


def _config(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = dict(
        scenario_id="cfm2",
        topology_id="nobel-eu",
        k_paths=2,
        num_spectrum_resources=320,
        nli_include_interferers=True,
        nli_modulation_correction="cfm2",
    )
    values.update(overrides)
    return ScenarioConfig(**values)  # type: ignore[arg-type]


def _long_path(topology: TopologyModel) -> PathRecord:
    return max(topology.paths, key=lambda path: path.hops)


def _request(service_id: int, path: PathRecord, topology: TopologyModel) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=topology.get_node_index(path.node_names[0]),
        destination_id=topology.get_node_index(path.node_names[-1]),
        bit_rate=100,
        arrival_time=float(service_id),
        holding_time=1e6,
    )


def _provision(
    engine: QoTEngine,
    state: RuntimeState,
    topology: TopologyModel,
    service_id: int,
    path: PathRecord,
    start: int,
    modulation: Modulation = QPSK,
) -> None:
    request = _request(service_id, path, topology)
    candidate = engine.build_candidate(request, path, modulation, start, 4)
    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=start,
        service_num_slots=4,
        occupied_slot_start=start,
        occupied_slot_end_exclusive=start + 5,
        modulation=modulation,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )


def _loaded(config: ScenarioConfig, topology: TopologyModel) -> tuple[QoTEngine, RuntimeState, PathRecord]:
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = _long_path(topology)
    for index, (start, modulation) in enumerate(((40, QPSK), (46, QAM16), (58, QAM64), (70, QPSK))):
        _provision(engine, state, topology, 100 + index, path, start, modulation)
    return engine, state, path


def _osnr(engine: QoTEngine, state: RuntimeState, topology: TopologyModel, path: PathRecord, modulation) -> float:
    candidate = engine.build_candidate(_request(0, path, topology), path, modulation, 52, 4)
    return engine.evaluate_candidate(state, candidate).osnr


def test_config_validates_the_correction() -> None:
    assert ScenarioConfig(scenario_id="x", topology_id="ring_4", k_paths=1, num_spectrum_resources=8).nli_modulation_correction == "egn_xci"
    with pytest.raises(ValueError, match="nli_modulation_correction"):
        ScenarioConfig(
            scenario_id="x", topology_id="ring_4", k_paths=1, num_spectrum_resources=8, nli_modulation_correction="egn"
        )
    keys = {
        ScenarioConfig(
            scenario_id="x", topology_id="ring_4", k_paths=1, num_spectrum_resources=8, nli_modulation_correction=mode
        ).runtime_structure_key()
        for mode in ("egn_xci", "cfm2", "gn")
    }
    assert len(keys) == 3


def test_cfm2_gsnr_depends_on_the_lightpath_format(topology: TopologyModel) -> None:
    engine, state, path = _loaded(_config(), topology)
    qpsk, qam16, qam64 = (_osnr(engine, state, topology, path, m) for m in (QPSK, QAM16, QAM64))
    # QPSK is the least Gaussian format (largest Phi): the least SCI.
    assert qpsk > qam16
    assert qpsk > qam64
    # The historical correction ignores the lightpath's own format.
    legacy_engine, legacy_state, _ = _loaded(_config(nli_modulation_correction="egn_xci"), topology)
    legacy = {_osnr(legacy_engine, legacy_state, topology, path, m) for m in (QPSK, QAM16, QAM64)}
    assert len(legacy) == 1


def test_cfm2_corrects_the_gn_model_downwards(topology: TopologyModel) -> None:
    cfm2_engine, cfm2_state, path = _loaded(_config(), topology)
    gn_engine, gn_state, _ = _loaded(_config(nli_modulation_correction="gn"), topology)
    egn_engine, egn_state, _ = _loaded(_config(nli_modulation_correction="egn_xci"), topology)
    for modulation in (QPSK, QAM16, QAM64):
        gn = _osnr(gn_engine, gn_state, topology, path, modulation)
        assert _osnr(cfm2_engine, cfm2_state, topology, path, modulation) > gn
        assert _osnr(egn_engine, egn_state, topology, path, modulation) > gn


def test_cfm2_applies_without_interferers(topology: TopologyModel) -> None:
    path = _long_path(topology)
    osnr = {}
    for mode in ("cfm2", "egn_xci"):
        config = _config(nli_include_interferers=False, nli_modulation_correction=mode)
        osnr[mode] = _osnr(QoTEngine(config, topology), RuntimeState(config, topology), topology, path, QPSK)
    assert osnr["cfm2"] > osnr["egn_xci"]


def test_interferer_dispersion_is_accumulated_from_its_transmitter(topology: TopologyModel) -> None:
    # Find a link that is the first hop of one path and a later hop of another.
    first_hop = {path.link_ids[0]: path for path in topology.paths}
    link_id, upstream = next(
        (path.link_ids[2], path)
        for path in topology.paths
        if len(path.link_ids) > 2 and path.link_ids[2] in first_hop
    )
    local = first_hop[link_id]
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    _provision(engine, state, topology, 1, upstream, 40)
    _provision(engine, state, topology, 2, local, 60)
    cache = engine._link_running_service_arrays(state, link_id)
    rho = cache.rho.reshape(-1, 2)
    assert np.all(rho[:, 0] > rho[:, 1])
    start_km = sum(float(topology.links[link].length_km) for link in upstream.link_ids[:2])
    span_start_km = engine._link_span_start_km[link_id]
    expected = cfm2.rho_interferer(1.0, 21.3 * (span_start_km + start_km))
    np.testing.assert_allclose(rho[:, 0], expected, rtol=1e-3)


def test_cfm2_batch_matches_scalar(topology: TopologyModel) -> None:
    engine, state, path = _loaded(_config(nli_coherence_epsilon=0.05), topology)
    starts = [0, 52, 90, 200]
    for modulation in (QPSK, QAM64):
        batch = engine.summarize_candidate_starts(
            state=state,
            service_id=0,
            path=path,
            modulation=modulation,
            service_num_slots=4,
            candidate_starts=starts,
        )
        scalar = [
            engine.summarize_candidate_at(
                state=state,
                service_id=0,
                path=path,
                modulation=modulation,
                service_slot_start=start,
                service_num_slots=4,
            ).osnr_margin
            for start in starts
        ]
        np.testing.assert_allclose(batch.osnr_margin, scalar, rtol=1e-10)


def test_cfm2_breakdown_and_recompute_are_consistent(topology: TopologyModel) -> None:
    config = _config(roadm_add_osnr_db=33.0)
    engine, state, path = _loaded(config, topology)
    request = _request(0, path, topology)
    candidate = engine.build_candidate(request, path, QAM16, 52, 4)
    evaluated = engine.evaluate_candidate(state, candidate)
    breakdown = engine.noise_breakdown(
        state,
        path=path,
        service_id=0,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
        modulation=QAM16,
    )
    assert breakdown.gsnr_db == pytest.approx(evaluated.osnr, abs=1e-9)
    with pytest.raises(ValueError, match="modulation"):
        engine.noise_breakdown(
            state,
            path=path,
            service_id=0,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
        )
    _provision(engine, state, topology, 0, path, 52, QAM16)
    assert engine.recompute_service(state, 0).osnr == pytest.approx(evaluated.osnr, abs=1e-9)
    assert engine.service_noise_breakdown(state, 0).gsnr_db == pytest.approx(evaluated.osnr, abs=1e-9)


def test_cfm2_environment_episode_runs() -> None:
    env = make_env(
        scenario="nobel_eu_baseline",
        modulations="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        seed=3,
        bit_rates=(10, 40, 100, 400),
        load=300.0,
        num_spectrum_resources=320,
        episode_length=40,
        modulations_to_consider=4,
        k_paths=3,
        overrides={"topology_id": "nobel-eu", "nli_include_interferers": True, "nli_modulation_correction": "cfm2"},
    )
    _, info = env.reset(seed=3)
    accepted = 0
    for _ in range(40):
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()
        _, _, terminated, truncated, info = env.step(select_first_fit_action(mask))
        accepted += info.get("status") == "accepted"
        if terminated or truncated:
            break
    assert accepted > 0
