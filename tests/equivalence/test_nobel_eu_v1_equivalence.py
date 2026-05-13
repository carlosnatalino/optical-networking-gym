from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import optical_networking_gym.core.osnr as legacy_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from optical_networking_gym.topology import Modulation as LegacyModulation
from optical_networking_gym.topology import get_topology

from optical_networking_gym_v2 import ActionMask, QoTEngine, RuntimeState, ServiceRequest, TopologyModel
from optical_networking_gym_v2.network.allocation import compute_required_slots
from optical_networking_gym_v2.utils import build_nobel_eu_ofc_v1_scenario

from .helpers import build_legacy_action_mask, build_legacy_service


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOBEL_EU_PATH = PROJECT_ROOT / "examples" / "topologies" / "nobel-eu.xml"


def _legacy_modulations() -> tuple[LegacyModulation, ...]:
    return (
        LegacyModulation("BPSK", 100_000, 1, minimum_osnr=3.71925646843142, inband_xt=-14),
        LegacyModulation("QPSK", 2_000, 2, minimum_osnr=6.72955642507124, inband_xt=-17),
        LegacyModulation("8QAM", 1_000, 3, minimum_osnr=10.8453935345953, inband_xt=-20),
        LegacyModulation("16QAM", 500, 4, minimum_osnr=13.2406469649752, inband_xt=-23),
        LegacyModulation("32QAM", 250, 5, minimum_osnr=16.1608982942870, inband_xt=-26),
        LegacyModulation("64QAM", 125, 6, minimum_osnr=19.0134649345090, inband_xt=-29),
    )


def _build_legacy_env() -> QRMSAEnv:
    topology = get_topology(
        str(NOBEL_EU_PATH),
        topology_name="nobel-eu",
        modulations=_legacy_modulations(),
        max_span_length=80.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=3,
    )
    return QRMSAEnv(
        topology=topology,
        seed=10,
        allow_rejection=True,
        load=300.0,
        episode_length=1000,
        num_spectrum_resources=320,
        launch_power_dbm=0.0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8 / 1565e-9,
        bandwidth=320 * 12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10, 40, 100, 400),
        margin=0.0,
        measure_disruptions=False,
        file_name="",
        k_paths=3,
        modulations_to_consider=6,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=False,
        qot_constraint="ASE+NLI",
        rl_mode=True,
        reset=False,
    )


def _first_request() -> ServiceRequest:
    return ServiceRequest(
        request_index=0,
        service_id=0,
        source_id=22,
        destination_id=3,
        bit_rate=400,
        arrival_time=10.701445579528809,
        holding_time=17117.12890625,
    )


def _second_request() -> ServiceRequest:
    return ServiceRequest(
        request_index=1,
        service_id=1,
        source_id=1,
        destination_id=22,
        bit_rate=10,
        arrival_time=24.90134048461914,
        holding_time=4339.18505859375,
    )


def test_nobel_eu_step1_mask_and_qot_match_legacy_ofc_v1() -> None:
    legacy_env = _build_legacy_env()
    legacy_env.reset(seed=10)

    config = build_nobel_eu_ofc_v1_scenario(seed=10, measure_disruptions=False)
    topology = TopologyModel.from_file(
        NOBEL_EU_PATH,
        topology_id=config.topology_id,
        k_paths=config.k_paths,
        max_span_length_km=config.max_span_length_km,
        default_attenuation_db_per_km=config.default_attenuation_db_per_km,
        default_noise_figure_db=config.default_noise_figure_db,
    )
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    builder = ActionMask(config, topology, engine)

    first_request = _first_request()
    second_request = _second_request()

    legacy_first = build_legacy_service(first_request, legacy_env)
    legacy_env.current_service = legacy_first
    legacy_path = legacy_env.k_shortest_paths[legacy_first.source, legacy_first.destination][0]
    legacy_modulation = _legacy_modulations()[4]
    first_service_num_slots = legacy_env.get_number_slots(legacy_first, legacy_modulation)
    legacy_first.path = legacy_path
    legacy_first.initial_slot = 0
    legacy_first.number_slots = first_service_num_slots
    legacy_first.center_frequency = legacy_env.frequency_start + legacy_env.frequency_slot_bandwidth * (
        first_service_num_slots / 2
    )
    legacy_first.bandwidth = legacy_env.frequency_slot_bandwidth * first_service_num_slots
    legacy_first.launch_power = legacy_env.launch_power
    legacy_first.current_modulation = legacy_modulation
    legacy_env._provision_path(legacy_path, 0, first_service_num_slots)

    v2_path = topology.get_paths_by_ids(first_request.source_id, first_request.destination_id)[0]
    first_candidate = engine.build_candidate(
        request=first_request,
        path=v2_path,
        modulation=config.modulations[4],
        service_slot_start=0,
        service_num_slots=first_service_num_slots,
    )
    state.apply_provision(
        request=first_candidate.request,
        path=v2_path,
        service_slot_start=first_candidate.service_slot_start,
        service_num_slots=first_candidate.service_num_slots,
        occupied_slot_start=first_candidate.service_slot_start,
        occupied_slot_end_exclusive=first_candidate.service_slot_start + first_candidate.service_num_slots + 1,
        modulation=first_candidate.modulation,
        center_frequency=first_candidate.center_frequency,
        bandwidth=first_candidate.bandwidth,
        launch_power=first_candidate.launch_power,
    )

    legacy_env.current_service = build_legacy_service(second_request, legacy_env)
    legacy_mask = build_legacy_action_mask(legacy_env)
    mask = builder.build(state, second_request)

    assert mask is not None
    assert np.array_equal(mask, legacy_mask)
    assert int(mask[:-1].sum()) == 4680
    assert int(mask[328]) == 1
    assert int(np.flatnonzero(mask[:-1])[0]) == 328

    analysis = builder.analysis_engine.build(state, second_request)
    selection = builder.decode_action(328, state, second_request)
    modulation_offset = analysis.modulation_offset_for_index(selection.modulation_index)
    assert modulation_offset == 1

    service_num_slots = int(
        analysis.required_slots_by_path_mod[selection.path_index, modulation_offset]
    )
    summary = engine.summarize_candidate_at(
        state=state,
        service_id=second_request.service_id,
        path=topology.get_paths_by_ids(second_request.source_id, second_request.destination_id)[
            selection.path_index
        ],
        modulation=config.modulations[selection.modulation_index],
        service_slot_start=selection.initial_slot,
        service_num_slots=service_num_slots,
    )

    legacy_second = legacy_env.current_service
    legacy_second.path = legacy_env.k_shortest_paths[legacy_second.source, legacy_second.destination][0]
    legacy_second.initial_slot = selection.initial_slot
    legacy_second.number_slots = compute_required_slots(
        bit_rate=second_request.bit_rate,
        spectral_efficiency=legacy_modulation.spectral_efficiency,
        channel_width=config.channel_width,
    )
    legacy_second.center_frequency = legacy_env.frequency_start + legacy_env.frequency_slot_bandwidth * (
        legacy_second.initial_slot + (legacy_second.number_slots / 2)
    )
    legacy_second.bandwidth = legacy_env.frequency_slot_bandwidth * legacy_second.number_slots
    legacy_second.launch_power = legacy_env.launch_power
    legacy_second.current_modulation = legacy_modulation
    legacy_gsnr, legacy_ase, legacy_nli = legacy_osnr.calculate_osnr(
        legacy_env,
        legacy_second,
        legacy_env.qot_constraint,
    )

    assert summary.meets_threshold is True
    assert summary.osnr == pytest.approx(legacy_gsnr, abs=1e-6)
    assert summary.ase == pytest.approx(legacy_ase, abs=1e-6)
    assert summary.nli == pytest.approx(legacy_nli, abs=1e-6)
    assert summary.osnr_margin == pytest.approx(legacy_gsnr - legacy_modulation.minimum_osnr, abs=1e-6)
