"""Network aging: degrade every span at regular intervals during one episode.

Every ``aging_interval`` requests, the fibre loss and the amplifier noise figure
of all spans grow by a fixed step over their initial values
(``env.update_spans``). The traffic state is kept, the established lightpaths
are re-evaluated on the aged physical layer (``refresh_active_services=True``),
and those that no longer meet their threshold are counted as disrupted. The
NLI is self-channel only (``nli_include_interferers=False``), so without aging
no lightpath is ever disrupted and every disruption is caused by the aging.
"""

from __future__ import annotations

import dataclasses

from optical_networking_gym import SpanUpdate, TopologyModel, make_env
from optical_networking_gym.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_LOAD,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_MODULATIONS_TO_CONSIDER,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
)
from optical_networking_gym.heuristics.runtime_heuristics import select_first_fit_action
from optical_networking_gym.utils import experiment_scenarios as scenario_utils


def aged_spans(
    base_topology: TopologyModel,
    *,
    age: int,
    attenuation_step_db_per_km: float,
    noise_figure_step_db: float,
) -> list[SpanUpdate]:
    """Updates that set every span to its initial values plus ``age`` steps."""
    return [
        SpanUpdate(
            link_id=link.id,
            span_index=span_index,
            attenuation_db_per_km=span.attenuation_db_per_km + age * attenuation_step_db_per_km,
            noise_figure_db=span.noise_figure_db + age * noise_figure_step_db,
        )
        for link in base_topology.links
        for span_index, span in enumerate(link.spans)
    ]


def run_episode(
    seed: int = DEFAULT_SEED,
    *,
    episode_length: int = 1000,
    aging_interval: int = 250,
    attenuation_step_db_per_km: float = 0.005,
    noise_figure_step_db: float = 0.25,
) -> dict[str, float | int | str]:
    scenario = scenario_utils.build_nobel_eu_graph_load_scenario(
        topology_id="nobel-eu",
        episode_length=episode_length,
        seed=seed,
        load=DEFAULT_LOAD,
        mean_holding_time=DEFAULT_MEAN_HOLDING_TIME,
        num_spectrum_resources=DEFAULT_NUM_SPECTRUM_RESOURCES,
        k_paths=DEFAULT_K_PATHS,
        launch_power_dbm=DEFAULT_LAUNCH_POWER_DBM,
        modulations_to_consider=DEFAULT_MODULATIONS_TO_CONSIDER,
        measure_disruptions=True,
        drop_on_disruption=False,
    )
    scenario = dataclasses.replace(scenario, nli_include_interferers=False)
    env = make_env(config=scenario)
    _, info = env.reset(seed=seed)
    base_topology = env.simulator.base_topology
    steps = 0

    while True:
        if steps and steps % aging_interval == 0:
            env.update_spans(
                aged_spans(
                    base_topology,
                    age=steps // aging_interval,
                    attenuation_step_db_per_km=attenuation_step_db_per_km,
                    noise_figure_step_db=noise_figure_step_db,
                ),
                refresh_active_services=True,
            )
        action = select_first_fit_action(env.heuristic_context())
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    return {
        "mode": "runtime",
        "policy": "first_fit",
        "steps": steps,
        "physical_layer_version": env.simulator.physical_layer_version,
        "blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
        "disrupted_services_rate": float(info.get("episode_disrupted_services", 0.0)),
    }


def main() -> None:
    print(run_episode())


if __name__ == "__main__":
    main()
