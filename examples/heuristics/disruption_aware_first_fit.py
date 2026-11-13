"""Disruption-aware first-fit versus plain first-fit.

Establishing a lightpath adds nonlinear interference to the services that
share its links, which can push some of them below the OSNR threshold of their
modulation format. With ``measure_disruptions=True`` the simulator detects and
counts these disruptions after every admission, and ``drop_on_disruption=True``
also tears the disrupted services down. The simulator does not refuse the new
request: whether to accept a request that would disrupt established services
is a decision of the provisioning algorithm, not of the environment.

This example compares two policies on the same traffic:

- ``first_fit``: the first action allowed by the action mask, which checks the
  QoT of the new service only;
- ``disruption_aware``: first-fit that also skips every candidate that would
  push an established service below its threshold, and rejects the request if
  no candidate is left.

The disruption-aware policy tries each candidate against the network state, so
it is much slower than plain first-fit, especially at high load.
"""

from __future__ import annotations

import argparse

from optical_networking_gym import (
    build_scenario,
    make_env,
    select_disruption_aware_first_fit_action,
    select_first_fit_action,
)

POLICIES = ("first_fit", "disruption_aware")


def run_episode(
    seed: int = 7,
    *,
    policy: str = "disruption_aware",
    load: float = 210.0,
    episode_length: int = 200,
    drop_on_disruption: bool = True,
) -> dict[str, float | int | str]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    scenario = build_scenario(
        "jocn_benchmark",
        topology_id="nobel-eu",
        load=load,
        episode_length=episode_length,
        margin=0.0,
        measure_disruptions=True,
        drop_on_disruption=drop_on_disruption,
    )
    env = make_env(scenario=scenario, seed=seed)
    _, info = env.reset(seed=seed)
    steps = 0

    while True:
        if policy == "first_fit":
            action = select_first_fit_action(env.action_masks())
        else:
            action = select_disruption_aware_first_fit_action(env.heuristic_context())
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    services_accepted = int(info.get("episode_services_accepted", 0))
    disrupted_rate = float(info.get("episode_disrupted_services", 0.0))
    return {
        "policy": policy,
        "steps": steps,
        "load": load,
        "blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
        "services_accepted": services_accepted,
        "disrupted_services": round(disrupted_rate * services_accepted),
        "disrupted_services_rate": disrupted_rate,
        "dropped_services": int(info.get("disrupted_or_dropped_services", 0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load", type=float, default=210.0)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument(
        "--keep-disrupted",
        dest="drop_on_disruption",
        action="store_false",
        help="count disrupted services but keep them established",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for policy in args.policies:
        summary = run_episode(
            seed=args.seed,
            policy=policy,
            load=args.load,
            episode_length=args.episode_length,
            drop_on_disruption=args.drop_on_disruption,
        )
        print(
            f"{summary['policy']:>16}: blocking {summary['blocking_rate']:.4f} | "
            f"accepted {summary['services_accepted']} | "
            f"disrupted {summary['disrupted_services']} | "
            f"dropped {summary['dropped_services']}"
        )


if __name__ == "__main__":
    main()
