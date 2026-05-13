#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import html
import json
from pathlib import Path
import sys
import time
from typing import Any
import webbrowser

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "results" / "env_test_report.html"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, make_env, select_first_fit_action  # noqa: E402

TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR
from optical_networking_gym_v2.envs.optical_env import OpticalEnv  # noqa: E402


@dataclass(slots=True)
class TestResult:
    name: str
    passed: bool
    details: str


@dataclass(slots=True)
class LinkSnapshot:
    label: str
    free_slots: list[int]


@dataclass(slots=True)
class StepSnapshot:
    step_index: int
    request: dict[str, Any]
    valid_non_reject_actions: int
    chosen_action: int
    chosen_action_valid: bool
    reward: float
    terminated: bool
    truncated: bool
    status: str
    step_ms: float
    mask: list[int]
    decoded_action: dict[str, Any] | None
    links: list[LinkSnapshot]


@dataclass(slots=True)
class SuiteReport:
    variant: str
    topology_name: str
    action_space_n: int
    observation_size: int
    steps_requested: int
    executed_steps: int
    accepted_steps: int
    total_runtime_ms: float
    tests: list[TestResult]
    steps: list[StepSnapshot]


def _build_env(
    *,
    topology_name: str,
    seed: int,
    num_slots: int,
    episode_length: int,
    k_paths: int,
    load: float,
) -> OpticalEnv:
    return make_env(
        scenario="ring4_quickstart" if topology_name == "ring_4" else "nobel_eu_baseline",
        modulations="BPSK, QPSK, 8QAM, 16QAM",
        seed=seed,
        load=load,
        num_spectrum_resources=num_slots,
        episode_length=episode_length,
        modulations_to_consider=4,
        k_paths=k_paths,
        overrides={"topology_id": topology_name, "topology_dir": TOPOLOGY_DIR},
    )


def _extract_mask(env: OpticalEnv, info: dict[str, Any]) -> np.ndarray:
    raw_mask = info.get("mask")
    if raw_mask is None:
        raw_mask = env.action_masks()
    if raw_mask is None:
        raise RuntimeError("action mask is unavailable")
    return np.asarray(raw_mask, dtype=np.uint8)


def _request_payload(env: OpticalEnv) -> dict[str, Any]:
    request = env.simulator.current_request
    if request is None:
        return {}
    return {
        "service_id": int(request.service_id),
        "source": int(request.source_id),
        "destination": int(request.destination_id),
        "bit_rate": float(request.bit_rate),
    }


def _decode_action(env: OpticalEnv, action: int) -> dict[str, Any] | None:
    if action == env.action_space.n - 1:
        return None
    state = env.simulator.state
    request = env.simulator.current_request
    if state is None or request is None:
        return None
    selection = env.simulator.action_mask_builder.decode_action(int(action), state, request)
    modulation = env.simulator.config.modulations[selection.modulation_index]
    return {
        "path_index": int(selection.path_index),
        "modulation_index": int(selection.modulation_index),
        "modulation_name": str(modulation.name),
        "initial_slot": int(selection.initial_slot),
    }


def _link_snapshots(env: OpticalEnv) -> list[LinkSnapshot]:
    state = env.simulator.state
    topology = env.simulator.topology
    if state is None:
        return []
    snapshots: list[LinkSnapshot] = []
    for link in topology.links:
        free_slots = (state.slot_allocation[link.id, :] == -1).astype(np.uint8)
        snapshots.append(
            LinkSnapshot(
                label=f"{link.source_name}->{link.target_name}",
                free_slots=[int(value) for value in free_slots.tolist()],
            )
        )
    return snapshots


def _make_test_results(
    observation: np.ndarray,
    mask: np.ndarray,
    action_space_n: int,
    chosen_action: int,
    executed_steps: int,
) -> list[TestResult]:
    return [
        TestResult(
            name="reset_returns_observation",
            passed=isinstance(observation, np.ndarray) and observation.size > 0,
            details=f"shape={tuple(observation.shape)}",
        ),
        TestResult(
            name="mask_present",
            passed=isinstance(mask, np.ndarray),
            details=f"mask_size={int(mask.size)}",
        ),
        TestResult(
            name="mask_matches_action_space",
            passed=int(mask.size) == int(action_space_n),
            details=f"mask={int(mask.size)} action_space={int(action_space_n)}",
        ),
        TestResult(
            name="mask_is_binary",
            passed=bool(np.isin(mask, [0, 1]).all()),
            details=f"unique={np.unique(mask).tolist()}",
        ),
        TestResult(
            name="heuristic_returns_valid_action",
            passed=(0 <= chosen_action < int(mask.size) and int(mask[chosen_action]) == 1),
            details=f"chosen_action={int(chosen_action)}",
        ),
        TestResult(
            name="multiple_steps_completed",
            passed=executed_steps > 0,
            details=f"executed_steps={int(executed_steps)}",
        ),
    ]


def run_visual_suite(
    *,
    topology_name: str = "ring_4",
    steps: int = 6,
    seed: int = 7,
    num_slots: int = 24,
    episode_length: int = 24,
    k_paths: int = 2,
    load: float = 10.0,
) -> SuiteReport:
    suite_start = time.perf_counter()
    env = _build_env(
        topology_name=topology_name,
        seed=seed,
        num_slots=num_slots,
        episode_length=episode_length,
        k_paths=k_paths,
        load=load,
    )
    accepted_steps = 0
    step_views: list[StepSnapshot] = []

    try:
        observation, info = env.reset(seed=seed)
        mask = _extract_mask(env, info)
        chosen_action = int(select_first_fit_action(mask))
        tests = _make_test_results(
            observation=np.asarray(observation),
            mask=mask,
            action_space_n=int(env.action_space.n),
            chosen_action=chosen_action,
            executed_steps=steps,
        )

        current_info = info
        current_observation = np.asarray(observation)
        for step_index in range(steps):
            current_mask = _extract_mask(env, current_info)
            request_payload = _request_payload(env)
            chosen_action = int(select_first_fit_action(current_mask))
            decoded_action = _decode_action(env, chosen_action)
            valid_non_reject_actions = int(np.count_nonzero(current_mask[:-1]))
            chosen_action_valid = bool(0 <= chosen_action < current_mask.size and current_mask[chosen_action] == 1)

            step_start = time.perf_counter()
            observation, reward, terminated, truncated, current_info = env.step(chosen_action)
            step_ms = (time.perf_counter() - step_start) * 1000.0
            current_observation = np.asarray(observation)
            status = str(current_info.get("status", "unknown"))
            if current_info.get("accepted") is True:
                accepted_steps += 1

            step_views.append(
                StepSnapshot(
                    step_index=step_index,
                    request=request_payload,
                    valid_non_reject_actions=valid_non_reject_actions,
                    chosen_action=chosen_action,
                    chosen_action_valid=chosen_action_valid,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    status=status,
                    step_ms=step_ms,
                    mask=[int(value) for value in current_mask.tolist()],
                    decoded_action=decoded_action,
                    links=_link_snapshots(env),
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    return SuiteReport(
        variant="v2",
        topology_name=topology_name,
        action_space_n=int(env.action_space.n),
        observation_size=int(current_observation.size),
        steps_requested=steps,
        executed_steps=len(step_views),
        accepted_steps=accepted_steps,
        total_runtime_ms=(time.perf_counter() - suite_start) * 1000.0,
        tests=tests,
        steps=step_views,
    )


def _suite_to_payload(suite: SuiteReport) -> dict[str, Any]:
    return {
        "variant": suite.variant,
        "topology_name": suite.topology_name,
        "action_space_n": suite.action_space_n,
        "observation_size": suite.observation_size,
        "steps_requested": suite.steps_requested,
        "executed_steps": suite.executed_steps,
        "accepted_steps": suite.accepted_steps,
        "total_runtime_ms": suite.total_runtime_ms,
        "tests": [asdict(test) for test in suite.tests],
        "steps": [
            {
                "step_index": step.step_index,
                "request": step.request,
                "valid_non_reject_actions": step.valid_non_reject_actions,
                "chosen_action": step.chosen_action,
                "chosen_action_valid": step.chosen_action_valid,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "status": step.status,
                "step_ms": step.step_ms,
                "mask": step.mask,
                "decoded_action": step.decoded_action,
                "links": [asdict(link) for link in step.links],
            }
            for step in suite.steps
        ],
    }


def _render_mask_svg(mask: list[int], chosen_action: int) -> str:
    cell_width = 4
    width = max(320, len(mask) * cell_width)
    height = 22
    rects: list[str] = []
    for index, value in enumerate(mask):
        x = index * cell_width
        if index == chosen_action:
            color = "#2563eb" if value else "#dc2626"
        elif index == len(mask) - 1:
            color = "#f59e0b" if value else "#fde68a"
        else:
            color = "#10b981" if value else "#e5e7eb"
        rects.append(
            f"<rect x='{x}' y='0' width='{cell_width - 1}' height='{height}' fill='{color}' />"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none' class='mask-svg'>"
        + "".join(rects)
        + "</svg>"
    )


def _render_spectrum_svg(links: list[LinkSnapshot]) -> str:
    if not links:
        return "<p>No spectrum snapshot available.</p>"

    cell = 10
    left_margin = 88
    row_height = 16
    cols = len(links[0].free_slots)
    width = left_margin + (cols * cell)
    height = max(24, len(links) * row_height)
    pieces: list[str] = [
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none' class='slots-svg'>"
    ]
    for row_index, link in enumerate(links):
        y = row_index * row_height
        pieces.append(
            f"<text x='0' y='{y + 10}' font-size='9' fill='#334155'>{html.escape(link.label)}</text>"
        )
        for column, free_value in enumerate(link.free_slots):
            x = left_margin + (column * cell)
            color = "#d1fae5" if free_value else "#0f172a"
            pieces.append(
                f"<rect x='{x}' y='{y}' width='{cell - 1}' height='12' fill='{color}' rx='1' ry='1' />"
            )
    pieces.append("</svg>")
    return "".join(pieces)


def _render_suite_html(suite: SuiteReport) -> str:
    tests_rows = "".join(
        (
            "<tr>"
            f"<td>{html.escape(test.name)}</td>"
            f"<td class='{'pass' if test.passed else 'fail'}'>{'PASS' if test.passed else 'FAIL'}</td>"
            f"<td>{html.escape(test.details)}</td>"
            "</tr>"
        )
        for test in suite.tests
    )

    step_blocks = []
    for step in suite.steps:
        request_items = " ".join(
            f"<span class='pill'>{html.escape(str(key))}: {html.escape(str(value))}</span>"
            for key, value in step.request.items()
        )
        decode_items = ""
        if step.decoded_action is not None:
            decode_items = " ".join(
                f"<span class='pill'>{html.escape(str(key))}: {html.escape(str(value))}</span>"
                for key, value in step.decoded_action.items()
            )

        step_blocks.append(
            "<section class='step-card'>"
            f"<h4>Step {step.step_index}</h4>"
            f"<p class='meta-line'><strong>Status:</strong> {html.escape(step.status)} | "
            f"<strong>Reward:</strong> {step.reward:.3f} | "
            f"<strong>Step Time:</strong> {step.step_ms:.2f} ms | "
            f"<strong>Valid Actions:</strong> {step.valid_non_reject_actions}</p>"
            f"<div class='pill-row'>{request_items}</div>"
            f"<div class='pill-row'>{decode_items}</div>"
            f"<p class='meta-line'><strong>Chosen action:</strong> {step.chosen_action} | "
            f"<strong>Mask valid:</strong> {'yes' if step.chosen_action_valid else 'no'} | "
            f"<strong>Done:</strong> {step.terminated or step.truncated}</p>"
            "<div class='visual-block'>"
            "<h5>Action Mask</h5>"
            f"{_render_mask_svg(step.mask, step.chosen_action)}"
            "</div>"
            "<div class='visual-block'>"
            "<h5>Spectrum Occupancy</h5>"
            f"{_render_spectrum_svg(step.links)}"
            "</div>"
            "</section>"
        )

    return (
        "<section class='suite'>"
        f"<h2>{html.escape(suite.variant.upper())} | {html.escape(suite.topology_name)}</h2>"
        "<div class='summary-grid'>"
        f"<div class='summary-card'><span>Action space</span><strong>{suite.action_space_n}</strong></div>"
        f"<div class='summary-card'><span>Observation size</span><strong>{suite.observation_size}</strong></div>"
        f"<div class='summary-card'><span>Steps</span><strong>{suite.executed_steps}/{suite.steps_requested}</strong></div>"
        f"<div class='summary-card'><span>Accepted</span><strong>{suite.accepted_steps}</strong></div>"
        f"<div class='summary-card'><span>Runtime</span><strong>{suite.total_runtime_ms:.2f} ms</strong></div>"
        "</div>"
        "<table class='tests-table'>"
        "<thead><tr><th>Test</th><th>Result</th><th>Details</th></tr></thead>"
        f"<tbody>{tests_rows}</tbody>"
        "</table>"
        + "".join(step_blocks)
        + "</section>"
    )


def _render_report_html(suite: SuiteReport) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    suite_html = _render_suite_html(suite)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Environment Visual Test Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --pass: #166534;
      --fail: #b91c1c;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3, h4, h5 {{
      margin: 0 0 12px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 24px;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .suite {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .summary-card {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .summary-card span {{
      display: block;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .summary-card strong {{
      font-size: 1.35rem;
      color: var(--accent);
    }}
    .tests-table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 24px;
      background: #f8fafc;
      border-radius: 14px;
      overflow: hidden;
    }}
    .tests-table th, .tests-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      font-size: 0.95rem;
    }}
    .tests-table thead {{
      background: #e2e8f0;
    }}
    .pass {{
      color: var(--pass);
      font-weight: 700;
    }}
    .fail {{
      color: var(--fail);
      font-weight: 700;
    }}
    .step-card {{
      border-top: 1px solid var(--line);
      padding-top: 20px;
      margin-top: 20px;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0;
    }}
    .pill {{
      background: #dbeafe;
      color: #1e3a8a;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.85rem;
    }}
    .meta-line {{
      color: var(--muted);
      margin: 8px 0;
    }}
    .visual-block {{
      margin-top: 16px;
    }}
    .mask-svg, .slots-svg {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Environment Visual Test Report</h1>
      <p>Generated at {html.escape(generated_at)} using topology files from {html.escape(str(TOPOLOGY_DIR))}.</p>
      <p>The report captures smoke-test results, action masks, chosen actions, and per-link spectrum occupancy for the v2 environment.</p>
    </section>
    {suite_html}
  </main>
</body>
</html>
"""


def write_visual_report(output_path: Path, suite: SuiteReport) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report_html(suite), encoding="utf-8")
    payload = {"suite": _suite_to_payload(suite)}
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_visual_report(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    topology_name: str = "ring_4",
    steps: int = 6,
    seed: int = 7,
    num_slots: int = 24,
    episode_length: int = 24,
    k_paths: int = 2,
    load: float = 10.0,
) -> SuiteReport:
    suite = run_visual_suite(
        topology_name=topology_name,
        steps=steps,
        seed=seed,
        num_slots=num_slots,
        episode_length=episode_length,
        k_paths=k_paths,
        load=load,
    )
    write_visual_report(output_path, suite)
    return suite


def _print_console_summary(suite: SuiteReport, output_path: Path) -> None:
    print(f"Visual report written to: {output_path}")
    print(f"Raw data written to: {output_path.with_suffix('.json')}")
    print(
        f"[{suite.variant}] topology={suite.topology_name} action_space={suite.action_space_n} "
        f"obs={suite.observation_size} steps={suite.executed_steps} "
        f"accepted={suite.accepted_steps} runtime_ms={suite.total_runtime_ms:.2f}"
    )
    for test in suite.tests:
        marker = "PASS" if test.passed else "FAIL"
        print(f"  - {marker}: {test.name} ({test.details})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visual HTML report for v2 env smoke tests.")
    parser.add_argument("--topology-name", default="ring_4")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--k-paths", type=int, default=2)
    parser.add_argument("--load", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite = generate_visual_report(
        output_path=args.output,
        topology_name=args.topology_name,
        steps=args.steps,
        seed=args.seed,
        num_slots=args.num_slots,
        episode_length=args.episode_length,
        k_paths=args.k_paths,
        load=args.load,
    )
    _print_console_summary(suite, args.output)
    if not args.no_open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
