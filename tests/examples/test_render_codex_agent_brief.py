from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "render_codex_agent_brief.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


def test_render_codex_agent_brief_enforces_distinct_analyst_and_reviewer_contracts() -> None:
    module = _load_module()
    build_agent_brief = module["build_agent_brief"]

    failure_pack = {
        "scenario_truth": {"load": 400.0},
        "acceptance_target": {"best_heuristic": "first_fit", "best_service_blocking_rate_mean": 0.015},
        "run_summary": {"final_blocking_rate": 0.020},
        "blocking_checkpoints": [{"step_index": 100, "episode_service_blocking_rate": 0.0}],
        "top_mismatch_pairs": [{"chosen_heuristic": "lowest_fragmentation", "reference_winner": "first_fit", "count": 4}],
        "top_decision_basis_mismatches": [{"decision_basis": "balanced_tie_break", "count": 3}],
        "top_fallback_reasons": [],
        "representative_cases": [{"case_id": "ep0-step10"}],
        "method_rules_digest": {
            "sections": {
                "Permanent Rules": ["Judge LLM is the real decision maker."],
                "Future Regret Rule": ["Future regret is offline evidence only."],
            }
        },
    }

    analyst_brief = build_agent_brief(
        failure_pack=failure_pack,
        role="analyst",
        failure_pack_path=Path("failure_pack.json"),
    )
    reviewer_brief = build_agent_brief(
        failure_pack=failure_pack,
        role="reviewer",
        failure_pack_path=Path("failure_pack.json"),
    )

    assert analyst_brief["role"] == "analyst"
    assert "Voce nao pode propor solucao." in analyst_brief["hard_constraints"]
    assert any("onset causal" in item for item in analyst_brief["hard_constraints"])
    assert "recommended_single_change" not in analyst_brief["expected_response_schema"]

    assert reviewer_brief["role"] == "reviewer"
    assert reviewer_brief["expected_response_schema"]["allowed_change_layer"] == "prompt|payload|shortlist|no_change"
    assert any("answer key" in item for item in reviewer_brief["hard_constraints"])
    assert any("uma mudanca unica" in item for item in reviewer_brief["hard_constraints"])
    assert any("antes do bloqueio aparecer" in item for item in reviewer_brief["hard_constraints"])
