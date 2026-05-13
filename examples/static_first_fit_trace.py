from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "heuristics" / "static_first_fit_trace.py"
_SPEC = importlib.util.spec_from_file_location(
    "optical_networking_gym_v2_examples_heuristics_static_first_fit_trace",
    _MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load example module at {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_env = _MODULE.build_env
main = _MODULE.main
run_episode = _MODULE.run_episode
save_results = _MODULE.save_results


__all__ = ["build_env", "main", "run_episode", "save_results"]


if __name__ == "__main__":
    main()
