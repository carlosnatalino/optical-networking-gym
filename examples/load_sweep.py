from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "heuristics" / "load_sweep.py"
_SPEC = importlib.util.spec_from_file_location("optical_networking_gym_v2_examples_heuristics_load_sweep", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load example module at {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

main = _MODULE.main
run_load_sweep = _MODULE.run_load_sweep


__all__ = ["main", "run_load_sweep"]


if __name__ == "__main__":
    main()
