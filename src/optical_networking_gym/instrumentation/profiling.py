from __future__ import annotations

import cProfile
from pathlib import Path
import pstats


def write_cprofile_stats(
    profiler: cProfile.Profile,
    output_path: str | Path,
    *,
    sort_by: str = "cumulative",
    top_n: int = 30,
) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as file_handler:
        pstats.Stats(profiler, stream=file_handler).strip_dirs().sort_stats(sort_by).print_stats(top_n)
    return resolved_path


__all__ = ["write_cprofile_stats"]
