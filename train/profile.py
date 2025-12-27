"""Profiling and timing utilities."""

import json
import os
import platform
import resource
import time
from datetime import datetime
from typing import List, Optional


def get_profile_dir() -> Optional[str]:
    """Get the directory for profiling output based on hostname."""
    if os.environ.get("TRAIN_PROFILE", "1") == "0":
        return None
    root = os.environ.get("TRAIN_ROOT", ".")
    hostname = platform.node().split(".")[0]
    return os.path.join(root, f".profile-{hostname}")


class Timer:
    """Timer for performance and resource usage instrumentation."""

    def __init__(self, name: str, output_path: Optional[str] = None):
        self.name = name
        self.output_path = output_path
        self.durations: List[float] = []
        self.start_time: float = 0.0
        self.start_resources: Optional[resource.struct_rusage] = None

        # Create/Clear the file if path provided
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            # We append in the loop, but typically we might want to start fresh or append?
            # User wants "flush frequently... machine locks up", implies streaming.
            # Let's just append to allow restart resilience, or mode='a'.

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.start_resources = resource.getrusage(resource.RUSAGE_SELF)

    def stop(self, epoch: int = 0, batch: int = 0) -> float:
        end_time = time.perf_counter()
        end_resources = resource.getrusage(resource.RUSAGE_SELF)

        d = end_time - self.start_time
        self.durations.append(d)

        if self.output_path and self.start_resources:
            # RSS is peak so specific diff doesn't mean much for this interval,
            # but tracking the peak growth is useful.
            # Faults are engaging counters, so diff is valid.
            majflt = end_resources.ru_majflt - self.start_resources.ru_majflt
            minflt = end_resources.ru_minflt - self.start_resources.ru_minflt

            entry = {
                "timestamp": datetime.now().isoformat(),
                "name": self.name,
                "epoch": epoch,
                "batch": batch,
                "duration_s": d,
                "maxrss": end_resources.ru_maxrss,
                "majflt": majflt,
                "minflt": minflt,
            }

            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                # os.fsync(f.fileno()) # Dropping fsync for speed/stability if desired, or keeping it?
                # Keeping fsync as per earlier logic, but NO TRY/EXCEPT
                os.fsync(f.fileno())

        return d

    def avg(self) -> float:
        return sum(self.durations) / max(1, len(self.durations))

    def reset(self) -> None:
        self.durations = []


def setup_profiling(script_prefix: str, include_kc_infix: bool = False) -> None:
    """Enable cProfile if TRAIN_PROFILE is set and register exit handler."""
    if os.environ.get("TRAIN_PROFILE", "1") == "0":
        return

    import atexit
    import cProfile
    import pstats
    import sys

    _profiler = cProfile.Profile()
    _profiler.enable()

    def _save_profile() -> None:
        if _profiler:
            _profiler.disable()

            prof_dir = get_profile_dir()
            if prof_dir:
                os.makedirs(prof_dir, exist_ok=True)

                infix = (
                    "_kc" if include_kc_infix and "--pretrain-kc" in sys.argv else ""
                )
                pid = os.getpid()

                # Write .pstats file
                pstats_file = os.path.join(
                    prof_dir, f"{script_prefix}{infix}_{pid}.pstats"
                )
                _profiler.dump_stats(pstats_file)

                # Write human-readable summary
                summary_file = os.path.join(
                    prof_dir, f"{script_prefix}{infix}_{pid}.txt"
                )
                with open(summary_file, "w", encoding="utf-8") as summary_file_handle:
                    stats = pstats.Stats(_profiler, stream=summary_file_handle)

                    stats.sort_stats("cumulative")
                    summary_file_handle.write("TOP 50 BY CUMULATIVE TIME\n")
                    summary_file_handle.write("=" * 80 + "\n")
                    stats.print_stats(50)

                    summary_file_handle.write("\n")

                    stats.sort_stats("calls")
                    summary_file_handle.write("TOP 50 BY INVOCATION COUNT\n")
                    summary_file_handle.write("=" * 80 + "\n")
                    stats.print_stats(50)

    atexit.register(_save_profile)
