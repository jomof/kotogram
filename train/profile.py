"""Profiling and timing utilities."""

import cProfile
import json
import os
import platform
import pstats
import re
import resource
import time
from datetime import datetime
from typing import Any, List, Optional


def profiling_enabled() -> bool:
    """Check if profiling is enabled via TRAIN_PROFILE environment variable."""
    return os.environ.get("TRAIN_PROFILE", "0") == "1"


def get_profile_dir() -> Optional[str]:
    """Get the directory for profiling output based on hostname."""
    if not profiling_enabled():
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
    if not profiling_enabled():
        return

    import atexit
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


class PhaseTimer:
    """Timer for measuring phases in scripts."""

    def __init__(self, console: Any, profile_dir: Optional[str] = None):
        self.console = console
        self.profile_dir = profile_dir
        self.pid = os.getpid()
        self.last = time.perf_counter()
        self.phase_idx = 1

        self.profiler: Optional[cProfile.Profile] = None
        if self.profile_dir:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def mark(self, phase_name: str) -> None:
        """Mark the end of a phase and print/dump stats."""
        now = time.perf_counter()
        elapsed = now - self.last
        self.last = now
        self.console.print(
            f"[dim]Stats: Phase '{phase_name}' took {elapsed:.1f}s[/dim]"
        )

        if self.profiler and self.profile_dir:
            self.profiler.disable()

            self._dump_stats(phase_name, elapsed)

            # Start new profiler/tracer for next phase
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.phase_idx += 1

    def stop(self, phase_name: str = "Final") -> None:
        """Stop the timer and dump final stats."""
        self.mark(phase_name)
        if self.profiler:
            self.profiler.disable()
            self.profiler = None

    def _dump_stats(self, phase_name: str, elapsed: float) -> None:
        if not self.profile_dir:
            return

        # Sanitize phase name for filename
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", phase_name).lower()
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")

        elapsed_str = f"{elapsed:.1f}s"
        base_path = os.path.join(
            self.profile_dir,
            f"label_p{self.phase_idx}_{clean_name}_{elapsed_str}_{self.pid}",
        )

        # Write .pstats
        if self.profiler:  # Check again for type safety
            self.profiler.dump_stats(f"{base_path}.pstats")

            # Write .txt summary
            with open(f"{base_path}.txt", "w", encoding="utf-8") as f:
                stats = pstats.Stats(self.profiler, stream=f)
                stats.sort_stats("cumulative")
                f.write(f"PHASE: {phase_name} ({elapsed_str})\n")
                f.write("=" * 80 + "\n")
                f.write("TOP 50 BY CUMULATIVE TIME\n")
                f.write("-" * 80 + "\n")
                stats.print_stats(50)
                f.write("\n")
                stats.sort_stats("calls")
                f.write("TOP 50 BY INVOCATION COUNT\n")
                f.write("-" * 80 + "\n")
                stats.print_stats(50)
