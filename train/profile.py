"""Profiling and timing utilities."""

import cProfile
import json
import os
import platform
import pstats
import re
import resource
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, List, Optional

import memray


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
    """Consolidated timer for performance, resource usage, and profiling."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        name: str,
        output_path: Optional[str] = None,
        profile_dir: Optional[str] = None,
        console: Any = None,
    ):
        self.name = name
        self.output_path = output_path
        self.profile_dir = profile_dir
        self.console = console

        self.durations: List[float] = []
        self.start_time: float = 0.0
        self.start_resources: Optional[resource.struct_rusage] = None

        # Profiling state
        self.profiler: Optional[cProfile.Profile] = None
        self.memray_tracker: Any = None
        self.memray_file: Optional[str] = None

        # Phase tracking
        self.phase_idx = 1
        self.last_phase_name: Optional[str] = None

        # Create/Clear the simplified log file if path provided
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Initialize global profilers immediately if this is a phase-style timer
        if self.profile_dir:
            os.makedirs(self.profile_dir, exist_ok=True)

    def start(self) -> None:
        """Start timing a block."""
        self.start_time = time.perf_counter()
        self.start_resources = resource.getrusage(resource.RUSAGE_SELF)

        # Start heavy profilers if configured
        if self.profile_dir and profiling_enabled():
            # cProfile
            self.profiler = cProfile.Profile()
            self.profiler.enable()

            # memray
            if memray:  # pylint: disable=using-constant-test
                pid = os.getpid()
                timestamp = int(time.time() * 1000)
                clean_name = self._clean_name(self.name)
                self.memray_file = os.path.join(
                    self.profile_dir,
                    f"{clean_name}_{timestamp}_{pid}.bin",
                )
                self.memray_tracker = memray.Tracker(self.memray_file)
                self.memray_tracker.__enter__()  # pylint: disable=unnecessary-dunder-call

    def stop(
        self, epoch: int = 0, batch: int = 0, phase_name: Optional[str] = None
    ) -> float:
        """Stop timing a block and record stats."""
        end_time = time.perf_counter()
        end_resources = resource.getrusage(resource.RUSAGE_SELF)

        d = end_time - self.start_time
        self.durations.append(d)

        # Stop profilers
        if self.memray_tracker:
            self.memray_tracker.__exit__(None, None, None)
            self.memray_tracker = None
        if self.profiler:
            self.profiler.disable()

        # 1. Write simple usage stats (JSONL)
        if self.output_path and self.start_resources:
            majflt = end_resources.ru_majflt - self.start_resources.ru_majflt
            minflt = end_resources.ru_minflt - self.start_resources.ru_minflt

            # If we had a memray file, we can't easily get peak without processing,
            # so we stick to RSS from getrusage for the lightweight log.
            entry = {
                "timestamp": datetime.now().isoformat(),
                "name": phase_name or self.name,
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
                os.fsync(f.fileno())

        # 2. Write heavy profiles (Artifacts)
        if self.profile_dir and (self.profiler or self.memray_file):
            self._save_heavy_stats(phase_name or self.name, d)
            self.profiler = None
            self.memray_file = None

        if self.console and phase_name:
            self.console.print(f"[dim]Stats: Phase '{phase_name}' took {d:.1f}s[/dim]")

        return d

    def mark(self, phase_name: str) -> None:
        """Sequential phase marker (stops previous, starts new)."""
        # Stop previous if running
        if self.start_time > 0:
            name = self.last_phase_name or f"Phase_{self.phase_idx}"
            self.stop(phase_name=name)

        # Setup for next
        self.name = phase_name  # Update current name for the next block
        self.last_phase_name = phase_name
        self.phase_idx += 1

        # Start new
        self.start()

    def avg(self) -> float:
        return sum(self.durations) / max(1, len(self.durations))

    def reset(self) -> None:
        self.durations = []

    def _clean_name(self, name: str) -> str:
        return _clean_name(name)

    def _save_heavy_stats(self, name: str, elapsed: float) -> None:
        """Save cProfile and memray stats to disk."""
        save_combined_stats(
            self.profile_dir,
            name,
            elapsed,
            self.profiler,
            self.memray_file,
        )


class PhaseTimer(Timer):
    """Backward compatible wrapper for sequential phases."""

    def __init__(self, console: Any, profile_dir: Optional[str] = None):
        super().__init__("start", profile_dir=profile_dir, console=console)
        self.start()

    def mark(self, phase_name: str) -> None:
        # For PhaseTimer, 'phase_name' labels the COMPLETED phase.
        # Timer.stop() takes phase_name to label the completed interval.
        self.stop(phase_name=phase_name)
        # Start next phase
        self.name = "phase_next"  # Placeholder until next mark
        self.start()

    def stop(self, phase_name: str = "Final") -> float:  # type: ignore # pylint: disable=arguments-differ
        return super().stop(phase_name=phase_name)


def _clean_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    return re.sub(r"_+", "_", clean).strip("_")


def save_combined_stats(
    profile_dir: Optional[str],
    name: str,
    elapsed: float,
    profiler: Optional[cProfile.Profile],
    memray_file: Optional[str],
) -> None:
    # pylint: disable=too-many-locals
    """Save cProfile and memray stats to disk (shared logic)."""
    if not profile_dir:
        return

    pid = os.getpid()
    clean = _clean_name(name)
    elapsed_str = f"{elapsed:.1f}s"
    base_name = f"{clean}_{elapsed_str}_{pid}"
    base_path = os.path.join(profile_dir, base_name)

    # 1. Save cProfile .pstats
    if profiler:
        pstats_file = f"{base_path}.pstats"
        profiler.dump_stats(pstats_file)

    # 2. Text Report (Combine cProfile + Memray)
    txt_file = f"{base_path}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"PROFILE: {name} ({elapsed_str})\n")
        f.write("=" * 80 + "\n")

        # cProfile Section
        if profiler:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative")
            f.write("TOP 50 BY CUMULATIVE TIME (cProfile)\n")
            f.write("-" * 80 + "\n")
            stats.print_stats(50)
            f.write("\n")

        # Memray Section
        if memray_file and os.path.exists(memray_file):
            f.write("=" * 80 + "\n")
            f.write("MEMORY REPORT (memray tree)\n")
            f.write("-" * 80 + "\n")
            f.flush()

            # Execute memray tree (better context than summary)
            # Execute memray stats (cleaner text output than tree/summary)
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("MEMORY REPORT (memray stats)\n")
            f.write("-" * 80 + "\n")
            f.flush()

            env = os.environ.copy()
            env["COLUMNS"] = "200"

            cmd_stats = [
                sys.executable,
                "-m",
                "memray",
                "stats",
                "-n",
                "20",
                memray_file,
            ]
            res = subprocess.run(cmd_stats, stdout=f, stderr=f, env=env, check=False)

            if res.returncode == 0:
                f.write("\n")
                f.write(f"Raw memray file: {os.path.basename(memray_file)}\n")
                # Cleanup huge bin file if successful
                if os.path.exists(memray_file):
                    os.remove(memray_file)
            else:
                f.write(
                    f"Error generating memray report: return code {res.returncode}\n"
                )


def setup_profiling(script_prefix: str, include_kc_infix: bool = False) -> None:
    """Enable cProfile if TRAIN_PROFILE is set and register exit handler."""
    if not profiling_enabled():
        return

    import atexit

    # Enable cProfile
    _profiler = cProfile.Profile()
    _profiler.enable()

    # Enable memray
    _memray_tracker = None
    _memray_file = None
    prof_dir = get_profile_dir()

    if memray and prof_dir:
        os.makedirs(prof_dir, exist_ok=True)
        pid = os.getpid()
        timestamp = int(time.time() * 1000)

        infix = "_kc" if include_kc_infix and "--pretrain-kc" in sys.argv else ""
        # Use a temporary name until we know elapsed time?
        # actually we can just name it broadly.
        _memray_file = os.path.join(
            prof_dir,
            f"{script_prefix}{infix}_global_{timestamp}_{pid}.bin",
        )
        _memray_tracker = memray.Tracker(_memray_file)
        _memray_tracker.__enter__()  # pylint: disable=unnecessary-dunder-call

    start_time = time.perf_counter()

    def _save_profile() -> None:
        elapsed = time.perf_counter() - start_time

        if _profiler:
            _profiler.disable()

        if _memray_tracker:
            _memray_tracker.__exit__(None, None, None)

        if prof_dir:
            infix = "_kc" if include_kc_infix and "--pretrain-kc" in sys.argv else ""
            # Use unified saver
            save_combined_stats(
                prof_dir,
                f"{script_prefix}{infix}",
                elapsed,
                _profiler,
                _memray_file,
            )

    atexit.register(_save_profile)
