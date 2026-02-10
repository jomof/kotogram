"""Profiling and timing utilities."""

import cProfile
import json
import os
import re
import resource
import sys
import time
from datetime import datetime
from typing import Any, List, Optional

import memray
from rich.console import Console

from train import paths

_console = Console()

memray: Any  # type: ignore # pylint: disable=no-member


def profiling_enabled() -> bool:
    """Check if profiling is enabled via TRAIN_PROFILE environment variable."""
    return os.environ.get("TRAIN_PROFILE", "0") == "1"


def get_profile_dir() -> Optional[str]:
    """Get the directory for profiling output based on hostname."""
    if not profiling_enabled():
        return None
    return paths.get_profile_dir()


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

            # cProfile - run alongside memray for CPU profiling
            self.profiler = cProfile.Profile()
            self.profiler.enable()

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
            self.profile_dir or "",
            name,
            elapsed,
            self.memray_file or "",
            self.profiler,
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


def _find_user_callsite(
    stack_trace: List[tuple[str, str, int]],
) -> Optional[tuple[str, str, int]]:
    """Find the first frame in user code (not in .venv/) from a stack trace.

    Args:
        stack_trace: List of (function, file_path, line_number) tuples.

    Returns:
        The first frame tuple that's in user code, or None if all frames are library code.
    """
    for frame in stack_trace:
        file_path = frame[1]
        if ".venv/" not in file_path and "site-packages/" not in file_path:
            return frame
    return None


def _format_location(func: str, file_path: str, line: int) -> str:
    """Format a location for display."""
    return f"{func}:{file_path}:{line}"


def _format_size(size_bytes: int) -> str:
    """Format a size in human-readable form."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.3f}GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.3f}MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.3f}kB"
    return f"{size_bytes}B"


def _generate_callsite_report(memray_file: str, top_n: int = 20) -> str:
    """Generate a custom top allocators report with user-code callsite lookup.

    For each allocation location that's in library code (.venv/), this looks up
    the stack trace to find the first frame in user code.

    Args:
        memray_file: Path to the memray .bin file.
        top_n: Number of top allocators to report.

    Returns:
        A formatted string report.
    """
    # pylint: disable=too-many-locals
    from collections import defaultdict

    from memray import FileReader

    # Aggregate allocations by location
    # Key: (function, file_path, line_number)
    # Value: {"size": total_bytes, "count": total_allocations, "callsites": set()}
    location_stats: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {"size": 0, "count": 0, "callsites": set()}
    )

    # Sampling rate: process every Nth record for speed while maintaining
    # representative distribution. With ~85M records, every 100th gives ~850K
    # which is still representative but processes in seconds not minutes.
    sample_rate = 100
    record_idx = 0

    with FileReader(memray_file) as reader:
        # Get total allocation count for progress
        metadata = reader.metadata
        total_records = getattr(metadata, "total_allocations", 0) or 0

        # If we don't have a total, use an estimate based on typical training runs
        if total_records == 0:
            total_records = 80_000_000  # Estimate ~80M allocations

        # Use all allocation records to find O(n) hotspots via allocation counts.
        # Sample for speed while maintaining representative distribution.
        progress_update_rate = (
            1_000_000  # Update every 1M records for reasonable display
        )
        start_time = time.perf_counter()
        for record in reader.get_allocation_records():
            record_idx += 1

            # Print progress every 1M records
            if record_idx % progress_update_rate == 0:
                pct = min(100, record_idx * 100 // total_records)
                elapsed = time.perf_counter() - start_time
                if pct > 0:
                    remaining = elapsed * (100 - pct) / pct
                else:
                    remaining = 0
                print(
                    f"\rCallsite analysis: {pct:3d}% "
                    f"({record_idx // 1_000_000}M/{total_records // 1_000_000}M) "
                    f"[{elapsed:.0f}s / {remaining:.0f}s remaining]",
                    end="",
                    flush=True,
                )

            if record_idx % sample_rate != 0:
                continue

            # Skip deallocations - they don't have stack traces
            # AllocatorType enum: PYMALLOC_FREE=1, FREE=5, MUNMAP=15
            allocator_int = int(record.allocator)
            if allocator_int in (1, 5, 15):
                continue

            stack = record.stack_trace()
            if not stack:
                continue

            # Top of stack is the immediate allocation location
            top_frame = stack[0]
            key = (top_frame[0], top_frame[1], top_frame[2])
            location_stats[key]["size"] += record.size
            location_stats[key]["count"] += record.n_allocations

            # If top frame is library code, find the user callsite
            if ".venv/" in top_frame[1] or "site-packages/" in top_frame[1]:
                user_frame = _find_user_callsite(stack)
                if user_frame:
                    location_stats[key]["callsites"].add(user_frame)

        # Print final progress with elapsed time
        final_elapsed = time.perf_counter() - start_time
        print(
            f"\rCallsite analysis: 100% ({record_idx // 1_000_000}M records) "
            f"[{final_elapsed:.1f}s]                    "
        )

    # Sort by size (descending) for top allocators by size
    by_size = sorted(
        location_stats.items(),
        key=lambda x: x[1]["size"],
        reverse=True,
    )[:top_n]

    # Sort by count (descending) for top allocators by count
    by_count = sorted(
        location_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )[:top_n]

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("CALLSITE REPORT (user code locations for library allocations)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(
        "🔍 Top 20 largest allocating locations (by size) with user callsites:"
    )

    for (func, file_path, line), stats in by_size:
        loc = _format_location(func, file_path, line)
        size_str = _format_size(stats["size"])
        lines.append(f"\t- {loc} -> {size_str}")
        # Show user callsites for library code
        if stats["callsites"]:
            for cs in sorted(stats["callsites"], key=lambda x: x[1]):
                cs_loc = _format_location(cs[0], cs[1], cs[2])
                lines.append(f"\t    <- {cs_loc}")

    lines.append("")
    lines.append(
        "🔍 Top 20 largest allocating locations (by count) with user callsites:"
    )

    for (func, file_path, line), stats in by_count:
        loc = _format_location(func, file_path, line)
        lines.append(f"\t- {loc} -> {stats['count']}")
        # Show user callsites for library code
        if stats["callsites"]:
            for cs in sorted(stats["callsites"], key=lambda x: x[1]):
                cs_loc = _format_location(cs[0], cs[1], cs[2])
                lines.append(f"\t    <- {cs_loc}")

    lines.append("")
    return "\n".join(lines)


def save_combined_stats(
    profile_dir: str,
    name: str,
    elapsed: float,
    memray_file: str,
    profiler: Optional[cProfile.Profile] = None,
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
    # (Removed dead code: profiler is always None when memray is active)

    # 2. Text Report (Combine CPU + Memray)
    txt_file = f"{base_path}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"PROFILE: {name} ({elapsed_str})\n")
        f.write("=" * 80 + "\n")

        # CPU Section using resource.getrusage()
        f.write("=" * 80 + "\n")
        f.write("CPU REPORT (resource usage)\n")
        f.write("-" * 80 + "\n")

        usage = resource.getrusage(resource.RUSAGE_SELF)
        user_time = usage.ru_utime
        sys_time = usage.ru_stime
        total_cpu = user_time + sys_time

        # Calculate CPU percentage
        if elapsed > 0:
            cpu_percent = (total_cpu / elapsed) * 100
        else:
            cpu_percent = 0

        # Format memory in MB
        max_rss_mb = usage.ru_maxrss / (1024 * 1024)  # Convert bytes to MB on macOS
        # Note: On Linux, ru_maxrss is in KB, on macOS it's in bytes

        f.write("\n⏱️ Timing:\n")
        f.write(f"\tWall time: {elapsed:.2f}s\n")
        f.write(f"\tUser time: {user_time:.2f}s\n")
        f.write(f"\tSystem time: {sys_time:.2f}s\n")
        f.write(f"\tCPU utilization: {cpu_percent:.1f}%\n")

        f.write("\n🧠 Memory:\n")
        f.write(f"\tPeak RSS: {max_rss_mb:.1f}MB\n")

        f.write("\n📊 Context Switches:\n")
        f.write(f"\tVoluntary: {usage.ru_nvcsw:,}\n")
        f.write(f"\tInvoluntary: {usage.ru_nivcsw:,}\n")

        f.write("\n📄 Page Faults:\n")
        f.write(f"\tMinor (no I/O): {usage.ru_minflt:,}\n")
        f.write(f"\tMajor (I/O required): {usage.ru_majflt:,}\n")

        f.write("\n")

        # cProfile Section - CPU hotspots
        if profiler is not None:
            import io
            import pstats

            f.write("=" * 80 + "\n")
            f.write("CPU HOTSPOTS (by cumulative time)\n")
            f.write("-" * 80 + "\n\n")

            # Capture pstats output - filter to project files only unless overridden
            step_start = time.perf_counter()
            with _console.status("[bold blue]Generating cProfile stats..."):
                stats_stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stats_stream)
                # Sort by cumulative time (total time including callees)
                stats.sort_stats("cumulative")
                f.write("Filter: None (show all)\n\n")
                stats.print_stats(30)
            step_time = time.perf_counter() - step_start
            _console.print(f"[green]✓[/green] cProfile stats ({step_time:.1f}s)")

            f.write(stats_stream.getvalue())
            f.write("\n")

        # Memray Section - generate callsite report directly (skip memray stats subprocess)
        if memray_file and os.path.exists(memray_file):
            f.write("=" * 80 + "\n")
            f.write("MEMORY REPORT (callsite analysis)\n")
            f.write("-" * 80 + "\n")
            f.flush()

            # Generate custom callsite report with user-code lookup (has its own progress bar)
            callsite_report = _generate_callsite_report(memray_file)
            f.write(callsite_report)

            f.write("\n")
            f.write(f"Raw memray file: {memray_file}\n")


def setup_profiling() -> None:
    """Enable cProfile if TRAIN_PROFILE is set and register exit handler."""
    if not profiling_enabled():
        return

    import atexit

    # Enable cProfile (Only if memray is NOT active)
    _profiler = None

    # Enable memray
    _memray_tracker = None
    _memray_file = None
    prof_dir = get_profile_dir()

    if memray and prof_dir:
        os.makedirs(prof_dir, exist_ok=True)
        pid = os.getpid()
        timestamp = int(time.time() * 1000)

        infix = "_kc" if "--pretrain-kc" in sys.argv else ""
        # Use a temporary name until we know elapsed time?
        # actually we can just name it broadly.
        _memray_file = os.path.join(
            prof_dir,
            f"train_style{infix}_global_{timestamp}_{pid}.bin",
        )
        _memray_tracker = memray.Tracker(_memray_file)
        _memray_tracker.__enter__()  # pylint: disable=unnecessary-dunder-call

    # Enable cProfile alongside memray
    _profiler = cProfile.Profile()
    _profiler.enable()

    start_time = time.perf_counter()

    def _save_profile() -> None:
        elapsed = time.perf_counter() - start_time

        if _profiler:
            _profiler.disable()

        if _memray_tracker:
            _memray_tracker.__exit__(None, None, None)

        if prof_dir:
            infix = "_kc" if "--pretrain-kc" in sys.argv else ""
            # Use unified saver
            save_combined_stats(
                prof_dir or "",
                f"train_style{infix}",
                elapsed,
                _memray_file or "",
                _profiler,
            )

    atexit.register(_save_profile)
