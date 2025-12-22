"""Profiling module for kotogram with cross-process atomic counters.

This module provides lightweight profiling capabilities that work across
threads and processes using file-based storage with file locking. Profiling
is only active when the PROFILE_KOTOGRAM environment variable is set to "1".

Usage:
    from kotogram.profile import increment_profile_counter, get_profile_report

    # In any function you want to profile:
    def my_function():
        increment_profile_counter()
        # ... function body

    # To get the profiling report:
    report = get_profile_report()
    print(report.counters)
"""

import fcntl
import inspect
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from kotogram.locations import ensure_dir, get_profile_dir

# Module-level state for thread safety within a process
_thread_lock = threading.Lock()


@dataclass
class ProfileReport:
    """Structured profiling report.

    Attributes:
        counters: Dictionary mapping fully-qualified function names to call counts.
        timestamp: ISO format timestamp when the report was generated.
    """

    counters: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def get_counter(self, name: str) -> int:
        """Get the sum of all counters containing the given name.

        Args:
            name: Substring to match in counter keys (e.g., "japanese_to_kotogram").

        Returns:
            Sum of counts for all matching counters.
        """
        return sum(count for key, count in self.counters.items() if name in key)


def _is_profiling_enabled() -> bool:
    """Check if profiling is enabled via environment variable."""
    return os.environ.get("PROFILE_KOTOGRAM") == "1"


def _get_counter_file_path() -> str:
    """Get the path to the counter storage file."""
    profile_dir = get_profile_dir()
    ensure_dir(profile_dir)
    return os.path.join(profile_dir, "counters.json")


def _get_caller_fqn(depth: int = 2) -> str:
    """Get the fully qualified name of the calling function.

    Args:
        depth: Stack depth to look at (2 = caller of the function calling this).

    Returns:
        Fully qualified name in format "module.class.function" or "module.function".
    """
    stack = inspect.stack()
    if len(stack) <= depth:
        return "unknown"

    frame_info = stack[depth]
    frame = frame_info.frame
    module = frame.f_globals.get("__name__", "")
    func_name = frame_info.function

    # Try to get class name if this is a method
    if "self" in frame.f_locals:
        cls = frame.f_locals["self"].__class__.__name__
        return f"{module}.{cls}.{func_name}"
    elif "cls" in frame.f_locals:
        cls = frame.f_locals["cls"].__name__
        return f"{module}.{cls}.{func_name}"

    return f"{module}.{func_name}"


def _read_counters_from_file(file_path: str) -> Dict[str, int]:
    """Read counters from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: Dict[str, int] = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def increment_profile_counter() -> None:
    """Increment the counter for the calling function.

    This function is designed to be called from any function you want to profile.
    The counter name is automatically determined from the caller's fully qualified
    name (module.class.function or module.function).

    When PROFILE_KOTOGRAM environment variable is not set to "1", this function
    returns immediately with no overhead.

    Thread-safe and process-safe using file locking.
    """
    if not _is_profiling_enabled():
        return

    caller = _get_caller_fqn()
    file_path = _get_counter_file_path()

    with _thread_lock:
        # Use file locking for cross-process atomicity
        # Open file for reading and writing, create if doesn't exist
        try:
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    # Read existing counters
                    with os.fdopen(os.dup(fd), "r", encoding="utf-8") as f:
                        try:
                            counters = json.load(f)
                        except (json.JSONDecodeError, ValueError):
                            counters = {}

                    # Increment counter
                    counters[caller] = counters.get(caller, 0) + 1

                    # Write back (truncate and write from beginning)
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.ftruncate(fd, 0)
                    content = json.dumps(counters).encode("utf-8")
                    os.write(fd, content)
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except OSError:
            # If we can't acquire lock, skip this increment (best effort)
            pass


def get_profile_report(profile_dir: Optional[str] = None) -> ProfileReport:
    """Get the current profiling report.

    Returns a structured ProfileReport containing all counter values.
    Also writes the report to .profile/report.yml.

    Args:
        profile_dir: Optional path to the .profile directory. If provided,
                    reads counters from that directory regardless of whether
                    profiling is enabled. If not provided, uses the default
                    location from get_profile_dir() (requires PROFILE_KOTOGRAM=1).

    Returns:
        ProfileReport with counters and timestamp.
    """
    timestamp = datetime.now().isoformat()

    # If profile_dir is explicitly provided, read from that location
    # Otherwise, require profiling to be enabled
    if profile_dir is None:
        if not _is_profiling_enabled():
            return ProfileReport(counters={}, timestamp=timestamp)
        profile_dir = get_profile_dir()

    file_path = os.path.join(profile_dir, "counters.json")
    counters = _read_counters_from_file(file_path)

    report = ProfileReport(counters=counters, timestamp=timestamp)

    # Write to JSON file
    ensure_dir(profile_dir)
    report_path = os.path.join(profile_dir, "report.json")

    report_dict = {"counters": report.counters, "timestamp": report.timestamp}

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    return report


def reset_profile_counters(profile_dir: Optional[str] = None) -> None:
    """Reset all profiling counters to zero.

    Useful for testing or starting a fresh profiling session.

    Args:
        profile_dir: Optional path to the .profile directory. If provided,
                    resets counters in that directory regardless of whether
                    profiling is enabled. If not provided, uses the default
                    location from get_profile_dir() (requires PROFILE_KOTOGRAM=1).
    """
    if profile_dir is None:
        if not _is_profiling_enabled():
            return
        profile_dir = get_profile_dir()

    ensure_dir(profile_dir)
    file_path = os.path.join(profile_dir, "counters.json")

    with _thread_lock:
        try:
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                os.write(fd, b"{}")
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except OSError:
            pass


def cleanup_shared_memory() -> None:
    """Clean up profiling resources.

    Removes the counter file. This is a legacy function name kept for
    backward compatibility - the profiling now uses file-based storage.
    """
    if not _is_profiling_enabled():
        return

    file_path = _get_counter_file_path()
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass
