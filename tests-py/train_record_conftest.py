from typing import Any

import instrumentation

# Whitelist public symbols (marking these as used)
__all__ = ["pytest_configure", "pytest_sessionstart", "pytest_sessionfinish"]


def pytest_configure(config: Any) -> None:  # pylint: disable=unused-argument
    # Enable instrumentation early
    instrumentation.auto_enable()


def pytest_sessionstart(session: Any) -> None:  # pylint: disable=unused-argument
    # Re-enforce profiling if pytest messed with it, or just ensure it's on.
    # instrumentation.auto_enable() does this.

    # Cleanup only on main node (not xdist workers)
    if not hasattr(session.config, "workerinput"):
        import os
        import shutil

        from train import paths

        profile_dir = paths.get_profile_dir()
        if os.path.exists(profile_dir):
            # Ensure clean state by removing stale profile artifacts from previous runs.
            shutil.rmtree(profile_dir, ignore_errors=True)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # pylint: disable=unused-argument
    _ = exitstatus

    # Generate report at end of session (visible in pytest capture)
    # This might duplicate atexit report but ensures visibility in test failure logs.
    instrumentation.generate_report()

    # If we want to prevent atexit from reporting again, we could clear the recorder?
    # But atexit might be useful if python crashes?
    # Let's clean up global recorder to disable atexit reporting mostly
    # But generate_report checks _RECORDER.
    # instrumentation._RECORDER = None
    # ^ warning: modifying private global.
