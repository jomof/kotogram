"""Subprocess I/O utilities for training scripts."""

import subprocess
import sys
from typing import IO, List


def tee_subprocess_output(
    process: "subprocess.Popen[bytes]",
    log_file: IO[bytes],
    chunk_size: int = 4096,
) -> None:
    """Copy subprocess stdout to both console and log file in real-time.

    This is a low-level binary passthrough for real-time streaming of
    subprocess output (e.g., progress bars). It is NOT display logic
    and does not go through view callbacks.

    Args:
        process: A Popen object with stdout=PIPE
        log_file: An open binary file to write to
        chunk_size: Size of chunks to read at a time
    """
    if process.stdout is None:
        raise RuntimeError("Failed to capture stdout from subprocess")

    while True:
        # Read chunks to allow real-time progress bar updates
        # without newline buffering issues
        chunk = process.stdout.read(chunk_size)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        log_file.write(chunk)
        log_file.flush()


def run_command_with_tee(
    cmd: List[str],
    env: dict,
    cwd: str,
    tee_log: str,
) -> int:
    """Run a command, teeing output to both console and a log file.

    Args:
        cmd: Command and arguments to run
        env: Environment variables
        cwd: Working directory
        tee_log: Path to log file (parent directory created if needed)

    Returns:
        The command's return code
    """
    import os

    # Create parent directory if needed
    os.makedirs(os.path.dirname(tee_log), exist_ok=True)

    with open(tee_log, "ab") as log_file:
        with subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        ) as process:
            tee_subprocess_output(process, log_file)
            process.wait()
            return process.returncode


# Explicit reference for static analysis - called from ./train_style
# pylint: disable=pointless-statement
run_command_with_tee
