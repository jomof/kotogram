"""
Library for executing processes in a restricted sandbox environment.

This module provides cross-platform primitives to run commands with restricted privileges,
specifically focusing on network and filesystem isolation where possible.

Platforms:
    - macOS: Uses `sandbox-exec` with dynamically generated Scheme profiles.
      (Note: Read access is allowed by default; only network and writes are restricted).
    - Linux: Unconfined (pass-through) by default.
"""

import logging
import os
import platform
import subprocess
import tempfile
from typing import Any, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)


def _build_mac_sandbox_profile(
    allow_network: bool = False,
    allow_read: Optional[List[str]] = None,
    allow_write: Optional[List[str]] = None,
) -> str:
    """Generates a macOS sandbox-exec (Scheme) profile."""
    # Use DENY DEFAULT to blacklist everything by default, as requested.
    # We must explicitly whitelist essential system capabilities.
    lines = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        # Essential non-filesystem permissions
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow signal)",
        "(allow mach*)",
        "(allow sysctl*)",
        "(allow ipc*)",
        "(allow file-read-metadata)",
        # Allow basic system reads? No, user wants strict control via allow_read.
        # But we must ensure allow_read handles paths correctly (literals vs subpaths).
    ]

    if allow_network:
        lines.append("(allow network*)")
    else:
        # Default denied, but good to be explicit/consistent
        lines.append("(deny network*)")

    # Filesystem: Read
    if allow_read:
        for path in allow_read:
            safe_path = path.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'(allow file-read* (subpath "{safe_path}"))')
            lines.append(f'(allow file-read* (literal "{safe_path}"))')

    # Filesystem: Write
    if allow_write:
        for path in allow_write:
            safe_path = path.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'(allow file-write* (subpath "{safe_path}"))')
            lines.append(f'(allow file-write* (literal "{safe_path}"))')

    # Allow sandbox-exec to break out of the sandbox to support nesting
    lines.append(
        '(allow process-exec (literal "/usr/bin/sandbox-exec") (with no-sandbox))'
    )

    # If allow_write is NOT specified, we do nothing (allow default handles it).
    # If allow_write IS specified but empty, we deny all writes?
    # Yes, standard security pattern.

    return "\n".join(lines)


def _build_command(
    args: List[str],
    allow_network: bool,
    allow_read: Optional[List[str]],
    allow_write: Optional[List[str]],
) -> List[str]:
    """Builds the full command with sandbox prefix."""
    # pylint: disable=unused-argument
    system = platform.system()

    if system == "Darwin":
        profile = _build_mac_sandbox_profile(allow_network, allow_read, allow_write)
        # sandbox-exec usage: sandbox-exec -p "profile" command...
        return ["sandbox-exec", "-p", profile] + args

    if system == "Linux":
        # Per user request: Do nothing on Linux (unconfined)
        return args

    logger.warning("Sandboxing not implemented for %s.", system)
    return args


def _expand_variables(path: str, context: Mapping[str, str]) -> str:
    """Expands variables like [exec-root] in paths."""
    for key, value in context.items():
        path = path.replace(f"[{key}]", value)
    return path


def _process_config(
    config: Mapping[str, Any], kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Resolves configuration variables and defaults."""
    # Context for variable expansion
    # Priority for exec-root: kwargs.get('cwd') -> os.getcwd()
    cwd = kwargs.get("cwd", os.getcwd())
    context = {
        "exec-root": cwd,
        "tmp": tempfile.gettempdir(),
        "home": os.path.expanduser("~"),
        "user": os.getenv("USER", "unknown"),
    }

    processed = dict(config)

    # Expand paths in allow_read/allow_write
    for list_key in ["allow_read", "allow_write"]:
        if list_key in processed and processed[list_key]:
            processed[list_key] = [
                os.path.realpath(_expand_variables(p, context))
                for p in processed[list_key]
            ]

    return processed


def confine(
    args: List[str],
    config: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
    **subprocess_kwargs: Any,
) -> Union["subprocess.CompletedProcess[Any]", None]:
    """
    Executes a command in a sandbox based on the provided configuration.

    Args:
        args: Command and arguments to run.
        config: Configuration dictionary.
            Keys:
            - allow_network: bool (default False)
            - allow_read: List[str] containing paths to allow reading.
              Supports variables: [exec-root], [tmp], [home], [user].
            - allow_write: List[str] containing paths to allow writing.
              Supports variables.
            - mode: "run" (default) or "exec".
              "run": Uses subprocess.run and returns CompletedProcess.
              "exec": Uses os.execvpe and does not return.
        env: Environment variables.
        **subprocess_kwargs: Arguments passed to subprocess.run (only used in 'run' mode).

    Returns:
        subprocess.CompletedProcess if mode is 'run'.
        None if mode is 'exec' (process replaced).
    """
    processed_config = _process_config(config, subprocess_kwargs)

    allow_network = processed_config.get("allow_network", False)
    allow_read = processed_config.get("allow_read")
    allow_write = processed_config.get("allow_write")
    mode = processed_config.get("mode", "run")

    cmd = _build_command(args, allow_network, allow_read, allow_write)

    if mode == "exec":
        # clean up kwargs not used by exec
        if env is None:
            env = os.environ

        # We cannot respect all subprocess_kwargs in exec mode (e.g. capture_output),
        # but we should respect cwd if possible? os.execvpe doesn't change cwd.
        # If user passed cwd, we should chdir?
        # Confine library usually assumes caller handles CWD for exec,
        # but subprocess.run handles it for run.
        # For parity, if cwd is in kwargs, we chdir.
        if "cwd" in subprocess_kwargs:
            os.chdir(subprocess_kwargs["cwd"])

        executable = cmd[0]
        os.execvpe(executable, cmd, env)
        return None

    # Default to check=False if not specified to match standard subprocess.run behavior
    check = subprocess_kwargs.pop("check", False)

    return subprocess.run(cmd, env=env, check=check, **subprocess_kwargs)


def main() -> None:
    """CLI Entry point."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run a command in a confined sandbox.")
    parser.add_argument("config", help="Path to JSON configuration file.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run.")

    args = parser.parse_args()

    if not args.command:
        parser.error("No command specified.")

    # argparse might put '--' in command if used as separator
    cmd_args = args.command
    if cmd_args[0] == "--":
        cmd_args = cmd_args[1:]

    if not cmd_args:
        parser.error("No command specified after --.")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Force exec mode for CLI wrapper usually
    # But let config override if it really wants to 'run' and exit?
    # Usually CLI wrapper implies exec.
    if "mode" not in config:
        config["mode"] = "exec"

    confine(cmd_args, config)


if __name__ == "__main__":
    main()
