"""
Parameter Constant Value Recorder (Instrumentation).

This module provides a mechanism to record function parameter values during execution
and report parameters that effectively act as constants (single unique value).
It is intended for use in both Pytest sessions and standalone script executions.
"""

import ast
import atexit
import os
import sys
from types import FrameType
from typing import Any, Dict, List, Optional, Set, Tuple

# Export public symbols
__all__ = ["auto_enable", "ParameterRecorder"]

# Environment variable to enable recording
ENV_VAR_ROOTS = "TRAIN_RECORD_ROOTS"
ENV_VAR_FAIL = "TRAIN_RECORD_FAIL_ON_CONST"
ENV_VAR_OUTPUT = "TRAIN_RECORD_OUTPUT_DIR"

# Sentinel to mark that a parameter has seen multiple distinct values
MANY_VALUES_SENTINEL = object()

# ANSI Colors
GREEN = "\033[1;32m"
RED = "\033[1;31m"
BLUE = "\033[1;34m"
RESET = "\033[0m"
BOLD_YELLOW = "\033[1;33m"


class ParameterRecorder:
    """Records unique values seen for function parameters."""

    def __init__(self, roots: List[str], fail_on_const: bool):
        self.roots = [os.path.abspath(r) + os.sep for r in roots]
        self.fail_on_const = fail_on_const
        # Key: (filename, firstlineno, qualname, param_name)
        # Value: (value, set_of_test_ids) OR MANY_VALUES_SENTINEL
        # Note: set_of_test_ids is discarded if value becomes MANY_VALUES_SENTINEL
        self.param_states: Dict[Tuple[str, int, str, str], Any] = {}
        # Set of keys that have been seen with a None value at least once
        self.seen_none_params: set[Tuple[str, int, str, str]] = set()

    def should_trace(self, filename: str) -> bool:
        """Determines if a file should be traced based on configured roots."""
        # Check exclusion patterns first
        if any(
            p in filename
            for p in (
                "/.venv/",
                "/.git/",
                "/__pycache__/",
                "/.mypy_cache/",
                "/tests-py/",
            )
        ):
            return False

        # Filter out synthetic filenames (e.g. <string>, <frozen...>)
        if filename.startswith("<"):
            return False

        abs_path = os.path.abspath(filename)
        # Must be under one of the roots
        return any(
            abs_path.startswith(root) or abs_path == root[:-1] for root in self.roots
        )

    def _freeze_value(self, value: Any) -> Any:
        """Freezes basic types for recording. Complex types are ignored (treated as many)."""
        # Accepted: None, bool, int, float, str, bytes
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str, bytes)):
            return value
        if isinstance(value, tuple):
            # Recurse.
            return tuple(self._freeze_value(x) for x in value)

        # Everything else is ignored / treated as "too complex to track as constant"
        return MANY_VALUES_SENTINEL

    def profile_func(self, frame: FrameType, event: str, _arg: Any) -> None:  # pylint: disable=too-many-locals
        """The sys.setprofile hook."""
        if event != "call":
            return

        code = frame.f_code
        filename = code.co_filename

        if not self.should_trace(filename):
            return

        # We assume code.co_name is sufficient for identification

        argcount = code.co_argcount
        varnames = code.co_varnames

        # Stable key base
        func_key_base = (filename, code.co_firstlineno, code.co_name)

        # Identify current test context
        current_test = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
        # Strip phase info (e.g. " (call)") from pytest current test string
        if " (" in current_test:
            current_test = current_test.split(" (")[0]

        for i in range(argcount):
            param_name = varnames[i]
            if param_name in frame.f_locals:
                val = frame.f_locals[param_name]
                frozen = self._freeze_value(val)

                if frozen is MANY_VALUES_SENTINEL:
                    new_val = MANY_VALUES_SENTINEL
                else:
                    new_val = frozen

                key = func_key_base + (param_name,)

                # Get current state
                # State is either (val, {sources}) or MANY_VALUES_SENTINEL
                current_state = self.param_states.get(key)
                if val is None:
                    self.seen_none_params.add(key)

                if current_state is MANY_VALUES_SENTINEL:
                    continue  # Already marked as varying

                if key not in self.param_states:
                    # First time seeing this param
                    if new_val is MANY_VALUES_SENTINEL:
                        self.param_states[key] = MANY_VALUES_SENTINEL
                    else:
                        self.param_states[key] = (new_val, {current_test})
                else:
                    # Seen before. Check if same.
                    old_val, sources = current_state

                    if new_val is MANY_VALUES_SENTINEL or old_val != new_val:
                        # Value changed or is un-trackable
                        self.param_states[key] = MANY_VALUES_SENTINEL
                    else:
                        # Same value, add source
                        sources.add(current_test)


_RECORDER: Optional[ParameterRecorder] = None


def generate_report() -> None:
    """Generates and prints the constant parameter report by aggregating files."""
    # pylint: disable=global-statement,global-variable-not-assigned
    global _RECORDER

    # Flush current process state to file
    if _RECORDER:
        persist_state()

    # If instrumentation was never configured (roots not set), do not generate a report.
    if not os.environ.get(ENV_VAR_ROOTS):
        return

    from train import paths

    output_dir = os.environ.get(ENV_VAR_OUTPUT)
    if not output_dir:
        output_dir = paths.get_profile_dir()

    if output_dir and os.path.exists(output_dir):
        # Determine project root for relative paths (heuristic: current dir or one up)
        project_root = os.environ.get("TRAIN_RECORD_ROOTS", os.getcwd()).split(
            os.pathsep
        )[0]
        aggregate_reports(
            output_dir,
            project_root=project_root,
            fail_on_const=os.environ.get(ENV_VAR_FAIL) == "1",
        )


def _build_serializable_state(recorder: "ParameterRecorder") -> List[Dict[str, Any]]:
    """Convert recorder state to JSON-serializable format."""
    serializable_state = []
    for key, value in recorder.param_states.items():
        if value is MANY_VALUES_SENTINEL:
            val_str = "<<MANY>>"
            test_sources: List[str] = []
        else:
            val, sources = value
            val_str = repr(val)
            test_sources = list(sources)

        has_seen_none = key in recorder.seen_none_params
        serializable_state.append(
            {
                "key": list(key),
                "value": val_str,
                "seen_none": has_seen_none,
                "sources": test_sources,
            }
        )
    return serializable_state


def _write_state_to_file(
    serializable_state: List[Dict[str, Any]], output_dir: str
) -> None:
    """Write serializable state to a JSON file atomically."""
    import json
    import tempfile

    pid = os.getpid()
    fname = os.path.join(output_dir, f"record_{pid}.json")
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f"record_{pid}.",
            suffix=".tmp",
            delete=False,
        )
        with temp_file as f:
            json.dump(serializable_state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file.name, fname)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f"[train-record] Failed to write report to {fname}: {e}",
            file=sys.stderr,
        )
        if temp_file is not None:
            try:
                os.remove(temp_file.name)
            except OSError:
                pass


def persist_state() -> None:
    """Persists the recorded state to file or stdout on exit."""
    # pylint: disable=global-variable-not-assigned
    global _RECORDER
    if not _RECORDER:
        return

    # Disable profiling
    sys.setprofile(None)

    output_dir = os.environ.get(ENV_VAR_OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    from train import paths

    if not output_dir:
        output_dir = paths.get_profile_dir()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        serializable_state = _build_serializable_state(_RECORDER)
        if serializable_state:
            _write_state_to_file(serializable_state, output_dir)


def aggregate_reports(
    output_dir: str, project_root: Optional[str] = None, fail_on_const: bool = False
) -> None:
    """Reads all JSON reports in output_dir and prints a merged report."""
    # pylint: disable=too-many-locals
    import glob
    import json

    # We do NOT print a big header up front anymore.
    # We check content first.

    # Key: (filename, line, func, param) -> set of seen values (reprs)
    aggregated: Dict[Tuple[str, int, str, str], set] = {}
    # Key: (filename, line, func, param) -> set of test sources
    source_map: Dict[Tuple[str, int, str, str], set] = {}
    seen_none_keys: Set[Tuple[str, int, str, str]] = set()

    files = glob.glob(os.path.join(output_dir, "record_*.json"))

    for fname in files:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    # item["key"] is list [filename, line, func, param]
                    key = tuple(item["key"])
                    val_str = item["value"]
                    has_seen_none = item.get("seen_none", False)
                    sources = item.get("sources", [])

                    if project_root and not key[0].startswith(project_root):
                        continue

                    if key not in aggregated:
                        aggregated[key] = set()
                        source_map[key] = set()

                    aggregated[key].add(val_str)
                    for s in sources:
                        source_map[key].add(s)

                    if has_seen_none:
                        seen_none_keys.add(key)

            # Clean up intermediate file
            os.remove(fname)

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"{RED}[train-record] Error reading {fname}: {e}{RESET}")

    # Determine constants
    constants_found = []

    for key in sorted(aggregated.keys()):
        seen_values = aggregated[key]
        filename, line, funcname, param = key

        if "<<MANY>>" in seen_values:
            continue
        if len(seen_values) > 1:
            continue  # Varying across runs

        # Only 1 unique value seen
        val_repr = list(seen_values)[0]
        if len(val_repr) > 100:
            val_repr = val_repr[:97] + "..."

        sources = source_map.get(key, set())

        # Format source info
        if not sources:
            source_summary = ""
        elif len(sources) == 1:
            source_summary = f" (from {list(sources)[0]})"
        else:
            first_5 = sorted(list(sources))[:5]
            source_summary = f" (from {len(sources)} tests: {', '.join(first_5)}{'...' if len(sources) > 5 else ''})"

        constants_found.append(
            (filename, line, funcname, param, val_repr, source_summary)
        )

    # Determine never-none
    never_none_found = _scan_optional_never_none(aggregated, seen_none_keys)

    # OUTPUT SECTION

    # 1. Constant Parameters
    if not constants_found:
        print(f"{GREEN}✅ Parameter constant value check OK{RESET}")
    else:
        # Header: Combined line, no separators
        print(
            f"\n{BLUE}Found {len(constants_found)} universally constant parameters{RESET}"
        )

        for item in constants_found:
            filename, line, funcname, param, val, s_summ = item
            # Relative path if project_root is provided
            display_path = filename
            if project_root:
                try:
                    display_path = os.path.relpath(filename, project_root)
                except ValueError:
                    pass

            # Split path into dir and basename for coloring
            dirname, basename = os.path.split(display_path)
            if dirname:
                dirname += os.sep

            # Colors
            nb_cyan = "\033[36m"
            nb_green = "\033[32m"
            nb_grey = "\033[90m"  # Grey for attribution

            # Format:
            # "  " prefix
            # {cyan}dirname/{reset}basename:{line}
            # {green}funcname{reset}
            # (parameter=value) -> param and = are white/reset, val is BOLD_YELLOW
            # {grey} (from test_...){reset}

            print(
                f"  {nb_cyan}{dirname}{RESET}{basename}:{line} {nb_green}{funcname}{RESET}({param}={BOLD_YELLOW}{val}{RESET}){nb_grey}{s_summ}{RESET}"
            )
        print("")

    # 2. Never None
    if not never_none_found:
        print(f"{GREEN}✅ Optional parameter 'never none' check OK{RESET}")
    else:
        # Header: Combined line, no separators (Matching style)
        print(
            f"\n{BLUE}Found {len(never_none_found)} optional parameters that were never None{RESET}"
        )

        for item in never_none_found:
            filename, line, funcname, param = item
            display_path = filename
            if project_root:
                try:
                    display_path = os.path.relpath(filename, project_root)
                except ValueError:
                    pass

            dirname, basename = os.path.split(display_path)
            if dirname:
                dirname += os.sep

            nb_cyan = "\033[36m"
            nb_green = "\033[32m"
            nb_yellow = "\033[33m"

            print(
                f"  {nb_cyan}{dirname}{RESET}{basename}:{line} {nb_green}{funcname}{RESET}({nb_yellow}{param}{RESET})"
            )
        print("")

    if fail_on_const and constants_found:
        print(f"\n{RED}[train-record] FAILING session due to {ENV_VAR_FAIL}=1{RESET}")
        sys.exit(1)


def _scan_optional_never_none(
    param_states: Dict[Tuple[str, int, str, str], Any],
    seen_none_keys: Set[Tuple[str, int, str, str]],
) -> List[Tuple[str, int, str, str]]:
    """Scans source code to find optional parameters that were never None. Returns list of matches."""
    # pylint: disable=too-many-locals,too-many-nested-blocks

    files_to_scan: Dict[str, List[Tuple[str, int, str, str]]] = {}
    for key in param_states:  # pylint: disable=consider-iterating-dictionary
        filename = key[0]
        if filename not in files_to_scan:
            files_to_scan[filename] = []
        files_to_scan[filename].append(key)

    never_none_found = []

    for filename, keys in files_to_scan.items():
        if not os.path.exists(filename):
            continue

        try:
            with open(filename, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=filename)

            # Helper to find functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_lineno = node.lineno

                    func_keys = [k for k in keys if k[1] == func_lineno]
                    if not func_keys:
                        continue

                    all_args = (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    )

                    for arg in all_args:
                        param_name = arg.arg

                        matching_key = next(
                            (k for k in func_keys if k[3] == param_name), None
                        )
                        if not matching_key:
                            continue

                        if matching_key in seen_none_keys:
                            continue

                        is_optional = False

                        if arg.annotation:
                            ann_str = ast.unparse(arg.annotation)
                            if (
                                "Optional" in ann_str
                                or ("Union" in ann_str and "None" in ann_str)
                                or "None |" in ann_str
                                or "| None" in ann_str
                            ):
                                is_optional = True

                        if is_optional:
                            never_none_found.append(
                                (filename, func_lineno, node.name, param_name)
                            )

        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return never_none_found


def auto_enable() -> None:
    """Automatically enables the recorder if the environment variable is set."""
    # pylint: disable=global-statement
    global _RECORDER

    if _RECORDER:
        return  # Already enabled

    roots_env = os.environ.get(ENV_VAR_ROOTS, "")
    fail_on_const = os.environ.get(ENV_VAR_FAIL, "") == "1"

    if not roots_env:
        return

    roots = [r.strip() for r in roots_env.split(os.pathsep) if r.strip()]
    if not roots:
        return

    _RECORDER = ParameterRecorder(roots, fail_on_const)
    sys.setprofile(_RECORDER.profile_func)
    atexit.register(persist_state)
