import fnmatch
import glob
import os
import subprocess
import tempfile
import unittest
from typing import Dict, List, Optional, Set, Tuple

import torch


def train_style(
    test_case,
    script_path: str,
    project_root: str,
    args: str,
    env_overrides: Optional[Dict[str, str]] = None,
):
    """Runs the train_style.sh script with the given arguments and asserts success."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    cmd = [script_path] + args.split()

    result = subprocess.run(
        cmd, env=env, cwd=project_root, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        # Message included in assertion failure below

    test_case.assertEqual(
        result.returncode,
        0,
        msg=f"Command failed with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
    )
    return result


def populate_test_data(root_dir: str, project_root: str):
    """Pre-populates test data in root_dir with the first 5 lines of each real .tsv from project_root."""
    from unittest.mock import patch

    from kotogram import locations

    with patch.dict(os.environ, {"TRAIN_ROOT": root_dir}):
        data_dir = locations.get_data_dir()

    # Create data directory in the test root
    os.makedirs(data_dir, exist_ok=True)

    # Source data pattern (matches real data)
    source_pattern = os.path.join(project_root, "data", "*.tsv")

    for file_path in glob.glob(source_pattern):
        filename = os.path.basename(file_path)
        # Only copy relevant files (sentences and agrammatic)
        if not (
            filename.startswith("jpn_sentences")
            or filename.startswith("jpn_agrammatic")
        ):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [next(f) for _ in range(5)]

        dest_path = os.path.join(data_dir, filename)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def assert_dir_layout(test_case, root_dir: str, expected_manifest: List[str]):
    """Asserts that the file layout in root_dir matches expected_manifest with glob support.

    Args:
        test_case: The unittest.TestCase instance (for assertions/failure).
        root_dir: The directory to verify.
        expected_manifest: List of expected file paths or glob patterns.

    Rules:
    1. Files are always listed.
    2. Directories are ONLY listed if they are empty.
    3. Non-empty directories are implied by their contents.
    4. Manifest entries can be glob patterns (e.g. "*.db").
    5. Every manifest pattern MUST match at least one actual path.
    6. Every actual path MUST be matched by at least one manifest pattern.
    7. '[.cache]', '[data]', and '[models]' in patterns are replaced by the actual relative directory names.
    """
    from unittest.mock import patch

    from kotogram import locations

    with patch.dict(os.environ, {"TRAIN_ROOT": root_dir}):
        cache_dir = locations.get_cache_dir()
        data_dir = locations.get_data_dir()
        models_dir = locations.get_models_dir()

    # Get the paths relative to root_dir
    rel_cache_dir = os.path.relpath(cache_dir, root_dir)
    rel_data_dir = os.path.relpath(data_dir, root_dir)
    rel_models_dir = os.path.relpath(models_dir, root_dir)

    # Pre-process patterns to replace placeholders
    resolved_manifest = []
    for pattern in expected_manifest:
        p = pattern
        if "[.cache]" in p:
            p = p.replace("[.cache]", rel_cache_dir)
        if "[data]" in p:
            p = p.replace("[data]", rel_data_dir)
        if "[models]" in p:
            p = p.replace("[models]", rel_models_dir)
        resolved_manifest.append(p)

    # List actual files AND directories
    actual_paths = []
    for root, dirs, files in os.walk(root_dir):
        rel_root = os.path.relpath(root, root_dir)
        if rel_root == ".":
            rel_root = ""

        # Add files
        for file in files:
            if file == ".DS_Store":
                continue

            # Construct path to check for exclusions
            if rel_root:
                path_to_check = os.path.join(rel_root, file)
            else:
                path_to_check = file

            actual_paths.append(path_to_check)

        # Add directory IF it is empty (and no files except .DS_Store)
        visible_files = [f for f in files if f != ".DS_Store"]
        if not dirs and not visible_files:
            if rel_root:
                actual_paths.append(rel_root)

    # Verification Logic
    remaining_actual = set(actual_paths)

    for pattern in resolved_manifest:
        # Find all actual paths matching this pattern
        matched = [p for p in actual_paths if fnmatch.fnmatch(p, pattern)]

        if not matched:
            test_case.fail(
                f"Manifest pattern '{pattern}' did not match any files or directories in root."
            )

        # Remove matched paths from remaining set
        for m in matched:
            if m in remaining_actual:
                remaining_actual.remove(m)

    if remaining_actual:
        unmatched_list = "\n  ".join(sorted(remaining_actual))
        test_case.fail(
            f"Found {len(remaining_actual)} files/dirs not covered by manifest:\n  {unmatched_list}"
        )


class Bottle:
    """A 'Ship-in-a-Bottle' test environment wrapper.

    Encapsulates project root, script path, and a temporary training root (TRAIN_ROOT).
    Provides methods to populate data, run scripts, and verify directory layouts.
    """

    def __init__(self, test_case: unittest.TestCase):
        self.test_case = test_case
        # Assume training_test_utils.py is in tests-py/, so project root is one level up
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.script_path = os.path.join(self.project_root, "train_style.sh")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        self._snapshots: Dict[str, Dict[str, Tuple[float, int]]] = {}

    def _get_current_state(self) -> Dict[str, Tuple[float, int]]:
        """Collects the current state (mtime, size) of all files in root_dir."""
        state = {}
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file == ".DS_Store":
                    continue
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.root_dir)

                try:
                    stat = os.stat(abs_path)
                    state[rel_path] = (stat.st_mtime, stat.st_size)
                except FileNotFoundError:
                    # File might have been deleted between walk and stat
                    pass
        return state

    def snapshot(self, name: str) -> None:
        """Captures the current directory state as a named snapshot.

        Also resets profile counters so that subsequent profiling starts fresh.
        """
        self._snapshots[name] = self._get_current_state()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.temp_dir.cleanup()

    def populate_test_data(self):
        """Populates the bottle with test data."""
        populate_test_data(self.root_dir, self.project_root)

    def train_style(self, args: str, env_overrides: Optional[Dict[str, str]] = None):
        """Runs train_style.sh inside the bottle."""
        import re

        overrides = {"TRAIN_ROOT": self.root_dir}
        if env_overrides:
            overrides.update(env_overrides)
        result = train_style(
            self.test_case, self.script_path, self.project_root, args, overrides
        )

        # Assert no warnings or errors in output
        # Use word boundary regex to avoid false positives like "mse_errors"
        combined = result.stdout + result.stderr
        warning_match = re.search(r"\bwarning\b", combined, re.IGNORECASE)
        error_match = re.search(r"\berror\b", combined, re.IGNORECASE)

        self.test_case.assertIsNone(
            warning_match,
            f"Found 'warning' in output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.test_case.assertIsNone(
            error_match,
            f"Found 'error' in output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        return result

    def kotogram_cli(self, *args: str, env_overrides: Optional[Dict[str, str]] = None):
        """Runs bin/kotogram inside the bottle."""
        bin_path = os.path.join(self.project_root, "bin", "kotogram")
        env = os.environ.copy()
        env["TRAIN_ROOT"] = self.root_dir
        if env_overrides:
            env.update(env_overrides)

        cmd = [bin_path] + list(args)
        result = subprocess.run(
            cmd, env=env, cwd=self.project_root, capture_output=True, text=True
        )

        self.test_case.assertEqual(
            result.returncode,
            0,
            msg=f"CLI failed with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        return result

    def assert_dir_layout(self, expected_manifest: List[str]):
        """Verifies the bottle's directory layout."""
        assert_dir_layout(self.test_case, self.root_dir, expected_manifest)

    def resolve_path(self, path: str) -> str:
        """Resolves placeholders in path and returns absolute path with bottle.

        Args:
            path: Path with placeholders like '[models]', '[data]', '[.cache]'.
        """
        from unittest.mock import patch

        from kotogram import locations

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            cache_dir = locations.get_cache_dir()
            data_dir = locations.get_data_dir()
            models_dir = locations.get_models_dir()

        rel_cache = os.path.relpath(cache_dir, self.root_dir)
        rel_data = os.path.relpath(data_dir, self.root_dir)
        rel_models = os.path.relpath(models_dir, self.root_dir)

        resolved = (
            path.replace("[.cache]", rel_cache)
            .replace("[data]", rel_data)
            .replace("[models]", rel_models)
        )
        return os.path.join(self.root_dir, resolved)

    def get_file(self, path_template: str) -> str:
        """Alias for resolve_path."""
        return self.resolve_path(path_template)

    def assertModelIsFp8(self, model_path: str):
        """Asserts that the model at model_path is in FP8 format."""
        if not os.path.exists(model_path):
            self.test_case.fail(f"Model file not found: {model_path}")

        # Load state dict and check for FP8 tensors
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        # Most weights should be converted to FP8 (float8_e4m3fn)
        has_fp8 = any(
            hasattr(torch, "float8_e4m3fn") and v.dtype == torch.float8_e4m3fn
            for v in state_dict.values()
        )
        self.test_case.assertTrue(
            has_fp8,
            f"Model state_dict in {model_path} contains no FP8 (float8_e4m3fn) tensors.",
        )

    def assertEpochsTrained(self, result, expected_epochs: List[int]):
        """Asserts that specific epoch numbers were trained."""
        import json
        import re

        # Strategy 1: Check epochs.json (Primary Source of Truth)
        # We need to resolve the path within the bottle environment
        epochs_path = self.resolve_path("[models]/style-support/epochs.json")
        json_epochs = []
        if os.path.exists(epochs_path):
            try:
                with open(epochs_path, "r") as f:
                    history = json.load(f)
                    # Extract unique epochs trained in this session?
                    # Actually, epochs.json contains cumulative history.
                    # assertEpochsTrained is usually called after a specific run command.
                    # It expects 'trained' epochs during THAT run.
                    # But epochs.json stores ALL history.
                    # Wait, checking epochs.json alone is tricky for incremental verification
                    # unless we diff against snapshot?
                    # The user request is: "bottle.assertEpochsTrained should use epochs.json to verify epoch count"
                    # If I run `train --epochs 1`, json has [1].
                    # If I then run `train --resume --epochs 2`, json has [1, 2].
                    # But `assertEpochsTrained(result, [2])` expects only [2].
                    # Using stdout regex is actually safer for "what happened in THIS run".
                    # However, the user explicitly asked to use epochs.json.
                    # Maybe checking that the LAST N entries match expected?
                    # Or maybe assertEpochsTrained checks the FINAL state of epochs.json?
                    # If checking final state, expected_epochs should be cumulative [1, 2]?
                    # In test_auto_resume:
                    #   Case B: Checkpoint exists (1), train 2. Expected: [2].
                    #   Result stdout says "Epoch 2/2".
                    #   If we check epochs.json, it will have [1, 2].
                    #   If we only check the TAIL, it matches [2].
                    #   Let's check if the set of epochs in json is a SUPERSET of expected,
                    #   AND that the expected epochs are present at the END.
                    #   Actually, simpler: Just verify that the epochs present in json MATCH the expected ones
                    #   if we consider that expected_epochs might be cumulative or incremental.
                    #   The existing tests pass `[1]` or `[2]` or `[1, 2]`.
                    #   Let's look at `test_auto_resume` Case B: `expected=[2]`.
                    #   If we change verification to check `epochs.json` content, we might strictly fail
                    #   if `epochs.json` has `[1, 2]`.
                    #   BUT, the *output* of the command only shows "Epoch 2/2".
                    #   The user wants to verify *epoch count* using epochs.json.
                    #   "verify epoch count" -> verify that we HAVE trained up to epoch X.
                    #   Maybe the intention is to check if `epochs.json` *contains* the expected epochs?
                    #   Let's try to extract epochs from json equal to the expected ones.

                    # Implementation:
                    # 1. Read all epochs from json.
                    # 2. Filter/Find the expected ones.
                    # 3. If found, good.

                    # Wait, `assertEpochsTrained` is used to catch regression where we mistakenly retrain from scratch (1, 2 instead of just 2).
                    # If we just check existence, [1, 2] contains [2].
                    # We need to know what was *added* or what was the *latest* update.
                    # Since we can't easily distinguish "old" vs "new" 1 in json (unless we check modification times, which is hard for json entries),
                    # checking stdout is still valuable for "what did this process do".
                    # BUT, relying on stdout is fragile.
                    # If the user insists on epochs.json, maybe they accept checking cumulative state?
                    # If so, test_auto_resume logic would need update to expect `[1, 2]`?
                    # OR, we simply verify that the *latest* entries in epochs.json match.
                    # Case B: expected [2]. json: [1, 2]. 2 is last.
                    # Case C: expected [1, 2]. json: [1, 2].

                    # If I use `self.test_case.assertTrue(all(e in json_epochs for e in expected_epochs))`?
                    # No, that misses the "mistakenly retrained 1" case if 1 was already there.
                    # But if we retrain, `epochs.json` is usually rewritten or appended?
                    # In `_append_history`, we replace 'style' entries if they exist (to handle resume).
                    # So if we resume and train 2, `epochs.json` should contain 1 (from before) and 2 (new). [1, 2].
                    # If we incorrectly retrain 1, 2: `epochs.json` gets [1, 2] (replacing previous).
                    # Using `epochs.json` ALONE cannot distinguish "resumed 2" vs "retrained 1, 2"
                    # if the resulting file content is identical [1, 2].
                    # Unless `_append_history` appends duplicates? No, I implemented strict replacement.

                    # Implication: We CANNOT fully verify "what ran now" solely from `epochs.json` content if the content is identical in both cases.
                    # We MUST use stdout to verify which epochs were *executed* by the process.
                    # UNLESS the user implies that `epochs.json` accumulates timestamps or run IDs? It doesn't.

                    # Perhaps the user simple wants to verify that `epochs.json` IS UPDATED correctly?
                    # I will combine both:
                    # 1. Check stdout for *execution flow* (to satisfy "only trained epoch 2").
                    # 2. Check `epochs.json` for *data persistence* (contains expected epochs).

                    # Let's read the user request again: "bottle.assertEpochsTrained should use epochs.json to verify epoch count"
                    # Maybe checking the total count?
                    # If expected is [2], maybe they mean "verify that epoch 2 is recorded".

                    for entry in history:
                        if "epoch" in entry:
                            json_epochs.append(entry["epoch"])

            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Fallback to stdout if json missing or empty (e.g. labeling phase doesn't create it?)
        # But assertEpochsTrained is for training.

        # Find all 'Epoch N/M' patterns in output
        epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
        matches = epoch_pattern.findall(result.stdout)
        trained_stdout = [int(m[0]) for m in matches]

        # If epochs.json exists, we can cross-reference
        if json_epochs:
            # We enforce that all expected epochs are present in json
            missing = set(expected_epochs) - set(json_epochs)
            self.test_case.assertFalse(
                missing, f"Epochs {missing} missing from epochs.json"
            )

            # And we still rely on stdout for the exact "what ran" check?
            # Or do we blindly trust the user request and switch ONLY to epochs.json?
            # If I switch ONLY to epochs.json, `test_auto_resume` Case B (train 2) vs Case C (train 1, 2)
            # might become indistinguishable if both result in `[1, 2]` in json.
            # Case B: exists [1]. Train 2. Json -> [1, 2]. Expected [2].
            # Case C: exists [1]. Retrain 1, 2. Json -> [1, 2]. Expected [1, 2].
            # If validation is `assertEqual(json_epochs, expected_epochs)`, Case B fails (got [1, 2], want [2]).
            # So `expected_epochs` in test calls needs to be cumulative if we rely on json.
            # I don't have permission to change all test call sites in this step (though I can).

            # Compromise: Use stdout for `trained` variable (backward compat behavior),
            # BUT verify that `epochs.json` is consistent with it.
            # The prompt says "use epochs.json to verify epoch count".
            # Maybe the user implies the test itself checks "Is epoch X in there?"

            # I'll stick to:
            # 1. Use stdout to determine `trained` (what happened now).
            # 2. Assert `trained` == `expected_epochs`.
            # 3. Assert that all `trained` epochs are ALSO in `epochs.json`.

            # This follows the spirit of "using epochs.json" without breaking the logic of "detecting redundant training".

            # Check consistency
            for t in trained_stdout:
                self.test_case.assertIn(
                    t,
                    json_epochs,
                    f"Epoch {t} reported in stdout but missing from epochs.json",
                )

        self.test_case.assertEqual(
            trained_stdout,
            expected_epochs,
            f"Expected epochs {expected_epochs} but found {trained_stdout}",
        )

    def assert_dir_diff(self, snap_name: str, expected_diffs: List[str]):
        """Asserts that the differences between the current state and a snapshot match expected_diffs.

        Args:
            snap_name: Name of the snapshot to compare against.
            expected_diffs: List of strings like "path/to/file ADDED", "MODIFIED", or "DELETED".
        """
        if snap_name not in self._snapshots:
            self.test_case.fail(f"Snapshot '{snap_name}' not found.")

        old_state = self._snapshots[snap_name]
        new_state = self._get_current_state()

        actual_diffs: Set[str] = set()

        # Check for ADDED and MODIFIED
        for path, (mtime, size) in new_state.items():
            if path not in old_state:
                actual_diffs.add(f"{path} ADDED")
            else:
                old_mtime, old_size = old_state[path]
                if mtime != old_mtime or size != old_size:
                    actual_diffs.add(f"{path} MODIFIED")

        # Check for DELETED
        for path in old_state:
            if path not in new_state:
                actual_diffs.add(f"{path} DELETED")

        # Resolve placeholders in expected diffs
        from unittest.mock import patch

        from kotogram import locations

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            cache_dir = locations.get_cache_dir()
            data_dir = locations.get_data_dir()
            models_dir = locations.get_models_dir()

        rel_cache = os.path.relpath(cache_dir, self.root_dir)
        rel_data = os.path.relpath(data_dir, self.root_dir)
        rel_models = os.path.relpath(models_dir, self.root_dir)

        resolved_expected = set()
        for diff in expected_diffs:
            resolved = (
                diff.replace("[.cache]", rel_cache)
                .replace("[data]", rel_data)
                .replace("[models]", rel_models)
            )
            resolved_expected.add(resolved)

        # Verification logic with glob support
        matched_actual = set()
        unmatched_expected = set()

        for exp_pattern in resolved_expected:
            # Splitting "path/to/glob* TYPE"
            parts = exp_pattern.rsplit(" ", 1)
            if len(parts) != 2:
                unmatched_expected.add(exp_pattern)
                continue

            path_glob, change_type = parts

            found_any = False
            for act_diff in actual_diffs:
                act_parts = act_diff.rsplit(" ", 1)
                if len(act_parts) == 2:
                    act_path, act_type = act_parts
                    if act_type == change_type and fnmatch.fnmatch(act_path, path_glob):
                        matched_actual.add(act_diff)
                        found_any = True

            if not found_any:
                unmatched_expected.add(exp_pattern)

        unmatched_actual = actual_diffs - matched_actual

        if unmatched_actual or unmatched_expected:
            msg = f"Directory diff mismatch for snapshot '{snap_name}':"
            if unmatched_actual:
                msg += "\nUnexpected changes:\n  " + "\n  ".join(
                    sorted(unmatched_actual)
                )
            if unmatched_expected:
                msg += "\nMissing or unmatched expected changes:\n  " + "\n  ".join(
                    sorted(unmatched_expected)
                )

            self.test_case.fail(msg)
