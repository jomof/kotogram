import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from typing import Dict, List, Optional, Set

import torch


# pylint: disable=too-many-positional-arguments
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
        cmd,
        env=env,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    else:
        # Help iteration by printing output even on success
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

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


# pylint: disable=too-many-locals
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

    # Check for duplicates in expected_manifest
    if len(expected_manifest) != len(set(expected_manifest)):
        duplicates = {x for x in expected_manifest if expected_manifest.count(x) > 1}
        test_case.fail(f"Duplicate entries found in expected_manifest: {duplicates}")

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

            # Skip profiling artifacts (any file/dir starting with .profile)
            if ".profile" in root or file.startswith(".profile"):
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
            # Skip profiling artifacts (directories starting with .profile)
            if rel_root.startswith(".profile") or "/.profile" in rel_root:
                continue

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
        self.script_path = os.path.join(self.project_root, "train_style")
        # pylint: disable=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = self.temp_dir.name
        # Stores snapshots as Dict[snap_name, Dict[rel_path, hash]]
        self._snapshots: Dict[str, Dict[str, str]] = {}

    def _get_file_hash(self, path: str) -> str:
        """Calculates SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_current_state(self) -> Dict[str, str]:
        """Collects the current state (file hashes) of all files in root_dir."""
        state = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file == ".DS_Store":
                    continue
                # Skip profiling artifacts (any file/dir starting with .profile)
                if ".profile" in root or file.startswith(".profile"):
                    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.root_dir)

                if os.path.exists(abs_path):
                    state[rel_path] = self._get_file_hash(abs_path)
        return state

    def snapshot(self, name: str) -> None:
        """Captures the current directory state as a named snapshot."""
        self._snapshots[name] = self._get_current_state()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.temp_dir.cleanup()

    def populate_test_data(self):
        """Populates the bottle with test data."""
        populate_test_data(self.root_dir, self.project_root)

    def calculate_expected_counts(self) -> Dict[str, int]:
        """Calculates expected sentence counts for MLM and KC pretraining.

        Simulates the logic of StyleDataset and KCTrainer to provide ground truth.
        """
        from unittest.mock import patch

        from kotogram import locations

        # Find cache directory in bottle (where script puts processed data)
        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            cache_dir = locations.get_cache_dir()

        gram_cache = os.path.join(cache_dir, "grammatic_combined.tsv")
        agram_cache = os.path.join(cache_dir, "agrammatic_combined.tsv")

        def count_lines(path):
            if not os.path.exists(path):
                return 0
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        # In cache, grammatic_combined contains all grammatic sentences
        mlm_count = count_lines(gram_cache)
        num_gram = mlm_count
        num_agram = count_lines(agram_cache)

        # Total dataset size
        total_len = num_gram + num_agram

        # Simulate StyleDataset.split(seed=42, train_ratio=0.8)
        # We need to know which indices are grammatic (0..num_gram-1)
        # and which are agrammatic (num_gram..total_len-1).
        # This assumes data is loaded in that block order (grammatic first).
        # Based on train_style.py: data_files = [gram, agram]. Yes.

        torch.manual_seed(42)
        indices = torch.randperm(total_len)
        n_train = int(total_len * 0.8)
        train_indices = indices[:n_train]

        # Count how many of train_indices correspond to grammatic part (< num_gram)
        kc_count = (train_indices < num_gram).sum().item()

        return {
            "total_grammatic_sentences": mlm_count,
            "grammatic_sentences_in_train_split": kc_count,
            "total_train_split_size": n_train,
        }

    def train_style(
        self,
        args: str,
        env_overrides: Optional[Dict[str, str]] = None,
    ):
        """Runs train_style.sh inside the bottle."""
        import re

        overrides = {"TRAIN_ROOT": self.root_dir}
        if env_overrides:
            if "TRAIN_ROOT" in env_overrides:
                raise ValueError(
                    "TRAIN_ROOT cannot be overridden in bottle.train_style()"
                )
            if "SKIP_DEPS" in env_overrides:
                raise ValueError(
                    "SKIP_DEPS cannot be overridden in bottle.train_style()"
                )
            overrides.update(env_overrides)
        result = train_style(
            self.test_case,
            self.script_path,
            self.project_root,
            args,
            overrides,
        )

        # Assert no warnings or errors in output
        # Use word boundary regex to avoid false positives like "mse_errors"
        combined = result.stdout + result.stderr

        # Filter out harmless distributed warnings
        filtered_combined = []
        for line in combined.splitlines():
            # PyTorch internal socket warnings [W socket.cpp:...]
            if "[W " in line and "socket.cpp" in line:
                continue
            # torch.distributed.run noise
            if "torch.distributed.run" in line and "WARNING" in line:
                continue
            # Generic distributed init noise
            if "failed to connect" in line and "localhost" in line:
                continue
            # Gloo loopback fallback warning
            if "ProcessGroupGloo.cpp" in line and "loopback" in line:
                continue
            filtered_combined.append(line)

        combined_filtered = "\n".join(filtered_combined)
        warning_match = re.search(r"\bwarning\b", combined_filtered, re.IGNORECASE)
        error_match = re.search(r"\berror\b", combined_filtered, re.IGNORECASE)

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
        env["TRAIN_PROFILE"] = "0"
        if env_overrides:
            env.update(env_overrides)

        cmd = [bin_path] + list(args)
        result = subprocess.run(
            cmd,
            env=env,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
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

        if path == "[models]/style-support/epochs.json":
            raise ValueError(
                "Use bottle.get_epoch_history() instead of resolving epochs.json directly."
            )

        resolved = (
            path.replace("[.cache]", rel_cache)
            .replace("[data]", rel_data)
            .replace("[models]", rel_models)
        )
        return os.path.join(self.root_dir, resolved)

    def get_file(self, path_template: str) -> str:
        """Alias for resolve_path."""
        return self.resolve_path(path_template)

    # pylint: disable=invalid-name
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

    def get_epoch_history(self) -> List[Dict]:
        """Reads and returns the parsed content of epochs.json."""
        # Use resolve_path logic internally bypassing the check
        # We know epochs.json is in support dir.
        from unittest.mock import patch

        from kotogram import locations

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            support_dir = locations.get_style_support_dir()

        epochs_path = os.path.join(support_dir, "epochs.json")
        if not os.path.exists(epochs_path):
            return []

        with open(epochs_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # pylint: disable=invalid-name
    def assertEpochsTrained(self, result, expected_epochs: List[int]):
        """Asserts that specific epoch numbers were trained."""
        import re

        # Strategy 1: Check epochs.json (Primary Source of Truth)
        # Strategy 1: Check epochs.json (Primary Source of Truth)
        history = self.get_epoch_history()
        json_epochs = []
        for entry in history:
            if "epoch" in entry:
                json_epochs.append(entry["epoch"])

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

            # Check consistency: epochs reported in stdout MUST be in json
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

    # pylint: disable=too-many-locals,too-many-nested-blocks
    def assert_dir_diff(self, snap_name: str, expected_diffs: List[str]):
        """Asserts that the differences between the current state and a snapshot match expected_diffs.

        Args:
            snap_name: Name of the snapshot to compare against.
            expected_diffs: List of strings like "path/to/file ADDED", "MODIFIED", or "DELETED".
        """
        if len(expected_diffs) != len(set(expected_diffs)):
            duplicates = {x for x in expected_diffs if expected_diffs.count(x) > 1}
            self.test_case.fail(
                f"Duplicate entries found in expected_diffs: {duplicates}"
            )

        if snap_name not in self._snapshots:
            self.test_case.fail(f"Snapshot '{snap_name}' not found.")

        old_state = self._snapshots[snap_name]
        new_state = self._get_current_state()

        actual_diffs: Set[str] = set()

        # Check for ADDED and MODIFIED
        for path, current_hash in new_state.items():
            if path not in old_state:
                actual_diffs.add(f"{path} ADDED")
            else:
                old_hash = old_state[path]
                if current_hash != old_hash:
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

                    # Handle MAYBE-MODIFIED: match if modified OR if file exists but wasn't modified
                    if change_type == "MAYBE-MODIFIED":
                        # If actual is MODIFIED, it matches.
                        if act_type == "MODIFIED" and fnmatch.fnmatch(
                            act_path, path_glob
                        ):
                            matched_actual.add(act_diff)
                            found_any = True
                        # If actual is not in diffs, we fall through. But wait, actual_diffs ONLY contain changes.
                        # MAYBE-MODIFIED means: if it IS modified, it's consumed. If it's NOT modified, it's also okay.
                        # So if we find a MODIFIED, we claim it.
                    elif act_type == change_type and fnmatch.fnmatch(
                        act_path, path_glob
                    ):
                        matched_actual.add(act_diff)
                        found_any = True

            if change_type == "MAYBE-MODIFIED":
                # Always considered "found" unless DELETED (which would be a separate diff)
                # But we need to verify the file actually exists if it wasn't modified.
                # Since actual_diffs only has changes, we can't check existence here easily without re-scanning.
                # However, assert_dir_diff works on snapshots.
                # If it's not in actual_diffs, it means it's same as old state (so it exists) OR it was deleted (would be in actual_diffs as DELETED).

                # Check if we found a MODIFIED match
                if found_any:
                    pass
                else:
                    # Check if it was DELETED
                    is_deleted = False
                    for act_diff in actual_diffs:
                        if act_diff.endswith(" DELETED"):
                            act_path = act_diff.rsplit(" ", 1)[0]
                            if fnmatch.fnmatch(act_path, path_glob):
                                is_deleted = True
                                break

                    if is_deleted:
                        # If deleted, MAYBE-MODIFIED fails (it implies existence)
                        unmatched_expected.add(exp_pattern)
                    else:
                        # Not modified and not deleted -> Exists and unchanged -> OK
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

    def assert_coherent_performance_profile(self):
        """Asserts that .profile-<machine name> has no .jsonl files but has .txt summary."""
        import platform
        from unittest.mock import patch

        from train.profile import get_profile_dir

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            profile_dir = get_profile_dir()

        if not profile_dir:
            # Profiling disabled
            return

        self.test_case.assertTrue(
            os.path.exists(profile_dir),
            f"Profile directory {profile_dir} should exist",
        )

        # Verify directory naming invariant (.profile-<hostname>)
        hostname = platform.node().split(".")[0]
        expected_dirname = f".profile-{hostname}"
        self.test_case.assertEqual(
            os.path.basename(profile_dir),
            expected_dirname,
            f"Profile directory name mismatch. Expected {expected_dirname}, got {os.path.basename(profile_dir)}",
        )

        # Check for jsonl files (should be gone)
        jsonl_files = glob.glob(os.path.join(profile_dir, "*.jsonl"))
        self.test_case.assertEqual(
            len(jsonl_files),
            0,
            f"Profile directory should not contain .jsonl files, found: {jsonl_files}",
        )

        # Check for txt summary reports (should be present) and contain required sections
        txt_files = glob.glob(os.path.join(profile_dir, "*.txt"))
        self.test_case.assertTrue(
            len(txt_files) > 0,
            f"Profile directory {profile_dir} should contain .txt summary report",
        )

        for txt_file in txt_files:
            # We only care about checking the cProfile outputs (train_style_*.txt),
            # not necessarily the custom "training-profile.txt" aggregate (though it's good if valid).
            # The cProfile ones are named train_style_<pid>.txt
            if "train_style_" in os.path.basename(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.test_case.assertIn(
                        "TOP 50 BY INVOCATION COUNT",
                        content,
                        f"Profile {txt_file} missing 'TOP 50 BY INVOCATION COUNT'",
                    )
                    self.test_case.assertIn(
                        "TOP 50 BY CUMULATIVE TIME",
                        content,
                        f"Profile {txt_file} missing 'TOP 50 BY CUMULATIVE TIME'",
                    )


def setup_mock_style_model(test_case):
    """Sets up a mock style model and tokenizer for testing analysis."""
    from unittest.mock import patch

    from kotogram.model import ModelConfig, StyleClassifier
    from kotogram.tokenizer import Tokenizer

    # Create dummy tokenizer
    test_case.tokenizer = Tokenizer()
    # pylint: disable=protected-access
    test_case.tokenizer._frozen = True

    # Create dummy model
    config = ModelConfig(vocab_sizes=test_case.tokenizer.get_vocab_sizes())
    test_case.model = StyleClassifier(config)
    test_case.model.eval()

    # Patch the internal loader
    patcher = patch(
        "kotogram.analysis._load_style_model",
        return_value=(test_case.model, test_case.tokenizer),
    )
    patcher.start()
    test_case.addCleanup(patcher.stop)
