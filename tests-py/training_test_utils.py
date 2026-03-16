# pylint: disable=too-many-lines
import contextlib
import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Set
from unittest.mock import patch

import torch

from kotogram import locations
from train import history
from train import paths as train_paths


# pylint: disable=too-many-positional-arguments
def train_style(
    test_case,
    script_path: str,
    project_root: str,
    args: str,
):
    """Runs the train_style.sh script with the given arguments and asserts success."""
    env = os.environ.copy()

    cmd = [script_path] + args.split()
    if script_path.endswith(".py"):
        cmd = [sys.executable, script_path] + args.split()

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


# pylint: disable=too-many-locals
def populate_test_data(root_dir: str, project_root: str):
    """Pre-populates test data in root_dir by sampling from data/corpus.db."""
    import sqlite3

    with patch.dict(os.environ, {"TRAIN_ROOT": root_dir}):
        data_dir = train_paths.get_data_dir()

    # Create data directory in the test root
    os.makedirs(data_dir, exist_ok=True)

    db_path = os.path.join(project_root, "data", "corpus.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Corpus database not found at {db_path}")

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    grammatic_samples: Set[str] = set()
    agrammatic_samples: Set[str] = set()

    def add_samples(condition: str, target_set: Set[str], count: int = 5):
        query = f"SELECT sentence FROM corpus WHERE {condition} LIMIT {count}"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            target_set.add(row[0])

    # 1. Gender Diversity
    # gender: -1.0 (Male), 1.0 (Female), 0.0 (Neutral), NULL (Unpragmatic)
    add_samples("gender = -1.0", grammatic_samples)
    add_samples("gender = 1.0", grammatic_samples)
    add_samples("gender = 0.0", grammatic_samples)
    add_samples("gender IS NULL", grammatic_samples)

    # 2. Register Diversity (All registers)
    register_samples: Dict[str, Set[str]] = {}

    # Read registers from the DB
    # Read registers from the DB
    cursor.execute("SELECT id, label FROM register")
    db_registers = cursor.fetchall()  # List of (id, label)

    for reg_id, reg_label_str in db_registers:
        reg_name = reg_label_str.lower()
        if reg_name not in register_samples:
            register_samples[reg_name] = set()

        # Query for sentences with this register
        add_samples(f"register_ids = '{reg_id}'", register_samples[reg_name])

    # 3. Formality Diversity (All 5 levels + Unpragmatic)
    # Formality levels seen: -1.0, -0.5, 0.0, 0.5, 1.0
    for f_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        add_samples(f"formality = {f_val}", grammatic_samples)
    add_samples("formality IS NULL", grammatic_samples)

    # 4. Grammaticality
    add_samples("grammatic = 1", grammatic_samples)
    add_samples("grammatic = 0", agrammatic_samples)

    # 5. Grammar Point Coverage (ensure we have sentences with grammar labels for PNU)
    add_samples("grammar IS NOT NULL AND grammar != ''", grammatic_samples)

    conn.close()

    # Write sampled data to corpus.db in data_dir for train_style to consume

    # Schema must match what label.py expects:
    # sentence, formality, gender, grammatic, register_ids (TEXT)

    new_db_path = os.path.join(data_dir, "corpus.db")

    # Check if target DB is already initialized or just delete it to be safe
    if os.path.exists(new_db_path):
        os.remove(new_db_path)

    # Re-open source DB to fetch full rows for the collected sentences
    source_conn = sqlite3.connect(db_path)
    source_c = source_conn.cursor()

    # Init target DB
    target_conn = sqlite3.connect(new_db_path)
    target_c = target_conn.cursor()
    target_c.execute(
        "CREATE TABLE corpus (sentence TEXT, formality REAL, gender REAL, grammatic INTEGER, register_ids TEXT, grammar TEXT, grammar_negative TEXT)"
    )

    # Create register table with mapping from kotogram.constants (source of truth)
    from kotogram.constants import REGISTER_ID_TO_LABEL

    target_c.execute("CREATE TABLE register (id INTEGER PRIMARY KEY, label TEXT)")
    register_data = [
        (id_val, label.name) for id_val, label in REGISTER_ID_TO_LABEL.items()
    ]
    target_c.executemany("INSERT INTO register VALUES (?, ?)", register_data)

    # Create grammar dictionary table (strict schema expected by scripts/label.py)
    # NOTE: back-compat is an anti-goal; tests must provision this table.
    target_c.execute(
        "CREATE TABLE grammar (id TEXT PRIMARY KEY, name TEXT NOT NULL, prior REAL)"
    )
    source_c.execute("SELECT id, name FROM grammar")
    grammar_rows = source_c.fetchall()
    target_c.executemany(
        "INSERT INTO grammar(id, name, prior) VALUES (?, ?, NULL)", grammar_rows
    )

    all_sentences = grammatic_samples.union(agrammatic_samples)
    for s in register_samples.values():
        all_sentences.update(s)

    # Fetch and insert
    for sent in all_sentences:
        # Fetch detailed info from source
        # Note: source corpus schema might differ slightly?
        # Source DB in `project_root/data/corpus.db` has likely full schema.
        # We need sentence, formality, gender, grammatic, register_ids.
        # register_ids might not be a column in source if it is old.
        # But let's try.
        source_c.execute(
            "SELECT sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative FROM corpus WHERE sentence = ?",
            (sent,),
        )
        row = source_c.fetchone()
        if row:
            target_c.execute("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?, ?)", row)
        else:
            # Fallback if sentence not found (unlikely)
            pass

    source_conn.close()
    target_conn.commit()
    target_conn.close()


def generate_test_chive(root_dir: str) -> None:
    """Write a tiny word2vec-format text file for test-only chiVe vectors."""
    import random as _rng

    from train.chive import CHIVE_DIM, get_chive_dir, get_chive_txt_path

    with patch.dict(os.environ, {"TRAIN_ROOT": root_dir}):
        chive_dir = get_chive_dir()
        txt_path = get_chive_txt_path()

    os.makedirs(chive_dir, exist_ok=True)

    # Common words likely to appear in any test corpus sample.
    words = [
        "する",
        "いる",
        "なる",
        "ある",
        "言う",
        "行く",
        "来る",
        "見る",
        "思う",
        "知る",
        "取る",
        "私",
        "彼",
        "彼女",
        "人",
        "事",
        "物",
        "所",
        "時",
        "年",
        "日",
        "月",
        "今日",
        "何",
        "方",
        "前",
        "後",
        "中",
        "上",
        "下",
        "大きい",
        "小さい",
        "良い",
        "悪い",
        "新しい",
        "多い",
        "少ない",
        "高い",
        "長い",
        "日本",
        "世界",
        "水",
        "手",
        "子供",
        "男",
        "女",
        "家",
        "学校",
        "仕事",
        "話",
        "言葉",
        "食べる",
        "飲む",
        "書く",
        "読む",
        "聞く",
        "話す",
        "走る",
        "歩く",
        "立つ",
        "座る",
        "猫",
        "犬",
        "花",
        "木",
        "空",
        "こんにちは",
        "です",
        "ます",
        "の",
        "が",
        "を",
        "に",
        "は",
        "と",
        "で",
        "も",
        "から",
        "まで",
        "よ",
        "ね",
        "か",
    ]

    _rng.seed(42)
    dim = CHIVE_DIM
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\n")
        for w in words:
            vec = [_rng.gauss(0, 0.3) for _ in range(dim)]
            f.write(f"{w} {' '.join(f'{v:.6f}' for v in vec)}\n")


# pylint: disable=too-many-locals
def assert_directory_matches_manifest(
    test_case, root_dir: str, expected_manifest: List[str]
):
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

    # pylint: disable=import-outside-toplevel
    from train.profile import get_profile_dir

    with patch.dict(os.environ, {"TRAIN_ROOT": root_dir}):
        cache_dir = train_paths.get_cache_dir()
        data_dir = train_paths.get_data_dir()
        models_dir = locations.get_models_dir()
        history_dir = train_paths.get_style_history_dir()
        profile_dir = get_profile_dir()

    # Check for duplicates in expected_manifest
    if len(expected_manifest) != len(set(expected_manifest)):
        duplicates = {x for x in expected_manifest if expected_manifest.count(x) > 1}
        test_case.fail(f"Duplicate entries found in expected_manifest: {duplicates}")

    # Get the paths relative to root_dir
    rel_cache_dir = os.path.relpath(cache_dir, root_dir)
    rel_data_dir = os.path.relpath(data_dir, root_dir)
    rel_models_dir = os.path.relpath(models_dir, root_dir)
    rel_history_dir = os.path.relpath(history_dir, root_dir)
    rel_profile_dir = (
        os.path.relpath(profile_dir, root_dir) if profile_dir else ".profile-disabled"
    )

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
        if "[history]" in p:
            p = p.replace("[history]", rel_history_dir)
        if "[.profile]" in p:
            p = p.replace("[.profile]", rel_profile_dir)
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
            path_for_exclusion = os.path.join(rel_root, file)
            # Skip .pyc files if unwanted (but explicit in manifest?)
            # Skip known logs if excluded?
            # We skip 'tmp*' directories commonly used for tests inside tests
            if "tmp" in path_for_exclusion.split(os.sep)[0]:
                continue

            actual_paths.append(path_for_exclusion)

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

    def __init__(
        self, test_case: unittest.TestCase, env: Optional[Dict[str, str]] = None
    ):
        self.test_case = test_case
        self.env = env.copy() if env else {}
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
                # Skip profiling artifacts (any file/dir starting with .profile)
                # if ".profile" in root or file.startswith(".profile"):
                #    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, self.root_dir)

                if os.path.exists(abs_path):
                    state[rel_path] = self._get_file_hash(abs_path)
        return state

    def snapshot(self, name: str) -> None:
        """Captures the current directory state as a named snapshot."""
        self._snapshots[name] = self._get_current_state()

    def __enter__(self):
        generate_test_chive(self.root_dir)
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.temp_dir.cleanup()

    def populate_test_data(self):
        """Populates the bottle with test data."""
        populate_test_data(self.root_dir, self.project_root)

    def calculate_expected_counts(self) -> Dict[str, int]:
        """Calculates expected sentence counts for KC pretraining.

        Queries the bottle's corpus.db to get ground truth counts.
        """
        import sqlite3

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            data_dir = train_paths.get_data_dir()
            db_path = os.path.join(data_dir, "corpus.db")

        if not os.path.exists(db_path):
            # Fallback if DB missing (shouldn't happen in valid test flow)
            return {
                "total_grammatic_sentences": 0,
                "grammatic_sentences_in_train_split": 0,
                "total_train_split_size": 0,
            }

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count total grammatical (label=1)
        cursor.execute("SELECT COUNT(*) FROM corpus WHERE grammatic = 1")
        total_grammatic = cursor.fetchone()[0]
        # Count total sentences
        cursor.execute("SELECT COUNT(*) FROM corpus")
        total_predictions = cursor.fetchone()[0]  # Total loaded dataset size

        conn.close()

        # Simulate StyleDataset.split(seed=42, train_ratio=0.8)
        # We need to simulate the split on the FULL dataset (grammatic + agrammatic).
        # But we don't know the exact order SQLite returns vs how StyleDataset loads?
        # StyleDataset reads offsets.bin which follows labeling order.
        # Labeling iterates corpus.db.
        # Assuming order is preserved (or stable sort?), we can try simulation.
        # But the split is random permutation.
        # If we just need total sizes:
        n_train = int(total_predictions * 0.8)

        # For "grammatic_sentences_in_train_split", we need exact indices.
        # This is hard without exact ordering.
        # However, test_train_style_script ONLY checks "total_grammatic_sentences" for proper KC.
        # It does NOT check "grammatic_sentences_in_train_split" for KC anymore (KC uses FULL dataset).
        # So we can omit or approximate the split metric if it's unused for KC.
        # Checking usage: test_train_style_script line 226 uses expected_counts["total_grammatic_sentences"].

        return {
            "total_grammatic_sentences": total_grammatic,
            "grammatic_sentences_in_train_split": 0,  # Unused for KC full dataset check
            "total_train_split_size": n_train,
        }

    def train_style(
        self,
        args: str,
    ):
        """Runs train_style.sh inside the bottle."""
        import re

        overrides = {"TRAIN_ROOT": self.root_dir}
        if self.env:
            if "TRAIN_ROOT" in self.env:
                raise ValueError(
                    "TRAIN_ROOT cannot be overridden in bottle.train_style()"
                )
            if "SKIP_DEPS" in self.env:
                raise ValueError(
                    "SKIP_DEPS cannot be overridden in bottle.train_style()"
                )
            overrides.update(self.env)

        # Prepare environment
        env = os.environ.copy()
        # Ensure imports work by adding project root to PYTHONPATH
        env["PYTHONPATH"] = (
            f"{self.project_root}:{os.path.join(self.project_root, 'tests-py')}:{env.get('PYTHONPATH', '')}"
        )
        env.update(overrides)

        cmd = [sys.executable, self.script_path] + args.split()
        if self.script_path.endswith(".py"):
            pass  # Already handled above

        # Run confined with CWD = bottle root to prevent write errors in project root
        result = subprocess.run(
            cmd,
            env=env,
            cwd=self.root_dir,
            check=False,
            capture_output=True,
            text=True,
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

        self.test_case.assertEqual(
            result.returncode,
            0,
            msg=f"Command failed with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        # Assert no warnings or errors in output
        # Use word boundary regex to avoid false positives like "mse_errors"
        combined = str(result.stdout or "") + str(result.stderr or "")

        # Filter out known safe warnings

        # Allow "Warning: Missing KC head weights" which is expected when upgrading checkpoints
        combined = re.sub(
            r"Warning: Missing KC head weights in checkpoint.*?(?=\n)",
            "",
            combined,
            flags=re.DOTALL,
        )

        # We match "warn" (case insensitive) to catch "Warning", "WARN", etc.
        warning_match = re.search(r"\bwarn", combined, re.IGNORECASE)
        error_match = re.search(r"\berror\b", combined, re.IGNORECASE)

        self.test_case.assertIsNone(
            warning_match,
            f"Found 'warn' in output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.test_case.assertIsNone(
            error_match,
            f"Found 'error' in output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        return result

    def kotogram_cli(self, *args: str):
        """Runs bin/kotogram inside the bottle."""
        bin_path = os.path.join(self.project_root, "bin", "kotogram")
        env = os.environ.copy()
        env["TRAIN_ROOT"] = self.root_dir
        # Ensure imports work by adding project root to PYTHONPATH
        env["PYTHONPATH"] = (
            f"{self.project_root}:{os.path.join(self.project_root, 'tests-py')}:{env.get('PYTHONPATH', '')}"
        )
        if self.env:
            env.update(self.env)

        cmd = [sys.executable, bin_path] + list(args)

        # Run confined with CWD = bottle root
        result = subprocess.run(
            cmd,
            env=env,
            cwd=self.root_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        self.test_case.assertEqual(
            result.returncode,
            0,
            msg=f"CLI failed with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        return result

    def run_script(
        self,
        rel_path: str,
        args: List[str],
        env_overrides: Optional[Dict[str, str]] = None,
    ):
        """Runs a python script inside the bottle.

        Args:
            rel_path: Relative path to the script from project root (e.g. 'scripts/label.py').
            args: List of command line arguments.
            env_overrides: Dictionary of environment variables to override.
        """
        script_path = os.path.join(self.project_root, rel_path)
        env = os.environ.copy()
        env["TRAIN_ROOT"] = self.root_dir
        # Ensure imports work for scripts not having path boilerplate (like scripts/label.py)
        env["PYTHONPATH"] = (
            f"{self.project_root}:{os.path.join(self.project_root, 'tests-py')}:{env.get('PYTHONPATH', '')}"
        )
        if env_overrides:
            env.update(env_overrides)

        cmd = [sys.executable, script_path] + args

        # Run confined with CWD = bottle root
        result = subprocess.run(
            cmd,
            env=env,
            cwd=self.root_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        # Helper logging on failure
        if result.returncode != 0:
            # Just print for visibility, let assertion handle failure
            print(f"Script failed: {cmd}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        self.test_case.assertEqual(
            result.returncode,
            0,
            msg=f"Script {rel_path} failed with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        return result

    @contextlib.contextmanager
    def environment(self):
        """Context manager that sets TRAIN_ROOT to the bottle's root for in-process checks."""

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            yield

    def assert_dir_layout(self, expected_manifest: List[str]):
        """Verifies the bottle's directory layout."""
        assert_directory_matches_manifest(
            self.test_case, self.root_dir, expected_manifest
        )

    def resolve_path(self, path: str) -> str:
        """Resolves placeholders in path and returns absolute path with bottle.

        Args:
            path: Path with placeholders like '[models]', '[data]', '[.cache]'.
        """

        # pylint: disable=import-outside-toplevel
        from train.profile import get_profile_dir

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            cache_dir = (
                train_paths.get_cache_dir()
            )  # Changed from locations.get_cache_dir()
            data_dir = (
                train_paths.get_data_dir()
            )  # Changed from locations.get_data_dir()
            models_dir = locations.get_models_dir()
            history_dir = train_paths.get_style_history_dir()
            profile_dir = get_profile_dir()

        rel_cache = os.path.relpath(cache_dir, self.root_dir)
        rel_data = os.path.relpath(data_dir, self.root_dir)
        rel_models = os.path.relpath(models_dir, self.root_dir)
        rel_history = os.path.relpath(history_dir, self.root_dir)
        rel_profile = (
            os.path.relpath(profile_dir, self.root_dir)
            if profile_dir
            else ".profile-disabled"
        )

        resolved = (
            path.replace("[.cache]", rel_cache)
            .replace("[data]", rel_data)
            .replace("[models]", rel_models)
            .replace("[history]", rel_history)
            .replace("[.profile]", rel_profile)
        )
        return os.path.join(self.root_dir, resolved)

    def get_file(self, path_template: str) -> str:
        """Alias for resolve_path."""
        return self.resolve_path(path_template)

    def assert_files_exist(self, paths: List[str]):
        """Asserts that all files in the list exist."""
        for p in paths:
            abs_path = self.resolve_path(p)
            self.test_case.assertTrue(os.path.exists(abs_path), f"File missing: {p}")

    # pylint: disable=invalid-name
    def assert_model_is_fp8(self, model_path: str):
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

    def get_epoch_history(self) -> List[history.HistoryEvent]:
        """Reads and returns the parsed content of training-history.tsv."""
        # Use resolve_path logic internally
        from train.paths import get_style_history_dir

        with patch.dict(os.environ, {"TRAIN_ROOT": self.root_dir}):
            history_dir = get_style_history_dir()

        history_path = os.path.join(history_dir, "training-history.tsv")
        return history.read_events(history_path)

    def assert_kc_epochs_trained(self, expected_epochs: List[int]):
        """Asserts that specific KC epoch numbers are present in history."""
        history_data = self.get_epoch_history()
        epochs_found = [
            e.epoch for e in history_data if isinstance(e, history.KcEpochEvent)
        ]
        self.test_case.assertEqual(
            epochs_found,
            expected_epochs,
            f"Expected KC epochs {expected_epochs}, found {epochs_found} in history.",
        )

    def assert_style_epochs_trained(self, expected_epochs: List[int]):
        """Asserts that specific Style epoch numbers are present in history."""
        history_data = self.get_epoch_history()
        epochs_found = [
            e.epoch for e in history_data if isinstance(e, history.StyleEpochEvent)
        ]
        self.test_case.assertEqual(
            epochs_found,
            expected_epochs,
            f"Expected Style epochs {expected_epochs}, found {epochs_found} in history.",
        )

    def assert_continuation_epochs(self, kc: int, style: int):
        """Asserts that continuation.json contains expected epoch counts."""
        cont_path = self.resolve_path("[models]/style/continuation.json")
        with open(cont_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.test_case.assertEqual(data["kc_epochs_trained"], kc)
        self.test_case.assertEqual(data["style_epochs_trained"], style)

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

            # pylint: disable=import-outside-toplevel
            from train.profile import get_profile_dir

        env_patch = {"TRAIN_ROOT": self.root_dir}
        if self.env:
            env_patch.update(self.env)
        with patch.dict(os.environ, env_patch):
            cache_dir = train_paths.get_cache_dir()
            data_dir = train_paths.get_data_dir()
            models_dir = locations.get_models_dir()
            history_dir = train_paths.get_style_history_dir()
            profile_dir = get_profile_dir()

        rel_cache = os.path.relpath(cache_dir, self.root_dir)
        rel_data = os.path.relpath(data_dir, self.root_dir)
        rel_models = os.path.relpath(models_dir, self.root_dir)
        rel_history = os.path.relpath(history_dir, self.root_dir)
        rel_profile = (
            os.path.relpath(profile_dir, self.root_dir)
            if profile_dir
            else ".profile-disabled"
        )

        resolved_expected = set()
        for diff in expected_diffs:
            resolved = (
                diff.replace("[.cache]", rel_cache)
                .replace("[data]", rel_data)
                .replace("[models]", rel_models)
                .replace("[history]", rel_history)
                .replace("[.profile]", rel_profile)
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
                        "TOP 50 BY CUMULATIVE TIME",
                        content,
                        f"Profile {txt_file} missing 'TOP 50 BY CUMULATIVE TIME'",
                    )

    def assert_no_nans_in_history(self):
        """Asserts that no metric values in training-history.tsv are NaN."""
        import math

        def check_no_nans(obj, path=""):
            if isinstance(obj, float):
                if math.isnan(obj):
                    self.test_case.fail(f"NaN found in history at '{path}'")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_nans(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_nans(v, f"{path}[{i}]")

        history_events = self.get_epoch_history()
        for event in history_events:
            if hasattr(event, "metrics"):
                check_no_nans(event.metrics, f"Epoch {event.epoch} metrics")
            if hasattr(event, "stats"):
                check_no_nans(event.stats, f"Epoch {event.epoch} stats")
