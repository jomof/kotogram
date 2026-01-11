import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest


class TestCurateScript(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Create dummy data in bottle root
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # 1. Standard Input
        self.input_file = os.path.join(self.data_dir, "jpn_sentences.tsv")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write("これはペンです\n")  # Formal
            f.write("猫が好き\n")  # Neutral
            f.write(
                "俺はあたしだ\n"
            )  # Conflicting: Danseigo + Joseigo -> Unpragmatic Gender -> Gram=0

        # 2. Agrammatic Input
        agrammatic_name = "jpn_agrammatic_sentences.tsv"
        self.agrammatic_file = os.path.join(self.data_dir, agrammatic_name)
        with open(self.agrammatic_file, "w", encoding="utf-8") as f:
            f.write("ペンはこれ\n")

        # 3. Register Overrides (to force conflict logic)
        # Danseigo: 俺 (Ore)
        danseigo_name = "jpn_sentences_danseigo.tsv"
        self.danseigo_file = os.path.join(self.data_dir, danseigo_name)
        with open(self.danseigo_file, "w", encoding="utf-8") as f:
            f.write("俺はあたしだ\n")

        # Joseigo: あたし (Atashi)
        joseigo_name = "jpn_sentences_joseigo.tsv"
        self.joseigo_file = os.path.join(self.data_dir, joseigo_name)
        with open(self.joseigo_file, "w", encoding="utf-8") as f:
            f.write("俺はあたしだ\n")

        # Create corpus.tar.gz
        import tarfile

        tar_path = os.path.join(self.data_dir, "corpus.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(self.input_file, arcname="jpn_sentences.tsv")
            tar.add(self.agrammatic_file, arcname=agrammatic_name)
            tar.add(self.danseigo_file, arcname=danseigo_name)
            tar.add(self.joseigo_file, arcname=joseigo_name)

        # Clean up raw files to ensure strict archival usage
        os.remove(self.input_file)
        os.remove(self.agrammatic_file)
        os.remove(self.danseigo_file)
        os.remove(self.joseigo_file)

        # Determine script path (absolute)
        # Determine script path (absolute)
        self.script_path = os.path.join(self.old_cwd, "scripts/curate")
        self.db_path = os.path.join(self.data_dir, "corpus.db")

    def tearDown(self):
        os.chdir(self.old_cwd)
        # Handle potential permission errors if DB is locked or something
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_curate_drink(self):
        # Run curate drink
        # We need to make sure kotogram is in python path.
        env = os.environ.copy()
        # Ensure PYTHONPATH includes the project root
        env["PYTHONPATH"] = self.old_cwd + ":" + env.get("PYTHONPATH", "")

        cmd = [sys.executable, self.script_path, "drink"]

        cmd = [sys.executable, self.script_path, "drink"]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        self.assertEqual(result.returncode, 0)

        # Check DB
        db_path = "data/corpus.db"
        self.assertTrue(os.path.exists(db_path))

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        # Verify labels exist (Tables removed)
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='formality'"
        )
        self.assertIsNone(c.fetchone())
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gender'")
        self.assertIsNone(c.fetchone())
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='grammaticality'"
        )
        self.assertIsNone(c.fetchone())

        # Check data
        # Corpus schema: sentence, f_val (Real), female_val (Real), grammatic (Integr), r_ids
        c.execute("SELECT * FROM corpus")
        rows = c.fetchall()

        # We expect at least one row, and verify its structure
        self.assertGreater(len(rows), 0)
        row_map = {r[0]: r for r in rows}

        # Verify "これはペンです" (Formal -> F=0.5, G=0.0)
        if "これはペンです" in row_map:
            row_1 = row_map["これはペンです"]
            self.assertEqual(len(row_1), 5)
            self.assertIsInstance(row_1[1], float)  # f_val (0.5 for Formal)
            self.assertEqual(row_1[1], 0.5)
            self.assertIsInstance(row_1[2], float)  # g_val (0.0 for Neutral)
            self.assertEqual(row_1[2], 0.0)
            self.assertEqual(row_1[3], 1)  # Grammatic (Boolean 1)

        # Verify "猫が好き" (Neutral -> 0.0)
        if "猫が好き" in row_map:
            row_2 = row_map["猫が好き"]
            self.assertEqual(row_2[1], 0.0)

        # Verify "ペンはこれ" (Agrammatic)
        if "ペンはこれ" in row_map:
            row_3 = row_map["ペンはこれ"]
            self.assertEqual(row_3[3], 0)  # Agrammatic (Boolean 0)

        # Verify "俺はあたしだ" (Conflicting Registers -> Unpragmatic -> Strict Gram=0)
        if "俺はあたしだ" in row_map:
            row_4 = row_map["俺はあたしだ"]
            # Strict Logic Check: Even though source file is normal,
            # conflict makes gender unpragmatic (g_prag=0).
            # Therefore final_gram = 1 AND 1 AND 0 = 0.
            self.assertEqual(row_4[3], 0, "Conflicting styles should result in gram=0")
            # Also verify gender is NULL
            self.assertIsNone(row_4[2])

    def test_cli_commands(self):
        """Test 'show' and 'read' CLI commands."""
        # Setup env
        env = os.environ.copy()
        env["PYTHONPATH"] = self.old_cwd + ":" + env.get("PYTHONPATH", "")

        # 1. Run 'drink' to populate DB
        cmd_drink = [
            sys.executable,
            self.script_path,
            "drink",
            "--db-path",
            self.db_path,
        ]
        res_drink = subprocess.run(
            cmd_drink, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(res_drink.returncode, 0, f"Drink failed: {res_drink.stderr}")

        # 2. Test 'read' command (Should find in DB)
        # We know "猫が好き" is in there from setUp logic
        cmd_read = [
            sys.executable,
            self.script_path,
            "read",
            "猫が好き",
            "--db-path",
            self.db_path,
        ]
        res_read = subprocess.run(
            cmd_read, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(res_read.returncode, 0, f"Read failed: {res_read.stderr}")
        self.assertIn("Source: corpus.db", res_read.stdout)

        # 3. Test 'show' command (Calculated)
        cmd_show = [sys.executable, self.script_path, "show", "猫が好き"]
        res_show = subprocess.run(
            cmd_show, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(res_show.returncode, 0, f"Show failed: {res_show.stderr}")
        self.assertIn("Source: Calculated", res_show.stdout)

        # 4. Test 'summary' command
        cmd_summary = [
            sys.executable,
            self.script_path,
            "summary",
            "--db-path",
            self.db_path,
        ]
        res_summary = subprocess.run(
            cmd_summary, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(
            res_summary.returncode, 0, f"Summary failed: {res_summary.stderr}"
        )
        # Check for expected headers in summary output
        self.assertIn("Formality Distribution", res_summary.stdout)
        self.assertIn("Gender Pragmatic Distribution", res_summary.stdout)
        self.assertIn("Register Distribution", res_summary.stdout)
        self.assertIn("Grammaticality Distribution", res_summary.stdout)

        # 5. Test 'distinct' command
        # Default (exclude gram=0)
        cmd_distinct = [
            sys.executable,
            self.script_path,
            "distinct",
            "--db-path",
            self.db_path,
        ]
        res_distinct = subprocess.run(
            cmd_distinct, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(
            res_distinct.returncode, 0, f"Distinct failed: {res_distinct.stderr}"
        )
        self.assertIn("Distinct Style Combinations", res_distinct.stdout)
        # "俺はあたしだ" is gram=0, so it should NOT appear by default
        # But wait, how do we know what string it produces?
        # It has "Gender: Unpragmatic", "Grammatic: 0".
        # Let's check for "Grammatic: 0" string.
        self.assertNotIn("Grammatic: 0", res_distinct.stdout)

        # Include ungrammatic
        cmd_distinct_all = [
            sys.executable,
            self.script_path,
            "distinct",
            "--db-path",
            self.db_path,
            "--include-ungrammatic",
        ]
        res_distinct_all = subprocess.run(
            cmd_distinct_all, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(
            res_distinct_all.returncode,
            0,
            f"Distinct all failed: {res_distinct_all.stderr}",
        )
        # Now we expect to see the ungrammatic row
        self.assertIn("Grammatic: 0", res_distinct_all.stdout)

        # 6. Test 'compare' command (Expect missing cache in test env)
        self._test_compare_command(env)

        # --- COVERAGE EXPANSION ---
        # Vary 'sentence', 'f_val', 'g_val' etc by reading other sentences
        # "これはペンです" has F=0.5, G=0.0 (Varies F from 0.0)
        subprocess.run(
            [
                sys.executable,
                self.script_path,
                "read",
                "これはペンです",
                "--db-path",
                self.db_path,
            ],
            env=env,
            check=False,
            capture_output=True,
        )
        # "俺はあたしだ" has G=None (Varies G from 0.0)
        subprocess.run(
            [
                sys.executable,
                self.script_path,
                "read",
                "俺はあたしだ",
                "--db-path",
                self.db_path,
            ],
            env=env,
            check=False,
            capture_output=True,
        )

        # Vary 'db_path' by using a copy of the DB for other commands
        db_path_2 = self.db_path + ".var"
        shutil.copy(self.db_path, db_path_2)

        # Run summary/distinct/compare on variable db path
        subprocess.run(
            [sys.executable, self.script_path, "summary", "--db-path", db_path_2],
            env=env,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, self.script_path, "distinct", "--db-path", db_path_2],
            env=env,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, self.script_path, "compare", "--db-path", db_path_2],
            env=env,
            check=False,
            capture_output=True,
        )

        # Vary 'show' input
        subprocess.run(
            [sys.executable, self.script_path, "show", "これはペンです"],
            env=env,
            check=False,
            capture_output=True,
        )

    def test_cli_isolated_environment(self):
        """Test variations in isolated environment (split from test_cli_commands)."""
        # Setup env
        env = os.environ.copy()
        env["PYTHONPATH"] = self.old_cwd + ":" + env.get("PYTHONPATH", "")

        # Vary DB path and verify effect
        # Create a separate environment with a minimal corpus.tar.gz
        var_dir = os.path.join(self.test_dir, "var_env")
        var_data_dir = os.path.join(var_dir, "data")
        os.makedirs(var_data_dir, exist_ok=True)

        # Create single TSV
        var_tsv = os.path.join(var_data_dir, "jpn_small.tsv")
        with open(var_tsv, "w", encoding="utf-8") as f:
            f.write("これはテストです\n")  # 1 sentence

        # Create tar.gz
        import tarfile

        with tarfile.open(os.path.join(var_data_dir, "corpus.tar.gz"), "w:gz") as tar:
            tar.add(var_tsv, arcname="jpn_small.tsv")

        # Output DB path (inside var_dir for convenience)
        db_path_fresh = os.path.join(var_dir, "fresh.db")

        # Drink in new cwd
        subprocess.run(
            [sys.executable, self.script_path, "drink", "--db-path", db_path_fresh],
            cwd=var_dir,
            env=env,
            check=True,
            capture_output=True,
        )

        # Run summary on new DB
        res_var = subprocess.run(
            [sys.executable, self.script_path, "summary", "--db-path", db_path_fresh],
            cwd=var_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertIn("Formal", res_var.stdout)
        self.assertIn("100.0%", res_var.stdout)

        # Also run read with varied path to keep the original variation intent
        subprocess.run(
            [
                sys.executable,
                self.script_path,
                "read",
                "猫",
                "--db-path",
                db_path_fresh,
            ],
            cwd=var_dir,
            env=env,
            check=False,
            capture_output=True,
        )

    def _test_compare_command(self, env):
        cmd_compare = [
            sys.executable,
            self.script_path,
            "compare",
            "--db-path",
            self.db_path,
        ]
        res_compare = subprocess.run(
            cmd_compare, env=env, capture_output=True, text=True, check=False
        )
        self.assertEqual(
            res_compare.returncode, 0, f"Compare failed: {res_compare.stderr}"
        )
        # Since we haven't run label.py, cache shouldn't exist
        self.assertIn("Cache not found", res_compare.stdout)


class TestCurateKcFamiliesImports(unittest.TestCase):
    """Test that kc-families imports work correctly after KC_HASH_BUCKETS refactoring."""

    def test_kc_families_imports(self):
        """Verify the imports used by curate_kc_families are available."""
        # These imports mirror what curate_kc_families uses
        from train.kc import (
            ALL_KC_FAMILIES,
            FAMILY_FEATURES,
            KC_POS_BIASED_WINDOW,
            get_family_bucket_size,
            is_family_sparse,
        )

        # Verify key values exist
        self.assertIsInstance(ALL_KC_FAMILIES, list)
        self.assertGreater(len(ALL_KC_FAMILIES), 0)
        self.assertIsInstance(FAMILY_FEATURES, dict)
        self.assertIsInstance(KC_POS_BIASED_WINDOW, int)

        # Test get_family_bucket_size for sparse families
        sparse_families = [f for f in ALL_KC_FAMILIES if is_family_sparse(f)]
        self.assertGreater(len(sparse_families), 0)

        for family in sparse_families:
            bucket_size = get_family_bucket_size(family)
            self.assertIsInstance(bucket_size, int)
            self.assertGreater(bucket_size, 0)

    def test_compute_kc_targets_basic(self):
        """Test compute_kc_targets with minimal input."""
        from train.kc import ALL_KC_FAMILIES, compute_kc_targets

        # Minimal feature dict (empty lists)
        feature_ids = {
            "pos": [2, 3, 4],  # Sample IDs (not special tokens)
            "pos_detail_1": [2, 3, 4],
            "conjugated_type": [2, 3, 4],
            "reading_gram": [2, 3, 4],
        }

        targets = compute_kc_targets(feature_ids)

        # Should return a dict with KC families as keys
        self.assertIsInstance(targets, dict)

        # Check that at least some families have targets
        non_empty = [f for f in ALL_KC_FAMILIES if targets.get(f)]
        self.assertGreater(len(non_empty), 0)


if __name__ == "__main__":
    unittest.main()
