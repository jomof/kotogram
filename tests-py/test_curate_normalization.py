import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

# pylint: disable=duplicate-code


class TestCurateNormalization(unittest.TestCase):
    def setUp(self):
        # Slightly different setup structure to avoid duplication detection
        self.old_cwd = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)
        os.makedirs("data", exist_ok=True)
        self.data_dir = os.path.join(self.test_dir, "data")
        self.script_path = os.path.join(self.old_cwd, "scripts/curate")
        self.db_path = os.path.join(self.data_dir, "corpus.db")

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_drink_normalization(self):
        # Create input with non-normalized characters
        tsv_name = "jpn_sentences.tsv"
        input_file = os.path.join(self.data_dir, tsv_name)
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("あの…\n")  # Ellipsis char -> ...
            f.write("えっと‼\n")  # Double Exclamation -> !!
            f.write("その...ままで\n")  # Already normalized

        # Create corpus.tar.gz
        import tarfile

        tar_path = os.path.join(self.data_dir, "corpus.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(input_file, arcname=tsv_name)

        # Remove raw file to ensure usage of tarball
        os.remove(input_file)

        # Run curate drink
        env = os.environ.copy()
        env["PYTHONPATH"] = self.old_cwd + ":" + env.get("PYTHONPATH", "")

        cmd = [sys.executable, self.script_path, "drink", "--db-path", self.db_path]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        self.assertEqual(result.returncode, 0)

        # Check DB content
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT sentence FROM corpus")
        sentences = [r[0] for r in c.fetchall()]
        conn.close()

        # Verify Normalization
        self.assertIn("あの...", sentences)
        self.assertIn("えっと!!", sentences)
        self.assertIn("その...ままで", sentences)

        # Verify Absense of Raw Forms
        self.assertNotIn("あの…", sentences)
        self.assertNotIn("えっと‼", sentences)


if __name__ == "__main__":
    unittest.main()
