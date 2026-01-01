import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import pytest

PYTHON_BASELINE = "tests/python_package_baseline.txt"
TS_BASELINE = "tests/typescript_package_baseline.txt"


class TestPackageHygiene:
    def run_command(self, cmd):
        return subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_python_package_integrity(self):
        """
        Verify the Python package build artifact integrity.
        """
        # Create a temporary directory for building.
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy necessary files
            if os.path.exists("pyproject.toml"):
                shutil.copy("pyproject.toml", temp_dir)
            if os.path.exists("README.md"):
                shutil.copy("README.md", temp_dir)
            if os.path.exists("LICENSE"):
                shutil.copy("LICENSE", temp_dir)

            shutil.copytree("kotogram", os.path.join(temp_dir, "kotogram"))

            # Create dummy models/style to satisfy build config if it expects it
            # The error said: error: package directory 'models/style' does not exist
            # This implies setup config expects it.
            # AND pyproject.toml maps "kotogram.model_data" = "models/style"
            # So we must put files in models/style for them to appear in kotogram/model_data in the wheel.
            models_style_dir = os.path.join(temp_dir, "models/style")
            os.makedirs(models_style_dir, exist_ok=True)
            # Create an empty init and data files
            with open(
                os.path.join(models_style_dir, "__init__.py"), "w", encoding="utf-8"
            ) as f:
                f.write("")

            for fname in [
                "labels.json",
                "model.json",
                "model.pt",
                "model_type.txt",
                "tokenizer.json",
            ]:
                with open(
                    os.path.join(models_style_dir, fname), "w", encoding="utf-8"
                ) as f:
                    f.write("")

            # Run build in temp dir
            res = self.run_command(
                f"python3 -m build --no-isolation --outdir {temp_dir}/dist_py {temp_dir}"
            )
            if res.returncode != 0:
                pytest.fail(f"Build failed:\n{res.stdout}{res.stderr}")

            dist_dir = os.path.join(temp_dir, "dist_py")
            if not os.path.exists(dist_dir):
                pytest.fail("dist_py not found")

            whls = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
            if not whls:
                pytest.fail("No wheel file generated")
            whl_path = os.path.join(dist_dir, whls[0])

            with zipfile.ZipFile(whl_path, "r") as z:
                files = z.namelist()

        # Normalize file names to avoid version-dependent diffs
        # e.g., kotogram-0.1.0.dist-info -> kotogram-*.dist-info
        norm_files = []
        for f in files:
            f = re.sub(r"kotogram-.*\.dist-info", "kotogram-*.dist-info", f)
            f = f.replace(
                "kotogram-*.dist-info/licenses/LICENSE", "kotogram-*.dist-info/LICENSE"
            )
            norm_files.append(f)

        norm_files.sort()

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("\n".join(norm_files) + "\n")
            tmp_path = tmp.name

        try:
            diff_res = self.run_command(f"diff -u {PYTHON_BASELINE} {tmp_path}")
            if diff_res.returncode != 0:
                pytest.fail(
                    f"Python package contents do not match baseline!\n{diff_res.stdout}"
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_typescript_package_integrity(self):
        """
        Verify the TypeScript package build artifact integrity.
        """
        if not os.path.exists("package.json"):
            pytest.skip("Skipped (no package.json)")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy minimal files for TS build
            shutil.copy("package.json", temp_dir)
            if os.path.exists("tsconfig.json"):
                shutil.copy("tsconfig.json", temp_dir)
            if os.path.exists("tsconfig.dist.json"):
                shutil.copy("tsconfig.dist.json", temp_dir)

            # Strategy: Symlink node_modules to avoid copy.
            if os.path.exists("node_modules"):
                os.symlink(
                    os.path.abspath("node_modules"),
                    os.path.join(temp_dir, "node_modules"),
                )

            # Copy source. I don't know exact source layout from here.
            # I will copy known files and dirs that look like source, excluding dist/
            for item in os.listdir("."):
                if item in [
                    "dist",
                    "dist_py",
                    "__pycache__",
                    ".git",
                    ".venv",
                    "node_modules",
                ]:
                    continue
                if os.path.isfile(item):
                    shutil.copy(item, temp_dir)
                elif os.path.isdir(item):
                    # Copy directory.
                    shutil.copytree(item, os.path.join(temp_dir, item))

            # Run build in temp dir
            cwd_arg = f"cd {temp_dir} && "

            # Isolate npm cache to avoid confinement issues (~/.npm permission)
            # We simply set npm_config_cache environment variable to a subdir of temp_dir
            npm_env = os.environ.copy()
            npm_env["npm_config_cache"] = os.path.join(temp_dir, ".npm_cache")

            # We need to pass this environment to the shell commands.
            # Since shell=True, we can prepend env vars or use env argument.
            # But run_command uses subprocess.run.
            # I need to update run_command to accept env.
            # For now, I will prepend the export to the command string for simplicity in shell.
            # export npm_config_cache=... && ...
            env_setup = f"export npm_config_cache={temp_dir}/.npm_cache && "

            res = self.run_command(f"{env_setup}{cwd_arg}npm run build")
            if res.returncode != 0:
                pytest.fail(f"npm build failed:\n{res.stdout}{res.stderr}")

            res = self.run_command(f"{env_setup}{cwd_arg}npm pack --quiet")
            if res.returncode != 0:
                pytest.fail(f"npm pack failed:\n{res.stdout}{res.stderr}")

            pack_file_name = res.stdout.strip().splitlines()[-1]
            pack_file = os.path.join(temp_dir, pack_file_name)

            try:
                res = self.run_command(f"tar -tf {pack_file}")
                if res.returncode != 0:
                    pytest.fail(f"tar failed:\n{res.stdout}{res.stderr}")

                files = res.stdout.strip().splitlines()
                # Normalize paths: 'package/lib/index.js' -> 'lib/index.js'
                norm_files = sorted([f.replace("package/", "", 1) for f in files])

                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
                    tmp.write("\n".join(norm_files) + "\n")
                    tmp_path = tmp.name

                try:
                    diff_res = self.run_command(f"diff -u {TS_BASELINE} {tmp_path}")
                    if diff_res.returncode != 0:
                        pytest.fail(
                            f"TypeScript package contents do not match baseline!\n{diff_res.stdout}"
                        )
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            finally:
                pass
