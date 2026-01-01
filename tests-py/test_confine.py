import os
import tempfile
import unittest
from unittest.mock import patch

# pylint: disable=import-private-name
from lib_confine import _build_command, confine


class TestConfine(unittest.TestCase):
    def test_mac_profile_generation(self):
        """Test that macOS generates a sandbox-exec command with correct profile."""
        with patch("platform.system", return_value="Darwin"):
            # Patch existence of sandbox-exec and success of (version 1) probe
            with patch("os.path.exists", return_value=True):
                with patch("subprocess.call", return_value=0):
                    # Test default: no network
                    cmd = _build_command(
                        ["ls", "-l"],
                        allow_network=False,
                        allow_read=None,
                        allow_write=None,
                    )
                    self.assertEqual(cmd[0], "sandbox-exec")
                    self.assertEqual(cmd[1], "-p")
                    profile = cmd[2]
                    self.assertIn("(version 1)", profile)
                    self.assertIn("(deny default)", profile)
                    self.assertIn("(deny network*)", profile)
                    self.assertEqual(cmd[3:], ["ls", "-l"])

                    # Test allow network
                    cmd_net = _build_command(
                        ["curl"], allow_network=True, allow_read=None, allow_write=None
                    )
                    profile_net = cmd_net[2]
                    self.assertIn("(version 1)", profile_net)
                    self.assertNotIn("(deny network*)", profile_net)
                    self.assertIn("(allow network*)", profile_net)

    def test_linux_pass_through(self):
        """Test that Linux simply passes through commands (no-op sandbox)."""
        with patch("platform.system", return_value="Linux"):
            cmd = _build_command(
                ["bash"], allow_network=False, allow_read=None, allow_write=None
            )
            self.assertEqual(cmd, ["bash"])

    def test_confine_run_mode(self):
        """Test confine in default 'run' mode."""
        config = {"allow_network": True, "allow_write": ["[exec-root]/output"]}
        with patch(
            "lib_confine._build_command", return_value=["sandbox", "ls"]
        ) as mock_build:
            with patch("subprocess.run") as mock_run:
                confine(["ls"], config, env={"TEST": "1"}, cwd="/custom/cwd")

                # Check variable expansion in config passed to build
                mock_build.assert_called_once()
                args, _ = mock_build.call_args
                self.assertEqual(
                    args[3], [os.path.realpath("/custom/cwd/output")]
                )  # allow_write expanded

                # Check subprocess.run call
                mock_run.assert_called_once()
                run_args, run_kwargs = mock_run.call_args
                self.assertEqual(run_args[0], ["sandbox", "ls"])
                self.assertEqual(run_kwargs["env"], {"TEST": "1"})
                self.assertEqual(run_kwargs["cwd"], "/custom/cwd")

    def test_confine_exec_mode(self):
        """Test confine in 'exec' mode."""
        config = {"mode": "exec", "allow_network": False}
        with patch("lib_confine._build_command", return_value=["sandbox", "ls"]):
            with patch("os.execvpe") as mock_exec:
                with patch("os.chdir") as mock_chdir:
                    with patch("os.environ", {"PATH": "/bin"}):
                        confine(["ls"], config, cwd="/new/dir")

                        mock_chdir.assert_called_with("/new/dir")
                        mock_exec.assert_called_once()
                        exec_args = mock_exec.call_args[0]
                        self.assertEqual(exec_args[0], "sandbox")
                        self.assertEqual(exec_args[1], ["sandbox", "ls"])

    def test_variable_expansion_defaults(self):
        """Test variable expansion uses os.getcwd() if cwd not provided."""
        config = {"allow_read": ["[exec-root]/file", "[tmp]/temp"]}
        with patch("lib_confine._build_command") as mock_build:
            with patch("subprocess.run"):
                with patch("os.getcwd", return_value="/current/dir"):
                    confine(["ls"], config)

                    mock_build.assert_called_once()
                    args = mock_build.call_args[0]
                    allow_read = args[2]
                    self.assertIn(os.path.realpath("/current/dir/file"), allow_read)
                    self.assertIn(
                        os.path.realpath(f"{tempfile.gettempdir()}/temp"), allow_read
                    )


if __name__ == "__main__":
    unittest.main()
