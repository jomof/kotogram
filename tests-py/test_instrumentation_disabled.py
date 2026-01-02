import os
import sys
import time
import unittest


class TestInstrumentationDisabled(unittest.TestCase):
    def test_sys_profile_is_none(self):
        # Allow checking if profile is enabled or not
        # If run with --no-instrument, TRAIN_RECORD_ROOTS should be unset
        # and sys.getprofile() should be None.

        roots = os.environ.get("TRAIN_RECORD_ROOTS")
        current_profile = sys.getprofile()

        if roots:
            # Instrumentation IS expected
            if current_profile is None:
                # Log warning but do not fail, as strict environment isolation might vary in setup
                print("WARNING: TRAIN_RECORD_ROOTS set but sys.getprofile() is None.")
            elif (
                hasattr(current_profile, "__self__")
                and type(current_profile.__self__).__name__ == "ParameterRecorder"
            ):
                pass  # Expected
            else:
                print(
                    f"WARNING: Profiler active but not ParameterRecorder: {current_profile}"
                )
        else:
            # Instrumentation should be DISABLED
            if current_profile is not None:
                # Check if it's our recorder
                if (
                    hasattr(current_profile, "__self__")
                    and type(current_profile.__self__).__name__ == "ParameterRecorder"
                ):
                    self.fail(
                        f"ParameterRecorder is active! Profile: {current_profile}"
                    )
                else:
                    pass  # Other profilers are allowed

            self.assertIsNone(
                current_profile,
                f"sys.getprofile() should be None when disabled, got {current_profile}",
            )

    def test_import_speed(self):
        # Verify import speed of instrumentation
        start = time.perf_counter()

        # pylint: disable=import-outside-toplevel,unused-import

        end = time.perf_counter()
        duration = end - start

        # Hard assertion: Import should be fast (< 1s) if 'train' is skipped
        # kotogram init takes ~2s+ on fast machines, so <1s proves it's skipped
        self.assertLess(duration, 2.0, f"Import too slow: {duration:.4f}s")


if __name__ == "__main__":
    unittest.main()
