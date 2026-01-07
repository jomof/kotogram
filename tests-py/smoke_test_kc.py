import re
import sys

sys.path.append("tests-py")
import unittest

from training_test_utils import Bottle


class SmokeTest(unittest.TestCase):
    def test_smoke_auto_scaling(self):
        """Runs a short training session to verify saturation auto-scaling."""
        print("\n--- Starting Smoke Test for Saturation Auto-Scaling ---")

        # Run training for a few epochs to let sat_w ramp up
        # We need enough epochs to get past freeze_encoder_epochs (usually 1)
        # and let sat_w become > 0.

        with Bottle(self, env={"KOTOGRAM_FORCE_TERMINAL": "false"}) as bottle:
            bottle.populate_test_data()

            # Step 1: Label first (creates cached dataset)
            bottle.train_style("--label --force-relabel")

            # Step 2: Train
            res = bottle.train_style(
                "--pretrain-kc --kc-epochs=3 --kc-freeze-encoder-epochs=0 --batch-size=32"
            )

            log_path = bottle.get_file("[models]/style-support/training.log")
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()

        print("\n--- Training Output Dump (stdout) ---")
        print(res.stdout)
        print("\n--- Training Log Dump (file) ---")
        print(log_content)
        print("----------------------------")

        # Verify Saturation Stats in Output (check both)
        found_sat_line = False
        combined_output = (res.stdout or "") + "\n" + (log_content or "")

        # Search entire output for the saturation stats pattern, allowing for newlines (wrapping)
        # Pattern: Sat: ... alpha=... ... contrib/prim=...
        # We use re.DOTALL so '.' matches newlines
        match = re.search(
            r"Sat:.*?alpha=([\d\.]+).*?contrib/prim=([\d\.]+)%",
            combined_output,
            re.DOTALL,
        )

        if match:
            found_sat_line = True
            alpha_val = float(match.group(1))
            ratio_pct_val = float(match.group(2))

            print(f"  -> Found Sat Scaling: alpha={alpha_val}, ratio={ratio_pct_val}%")

            if alpha_val > 0.0:
                if ratio_pct_val == 0.0:
                    print(
                        "WARNING: Sat contribution is 0.0%. This might be okay if max(logits) < 3.0 everywhere."
                    )
                else:
                    print(
                        f"     Confirmed active saturation penalty (ratio={ratio_pct_val}%)"
                    )

        if not found_sat_line:
            # Create a helpful failure message with a snippet of log
            snippet = (
                combined_output[-2000:]
                if len(combined_output) > 2000
                else combined_output
            )
            self.fail(
                f"Did not find any 'Sat:' summary line with alpha/contrib stats.\n\nCaptured Output Snippet:\n{snippet}"
            )


if __name__ == "__main__":
    unittest.main()
