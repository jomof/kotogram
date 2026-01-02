import unittest

from training_test_utils import Bottle

from train import history


class TestKCTargetVerification(unittest.TestCase):
    def test_kc_targets_are_trained(self):
        # Using Bottle ensures correct environment setup
        with Bottle(self) as bottle:
            bottle.populate_test_data()

            # Labeling is handled automatically by train_style if needed

            # Train with specific KC targets
            # Use minimal arguments for speed
            cmd = "--pretrain-kc --kc-epochs 1 --epochs 0 --kc-freeze-encoder-epochs 0 "

            bottle.train_style(cmd)

            # Get history directly from bottle
            events = bottle.get_epoch_history()
            kc_events = [e for e in events if isinstance(e, history.KcEpochEvent)]

            self.assertTrue(len(kc_events) > 0, "No KC epoch events found")

            last_event = kc_events[-1]
            metrics = last_event.metrics

            # Check active_kc_targets
            self.assertIn(
                "active_kc_targets", metrics, "active_kc_targets not in metrics"
            )

            active_str = metrics["active_kc_targets"]
            # Convert to list for analysis
            active_targets = active_str.split(",") if active_str else []
            active_set = set(active_targets)

            expected_subset = {
                "bag_reading",
                "tail_reading",
                "ngram_reading",
                "tail_ngram_reading",
                "bag_pos",
                "tail_pos",
                "ngram_pos",
                "tail_ngram_pos",
                "bag_conjugated_form",
                "tail_conjugated_form",
                "ngram_conjugated_form",
                "tail_ngram_conjugated_form",
                "pair_pos_conj",
            }

            missing = expected_subset - active_set
            self.assertEqual(
                len(missing),
                0,
                f"Missing expected active targets: {missing}. Found: {active_set}",
            )


if __name__ == "__main__":
    unittest.main()
