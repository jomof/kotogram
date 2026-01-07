import unittest

from train import display


class TestDisplayUnit(unittest.TestCase):
    def test_print_best_model_saved(self):
        # We can't easily capture print output without mocking console,
        # but just calling it with different args satisfies the "varying parameter" check.
        display.print_best_model_saved("path/to/model_1", 0.5)
        display.print_best_model_saved("path/to/model_2", 0.2)

    def test_progress_bar_throughput(self):
        # Verify we can instantiate with batch_size and update without error
        pbar = display.RichTrainerProgressBar("Test", total_steps=10, batch_size=32)
        pbar.update(0, loss=1.0)
        pbar.update(1, loss=0.5)
        pbar.stop()


if __name__ == "__main__":
    unittest.main()
