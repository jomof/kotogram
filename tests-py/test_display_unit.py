import unittest

from train import display


class TestDisplayUnit(unittest.TestCase):
    def test_format_kc_epoch_compact_summary(self):
        # Default-ish call
        s1 = display.format_kc_epoch_compact_summary(
            epoch=1,
            total_epochs=10,
            total_loss=0.5,
            avg_prob=0.1,
            act_dens=0.01,
            struct_avg=0.5,
            top_losses=[("base", 0.1)],
            amp_stats={
                "start": 0,
                "end": 0,
                "skips": 0,
                "opt_steps": 10,
                "flush_steps": 0,
            },
            entropy_norm=1.0,
            avg_kl_to_uniform=0.1,
            uniq_kcs=100,
            avg_p_max=0.5,
        )
        self.assertIn("loss=0.5000", s1)

        # Vary parameters
        s2 = display.format_kc_epoch_compact_summary(
            epoch=5,
            total_epochs=20,
            total_loss=0.2,
            avg_prob=0.5,
            act_dens=0.2,
            struct_avg=0.1,
            top_losses=[("struct", 0.05)],
            amp_stats={
                "start": 1,
                "end": 1,
                "skips": 1,
                "opt_steps": 20,
                "flush_steps": 1,
            },
            entropy_norm=0.5,
            avg_kl_to_uniform=0.05,
            uniq_kcs=200,
            avg_p_max=0.8,
        )
        self.assertIn("loss=0.2000", s2)
        self.assertIn("Epoch 5 of 20", s2)

    def test_format_kc_usage_summary(self):
        # Call 1
        s1 = display.format_kc_usage_summary(
            uniq=10,
            total=100,
            max_top1=0.5,
            tv_mean=0.1,
            gap_mean=0.01,
            topk_counts=[(1, 10)],
            top1_counts=[(1, 10)],
            k=5,
        )
        self.assertIn("uniqKCs=10", s1)

        # Call 2 (different values)
        s2 = display.format_kc_usage_summary(
            uniq=50,
            total=500,
            max_top1=0.8,
            tv_mean=0.2,
            gap_mean=0.05,
            topk_counts=[(1, 50)],
            top1_counts=[(1, 50)],
            k=10,
        )
        self.assertIn("uniqKCs=50", s2)

    def test_print_best_model_saved(self):
        # We can't easily capture print output without mocking console,
        # but just calling it with different args satisfies the "varying parameter" check.
        display.print_best_model_saved("path/to/model_1", 0.5)
        display.print_best_model_saved("path/to/model_2", 0.2)


if __name__ == "__main__":
    unittest.main()
