"""Tests for discrete accuracy in MSE family diagnostics."""

import torch

from train.kc_diagnostics import KCEpochDiag, discretize_mse


class TestDiscretizeMse:
    """Tests for discretize_mse bucket assignment."""

    def test_formality_five_buckets(self) -> None:
        """Formality thresholds: -0.75, -0.25, 0.25, 0.75 → 5 buckets."""
        values = torch.tensor([-1.0, -0.8, -0.5, -0.1, 0.0, 0.3, 0.5, 0.8, 1.0])
        buckets = discretize_mse(values, "formality")
        # bucket 0: < -0.75  → very_casual
        # bucket 1: [-0.75, -0.25) → casual
        # bucket 2: [-0.25,  0.25) → neutral
        # bucket 3: [ 0.25,  0.75) → formal
        # bucket 4: >= 0.75  → very_formal
        expected = torch.tensor([0, 0, 1, 2, 2, 3, 3, 4, 4])
        assert torch.equal(buckets, expected), f"Got {buckets.tolist()}"

    def test_gender_three_buckets(self) -> None:
        """Gender thresholds: -0.5, 0.5 → 3 buckets."""
        values = torch.tensor([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
        buckets = discretize_mse(values, "gender")
        # bucket 0: < -0.5  → masculine
        # bucket 1: [-0.5, 0.5) → neutral
        # bucket 2: >= 0.5  → feminine
        expected = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        assert torch.equal(buckets, expected), f"Got {buckets.tolist()}"

    def test_grammatic_binary(self) -> None:
        """Grammatic threshold: 0.5 → 2 buckets."""
        values = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        buckets = discretize_mse(values, "grammatic")
        # bucket 0: <= 0.5 → ungrammatical
        # bucket 1: > 0.5  → grammatical
        expected = torch.tensor([0, 0, 0, 1, 1])
        assert torch.equal(buckets, expected), f"Got {buckets.tolist()}"

    def test_unknown_family_all_same_bucket(self) -> None:
        """Unknown family falls back to single bucket (always matches)."""
        values = torch.tensor([-1.0, 0.0, 1.0])
        buckets = discretize_mse(values, "unknown_family")
        expected = torch.zeros(3, dtype=torch.long)
        assert torch.equal(buckets, expected)


class TestMseFamilyDiscreteAccuracy:
    """Tests for update_mse_family discrete accuracy accumulation."""

    def test_formality_same_bucket_counts_correct(self) -> None:
        """Predictions in the same formality bucket as targets count as correct."""
        diag = KCEpochDiag()
        # Both pred and target in neutral bucket [-0.25, 0.25)
        preds = torch.tensor([0.0, 0.1, -0.1])
        targets = torch.tensor([0.0, 0.2, -0.2])
        diag.update_mse_family("formality", preds, targets, loss=0.01)
        stats = diag.mse_families["formality"]
        assert stats.correct_discrete == 3

    def test_formality_different_bucket_not_correct(self) -> None:
        """Prediction in a different bucket from target does not count."""
        diag = KCEpochDiag()
        # pred=0.8 (very_formal), target=-0.8 (very_casual) → different
        preds = torch.tensor([0.8])
        targets = torch.tensor([-0.8])
        diag.update_mse_family("formality", preds, targets, loss=0.5)
        stats = diag.mse_families["formality"]
        assert stats.correct_discrete == 0

    def test_gender_accuracy_across_batches(self) -> None:
        """Discrete accuracy accumulates correctly across multiple batches."""
        diag = KCEpochDiag()
        # Batch 1: both neutral → correct
        diag.update_mse_family(
            "gender",
            torch.tensor([0.0]),
            torch.tensor([0.1]),
            loss=0.01,
        )
        # Batch 2: one correct (both feminine), one wrong (masc vs neutral)
        diag.update_mse_family(
            "gender",
            torch.tensor([0.7, -0.8]),
            torch.tensor([0.6, 0.0]),
            loss=0.1,
        )
        stats = diag.mse_families["gender"]
        assert stats.correct_discrete == 2  # 1 from batch 1 + 1 from batch 2
        assert stats.sample_count == 3

    def test_discrete_accuracy_in_report(self) -> None:
        """get_stats() reports discrete_accuracy correctly."""
        diag = KCEpochDiag()
        # 2 correct out of 4
        preds = torch.tensor([0.0, 0.8, -0.9, 0.3])
        targets = torch.tensor([0.1, 0.9, 0.0, 0.9])
        diag.update_mse_family("formality", preds, targets, loss=0.1)
        report = diag.get_stats()
        mse_stats = report.mse_families["formality"]
        # pred buckets:   [neutral(2), very_formal(4), very_casual(0), formal(3)]
        # target buckets: [neutral(2), very_formal(4), neutral(2),     very_formal(4)]
        # matches: idx 0 and 1 → 2 out of 4
        assert abs(mse_stats.discrete_accuracy - 0.5) < 1e-6
