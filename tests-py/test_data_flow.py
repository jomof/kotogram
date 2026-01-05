import unittest

import torch

from kotogram.model import NUM_REGISTER_CLASSES
from train.dataset import collate_fn
from train.types import Sample


class TestDataFlow(unittest.TestCase):
    def test_collate_fn_types_and_shapes(self):
        # Create dummy samples
        # Note: In the actual script, register_labels is a list of ints.
        s1 = Sample(
            feature_ids={
                "surface": [1, 2, 3],
                "pos": [1, 2, 3],
                "pos_detail_1": [1, 2, 3],
                "pos_detail_2": [1, 2, 3],
                "pos_detail_3": [1, 2, 3],
                "conjugated_type": [1, 2, 3],
                "conjugated_form": [1, 2, 3],
                "lemma": [1, 2, 3],
                "base_orth": [1, 2, 3],
                "reading": [1, 2, 3],
            },
            formality_value=-1.0,  # Very Casual
            formality_pragmatic=1,
            gender_value=0.0,
            gender_pragmatic=0,
            register_labels=[0],  # Neutral
            grammaticality_label=1,
            original_sentence="Test 1",
            kotogram="Test 1",
        )
        s2 = Sample(
            feature_ids={
                "surface": [4, 5],
                "pos": [4, 5],
                "pos_detail_1": [4, 5],
                "pos_detail_2": [4, 5],
                "pos_detail_3": [4, 5],
                "conjugated_type": [4, 5],
                "conjugated_form": [4, 5],
                "lemma": [4, 5],
                "base_orth": [4, 5],
                "reading": [4, 5],
            },
            formality_value=1.0,  # Very Formal
            formality_pragmatic=1,
            gender_value=1.0,
            gender_pragmatic=1,
            register_labels=[1, 2],  # Sonkeigo + Kenjogo (example)
            grammaticality_label=0,  # Agrammatic
            original_sentence="Test 2",
            kotogram="Test 2",
        )

        batch = [s1, s2]
        collated = collate_fn(batch)

        # Check Register Labels (BCEWithLogitsLoss requires Float)
        reg_labels = collated.register_labels
        self.assertTrue(
            torch.is_floating_point(reg_labels),
            f"Register labels must be float, got {reg_labels.dtype}",
        )
        self.assertEqual(reg_labels.dtype, torch.float32)  # Specifically float32
        self.assertEqual(reg_labels.shape, (2, NUM_REGISTER_CLASSES))

        # Check content of register labels
        # s1: [0] -> 1.0 at index 0, others 0
        self.assertEqual(reg_labels[0, 0].item(), 1.0)
        self.assertEqual(reg_labels[0, 1].item(), 0.0)

        # s2: [1, 2] -> 1.0 at index 1 and 2
        self.assertEqual(reg_labels[1, 1].item(), 1.0)
        self.assertEqual(reg_labels[1, 2].item(), 1.0)
        self.assertEqual(reg_labels[1, 0].item(), 0.0)

        # Check Grammaticality Labels (CrossEntropy requires Long)
        gram_labels = collated.grammaticality_labels
        self.assertFalse(
            torch.is_floating_point(gram_labels),
            f"Grammaticality labels must be long, got {gram_labels.dtype}",
        )
        self.assertEqual(gram_labels.dtype, torch.long)
        self.assertEqual(gram_labels.shape, (2,))
        self.assertEqual(gram_labels[0].item(), 1)
        self.assertEqual(gram_labels[1].item(), 0)

        # Check Formality/Gender Labels
        form_val = collated.formality_value
        form_prag = collated.formality_pragmatic
        self.assertEqual(form_val.dtype, torch.float32)
        self.assertEqual(form_prag.dtype, torch.long)

        gender_labels = collated.gender_pragmatic
        self.assertEqual(gender_labels.dtype, torch.long)

    def test_sample_defaults(self):
        # Verify default behavior
        s = Sample(
            feature_ids={
                "surface": [1],
                "pos": [1],
                "pos_detail_1": [1],
                "pos_detail_2": [1],
                "pos_detail_3": [1],
                "conjugated_type": [1],
                "conjugated_form": [1],
                "lemma": [1],
                "base_orth": [1],
                "reading": [1],
            },
            formality_value=0.0,
            formality_pragmatic=1,
            gender_value=0.0,
            gender_pragmatic=0,
            # Missing register_labels, grammaticality_label
        )
        # Should default to Neutral ([0]) and Grammatic (1)
        self.assertEqual(s.register_labels, [0])
        self.assertEqual(s.grammaticality_label, 1)

    def test_collate_fn_variations(self):
        """Test collate_fn with non-default parameters."""
        s1 = Sample(
            feature_ids={"surface": [1, 2]},
            formality_value=0.0,
            formality_pragmatic=0,
            gender_value=0.0,
            gender_pragmatic=0,
            register_labels=[0],
            grammaticality_label=1,
            original_sentence="S1",
            kotogram="S1",
        )
        s2 = Sample(
            feature_ids={"surface": [3, 4, 5]},
            formality_value=0.0,
            formality_pragmatic=0,
            gender_value=0.0,
            gender_pragmatic=0,
            register_labels=[0],
            grammaticality_label=1,
            original_sentence="S2",
            kotogram="S2",
        )
        batch = [s1, s2]

        # Vary max_seq_len (truncation) and implicit pad_id=0
        collated = collate_fn(batch, max_seq_len=2)

        # Check shapes
        self.assertEqual(collated.feature_inputs["input_ids_surface"].shape, (2, 2))
        # Check padding/truncation
        # s1: [1, 2] -> [1, 2] (exact fit)
        # s2: [3, 4, 5] -> [3, 4] (truncated)
        self.assertTrue(
            torch.equal(
                collated.feature_inputs["input_ids_surface"][1],
                torch.tensor([3, 4], dtype=torch.long),
            )
        )

        # Check pad_id usage (needs a short sequence, should be 0)
        s3 = Sample(
            feature_ids={"surface": [6]},
            formality_value=0.0,
            formality_pragmatic=0,
            gender_value=0.0,
            gender_pragmatic=0,
            register_labels=[0],
            grammaticality_label=1,
            original_sentence="S3",
            kotogram="S3",
        )
        batch2 = [s1, s3]
        collated2 = collate_fn(batch2, max_seq_len=3)
        # s1: [1, 2] -> [1, 2, 0]
        # s3: [6] -> [6, 0, 0]
        self.assertEqual(collated2.feature_inputs["input_ids_surface"][0, 2].item(), 0)
        self.assertEqual(collated2.feature_inputs["input_ids_surface"][1, 1].item(), 0)

        # Vary batch size (3 items)
        batch3 = [s1, s2, s3]
        collated3 = collate_fn(batch3, max_seq_len=2)
        self.assertEqual(collated3.feature_inputs["input_ids_surface"].shape[0], 3)

    def test_filter_by_grammaticality_label(self):
        """Vary label in filter_by_grammaticality."""
        from train.dataset import StyleDataset

        # Setup dataset with mocked I/O helper methods
        # Setup dataset with mocked I/O helper methods
        with (
            unittest.mock.patch.object(
                StyleDataset, "_check_exists", return_value=True
            ),
            unittest.mock.patch.object(StyleDataset, "_get_size", return_value=40),
            unittest.mock.patch.object(
                StyleDataset,
                "_load_tensor",
                return_value=torch.zeros(10, dtype=torch.int32),
            ),
        ):
            dataset = StyleDataset(
                "dummy_path", unittest.mock.MagicMock(), indices=torch.arange(10)
            )
            # dataset.labels["gram"] needs to exist
            dataset.labels = {"gram": torch.randint(0, 3, (10,))}
            dataset.features = {}
            dataset.offsets = {}
            dataset.kc_maps = {}
            dataset.data_dir = "dummy_path"

            # Filter for unlikely label 2
            ds_2 = dataset.filter_by_grammaticality(label=2)
            # Check filtered indices logic
            expected = dataset.indices[dataset.labels["gram"][dataset.indices] == 2]
            self.assertTrue(torch.equal(ds_2.indices, expected))


if __name__ == "__main__":
    unittest.main()
