import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer
from train.dataset import DatasetConfig, StyleDataset, collate_fn
from train.types import ProcessedSample, Sample


class TestDatasetRefactor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.msg_tsv = os.path.join(self.test_dir, "test.tsv")
        with open(self.msg_tsv, "w", encoding="utf-8") as f:
            f.write("A:こんにちは\n")
            f.write("A:サヨナラ\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test basic initialization."""
        tokenizer = MagicMock(spec=Tokenizer)
        dataset = StyleDataset([], tokenizer)
        self.assertEqual(len(dataset), 0)

    @patch("train.dataset.StyleDataset._process_parallel")
    def test_from_tsv_mocked_processing(self, mock_process):
        """Test from_tsv flow with mocked processing."""
        # Setup mock return
        mock_sample = ProcessedSample(
            sentence="A:こんにちはB:元気？",
            kotogram="dummy",
            formality_id=2,
            gender_value=0.0,
            gender_pragmatic=0,
            register_ids=[0],
            gram_label=1,
            success=1,
            feature_ids={"surface": [1, 2, 3]},
        )
        mock_process.return_value = [mock_sample]

        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.field_vocabs = {"surface": [1, 2, 3, 4, 5]}  # Ensure valid vocab

        dataset = StyleDataset.from_tsv(
            self.msg_tsv,
            tokenizer,
            config=DatasetConfig(verbose=False, use_cache=False),
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].original_sentence, "A:こんにちはB:元気？")

    def test_collate_fn_shapes(self):
        """Test collate_fn produces correct shapes."""
        # Dynamically create feature_ids
        f_ids_1 = {k: [1, 2] for k in FEATURE_FIELDS}
        f_ids_2 = {k: [3] for k in FEATURE_FIELDS}

        batch = [
            Sample(
                feature_ids=f_ids_1,
                formality_value=0.0,
                formality_pragmatic=0,
                gender_value=0.0,
                gender_pragmatic=0,
                register_labels=[0],
                grammaticality_label=1,
                original_sentence="s1",
                kotogram="k1",
                kc_targets={"bag_surface": [1], "ngram_pos": [10, 20]},
            ),
            Sample(
                feature_ids=f_ids_2,
                formality_value=1.0,
                formality_pragmatic=1,
                gender_value=1.0,
                gender_pragmatic=1,
                register_labels=[1],
                grammaticality_label=1,
                original_sentence="s2",
                kotogram="k2",
                kc_targets={"bag_surface": [3], "ngram_pos": [30]},
            ),
        ]

        vocab_sizes = {"surface": 100, "pos": 50}
        collated = collate_fn(batch, vocab_sizes=vocab_sizes)

        self.assertIn("attention_mask", collated)
        self.assertEqual(
            collated["input_ids_surface"].shape, (2, 2)
        )  # Max seq len is 2
        self.assertEqual(collated["formality_value"].shape, (2,))
        self.assertIn("kc_targets_bag_surface", collated)
        self.assertIn("kc_targets_ngram_pos", collated)


if __name__ == "__main__":
    unittest.main()
