"""Unit tests for scripts.dataset — the professional dataset pipeline."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from kotogram.tokenizer import (
    CLS_ID,
    CLS_TOKEN,
    FEATURE_FIELDS,
    MASK_ID,
    MASK_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)
from scripts.dataset import (
    SCHEMA_VERSION,
    BundledStyleDataset,
    compute_chive_hash,
    compute_dataset_id,
    load_dataset,
    merge_content_mask,
    merge_vocabs,
    read_lock,
    remap_feature_tensor,
    write_lock,
)
from scripts.dataset_token_histogram import grammatical_token_length_counts
from train.dataset import collate_fn


def _make_minimal_bundle(
    n_sentences: int = 10,
    extra_surface_tokens: int = 20,
) -> dict:
    """Create a synthetic dataset bundle for testing."""
    surface_vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        CLS_TOKEN: CLS_ID,
        MASK_TOKEN: MASK_ID,
    }
    for i in range(extra_surface_tokens):
        surface_vocab[f"tok_{i}"] = len(surface_vocab)

    vocab = {}
    for field in FEATURE_FIELDS:
        if field == "surface":
            vocab[field] = dict(surface_vocab)
        else:
            vocab[field] = {
                PAD_TOKEN: PAD_ID,
                UNK_TOKEN: UNK_ID,
                CLS_TOKEN: CLS_ID,
                MASK_TOKEN: MASK_ID,
                "val_a": 4,
                "val_b": 5,
            }

    tokens_per_sentence = 5
    total_tokens = n_sentences * tokens_per_sentence
    offsets = torch.tensor(
        [i * tokens_per_sentence for i in range(n_sentences + 1)], dtype=torch.int32
    )

    features = {
        "surface": torch.randint(
            4, len(surface_vocab), (total_tokens,), dtype=torch.int32
        ),
    }

    gram = torch.ones(n_sentences, dtype=torch.uint8)
    gram[:3] = 0
    labels = {"gram": gram}

    content_mask = torch.zeros(len(surface_vocab), dtype=torch.bool)
    content_mask[4:10] = True

    sentences = [f"sentence {i}" for i in range(n_sentences)]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "test0000",
        "created_at": "2026-01-01T00:00:00Z",
        "git_commit": "abc1234",
        "base_dataset_id": None,
        "chive_id": "chive0000",
        "sentence_count": n_sentences,
        "token_count": total_tokens,
        "vocab": vocab,
        "offsets": offsets,
        "features": features,
        "labels": labels,
        "content_mask": content_mask,
        "sentences": sentences,
    }
    bundle["token_length_counts"] = torch.from_numpy(
        grammatical_token_length_counts(bundle)
    )
    return bundle


class TestVocabMerge(unittest.TestCase):
    """Test vocabulary inheritance (append-only merge)."""

    def test_no_base_uses_local_ids(self):
        local_vocab = {
            "surface": {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
                "cat": 4,
                "dog": 5,
            },
        }
        merged, remap = merge_vocabs(local_vocab, None)
        self.assertEqual(merged["surface"]["cat"], 4)
        self.assertEqual(merged["surface"]["dog"], 5)
        self.assertEqual(remap["surface"][4], 4)
        self.assertEqual(remap["surface"][5], 5)

    def test_base_ids_preserved(self):
        base_vocab = {
            "surface": {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
                "cat": 4,
                "dog": 5,
            },
        }
        local_vocab = {
            "surface": {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
                "cat": 4,
                "bird": 5,
            },
        }
        merged, remap = merge_vocabs(local_vocab, base_vocab)

        self.assertEqual(merged["surface"]["cat"], 4, "Existing token keeps base ID")
        self.assertEqual(merged["surface"]["dog"], 5, "Base-only token kept")
        self.assertEqual(
            merged["surface"]["bird"], 6, "New token appended after base max"
        )

        self.assertEqual(remap["surface"][4], 4, "local cat -> base cat")
        self.assertEqual(remap["surface"][5], 6, "local bird (was 5) -> merged 6")

    def test_append_only_semantics(self):
        base_vocab = {
            "surface": {PAD_TOKEN: 0, UNK_TOKEN: 1, CLS_TOKEN: 2, MASK_TOKEN: 3, "a": 4}
        }
        local_vocab = {
            "surface": {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
                MASK_TOKEN: 3,
                "a": 4,
                "b": 5,
                "c": 6,
            }
        }
        merged, _ = merge_vocabs(local_vocab, base_vocab)

        self.assertEqual(merged["surface"]["a"], 4)
        self.assertIn("b", merged["surface"])
        self.assertIn("c", merged["surface"])
        self.assertGreater(merged["surface"]["b"], 4)
        self.assertGreater(merged["surface"]["c"], 4)


class TestRemapFeatureTensor(unittest.TestCase):
    def test_identity_remap(self):
        tensor = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        remap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        result = remap_feature_tensor(tensor, remap)
        self.assertTrue(torch.equal(result, tensor))

    def test_non_trivial_remap(self):
        tensor = torch.tensor([4, 5, 4, 5], dtype=torch.int32)
        remap = {4: 10, 5: 20}
        result = remap_feature_tensor(tensor, remap)
        expected = torch.tensor([10, 20, 10, 20], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_empty_remap_returns_original(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = remap_feature_tensor(tensor, {})
        self.assertTrue(torch.equal(result, tensor))


class TestContentMaskMerge(unittest.TestCase):
    def test_no_base_uses_local(self):
        local_mask = torch.tensor([False, True, True, False, True], dtype=torch.bool)
        remap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        merged = merge_content_mask(local_mask, remap, None, 5)
        self.assertTrue(torch.equal(merged, local_mask))

    def test_base_carried_forward(self):
        local_mask = torch.tensor([False, True], dtype=torch.bool)
        base_mask = torch.tensor([False, False, False, True, True], dtype=torch.bool)
        remap = {0: 0, 1: 1}
        merged = merge_content_mask(local_mask, remap, base_mask, 5)

        self.assertTrue(merged[1], "Local overwrites base for overlapping tokens")
        self.assertTrue(merged[3], "Base-only token carried forward")
        self.assertTrue(merged[4], "Base-only token carried forward")

    def test_new_tokens_get_local_classification(self):
        local_mask = torch.tensor([False, False, False, False, True], dtype=torch.bool)
        base_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
        remap = {4: 5}
        merged = merge_content_mask(local_mask, remap, base_mask, 6)

        self.assertTrue(
            merged[5], "New token (local 4 -> merged 5) gets local classification"
        )
        self.assertFalse(merged[4], "Base token at 4 unchanged")

    def test_local_overwrites_base(self):
        local_mask = torch.tensor([False, True, False], dtype=torch.bool)
        base_mask = torch.tensor([False, False, True], dtype=torch.bool)
        remap = {1: 1, 2: 2}
        merged = merge_content_mask(local_mask, remap, base_mask, 3)

        self.assertTrue(merged[1], "Local True overwrites base False")
        self.assertFalse(merged[2], "Local False overwrites base True")


class TestDatasetSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_round_trip(self):
        bundle = _make_minimal_bundle()
        path = os.path.join(self.tmpdir, "test.pt")
        torch.save(bundle, path)
        loaded = load_dataset(path)

        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded["dataset_id"], bundle["dataset_id"])
        self.assertEqual(loaded["sentence_count"], bundle["sentence_count"])
        self.assertTrue(torch.equal(loaded["offsets"], bundle["offsets"]))
        self.assertTrue(
            torch.equal(loaded["features"]["surface"], bundle["features"]["surface"])
        )

    def test_wrong_schema_version_raises(self):
        bundle = _make_minimal_bundle()
        bundle["schema_version"] = 999
        path = os.path.join(self.tmpdir, "bad.pt")
        torch.save(bundle, path)
        with self.assertRaises(ValueError):
            load_dataset(path)

    def test_missing_keys_raises(self):
        bundle = _make_minimal_bundle()
        del bundle["vocab"]
        path = os.path.join(self.tmpdir, "missing.pt")
        torch.save(bundle, path)
        with self.assertRaises(ValueError):
            load_dataset(path)

    def test_missing_token_length_counts_raises(self):
        bundle = _make_minimal_bundle()
        del bundle["token_length_counts"]
        path = os.path.join(self.tmpdir, "missing_hist.pt")
        torch.save(bundle, path)
        with self.assertRaises(ValueError):
            load_dataset(path)


class TestDatasetIdDeterminism(unittest.TestCase):
    def test_same_data_same_id(self):
        b1 = _make_minimal_bundle()
        b2 = _make_minimal_bundle()
        b2["features"]["surface"] = b1["features"]["surface"].clone()
        b2["labels"] = {k: v.clone() for k, v in b1["labels"].items()}
        b2["offsets"] = b1["offsets"].clone()
        b2["content_mask"] = b1["content_mask"].clone()
        b2["sentences"] = list(b1["sentences"])
        b2["vocab"] = json.loads(json.dumps(b1["vocab"]))

        id1 = compute_dataset_id(b1)
        id2 = compute_dataset_id(b2)
        self.assertEqual(id1, id2)

    def test_different_data_different_id(self):
        b1 = _make_minimal_bundle()
        b2 = _make_minimal_bundle()
        b2["sentences"][0] = "CHANGED"

        id1 = compute_dataset_id(b1)
        id2 = compute_dataset_id(b2)
        self.assertNotEqual(id1, id2)

    def test_schema_version_affects_id(self):
        b1 = _make_minimal_bundle()
        b2 = _make_minimal_bundle()
        b2["features"]["surface"] = b1["features"]["surface"].clone()
        b2["labels"] = {k: v.clone() for k, v in b1["labels"].items()}
        b2["offsets"] = b1["offsets"].clone()
        b2["content_mask"] = b1["content_mask"].clone()
        b2["sentences"] = list(b1["sentences"])
        b2["vocab"] = json.loads(json.dumps(b1["vocab"]))
        b2["token_length_counts"] = b1["token_length_counts"].clone()
        b2["schema_version"] = SCHEMA_VERSION + 1
        self.assertNotEqual(compute_dataset_id(b1), compute_dataset_id(b2))


class TestChiveHash(unittest.TestCase):
    def test_deterministic(self):
        t = torch.randn(10, 300)
        h1 = compute_chive_hash(t)
        h2 = compute_chive_hash(t)
        self.assertEqual(h1, h2)

    def test_different_vectors_different_hash(self):
        t1 = torch.randn(10, 300)
        t2 = torch.randn(10, 300)
        self.assertNotEqual(compute_chive_hash(t1), compute_chive_hash(t2))


class TestBundledStyleDataset(unittest.TestCase):
    def test_from_bundle_length(self):
        bundle = _make_minimal_bundle(n_sentences=20)
        ds = BundledStyleDataset.from_bundle(bundle, verbose=False)
        self.assertEqual(len(ds), 20)

    def test_filter_by_grammaticality(self):
        bundle = _make_minimal_bundle(n_sentences=20)
        bundle["labels"]["gram"][:5] = 0
        bundle["labels"]["gram"][5:] = 1

        ds = BundledStyleDataset.from_bundle(bundle, verbose=False)
        gram_ds = ds.filter_by_grammaticality(label=1)

        self.assertEqual(len(gram_ds), 15)
        self.assertIsInstance(gram_ds, BundledStyleDataset)

    def test_getitem_returns_sample(self):
        bundle = _make_minimal_bundle(n_sentences=5)
        ds = BundledStyleDataset.from_bundle(bundle, verbose=False)
        sample = ds[0]
        self.assertIn("surface", sample.feature_ids)
        self.assertIsInstance(sample.grammaticality_label, int)

    def test_get_sentence_by_idx(self):
        bundle = _make_minimal_bundle(n_sentences=5)
        ds = BundledStyleDataset.from_bundle(bundle, verbose=False)
        self.assertEqual(ds.get_sentence_by_idx(0), "sentence 0")
        self.assertEqual(ds.get_sentence_by_idx(4), "sentence 4")
        self.assertEqual(ds.get_sentence_by_idx(999), "")

    def test_collate_fn_compatibility(self):
        bundle = _make_minimal_bundle(n_sentences=10)
        ds = BundledStyleDataset.from_bundle(bundle, verbose=False)
        samples = [ds[i] for i in range(min(4, len(ds)))]
        batch = collate_fn(samples)
        self.assertIn("input_ids_surface", batch.feature_inputs)
        self.assertEqual(batch.attention_mask.shape[0], len(samples))


class TestDatasetLock(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("scripts.dataset._find_repo_root")
    def test_write_read_round_trip(self, mock_root):
        mock_root.return_value = self.tmpdir
        write_lock("abc123", "chive456")
        lock = read_lock()
        self.assertIsNotNone(lock)
        self.assertEqual(lock["dataset_id"], "abc123")
        self.assertEqual(lock["chive_id"], "chive456")
        self.assertIn("created_at", lock)

    @patch("scripts.dataset._find_repo_root")
    def test_read_missing_returns_none(self, mock_root):
        mock_root.return_value = self.tmpdir
        self.assertIsNone(read_lock())


class TestGCSOperationsMocked(unittest.TestCase):
    """Verify GCS-dependent functions handle mocked responses correctly."""

    @patch("scripts.dataset._gcs_read_json")
    @patch("scripts.dataset._ensure_dataset_local")
    @patch("scripts.dataset._ensure_chive_local")
    @patch("scripts.dataset.load_dataset")
    @patch("scripts.dataset.load_chive")
    def test_resolve_latest(  # pylint: disable=too-many-positional-arguments
        self,
        mock_load_chive,
        mock_load_ds,
        mock_chive_local,
        mock_ds_local,
        mock_gcs_json,
    ):
        mock_gcs_json.return_value = {"dataset_id": "ds123", "chive_id": "ch456"}
        mock_ds_local.return_value = "/tmp/ds-ds123.pt"
        mock_chive_local.return_value = "/tmp/chive-ch456.pt"
        bundle = _make_minimal_bundle()
        mock_load_ds.return_value = bundle
        mock_load_chive.return_value = torch.randn(10, 300)

        from scripts.dataset import resolve_dataset

        result_bundle, chive_tensor = resolve_dataset("latest")

        mock_gcs_json.assert_called_once()
        mock_ds_local.assert_called_with("ds123")
        mock_chive_local.assert_called_with("ch456")
        self.assertEqual(result_bundle["dataset_id"], "test0000")
        self.assertEqual(chive_tensor.shape, (10, 300))

    @patch("scripts.dataset.read_lock")
    def test_resolve_no_lock_raises(self, mock_lock):
        mock_lock.return_value = None
        from scripts.dataset import resolve_dataset

        with self.assertRaises(FileNotFoundError):
            resolve_dataset(None)


if __name__ == "__main__":
    unittest.main()
