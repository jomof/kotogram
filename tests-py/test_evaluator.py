import unittest
from unittest.mock import MagicMock

import torch
from torch.utils.data import DataLoader

from kotogram.model import InferenceClassifier, StylePrediction
from kotogram.tokenizer import Tokenizer
from train.evaluator import EvalResult, Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Mock model and tokenizer
        self.tokenizer = MagicMock(spec=Tokenizer)
        self.tokenizer.pad_id = 0

        self.model = MagicMock(spec=InferenceClassifier)
        self.model.eval.return_value = None
        self.model.to.return_value = self.model

        # Setup mock return values for predict()
        batch_size = 2

        # StylePrediction fields are tensors, not logits inside the prediction object (usually)
        # But predict() returns StylePrediction with *probs* and *values*.
        self.model.predict.return_value = StylePrediction(
            formality_value=torch.randn(batch_size, 1),
            formality_pragmatic_probs=torch.randn(batch_size, 2),
            gender_value=torch.randn(batch_size, 1),
            gender_pragmatic_probs=torch.randn(batch_size, 2),
            grammaticality_probs=torch.randn(batch_size, 2),
            register_probs=torch.randn(batch_size, 9),
        )

    def test_initialization(self):
        evaluator = Evaluator(self.model, self.device, verbose=False)
        self.assertEqual(evaluator.model, self.model)
        self.assertEqual(evaluator.device, self.device)
        self.assertFalse(evaluator.verbose)
        self.assertIsNotNone(evaluator.console)  # Rich is mandatory now

    def test_evaluate_empty_loader(self):
        evaluator = Evaluator(self.model, self.device, verbose=False)
        loader = DataLoader([], batch_size=1)
        result = evaluator.evaluate(loader)
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(len(result.formality_val_preds), 0)

    def test_evaluate_batch(self):
        # Create a dummy batch
        batch = unittest.mock.Mock()
        batch.feature_inputs = {
            # ENCODER_FEATURE_FIELDS: pos, pos_detail_1, pos_detail_2, pos_detail_3,
            # conjugated_form, conjugated_type, reading
            "input_ids_pos": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_pos_detail_1": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_pos_detail_2": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_pos_detail_3": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_conjugated_form": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_conjugated_type": torch.tensor([[1, 2], [3, 4]]),
            "input_ids_reading": torch.tensor([[1, 2], [3, 4]]),
        }
        batch.attention_mask = torch.tensor([[1, 1], [1, 1]])
        batch.formality_value = torch.tensor([0.0, 1.0])
        batch.formality_pragmatic = torch.tensor([0, 1])
        batch.gender_value = torch.tensor([0.0, 1.0])
        batch.gender_pragmatic = torch.tensor([0, 1])
        batch.grammaticality_labels = torch.tensor([1, 1])
        batch.register_labels = torch.zeros(2, 9)
        batch.indices = None
        batch.original_sentence = ["Sentence 1", "Sentence 2"]
        batch.kotogram = [
            "私/代名詞/ワタシ/ワタシ",
            "彼/代名詞/カレ/カレ",
        ]  # Realistic dummy kotograms

        # Mock DataLoader
        loader = [batch]

        evaluator = Evaluator(self.model, self.device, verbose=True)
        result = evaluator.evaluate(loader)

        self.assertEqual(len(result.formality_val_preds), 2)
        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(result.sentences[0], "Sentence 1")
        self.assertEqual(result.kotograms[0], "私/代名詞/ワタシ/ワタシ")

        # Check model call
        self.model.predict.assert_called()

    def test_keyboard_interrupt(self):
        # Simulate KeyboardInterrupt during iteration
        evaluator = Evaluator(self.model, self.device, verbose=False)

        # Mock loader that raises KeyboardInterrupt
        loader = MagicMock()
        loader.__iter__.side_effect = KeyboardInterrupt()

        with self.assertRaises(SystemExit):
            evaluator.evaluate(loader)


if __name__ == "__main__":
    unittest.main()
