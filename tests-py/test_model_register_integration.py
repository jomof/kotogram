import unittest
from unittest.mock import MagicMock, patch

import torch

from kotogram.analysis import FormalityLevel, RegisterLevel, grammar
from kotogram.model import StylePrediction


class TestModelRegisterIntegration(unittest.TestCase):
    def setUp(self):
        # Create a mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        # Setup tokenizer mock
        self.mock_tokenizer.encode.return_value = {
            "surface": [1, 2, 3],
            "pos": [4, 5, 6],
            "pos_detail1": [7, 8, 9],
            "pos_detail2": [10, 11, 12],
            "pos_detail3": [10, 11, 12],
            "conjugated_type": [13, 14, 15],
            "conjugated_form": [16, 17, 18],
            "lemma": [19, 20, 21],
            "base_orth": [22, 23, 24],
            "reading": [25, 26, 27],
        }

    @patch("kotogram.analysis._load_style_model")
    def test_register_prediction(self, mock_load):
        mock_load.return_value = (self.mock_model, self.mock_tokenizer)

        # Setup model mock prediction for register
        # REGISTER_ID_TO_LABEL mapping (from model.py):
        # 0:NEUTRAL, 1:SONKEIGO, 2:KENJOGO, 3:KANSAIBEN, 4:HAKATABEN, 5:KYOSHIGO, 6:NETSLANG, 7:OJOUSAMA, 8:GUNTAI

        # Mock probabilities: Batch size 1, 14 classes.
        # Use values > 0.5 for active, < 0.5 for inactive
        register_probs = torch.tensor([[0.1] * 14])
        # Let's make index 1 (SONKEIGO) the winner
        register_probs[0, 1] = 0.9

        self.mock_model.predict.return_value = StylePrediction(
            formality_value=torch.zeros(1, 1),
            formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            gender_value=torch.zeros(1, 1),
            gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            grammaticality_probs=torch.tensor([[0.1, 0.9]]),
            register_probs=register_probs,
        )

        # Test 1: Expect {SONKEIGO}
        result = grammar("dummy kotogram")
        self.assertEqual(result.registers, {RegisterLevel.SONKEIGO})
        self.assertEqual(len(result.register_scores), 14)

        # Test 2: Expect {KANSAIBEN} (index 3)
        register_probs[0, 1] = 0.1
        register_probs[0, 3] = 0.9
        self.mock_model.predict.return_value = StylePrediction(
            formality_value=torch.zeros(1, 1),
            formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            gender_value=torch.zeros(1, 1),
            gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            grammaticality_probs=torch.tensor([[0.1, 0.9]]),
            register_probs=register_probs,
        )
        result = grammar("dummy kotogram")
        self.assertEqual(result.registers, {RegisterLevel.KANSAIBEN})
        self.assertEqual(len(result.register_scores), 14)

        # Test 3: Expect {NEUTRAL} (all low)
        register_probs[0, 3] = 0.1
        register_probs[0, 0] = 0.1
        self.mock_model.predict.return_value = StylePrediction(
            formality_value=torch.zeros(1, 1),
            formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            gender_value=torch.zeros(1, 1),
            gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            grammaticality_probs=torch.tensor([[0.1, 0.9]]),
            register_probs=register_probs,
        )
        result = grammar("dummy kotogram")
        self.assertEqual(result.registers, {RegisterLevel.NEUTRAL})
        self.assertEqual(len(result.register_scores), 14)

        # Test 4: Expect {SONKEIGO, KANSAIBEN} (multi-label)
        register_probs[0, 1] = 0.9  # SONKEIGO
        register_probs[0, 3] = 0.9  # KANSAIBEN
        self.mock_model.predict.return_value = StylePrediction(
            formality_value=torch.zeros(1, 1),
            formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            gender_value=torch.zeros(1, 1),
            gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]),
            grammaticality_probs=torch.tensor([[0.1, 0.9]]),
            register_probs=register_probs,
        )
        result = grammar("dummy kotogram")
        self.assertEqual(
            result.registers, {RegisterLevel.SONKEIGO, RegisterLevel.KANSAIBEN}
        )
        self.assertEqual(len(result.register_scores), 14)

    @patch("kotogram.analysis._load_style_model")
    def test_style_function_includes_register(self, mock_load):
        mock_load.return_value = (self.mock_model, self.mock_tokenizer)

        # Mock predictions
        formality_val = torch.tensor([1.0])  # Formal (0.5 actually, 1.0 is very formal)
        # Using 0.5 because style() logic buckets: >=0.25 is Formal
        formality_val = torch.tensor([0.5])

        formality_prag = torch.zeros(1, 2)
        formality_prag[0, 1] = 5.0  # Pragmatic

        gender_logits = torch.zeros(1, 4)
        gender_logits[0, 0] = 5.0  # MASCULINE

        gram_logits = torch.zeros(1, 2)
        gram_logits[0, 1] = 5.0  # Grammatic

        register_probs = torch.zeros(1, 14)
        register_probs[0, 3] = 0.9  # KANSAIBEN

        gender_val = torch.tensor([-1.0])
        gender_prag = torch.zeros(1, 2)
        gender_prag[0, 1] = 5.0  # Pragmatic

        self.mock_model.predict.return_value = StylePrediction(
            formality_value=formality_val,
            formality_pragmatic_probs=formality_prag,
            gender_value=gender_val,
            gender_pragmatic_probs=gender_prag,
            grammaticality_probs=gram_logits,
            register_probs=register_probs,
        )

        res = grammar("dummy kotogram")

        self.assertEqual(res.formality, FormalityLevel.FORMAL)
        self.assertEqual(res.gender_score, -1.0)
        self.assertEqual(res.registers, {RegisterLevel.KANSAIBEN})
        self.assertEqual(len(res.register_scores), 14)
        self.assertTrue(res.is_grammatic)


if __name__ == "__main__":
    unittest.main()
