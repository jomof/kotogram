import unittest
from unittest.mock import MagicMock, patch
import torch
from kotogram.analysis import register, style, RegisterLevel, FormalityLevel, GenderLevel

class TestModelRegisterIntegration(unittest.TestCase):
    def setUp(self):
        # Create a mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock
        self.mock_tokenizer.encode.return_value = {
            'surface': [1, 2, 3],
            'pos': [4, 5, 6],
            'pos_detail1': [7, 8, 9],
            'pos_detail2': [10, 11, 12],
            'conjugated_type': [13, 14, 15],
            'conjugated_form': [16, 17, 18],
            'lemma': [19, 20, 21],
        }

    @patch('kotogram.analysis._load_style_model')
    def test_register_prediction(self, mock_load):
        mock_load.return_value = (self.mock_model, self.mock_tokenizer)
        
        # Setup model mock prediction for register
        # register: 0=sonkeigo (from model.py or just testing logic)
        # We need to match REGISTER_ID_TO_LABEL in model.py
        # Assuming mappings: 0:SONKEIGO, 1:KENJOGO, 2:KANSAIBEN, 3:HAKATABEN, 4:KYOSHIGO, 5:NETSLANG, 6:NEUTRAL
        
        # Mock probabilities: Batch size 1, 7 classes.
        # Use values > 0.5 for active, < 0.5 for inactive
        register_probs = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
        # Let's make index 0 (SONKEIGO) the winner
        register_probs[0, 0] = 0.9
        
        # model.predict returns 4 tensors (probabilities)
        self.mock_model.predict.return_value = (
            torch.zeros(1, 6), # formality
            torch.zeros(1, 4), # gender
            torch.zeros(1, 2), # grammaticality
            register_probs     # register
        )
        
        # Test 1: Expect {SONKEIGO}
        result = register("dummy kotogram")
        self.assertEqual(result, {RegisterLevel.SONKEIGO})
        
        # Test 2: Expect {KANSAIBEN} (index 2)
        register_probs[0, 0] = 0.1
        register_probs[0, 2] = 0.9
        self.mock_model.predict.return_value = (
            torch.zeros(1, 6), 
            torch.zeros(1, 4), 
            torch.zeros(1, 2), 
            register_probs
        )
        result = register("dummy kotogram")
        self.assertEqual(result, {RegisterLevel.KANSAIBEN})
        
        # Test 3: Expect {NEUTRAL} (all low)
        # Using index 6 (NEUTRAL) explicit logic in analysis.py: 
        # "if not detected_registers: detected_registers.add(RegisterLevel.NEUTRAL)"
        # So even if all are 0.1, it should return {NEUTRAL}
        register_probs[0, 2] = 0.1
        register_probs[0, 6] = 0.1 # even if neutral logit is low, it falls back
        self.mock_model.predict.return_value = (
            torch.zeros(1, 6), 
            torch.zeros(1, 4), 
            torch.zeros(1, 2), 
            register_probs
        )
        result = register("dummy kotogram")
        self.assertEqual(result, {RegisterLevel.NEUTRAL})

        # Test 4: Expect {SONKEIGO, KANSAIBEN} (multi-label)
        register_probs[0, 0] = 0.9 # SONKEIGO
        register_probs[0, 2] = 0.9 # KANSAIBEN
        self.mock_model.predict.return_value = (
            torch.zeros(1, 6), 
            torch.zeros(1, 4), 
            torch.zeros(1, 2), 
            register_probs
        )
        result = register("dummy kotogram")
        self.assertEqual(result, {RegisterLevel.SONKEIGO, RegisterLevel.KANSAIBEN})

    @patch('kotogram.analysis._load_style_model')
    def test_style_function_includes_register(self, mock_load):
        mock_load.return_value = (self.mock_model, self.mock_tokenizer)
        
        # Mock predictions
        formality_logits = torch.zeros(1, 6)
        formality_logits[0, 1] = 5.0 # FORMAL
        
        gender_logits = torch.zeros(1, 4)
        gender_logits[0, 0] = 5.0 # MASCULINE
        
        gram_logits = torch.zeros(1, 2)
        gram_logits[0, 1] = 5.0 # Grammatic
        
        register_probs = torch.zeros(1, 7)
        register_probs[0, 2] = 0.9 # KANSAIBEN
        
        self.mock_model.predict.return_value = (
            formality_logits,
            gender_logits,
            gram_logits,
            register_probs
        )
        
        f, g, r, is_gram = style("dummy kotogram")
        
        f, g, r, is_gram = style("dummy kotogram")
        
        self.assertEqual(f, FormalityLevel.FORMAL)
        self.assertEqual(g, GenderLevel.MASCULINE)
        self.assertEqual(r, {RegisterLevel.KANSAIBEN})
        self.assertTrue(is_gram)

if __name__ == '__main__':
    unittest.main()
