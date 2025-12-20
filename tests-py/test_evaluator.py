
import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader
from kotogram.evaluator import Evaluator, EvalResult
from kotogram.model import Tokenizer, StyleClassifier

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        
        # Mock model and tokenizer
        self.tokenizer = MagicMock(spec=Tokenizer)
        self.tokenizer.pad_id = 0
        
        self.model = MagicMock(spec=StyleClassifier)
        self.model.eval.return_value = None
        self.model.to.return_value = self.model
        
        # Setup mock return values for model()
        # Returns: (formality_logits, gender_val, gender_prag, grammaticality, register_logits)
        batch_size = 2
        self.model.return_value = (
            torch.randn(batch_size, 6),  # formality: 6 classes
            torch.randn(batch_size, 1),  # gender_val: continuous
            torch.randn(batch_size, 4),  # gender_prag: 4 classes (0-3: M, F, N, U) - Wait, prags are specific classes? 
            # Actually gender_pragmatic is trained as classification?
            # Creating dummy outputs
            torch.randn(batch_size, 2),  # grammaticality: 2 classes
            torch.randn(batch_size, 9)   # register: ~9 classes
        )

    def test_initialization(self):
        evaluator = Evaluator(self.model, self.device, verbose=False)
        self.assertEqual(evaluator.model, self.model)
        self.assertEqual(evaluator.device, self.device)
        self.assertFalse(evaluator.verbose)
        self.assertIsNotNone(evaluator.console) # Rich is mandatory now

    def test_evaluate_empty_loader(self):
        evaluator = Evaluator(self.model, self.device, verbose=False)
        loader = DataLoader([], batch_size=1)
        result = evaluator.evaluate(loader)
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(len(result.formality_preds), 0)

    def test_evaluate_batch(self):
        # Create a dummy batch
        batch = {
            'input_ids_surface': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_lemma': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_pos': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_pos_detail1': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_pos_detail2': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_conjugated_type': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_conjugated_form': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_base_orth': torch.tensor([[1, 2], [3, 4]]),
            'input_ids_reading': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]]),
            
            'formality_labels': torch.tensor([0, 1]),
            'gender_value': torch.tensor([0.0, 1.0]),
            'gender_pragmatic': torch.tensor([0, 1]),
            'grammaticality_labels': torch.tensor([1, 1]),
            'register_labels': torch.zeros(2, 9),
            
            'original_sentence': ['Sentence 1', 'Sentence 2'],
            'kotogram': ['K1', 'K2']
        }
        
        # Mock DataLoader
        loader = [batch]
        
        evaluator = Evaluator(self.model, self.device, verbose=True)
        result = evaluator.evaluate(loader)
        
        self.assertEqual(len(result.formality_preds), 2)
        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(result.sentences[0], 'Sentence 1')
        self.assertEqual(result.kotograms[0], 'K1')
        
        # Check model call
        self.model.assert_called()

    def test_keyboard_interrupt(self):
        # Simulate KeyboardInterrupt during iteration
        evaluator = Evaluator(self.model, self.device, verbose=False)
        
        # Mock loader that raises KeyboardInterrupt
        loader = MagicMock()
        loader.__iter__.side_effect = KeyboardInterrupt()
        
        with self.assertRaises(SystemExit):
            evaluator.evaluate(loader)

if __name__ == '__main__':
    unittest.main()
