import torch
import unittest
from scripts.train_style import collate_fn, Sample, NUM_REGISTER_CLASSES

class TestDataFlow(unittest.TestCase):
    def test_collate_fn_types_and_shapes(self):
        # Create dummy samples
        # Note: In the actual script, register_labels is a list of ints.
        s1 = Sample(
            feature_ids={
                'surface': [1, 2, 3],
                'pos': [1, 2, 3],
                'pos_detail1': [1, 2, 3],
                'pos_detail2': [1, 2, 3],
                'conjugated_type': [1, 2, 3],
                'conjugated_form': [1, 2, 3],
                'lemma': [1, 2, 3]
            },
            formality_label=0,
            gender_value=0.0,
            gender_pragmatic=0,
            register_labels=[0], # Neutral
            grammaticality_label=1,
            original_sentence="Test 1",
            kotogram="Test 1"
        )
        s2 = Sample(
            feature_ids={
                'surface': [4, 5],
                'pos': [4, 5],
                'pos_detail1': [4, 5],
                'pos_detail2': [4, 5],
                'conjugated_type': [4, 5],
                'conjugated_form': [4, 5],
                'lemma': [4, 5]
            },
            formality_label=1,
            gender_value=1.0,
            gender_pragmatic=1,
            register_labels=[1, 2], # Sonkeigo + Kenjogo (example)
            grammaticality_label=0, # Agrammatic
            original_sentence="Test 2",
            kotogram="Test 2"
        )
        
        batch = [s1, s2]
        collated = collate_fn(batch, pad_id=0)
        
        # Check Register Labels (BCEWithLogitsLoss requires Float)
        reg_labels = collated['register_labels']
        self.assertTrue(torch.is_floating_point(reg_labels), f"Register labels must be float, got {reg_labels.dtype}")
        self.assertEqual(reg_labels.dtype, torch.float32) # Specifically float32
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
        gram_labels = collated['grammaticality_labels']
        self.assertFalse(torch.is_floating_point(gram_labels), f"Grammaticality labels must be long, got {gram_labels.dtype}")
        self.assertEqual(gram_labels.dtype, torch.long)
        self.assertEqual(gram_labels.shape, (2,))
        self.assertEqual(gram_labels[0].item(), 1)
        self.assertEqual(gram_labels[1].item(), 0)
        
        # Check Formality/Gender Labels
        form_labels = collated['formality_labels']
        self.assertEqual(form_labels.dtype, torch.long)
        
        gender_labels = collated['gender_pragmatic']
        self.assertEqual(gender_labels.dtype, torch.long)

    def test_sample_defaults(self):
        # Verify default behavior
        s = Sample(
            feature_ids={
                'surface': [1],
                'pos': [1],
                'pos_detail1': [1],
                'pos_detail2': [1],
                'conjugated_type': [1],
                'conjugated_form': [1],
                'lemma': [1]
            },
            formality_label=0,
            gender_value=0.0,
            gender_pragmatic=0
            # Missing register_labels, grammaticality_label
        )
        # Should default to Neutral ([0]) and Grammatic (1)
        self.assertEqual(s.register_labels, [0]) 
        self.assertEqual(s.grammaticality_label, 1)

if __name__ == '__main__':
    unittest.main()
