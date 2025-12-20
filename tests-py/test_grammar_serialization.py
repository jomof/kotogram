"""Tests for GrammarAnalysis JSON serialization."""

import unittest
import json
from kotogram.analysis import GrammarAnalysis
from kotogram.constants import FormalityLevel, GenderLevel, RegisterLevel

class TestGrammarSerialization(unittest.TestCase):
    """Test to_json and from_json methods of GrammarAnalysis."""

    def test_serialization_roundtrip(self):
        """Verify that GrammarAnalysis can be serialized and deserialized correctly."""
        analysis = GrammarAnalysis(
            kotogram="dummy_kotogram",
            formality=FormalityLevel.FORMAL,
            formality_score=0.9,
            formality_is_pragmatic=True,
            gender=GenderLevel.NEUTRAL,
            gender_score=0.1,
            gender_is_pragmatic=True,
            registers={RegisterLevel.KANSAIBEN, RegisterLevel.SONKEIGO},
            register_scores={
                RegisterLevel.KANSAIBEN: 0.8,
                RegisterLevel.SONKEIGO: 0.7,
                RegisterLevel.NEUTRAL: 0.2
            },
            is_grammatic=True,
            grammaticality_score=0.95
        )

        # To JSON
        json_str = analysis.to_json()
        
        # Verify JSON content
        data = json.loads(json_str)
        self.assertEqual(data['formality'], "formal")
        self.assertEqual(data['gender'], "neutral")
        self.assertEqual(sorted(data['registers']), ["kansaiben", "sonkeigo"])
        self.assertEqual(data['register_scores']['kansaiben'], 0.8)
        self.assertEqual(data['register_scores']['neutral'], 0.2)

        # From JSON
        restored = GrammarAnalysis.from_json(json_str)
        
        # Verify restored object
        self.assertEqual(restored.kotogram, analysis.kotogram)
        self.assertEqual(restored.formality, analysis.formality)
        self.assertEqual(restored.gender, analysis.gender)
        self.assertEqual(restored.registers, analysis.registers)
        self.assertEqual(restored.register_scores, analysis.register_scores)
        self.assertEqual(restored.is_grammatic, analysis.is_grammatic)
        self.assertAlmostEqual(restored.grammaticality_score, analysis.grammaticality_score)

if __name__ == "__main__":
    unittest.main()
