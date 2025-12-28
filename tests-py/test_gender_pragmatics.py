import os
import sys
import unittest

# Add project root to path
sys.path.append(os.getcwd())

# pylint: disable=wrong-import-position
from kotogram.analysis import GenderLevel, RegisterLevel
from scripts.label import infer_gender_from_register


class TestGenderPragmatics(unittest.TestCase):
    def test_infer_gender_basics(self):
        # Explict Masculine
        val, prag = infer_gender_from_register(GenderLevel.MASCULINE, [])
        self.assertEqual(val, -1.0)
        self.assertEqual(prag, 1)

        # Explicit Feminine
        val, prag = infer_gender_from_register(GenderLevel.FEMININE, [])
        self.assertEqual(val, 1.0)
        self.assertEqual(prag, 1)

    def test_infer_gender_from_registers(self):
        # Neutral + Masc Register
        val, prag = infer_gender_from_register(
            GenderLevel.NEUTRAL, [RegisterLevel.DANSEIGO]
        )
        self.assertEqual(val, -1.0)
        self.assertEqual(prag, 1)

        # Neutral + Fem Register
        val, prag = infer_gender_from_register(
            GenderLevel.NEUTRAL, [RegisterLevel.JOSEIGO]
        )
        self.assertEqual(val, 1.0)
        self.assertEqual(prag, 1)

        # Neutral + Mixed (Conflicting) -> Unpragmatic
        val, prag = infer_gender_from_register(
            GenderLevel.NEUTRAL, [RegisterLevel.DANSEIGO, RegisterLevel.JOSEIGO]
        )
        self.assertEqual(prag, 0)

    def test_neutral_is_pragmatic(self):
        """
        Verify that a standard neutral sentence (no specific register)
        is considered 'Pragmatic' (valid training data for Neutral class).
        """
        # This is the bug case: currently returns (0.0, 0)
        val, prag = infer_gender_from_register(
            GenderLevel.NEUTRAL, [RegisterLevel.NEUTRAL]
        )
        self.assertEqual(val, 0.0)
        self.assertEqual(
            prag, 1, "Neutral sentences should be pragmatic (valid training samples)"
        )

        # Empty registers fallback
        val, prag = infer_gender_from_register(GenderLevel.NEUTRAL, [])
        self.assertEqual(val, 0.0)
        self.assertEqual(
            prag, 1, "Neutral sentences without registers should be pragmatic"
        )


if __name__ == "__main__":
    unittest.main()
