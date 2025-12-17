
import unittest
from kotogram.augment import augment
from kotogram.analysis import formality, gender, style, grammaticality, register
from kotogram.kotogram import kotogram_to_japanese, split_kotogram, extract_token_features

class TestTypeCheckingRepro(unittest.TestCase):
    def test_augment_bad_types(self):
        """Test augment with incorrect types."""
        print("\n--- Testing augment with bad types ---")
        try:
            # Should fail because it expects a list of strings, not a single string
            augment("not a list") 
            print("FAILURE: augment accepted a string instead of a list of strings")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a list of strings", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")

        try:
            # Should fail because list contains int, not str
            augment(["valid string", 123])
            print("FAILURE: augment accepted a list containing an integer")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a list of strings", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")

    def test_grammaticality_bad_types(self):
        """Test grammaticality with incorrect types."""
        print("\n--- Testing grammaticality with bad types ---")
        try:
            # Should fail because expects str
            grammaticality(123)
            print("FAILURE: grammaticality accepted an integer")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a string", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")
            
    def test_formality_bad_types(self):
        print("\n--- Testing formality with bad types ---")
        try:
            formality(None)
            print("FAILURE: formality accepted None")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a string", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")

    def test_kotogram_utils_bad_types(self):
        print("\n--- Testing kotogram utils with bad types ---")
        try:
            split_kotogram(123)
            print("FAILURE: split_kotogram accepted int")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a string", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")
            
        try:
            extract_token_features(None)
            print("FAILURE: extract_token_features accepted None")
        except TypeError as e:
            print(f"SUCCESS: {e}")
            self.assertIn("must be a string", str(e))
        except Exception as e:
            print(f"FAILURE (wrong error type): {type(e).__name__}: {e}")

if __name__ == "__main__":
    unittest.main()
