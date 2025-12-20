
import unittest
from kotogram.preprocess import preprocess
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.parser = SudachiJapaneseParser()

    def test_strip_quotes(self):
        self.assertEqual(preprocess("「こんにちは」"), ("こんにちは", ["strip_surrounding_quotes"]))
        self.assertEqual(preprocess("「あいうえお」"), ("あいうえお", ["strip_surrounding_quotes"]))

    def test_replace_names(self):
        # 太郎 is a common given name recognized by Sudachi
        res, types = preprocess("太郎", parser=self.parser)
        self.assertEqual(res, "ひろし")
        self.assertIn("replace_names", types)
        
        # Combined
        res, types = preprocess("「太郎」", parser=self.parser)
        self.assertEqual(res, "ひろし")
        self.assertIn("strip_surrounding_quotes", types)
        self.assertIn("replace_names", types)

    def test_no_replace_surname(self):
        # 田中 is a common surname
        res, types = preprocess("田中", parser=self.parser)
        self.assertEqual(res, "田中")
        self.assertNotIn("replace_names", types)

    def test_exception_on_parsing_failure(self):
        # We can mock a parser that returns a wrong POS for ひろし
        class MockParser:
            def japanese_to_kotogram(self, text):
                if text == "太郎":
                    return "⌈ˢ太郎ᵖnoun:proper-noun:person-name:given-name⌉"
                if text == "ひろし":
                    return "⌈ˢひろしᵖnoun:proper-noun:general⌉"
                return ""
        
        with self.assertRaises(Exception) as cm:
            preprocess("太郎", parser=MockParser())
        self.assertIn("ひろし' parsed as 'general:' instead of 'person-name:given-name'", str(cm.exception))

    def test_empty(self):
        self.assertEqual(preprocess(""), ("", []))

if __name__ == '__main__':
    unittest.main()
