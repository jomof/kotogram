import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from rule_based_analysis import analyze_register
from kotogram.analysis import RegisterLevel
# Use the actual parser
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

class TestRegisterLabeler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize parser once (dictionary loading might be slow)
        cls.parser = SudachiJapaneseParser()

    def _check(self, sentence, expected_registers):
        # Parse sentence to kotogram string
        kotogram = self.parser.japanese_to_kotogram(sentence)
        # Analyze
        result = analyze_register(kotogram)
        
        # Check if expected registers are a subset of result
        # (Since we might detect multiple, e.g. SONKEIGO + KYOSHIGO)
        # For strict matching in tests, we can check equality if we are sure.
        # But analyze_register returns a Set.
        # Let's enforce that ALL expected registers are present.
        # And if Expected contains NEUTRAL, checks exactly NEUTRAL.
        
        if RegisterLevel.NEUTRAL in expected_registers and len(expected_registers) == 1:
            self.assertEqual(result, {RegisterLevel.NEUTRAL}, 
                             f"Expected Neutral for '{sentence}', got {result}\nKotogram: {kotogram}")
        else:
            # Check presence
            self.assertTrue(expected_registers.issubset(result), 
                            f"Expected {expected_registers} in {result} for '{sentence}'\nKotogram: {kotogram}")

    def test_netslang(self):
        self._check("それなwww", {RegisterLevel.NETSLANG})
        self._check("草生える", {RegisterLevel.NETSLANG})
        self._check("ワンチャンある", {RegisterLevel.NETSLANG})

    def test_kansaiben(self):
        self._check("なんでやねん", {RegisterLevel.KANSAIBEN})
        self._check("行かへん", {RegisterLevel.KANSAIBEN})
        self._check("知らんけど", {RegisterLevel.KANSAIBEN})

    def test_hakataben(self):
        self._check("行くばい", {RegisterLevel.HAKATABEN})
        self._check("好いとー", {RegisterLevel.HAKATABEN})
        self._check("そげんこと", {RegisterLevel.HAKATABEN})

    def test_kyoshigo(self):
        self._check("しなさい", {RegisterLevel.KYOSHIGO})
        self._check("廊下を走ってはいけません", {RegisterLevel.KYOSHIGO})
        self._check("よくできました", {RegisterLevel.KYOSHIGO})
        # Check context specific
        self._check("先生の話を聞いてください", {RegisterLevel.KYOSHIGO})

    def test_sonkeigo(self):
        self._check("先生がいらっしゃる", {RegisterLevel.SONKEIGO})
        self._check("どうぞ召し上がってください", {RegisterLevel.SONKEIGO})
        self._check("お忙しいところ", {RegisterLevel.SONKEIGO})
        self._check("こちらにお掛けください", {RegisterLevel.SONKEIGO})
        
    def test_ojousama(self):
        self._check("ごきげんよう", {RegisterLevel.OJOUSAMA})
        self._check("素敵ですわ", {RegisterLevel.OJOUSAMA})
        self._check("存じておりますの", {RegisterLevel.OJOUSAMA}) # "zonjite" -> Kenjogo too? Test specific first.
        # "my rule" triggers ojousama for 'masu no'.

    def test_guntai(self):
        self._check("了解であります", {RegisterLevel.GUNTAI})
        self._check("全員集合", {RegisterLevel.GUNTAI})
        self._check("異常なし", {RegisterLevel.GUNTAI})
        self._check("自分はそう思います", {RegisterLevel.GUNTAI})

    def test_kenjogo(self):
        self._check("私が申す", {RegisterLevel.KENJOGO})
        self._check("拝見いたしました", {RegisterLevel.KENJOGO})
        self._check("お電話差し上げます", {RegisterLevel.KENJOGO})
        self._check("よろしくお願いいたします", {RegisterLevel.KENJOGO})

    def test_neutral(self):
        self._check("これはペンです", {RegisterLevel.NEUTRAL})
        self._check("明日は晴れるでしょう", {RegisterLevel.NEUTRAL})

    def test_multi_label(self):
        # "静かにしなさい" -> Kyoshigo (nasai) + Sonkeigo (nasaru)
        # Verify both are detected
        self._check("静かにしなさい", {RegisterLevel.KYOSHIGO, RegisterLevel.SONKEIGO})

    def test_dataset_coverage(self):
        """Verify that all sentences in data/jpn_sentences_register.tsv are correctly labeled."""
        import csv
        
        # Path relative to this test file
        tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/jpn_sentences_register.tsv'))
        self.assertTrue(os.path.exists(tsv_path), f"TSV file not found at {tsv_path}")
        
        label_map = {
            'sonkeigo': RegisterLevel.SONKEIGO,
            'kenjogo': RegisterLevel.KENJOGO,
            'kansaiben': RegisterLevel.KANSAIBEN,
            'hakataben': RegisterLevel.HAKATABEN,
            'kyoshigo': RegisterLevel.KYOSHIGO,
            'netslang': RegisterLevel.NETSLANG,
            'ojousama': RegisterLevel.OJOUSAMA,
            'guntai': RegisterLevel.GUNTAI,
            'neutral': RegisterLevel.NEUTRAL,
        }

        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3: continue
                
                row_id = row[0]
                sentence = row[2]
                
                # Extract expected label from ID (e.g. "sonkeigo_001")
                # Handle cases like "kyoshigo_10"
                expected_str = row_id.split('_')[0]
                expected_enum = label_map.get(expected_str)
                
                if not expected_enum:
                    print(f"Skipping unknown register ID prefix: {row_id}") # Optional logging
                    continue
                
                # Use _check helper logic but customized for this loop for better error messages
                kotogram = self.parser.japanese_to_kotogram(sentence)
                result = analyze_register(kotogram)
                
                if expected_enum == RegisterLevel.NEUTRAL:
                    # Expect ONLY neutral
                    self.assertEqual(result, {RegisterLevel.NEUTRAL}, 
                                     f"Failed ID: {row_id}. Expected {{NEUTRAL}}, got {result} for '{sentence}'")
                else:
                    # Expect specific register in set
                    self.assertIn(expected_enum, result, 
                                  f"Failed ID: {row_id}. Expected {expected_enum} in {result} for '{sentence}'")

if __name__ == '__main__':
    unittest.main()
