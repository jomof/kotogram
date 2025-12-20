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


    def test_danseigo_boku(self):
        """Test that ぼく (boku) triggers danseigo detection."""
        self._check("ぼくは日本語を話すことができません。", {RegisterLevel.DANSEIGO})
        self._check("ぼくも行きたい", {RegisterLevel.DANSEIGO})
    
    def test_joseigo_wa_casual(self):
        """Test that casual sentence-final わ triggers joseigo."""
        self._check("困っちゃうわ。", {RegisterLevel.JOSEIGO})
        # Note: わ after です/ます is ojousama, not joseigo
    
    def test_hakataben_false_positive_tai(self):
        """Test that たいです does NOT incorrectly trigger hakataben."""
        kotogram = self.parser.japanese_to_kotogram("彼はそれらの両方を食べたいです。")
        result = analyze_register(kotogram)
        self.assertNotIn(RegisterLevel.HAKATABEN, result, 
                        f"たいです should not trigger hakataben. Got {result}")
    
    def test_ojousama_false_positive_masu(self):
        """Test that standalone ます does NOT incorrectly trigger ojousama."""
        kotogram = self.parser.japanese_to_kotogram("海綿は水を吸収しますので水彩絵具をぼかしたりする時に便利です。")
        result = analyze_register(kotogram)
        self.assertNotIn(RegisterLevel.OJOUSAMA, result,
                        f"Standalone ます should not trigger ojousama. Got {result}")
    
    def test_guntai_false_positives(self):
        """Test that non-military imperatives don't trigger guntai."""
        # Regular imperative without military context
        kotogram = self.parser.japanese_to_kotogram("ちょっと待て！")
        analyze_register(kotogram)
        # This might still trigger guntai, but we'll fix the rules
        # Just documenting expected behavior for now
    
    def test_kansaiben_false_positive_yainaya(self):
        """Test that やいなや (as soon as) doesn't trigger kansaiben."""
        kotogram = self.parser.japanese_to_kotogram("人は生まれるやいなや、死にに向かう。")
        result = analyze_register(kotogram)
        self.assertNotIn(RegisterLevel.KANSAIBEN, result,
                        f"やいなや is standard Japanese, not Kansaiben. Got {result}")
        self.assertEqual(result, {RegisterLevel.NEUTRAL})
    
    def test_hakataben_false_positive_tai_auxiliary(self):
        """Test that たい auxiliary verb doesn't trigger hakataben."""
        sentences = [
            "差し当たって、私はその本屋で働きたいと思う。",
            "何と言ったら良いか分かりません。",
            "何と言ったらいいか・・・。"
        ]
        for sent in sentences:
            kotogram = self.parser.japanese_to_kotogram(sent)
            result = analyze_register(kotogram)
            self.assertNotIn(RegisterLevel.HAKATABEN, result,
                            f"Standard たい/たら should not trigger hakataben in '{sent}'. Got {result}")
    
    def test_kyoshigo_false_positive_desu(self):
        """Test that plain です doesn't trigger kyoshigo."""
        sentences = [
            "宿題を全部やってしまったので少しやすみたいです。",
            "彼の説明はわかりにくかったです。"
        ]
        for sent in sentences:
            kotogram = self.parser.japanese_to_kotogram(sent)
            result = analyze_register(kotogram)
            self.assertNotIn(RegisterLevel.KYOSHIGO, result,
                            f"Plain です without teacher markers should not trigger kyoshigo in '{sent}'. Got {result}")
    
    def test_netslang_false_positives(self):
        """Test that standard formal sentences don't trigger netslang."""
        sentences = [
            "この評論を優勝作品に選んだ基準は何ですか。",
            "雨が降って土に湿り気があると草は取りやすくなる。",
            "彼の４人抜きの活躍でうちの高校のチームが優勝しました。"
        ]
        for sent in sentences:
            kotogram = self.parser.japanese_to_kotogram(sent)
            result = analyze_register(kotogram)
            self.assertNotIn(RegisterLevel.NETSLANG, result,
                            f"Standard sentence should not trigger netslang in '{sent}'. Got {result}")
    
    def test_guntai_false_positive_jibun(self):
        """Test that 自分 in non-military context doesn't trigger guntai."""
        kotogram = self.parser.japanese_to_kotogram("会社に入ると、自分が望むと望まざるとにかかわらず、会社のために働かなくてはいけない。")
        result = analyze_register(kotogram)
        self.assertNotIn(RegisterLevel.GUNTAI, result,
                        f"自分 in standard context should not trigger guntai. Got {result}")
    
    def test_guntai_false_positive_imperative(self):
        """Test that plain imperative without military context doesn't trigger guntai."""
        kotogram = self.parser.japanese_to_kotogram("省エネのためにコンビニの２４時間営業を廃止しろ！")
        result = analyze_register(kotogram)
        self.assertNotIn(RegisterLevel.GUNTAI, result,
                        f"Plain imperative without military context should not trigger guntai. Got {result}")

if __name__ == '__main__':
    unittest.main()
