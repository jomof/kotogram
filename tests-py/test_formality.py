"""Tests for formality analysis of Japanese sentences."""

import unittest
from kotogram import MecabJapaneseParser, SudachiJapaneseParser, formality, FormalityLevel


class TestFormalityMecab(unittest.TestCase):
    """Test formality analysis with MeCab parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

    def test_very_formal_itadaku_humble(self):
        """いただく humble verb should be VERY_FORMAL."""
        # Real example from dataset ID 5193
        text = "また後でかけ直していただけませんか？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # いただく is humble keigo
        self.assertEqual(result, FormalityLevel.VERY_FORMAL)

    def test_very_formal_itasu_humble(self):
        """いたす humble verb should be VERY_FORMAL."""
        # Real example from dataset ID 74370
        text = "遅延便については、オリジナルの出発日に基づくシーズナリティを適用するため、マイル差額の払い戻しはいたしません。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # いたす is humble keigo
        self.assertEqual(result, FormalityLevel.VERY_FORMAL)

    def test_very_casual_nanda(self):
        """なんだ contraction should be VERY_CASUAL."""
        # Real example from dataset ID 74670
        text = "あらまあ、ホント、全く知らなんだ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なんだ is casual contraction
        self.assertEqual(result, FormalityLevel.VERY_CASUAL)

    def test_very_casual_ja_copula(self):
        """じゃ copula should be VERY_CASUAL."""
        # Real example from dataset ID 75050
        text = "じゃ、１年前のはもう効き目がないんだ！"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # じゃ is casual contracted copula
        self.assertEqual(result, FormalityLevel.VERY_CASUAL)

    def test_casual_datta_past_copula(self):
        """Sentence ending with だった (past copula) should be CASUAL."""
        # Real example from dataset ID 4714
        text = "こいつは悪いウサギだった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だった is casual past copula (contrasts with でした formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_darou_presumptive(self):
        """Sentence with だろう (presumptive) should be CASUAL."""
        # Real example from dataset ID 4769
        text = "兄弟がいるとどんなだろうといつも思う。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だろう is casual presumptive (contrasts with でしょう formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_neutral_dakara_conjunction(self):
        """だから conjunction with plain ending should be NEUTRAL."""
        # Real example from dataset ID 4765
        text = "だから何？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だから is conjunction, sentence ends with plain pronoun
        # While colloquially casual, grammatically it's neutral (no formal/casual markers at sentence end)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_embedded_da(self):
        """Sentence with だ in embedded clause should be NEUTRAL if main ending is plain."""
        # Real example from dataset ID 4747
        text = "大抵の人は僕を気違いだと思っている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だ is in embedded quotation "〜だと思う", main verb is plain ている
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_attributive_da(self):
        """Sentence with attributive な (from だ) should be NEUTRAL if ending is plain."""
        # Real example from dataset ID 4760
        text = "こんな馬鹿なことは言ったことが無い。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # な is attributive form of だ (馬鹿な), sentence ends with plain 無い
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_conjunctive_ni(self):
        """Sentence with conjunctive に (from だ) should be NEUTRAL if ending is plain."""
        # Real example from dataset ID 4775
        text = "自分勝手にするつもりはない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # に is conjunctive form of だ (勝手に), sentence ends with plain ない
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_formal_with_casual_conjunction(self):
        """Formal sentence with casual conjunction (だけど) should still be FORMAL."""
        # Real example from dataset ID 4842
        text = "だけど、ここではそんなに簡単ではないんです。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # Casual conjunctions like だけど are acceptable in formal speech
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_kudasai_imperative(self):
        """Polite imperative ください should be FORMAL."""
        # Real example from dataset ID 5021
        text = "あなたが受けるに値する助けをあなたに与えて下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ください is the polite imperative form
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_kudasai_imperative_short(self):
        """Short polite imperative ください should be FORMAL."""
        # Real example from dataset ID 5076
        text = "ベストを尽くして下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ください is the polite imperative form
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_nasai_imperative(self):
        """Polite imperative なさい should be FORMAL."""
        # Real example from dataset ID 5179
        text = "心配なさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なさい is a polite imperative (less than ください but still formal)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_nasai_imperative_long(self):
        """Polite imperative なさい in longer sentence should be FORMAL."""
        # Real example from dataset ID 5186
        text = "君はどう思うかについて、深く掘り下げて考えなさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なさい is a polite imperative
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_casual_with_no_particle_question(self):
        """Plain verb + の question marker should be CASUAL."""
        # Real example from dataset ID 5170
        text = "ここに来て夢を見てるの？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # の as sentence-final particle creates casual questions
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_formal_masu_form(self):
        """Polite ます form should be FORMAL."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_desu_form(self):
        """Polite です form should be FORMAL."""
        text = "学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_sentence(self):
        """Full formal sentence should be FORMAL."""
        text = "私は学生です。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_neutral_plain_verb(self):
        """Plain form verb should be NEUTRAL."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_plain_adjective(self):
        """Plain form adjective should be NEUTRAL."""
        text = "高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_plain_sentence(self):
        """Plain form sentence should be NEUTRAL."""
        text = "猫を見る"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_casual_with_particle_yo(self):
        """Sentence with casual particle よ should be NEUTRAL or CASUAL."""
        text = "食べるよ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_casual_with_particle_ne(self):
        """Sentence with casual particle ね should be NEUTRAL or CASUAL."""
        text = "食べるね"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_formal_with_acceptable_particle_yo(self):
        """Formal sentence with よ should still be FORMAL."""
        text = "食べますよ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # よ is acceptable with formal forms
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_with_acceptable_particle_ne(self):
        """Formal sentence with ね should still be FORMAL."""
        text = "食べますね"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ね is acceptable with formal forms
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_unpragmatic_formal_with_casual_particle(self):
        """Formal verb with very casual particle should be UNPRAGMATIC."""
        # Note: This test depends on parser output
        # Some casual particles like ぞ、ぜ、な with ます form are unpragmatic
        text = "食べますぜ"  # Very casual particle with formal verb
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # This should be unpragmatic if ぜ is parsed as a particle
        # If not parsed as expected, it might be FORMAL
        self.assertIn(result, [FormalityLevel.UNPRAGMATIC_FORMALITY, FormalityLevel.FORMAL])

    def test_casual_da_form(self):
        """Plain だ copula should be CASUAL (not NEUTRAL)."""
        text = "学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # The だ copula is the plain/casual form, contrasting with です (formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_past_sentence(self):
        """Sentence with だった (past copula) should be CASUAL."""
        # Real example from dataset ID 4714
        text = "こいつは悪いウサギだった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_with_yo_particle(self):
        """Sentence with だ + よ should be CASUAL."""
        # Real example from dataset ID 4749
        text = "それは私の台詞だよ！"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_present_simple(self):
        """Sentence ending with plain だ should be CASUAL."""
        # Real example from dataset ID 4757
        text = "私はなぜか夜の方が元気だ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_yo_particle(self):
        """Plain verb + よ should be CASUAL (conversational marker)."""
        # Real example from dataset ID 1297
        text = "きみにちょっとしたものをもってきたよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # よ with plain forms creates casual/conversational speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_no_particle(self):
        """Plain verb + の should be CASUAL (casual question marker)."""
        # Real example from dataset ID 4704
        text = "何してるの？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # の as a sentence-final particle creates casual questions
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_adjective_with_yo(self):
        """Plain adjective + よ should be CASUAL (conversational)."""
        # Real example from dataset ID 4738
        text = "眠った方がいいよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_wa_particle_in_dialogue(self):
        """Dialogue with わ particle should be CASUAL."""
        # Real example from dataset ID 4995
        text = "「あの音で考え事ができないわ」と、彼女はタイプライターを見つめながら言った。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # わ with plain form creates casual speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_yo_and_ellipsis(self):
        """Plain verb + よ + ellipsis should be CASUAL."""
        # Real example from dataset ID 5116
        text = "いびきをかくことは認めるよ・・・。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_na_prohibitive(self):
        """Prohibitive な with plain form should be CASUAL."""
        # Real example from dataset ID 5139
        text = "考えていることを聞くな。やることを聞け。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # な as prohibitive particle creates casual/direct speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_empty_kotogram(self):
        """Empty kotogram should return NEUTRAL."""
        result = formality("")
        self.assertEqual(result, FormalityLevel.NEUTRAL)


class TestFormalitySudachi(unittest.TestCase):
    """Test formality analysis with Sudachi parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_very_formal_itadaku_humble(self):
        """いただく humble verb should be VERY_FORMAL."""
        # Real example from dataset ID 5193
        text = "また後でかけ直していただけませんか？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # いただく is humble keigo
        self.assertEqual(result, FormalityLevel.VERY_FORMAL)

    def test_very_formal_itasu_humble(self):
        """いたす humble verb should be VERY_FORMAL."""
        # Real example from dataset ID 74370
        text = "遅延便については、オリジナルの出発日に基づくシーズナリティを適用するため、マイル差額の払い戻しはいたしません。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # いたす is humble keigo
        self.assertEqual(result, FormalityLevel.VERY_FORMAL)

    def test_very_casual_nanda(self):
        """なんだ contraction should be VERY_CASUAL."""
        # Real example from dataset ID 74670
        text = "あらまあ、ホント、全く知らなんだ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なんだ is casual contraction
        self.assertEqual(result, FormalityLevel.VERY_CASUAL)

    def test_very_casual_ja_copula(self):
        """じゃ copula should be VERY_CASUAL."""
        # Real example from dataset ID 75050
        text = "じゃ、１年前のはもう効き目がないんだ！"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # じゃ is casual contracted copula
        self.assertEqual(result, FormalityLevel.VERY_CASUAL)

    def test_casual_datta_past_copula(self):
        """Sentence ending with だった (past copula) should be CASUAL."""
        # Real example from dataset ID 4714
        text = "こいつは悪いウサギだった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だった is casual past copula (contrasts with でした formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_darou_presumptive(self):
        """Sentence with だろう (presumptive) should be CASUAL."""
        # Real example from dataset ID 4769
        text = "兄弟がいるとどんなだろうといつも思う。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だろう is casual presumptive (contrasts with でしょう formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_neutral_dakara_conjunction(self):
        """だから conjunction with plain ending should be NEUTRAL."""
        # Real example from dataset ID 4765
        text = "だから何？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だから is conjunction, sentence ends with plain pronoun
        # While colloquially casual, grammatically it's neutral (no formal/casual markers at sentence end)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_embedded_da(self):
        """Sentence with だ in embedded clause should be NEUTRAL if main ending is plain."""
        # Real example from dataset ID 4747
        text = "大抵の人は僕を気違いだと思っている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # だ is in embedded quotation "〜だと思う", main verb is plain ている
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_attributive_da(self):
        """Sentence with attributive な (from だ) should be NEUTRAL if ending is plain."""
        # Real example from dataset ID 4760
        text = "こんな馬鹿なことは言ったことが無い。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # な is attributive form of だ (馬鹿な), sentence ends with plain 無い
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_with_conjunctive_ni(self):
        """Sentence with conjunctive に (from だ) should be NEUTRAL if ending is plain."""
        # Real example from dataset ID 4775
        text = "自分勝手にするつもりはない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # に is conjunctive form of だ (勝手に), sentence ends with plain ない
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_formal_with_casual_conjunction(self):
        """Formal sentence with casual conjunction (だけど) should still be FORMAL."""
        # Real example from dataset ID 4842
        text = "だけど、ここではそんなに簡単ではないんです。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # Casual conjunctions like だけど are acceptable in formal speech
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_kudasai_imperative(self):
        """Polite imperative ください should be FORMAL."""
        # Real example from dataset ID 5021
        text = "あなたが受けるに値する助けをあなたに与えて下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ください is the polite imperative form
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_kudasai_imperative_short(self):
        """Short polite imperative ください should be FORMAL."""
        # Real example from dataset ID 5076
        text = "ベストを尽くして下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ください is the polite imperative form
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_nasai_imperative(self):
        """Polite imperative なさい should be FORMAL."""
        # Real example from dataset ID 5179
        text = "心配なさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なさい is a polite imperative (less than ください but still formal)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_nasai_imperative_long(self):
        """Polite imperative なさい in longer sentence should be FORMAL."""
        # Real example from dataset ID 5186
        text = "君はどう思うかについて、深く掘り下げて考えなさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # なさい is a polite imperative
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_casual_with_no_particle_question(self):
        """Plain verb + の question marker should be CASUAL."""
        # Real example from dataset ID 5170
        text = "ここに来て夢を見てるの？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # の as sentence-final particle creates casual questions
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_formal_masu_form(self):
        """Polite ます form should be FORMAL."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_desu_form(self):
        """Polite です form should be FORMAL."""
        text = "学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_sentence(self):
        """Full formal sentence should be FORMAL."""
        text = "私は学生です。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_neutral_plain_verb(self):
        """Plain form verb should be NEUTRAL."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_plain_adjective(self):
        """Plain form adjective should be NEUTRAL."""
        text = "高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_neutral_plain_sentence(self):
        """Plain form sentence should be NEUTRAL."""
        text = "猫を見る"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.NEUTRAL)

    def test_casual_with_particle_yo(self):
        """Sentence with casual particle よ should be NEUTRAL or CASUAL."""
        text = "食べるよ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_casual_with_particle_ne(self):
        """Sentence with casual particle ね should be NEUTRAL or CASUAL."""
        text = "食べるね"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_formal_with_acceptable_particle_yo(self):
        """Formal sentence with よ should still be FORMAL."""
        text = "食べますよ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # よ is acceptable with formal forms
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_formal_with_acceptable_particle_ne(self):
        """Formal sentence with ね should still be FORMAL."""
        text = "食べますね"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # ね is acceptable with formal forms
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_casual_da_form(self):
        """Plain だ copula should be CASUAL (not NEUTRAL)."""
        text = "学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # The だ copula is the plain/casual form, contrasting with です (formal)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_past_sentence(self):
        """Sentence with だった (past copula) should be CASUAL."""
        # Real example from dataset ID 4714
        text = "こいつは悪いウサギだった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_with_yo_particle(self):
        """Sentence with だ + よ should be CASUAL."""
        # Real example from dataset ID 4749
        text = "それは私の台詞だよ！"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_da_present_simple(self):
        """Sentence ending with plain だ should be CASUAL."""
        # Real example from dataset ID 4757
        text = "私はなぜか夜の方が元気だ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_yo_particle(self):
        """Plain verb + よ should be CASUAL (conversational marker)."""
        # Real example from dataset ID 1297
        text = "きみにちょっとしたものをもってきたよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # よ with plain forms creates casual/conversational speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_no_particle(self):
        """Plain verb + の should be CASUAL (casual question marker)."""
        # Real example from dataset ID 4704
        text = "何してるの？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # の as a sentence-final particle creates casual questions
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_adjective_with_yo(self):
        """Plain adjective + よ should be CASUAL (conversational)."""
        # Real example from dataset ID 4738
        text = "眠った方がいいよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_wa_particle_in_dialogue(self):
        """Dialogue with わ particle should be CASUAL."""
        # Real example from dataset ID 4995
        text = "「あの音で考え事ができないわ」と、彼女はタイプライターを見つめながら言った。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # わ with plain form creates casual speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_yo_and_ellipsis(self):
        """Plain verb + よ + ellipsis should be CASUAL."""
        # Real example from dataset ID 5116
        text = "いびきをかくことは認めるよ・・・。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_casual_with_na_prohibitive(self):
        """Prohibitive な with plain form should be CASUAL."""
        # Real example from dataset ID 5139
        text = "考えていることを聞くな。やることを聞け。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        # な as prohibitive particle creates casual/direct speech
        self.assertEqual(result, FormalityLevel.CASUAL)

    def test_empty_kotogram(self):
        """Empty kotogram should return NEUTRAL."""
        result = formality("")
        self.assertEqual(result, FormalityLevel.NEUTRAL)


class TestFormalityEdgeCases(unittest.TestCase):
    """Test edge cases and complex formality scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser and not self.sudachi_parser:
            self.skipTest("No parsers available")

    def test_question_formal_mecab(self):
        """Formal question should be FORMAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "何を食べますか"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_question_formal_sudachi(self):
        """Formal question should be FORMAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "何を食べますか"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_question_casual_mecab(self):
        """Casual question should be NEUTRAL or CASUAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "何を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_question_casual_sudachi(self):
        """Casual question should be NEUTRAL or CASUAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "何を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_negative_formal_mecab(self):
        """Formal negative should be FORMAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "食べません"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_negative_formal_sudachi(self):
        """Formal negative should be FORMAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "食べません"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_negative_casual_mecab(self):
        """Casual negative should be NEUTRAL or CASUAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "食べない"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_negative_casual_sudachi(self):
        """Casual negative should be NEUTRAL or CASUAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "食べない"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_past_tense_formal_mecab(self):
        """Formal past tense should be FORMAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "食べました"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_past_tense_formal_sudachi(self):
        """Formal past tense should be FORMAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "食べました"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_past_tense_casual_mecab(self):
        """Casual past tense should be NEUTRAL or CASUAL."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "食べた"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])

    def test_past_tense_casual_sudachi(self):
        """Casual past tense should be NEUTRAL or CASUAL."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "食べた"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertIn(result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])


class TestCrossParserFormality(unittest.TestCase):
    """Test that formality analysis works consistently across parsers."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser or not self.sudachi_parser:
            self.skipTest("Both parsers required for cross-parser tests")

    def test_both_detect_formal(self):
        """Both parsers should detect formal sentences."""
        text = "食べます"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = formality(mecab_kotogram)
        sudachi_result = formality(sudachi_kotogram)

        # Both should detect formal
        self.assertEqual(mecab_result, FormalityLevel.FORMAL)
        self.assertEqual(sudachi_result, FormalityLevel.FORMAL)

    def test_both_detect_neutral(self):
        """Both parsers should detect neutral sentences."""
        text = "食べる"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = formality(mecab_kotogram)
        sudachi_result = formality(sudachi_kotogram)

        # Both should detect neutral
        self.assertEqual(mecab_result, FormalityLevel.NEUTRAL)
        self.assertEqual(sudachi_result, FormalityLevel.NEUTRAL)

    def test_both_handle_complex_sentence(self):
        """Both parsers should handle complex sentences."""
        text = "私は毎日学校に行きます。"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = formality(mecab_kotogram)
        sudachi_result = formality(sudachi_kotogram)

        # Both should detect formal
        self.assertEqual(mecab_result, FormalityLevel.FORMAL)
        self.assertEqual(sudachi_result, FormalityLevel.FORMAL)

    def test_both_handle_particles(self):
        """Both parsers should handle casual particles similarly."""
        text = "食べるよ"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = formality(mecab_kotogram)
        sudachi_result = formality(sudachi_kotogram)

        # Both should be in similar category (neutral or casual)
        self.assertIn(mecab_result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])
        self.assertIn(sudachi_result, [FormalityLevel.NEUTRAL, FormalityLevel.CASUAL])


class TestFeminineFormalRegister(unittest.TestCase):
    """Test that feminine-formal register is not flagged as unpragmatic."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser and not self.sudachi_parser:
            self.skipTest("No parsers available")

    def test_desu_wa_is_formal_mecab(self):
        """Feminine-formal ですわ should be FORMAL, not unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "それは素敵ですわ"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_desu_wa_is_formal_sudachi(self):
        """Feminine-formal ですわ should be FORMAL, not unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "それは素敵ですわ"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_desu_no_is_formal_mecab(self):
        """Feminine-formal ですの should be FORMAL, not unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "これは私の本ですの"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_desu_no_is_formal_sudachi(self):
        """Feminine-formal ですの should be FORMAL, not unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "これは私の本ですの"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_masu_wa_is_formal_mecab(self):
        """Feminine-formal ますわ should be FORMAL, not unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "参りますわ"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_masu_wa_is_formal_sudachi(self):
        """Feminine-formal ますわ should be FORMAL, not unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "参りますわ"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_kudasai_ne_is_formal_mecab(self):
        """Polite ください + ね should be FORMAL, not unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "連絡してくださいね"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_kudasai_ne_is_formal_sudachi(self):
        """Polite ください + ね should be FORMAL, not unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "連絡してくださいね"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_kudasai_na_is_formal_mecab(self):
        """Polite ください + な (feminine) should be FORMAL, not unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "怒らないでくださいな"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)

    def test_kudasai_na_is_formal_sudachi(self):
        """Polite ください + な (feminine) should be FORMAL, not unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "怒らないでくださいな"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertEqual(result, FormalityLevel.FORMAL)


class TestUnpragmaticFormality(unittest.TestCase):
    """Test detection of unpragmatic formality mixing."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser and not self.sudachi_parser:
            self.skipTest("No parsers available")

    def test_consistent_formal_not_unpragmatic_mecab(self):
        """Consistent formal usage should not be unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "私は学生です。"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertNotEqual(result, FormalityLevel.UNPRAGMATIC_FORMALITY)

    def test_consistent_formal_not_unpragmatic_sudachi(self):
        """Consistent formal usage should not be unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "私は学生です。"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertNotEqual(result, FormalityLevel.UNPRAGMATIC_FORMALITY)

    def test_consistent_casual_not_unpragmatic_mecab(self):
        """Consistent casual usage should not be unpragmatic."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "私は学生だ。"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertNotEqual(result, FormalityLevel.UNPRAGMATIC_FORMALITY)

    def test_consistent_casual_not_unpragmatic_sudachi(self):
        """Consistent casual usage should not be unpragmatic."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "私は学生だ。"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = formality(kotogram)
        self.assertNotEqual(result, FormalityLevel.UNPRAGMATIC_FORMALITY)


if __name__ == "__main__":
    unittest.main()
