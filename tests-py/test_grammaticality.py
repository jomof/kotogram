"""Tests for grammaticality analysis of Japanese sentences."""

import unittest
from kotogram import SudachiJapaneseParser, grammaticality, style


class TestGrammaticalityRuleBased(unittest.TestCase):
    """Test rule-based grammaticality analysis with Sudachi parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    # --- Ungrammatical patterns: adjectival predicate + だ ---

    def test_ungrammatical_suffix_rashii_da(self):
        """学生らしいだ - adjectival suffix らしい + だ should be ungrammatical."""
        text = "学生らしいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_auxv_tai_da(self):
        """行きたいだ - auxiliary たい + だ should be ungrammatical."""
        text = "行きたいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_suffix_yasui_da(self):
        """読みやすいだ - adjectival suffix やすい + だ should be ungrammatical."""
        text = "読みやすいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_suffix_nikui_da(self):
        """読みにくいだ - adjectival suffix にくい + だ should be ungrammatical."""
        text = "読みにくいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_auxv_nai_da(self):
        """食べないだ - auxiliary ない + だ should be ungrammatical."""
        text = "食べないだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_auxv_rashii_da(self):
        """行くらしいだ - auxiliary らしい (after verb) + だ should be ungrammatical."""
        text = "行くらしいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_tabetai_da(self):
        """食べたいだ - auxiliary たい + だ should be ungrammatical."""
        text = "食べたいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Grammatical patterns ---

    def test_grammatical_noun_da(self):
        """学生だ - noun + だ should be grammatical."""
        text = "学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_beki_da(self):
        """行くべきだ - べき + だ should be grammatical."""
        text = "行くべきだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_you_da(self):
        """雨が降るようだ - よう + だ should be grammatical."""
        text = "雨が降るようだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_masu_form(self):
        """食べます - verb + ます should be grammatical."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_rashiku_adverbial(self):
        """学生らしく振る舞う - らしく (adverbial) should be grammatical."""
        text = "学生らしく振る舞う"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_rashii_desu(self):
        """学生らしいです - らしい + です should be grammatical."""
        text = "学生らしいです"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_tai_alone(self):
        """食べたい - たい alone without だ should be grammatical."""
        text = "食べたい"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_tai_desu(self):
        """食べたいです - たい + です should be grammatical."""
        text = "食べたいです"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_nai_alone(self):
        """食べない - ない alone without だ should be grammatical."""
        text = "食べない"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_nai_desu(self):
        """食べないです - ない + です should be grammatical."""
        text = "食べないです"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_plain_verb(self):
        """食べる - plain verb should be grammatical."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_adjective_alone(self):
        """美しい - i-adjective alone should be grammatical."""
        text = "美しい"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_sou_da(self):
        """雨が降りそうだ - そう + だ should be grammatical."""
        text = "雨が降りそうだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)


class TestGrammaticalityEdgeCases(unittest.TestCase):
    """Test edge cases for grammaticality analysis."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_empty_kotogram(self):
        """Empty kotogram should return True (grammatical)."""
        result = grammaticality("", use_model=False)
        self.assertTrue(result)

    def test_single_token(self):
        """Single token should be grammatical (no pair to check)."""
        text = "猫"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_complex_sentence_grammatical(self):
        """Complex grammatical sentence should pass."""
        text = "私は毎日日本語を勉強しています"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_ungrammatical_mid_sentence(self):
        """Ungrammatical pattern in middle of sentence should be detected."""
        # This creates らしいだ pattern in the middle
        text = "彼は学生らしいだと言った"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)


class TestStyleGrammaticality(unittest.TestCase):
    """Test that style() function correctly uses rule-based grammaticality."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_style_ungrammatical(self):
        """style() should return is_grammatic=False for ungrammatical input."""
        text = "行きたいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        formality_level, gender_level, is_grammatic = style(kotogram, use_model=False)
        self.assertFalse(is_grammatic)

    def test_style_grammatical(self):
        """style() should return is_grammatic=True for grammatical input."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        formality_level, gender_level, is_grammatic = style(kotogram, use_model=False)
        self.assertTrue(is_grammatic)

    def test_style_grammatical_complex(self):
        """style() should return is_grammatic=True for complex grammatical sentence."""
        text = "私は日本語を勉強したいです"
        kotogram = self.parser.japanese_to_kotogram(text)
        formality_level, gender_level, is_grammatic = style(kotogram, use_model=False)
        self.assertTrue(is_grammatic)


class TestGrammaticalityFromAgrammaticFile(unittest.TestCase):
    """Test grammaticality against sentences from data/agrammatic_sentences.tsv."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    # --- Double past tense (たた) ---

    def test_ungrammatical_double_past_155191(self):
        """私は大勢の人が餓死して行くのをテレビで見たた。- double past た should be ungrammatical."""
        text = "私は大勢の人が餓死して行くのをテレビで見たた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_past_168420(self):
        """子供達は人形で楽しく遊んでいたた。- double past た should be ungrammatical."""
        text = "子供達は人形で楽しく遊んでいたた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_past_simple(self):
        """食べたた - simple double past should be ungrammatical."""
        text = "食べたた"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Doubled particles ---

    def test_ungrammatical_double_ga_153516(self):
        """私は彼女がが一人で行った方が良いといいました。- double が should be ungrammatical."""
        text = "私は彼女がが一人で行った方が良いといいました。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_wo_166186(self):
        """私たちはみな収穫の手伝いををした。- double を should be ungrammatical."""
        text = "私たちはみな収穫の手伝いををした。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_ga_simple(self):
        """飴ががほしい。- double が should be ungrammatical."""
        text = "飴ががほしい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_ni(self):
        """本学にに入学したい - double に should be ungrammatical."""
        text = "本学にに入学したい"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_de(self):
        """東京でで会った - double で should be ungrammatical."""
        # Note: The original file sentence "彼女は働きすぎでで疲れている" is parsed
        # with the first で as copula (auxv) and second as conjunction (conj),
        # so we use a simpler example that Sudachi parses as two particles.
        text = "東京でで会った"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_to(self):
        """彼とと話した - double と should be ungrammatical."""
        text = "彼とと話した"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- i-adjective + だ ---

    def test_ungrammatical_i_adj_da_164578(self):
        """私には心配がないだ。- i-adjective ない + だ should be ungrammatical."""
        text = "私には心配がないだ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_i_adj_da_simple(self):
        """美しいだ - i-adjective + だ should be ungrammatical."""
        text = "美しいだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- i-adjective with な (should use い form) ---

    def test_ungrammatical_i_adj_na_8898667(self):
        """そんなオイシイな話がある - i-adjective with な should be ungrammatical."""
        text = "そんなオイシイな話がある"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_i_adj_na_118167(self):
        """彼の意見はその問題に新しいな見方を加える - i-adjective 新しい with な should be ungrammatical."""
        text = "彼の意見はその問題に新しいな見方を加える"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- だです redundancy ---

    def test_ungrammatical_da_desu_165821(self):
        """私たちは幸福だです。- だ + です redundancy should be ungrammatical."""
        text = "私たちは幸福だです。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_da_desu_simple(self):
        """学生だです - だ + です redundancy should be ungrammatical."""
        text = "学生だです"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- te/de wrong voicing ---

    def test_ungrammatical_te_de_voicing_83250(self):
        """弁論大会で優勝されでおめでとうございます。- されで should be されて."""
        text = "弁論大会で優勝されでおめでとうございます。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_te_de_voicing_simple(self):
        """食べでいる - should be 食べている."""
        text = "食べでいる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_te_de_voicing_godan_geminate(self):
        """黙っでいる - godan verb っ form + で should be て."""
        text = "黙っでいる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_te_de_voicing_godan_i_sound(self):
        """続いで - godan-ka verb い音便 + で should be て."""
        text = "続いで勉強する"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_te_de_voicing_sa_irregular(self):
        """対しで - sa-irregular verb + で should be て."""
        text = "それに対しで"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Verb terminal + を ---

    def test_ungrammatical_verb_terminal_wo_5132379(self):
        """そういうをこともありますよ。- いうを is ungrammatical."""
        text = "そういうをこともありますよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Double で (various parse combinations) ---

    def test_ungrammatical_double_de_auxv_conj(self):
        """彼女は働きすぎでで疲れている - double で (auxv + conj) should be ungrammatical."""
        text = "彼女は働きすぎでで疲れている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_double_de_prt_conj(self):
        """序文でで著者は - double で (prt + conj) should be ungrammatical."""
        text = "序文でで著者は述べている"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Verb conjunctive + を ---

    def test_ungrammatical_verb_conjunctive_wo(self):
        """食べを慣れない - verb conjunctive + を should be ungrammatical."""
        text = "食べを慣れない"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_verb_conjunctive_wo_ai(self):
        """お会いをした - verb conjunctive + を should be ungrammatical."""
        text = "お会いをした"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- na-adjective missing な ---

    def test_ungrammatical_na_adj_missing_na(self):
        """きれい花 - na-adjective + noun without な should be ungrammatical.

        Note: We use きれい (hiragana) instead of 新鮮 (kanji) because all-kanji
        na-adjective + all-kanji noun combinations are allowed as compound nouns
        (e.g., 新鮮空気 is accepted as a sino-Japanese compound).
        """
        text = "きれい花を見て"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Grammatical counterparts (to make sure we don't over-detect) ---

    def test_grammatical_te_form_correct(self):
        """食べている - correct て form should be grammatical."""
        text = "食べている"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_sarete_correct(self):
        """優勝されておめでとう - correct されて form should be grammatical."""
        text = "優勝されておめでとう"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_verb_attributive_wo(self):
        """そういうことをする - verb in attributive + こと + を is grammatical."""
        text = "そういうことをする"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_single_ta(self):
        """食べた - single past tense should be grammatical."""
        text = "食べた"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_single_ga(self):
        """猫が好きだ - single が should be grammatical."""
        text = "猫が好きだ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_na_adjective_na(self):
        """静かな部屋 - na-adjective with な should be grammatical."""
        text = "静かな部屋"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_i_adjective_attributive(self):
        """新しい本 - i-adjective in attributive form (no な) should be grammatical."""
        text = "新しい本"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_godan_geminate_te(self):
        """黙っている - godan verb っ form + て is grammatical."""
        text = "黙っている"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_godan_i_sound_te(self):
        """続いて勉強する - godan-ka verb い音便 + て is grammatical."""
        text = "続いて勉強する"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_godan_de_correct(self):
        """読んでいる - godan-ma verb ん + で is grammatical."""
        text = "読んでいる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_godan_ga_de_correct(self):
        """泳いでいる - godan-ga verb い + で is grammatical."""
        text = "泳いでいる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_single_de(self):
        """東京で会った - single で is grammatical."""
        text = "東京で会った"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_verb_nominalized_wo(self):
        """食べることをする - verb nominalized with こと + を is grammatical."""
        text = "食べることをやめる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_na_adj_with_na(self):
        """新鮮な空気 - na-adjective + な + noun is grammatical."""
        text = "新鮮な空気"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)


class TestGrammaticalityHelperFunctions(unittest.TestCase):
    """Test internal helper functions for grammaticality checking."""

    def test_is_adjectival_predicate_terminal_suffix(self):
        """Test _is_adjectival_predicate_terminal for suffix pattern."""
        from kotogram.analysis import _is_adjectival_predicate_terminal

        # Adjectival suffix in terminal form
        feature = {
            'pos': 'suff',
            'pos_detail1': 'adjectival',
            'conjugated_form': 'terminal',
            'conjugated_type': 'adjective',
        }
        self.assertTrue(_is_adjectival_predicate_terminal(feature))

    def test_is_adjectival_predicate_terminal_auxv_tai(self):
        """Test _is_adjectival_predicate_terminal for auxv-tai pattern."""
        from kotogram.analysis import _is_adjectival_predicate_terminal

        # Auxiliary たい in terminal form
        feature = {
            'pos': 'auxv',
            'pos_detail1': '',
            'conjugated_form': 'terminal',
            'conjugated_type': 'auxv-tai',
        }
        self.assertTrue(_is_adjectival_predicate_terminal(feature))

    def test_is_adjectival_predicate_terminal_auxv_nai(self):
        """Test _is_adjectival_predicate_terminal for auxv-nai pattern."""
        from kotogram.analysis import _is_adjectival_predicate_terminal

        # Auxiliary ない in terminal form
        feature = {
            'pos': 'auxv',
            'pos_detail1': '',
            'conjugated_form': 'terminal',
            'conjugated_type': 'auxv-nai',
        }
        self.assertTrue(_is_adjectival_predicate_terminal(feature))

    def test_is_adjectival_predicate_terminal_not_terminal(self):
        """Test _is_adjectival_predicate_terminal returns False for non-terminal."""
        from kotogram.analysis import _is_adjectival_predicate_terminal

        # Adjectival suffix but NOT in terminal form
        feature = {
            'pos': 'suff',
            'pos_detail1': 'adjectival',
            'conjugated_form': 'conjunctive',  # Not terminal
            'conjugated_type': 'adjective',
        }
        self.assertFalse(_is_adjectival_predicate_terminal(feature))

    def test_is_adjectival_predicate_terminal_non_adjectival_auxv(self):
        """Test _is_adjectival_predicate_terminal returns False for non-adjectival auxv."""
        from kotogram.analysis import _is_adjectival_predicate_terminal

        # Auxiliary べき is NOT adjectival
        feature = {
            'pos': 'auxv',
            'pos_detail1': '',
            'conjugated_form': 'terminal',
            'conjugated_type': 'auxv-beki',  # Not in adjectival set
        }
        self.assertFalse(_is_adjectival_predicate_terminal(feature))

    def test_is_da_copula_true(self):
        """Test _is_da_copula returns True for だ."""
        from kotogram.analysis import _is_da_copula

        feature = {
            'pos': 'auxv',
            'conjugated_type': 'auxv-da',
            'conjugated_form': 'terminal',
        }
        self.assertTrue(_is_da_copula(feature))

    def test_is_da_copula_false_desu(self):
        """Test _is_da_copula returns False for です."""
        from kotogram.analysis import _is_da_copula

        feature = {
            'pos': 'auxv',
            'conjugated_type': 'auxv-desu',  # Not だ
            'conjugated_form': 'terminal',
        }
        self.assertFalse(_is_da_copula(feature))

    def test_is_da_copula_false_noun(self):
        """Test _is_da_copula returns False for non-auxv."""
        from kotogram.analysis import _is_da_copula

        feature = {
            'pos': 'n',  # Not auxv
            'conjugated_type': '',
            'conjugated_form': '',
        }
        self.assertFalse(_is_da_copula(feature))


class TestGrammaticalFalsePositiveFixes(unittest.TestCase):
    """Test that previously false positive cases are now correctly handled as grammatical."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    # --- Nominalized verbs that should not trigger verb_conjunctive + を ---

    def test_grammatical_hikitsuke_wo(self):
        """ひきつけを起こす - ひきつけ is a nominalized verb (seizure)."""
        text = "熱が出るとひきつけを起こすことがあります。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_oide_wo(self):
        """おいでを願う - おいで is an honorific nominalized verb (coming)."""
        text = "明日おいでを願えますか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_uketori_wo(self):
        """受取りを諦める - 受取り is a nominalized verb (receipt)."""
        text = "未払い給料の受取りを諦めました。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- すぎで pattern (excess + cause) ---

    def test_grammatical_sugide_cause(self):
        """すぎでそれ以上 - すぎで indicating cause is valid."""
        text = "彼は疲れすぎでそれ以上歩けなかった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_sugi_dewanai(self):
        """すぎではない - is not too much pattern."""
        text = "友を選ぶときにはどれだけ注意してもしすぎではない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- ででも pattern (at anywhere) ---

    def test_grammatical_dokodedemo(self):
        """どこででも - at anywhere pattern is valid."""
        text = "子供のころどこででも眠ることができた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_gengodedemo(self):
        """言語ででも - in any language pattern is valid."""
        text = "どんな言語ででも書き込めます。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- へへ onomatopoeia ---

    def test_grammatical_hehe_onomatopoeia(self):
        """へへ - laughter onomatopoeia is not doubled particle."""
        text = "おとうさんはお前の実の父親じゃないんだからさ、へへ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Classical Japanese patterns ---

    def test_grammatical_classical_kitaru_wo(self):
        """來たるを見て - classical Japanese V-terminal + を is valid."""
        text = "その從ひ來たるを見て言ひ給ふ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Literary patterns ---

    def test_grammatical_iru_wo_literary(self):
        """ているを見て - literary nominalization pattern."""
        text = "かかっているを見て"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_yamu_wo_ezu(self):
        """やむを得ず - idiomatic expression."""
        text = "やむを得ず帰った"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Sino-Japanese compounds (na-adj + noun without な) ---

    def test_grammatical_kanijutaku(self):
        """簡易住宅 - sino-Japanese compound is valid."""
        text = "簡易住宅に住んでいる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_seiteki_henken(self):
        """性的偏見 - 的-ending compound is valid."""
        text = "彼女は科学における性的偏見について書いた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_taihen_adverb(self):
        """大変文才 - 大変 used as adverb is valid."""
        text = "彼女は大変文才のある女性だ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Nominalized verbs in te-de pattern ---

    def test_grammatical_oide_de(self):
        """お出でで - honorific coming + copula is valid."""
        text = "あなたとあなたのお友達のお出でをお待ちしています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_kodomoke_de(self):
        """子供むけでない - aimed at + copula pattern."""
        text = "その小説は子供むけでない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_gakekuzure_de(self):
        """がけくずれで - landslide (cause) pattern."""
        text = "がけくずれで交通は遮断された。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Parser misparse handled: こいだ ---

    def test_grammatical_koida(self):
        """こいだ - past tense of 漕ぐ (to pedal/row), not adj + だ."""
        text = "彼は、自転車を一生懸命こいだ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- ぶっつづけで pattern ---

    def test_grammatical_buttsudukede(self):
        """ぶっつづけで - continuously pattern."""
        text = "彼は５時間以上もぶっつづけで働かされた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)


class TestAgrammaticTruePositives(unittest.TestCase):
    """Test that agrammatic sentences from agrammatic_sentences.tsv are correctly detected."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception:
            self.skipTest("Sudachi not available")

    # --- Godan nasal + て (should be で) ---

    def test_ungrammatical_nasal_te_shinu(self):
        """死んて - godan-na nasal + て should be detected (should be 死んで)."""
        text = "けちん坊と太った豚は死んて初めて役に立つ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_nasal_te_sumu(self):
        """住んて - godan-ma nasal + て should be detected (should be 住んで)."""
        text = "大阪に住んています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_nasal_te_tanoshimu(self):
        """楽しんて - godan-ma nasal + て should be detected (should be 楽しんで)."""
        text = "スイスの美しい風景を楽しんているだろうね。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Incomplete ている (てい。) ---

    def test_ungrammatical_incomplete_te_iru_shukketsu(self):
        """してい。 - incomplete ている should be detected."""
        text = "歯茎から出血をしてい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_incomplete_te_iru_oshieru(self):
        """教えてい。 - incomplete ている should be detected."""
        text = "私は教えてい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_incomplete_te_iru_shiru(self):
        """知ってい。 - incomplete ている should be detected."""
        text = "彼女の住所を知ってい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Grammatical counterparts (correct forms) ---

    def test_grammatical_nasal_de_correct(self):
        """死んで - correct nasal + で should be grammatical."""
        text = "けちん坊と太った豚は死んで初めて役に立つ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_te_iru_complete(self):
        """している - complete ている should be grammatical."""
        text = "歯茎から出血をしている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- ついで (should be ついて) ---

    def test_ungrammatical_tsuide_ni_tsuite(self):
        """についで - should be について (wrong voicing)."""
        text = "彼は私の研究についで忠告してくれた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_tsuide_houhou(self):
        """についで - should be について (wrong voicing)."""
        text = "報告者は自分の研究方法についで詳しく述べた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- なければたらない (should be なければならない) ---

    def test_ungrammatical_nakereba_taranai(self):
        """なければたらない - should be なければならない."""
        text = "私達は今あるもので我慢しなければたらない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_nakereba_taranai_tegami(self):
        """なければたらない - should be なければならない."""
        text = "忘れずに手紙を出さなければたらない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- できるたら (should be できたら) ---

    def test_ungrammatical_dekiru_tara(self):
        """できるたら - should be できたら (wrong conditional form)."""
        text = "できるたらお手伝いします。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    # --- Grammatical counterparts ---

    def test_grammatical_ni_tsuite_correct(self):
        """について - correct form should be grammatical."""
        text = "彼は私の研究について忠告してくれた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_nakereba_naranai_correct(self):
        """なければならない - correct form should be grammatical."""
        text = "私達は今あるもので我慢しなければならない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_dekitara_correct(self):
        """できたら - correct conditional form should be grammatical."""
        text = "できたらお手伝いします。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Godan verb conjunctive + で (should be て) ---

    def test_ungrammatical_hanashide_iru(self):
        """話しでいます - should be 話していま す (godan sa-row + で + いる)."""
        text = "あの少年は英語を話しでいます。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_nakushide_shimau(self):
        """なくしでしまった - should be なくしてしまった."""
        text = "彼は父親からもらった時計をなくしでしまった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_doshide_yoi(self):
        """どうしでよい - should be どうしてよい."""
        text = "どうしでよいかわからなかったので、私は彼に助力を求めた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_kuchigotae_shide(self):
        """口答えしではいけません - should be 口答えしてはいけません."""
        text = "お母さんに口答えしではいけませんよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_hanashite_iru_correct(self):
        """話しています - correct te-form should be grammatical."""
        text = "あの少年は英語を話しています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- いる (terminal) + ます (should be い + ます) ---

    def test_ungrammatical_iru_terminal_masu(self):
        """ているます - いる in terminal form + ます is wrong (should be ています)."""
        text = "トムは私が困っているますときは、いつでも力になってくれる。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_te_imasu_correct(self):
        """ています - correct form should be grammatical."""
        text = "トムは私が困っていますときは、いつでも力になってくれる。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Noun + て + いる (missing っ in te-form) ---

    def test_ungrammatical_inotte_missing_geminate(self):
        """祈ています - missing っ (should be 祈っています)."""
        text = "イングランドでのお仕事がうまくいったことを祈ています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_shitte_missing_geminate(self):
        """知ている - missing っ (should be 知っている)."""
        text = "シンガポールの医師は殆どの場合皆お互いを知ている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_yashinatte_missing_geminate(self):
        """養ていけない - missing っ (should be 養っていけない)."""
        text = "彼は給料が安すぎて家族を養ていけない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_kaette_missing_geminate(self):
        """帰ていました - missing っ (should be 帰っていました)."""
        text = "彼女はずっと前に家に帰ていました。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_inotte_correct(self):
        """祈っています - correct te-form should be grammatical."""
        text = "イングランドでのお仕事がうまくいったことを祈っています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- た + たこ (doubled た misparsed) ---

    def test_ungrammatical_ta_tako_itta(self):
        """言ったたこと - doubled た (parser sees た + たこ)."""
        text = "彼の言ったたことは本当ではなかった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_ta_tako_kangaeta(self):
        """考えたたこと - doubled た (parser sees た + たこ)."""
        text = "看護婦になろうと考えたたことはありますか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_ta_tako_toukou(self):
        """投稿したたこと - doubled た (parser sees た + たこ)."""
        text = "ウィキペディアに記事を投稿したたことはありますか？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_ta_koto_correct(self):
        """言ったこと - correct form should be grammatical."""
        text = "彼の言ったことは本当ではなかった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Classical-irregular-ra terminal + 。 (incomplete sentence) ---

    def test_ungrammatical_ari_terminal(self):
        """あり。 - sentence ending with conjunctive あり is incomplete."""
        text = "机の上に本があり。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_arimasu_correct(self):
        """あります。 - correct form should be grammatical."""
        text = "机の上に本があります。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- た (terminal) + だ (terminal) pattern ---

    def test_ungrammatical_ta_da_nakatta(self):
        """なかっただ - past negative + だ is wrong."""
        text = "母はいい母親ではなかっただ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_ta_da_ureshikatta(self):
        """うれしかっただ - past adjective + だ is wrong."""
        text = "ここで君に会えてうれしかっただ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_ta_darou_correct(self):
        """ただろう - た + だろう is grammatically correct."""
        text = "彼は学生だっただろう。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- Godan verb imperfective + て (missing っ) ---

    def test_ungrammatical_imperfective_te_kakaru(self):
        """かかている - godan imperfective + て is wrong (should be かかっている)."""
        text = "繁栄は勤勉にかかている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_imperfective_te_mukau(self):
        """向かて - godan imperfective + て is wrong (should be 向かって)."""
        text = "彼はアメリカに向かて航海にでた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_geminate_te_correct(self):
        """かかっている - correct geminate te-form should be grammatical."""
        text = "繁栄は勤勉にかかっている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- i-adjective terminal + ない (wrong negation) ---

    def test_ungrammatical_i_adj_nai_tooi(self):
        """遠いない - i-adj terminal + ない is wrong (should be 遠くない)."""
        text = "駅はここから遠いない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_i_adj_nai_isogashii(self):
        """忙しいない - i-adj terminal + ない is wrong (should be 忙しくない)."""
        text = "忙しいないときにいつかお訪ね下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_i_adj_nai_takai(self):
        """高いない - i-adj terminal + ない is wrong (should be 高くない)."""
        text = "健は背が高いが、私は高いない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_i_adj_ku_nai_correct(self):
        """遠くない - correct i-adj negation should be grammatical."""
        text = "駅はここから遠くない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- shp (na-adj) + ない (wrong negation) ---

    def test_ungrammatical_shp_nai_suki(self):
        """好きない - shp + ない is wrong (should be 好きではない)."""
        text = "最近の服はピタッとした感じが多くて好きない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_shp_nai_akiraka(self):
        """明らかない - shp + ない is wrong (should be 明らかではない)."""
        text = "火元は明らかない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_shp_dewanai_correct(self):
        """好きではない - correct na-adj negation should be grammatical."""
        text = "最近の服は好きではない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- i-adjective conjunctive + な (wrong attributive form) ---

    def test_ungrammatical_adj_conjunctive_na_hayaku(self):
        """早くな出発 - i-adj conjunctive + な is wrong (should be 早い出発 or 早く出発)."""
        text = "私たちはできるだけ早くな出発したほうがよいと案内人は言った。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_adj_conjunctive_na_hidoku(self):
        """ひどくな軽蔑 - i-adj conjunctive + な is wrong."""
        text = "私は彼をひどくな軽蔑している。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_adj_conjunctive_na_ayauku(self):
        """危うくな溺死 - i-adj conjunctive + な is wrong."""
        text = "彼は危うくな溺死するところだった。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_adj_conjunctive_correct(self):
        """早く出発 - i-adj conjunctive without な should be grammatical."""
        text = "私たちはできるだけ早く出発したほうがよい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- ます conjunctive + た (wrong form for past) ---

    def test_ungrammatical_masu_ta_shimashita_koto(self):
        """しましたことがない - wrong form (should be したことがない)."""
        text = "僕らは一度もそれに挑戦しましたことがない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_masu_ta_ikimashita_koto(self):
        """行っましたことがある - wrong form (should be 行ったことがある)."""
        text = "ハワイへ行っましたことがありますか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_masu_ta_iimashita_koto(self):
        """言っましたこと - wrong form (should be 言ったこと)."""
        text = "彼の言っましたことを決してほんとうでない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_ta_koto_ga_aru_correct(self):
        """したことがある - correct form should be grammatical."""
        text = "僕らは一度もそれに挑戦したことがない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- godan-ka conjunctive + か (wrong question form) ---

    def test_ungrammatical_godan_conjunctive_ka_iki(self):
        """行きか - godan conjunctive + か is wrong (should be 行くか or 行きますか)."""
        text = "「ジムはどのようにして学校に行きか」「バスで行きます」"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_godan_conjunctive_ka_yaki(self):
        """焼きか - godan conjunctive + か is wrong."""
        text = "ステーキはどのように焼きか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_godan_terminal_ka_correct(self):
        """行くか - correct question form should be grammatical."""
        text = "学校に行くか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- ka-irregular (来る) conjunctive + で (particle) - should be て form ---

    def test_ungrammatical_verb_conjunctive_de_kite(self):
        """来でくれる - ka-irregular conjunctive + で is wrong (should be 来てくれる)."""
        text = "智子は友だちに、パーティに来でくれるよう頼んだ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_verb_conjunctive_de_kite2(self):
        """来でくれる - ka-irregular conjunctive + で is wrong (should be 来てくれる)."""
        text = "彼が来でくれるだろうと期待していた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_kite_kudasai_correct(self):
        """来てください - correct form should be grammatical."""
        text = "明日来てください。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb conjunctive + できる (missing て) ---

    def test_ungrammatical_wasure_dekita(self):
        """忘れできた - verb conjunctive + できる is wrong (should be 忘れてきた)."""
        text = "家にクレジットカードを忘れできた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_oboe_deki(self):
        """覚えできなさい - verb conjunctive + できる is wrong (should be 覚えてきなさい)."""
        text = "来週までにこの詩を覚えできなさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_wasurete_kita_correct(self):
        """忘れてきた - correct form should be grammatical."""
        text = "家にクレジットカードを忘れてきた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- と + い (verb) - should be といって ---

    def test_ungrammatical_toite_mazushii(self):
        """といて人を - should be といって人を."""
        text = "貧しいからといて人を軽蔑してはならない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_toite_yuui(self):
        """といて油断 - should be といって油断."""
        text = "いくら自分たちが優位だからといて油断してはならない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_toitte_correct(self):
        """といって - correct form should be grammatical."""
        text = "貧しいからといって人を軽蔑してはならない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb terminal + ます (wrong form) ---

    def test_ungrammatical_verb_terminal_masu_mamoru(self):
        """守るます - verb terminal + ます is wrong (should be 守ります)."""
        text = "公害からこの美しい地球を守るますために、私たちは何をしなければならないのか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_verb_terminal_masu_hairu(self):
        """入るます - verb terminal + ます is wrong (should be 入ります)."""
        text = "彼女は部屋へ入るますとき帽子を取った。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_verb_terminal_masu_suru(self):
        """するます - verb terminal + ます is wrong (should be します)."""
        text = "大きく強く成長するますためには野菜を食べなさい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_verb_conjunctive_masu_correct(self):
        """守ります - correct conjunctive + ます should be grammatical."""
        text = "この美しい地球を守りますために、努力しています。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb imperfective + あり (wrong negation form) ---

    def test_ungrammatical_imperfective_ari_ika(self):
        """行かあり - verb imperfective + あり is wrong (should be 行きません)."""
        text = "私は行かありませんことを忠告する。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_imperfective_ari_tsumara(self):
        """つまらあり - verb imperfective + あり is wrong (should be つまりません)."""
        text = "つまらありませんことで腹を立てるなよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_conjunctive_masen_correct(self):
        """行きません - correct conjunctive + ません should be grammatical."""
        text = "私は行きませんことを忠告する。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- shp (na-adjective with hiragana) + noun (missing な) ---

    def test_ungrammatical_shp_hiragana_noun_kirei(self):
        """きれい花 - na-adj with hiragana + noun without な is wrong (should be きれいな花)."""
        text = "きれい花を見て"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_shp_na_noun_correct(self):
        """きれいな花 - na-adj + な + noun should be grammatical."""
        text = "きれいな花を見て"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb conjunctive + なら (wrong conditional form) ---

    def test_ungrammatical_conjunctive_nara_shi(self):
        """しなら - verb conjunctive + なら is wrong (should be したら/するなら)."""
        text = "もっとお金がしなら、旅行に行きたい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_conjunctive_nara_iki(self):
        """行きなら - verb conjunctive + なら is wrong (should be 行ったら/行くなら)."""
        text = "彼が行きなら、私も行く。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_tara_conditional_correct(self):
        """行ったら - correct conditional form should be grammatical."""
        text = "彼が行ったら、私も行く。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_nara_conditional_correct(self):
        """行くなら - correct terminal + なら should be grammatical."""
        text = "彼が行くなら、私も行く。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- i-ichidan-a verb terminal + を (wrong verb ending) ---

    def test_ungrammatical_ichidan_terminal_wo_iru(self):
        """いるを思う - ichidan terminal + を is wrong (should be いると思う)."""
        text = "彼はいるを思いますか。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_ichidan_terminal_wo_deru(self):
        """出るを - ichidan terminal + を is wrong (should be 出るのを)."""
        text = "出るをやめた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_no_wo_nominalization_correct(self):
        """いるのを - correct nominalization with の should be grammatical."""
        text = "私のペンを持っているのを見てください。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- shp (na-adjective) + suffix (missing な, except nominalization suffixes) ---

    def test_ungrammatical_shp_suffix_kirei(self):
        """きれい上 - na-adj + suffix without な is wrong (should be きれいな上)."""
        text = "きれい上に高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_shp_suffix_taisetsu(self):
        """大切人 - na-adj + suffix without な is wrong (should be 大切な人)."""
        text = "彼女は大切人だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_shp_na_suffix_correct(self):
        """きれいな上 - na-adj + な + suffix should be grammatical."""
        text = "きれいな上に高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    def test_grammatical_shp_sa_nominalization(self):
        """勇敢さ - na-adj + さ (nominalization suffix) should be grammatical."""
        text = "彼の勇敢さを称える"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb terminal + を (wrong nominalization) ---

    def test_ungrammatical_verb_terminal_wo_iru(self):
        """いるを - verb terminal + を is wrong (should use の or こと)."""
        text = "草が生えているをところに水はない。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_verb_terminal_wo_aru(self):
        """あるをこと - verb terminal + を is wrong."""
        text = "鯨は最大の哺乳動物であるをことはよく知られている。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_yamu_wo_ezu_idiomatic(self):
        """やむを得ず - idiomatic expression should be grammatical."""
        text = "病気のため、彼はやむを得ず退学した。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)

    # --- verb conjunctive + で (prt) + verb (should be て) ---

    def test_ungrammatical_conjunctive_de_oyogi(self):
        """泳ぎで行く - conjunctive + で + verb is wrong (should be 泳ぎに行く)."""
        text = "私はよく川へ泳ぎで行く。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_ungrammatical_conjunctive_de_kake(self):
        """かけで下さい - conjunctive + で + verb is wrong (should be かけてください)."""
        text = "局部には必ずモザイクをかけで下さい。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertFalse(result)

    def test_grammatical_buttsudukede(self):
        """ぶっつづけで - nominalized verb + で is grammatical."""
        text = "彼は５時間以上もぶっつづけで働かされた。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = grammaticality(kotogram, use_model=False)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
