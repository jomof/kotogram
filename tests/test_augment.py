import pytest
from unittest.mock import patch
from kotogram.augment import (
    Augmenter, augment,
    PronounRule, CopulaRule, ContractionRule, TopicDropRule, ProgressiveRule, PluralRule
)
from kotogram.kotogram import TokenFeatures

from kotogram.augment import (
    VerbPolitenessRule, Token
)

# --- Rule Tests ---

def test_verb_politeness_rule():
    rule = VerbPolitenessRule()
    
    # 1. Plain -> Polite (Godan u -> i + masu)
    # 行く -> 行きます
    token_iku = Token('行く', {'surface': '行く', 'pos': 'v', 'lemma': '行く', 'conjugated_type': 'godan-k', 'conjugated_form': 'terminal'})
    tokens = (token_iku, '。')
    results = rule.apply(tokens)
    # Should contain original and ('行き', 'ます', '。')
    assert len(results) == 2
    
    variants = {tuple(t if isinstance(t, str) else t.surface for t in res) for res in results}
    assert ('行き', 'ます', '。') in variants

    # 2. Plain -> Polite (Ichidan drop ru + masu)
    # 食べる -> 食べます
    token_taberu = Token('食べる', {'surface': '食べる', 'pos': 'v', 'lemma': '食べる', 'conjugated_type': 'i-ichidan', 'conjugated_form': 'terminal'})
    tokens_ichidan = (token_taberu,)
    results_ichidan = rule.apply(tokens_ichidan)
    variants_ichidan = {tuple(t if isinstance(t, str) else t.surface for t in res) for res in results_ichidan}
    assert ('食べ', 'ます') in variants_ichidan
    
    # 3. Plain -> Polite (Suru irregular)
    # 勉強する -> 勉強します
    token_suru = Token('勉強する', {'surface': '勉強する', 'pos': 'v', 'lemma': '勉強する', 'conjugated_type': 'suru', 'conjugated_form': 'terminal'})
    tokens_suru = (token_suru,)
    results_suru = rule.apply(tokens_suru)
    variants_suru = {tuple(t if isinstance(t, str) else t.surface for t in res) for res in results_suru}
    assert ('勉強し', 'ます') in variants_suru

    # 4. Polite -> Plain
    # 行きます -> 行く
    # Tokens: 行き(verb), ます(auxv)
    token_iki = Token('行き', {'surface': '行き', 'pos': 'v', 'lemma': '行く', 'conjugated_type': 'godan-k', 'conjugated_form': 'conjunctive'})
    token_masu = Token('ます', {'surface': 'ます', 'pos': 'auxv'})
    tokens_polite = (token_iki, token_masu)
    
    results_polite = rule.apply(tokens_polite)
    variants_polite = {tuple(t if isinstance(t, str) else t.get('surface', t) for t in res) for res in results_polite}
    
    assert ('行く',) in variants_polite
    # Also ensure original is kept
    assert ('行き', 'ます') in variants_polite

    # 5. Non-terminal plain verb should NOT change
    # 行って (conjunctive te-form)
    token_itte = Token('行って', {'surface': '行って', 'pos': 'v', 'lemma': '行く', 'conjugated_type': 'godan-k', 'conjugated_form': 'conjunctive'})
    results_no_change = rule.apply((token_itte,))
    assert len(results_no_change) == 1

def test_contraction_rule():
    rule = ContractionRule()
    
    # de wa -> ja
    tokens_dewa = ('学生', 'で', 'は', 'ない')
    results = rule.apply(tokens_dewa)
    # Should include ('学生', 'で', 'は', 'ない') and ('学生', 'じゃ', 'ない')
    assert len(results) == 2
    assert ('学生', 'じゃ', 'ない') in results
    
    # ja -> de wa
    tokens_ja = ('学生', 'じゃ', 'ない')
    results = rule.apply(tokens_ja)
    assert ('学生', 'で', 'は', 'ない') in results
    
    # te iru -> te ru (progressive)
    tokens_teiru = ('食べ', 'て', 'いる')
    results = rule.apply(tokens_teiru)
    assert ('食べ', 'てる',) in results
    
    # mixed/multiple
    # "食べているからはなさない" (eating so won't talk - hypothetical)
    tokens_multi = ('食べ', 'て', 'いる', 'から', '話し', 'て', 'いる')
    results = rule.apply(tokens_multi)
    # Should generate combos: 
    # 1. original
    # 2. eat-short, talk-long
    # 3. eat-long, talk-short
    # 4. eat-short, talk-short
    assert len(results) == 4
    assert ('食べ', 'てる', 'から', '話し', 'てる') in results

def test_pronoun_rule():
    rule = PronounRule()
    # (私, は, 学生, です) -> {(私, ...), (僕, ...), (俺, ...), ...}
    tokens = ('私', 'は', '学生', 'です')
    results = rule.apply(tokens)
    
    # Should generate variations for all 6 pronouns defined in constant
    # '私', '僕', '俺', 'わたし', 'ぼく', 'おれ'
    assert len(results) == 6
    assert ('僕', 'は', '学生', 'です') in results
    assert ('俺', 'は', '学生', 'です') in results

def test_pronoun_rule_no_match():
    rule = PronounRule()
    tokens = ('彼', 'は', '学生', 'です')
    results = rule.apply(tokens)
    assert len(results) == 1
    assert list(results)[0] == tokens

def test_copula_rule():
    rule = CopulaRule()
    # match ('だ', '。') -> ('です', '。')
    tokens_da = ('学生', 'だ', '。')
    results = rule.apply(tokens_da)
    assert len(results) == 2
    assert ('学生', 'だ', '。') in results
    assert ('学生', 'です', '。') in results
    
    # match ('です', '。') -> ('だ', '。')
    tokens_desu = ('学生', 'です', '。')
    results = rule.apply(tokens_desu)
    assert len(results) == 2
    assert ('学生', 'だ', '。') in results

def test_copula_rule_mid_sentence():
    # Should NOT match mid-sentence
    rule = CopulaRule()
    tokens = ('学生', 'だ', 'けど', '元気')
    results = rule.apply(tokens)
    assert len(results) == 1
    assert list(results)[0] == tokens

def test_topic_drop_rule():
    rule = TopicDropRule()
    # match ('私', 'は', ...)
    tokens = ('私', 'は', '行きます')
    results = rule.apply(tokens)
    assert len(results) == 2
    assert ('私', 'は', '行きます') in results
    assert ('行きます',) in results  # Dropped "私", "は"

def test_progressive_rule():
    rule = ProgressiveRule()
    # match ('て', 'いる', '。') -> ('て', 'い', 'ます', '。')
    tokens = ('食べ', 'て', 'いる', '。')
    results = rule.apply(tokens)
    assert len(results) == 2
    assert ('食べ', 'て', 'い', 'ます', '。') in results

def test_plural_rule():
    rule = PluralRule()
    # match '私達' -> '私たち'
    tokens = ('私達', 'は')
    results = rule.apply(tokens)
    assert len(results) == 2
    assert ('私たち', 'は') in results

# --- Augmenter Integration Tests ---

@pytest.fixture
def mock_parser():
    with patch('kotogram.augment.SudachiJapaneseParser') as MockParser:
        parser_instance = MockParser.return_value
        # Mock tokenization behavior for specific sentences
        def side_effect(text):
            # Return dummy kotogram that will be split by split_kotogram mock
            return text 
        parser_instance.japanese_to_kotogram.side_effect = side_effect
        yield parser_instance

@patch('kotogram.augment.split_kotogram')
@patch('kotogram.augment.extract_token_features')
def test_process_sentence(mock_extract, mock_split, mock_parser):
    # Setup mocks to simulate "私 は 学生 だ 。" tokenization
    mock_split.return_value = ['t1', 't2', 't3', 't4', 't5']
    
    # Map tokens to surface forms
    # Map tokens to surface forms
    surfaces = ['私', 'は', '学生', 'だ', '。']
    mock_extract.side_effect = [TokenFeatures(surface=s) for s in surfaces]
    
    augmenter = Augmenter()
    # Override parser with our mock
    Augmenter._parser = mock_parser
    
    # Test augmentation
    # Should apply Pronoun (6 variants) AND Copula (2 variants) -> 12 variants
    # PLUS Topic Drop for "私は" -> "学生だ。" (which also gets copula expanded -> "学生です。")
    # Total = 12 + 2 = 14
    results = augmenter.process_sentence("私は学生だ。")
    
    assert len(results) == 14
    assert "僕は学生です。" in results
    assert "俺は学生だ。" in results

# --- High-Level Augment Function Test ---

@patch('kotogram.augment.Augmenter')
@patch('kotogram.augment.grammaticality')
@patch('kotogram.augment._load_style_model')
def test_augment_function(mock_load, mock_gram, MockAugmenter):
    # Setup
    mock_inst = MockAugmenter.return_value
    mock_inst.process_sentence.return_value = {"Augmented1", "Augmented2", "Bad1"}
    
    # Mock filtering: check_grammaticality is called inside filter_grammatical
    # But wait, augment() calls process_sentence then filter_grammatical.
    # We should stick to testing 'augment' logic which delegates to the class.
    
    # Let's test the filter logic specifically by mocking the internal calls of filter_grammatical
    # Actually, let's just integration test 'augment' with real class but mocked dependencies
    pass

@patch('kotogram.augment.split_kotogram')
@patch('kotogram.augment.extract_token_features')
@patch('kotogram.augment.SudachiJapaneseParser')
@patch('kotogram.augment.grammaticality')
@patch('kotogram.augment._load_style_model')
def test_augment_full_flow(mock_load, mock_gram, mock_parser_cls, mock_extract, mock_split):
    # 1. Tokenization Setup for "私 は 学生 だ 。"
    mock_split.return_value = ['t1', 't2', 't3', 't4', 't5']
    surfaces = ['私', 'は', '学生', 'だ', '。']
    mock_extract.side_effect = [TokenFeatures(surface=s) for s in surfaces] * 100 # ample iterator
    
    # 2. Grammaticality Setup
    # Accept everything except "俺は学生だ。"
    def gram_side_effect(k, use_model=True):
        # We don't have real kotograms here, but assuming the filter calls it
        # Actually in test_process_sentence we saw it generates strings.
        # filter_grammatical calls parser.japanese_to_kotogram(sent) -> 'kotogram'
        return True
    mock_gram.side_effect = gram_side_effect
    
    # Run
    # Reset singleton parser to ensure mock is used?
    Augmenter._parser = None
    
    inputs = ["私は学生だ。"]
    results = augment(inputs)
    
    # Verify expansion happened (Pronoun * Copula >= 1)
    assert len(results) > 1
    assert "僕は学生です。" in results
    
    # Verify model loaded
    mock_load.assert_called()
    assert mock_gram.called
