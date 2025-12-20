"""Abstract base class for Japanese text parsing with shared mapping constants."""

from abc import ABC, abstractmethod

# Global mapping constants shared across all Japanese parser implementations

# Part-of-speech mappings
POS_MAP = {
    "名詞": "noun",        # Noun  
    "動詞": "verb",        # Verb
    "形容詞": "adj",       # Adjective
    "副詞": "adv",         # Adverb
    "助詞": "particle",    # Particle
    "助動詞": "aux-verb",   # Auxiliary verb
    "接続詞": "conj",       # Conjunction
    "感動詞": "interj",     # Interjection
    "代名詞": "pron",       # Pronoun
    "連体詞": "adnom",      # Adnominal (rentaishi)
    "接頭辞": "prefix",     # Prefix
    "接尾辞": "suffix",     # Suffix
    "形状詞": "adjectival-noun",  # Adjectival noun (keiyōdōshi)
    "記号": "symbol",      # Symbol
    "補助記号": "aux-symbol",  # Auxiliary symbol
    "空白": "whitespace",  # Whitespace
    "*": '',              # Unspecified/empty field marker

}

# Part-of-speech detail level 1 mappings
POS1_MAP = {
    "一般": "general",                        # General
    "固有名詞": "proper-noun",                 # Proper noun
    "普通名詞": "common-noun",                 # Common noun
    "数詞": "numeral",                        # Numeral
    "格助詞": "case-particle",                 # Case particle (を, に, etc.)
    "係助詞": "binding-particle",              # Binding particle (は, も, etc.)
    "副助詞": "adverbial-particle",            # Adverbial particle
    "接続助詞": "conjunctive-particle",         # Conjunctive particle (て, ば, etc.)
    "終助詞": "sentence-final-particle",       # Sentence-final particle (ね, よ, etc.)
    "準体助詞": "nominal-particle",             # Nominal particle (の as nominalizer)
    "助動詞語幹": "aux-verb-stem",             # Auxiliary verb stem
    "非自立可能": "bound",                      # Bound (non-independent) form
    "動詞的": "verbal",                        # Verbal
    "形容詞的": "adjectival",                   # Adjectival
    "形状詞的": "adjectival-noun-like",         # Adjectival-noun-like
    "名詞的": "nominal",                       # Nominal
    "タリ": "tari",                           # Tari form
    "フィラー": "filler",                      # Filler word (ええと, あの, etc.)
    "文字": "letter",                         # Letter/character (α, β, γ, etc.)
    "ＡＡ": "ascii-art",                      # ASCII art / text emoticon
    "句点": "period",                         # Period (。)
    "読点": "comma",                          # Comma (、)
    "括弧開": "open-bracket",                  # Opening bracket
    "括弧閉": "close-bracket",                 # Closing bracket
    "*": '',                                 # Unspecified/empty field marker

}

# Part-of-speech detail level 2 mappings
POS2_MAP = {
    "一般": "general",                    # General
    "サ変可能": "verbal-suru",            # Can be used with suru (する)
    "サ変形状詞可能": "verbal-suru-adj",   # Verbal suru + adjectival noun
    "副詞可能": "adverbial",              # Can be used as adverb
    "形状詞可能": "adjectival-noun-possible",  # Can be adjectival noun
    "助数詞": "counter",                  # Counter word (個, 本, etc.)
    "助数詞可能": "counter-possible",      # Can be counter word
    "地名": "place-name",                # Place name
    "人名": "person-name",               # Person name
    "顔文字": "kaomoji",                  # Kaomoji/emoticon (^_^, etc.)
    "*": '',                            # Unspecified/empty field marker

}

# Part-of-speech detail level 3 mappings
POS3_MAP = {
    "一般": "general",      # General place/person names
    "国": "country",        # Country names (日本, アメリカ, etc.)
    "名": "given-name",     # Given names (太郎, 花子, etc.)
    "姓": "surname",        # Family names (山田, 佐藤, etc.)
    "*": '',               # Unspecified/empty field marker

}

# Conjugation type mappings
CONJUGATED_TYPE_MAP = {
    # Auxiliary verbs
    "助動詞-タ": "aux-ta",                  # Past tense auxiliary (だった)
    "助動詞-ダ": "aux-da",                  # Copula (だ)
    "助動詞-デス": "aux-desu",              # Polite copula (です)
    "助動詞-マス": "aux-masu",              # Polite auxiliary (ます)
    "助動詞-ナイ": "aux-nai",               # Negative auxiliary (ない)
    "助動詞-ヌ": "aux-nu",                  # Classical negative (ぬ)
    "助動詞-レル": "aux-reru",              # Passive/potential auxiliary (れる/られる)
    "助動詞-タイ": "aux-tai",               # Desiderative auxiliary (たい - want to)
    "助動詞-ラシイ": "aux-rashii",          # Evidential auxiliary (らしい - seems)
    "助動詞-マイ": "aux-mai",               # Negative volitional (まい)
    "助動詞-ジャ": "aux-ja",                # Contracted copula (じゃ)
    "助動詞-ヤ": "aux-ya",                  # Classical auxiliary (や)
    "助動詞-ナンダ": "aux-nanda",           # Colloquial (なんだ)
    "助動詞-ヘン": "aux-hen",               # Kansai dialect negative (へん)
    
    # Godan (五段) verbs - 5-row conjugation
    "五段-ラ行": "godan-ra",               # Ra-row godan (作る to make)
    "五段-カ行": "godan-ka",               # Ka-row godan (書く to write)
    "五段-ガ行": "godan-ga",               # Ga-row godan (泳ぐ to swim)
    "五段-サ行": "godan-sa",               # Sa-row godan (話す to speak)
    "五段-タ行": "godan-ta",               # Ta-row godan (立つ to stand)
    "五段-ナ行": "godan-na",               # Na-row godan (死ぬ to die, rare)
    "五段-バ行": "godan-ba",               # Ba-row godan (遊ぶ to play)
    "五段-マ行": "godan-ma",               # Ma-row godan (読む to read)
    "五段-ワア行": "godan-waa",            # Waa-row godan (言う to say)
    
    # Ichidan (一段) verbs - 1-row conjugation
    "上一段-ア行": "upper-ichidan-a",     # Upper a-row ichidan (いる to exist)
    "上一段-カ行": "upper-ichidan-ka",    # Upper ka-row ichidan (起きる to wake)
    "上一段-ガ行": "upper-ichidan-ga",    # Upper ga-row ichidan (過ぎる to pass)
    "上一段-ザ行": "upper-ichidan-za",    # Upper za-row ichidan (信じる to believe)
    "上一段-タ行": "upper-ichidan-ta",    # Upper ta-row ichidan (落ちる to fall)
    "上一段-ナ行": "upper-ichidan-na",    # Upper na-row ichidan (archaic)
    "上一段-ハ行": "upper-ichidan-ha",    # Upper ha-row ichidan (rare)
    "上一段-バ行": "upper-ichidan-ba",    # Upper ba-row ichidan (浴びる to bathe)
    "上一段-マ行": "upper-ichidan-ma",    # Upper ma-row ichidan (見る to see)
    "上一段-ラ行": "upper-ichidan-ra",    # Upper ra-row ichidan (居る archaic)
    "下一段-ア行": "lower-ichidan-a",     # Lower a-row ichidan (rare)
    "下一段-カ行": "lower-ichidan-ka",    # Lower ka-row ichidan (受ける to receive)
    "下一段-ガ行": "lower-ichidan-ga",    # Lower ga-row ichidan (上げる to raise)
    "下一段-サ行": "lower-ichidan-sa",    # Lower sa-row ichidan (せる causative)
    "下一段-ザ行": "lower-ichidan-za",    # Lower za-row ichidan (教える to teach)
    "下一段-タ行": "lower-ichidan-ta",    # Lower ta-row ichidan (捨てる to throw away)
    "下一段-ダ行": "lower-ichidan-da",    # Lower da-row ichidan (出る to exit)
    "下一段-ナ行": "lower-ichidan-na",    # Lower na-row ichidan (寝る to sleep)
    "下一段-ハ行": "lower-ichidan-ha",    # Lower ha-row ichidan (減る to decrease)
    "下一段-バ行": "lower-ichidan-ba",    # Lower ba-row ichidan (食べる to eat)
    "下一段-マ行": "lower-ichidan-ma",    # Lower ma-row ichidan (止める to stop)
    "下一段-ラ行": "lower-ichidan-ra",    # Lower ra-row ichidan (入れる to put in)
    
    # Irregular verbs
    "カ行変格": "ka-irregular",           # Ka-row irregular (来る to come)
    "サ行変格": "sa-irregular",           # Sa-row irregular (する to do)
    
    # Adjectives
    "形容詞": "i-adjective",              # I-adjectives (高い tall)
    
    # Classical Japanese conjugations
    "文語サ行変格": "classical-sa-irregular",       # Classical sa-irregular (す)
    "文語ラ行変格": "classical-ra-irregular",       # Classical ra-irregular
    "文語形容詞-ク": "classical-adj-ku",           # Classical ku-adjective
    "文語形容詞-シク": "classical-adj-shiku",       # Classical shiku-adjective
    "文語助動詞-タリ-完了": "classical-aux-tari-perfective",  # Classical perfective tari
    "文語助動詞-タリ-断定": "classical-aux-tari-assertive",   # Classical assertive tari
    "文語助動詞-ナリ-断定": "classical-aux-nari",    # Classical nari copula
    "文語助動詞-リ": "classical-aux-ri",            # Classical perfective ri
    "文語助動詞-ベシ": "classical-aux-beshi",       # Classical beshi (should)
    "文語助動詞-ズ": "classical-aux-zu",            # Classical negative zu
    "文語助動詞-キ": "classical-aux-ki",            # Classical past ki
    "文語助動詞-ケリ": "classical-aux-keri",        # Classical perfect keri
    "文語助動詞-ゴトシ": "classical-aux-gotoshi",    # Classical gotoshi (like)
    "文語助動詞-マジ": "classical-aux-maji",        # Classical negative presumptive
    "文語助動詞-ム": "classical-aux-mu",            # Classical presumptive mu
    "文語助動詞-ジ": "classical-aux-ji",            # Classical ji
    "文語助動詞-ヌ": "classical-aux-nu",            # Classical nu
    "文語助動詞-ラシ": "classical-aux-rashi",       # Classical evidential rashi
    "文語助動詞-ラム": "classical-aux-ramu",        # Classical presumptive ramu
    "文語助動詞-ザマス": "classical-aux-zamasu",     # Colloquial polite zamasu
    "文語上二段-タ行": "classical-upper-nidan-ta",  # Classical upper nidan ta
    "文語上二段-ダ行": "classical-upper-nidan-da",  # Classical upper nidan da
    "文語上二段-バ行": "classical-upper-nidan-ba",  # Classical upper nidan ba
    "文語下二段-ア行": "classical-lower-nidan-a",   # Classical lower nidan a
    "文語下二段-カ行": "classical-lower-nidan-ka",  # Classical lower nidan ka
    "文語下二段-ガ行": "classical-lower-nidan-ga",  # Classical lower nidan ga
    "文語下二段-サ行": "classical-lower-nidan-sa",  # Classical lower nidan sa
    "文語下二段-ダ行": "classical-lower-nidan-da",  # Classical lower nidan da
    "文語下二段-ナ行": "classical-lower-nidan-na",  # Classical lower nidan na
    "文語下二段-ハ行": "classical-lower-nidan-ha",  # Classical lower nidan ha
    "文語下二段-マ行": "classical-lower-nidan-ma",  # Classical lower nidan ma
    "文語下二段-ラ行": "classical-lower-nidan-ra",  # Classical lower nidan ra
    "文語四段-カ行": "classical-yodan-ka",         # Classical yodan ka
    "文語四段-サ行": "classical-yodan-sa",         # Classical yodan sa
    "文語四段-タ行": "classical-yodan-ta",         # Classical yodan ta
    "文語四段-ハ行": "classical-yodan-ha",         # Classical yodan ha
    "文語四段-マ行": "classical-yodan-ma",         # Classical yodan ma
    "文語四段-ラ行": "classical-yodan-ra",         # Classical yodan ra
    "*": '',                                      # Unspecified

}

# Conjugation form mappings
CONJUGATED_FORM_MAP = {
    "終止形-一般": "terminal",                    # Terminal/conclusive form
    "終止形-撥音便": "terminal-nasal",            # Terminal with nasal euphony
    "終止形-促音便": "terminal-geminate",          # Terminal with geminate
    "終止形-融合": "terminal-fused",              # Terminal fused form
    "終止形-ウ音便": "terminal-u-euphonic",        # Terminal u-euphonic change
    "連用形-一般": "continuative",                # Continuative/conjunctive form
    "連用形-促音便": "continuative-geminate",      # Continuative with geminate (っ)
    "連用形-撥音便": "continuative-nasal",         # Continuative with nasal (ん)
    "連用形-イ音便": "continuative-i-euphonic",    # Continuative i-euphonic change
    "連用形-ウ音便": "continuative-u-euphonic",    # Continuative u-euphonic change
    "連用形-ニ": "continuative-ni",               # Continuative ni form
    "連用形-省略": "continuative-abbreviated",     # Abbreviated continuative
    "連用形-融合": "continuative-fused",          # Fused continuative
    "連用形-補助": "continuative-auxiliary",       # Auxiliary continuative
    "連体形-一般": "attributive",                  # Attributive/adnominal form
    "連体形-撥音便": "attributive-nasal",          # Attributive with nasal
    "連体形-省略": "attributive-abbreviated",       # Abbreviated attributive
    "連体形-補助": "attributive-auxiliary",         # Auxiliary attributive
    "未然形-一般": "irrealis",                     # Irrealis/imperfective form
    "未然形-サ": "irrealis-sa",                    # Irrealis sa form
    "未然形-セ": "irrealis-se",                    # Irrealis se form (causative)
    "未然形-撥音便": "irrealis-nasal",             # Irrealis with nasal
    "未然形-補助": "irrealis-auxiliary",            # Auxiliary irrealis
    "仮定形-一般": "conditional",                  # Conditional/hypothetical form
    "仮定形-融合": "conditional-fused",            # Fused conditional
    "命令形": "imperative",                       # Imperative form
    "意志推量形": "volitional-presumptive",        # Volitional/presumptive form
    "已然形-一般": "realis",                       # Realis/perfective (classical)
    "語幹-一般": "stem",                          # Verb stem
    "語幹-サ": "stem-sa",                         # Stem sa form
    "ク語法": "ku-form",                          # Ku-form (classical)
    "*": '',                                     # Unspecified

}

# Part-of-speech to character mappings
POS_TO_CHARS = {
    "particle": ['は', 'が', 'を', 'に', 'へ', 'と', 'で', 'か', 'の', 'ね', 'よ', 'て',
            'わ', 'も', 'ぜ', 'ん', 'な', 'ば', 'ぞ', 'し', 'さ', 'や', 'ら', 'ど',
            'い', 'つ', 'べ', 'け', 'ょ'],
    "symbol": [],
    "aux-symbol": ['。', '、', '・', '：', '；', '？', '！', '…', '「', '」', '『', '』',
             '{', '}', '.', 'ー', ':', '?', 'っ', '-', '々', '(', ')', '[', ']',
             '<', '>', '／', '＼', '＊', '＋', '＝', '＠', '＃', '％', '＆', '＊',
             'ぇ', '〇', '（', '）', '* ', '*', '～', '"', '◯']
}

# Character to part-of-speech reverse mapping
CHAR_TO_POS = {
    ch: pos
    for pos, chars in POS_TO_CHARS.items()
    for ch in chars
}


class JapaneseParser(ABC):
    """Abstract base class for Japanese text parsing.

    Implementations should parse Japanese text into a compact representation
    (kotogram format) that encodes linguistic information about each token.
    """

    @abstractmethod
    def japanese_to_kotogram(self, text: str) -> str:
        """Convert Japanese text to kotogram compact representation.

        Args:
            text: Japanese text to parse

        Returns:
            Kotogram compact sentence representation
        """
        pass
