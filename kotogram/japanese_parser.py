"""Abstract base class for Japanese text parsing with shared mapping constants."""

from abc import ABC, abstractmethod

# Global mapping constants shared across all Japanese parser implementations

# Part-of-speech mappings
POS_MAP = {
    "名詞": "n",  # Noun
    "普通名詞": "cn",  # Common noun
    "動詞": "v",  # Verb
    "形容詞": "adj",  # Adjective
    "副詞": "adv",  # Adverb
    "助詞": "prt",  # Particle
    "接続詞": "conj",  # Conjunction
    "感動詞": "int",  # Interjection
    "空白": "space",  # Whitespace
    "記号": "sym",  # Symbol
    "助動詞": "auxv",  # Auxiliary verb
    "補助記号": "auxs",  # Auxiliary symbol
    "代名詞": "pron",  # Pronoun
    "接頭辞": "pref",  # Prefix
    "接尾辞": "suff",  # Suffix
    "形状詞": "shp",  # Shape word
    "連体詞": "at",  # Attributive
}

# Part-of-speech detail level 2 mappings
POS2_MAP = {
    "副詞可能": "adverb-possible",
    "一般": "general",
    "サ変可能": "suru-possible",
    "地名": "place-name",
    "状詞可能": "descriptive-possible",
    "形状詞可能": "pos2-unk1",
    "助数詞可能": "counter-possible",
    "サ変形状詞可能": "pos2-unk2",
    "人名": "person-name",
    "助数詞": "counter",
    "顔文字": "kaomoji",  # Emoticon/kaomoji
    "*": '',  # Unspecified/empty field marker
    "": ''
}

# Part-of-speech detail level 1 mappings
POS1_MAP = {
    "括弧開": 'bracket_open',  # Opening bracket
    "括弧閉": 'bracket_close',  # Closing bracket
    "読点": 'comma',  # Comma
    "固有名詞": 'proper_noun',  # Proper noun
    "格助詞": 'case_particle',  # Case particle
    "普通名詞": 'common_noun',  # Common noun
    "準体助詞": 'pre_noun_particle',  # Pre-noun particle
    "終助詞": 'sentence_final_particle',  # Sentence-final particle
    "句点": 'period',  # Period
    "係助詞": 'binding_particle',  # Binding particle
    "非自立可能": 'non_self_reliant',  # Non-self-reliant
    "一般": 'general',  # General
    "助動詞語幹": 'auxiliary_verb_stem',  # Auxiliary verb stem
    "形容詞的": 'adjectival',  # Adjectival
    "副助詞": 'adverbial_particle',  # Adverbial particle
    "接続助詞": 'conjunctive_particle',  # Conjunctive particle
    "数詞": 'numeral',  # Numeral
    "名詞的": 'noun_like',  # Noun-like
    "フィラー": 'filler',  # Filler
    "形状詞的": 'shape_word_like',  # Shape word-like
    "タリ": 'tari',  # tari (a form of auxiliary verb)
    "動詞的": 'verb_like',  # Verb-like
    "文字": 'character',  # Character/letter (e.g., Greek letters like α, β, γ)
    "ＡＡ": 'ascii_art',  # ASCII art / emoticon
    "*": '',  # Unspecified/empty field marker
    "": ''
}

# Conjugation type mappings
CONJUGATED_TYPE_MAP = {
    "助動詞-タ": "auxv-ta",        # だった (datta - "was")
    "助動詞-ダ": "auxv-da",        # だ (da - "is/am/are")
    "助動詞-マス": "auxv-masu",    # ます (masu - polite ending)
    "助動詞-ヌ": "auxv-nu",        # ぬ (nu - classical negative), classical
    "助動詞-デス": "auxv-desu",    # です (desu - polite copula)
    "助動詞-ナイ": "auxv-nai",     # ない (nai - "not")
    "助動詞-ラシイ": "auxv-rashii",  # らしい (rashii - "seems like/apparently")
    "助動詞-レル": "auxv-reru",    # られる (rareru - passive/potential)
    "助動詞-タイ": "auxv-tai",     # たい (tai - "want to")
    "文語助動詞-リ": "auxv-ri",  # り (ri - classical perfective), classical
    "文語助動詞-ベシ": "auxv-beshi",  # べし (beshi - "should/ought to"), classical
    "文語助動詞-ゴトシ": "auxv-gotoshi",  # ごとし (gotoshi - "like/as if"), classical
    "文語助動詞-ズ": "auxv-zu",  # ず (zu - classical negative auxiliary), classical
    "文語助動詞-キ": "auxv-ki",  # き (ki - classical past tense), classical
    "文語助動詞-ケリ": "auxv-keri",  # けり (keri - classical perfect/recollective), classical
    "文語助動詞-タリ-完了": "auxv-tari-perfective",  # たり (tari - classical perfective), classical
    "文語助動詞-ナリ-断定": "auxv-nari-assertive",  # なり (nari - classical assertive copula), classical
    "文語助動詞-マジ": "auxv-maji",  # まじ (maji - classical negative presumptive), classical
    "文語助動詞-ム": "auxv-mu",  # む (mu - classical presumptive/volitional), classical
    "文語形容詞-シク": "classical-adj-shiku",  # しく (shiku-inflection classical adjective)
    "文語ラ行変格": "classical-irregular-ra",  # Classical ra-row irregular verbs
    "助動詞-マイ": "auxv-mai",  # まい (mai - "probably won't/shouldn't")
    "助動詞-ジャ": "auxv-ja",  # じゃ (ja - contracted copula "is/am/are")
    "助動詞-ヤ": "auxv-ya",  # や (ya - classical question particle/auxiliary)
    "助動詞-ナンダ": "auxv-nanda",  # なんだ (nanda - colloquial past tense of だ)
    "助動詞-ヒン": "auxv-hin",  # ひん (hin - Kansai dialect negative)
    "助動詞-ヘン": "auxv-hen",  # へん (hen - Kansai dialect negative)
    "助動詞-ヤス": "auxv-yasu",  # やす (yasu - polite auxiliary)
    "助動詞-ンス": "auxv-nsu",  # んす (nsu - colloquial/dialectal auxiliary)
    "文語助動詞-タリ-断定": "auxv-tari",  # たり (tari - classical assertive), classical
    "形容詞": "adjective",          # 高い (takai - "tall/expensive")
    "五段-ラ行": "godan-ra",       # 作る (tsukuru - "to make")
    "五段-カ行": "godan-ka",       # 書く (kaku - "to write")
    "五段-ガ行": "godan-ga",       # 泳ぐ (oyogu - "to swim")
    "五段-サ行": "godan-sa",       # 話す (hanasu - "to speak")
    "五段-タ行": "godan-ta",       # 立つ (tatsu - "to stand")
    "五段-ナ行": "godan-na",       # 死ぬ (shinu - "to die"), rare
    "五段-バ行": "godan-ba",       # 遊ぶ (asobu - "to play")
    "五段-マ行": "godan-ma",       # 読む (yomu - "to read")
    "五段-ワ行": "godan-wa",       # 買う (kau - "to buy")
    "五段-ワア行": "godan-waa",    # 言う (iu - "to say")
    "上一段-ア行": "i-ichidan-a",  # いる (iru - "to exist")
    "上一段-カ行": "i-ichidan-ka", # 起きる (okiru - "to wake up")
    "上一段-ガ行": "i-ichidan-ga", # 過ぎる (sugiru - "to pass")
    "上一段-ザ行": "i-ichidan-za", # 信じる (shinjiru - "to believe")
    "上一段-タ行": "i-ichidan-ta", # 落ちる (ochiru - "to fall")
    "上一段-ナ行": "i-ichidan-na", # 死ぬる (shinuru), archaic
    "上一段-ハ行": "i-ichidan-ha", # 干る (hiru - "to dry"), rare
    "上一段-バ行": "i-ichidan-ba", # 浴びる (abiru - "to bathe")
    "上一段-マ行": "i-ichidan-ma", # 見る (miru - "to see")
    "上一段-ラ行": "i-ichidan-ra", # 居る (iru - "to be"), archaic
    "下一段-ハ行": "e-ichidan-ha",  # へる (heru - "to decrease"), rare
    "下一段-ア行": "e-ichidan-a",  # える (eru - "to get"), rare
    "下一段-サ行": "e-ichidan-sa", # せる (seru - causative), rare
    "下一段-バ行": "e-ichidan-ba", # 食べる (taberu - "to eat")
    "下一段-カ行": "e-ichidan-ka", # 受ける (ukeru - "to receive")
    "下一段-ガ行": "e-ichidan-ga", # 上げる (ageru - "to raise")
    "下一段-ザ行": "e-ichidan-za", # 教える (oshieru - "to teach")
    "下一段-タ行": "e-ichidan-ta", # 捨てる (suteru - "to throw away")
    "下一段-ダ行": "e-ichidan-da", # 出る (deru - "to exit")
    "下一段-ナ行": "e-ichidan-na", # 寝る (neru - "to sleep")
    "下一段-マ行": "e-ichidan-ma", # 止める (yameru - "to stop")
    "下一段-ラ行": "e-ichidan-ra", # 入れる (ireru - "to put in")
    "文語下二段-ア行": "nidan-a",   # 得 (u - "to get"), classical
    "文語下二段-カ行": "nidan-ka",  # 受く (uku - "to receive"), classical
    "文語下二段-ガ行": "nidan-ga",  # 上ぐ (agu - "to raise"), classical
    "文語下二段-ザ行": "nidan-za",  # 教ふ (oshefu - "to teach"), classical
    "文語下二段-タ行": "nidan-ta",  # 捨つ (sutsu - "to throw away"), classical
    "文語下二段-ダ行": "nidan-da",  # 出づ (idezu - "to exit"), classical
    "文語下二段-ナ行": "nidan-na",  # 寝ぬ (nenu - "to sleep"), classical
    "文語下二段-バ行": "nidan-ba",  # 食ぶ (tabu - "to eat"), classical
    "文語下二段-マ行": "nidan-ma",  # 止む (yamu - "to stop"), classical
    "文語下二段-ヤ行": "nidan-ya",  # 焼ゆ (yaku - "to burn"), classical
    "文語下二段-ラ行": "nidan-ra",  # 入る (iru - "to enter"), classical
    "文語下二段-ワ行": "nidan-wa",  # 植う (uu - "to plant"), classical
    "文語上二段-カ行": "upper-nidan-ka", # 起く (oku - "to wake up"), classical
    "文語上二段-ガ行": "upper-nidan-ga", # 過ぐ (sugu - "to pass"), classical
    "文語上二段-タ行": "upper-nidan-ta", # classical upper ni-dan ta-row
    "文語上二段-ダ行": "upper-nidan-da", # classical upper ni-dan da-row
    "文語上二段-バ行": "upper-nidan-ba", # classical upper ni-dan ba-row
    "文語下二段-サ行": "lower-nidan-sa", # classical lower ni-dan sa-row
    "文語下二段-ハ行": "lower-nidan-ha", # classical lower ni-dan ha-row
    "文語助動詞-ザマス": "auxv-zamasu",  # ザマス (zamasu - colloquial polite auxiliary)
    "文語助動詞-ジ": "auxv-ji",  # じ (ji - classical auxiliary)
    "文語助動詞-ヌ": "auxv-nu-classical",  # ぬ (nu - classical auxiliary)
    "文語助動詞-ラシ": "auxv-rashi",  # らし (rashi - classical evidential)
    "文語助動詞-ラム": "auxv-ramu",  # らむ (ramu - classical presumptive/conjecture)
    "カ行変格": "ka-irregular",    # 来る (kuru - "to come")
    "サ行変格": "sa-irregular",    # する (suru - "to do")
    "文語サ行変格": "classical-sa-irregular",  # す (su - classical "to do"), classical
    "文語四段-カ行": "yodan-ka",  # 書く (kaku - "to write"), classical
    "文語四段-ガ行": "yodan-ga",  # 泳ぐ (oyogu - "to swim"), classical
    "文語四段-サ行": "yodan-sa",  # 話す (hanasu - "to speak"), classical
    "文語四段-タ行": "yodan-ta",  # 立つ (tatsu - "to stand"), classical
    "文語四段-ナ行": "yodan-na",  # 死ぬ (shinu - "to die"), classical
    "文語四段-バ行": "yodan-ba",  # 遊ぶ (asobu - "to play"), classical
    "文語四段-マ行": "yodan-ma",  # 読む (yomu - "to read"), classical
    "文語四段-ラ行": "yodan-ra",  # 作る (tsukuru - "to make"), classical
    "文語四段-ワ行": "yodan-wa",  # 買ふ (kafu - "to buy"), classical
    "文語四段-ハ行": "yodan-ha",  # 笑ふ (warafu - "to laugh"), classical
    "文語形容詞-ク": "classical-adjective-ku",  # 高く (takaku), classical
    "": ""
}

# Conjugation form mappings
CONJUGATED_FORM_MAP = {
    "仮定形-一般": "conditional",
    "仮定形-融合": "conditional-fused",
    "命令形": "imperative",
    "意志推量形": "volitional-presumptive",
    "未然形-サ": "imperfective-sa",
    "未然形-一般": "imperfective",
    "未然形-撥音便": "imperfective-nasal",
    "終止形-一般": "terminal",
    "終止形-撥音便": "terminal-nasal",
    "終止形-促音便": "terminal-geminate",
    "終止形-融合": "terminal-fused",
    "語幹-一般": "stem",
    "語幹-サ": "stem-sa",
    "連体形-一般": "attributive",
    "連体形-省略": "attributive-abbreviated",
    "連用形-イ音便": "conjunctive-i-sound",
    "連用形-ニ": "conjunctive-ni",
    "連用形-ト": "conjunctive-to",
    "連用形-一般": "conjunctive",
    "連用形-促音便": "conjunctive-geminate",
    "連用形-撥音便": "conjunctive-nasal",
    "連用形-省略": "conjunctive-abbreviated",
    "連用形-補助": "conjunctive-auxiliary",
    "連用形-融合": "conjunctive-fused",
    "未然形-セ": "imperfective-se",
    "連用形-ウ音便": "conjunctive-u-sound",
    "連体形-撥音便": "attributive-nasal",
    "已然形-一般": "realis",
    "已然形-補助": "realis-auxiliary",
    "連体形-補助": "attributive-auxiliary",
    "未然形-補助": "imperfective-auxiliary",
    "ク語法": "ku-form",  # ku-form classical grammar
    "終止形-ウ音便": "terminal-u-sound",  # terminal u-sound change
    "": ""
}

# Part-of-speech to character mappings
POS_TO_CHARS = {
    "prt": ['は', 'が', 'を', 'に', 'へ', 'と', 'で', 'か', 'の', 'ね', 'よ', 'て',
            'わ', 'も', 'ぜ', 'ん', 'な', 'ば', 'ぞ', 'し', 'さ', 'や', 'ら', 'ど',
            'い', 'つ', 'べ', 'け', 'ょ'],
    "sym": [],
    "auxs": ['。', '、', '・', '：', '；', '？', '！', '…', '「', '」', '『', '』',
             '{', '}', '.', 'ー', ':', '?', 'っ', '-', '々', '(', ')', '[', ']',
             '<', '>', '／', '＼', '＊', '＋', '＝', '＠', '＃', '％', '＆', '＊',
             'ぇ', '〇', '（', '）', '* ', '*', '～', '"', '◯'],
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
