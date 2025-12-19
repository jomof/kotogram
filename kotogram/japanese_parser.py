"""Abstract base class for Japanese text parsing with shared mapping constants."""

from abc import ABC, abstractmethod

# Global mapping constants shared across all Japanese parser implementations

# Part-of-speech mappings
POS_MAP = {
    "名詞": "noun",  # Noun
    "動詞": "verb",  # Verb
    "形容詞": "adjective",  # Adjective
    "副詞": "adverb",  # Adverb
    "助詞": "particle",  # Particle
    "接続詞": "conjunction",  # Conjunction
    "感動詞": "interjection",  # Interjection
    "空白": "whitespace",  # Whitespace
    "記号": "symbol",  # Symbol
    "助動詞": "auxiliary-verb",  # Auxiliary verb
    "補助記号": "auxiliary-symbol",  # Auxiliary symbol
    "代名詞": "pronoun",  # Pronoun
    "接頭辞": "prefix",  # Prefix
    "接尾辞": "suffix",  # Suffix
    "形状詞": "shape-word",  # Shape word
    "連体詞": "attributive",  # Attributive
}

# Part-of-speech detail level 1 mappings
POS1_MAP = {
    "括弧開": 'bracket-open',  # Opening bracket
    "括弧閉": 'bracket-close',  # Closing bracket
    "読点": 'comma',  # Comma
    "固有名詞": 'proper-noun',  # Proper noun
    "格助詞": 'case-particle',  # Case particle
    "普通名詞": 'common-noun',  # Common noun
    "準体助詞": 'pre-noun-particle',  # Pre-noun particle
    "終助詞": 'sentence-final-particle',  # Sentence-final particle
    "句点": 'period',  # Period
    "係助詞": 'binding-particle',  # Binding particle
    "非自立可能": 'non-self-reliant',  # Non-self-reliant
    "一般": 'general',  # General
    "助動詞語幹": 'auxiliary-verb-stem',  # Auxiliary verb stem
    "形容詞的": 'adjectival',  # Adjectival
    "副助詞": 'adverbial-particle',  # Adverbial particle
    "接続助詞": 'conjunctive-particle',  # Conjunctive particle
    "数詞": 'numeral',  # Numeral
    "名詞的": 'noun-like',  # Noun-like
    "フィラー": 'filler',  # Filler
    "形状詞的": 'shape-word-like',  # Shape word-like
    "タリ": 'tari',  # tari (a form of auxiliary verb)
    "動詞的": 'verb-like',  # Verb-like
    "文字": 'character',  # Character/letter (e.g., Greek letters like α, β, γ)
    "ＡＡ": 'ascii-art',  # ASCII art / emoticon
    "*": '',  # Unspecified/empty field marker
}

# Part-of-speech detail level 2 mappings
POS2_MAP = {
    "副詞可能": "adverb-possible",
    "一般": "general",
    "サ変可能": "suru-possible",
    "地名": "place-name",
    "形状詞可能": "pos2-unk1",
    "助数詞可能": "counter-possible",
    "サ変形状詞可能": "pos2-unk2",
    "人名": "person-name",
    "助数詞": "counter",
    "顔文字": "kaomoji",  # Emoticon/kaomoji
    "*": '',  # Unspecified/empty field marker
}

# Part-of-speech detail level 3 mappings
POS3_MAP = {
    "国": "country",
    "姓": "surname",
    "名": "given-name",
    "一般": "general",
    "*": "",  # Unspecified/empty field marker
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
    "文語助動詞-リ": "classical-auxv-ri",  # り (ri - classical perfective), classical
    "文語助動詞-ベシ": "classical-auxv-beshi",  # べし (beshi - "should/ought to"), classical
    "文語助動詞-ゴトシ": "classical-auxv-gotoshi",  # ごとし (gotoshi - "like/as if"), classical
    "文語助動詞-ズ": "classical-auxv-zu",  # ず (zu - classical negative auxiliary), classical
    "文語助動詞-キ": "classical-auxv-ki",  # き (ki - classical past tense), classical
    "文語助動詞-ケリ": "classical-auxv-keri",  # けり (keri - classical perfect/recollective), classical
    "文語助動詞-タリ-完了": "classical-auxv-tari-perfective",  # たり (tari - classical perfective), classical
    "文語助動詞-ナリ-断定": "classical-auxv-nari-assertive",  # なり (nari - classical assertive copula), classical
    "文語助動詞-マジ": "classical-auxv-maji",  # まじ (maji - classical negative presumptive), classical
    "文語助動詞-ム": "classical-auxv-mu",  # む (mu - classical presumptive/volitional), classical
    "文語形容詞-シク": "classical-adj-shiku",  # しく (shiku-inflection classical adjective)
    "文語ラ行変格": "classical-irregular-ra",  # Classical ra-row irregular verbs
    "助動詞-マイ": "auxv-mai",  # まい (mai - "probably won't/shouldn't")
    "助動詞-ジャ": "auxv-ja",  # じゃ (ja - contracted copula "is/am/are")
    "助動詞-ヤ": "auxv-ya",  # や (ya - classical question particle/auxiliary)
    "助動詞-ナンダ": "auxv-nanda",  # なんだ (nanda - colloquial past tense of だ)
    "助動詞-ヘン": "auxv-hen",  # へん (hen - Kansai dialect negative)
    "文語助動詞-タリ-断定": "classical-auxv-tari",  # たり (tari - classical assertive), classical
    "形容詞": "adjective",          # 高い (takai - "tall/expensive")
    "五段-ラ行": "godan-ra",       # 作る (tsukuru - "to make")
    "五段-カ行": "godan-ka",       # 書く (kaku - "to write")
    "五段-ガ行": "godan-ga",       # 泳ぐ (oyogu - "to swim")
    "五段-サ行": "godan-sa",       # 話す (hanasu - "to speak")
    "五段-タ行": "godan-ta",       # 立つ (tatsu - "to stand")
    "五段-ナ行": "godan-na",       # 死ぬ (shinu - "to die"), rare
    "五段-バ行": "godan-ba",       # 遊ぶ (asobu - "to play")
    "五段-マ行": "godan-ma",       # 読む (yomu - "to read")
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
    "文語下二段-ア行": "classical-nidan-a",   # 得 (u - "to get"), classical
    "文語下二段-カ行": "classical-nidan-ka",  # 受く (uku - "to receive"), classical
    "文語下二段-ガ行": "classical-nidan-ga",  # 上ぐ (agu - "to raise"), classical
    "文語下二段-タ行": "classical-nidan-ta",  # 捨つ (sutsu - "to throw away"), classical
    "文語下二段-ダ行": "classical-nidan-da",  # 出づ (idezu - "to exit"), classical
    "文語下二段-ナ行": "classical-nidan-na",  # 寝ぬ (nenu - "to sleep"), classical
    "文語下二段-マ行": "classical-nidan-ma",  # 止む (yamu - "to stop"), classical
    "文語下二段-ラ行": "classical-nidan-ra",  # 入る (iru - "to enter"), classical
    "文語上二段-タ行": "classical-upper-nidan-ta", # classical upper ni-dan ta-row
    "文語上二段-ダ行": "classical-upper-nidan-da", # classical upper ni-dan da-row
    "文語上二段-バ行": "classical-upper-nidan-ba", # classical upper ni-dan ba-row
    "文語下二段-サ行": "classical-lower-nidan-sa", # classical lower ni-dan sa-row
    "文語下二段-ハ行": "classical-lower-nidan-ha", # classical lower ni-dan ha-row
    "文語助動詞-ザマス": "classical-auxv-zamasu",  # ザマス (zamasu - colloquial polite auxiliary)
    "文語助動詞-ジ": "classical-auxv-ji",  # じ (ji - classical auxiliary)
    "文語助動詞-ヌ": "classical-auxv-nu-classical",  # ぬ (nu - classical auxiliary)
    "文語助動詞-ラシ": "classical-auxv-rashi",  # らし (rashi - classical evidential)
    "文語助動詞-ラム": "classical-auxv-ramu",  # らむ (ramu - classical presumptive/conjecture)
    "カ行変格": "ka-irregular",    # 来る (kuru - "to come")
    "サ行変格": "sa-irregular",    # する (suru - "to do")
    "文語サ行変格": "classical-sa-irregular",  # す (su - classical "to do"), classical
    "文語四段-カ行": "classical-yodan-ka",  # 書く (kaku - "to write"), classical
    "文語四段-サ行": "classical-yodan-sa",  # 話す (hanasu - "to speak"), classical
    "文語四段-タ行": "classical-yodan-ta",  # 立つ (tatsu - "to stand"), classical
    "文語四段-ラ行": "classical-yodan-ra",  # 作る (tsukuru - "to make"), classical
    "文語四段-ハ行": "classical-yodan-ha",  # 笑ふ (warafu - "to laugh"), classical
    "文語四段-マ行": "classical-yodan-ma",  # 止む (yamu - "to stop"), classical
    "文語形容詞-ク": "classical-adjective-ku",  # 高く (takaku), classical
    "助動詞-ドス": "auxv-dosu",  # どす (dosu - Kansai polite)
    "文語上二段-ハ行": "classical-upper-nidan-ha", 
    "無変化型": "invariant",
    "*": "",
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
    "連体形-補助": "attributive-auxiliary",
    "未然形-補助": "imperfective-auxiliary",
    "ク語法": "ku-form",  # ku-form classical grammar
    "終止形-ウ音便": "terminal-u-sound",  # terminal u-sound change
    "連体形-一般+送り仮名省略": "attributive-okurigana-omitted",
    "連用形-一般+送り仮名省略": "conjunctive-okurigana-omitted",
    "*": "",
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
