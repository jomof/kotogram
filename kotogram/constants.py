from enum import Enum


class FormalityLevel(Enum):
    """Overall formality level of a Japanese sentence."""

    VERY_FORMAL = "very_formal"  # Keigo, honorific language (敬語)
    FORMAL = "formal"  # Polite/formal (-ます/-です forms)
    NEUTRAL = "neutral"  # Plain/dictionary form, balanced
    CASUAL = "casual"  # Colloquial, informal contractions
    VERY_CASUAL = "very_casual"  # Highly casual, slang
    UNPRAGMATIC_FORMALITY = "unpragmatic_formality"  # Mixed/awkward formality


class GenderLevel(Enum):
    """Gender association level of a Japanese sentence."""

    MASCULINE = "masculine"  # Male-associated speech (俺, ぜ, ぞ, etc.)
    FEMININE = "feminine"  # Female-associated speech (わ, の, あたし, etc.)
    NEUTRAL = "neutral"  # Gender-neutral speech
    UNPRAGMATIC_GENDER = "unpragmatic_gender"


FORMALITY_LABEL_TO_ID = {
    FormalityLevel.VERY_FORMAL: 0,
    FormalityLevel.FORMAL: 1,
    FormalityLevel.NEUTRAL: 2,
    FormalityLevel.CASUAL: 3,
    FormalityLevel.VERY_CASUAL: 4,
    FormalityLevel.UNPRAGMATIC_FORMALITY: 5,
}
FORMALITY_ID_TO_LABEL = {v: k for k, v in FORMALITY_LABEL_TO_ID.items()}

GENDER_LABEL_TO_ID = {
    GenderLevel.MASCULINE: 0,
    GenderLevel.FEMININE: 1,
    GenderLevel.NEUTRAL: 2,
    GenderLevel.UNPRAGMATIC_GENDER: 3,
}
GENDER_ID_TO_LABEL = {v: k for k, v in GENDER_LABEL_TO_ID.items()}


class RegisterLevel(Enum):
    """Specific register/dialect classifications."""

    SONKEIGO = "sonkeigo"  # Honorific (respectful)
    KENJOGO = "kenjogo"  # Humble
    KANSAIBEN = "kansaiben"  # Kansai dialect
    HAKATABEN = "hakataben"  # Hakata dialect
    KYOSHIGO = "kyoshigo"  # Teacher style
    NETSLANG = "netslang"  # Internet slang
    OJOUSAMA = "ojousama"  # Refined lady style
    GUNTAI = "guntai"  # Military style
    JOSEIGO = "joseigo"  # Feminine register
    DANSEIGO = "danseigo"  # Masculine register
    BURIKKO = "burikko"  # Burikko (exaggerated cuteness)
    NEUTRAL = "neutral"  # Standard Japanese
    TOHOKU = "tohoku"  # Tohoku dialect
    BUSHI = "bushi"  # Samurai/Archaic register


REGISTER_LABEL_TO_ID = {v: i for i, v in enumerate(RegisterLevel)}
REGISTER_ID_TO_LABEL = {v: k for k, v in REGISTER_LABEL_TO_ID.items()}
