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


# Source of truth for register label-to-ID mapping
# This must match the data/corpus.db register table and is used at inference time
# DO NOT use enumerate() as enum order does not guarantee these IDs
REGISTER_LABEL_TO_ID = {
    RegisterLevel.NEUTRAL: 0,
    RegisterLevel.SONKEIGO: 1,
    RegisterLevel.KENJOGO: 2,
    RegisterLevel.KANSAIBEN: 3,
    RegisterLevel.HAKATABEN: 4,
    RegisterLevel.KYOSHIGO: 5,
    RegisterLevel.NETSLANG: 6,
    RegisterLevel.OJOUSAMA: 7,
    RegisterLevel.GUNTAI: 8,
    RegisterLevel.JOSEIGO: 9,
    RegisterLevel.DANSEIGO: 10,
    RegisterLevel.BURIKKO: 11,
    RegisterLevel.TOHOKU: 12,
    RegisterLevel.BUSHI: 13,
}
REGISTER_ID_TO_LABEL = {v: k for k, v in REGISTER_LABEL_TO_ID.items()}


# Score-to-level classification thresholds
# These define how continuous scores [-1.0, 1.0] map to discrete levels


class FormalityThresholds:
    """Thresholds for classifying formality scores into discrete levels.

    Score range: -1.0 (very casual) to +1.0 (very formal)
    """

    VERY_FORMAL_MIN = 0.75  # >= 0.75 → VERY_FORMAL
    FORMAL_MIN = 0.25  # >= 0.25 → FORMAL
    NEUTRAL_MIN = -0.25  # >= -0.25 → NEUTRAL
    CASUAL_MIN = -0.75  # >= -0.75 → CASUAL
    # < -0.75 → VERY_CASUAL


class GenderThresholds:
    """Thresholds for classifying gender scores into discrete levels.

    Score range: -1.0 (masculine) to +1.0 (feminine)
    """

    MASCULINE_MAX = -0.5  # <= -0.5 → MASCULINE
    FEMININE_MIN = 0.5  # >= 0.5 → FEMININE
    # Between these → NEUTRAL


class PragmaticThresholds:
    """Thresholds for pragmatic vs unpragmatic classification."""

    PRAGMATIC_MIN = 0.5  # >= 0.5 probability → pragmatic


class GrammaticalityThresholds:
    """Thresholds for grammaticality classification."""

    GRAMMATIC_MIN = 0.5  # > 0.5 probability → grammatic
