from enum import Enum

class FormalityLevel(Enum):
    """Overall formality level of a Japanese sentence."""

    VERY_FORMAL = "very_formal"           # Keigo, honorific language (敬語)
    FORMAL = "formal"                     # Polite/formal (-ます/-です forms)
    NEUTRAL = "neutral"                   # Plain/dictionary form, balanced
    CASUAL = "casual"                     # Colloquial, informal contractions
    VERY_CASUAL = "very_casual"          # Highly casual, slang
    UNPRAGMATIC_FORMALITY = "unpragmatic_formality"  # Mixed/awkward formality


class GenderLevel(Enum):
    """Gender association level of a Japanese sentence."""

    MASCULINE = "masculine"               # Male-associated speech (俺, ぜ, ぞ, etc.)
    FEMININE = "feminine"                 # Female-associated speech (わ, の, あたし, etc.)
    NEUTRAL = "neutral"                   # Gender-neutral speech
    UNPRAGMATIC_GENDER = "unpragmatic_gender"  # Mixed/awkward gender markers


class RegisterLevel(Enum):
    """Specific register/dialect classifications."""

    SONKEIGO = "sonkeigo"                 # Honorific (respectful)
    KENJOGO = "kenjogo"                   # Humble
    KANSAIBEN = "kansaiben"               # Kansai dialect
    HAKATABEN = "hakataben"               # Hakata dialect
    KYOSHIGO = "kyoshigo"                 # Teacher style
    NETSLANG = "netslang"                 # Internet slang
    OJOUSAMA = "ojousama"                 # Refined lady style
    GUNTAI = "guntai"                     # Military style
    JOSEIGO = "joseigo"                   # Feminine register
    DANSEIGO = "danseigo"                 # Masculine register
    BURIKKO = "burikko"                   # Burikko (exaggerated cuteness)
    NEUTRAL = "neutral"                   # Standard/Neutral
    TOHOKU = "tohoku"                     # Tohoku dialect
    BUSHI = "bushi"                       # Samurai/Archaic register
