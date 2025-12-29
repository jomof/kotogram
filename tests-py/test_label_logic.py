from kotogram.analysis import GenderLevel, RegisterLevel
from scripts.rule_based_analysis import infer_gender_from_register


def test_explicit_gender():
    # If gender is already known, it should rely on that
    assert infer_gender_from_register(GenderLevel.MASCULINE, []) == (-1.0, 1)
    assert infer_gender_from_register(GenderLevel.FEMININE, []) == (1.0, 1)
    # Register shouldn't override explicit gender
    assert infer_gender_from_register(
        GenderLevel.MASCULINE, [RegisterLevel.OJOUSAMA]
    ) == (-1.0, 1)


def test_neutral_inference():
    # Masculine registers
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.DANSEIGO]
    ) == (-1.0, 1)
    assert infer_gender_from_register(GenderLevel.NEUTRAL, [RegisterLevel.GUNTAI]) == (
        -1.0,
        1,
    )
    assert infer_gender_from_register(GenderLevel.NEUTRAL, [RegisterLevel.BUSHI]) == (
        -1.0,
        1,
    )

    # Feminine registers
    assert infer_gender_from_register(GenderLevel.NEUTRAL, [RegisterLevel.JOSEIGO]) == (
        1.0,
        1,
    )
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.OJOUSAMA]
    ) == (1.0, 1)
    assert infer_gender_from_register(GenderLevel.NEUTRAL, [RegisterLevel.BURIKKO]) == (
        1.0,
        1,
    )


def test_kyoshigo_exclusion():
    # KYOSHIGO used to be masculine, should now be neutral (unless other markers exist)
    # Since it has no gender valence, a pure Kyoshigo sentence is gender-unpragmatic.
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.KYOSHIGO]
    ) == (0.0, 1)


def test_conflict_unpragmatic():
    # Masculine + Feminine register -> Unpragmatic
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.DANSEIGO, RegisterLevel.JOSEIGO]
    ) == (0.0, 0)
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.GUNTAI, RegisterLevel.OJOUSAMA]
    ) == (0.0, 0)

    # KYOSHIGO (Neutral) + Feminine -> Feminine (Valid combination, not a conflict)
    # Teacher style + Feminine speech is a perfectly valid "Female Teacher" style
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.KYOSHIGO, RegisterLevel.JOSEIGO]
    ) == (1.0, 1)


def test_neutral_default():
    # Default case (no markers) is now PRAGMATIC (1) as we want to train on neutral sentences.
    assert infer_gender_from_register(GenderLevel.NEUTRAL, []) == (0.0, 1)
    # KANSAIBEN is a region, not gendered, but still valid neutral/dialect training data?
    # User said "No registers should be excluded".
    assert infer_gender_from_register(
        GenderLevel.NEUTRAL, [RegisterLevel.KANSAIBEN]
    ) == (0.0, 1)
