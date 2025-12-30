"""Masking strategies for training data sanitization.

This module provides utilities to mask specific tokens in a kotogram stream,
primarily for anonymization or data augmentation purposes.
"""

from typing import List

from kotogram.kotogram import Token


def apply_training_mask(tokens: List[Token]) -> None:
    """Apply training mask to anonymize given names in place.

    Replaces Japanese given names (First Names) with the placeholder "太郎" (Taro).
    This operation is performed in-place on the provided token list.

    Args:
        tokens: List of kotogram.Token objects to process.
    """
    for i, token in enumerate(tokens):
        features = token.features
        pos = features.get("pos")
        detail1 = features.get("pos_detail_1")
        detail2 = features.get("pos_detail_2")
        detail3 = features.get("pos_detail_3")

        # Base Proper Noun Check
        if pos == "noun" and detail1 == "proper-noun":
            target_surface = "<proper-noun>"

            # Hierarchy: Person Name
            if detail2 == "person-name":
                target_surface = "<person-name>"
                if detail3 == "given-name":
                    target_surface = "<given-name>"
                elif detail3 == "surname":
                    target_surface = "<surname>"

            # Hierarchy: Place Name
            elif detail2 == "place-name":
                target_surface = "<place-name>"
                if detail3 == "country":
                    target_surface = "<country>"

            # Strict Assertions for specific types claimed in detail3
            if detail3 == "given-name" and target_surface != "<given-name>":
                raise RuntimeError(
                    f"Token has pos_detail_3='given-name' but failed hierarchy check. Features: {features}"
                )
            if detail3 == "surname" and target_surface != "<surname>":
                raise RuntimeError(
                    f"Token has pos_detail_3='surname' but failed hierarchy check. Features: {features}"
                )
            if detail3 == "country" and target_surface != "<country>":
                raise RuntimeError(
                    f"Token has pos_detail_3='country' but failed hierarchy check. Features: {features}"
                )

            # Apply Replacement
            # Create a NEW token with stripped features
            # Retain only POS information for grammatical stability
            new_features = {
                "pos": pos,
                "pos_detail_1": detail1,
                "pos_detail_2": detail2,
                "pos_detail_3": detail3,
                "lemma": "*",
            }
            # Replace token in list
            tokens[i] = Token(target_surface, features=new_features)
