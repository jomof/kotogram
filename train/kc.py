"""KC target computation logic."""

from typing import Any, Dict, List

# KC Configuration
KC_HASH_BUCKETS = 16384
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5


def compute_kc_targets(feature_ids: Dict[str, List[int]]) -> Dict[str, Any]:
    """Compute KC targets from feature IDs."""
    targets: Dict[str, Any] = {}

    # 1. Token bags (Multi-hot)
    # Target = unique token IDs appearing in the sentence
    for field in ["lemma", "pos", "conjugated_form"]:
        if field in feature_ids:
            targets[f"bag_{field}"] = list(set(feature_ids[field]))

    # 2. Position-biased token bags
    # Target = unique token IDs appearing in the last N tokens
    for field in ["surface", "lemma", "pos", "conjugated_form"]:
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            targets[f"tail_{field}"] = list(set(tail_ids))

    # 3. N-gram hash targets
    # Target = hashed IDs for bigrams/trigrams
    for field in ["pos", "conjugated_form"]:
        if field in feature_ids:
            ids = feature_ids[field]
            hashes = set()
            # Unigrams, Bigrams, Trigrams
            # "Token bags" (Priority 1A) covers unigrams basically.
            # So let's stick to n=2..Order.
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = tuple(ids[i : i + n_val])
                        # Simple hash: polynomial rolling hash or python hash
                        # Python hash is randomized per process, strictly we might want stable
                        # but "Stable KC IDs across runs" is a non-goal.
                        # Start with python hash for simplicity and speed.
                        h = hash(ngram) % KC_HASH_BUCKETS
                        hashes.add(h)
            targets[f"ngram_{field}"] = list(hashes)

    # 3b. (pos, conjugated_form) pairs
    if "pos" in feature_ids and "conjugated_form" in feature_ids:
        p_ids = feature_ids["pos"]
        c_ids = feature_ids["conjugated_form"]
        if len(p_ids) == len(c_ids):
            pair_hashes = set()
            for i, p_id in enumerate(p_ids):
                pair = (p_id, c_ids[i])
                h = hash(pair) % KC_HASH_BUCKETS
                pair_hashes.add(h)
            targets["pair_pos_conj"] = list(pair_hashes)

    return targets
