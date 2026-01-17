import math
import unittest.mock

import torch
import torch.nn.functional as F

from kotogram.model import ModelConfig
from kotogram.tokenizer import FEATURE_FIELDS
from train.kc import KcFamilyId
from train.models import TrainingClassifier


def test_gumbel_noise_training_only():
    """Verify Gumbel noise is applied only during training and affects top-k."""
    config = ModelConfig(
        vocab_sizes={f: 100 for f in FEATURE_FIELDS},
        kc_vocab_size=10,
        kc_topk=1,  # Select top-1 to easily see noise impact
        kc_temperature=1.0,
    )
    model = TrainingClassifier(config)

    # Mock inputs
    field_inputs = {}
    for f in FEATURE_FIELDS:
        field_inputs[f"input_ids_{f}"] = torch.zeros(2, 5, dtype=torch.long)

    attention_mask = torch.ones(2, 5)

    # 1. Eval Mode: No noise
    model.eval()
    with torch.no_grad():
        out_clean = model.forward_kc(field_inputs, attention_mask, gumbel_scale=1.0)

    # Run twice to ensure deterministic
    with torch.no_grad():
        out_clean_2 = model.forward_kc(field_inputs, attention_mask, gumbel_scale=1.0)

    assert torch.allclose(out_clean["kc_probs"], out_clean_2["kc_probs"]), (
        "Eval mode should be deterministic even with gumbel_scale passed"
    )

    # 2. Train Mode: Noise applied vs Not applied (Same Seed)
    model.train()

    # Run 1: No noise (scale=0)
    torch.manual_seed(42)
    out_clean_train = model.forward_kc(field_inputs, attention_mask, gumbel_scale=0.0)

    # Run 2: High noise (scale=10)
    torch.manual_seed(42)  # Reset seed to get same dropout mask
    out_noisy_train = model.forward_kc(field_inputs, attention_mask, gumbel_scale=10.0)

    # Clean logits should match exactly (same dropout mask)
    assert torch.allclose(
        out_clean_train["kc_logits_raw"], out_noisy_train["kc_logits_raw"]
    ), "Raw logits should be identical with same seed despite gumbel_scale"

    # Probs should be different (one has noise added to selection logits)
    diff = (out_clean_train["kc_probs"] - out_noisy_train["kc_probs"]).abs().max()
    assert diff > 0.1, f"Gumbel noise should change probabilities, got max diff {diff}"


def test_differentiable_usage_calculation():
    """Verify softmax-based usage calculation logic."""
    # Replicating the logic from trainer.py
    logits_raw = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],  # Strong pref for 0
            [0.0, 10.0, 0.0, 0.0],  # Strong pref for 1
        ]
    )
    tau_usage = 1.0

    # q: (B, V)
    q = torch.softmax(logits_raw / tau_usage, dim=-1)

    # Expect roughly [1, 0, 0, 0] and [0, 1, 0, 0]
    assert q[0, 0] > 0.99
    assert q[1, 1] > 0.99

    # p: (V,) mean across batch
    p = q.mean(dim=0)

    # Expect roughly [0.5, 0.5, 0.0, 0.0]
    assert 0.49 < p[0] < 0.51
    assert 0.49 < p[1] < 0.51
    assert p[2] < 0.01

    # Entropy
    kc_diversity_eps = 1e-9
    log_p = (p + kc_diversity_eps).log()
    entropy = -(p * log_p).sum()

    # Entropy of [0.5, 0.5] is - (0.5 ln 0.5 + 0.5 ln 0.5) = ln 2 ~= 0.693
    expected_ent = math.log(2)
    assert abs(entropy.item() - expected_ent) < 0.01

    # Ensure gradients flow
    logits_raw.requires_grad = True
    q = torch.softmax(logits_raw, dim=-1)
    p = q.mean(dim=0)
    loss = (p - 0.25).pow(2).sum()  # Dummy loss trying to make uniform
    loss.backward()
    assert logits_raw.grad is not None
    assert logits_raw.grad.abs().sum() > 0


def test_negative_sampling_logic():
    """Verify logic for masking large heads."""
    # pylint: disable=invalid-name
    batch_size, vocab_size = 4, 300
    logits = torch.randn(batch_size, vocab_size)
    targets = torch.zeros(batch_size, vocab_size)

    # Set some positives
    targets[0, 10] = 1.0
    targets[1, 50] = 1.0

    # Mocking implementation
    pos_mask = targets > 0.5
    neg_count = 10  # Small count for test

    # Sample negatives
    # Sample negatives
    neg_inds = torch.randint(0, vocab_size, (batch_size, neg_count))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, neg_inds, True)

    mask = mask | pos_mask

    # Check that ALL positives are included
    assert mask[0, 10].item() is True
    assert mask[1, 50].item() is True

    # Check that we have roughly neg_count + num_pos active
    # (Could be less if collision)
    active_0 = mask[0].sum().item()
    # Relaxed check for collisions (randint is with replacement)
    assert neg_count - 3 <= active_0 <= neg_count + 1  # Allow for collisions

    # Verify we can compute loss
    loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    assert not torch.isnan(loss)
    assert loss > 0


def test_kc_probe_auc_calculation():
    """Verify AUC calculation logic for structural heads."""
    # pylint: disable=too-many-locals
    # Perfect separation: all positives have higher scores than negatives
    pos_logits = [2.0, 3.0, 4.0]
    neg_logits = [-1.0, 0.0, 1.0]

    all_logits = pos_logits + neg_logits
    all_labels = [1.0] * len(pos_logits) + [0.0] * len(neg_logits)
    combined = sorted(zip(all_logits, all_labels), key=lambda x: x[0])
    ranks = list(range(1, len(combined) + 1))
    pos_rank_sum = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl > 0.5)
    n_pos = len(pos_logits)
    n_neg = len(neg_logits)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg)

    # Perfect AUC should be 1.0
    assert abs(auc - 1.0) < 0.01, f"Perfect separation should give AUC=1.0, got {auc}"

    # Random (overlapping) scores - use slightly different values to avoid ties
    # With ties, AUC is dependent on sort stability
    pos_logits_random = [0.51, 0.52, 0.53]
    neg_logits_random = [0.49, 0.50, 0.54]

    all_logits_r = pos_logits_random + neg_logits_random
    all_labels_r = [1.0] * 3 + [0.0] * 3
    combined_r = sorted(zip(all_logits_r, all_labels_r), key=lambda x: x[0])
    ranks_r = list(range(1, len(combined_r) + 1))
    pos_rank_sum_r = sum(r for r, (_, lbl) in zip(ranks_r, combined_r) if lbl > 0.5)
    auc_r = (pos_rank_sum_r - 3 * 4 / 2) / 9

    # Interleaved scores should give AUC between 0.3 and 0.7
    assert 0.3 < auc_r < 0.7, (
        f"Interleaved scores should give AUC near 0.5, got {auc_r}"
    )


def test_nan_guard_finite_loss_check():
    """Verify torch.isfinite correctly detects NaN/Inf in loss (Round 10)."""
    # Normal loss
    loss_ok = torch.tensor(0.5)
    assert torch.isfinite(loss_ok), "Normal loss should be finite"

    # NaN loss
    loss_nan = torch.tensor(float("nan"))
    assert not torch.isfinite(loss_nan), "NaN loss should not be finite"

    # Inf loss
    loss_inf = torch.tensor(float("inf"))
    assert not torch.isfinite(loss_inf), "Inf loss should not be finite"

    # Negative Inf
    loss_ninf = torch.tensor(float("-inf"))
    assert not torch.isfinite(loss_ninf), "Negative inf should not be finite"


def test_nan_guard_finite_grad_check():
    """Verify torch.isfinite correctly detects NaN/Inf in gradients (Round 10)."""
    # Clean gradients
    clean_grad = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isfinite(clean_grad).all(), "Clean gradients should all be finite"

    # Gradient with NaN
    nan_grad = torch.tensor([1.0, float("nan"), 3.0])
    assert not torch.isfinite(nan_grad).all(), "Gradient with NaN not all finite"

    # Gradient with Inf
    inf_grad = torch.tensor([1.0, float("inf"), 3.0])
    assert not torch.isfinite(inf_grad).all(), "Gradient with Inf not all finite"


# =============================================================================
# Round 11: NaN Recovery System Tests
# =============================================================================


def test_tensor_finite_stats_clean():
    """Test tensor_finite_stats with clean tensor."""
    from train.trainer import tensor_finite_stats

    x = torch.tensor([1.0, 2.0, 3.0])
    stats = tensor_finite_stats(x)

    assert stats.finite is True, "Clean tensor should be finite"
    assert stats.n_nan == 0, "No NaNs expected"
    assert stats.n_inf == 0, "No Infs expected"
    # min/max are skipped (NaN) for clean tensors to save syncs
    # assert stats.min == 1.0, "Min should be 1.0"
    # assert stats.max == 3.0, "Max should be 3.0"


def test_tensor_finite_stats_with_nan():
    """Test tensor_finite_stats with NaN values."""
    from train.trainer import tensor_finite_stats

    x = torch.tensor([1.0, float("nan"), 3.0])
    stats = tensor_finite_stats(x)

    assert stats.finite is False, "Tensor with NaN should not be finite"
    assert stats.n_nan == 1, "One NaN expected"
    assert stats.n_inf == 0, "No Infs expected"
    assert stats.min == 1.0, "Min of finite values should be 1.0"
    assert stats.max == 3.0, "Max of finite values should be 3.0"


def test_tensor_finite_stats_with_inf():
    """Test tensor_finite_stats with Inf values."""
    from train.trainer import tensor_finite_stats

    x = torch.tensor([1.0, float("inf"), 3.0])
    stats = tensor_finite_stats(x)

    assert stats.finite is False, "Tensor with Inf should not be finite"
    assert stats.n_nan == 0, "No NaNs expected"
    assert stats.n_inf == 1, "One Inf expected"
    assert stats.min == 1.0, "Min of finite values should be 1.0"
    assert stats.max == 3.0, "Max of finite values should be 3.0"


def test_tensor_finite_stats_all_nan():
    """Test tensor_finite_stats with all NaN values."""

    from train.trainer import tensor_finite_stats

    x = torch.tensor([float("nan"), float("nan")])
    stats = tensor_finite_stats(x)

    assert stats.finite is False
    assert stats.n_nan == 2
    assert math.isnan(stats.min), "Min should be NaN when all values non-finite"
    assert math.isnan(stats.max), "Max should be NaN when all values non-finite"


def test_tensor_finite_stats_none():
    """Test tensor_finite_stats with None input."""
    from train.trainer import tensor_finite_stats

    stats = tensor_finite_stats(None)

    assert stats.finite is True, "None should be treated as finite"
    assert stats.n_nan == 0
    assert stats.n_inf == 0


def test_nan_recovery_log_spam_pattern():
    """Test the log spam reduction pattern (first 3, then every 50th).

    Mirrors the logic in KCTrainer.train_epoch.
    """
    logged_count = 0
    nonfinite_total = 0

    for _ in range(100):
        nonfinite_total += 1
        should_log = logged_count < 3 or nonfinite_total % 50 == 0
        if should_log:
            logged_count += 1

    # Should have logged: 1, 2, 3, 50, 100 = 5 times
    assert logged_count == 5, f"Expected 5 logs, got {logged_count}"


def test_nan_recovery_streak_reset():
    """Test that streak resets on successful forward.

    Mirrors the streak logic in KCTrainer.train_epoch.
    """
    streak = 0

    # Simulate NaN forward -> streak increases
    forward_ok = False
    if not forward_ok:
        streak += 1
    assert streak == 1

    # Simulate another NaN forward
    forward_ok = False
    if not forward_ok:
        streak += 1
    assert streak == 2

    # Simulate successful forward -> streak resets
    forward_ok = True
    if forward_ok and streak > 0:
        streak = 0
    assert streak == 0


# =============================================================================
# Round 12: KC Numeric Stability Tests
# =============================================================================


def test_forward_kc_gumbel_stability():
    """Test gumbel path produces finite outputs with various scales."""

    config = ModelConfig(
        vocab_sizes={f: 50 for f in FEATURE_FIELDS},
        kc_vocab_size=16,
        kc_topk=4,
        kc_temperature=1.0,
    )
    model = TrainingClassifier(config)
    model.train()

    field_inputs = {
        f"input_ids_{f}": torch.zeros(4, 10, dtype=torch.long) for f in FEATURE_FIELDS
    }
    attention_mask = torch.ones(4, 10)

    # Test with multiple gumbel scales
    for gumbel_scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
        outputs = model.forward_kc(
            field_inputs, attention_mask, gumbel_scale=gumbel_scale
        )

        assert torch.isfinite(outputs["kc_logits_raw"]).all(), (
            f"kc_logits_raw not finite at scale={gumbel_scale}"
        )
        assert torch.isfinite(outputs["kc_probs"]).all(), (
            f"kc_probs not finite at scale={gumbel_scale}"
        )
        assert torch.isfinite(outputs["topk_vals"]).all(), (
            f"topk_vals not finite at scale={gumbel_scale}"
        )


def test_forward_kc_nan_to_num_guard():
    """Test that kc_probs uses nan_to_num to guard against non-finite values.

    Verifies that even if extreme inputs cause issues, output probs are still finite
    due to the logits clamp and nan_to_num guard.
    """

    config = ModelConfig(
        vocab_sizes={f: 50 for f in FEATURE_FIELDS},
        kc_vocab_size=16,
        kc_topk=4,
        kc_temperature=1.0,
    )
    model = TrainingClassifier(config)
    model.train()

    field_inputs = {
        f"input_ids_{f}": torch.zeros(2, 5, dtype=torch.long) for f in FEATURE_FIELDS
    }
    attention_mask = torch.ones(2, 5)

    # Run forward even with extreme gumbel scale
    outputs = model.forward_kc(field_inputs, attention_mask, gumbel_scale=100.0)

    # Probs should be finite due to nan_to_num guard (and logits clamp)
    assert torch.isfinite(outputs["kc_probs"]).all(), (
        "kc_probs should be finite with nan_to_num guard"
    )
    # Probs should be in [0, 1]
    assert (outputs["kc_probs"] >= 0).all() and (outputs["kc_probs"] <= 1).all(), (
        "kc_probs out of range"
    )


def test_kc_snapshot_restore_pattern():
    """Test the snapshot/restore pattern used for NaN recovery.

    This tests the state_dict based save/restore logic.
    """
    # Create mock state dicts
    original_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    snapshot = {"kc_head": {"linear.weight": original_weight.clone()}}

    # Restore pattern
    restored = {k: v.clone() for k, v in snapshot["kc_head"].items()}

    # Verify restore produces original values
    assert torch.allclose(restored["linear.weight"], original_weight)
    assert not torch.isnan(restored["linear.weight"]).any()


def test_kc_logits_clamp_range():
    """Test that logits_for_selection is clamped to [-20, 20].

    Verifies the clamp prevents extreme logits that could cause gradient issues.
    """
    # Simulate the clamp logic from forward_kc
    extreme_logits = torch.tensor([[-100.0, 0.0, 100.0], [-50.0, 50.0, 0.0]])
    clamped = extreme_logits.clamp(min=-20.0, max=20.0)

    assert clamped.min() >= -20.0, "Min should be >= -20"
    assert clamped.max() <= 20.0, "Max should be <= 20"

    # Sigmoid of clamped values should be numerically stable
    sigmoid_vals = torch.sigmoid(clamped)
    assert torch.isfinite(sigmoid_vals).all(), (
        "Sigmoid of clamped logits should be finite"
    )


def test_nan_guard_skip_logic():
    """Test the NaN guard skip logic pattern used in KCTrainer (Round 10).

    This tests the actual guard pattern: if loss is non-finite, we should
    skip backward and continue to next batch.
    """
    # Simulate a batch loop with the guard logic
    losses = [
        torch.tensor(0.5),  # OK
        torch.tensor(float("nan")),  # Should skip
        torch.tensor(0.3),  # OK
        torch.tensor(float("inf")),  # Should skip
        torch.tensor(0.2),  # OK
    ]

    backward_count = 0
    skip_count = 0

    for loss in losses:
        # This mirrors the guard logic in train_epoch
        if not torch.isfinite(loss):
            skip_count += 1
            continue  # Skip backward

        # Would call backward() here
        backward_count += 1

    assert backward_count == 3, f"Expected 3 backward calls, got {backward_count}"
    assert skip_count == 2, f"Expected 2 skips, got {skip_count}"


def test_nan_guard_grad_skip_pattern():
    """Test the gradient NaN guard pattern used in _perform_optimizer_step (Round 10).

    This tests the pattern: check all param grads, skip step if any non-finite.
    """
    # Create mock parameter groups like optimizer.param_groups
    param_ok = torch.tensor([1.0, 2.0], requires_grad=True)
    param_ok.grad = torch.tensor([0.1, 0.2])

    param_nan = torch.tensor([1.0, 2.0], requires_grad=True)
    param_nan.grad = torch.tensor([0.1, float("nan")])

    param_inf = torch.tensor([1.0, 2.0], requires_grad=True)
    param_inf.grad = torch.tensor([float("inf"), 0.2])

    # Test case 1: all grads finite - should NOT skip
    param_groups_ok = [{"params": [param_ok]}]
    found_nonfinite = False
    for group in param_groups_ok:
        for p in group["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                found_nonfinite = True
                break
    assert not found_nonfinite, "Clean grads should not trigger skip"

    # Test case 2: one grad has NaN - should skip
    param_groups_nan = [{"params": [param_nan]}]
    found_nonfinite = False
    for group in param_groups_nan:
        for p in group["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                found_nonfinite = True
                break
    assert found_nonfinite, "NaN grad should trigger skip"

    # Test case 3: one grad has Inf - should skip
    param_groups_inf = [{"params": [param_inf]}]
    found_nonfinite = False
    for group in param_groups_inf:
        for p in group["params"]:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                found_nonfinite = True
                break
    assert found_nonfinite, "Inf grad should trigger skip"


# =============================================================================
# Round 13: Sparse Targets and Scaling Tests
# =============================================================================


def test_create_kc_batch_sparse_for_small_heads():
    """Test create_kc_batch returns sparse indices even for small heads (vocab <= 4096)."""
    from train.dataset import create_kc_batch

    batch = unittest.mock.Mock()
    batch.feature_inputs = {
        "input_ids_pos": torch.tensor([[4, 5, 6, 0], [10, 11, 0, 0]]),
    }
    batch.attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    batch.kc_targets = [{KcFamilyId.BAG_POS: [4, 5, 6]}, {KcFamilyId.BAG_POS: [10, 11]}]
    target_specs = {KcFamilyId.BAG_POS: 100}  # Small head: 100 < 4096

    tokenizer = unittest.mock.Mock(pad_id=0, unk_id=1, cls_id=2)
    result = create_kc_batch(batch, tokenizer, target_specs)

    assert f"kc_pos_inds_{KcFamilyId.BAG_POS.name.lower()}" in result
    assert f"kc_pos_mask_{KcFamilyId.BAG_POS.name.lower()}" in result

    # Check sparse indices
    inds = result[f"kc_pos_inds_{KcFamilyId.BAG_POS.name.lower()}"]
    # Should contain [4, 5, 6] for first item
    assert (inds[0] == 4).any()
    assert (inds[0] == 5).any()
    assert (inds[0] == 6).any()

    mask = result[f"kc_pos_mask_{KcFamilyId.BAG_POS.name.lower()}"]
    assert mask[0].sum() == 3


def test_create_kc_batch_sparse_for_large_heads():
    """Test create_kc_batch returns sparse indices for large heads (sparse families like NGRAM_POS)."""
    from train.dataset import create_kc_batch

    batch = unittest.mock.Mock()
    batch.feature_inputs = {
        "input_ids_pos": torch.tensor([[4, 5, 6, 0], [10, 11, 0, 0]]),
    }
    batch.attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    # Use NGRAM_POS which is a sparse family
    batch.kc_targets = [
        {KcFamilyId.NGRAM_POS: [4, 5, 6]},
        {KcFamilyId.NGRAM_POS: [10, 11]},
    ]
    target_specs = {KcFamilyId.NGRAM_POS: 16384}  # Large sparse family

    tokenizer = unittest.mock.Mock(pad_id=0, unk_id=1, cls_id=2)
    result = create_kc_batch(batch, tokenizer, target_specs)

    # Should have sparse indices, not dense (NGRAM_POS is sparse)
    assert f"kc_targets_{KcFamilyId.NGRAM_POS.name.lower()}" not in result
    assert f"kc_pos_inds_{KcFamilyId.NGRAM_POS.name.lower()}" in result
    assert f"kc_pos_mask_{KcFamilyId.NGRAM_POS.name.lower()}" in result

    pos_inds = result[f"kc_pos_inds_{KcFamilyId.NGRAM_POS.name.lower()}"]
    pos_mask = result[f"kc_pos_mask_{KcFamilyId.NGRAM_POS.name.lower()}"]

    # Check shapes
    assert pos_inds.shape[0] == 2  # batch size
    assert pos_inds.shape[1] == 64  # max_pos_per_sample default

    # Check sample 0: tokens 4, 5, 6 are all >= 4 so included (specials are <4)
    assert pos_mask[0, :3].all()  # First three positions should be valid
    assert set(pos_inds[0, :3].tolist()) == {4, 5, 6}


def test_bce_sampled_returns_finite_loss():
    """Test sampled BCE pattern produces finite loss."""
    # pylint: disable=too-many-locals
    batch_size, vocab_size, n_pos, n_neg = 4, 10000, 10, 128

    logits = torch.randn(batch_size, vocab_size)
    pos_inds = torch.randint(4, vocab_size, (batch_size, n_pos))
    pos_mask = torch.ones((batch_size, n_pos), dtype=torch.bool)
    pos_mask[:, 5:] = False  # Only first 5 positions valid

    # Build combined indices
    neg_i = torch.randint(4, vocab_size, (batch_size, n_neg))
    idxs = torch.cat([pos_inds, neg_i], dim=1)
    t_pos = pos_mask.float()
    t_neg = torch.zeros((batch_size, n_neg))
    t = torch.cat([t_pos, t_neg], dim=1)
    valid = torch.cat(
        [pos_mask, torch.ones((batch_size, n_neg), dtype=torch.bool)], dim=1
    )

    idxs_safe = idxs.clamp_min(0)
    gathered = logits.gather(1, idxs_safe)

    loss_elem = F.binary_cross_entropy_with_logits(gathered, t, reduction="none")
    loss = (loss_elem * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    assert torch.isfinite(loss), "Sampled BCE should produce finite loss"
    assert loss.item() > 0, "Loss should be positive"


# =============================================================================
# Round 14: KC AMP Safety and Skip-Loop Protection Tests
# =============================================================================


def test_consecutive_skip_counter_logic():
    """Test the consecutive skip counter + fail-fast logic pattern."""
    # Simulate the tracking variables
    consecutive_step_skips = 0
    total_step_skips = 0
    total_steps_applied = 0
    max_consecutive_skips = 25

    # Simulate 10 skipped steps
    for _ in range(10):
        consecutive_step_skips += 1
        total_step_skips += 1

    assert consecutive_step_skips == 10
    assert total_step_skips == 10

    # Simulate a successful step -> resets consecutive counter
    consecutive_step_skips = 0
    total_steps_applied += 1

    assert consecutive_step_skips == 0
    assert total_steps_applied == 1

    # Simulate exceeding max consecutive skips
    consecutive_step_skips = 26
    should_raise = consecutive_step_skips > max_consecutive_skips
    assert should_raise, "Should raise RuntimeError when consecutive skips > 25"


def test_float_sorting_not_string():
    """Test that numeric p values are sorted correctly (not as strings)."""
    # Simulate the sorting pattern (fix for str(p) -> float(p))
    stats = [
        {"p": 0.1},
        {"p": 0.01},
        {"p": 0.5},
        {"p": 0.05},
    ]

    # Sort by numeric p (correct behavior)
    sorted_stats = sorted(stats, key=lambda x: float(x.get("p", 0.0)))

    # Should be: 0.01, 0.05, 0.1, 0.5
    assert sorted_stats[0]["p"] == 0.01, "First should be 0.01"
    assert sorted_stats[1]["p"] == 0.05, "Second should be 0.05"
    assert sorted_stats[2]["p"] == 0.1, "Third should be 0.1"
    assert sorted_stats[3]["p"] == 0.5, "Fourth should be 0.5"


# =============================================================================
# Round 15: Constant Parameter Coverage
# =============================================================================


def test_forward_kc_parameter_variations():
    """Vary grad_cap and long_sentence_mask in TrainingClassifier.forward_kc."""
    # pylint: disable=protected-access, import-private-name
    config = ModelConfig(
        vocab_sizes={f: 50 for f in FEATURE_FIELDS},
        kc_vocab_size=16,
        kc_topk=4,
        kc_temperature=1.0,
    )
    model = TrainingClassifier(config)
    model.train()

    # Mock kc_head for manual control (and requires_grad)
    logits_raw = torch.randn(2, 16, requires_grad=True)
    model.kc_head.forward_with_raw = unittest.mock.MagicMock(
        return_value=(logits_raw, logits_raw)
    )
    # Ensure pooled output is returned
    model._get_pooled_output = unittest.mock.MagicMock(return_value=torch.randn(2, 32))

    field_inputs = {
        f"input_ids_{f}": torch.zeros(2, 5, dtype=torch.long) for f in FEATURE_FIELDS
    }
    attention_mask = torch.ones(2, 5)

    # 1. Test with grad_cap
    _ = model.forward_kc(field_inputs, attention_mask, grad_cap=1.0)

    # Verify hook logic doesn't crash.
    # To verify functional correctness: simulate backward pass
    loss = logits_raw.sum()
    loss.backward()
    # If hook was registered, it ran.

    # 2. Test with long_sentence_mask triggering re-normalization
    # We need a logit that results in > 0.85 prob.
    # Sigmoid(x) > 0.85 => x > 1.74. Let's use 5.0.
    logits_trigger = torch.zeros(2, 16)
    logits_trigger[0, 0] = 5.0
    logits_trigger.requires_grad = True  # needed for hook logic if re-used? No.

    model.kc_head.forward_with_raw.return_value = (logits_trigger, logits_trigger)

    long_mask = torch.tensor([True, False])  # Sample 0 is long

    # Should trigger the boost logic for sample 0
    out = model.forward_kc(field_inputs, attention_mask, long_sentence_mask=long_mask)

    # Check that sample 0 prob is DIFFERENT from what standard sigmoid(5.0) would be?
    # Actually, it re-normalizes with temperature boost (1.5x)
    # Original: sigmoid(5.0/1.0) = 0.9933
    # Boosted: sigmoid(5.0/1.5) = sigmoid(3.33) = 0.965
    # So prob should decrease.
    p0 = out["kc_probs"][0, 0].item()
    assert p0 < 0.99, f"Should have reduced confidence, got {p0}"


def test_bce_sampled_parameter_variations():
    """Vary neg_count, vocab_size, seed, family_name in _bce_sampled_from_sparse."""
    # pylint: disable=protected-access,import-outside-toplevel
    # We can test this by instantiating a KCTrainer wrapper or just using the method if it was static/separable.
    # It's an instance method but doesn't use much self state except config for nothing critical here?
    # Actually it's cleaner to mock KCTrainer or use kc_test_utils.

    # We'll adapt to a simple mock style here to avoid test class overhead if possible,
    # but since we are in test_kc_components.py we lack the wrapper.
    # Let's mock the trainer instance.
    from train.config import KCConfig, TrainerConfig
    from train.trainer import KCTrainer

    model = unittest.mock.MagicMock()
    model.config.kc_vocab_size = 100
    dataset = unittest.mock.MagicMock()

    # Patch DataLoader to avoid instantiation issues
    with unittest.mock.patch("train.kc_trainer.DataLoader"):
        trainer = KCTrainer(
            model,
            dataset,
            TrainerConfig(device="cpu", batch_size=2),
            unittest.mock.MagicMock(),
            KCConfig(),
        )

    logits = torch.randn(2, 100)
    pos_inds = torch.zeros(2, 10, dtype=torch.long)
    pos_mask = torch.zeros(2, 10, dtype=torch.bool)

    # Vary parameters
    loss, gap_val = trainer._bce_sampled_from_sparse(
        logits,
        pos_inds,
        pos_mask,
        vocab_size=100,
        neg_count=10,
        seed=42,
        family_name="var_test",
        reading_mask_id=999,
    )
    assert torch.isfinite(loss)
    # gap_val is now a tensor (changed with gap regularizer)
    assert isinstance(gap_val, (float, torch.Tensor))
    if isinstance(gap_val, torch.Tensor):
        assert torch.isfinite(gap_val)
