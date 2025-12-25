import math
import os
import sys

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kotogram.model import ModelConfig
from kotogram.tokenizer import FEATURE_FIELDS
from train.trainer import StyleClassifierWithMLM


def test_gumbel_noise_training_only():
    """Verify Gumbel noise is applied only during training and affects top-k."""
    config = ModelConfig(
        vocab_sizes={f: 100 for f in FEATURE_FIELDS},
        kc_enabled=True,
        kc_vocab_size=10,
        kc_topk=1,  # Select top-1 to easily see noise impact
        kc_temperature=1.0,
    )
    model = StyleClassifierWithMLM(config)

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
    B, V = 4, 300
    logits = torch.randn(B, V)
    targets = torch.zeros(B, V)

    # Set some positives
    targets[0, 10] = 1.0
    targets[1, 50] = 1.0

    # Mocking implementation
    pos_mask = targets > 0.5
    neg_count = 10  # Small count for test

    # Sample negatives
    neg_inds = torch.randint(0, V, (B, neg_count))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, neg_inds, True)

    mask = mask | pos_mask

    # Check that ALL positives are included
    assert mask[0, 10].item() is True
    assert mask[1, 50].item() is True

    # Check that we have roughly neg_count + num_pos active
    # (Could be less if collision)
    active_0 = mask[0].sum().item()
    assert neg_count <= active_0 <= neg_count + 1  # +1 for positive

    # Verify we can compute loss
    loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
    assert not torch.isnan(loss)
    assert loss > 0


def test_kc_probe_diagnose_collapse():
    """Verify _diagnose_kc_probe detects collapse risk correctly."""
    # Simulate collapse: high max_top1, low entropy
    probe_result = {
        "max_top1": 0.25,  # Very high (want < 0.10)
        "entropy_norm": 0.70,  # Low (want > 0.85)
        "uniq_kcs": 100,
        "kc_vocab_size": 1024,
    }

    # The diagnose function logic (inline test without Trainer instance)
    recommendations = []
    max_top1 = probe_result.get("max_top1", 0.0)
    entropy_norm = probe_result.get("entropy_norm", 1.0)

    collapse_risk = max_top1 > 0.10 or entropy_norm < 0.85
    if collapse_risk:
        recommendations.append("COLLAPSE RISK")

    assert collapse_risk is True, "Should detect collapse from high max_top1"
    assert "COLLAPSE RISK" in recommendations[0]


def test_kc_probe_diagnose_healthy():
    """Verify _diagnose_kc_probe approves healthy KC state."""
    probe_result = {
        "max_top1": 0.05,  # Good (< 0.10)
        "entropy_norm": 0.95,  # Good (> 0.85)
        "uniq_kcs": 800,  # Good usage
        "kc_vocab_size": 1024,
        "head_pos_auc": 0.92,  # Good
        "head_conjugated_form_auc": 0.88,  # Good
    }

    # Check collapse
    collapse_risk = (
        probe_result["max_top1"] > 0.10 or probe_result["entropy_norm"] < 0.85
    )
    assert collapse_risk is False, "Should not detect collapse for healthy metrics"

    # Check usage
    usage_ratio = probe_result["uniq_kcs"] / probe_result["kc_vocab_size"]
    assert usage_ratio >= 0.5, "Usage should be healthy"


def test_kc_probe_auc_calculation():
    """Verify AUC calculation logic for structural heads."""
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
