"""Regression tests for logit filtering and sampling.

These need no model weights, so they run in a second and can gate every change to the
sampling path -- which matters because these functions decide what the model is allowed
to say, and two of the defects covered here were silent rather than loud.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kronos import sample_from_logits, top_k_top_p_filtering  # noqa: E402


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(4, 64)


def test_returns_tensor_when_no_filter_requested(logits):
    """Used to fall off the end of the function and return None."""
    out = top_k_top_p_filtering(logits, top_k=0, top_p=1.0)
    assert out is not None
    torch.testing.assert_close(out, logits)


def test_top_k_and_top_p_compose(logits):
    """Used to `return` inside the top-k branch, silently discarding top_p."""
    k_only = top_k_top_p_filtering(logits, top_k=20, top_p=1.0)
    both = top_k_top_p_filtering(logits, top_k=20, top_p=0.5)

    n_k = torch.isfinite(k_only).sum().item()
    n_both = torch.isfinite(both).sum().item()
    assert n_both < n_k, "nucleus filter had no effect on top of top-k"
    # everything top-k removed must stay removed
    assert torch.all(torch.isfinite(both) <= torch.isfinite(k_only))


def test_does_not_mutate_caller_tensor(logits):
    before = logits.clone()
    top_k_top_p_filtering(logits, top_k=8, top_p=0.9)
    torch.testing.assert_close(logits, before)


@pytest.mark.parametrize("kw", [{"top_p": 0.9}, {"top_k": 5}, {}, {"top_k": None, "top_p": None}])
def test_sample_from_logits_accepts_partial_filter_args(logits, kw):
    """Passing only one of the two used to raise TypeError on None comparison."""
    out = sample_from_logits(logits, temperature=1.0, **kw)
    assert out.shape == (4, 1)
    assert out.dtype == torch.long
    assert int(out.min()) >= 0 and int(out.max()) < logits.size(-1)


def test_greedy_is_deterministic_and_is_the_argmax(logits):
    a = sample_from_logits(logits, sample_logits=False)
    b = sample_from_logits(logits, sample_logits=False)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a.squeeze(-1), logits.argmax(dim=-1))


def test_single_filter_paths_are_numerically_unchanged(logits):
    """Guards the claim that only the top_k-AND-top_p combination changed behaviour."""
    k = top_k_top_p_filtering(logits.clone(), top_k=10, top_p=1.0)
    expect_k = logits.clone()
    idx = expect_k < torch.topk(expect_k, 10)[0][..., -1, None]
    expect_k[idx] = -float("Inf")
    torch.testing.assert_close(k, expect_k)
