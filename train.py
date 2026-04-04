#!/usr/bin/env python3
"""
AdderBoard training script for Orze.

Trains small autoregressive transformers for 10-digit addition.
Supports configurable architecture (Qwen3-style), weight tying,
circular arc embeddings, curriculum learning, and more.

Orze contract:
    python train.py --idea-id <id> --results-dir <dir> --ideas-md <file> --config <yaml>
    Writes: results/{idea_id}/metrics.json
"""

import argparse
import json
import sys

# Force unbuffered output so orze can see training progress
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Data: 10-digit addition
# ---------------------------------------------------------------------------

MAX_ADDEND = 10**10 - 1
VOCAB_SIZE = 10  # digits 0-9 only (0 doubles as separator, like competition models)
NUM_DIGITS = 10
OUTPUT_LEN = 11  # up to 11 digits in result (10+10 can be 11 digits)
PROMPT_LEN = 24  # [0] + a_digits(10) + [0,0] + b_digits(10) + [0] — matches competition format


def encode_pair(a: int, b: int):
    """Encode a+b as token sequence. LSB-first digits, 0 as separator.

    Format: [0] + rev(a, 10) + [0, 0] + rev(b, 10) + [0]
    This matches the competition winner's encoding which places digits at
    specific RoPE positions critical for attention routing.
    """
    a_digits = [(a // 10**i) % 10 for i in range(NUM_DIGITS)]
    b_digits = [(b // 10**i) % 10 for i in range(NUM_DIGITS)]
    s = a + b
    s_digits = [(s // 10**i) % 10 for i in range(OUTPUT_LEN)]
    prompt = [0] + a_digits + [0, 0] + b_digits + [0]  # competition format
    target = s_digits
    return prompt, target


# Pre-compute divisors for vectorized digit extraction
_DIVISORS = torch.tensor([10**i for i in range(NUM_DIGITS)], dtype=torch.long)
_DIVISORS_OUT = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)


def generate_batch(batch_size, max_digits=NUM_DIGITS, device="cpu", commutative_aug=False):
    """Generate a batch of addition problems (vectorized)."""
    max_val = 10**max_digits - 1
    a = torch.randint(0, max_val + 1, (batch_size,), dtype=torch.long)
    b = torch.randint(0, max_val + 1, (batch_size,), dtype=torch.long)
    # Commutative augmentation: randomly swap a and b (since a+b = b+a)
    # This halves the effective problem space the model must learn
    if commutative_aug:
        swap_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        a_new = torch.where(swap_mask, b, a)
        b_new = torch.where(swap_mask, a, b)
        a, b = a_new, b_new
    s = a + b
    # Extract digits LSB-first: (batch, NUM_DIGITS)
    a_digits = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    s_digits = (s.unsqueeze(1) // _DIVISORS_OUT) % 10
    # Pad to full NUM_DIGITS with zeros
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(batch_size, NUM_DIGITS - max_digits, dtype=torch.long)
        a_digits = torch.cat([a_digits, pad], dim=1)
        b_digits = torch.cat([b_digits, pad], dim=1)
    # Build prompt: [0] + a_digits(10) + [0,0] + b_digits(10) + [0] — competition format
    sep1 = torch.zeros(batch_size, 1, dtype=torch.long)
    sep2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1)
    return prompts.to(device), s_digits.to(device)


def compute_carry_targets(a, b, max_digits=NUM_DIGITS):
    """Compute per-digit carry bits for addition a + b.

    Returns carry tensor of shape (batch, OUTPUT_LEN) where carry[i] = 1 if
    there's a carry INTO digit position i (i.e., the sum at position i-1 overflowed).
    Carry[0] is always 0 (no carry into the ones place).
    """
    dev = a.device
    a_digits = (a.unsqueeze(1) // _DIVISORS_OUT.to(dev)) % 10  # (B, OUTPUT_LEN)
    b_digits_padded = torch.zeros_like(a_digits)
    b_digs = (b.unsqueeze(1) // _DIVISORS[:max_digits].to(dev)) % 10
    b_digits_padded[:, :max_digits] = b_digs
    a_digits_padded = torch.zeros_like(a_digits)
    a_digs = (a.unsqueeze(1) // _DIVISORS[:max_digits].to(dev)) % 10
    a_digits_padded[:, :max_digits] = a_digs

    carries = torch.zeros(a.shape[0], OUTPUT_LEN, dtype=torch.long, device=dev)
    carry = torch.zeros(a.shape[0], dtype=torch.long, device=dev)
    for i in range(OUTPUT_LEN):
        digit_sum = a_digits_padded[:, i] + b_digits_padded[:, i] + carry
        carry = (digit_sum >= 10).long()
        if i + 1 < OUTPUT_LEN:
            carries[:, i + 1] = carry
    return carries  # (B, OUTPUT_LEN), values 0 or 1


def compute_max_carry_chain_length(a, b, max_digits=NUM_DIGITS):
    """Compute the length of the longest consecutive carry chain for each example.

    Returns tensor of shape (batch,) with the longest consecutive carry chain length.
    A carry chain of length N means N consecutive digit positions all had incoming carries.
    This measures the difficulty of the hardest carry pattern in each example — the key
    predictor of failure at 99%+ accuracy where nearly all remaining errors are long
    consecutive carry chain mistakes.
    """
    carries = compute_carry_targets(a, b, max_digits)  # (B, OUTPUT_LEN), values 0 or 1
    # Find longest consecutive run of 1s in each row
    batch_size = carries.shape[0]
    dev = carries.device
    max_chain = torch.zeros(batch_size, dtype=torch.long, device=dev)
    current = torch.zeros(batch_size, dtype=torch.long, device=dev)
    for i in range(OUTPUT_LEN):
        current = (current + 1) * carries[:, i]  # reset to 0 when carry=0, increment when carry=1
        max_chain = torch.maximum(max_chain, current)
    return max_chain  # (B,)


def generate_carry_biased_batch(batch_size, max_digits=NUM_DIGITS, carry_prob=0.5, device="cpu", commutative_aug=False):
    """Generate batch biased toward carry-heavy problems (vectorized)."""
    max_val = 10**max_digits - 1
    n_carry = int(batch_size * carry_prob)
    n_random = batch_size - n_carry
    # Random portion
    a_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)
    b_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)
    # Carry-biased portion: digits that sum > 9
    a_carry_digits = torch.randint(5, 10, (n_carry, max_digits), dtype=torch.long)
    b_carry_digits = torch.zeros(n_carry, max_digits, dtype=torch.long)
    for d in range(max_digits):
        lo = (10 - a_carry_digits[:, d]).clamp(min=0)
        b_carry_digits[:, d] = lo + torch.randint(0, 1, (n_carry,)) * (9 - lo)
    powers = _DIVISORS[:max_digits]
    a_carry = (a_carry_digits * powers).sum(dim=1).clamp(max=max_val)
    b_carry = (b_carry_digits * powers).sum(dim=1).clamp(max=max_val)
    a = torch.cat([a_rand, a_carry])
    b = torch.cat([b_rand, b_carry])
    # Shuffle
    perm = torch.randperm(batch_size)
    a, b = a[perm], b[perm]
    # Commutative augmentation: randomly swap a and b
    if commutative_aug:
        swap_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        a_new = torch.where(swap_mask, b, a)
        b_new = torch.where(swap_mask, a, b)
        a, b = a_new, b_new
    s = a + b
    a_digits = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    s_digits = (s.unsqueeze(1) // _DIVISORS_OUT) % 10
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(batch_size, NUM_DIGITS - max_digits, dtype=torch.long)
        a_digits = torch.cat([a_digits, pad], dim=1)
        b_digits = torch.cat([b_digits, pad], dim=1)
    sep1 = torch.zeros(batch_size, 1, dtype=torch.long)
    sep2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1)
    return prompts.to(device), s_digits.to(device)


def generate_long_carry_chain_batch(batch_size, max_digits=NUM_DIGITS, chain_prob=0.5,
                                    min_chain_len=3, device="cpu", commutative_aug=False):
    """Generate batch biased toward problems with long consecutive carry chains.

    Unlike carry_bias (which forces individual per-digit carries uniformly),
    this targets CONSECUTIVE carries (e.g., 999..9 + 1 style) which are the
    hardest patterns for tiny models. The remaining 2% errors at 97-98% accuracy
    are almost always long carry chain failures.

    Args:
        chain_prob: fraction of batch with forced long carry chains
        min_chain_len: minimum consecutive digits forced to carry
    """
    max_val = 10**max_digits - 1
    n_chain = int(batch_size * chain_prob)
    n_random = batch_size - n_chain
    # Random portion
    a_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)
    b_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)
    # Long carry chain portion: force consecutive digit positions to sum >= 10
    a_chain_digits = torch.randint(0, 10, (n_chain, max_digits), dtype=torch.long)
    b_chain_digits = torch.randint(0, 10, (n_chain, max_digits), dtype=torch.long)
    max_start = max(max_digits - min_chain_len, 0)
    starts = torch.randint(0, max_start + 1, (n_chain,))
    extra_lens = torch.randint(0, 3, (n_chain,))  # chain_len = min_chain_len + 0..2
    # Build carry mask: True at positions that should carry
    carry_mask = torch.zeros(n_chain, max_digits, dtype=torch.bool)
    for i in range(n_chain):
        end = min(starts[i].item() + min_chain_len + extra_lens[i].item(), max_digits)
        carry_mask[i, starts[i].item():end] = True
    # At carry positions: force b_d so that a_d + b_d >= 10
    a_at_carry = a_chain_digits[carry_mask]
    lo = (10 - a_at_carry).clamp(min=0)
    range_size = (10 - lo).clamp(min=1)
    b_at_carry = lo + (torch.rand(len(lo)) * range_size.float()).long().clamp(max=9)
    b_chain_digits[carry_mask] = b_at_carry
    powers = _DIVISORS[:max_digits]
    a_chain = (a_chain_digits * powers).sum(dim=1).clamp(max=max_val)
    b_chain = (b_chain_digits * powers).sum(dim=1).clamp(max=max_val)
    a = torch.cat([a_rand, a_chain])
    b = torch.cat([b_rand, b_chain])
    perm = torch.randperm(batch_size)
    a, b = a[perm], b[perm]
    if commutative_aug:
        swap_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        a_new = torch.where(swap_mask, b, a)
        b_new = torch.where(swap_mask, a, b)
        a, b = a_new, b_new
    s = a + b
    a_digits = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    s_digits = (s.unsqueeze(1) // _DIVISORS_OUT) % 10
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(batch_size, NUM_DIGITS - max_digits, dtype=torch.long)
        a_digits = torch.cat([a_digits, pad], dim=1)
        b_digits = torch.cat([b_digits, pad], dim=1)
    sep1 = torch.zeros(batch_size, 1, dtype=torch.long)
    sep2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1)
    return prompts.to(device), s_digits.to(device)


def generate_targeted_carry_batch(batch_size, max_digits=NUM_DIGITS, digit_acc_ema=None,
                                  target_fraction=0.5, device="cpu", commutative_aug=False):
    """Generate batch with carry chains targeted at the model's weakest digit positions.

    Unlike carry_bias (uniform random carries) or long_carry_chain (consecutive carries),
    this analyzes per-digit accuracy EMA and forces carries specifically at the digit
    positions where the model currently struggles. If digit positions 7-9 have the lowest
    accuracy, this generates problems where positions 7-9 specifically have incoming carries.

    Args:
        digit_acc_ema: (OUTPUT_LEN,) tensor of per-digit accuracy EMA. If None, falls back
                       to uniform carry biasing.
        target_fraction: fraction of batch with targeted carries (rest is random).
    """
    max_val = 10**max_digits - 1
    n_targeted = int(batch_size * target_fraction)
    n_random = batch_size - n_targeted

    # Random portion
    a_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)
    b_rand = torch.randint(0, max_val + 1, (n_random,), dtype=torch.long)

    if n_targeted > 0 and digit_acc_ema is not None:
        # Compute per-position targeting weight: lower accuracy → higher weight
        # Only consider positions within max_digits (carries can't occur beyond input digits)
        weights = (1.0 - digit_acc_ema[:max_digits].clamp(0, 1)).cpu()
        weights = weights / (weights.sum() + 1e-8)  # normalize to probability
        # If all positions have similar accuracy, fall back to uniform
        if weights.max() < 1.5 / max_digits:
            weights = torch.ones(max_digits) / max_digits

        # For each targeted sample, pick 1-3 digit positions weighted by difficulty
        a_targ_digits = torch.randint(0, 10, (n_targeted, max_digits), dtype=torch.long)
        b_targ_digits = torch.randint(0, 10, (n_targeted, max_digits), dtype=torch.long)

        # Sample target positions for each example
        n_carry_positions = torch.randint(1, min(4, max_digits + 1), (n_targeted,))
        for i in range(n_targeted):
            # Sample positions weighted by difficulty (lower accuracy = higher weight)
            n_pos = n_carry_positions[i].item()
            chosen = torch.multinomial(weights, min(n_pos, max_digits), replacement=False)
            for pos in chosen:
                p = pos.item()
                # Force a carry at this position: ensure a_d + b_d >= 10
                a_d = torch.randint(5, 10, (1,)).item()
                a_targ_digits[i, p] = a_d
                lo = max(0, 10 - a_d)
                b_targ_digits[i, p] = torch.randint(lo, 10, (1,)).item()

        powers = _DIVISORS[:max_digits]
        a_targ = (a_targ_digits * powers).sum(dim=1).clamp(max=max_val)
        b_targ = (b_targ_digits * powers).sum(dim=1).clamp(max=max_val)
    else:
        # Fallback: uniform carry biasing
        a_targ = torch.randint(0, max_val + 1, (n_targeted,), dtype=torch.long)
        b_targ = torch.randint(0, max_val + 1, (n_targeted,), dtype=torch.long)

    a = torch.cat([a_rand, a_targ])
    b = torch.cat([b_rand, b_targ])
    perm = torch.randperm(batch_size)
    a, b = a[perm], b[perm]
    if commutative_aug:
        swap_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        a_new = torch.where(swap_mask, b, a)
        b_new = torch.where(swap_mask, a, b)
        a, b = a_new, b_new
    s = a + b
    a_digits = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    s_digits = (s.unsqueeze(1) // _DIVISORS_OUT) % 10
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(batch_size, NUM_DIGITS - max_digits, dtype=torch.long)
        a_digits = torch.cat([a_digits, pad], dim=1)
        b_digits = torch.cat([b_digits, pad], dim=1)
    sep1 = torch.zeros(batch_size, 1, dtype=torch.long)
    sep2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1)
    return prompts.to(device), s_digits.to(device)


def generate_no_carry_batch(batch_size, max_digits=NUM_DIGITS, device="cpu", commutative_aug=False):
    """Generate batch of addition problems with guaranteed NO carry chains.

    Every digit pair satisfies a[i] + b[i] < 10, so no carry propagation occurs.
    Training on explicit no-carry examples helps calibrate the carry/no-carry
    decision boundary. At 99%+ accuracy, remaining errors include not just missed
    carries but also spurious carries (model adds a carry where none exists).
    Seeing clean no-carry examples sharpens this boundary.

    Use no_carry_fraction config to mix these into training batches.
    """
    max_val = 10**max_digits - 1
    # Construct digit-by-digit to guarantee no carries: a[i] + b[i] < 10
    a_digits = torch.randint(0, 10, (batch_size, max_digits), dtype=torch.long)
    b_digits = torch.zeros(batch_size, max_digits, dtype=torch.long)
    for d in range(max_digits):
        upper = (9 - a_digits[:, d]).clamp(min=0)
        b_digits[:, d] = (torch.rand(batch_size) * (upper.float() + 1)).long().clamp(max=upper)
    powers = _DIVISORS[:max_digits]
    a = (a_digits * powers).sum(dim=1).clamp(max=max_val)
    b = (b_digits * powers).sum(dim=1).clamp(max=max_val)
    if commutative_aug:
        swap_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        a_new = torch.where(swap_mask, b, a)
        b_new = torch.where(swap_mask, a, b)
        a, b = a_new, b_new
    s = a + b
    a_full = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_full = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    s_digits = (s.unsqueeze(1) // _DIVISORS_OUT) % 10
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(batch_size, NUM_DIGITS - max_digits, dtype=torch.long)
        a_full = torch.cat([a_full, pad], dim=1)
        b_full = torch.cat([b_full, pad], dim=1)
    sep1 = torch.zeros(batch_size, 1, dtype=torch.long)
    sep2 = torch.zeros(batch_size, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_full, sep2, b_full, sep1], dim=1)
    return prompts.to(device), s_digits.to(device)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, learnable=True):
        super().__init__()
        self.eps = eps
        if learnable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class UnitRMSNorm(nn.Module):
    """Parameterless RMS normalization."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class ScalarRMSNorm(nn.Module):
    """RMS normalization with a single learnable scale parameter.

    Saves d_model-1 params vs standard RMSNorm(d_model). For d_model=3, saves 2 params
    per norm layer. Three norms (ln1, ln2, ln_f) save 6 params total.
    Use norm_type="scalar_rms" to enable.
    """
    def __init__(self, dim=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class LerpRMSNorm(nn.Module):
    """RMSNorm with learnable interpolation: (1-t)*norm(x) + t*x.

    For tiny models, full normalization can destroy fine-grained information
    about carry chains. This lets the model learn how much normalization to apply.
    Initialized to nearly full normalization (t≈0), can learn to reduce it.
    Adds 1 extra param (the lerp weight) on top of RMSNorm's dim params.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.t = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007, nearly full norm

    def forward(self, x):
        t = torch.sigmoid(self.t)
        return (1 - t) * self.norm(x) + t * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0, max_seq_len=128, rope_period=None, learnable=False):
        super().__init__()
        if rope_period is not None and rope_period > 0:
            # Fixed period: all frequencies set to 2*pi/period
            inv_freq = torch.full((max(1, dim // 2),), 2 * math.pi / rope_period)
        else:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        if learnable:
            # Learnable RoPE: let the model discover optimal positional frequencies
            # for carry-chain routing. Same param count as fixed (replaces buffer).
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, offset=0):
        T = x.shape[-2]
        t = torch.arange(offset, offset + T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos_f = freqs.cos()
        sin_f = freqs.sin()
        # x shape: (B, H, T, D) or (B, T, D)
        # Handle odd head_dim: only rotate the even-length prefix
        D = x.shape[-1]
        rot_dim = (D // 2) * 2
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]  # empty if D is even
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        half = rot_dim // 2
        cos_f = cos_f[..., :half]
        sin_f = sin_f[..., :half]
        rotated = torch.stack([
            x1 * cos_f - x2 * sin_f,
            x1 * sin_f + x2 * cos_f,
        ], dim=-1).flatten(-2)
        if x_pass.shape[-1] > 0:
            out = torch.cat([rotated, x_pass], dim=-1)
        else:
            out = rotated
        return out


class CircularArcEmbedding(nn.Module):
    """Circular arc embedding: 3 learnable params instead of vocab_size * tok_dim."""
    def __init__(self, vocab_size, tok_dim=2):
        super().__init__()
        assert tok_dim == 2, "CircularArcEmbedding only supports tok_dim=2"
        self.vocab_size = vocab_size
        self.arc_A = nn.Parameter(torch.tensor(2.5))
        self.arc_start = nn.Parameter(torch.tensor(-1.2))
        self.arc_stride = nn.Parameter(torch.tensor(0.29))

    def table(self):
        d = torch.arange(self.vocab_size, device=self.arc_A.device, dtype=self.arc_A.dtype)
        angles = self.arc_start + d * self.arc_stride
        return torch.stack([self.arc_A * torch.cos(angles), self.arc_A * torch.sin(angles)], dim=1)

    def forward(self, tokens):
        return self.table()[tokens]

    def num_params(self):
        return 3


class QuadraticEmbedding(nn.Module):
    """Parametric quadratic embedding: e(d) = [w0 - w1*d^2, -d]. 2 params."""
    def __init__(self, vocab_size, tok_dim=2):
        super().__init__()
        assert tok_dim == 2
        self.vocab_size = vocab_size
        self.w0 = nn.Parameter(torch.tensor(1000.0))
        self.w1 = nn.Parameter(torch.tensor(0.001))

    def table(self):
        d = torch.arange(self.vocab_size, device=self.w0.device, dtype=self.w0.dtype)
        return torch.stack([self.w0 - self.w1 * d * d, -d], dim=1)

    def forward(self, tokens):
        return self.table()[tokens]

    def num_params(self):
        return 2


class LookupEmbedding(nn.Module):
    """Standard lookup table embedding."""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def table(self):
        return self.embed.weight

    def forward(self, tokens):
        return self.embed(tokens)

    def num_params(self):
        return self.embed.weight.numel()


class LearnedPhaseEmbedding(nn.Module):
    """Learned phase embedding: e(d) = A * [cos(base + d*stride + phase_d), sin(base + d*stride + phase_d)].

    Like circular_arc but with an additional per-digit learnable phase offset.
    This adds vocab_size params (10 for digits) to the 3 base params (A, start, stride),
    totaling 13 params. The per-digit phases let the model break the rigid arc geometry
    to place digit embeddings at optimal positions for carry-chain routing, while still
    sharing the global amplitude and base angle structure.

    Use embed_type="learned_phase" to enable. 13 params total (vs 3 for circular_arc,
    20 for full lookup with d_model=2).
    """
    def __init__(self, vocab_size, tok_dim=2):
        super().__init__()
        assert tok_dim == 2, "LearnedPhaseEmbedding only supports tok_dim=2"
        self.vocab_size = vocab_size
        self.arc_A = nn.Parameter(torch.tensor(2.5))
        self.arc_start = nn.Parameter(torch.tensor(-1.2))
        self.arc_stride = nn.Parameter(torch.tensor(0.29))
        self.phase_offsets = nn.Parameter(torch.zeros(vocab_size) * 0.01)

    def table(self):
        d = torch.arange(self.vocab_size, device=self.arc_A.device, dtype=self.arc_A.dtype)
        angles = self.arc_start + d * self.arc_stride + self.phase_offsets
        return torch.stack([self.arc_A * torch.cos(angles), self.arc_A * torch.sin(angles)], dim=1)

    def forward(self, tokens):
        return self.table()[tokens]

    def num_params(self):
        return 3 + self.vocab_size  # A, start, stride + per-digit phase offsets


class MixedNorm(nn.Module):
    """Learnable mix of RMSNorm and identity: out = sigmoid(t) * RMSNorm(x) + (1-sigmoid(t)) * x.

    Like LerpRMSNorm but parameterized differently: uses the SAME weight vector as RMSNorm
    (d_model params) plus 1 mixing param. The key difference from LerpRMSNorm is that MixedNorm
    initializes with t=0 (50/50 mix) rather than nearly-full-norm, which helps ultra-compressed
    models that need to preserve magnitude information through the residual stream for carry chains.
    Total params: d_model + 1.

    Use norm_type="mixed" to enable.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.mix = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5, balanced start

    def forward(self, x):
        t = torch.sigmoid(self.mix)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return t * normed + (1 - t) * x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def _apply_2d_rotation(x, angle):
    """Apply learned 2D rotation to pairs of dims. x: [..., D], angle: scalar."""
    c, s = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _apply_multi_angle_rotation(x, angles):
    """Apply per-pair learned 2D rotations. x: [..., D], angles: (D//2,).

    More expressive than single-angle rotation: each dimension pair gets
    its own rotation angle. For head_dim=4, uses 2 angles instead of 1.
    This better matches competition-winning rotation tying approaches.
    """
    c = torch.cos(angles)
    s = torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _build_givens_rotation(angles, dim):
    """Build dim x dim orthogonal matrix from product of Givens rotations.

    Uses dim*(dim-1)/2 angles for a full parametrization of SO(dim).
    For dim=4 (head_dim=4), uses 6 angles to create a 4x4 orthogonal matrix.
    More expressive than pair-wise rotation (_apply_multi_angle_rotation which
    only rotates adjacent pairs [0,1] and [2,3]): can represent ANY orthogonal
    transform including cross-pair mixing (e.g., dim 0 with dim 2).
    """
    R = torch.eye(dim, device=angles.device, dtype=angles.dtype)
    idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            if idx >= len(angles):
                break
            c = torch.cos(angles[idx])
            s = torch.sin(angles[idx])
            G = torch.eye(dim, device=angles.device, dtype=angles.dtype)
            G[i, i] = c
            G[j, j] = c
            G[i, j] = -s
            G[j, i] = s
            R = G @ R
            idx += 1
    return R


def _build_cayley_rotation(params, dim):
    """Build orthogonal matrix via the Cayley transform of a skew-symmetric matrix.

    R = (I - A) @ (I + A)^{-1}, where A is skew-symmetric (A^T = -A).
    This parameterization is:
    - Guaranteed orthogonal (det = +1 or -1) by construction
    - Numerically stable: no trigonometric functions, just a matrix solve
    - Smoother gradients near identity than Givens angles (no periodicity)
    - Better conditioned for SGD: the Cayley map is a rational function,
      avoiding the winding-number/vanishing-gradient issues of exp/trig maps

    For dim=4 (head_dim=4), uses 6 parameters (same as Givens).
    Initialized near zero → R ≈ I (near-identity rotation).
    """
    A = torch.zeros(dim, dim, device=params.device, dtype=params.dtype)
    idx = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            A[i, j] = params[idx]
            A[j, i] = -params[idx]
            idx += 1
    I = torch.eye(dim, device=params.device, dtype=params.dtype)
    R = (I - A) @ torch.linalg.solve(I + A, I)
    return R


def _build_hadamard(dim):
    """Build normalized Hadamard matrix of size dim (must be power of 2).

    The Walsh-Hadamard matrix is a fixed orthogonal transform that mixes ALL
    dimension pairs simultaneously. For dim=4, K = Q @ H creates non-trivial
    Q-K interactions across every pair of dimensions, unlike cyclic permutation
    (which only shuffles indices) or pair-wise rotation (which only mixes
    adjacent pairs [0,1] and [2,3]). Zero learnable parameters.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < dim:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    H = H[:dim, :dim] / math.sqrt(dim)
    return H


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim,
                 rope_theta=3.0, qk_norm=True, tie_kv=False, tie_qo=False,
                 tie_qk=False, k_rot_q=False, v_eq_q=False, tie_qk_norm=False,
                 attn_out_rank=0, use_rope=True, rope_period=None,
                 attn_temp=None, rope_learnable=False,
                 multi_angle_rot=False, rot_init=0.1, k_diag_q=False,
                 sigmoid_attn=False, k_householder_q=False, k_negate_q=False,
                 attn_dropout=0.0, freeze_rot=False, k_givens_q=False,
                 qk_norm_type="rms", k_perm_q=False, k_perm_shift=-1,
                 k_signperm_q=False, k_cayley_q=False,
                 k_hadamard_q=False, anneal_tying_steps=0,
                 k_factored_rank=0, k_residual_rank=0, k_affine_q=False,
                 k_partial_proj_dims=0, k_transform_after_norm=False,
                 k_proj_dropout_prob=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.sigmoid_attn = sigmoid_attn
        self.attn_dropout_p = attn_dropout
        self.k_factored = k_factored_rank > 0
        self.k_proj_dropout_prob = k_proj_dropout_prob

        # Attention temperature: learnable scaling on attention logits
        # "learned" = learnable scalar init to 1.0 (1 param), float = fixed value
        self.attn_temp_mode = attn_temp
        if attn_temp == "learned":
            self.attn_temp = nn.Parameter(torch.tensor(1.0))
        elif attn_temp is not None and attn_temp != "learned":
            self.register_buffer("attn_temp", torch.tensor(float(attn_temp)))
        else:
            self.attn_temp = None

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # K projection options
        self.k_rot_q = k_rot_q
        self.k_diag_q = k_diag_q
        self.k_householder_q = k_householder_q
        self.k_negate_q = k_negate_q
        self.k_givens_q = k_givens_q
        self.k_cayley_q = k_cayley_q
        self.k_perm_q = k_perm_q
        self.k_signperm_q = k_signperm_q
        self.k_hadamard_q = k_hadamard_q
        self.k_affine_q = k_affine_q
        self.multi_angle_rot = multi_angle_rot
        self.tie_qk = tie_qk
        if k_hadamard_q:
            # Fixed Walsh-Hadamard matrix: orthogonal transform, zero learnable params.
            # Mixes all dimension pairs simultaneously (more expressive than permutation).
            H = _build_hadamard(head_dim)
            self.register_buffer('k_hadamard_matrix', H)
        elif k_cayley_q:
            # Cayley transform: R = (I - A)(I + A)^{-1}, A is skew-symmetric
            # For head_dim=4: 6 upper-triangle params (same as Givens)
            # Initialized near zero → R ≈ I → K ≈ Q (near-identity start)
            n_skew = head_dim * (head_dim - 1) // 2
            self.k_cayley_params = nn.Parameter(torch.zeros(n_skew) + float(rot_init) * 0.1)
        elif k_givens_q:
            # Full Givens rotation: dim*(dim-1)/2 angles for SO(head_dim)
            # For head_dim=4: 6 angles. More expressive than pair-wise (2 angles).
            n_angles = head_dim * (head_dim - 1) // 2
            if freeze_rot:
                self.register_buffer('k_givens_angles', torch.full((n_angles,), float(rot_init)))
            else:
                self.k_givens_angles = nn.Parameter(torch.full((n_angles,), float(rot_init)))
        elif k_rot_q:
            if multi_angle_rot:
                n_angles = max(1, head_dim // 2)
                if freeze_rot:
                    self.register_buffer('k_rot_angles', torch.full((n_angles,), float(rot_init)))
                else:
                    self.k_rot_angles = nn.Parameter(torch.full((n_angles,), float(rot_init)))
            else:
                if freeze_rot:
                    self.register_buffer('k_rot_angle', torch.tensor(float(rot_init)))
                else:
                    self.k_rot_angle = nn.Parameter(torch.tensor(float(rot_init)))
        elif k_householder_q:
            # K = Householder(v) * Q: norm-preserving orthogonal transformation
            # Uses head_dim params (the reflection vector v). H = I - 2*vv^T/(v^Tv)
            # Easier to optimize than rotation, same norm-preservation property.
            self.householder_v = nn.Parameter(torch.randn(head_dim) * 0.1)
        elif k_perm_q:
            # K = Q with dimensions cyclically permuted. Zero extra params.
            # Unlike k_negate_q (which always gets 0%), cyclic perm preserves info
            # and interacts non-trivially with RoPE since it shuffles which dims
            # get paired for rotation (e.g., [0,1],[2,3] vs [1,2],[3,0]).
            perm_indices = torch.arange(head_dim)
            perm_indices = torch.roll(perm_indices, shifts=k_perm_shift)
            self.register_buffer('k_perm_indices', perm_indices)
        elif k_signperm_q:
            # K = sign_flip(cyclic_perm(Q)). Zero extra params.
            # Combines cyclic permutation with alternating sign flips: K[i] = (-1)^i * Q[perm[i]].
            # The sign flips reverse the effective RoPE rotation direction for odd dim pairs,
            # creating complementary Q-K attention patterns that may be more expressive than
            # plain permutation for carry-chain routing in ultra-compressed models.
            perm_indices = torch.arange(head_dim)
            perm_indices = torch.roll(perm_indices, shifts=k_perm_shift)
            self.register_buffer('k_signperm_indices', perm_indices)
            signs = torch.ones(head_dim)
            signs[1::2] = -1  # negate odd-indexed dimensions after permutation
            self.register_buffer('k_signperm_signs', signs)
        elif k_negate_q:
            pass  # K = -Q, zero extra params
        elif k_diag_q:
            # K = diag(s) * Q: per-dimension scaling, head_dim params
            # More expressive than rotation but cheaper than full K_proj
            self.k_scale = nn.Parameter(torch.ones(head_dim))
        elif k_affine_q:
            # K = scale ⊙ Q + offset: per-dimension affine transform.
            # 2*head_dim params (8 for hd=4) vs 12 for full K_proj, saving 4p.
            # Subsumes k_diag_q (scale only). The offset breaks Q=K symmetry
            # in a way that persists through QK-norm, unlike norm-preserving
            # transforms (rotation, permutation). Initialized near identity:
            # scale=1 (K≈Q), offset=0 (no bias). The offset provides a fixed
            # attention prior that shifts K toward/away from certain directions,
            # complementing RoPE's position-dependent rotation.
            self.k_affine_scale = nn.Parameter(torch.ones(head_dim))
            self.k_affine_offset = nn.Parameter(torch.zeros(head_dim))
        elif k_partial_proj_dims > 0 and k_partial_proj_dims < head_dim:
            # Partial K projection: first k_partial_proj_dims dims of K are tied to Q
            # (free, 0 params), remaining dims use a separate small K_proj.
            # For k_partial_proj_dims=2 with d=3, hd=4:
            #   K[:2] = Q[:2] (0 params), K[2:4] = partial_k_proj(x) (3*2=6 params)
            #   Saves 12-6=6p vs full K_proj. With tie_kv+tie_down_gate: 52->46p.
            # Bridges the gap between zero-param K-tying (always fails) and full K_proj:
            # the tied dims capture the coarse Q-K relationship, while the projected
            # dims provide input-dependent corrections for carry-chain routing.
            self.k_partial_proj_dims = k_partial_proj_dims
            _free_dims = head_dim - k_partial_proj_dims
            self.k_partial_proj = nn.Linear(d_model, n_kv_heads * _free_dims, bias=False)
        elif k_factored_rank > 0:
            # Low-rank factored K projection: K(x) = x @ A @ B
            # For rank=1 with d_model=3, head_dim=4: 3+4=7 params instead of 12.
            # Middle ground between full K_proj (12p) and zero-param Q-tying
            # (k_perm_q, k_rot_q, etc. which all fail to converge).
            # Preserves an explicit input-dependent K projection but with reduced
            # rank, forcing the model to compress attention routing through a
            # low-dimensional bottleneck. V follows tie_kv logic (V=K if tie_kv).
            self.k_factor_A = nn.Linear(d_model, k_factored_rank, bias=False)
            self.k_factor_B = nn.Linear(k_factored_rank, n_kv_heads * head_dim, bias=False)
        elif not tie_qk:
            self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # Annealed tying: gradually transition from independent K_proj to tied K.
        # Creates a staging K_proj used during the annealing period. At progress=0,
        # K = K_staging(x) (full expressivity). At progress=1, K = K_tied(Q).
        # Between: K = lerp(K_staging, K_tied, progress). The staging K_proj is
        # excluded from inference param count (see count_params).
        # Post-normalization K transform: apply K=transform(Q) AFTER QK-norm instead of before.
        # Standard: Q=Q_proj(x), K=transform(Q), Q=norm(Q), K=norm(K) → transform on raw Q
        # Post-norm: Q=Q_proj(x), Q=norm(Q), K=transform(Q_normed) → transform on unit vectors
        # Key differences: (1) K inherits normalized magnitude, no separate K norm needed;
        # (2) transforms like rotation/permutation operate on the unit sphere (more stable);
        # (3) gradients from K don't flow through the norm's computation of Q statistics.
        # This changes the optimization landscape and may help K-tying converge.
        self.k_transform_after_norm = k_transform_after_norm

        _has_k_tying = (k_hadamard_q or k_cayley_q or k_givens_q or k_rot_q or
                        k_diag_q or k_householder_q or k_negate_q or k_perm_q or
                        k_signperm_q or k_affine_q)
        self.anneal_tying = anneal_tying_steps > 0 and _has_k_tying
        self._anneal_progress = 1.0  # 1.0 = fully tied (default, backward compat)
        if self.anneal_tying:
            self._staging_k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # Low-rank residual added to tied K: K = f(Q) + x @ A @ B.
        # Provides a small input-dependent correction that breaks the pure Q-K tying
        # symmetry, bridging the gap between full K_proj (12p for d=3,hd=4) and
        # zero-param tying (k_perm_q etc. that fail to converge). For rank=1 with
        # d=3, hd=4: 3+4=7 params. Combined with tie_kv (V=K), the model gets
        # input-dependent K/V without the full 12p K_proj cost.
        self.k_residual = k_residual_rank > 0 and _has_k_tying
        if self.k_residual:
            self.k_res_A = nn.Linear(d_model, k_residual_rank, bias=False)
            self.k_res_B = nn.Linear(k_residual_rank, n_kv_heads * head_dim, bias=False)
            # Initialize near zero so residual starts small, preserving tying behavior
            nn.init.normal_(self.k_res_A.weight, std=0.01)
            nn.init.normal_(self.k_res_B.weight, std=0.01)

        # V projection options
        self.v_eq_q = v_eq_q
        self.tie_kv = tie_kv
        self.k_partial_proj_active = k_partial_proj_dims > 0 and k_partial_proj_dims < head_dim
        if not v_eq_q and not tie_kv and not tie_qk and not k_rot_q and not k_diag_q and not k_householder_q and not k_negate_q and not k_givens_q and not k_cayley_q and not k_perm_q and not k_signperm_q and not k_hadamard_q and not k_affine_q and not self.k_partial_proj_active:
            self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        self.tie_qo = tie_qo
        if attn_out_rank > 0:
            self.out_A = nn.Parameter(torch.empty(n_heads * head_dim, attn_out_rank))
            self.out_B = nn.Parameter(torch.empty(attn_out_rank, d_model))
            nn.init.kaiming_uniform_(self.out_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.out_B, a=math.sqrt(5))
            self.o_proj = None
        elif not tie_qo:
            self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
            self.out_A = None
        else:
            self.o_proj = None
            self.out_A = None

        self.qk_norm = qk_norm
        self.tie_qk_norm = tie_qk_norm
        if qk_norm:
            if qk_norm_type == "unit":
                self.q_norm = UnitRMSNorm()
                if not tie_qk_norm:
                    self.k_norm = UnitRMSNorm()
            elif qk_norm_type == "scalar_rms":
                self.q_norm = ScalarRMSNorm()
                if not tie_qk_norm:
                    self.k_norm = ScalarRMSNorm()
            else:
                self.q_norm = RMSNorm(head_dim)
                if not tie_qk_norm:
                    self.k_norm = RMSNorm(head_dim)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(head_dim, theta=rope_theta, rope_period=rope_period,
                                        learnable=rope_learnable)

    def _apply_k_transform(self, q):
        """Apply the K=transform(Q) operation. Factored out for use in both
        pre-norm and post-norm K transform modes."""
        if self.k_hadamard_q:
            return (q.reshape(-1, self.head_dim) @ self.k_hadamard_matrix).reshape_as(q)
        elif self.k_cayley_q:
            R = _build_cayley_rotation(self.k_cayley_params, self.head_dim)
            return (q.reshape(-1, self.head_dim) @ R).reshape_as(q)
        elif self.k_givens_q:
            R = _build_givens_rotation(self.k_givens_angles, self.head_dim)
            return (q.reshape(-1, self.head_dim) @ R).reshape_as(q)
        elif self.k_rot_q:
            if self.multi_angle_rot:
                return _apply_multi_angle_rotation(q, self.k_rot_angles)
            else:
                return _apply_2d_rotation(q, self.k_rot_angle)
        elif self.k_householder_q:
            v_hat = self.householder_v / (self.householder_v.norm() + 1e-8)
            proj = (q * v_hat).sum(dim=-1, keepdim=True)
            return q - 2 * proj * v_hat
        elif self.k_perm_q:
            return q[:, :, :, self.k_perm_indices]
        elif self.k_signperm_q:
            return q[:, :, :, self.k_signperm_indices] * self.k_signperm_signs
        elif self.k_negate_q:
            return -q
        elif self.k_diag_q:
            return q * self.k_scale
        elif self.k_affine_q:
            return q * self.k_affine_scale + self.k_affine_offset
        return q  # fallback: K = Q

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Check if this is a K-tying transform (as opposed to independent K_proj)
        _is_k_transform = (self.k_hadamard_q or self.k_cayley_q or self.k_givens_q or
                           self.k_rot_q or self.k_householder_q or self.k_perm_q or
                           self.k_signperm_q or self.k_negate_q or self.k_diag_q or
                           self.k_affine_q)

        # K computation — two modes:
        # Standard (k_transform_after_norm=False): K=transform(Q), then norm both Q and K
        # Post-norm (k_transform_after_norm=True): norm Q first, then K=transform(Q_normed)
        if self.k_transform_after_norm and _is_k_transform:
            # Post-norm mode: normalize Q first, then derive K from normalized Q.
            # K inherits the normalized magnitude, no separate K norm needed.
            k = None  # will be computed after QK-norm below
        elif _is_k_transform:
            k = self._apply_k_transform(q)
        elif self.k_partial_proj_active:
            # Partial K projection: first k_partial_proj_dims dims from Q, rest from projection
            _tied_dims = self.k_partial_proj_dims
            k_tied = q[..., :_tied_dims]
            _free_dims = self.head_dim - _tied_dims
            k_free = self.k_partial_proj(x).view(B, T, self.n_kv_heads, _free_dims).transpose(1, 2)
            k = torch.cat([k_tied, k_free], dim=-1)
        elif self.k_factored:
            k = self.k_factor_B(self.k_factor_A(x)).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        elif self.tie_qk:
            k = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            # K-projection dropout: during training, randomly replace K=K_proj(x) with K=Q.
            # This teaches the model to work with both independent and tied K, potentially
            # enabling post-training K_proj removal (52p → 40p) or smoother annealed tying.
            if self.training and self.k_proj_dropout_prob > 0:
                if torch.rand(1).item() < self.k_proj_dropout_prob:
                    k = q.clone()

        # Annealed tying: interpolate between staging K_proj and tied K
        if self.anneal_tying and self._anneal_progress < 1.0 and k is not None:
            k_free = self._staging_k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            t = self._anneal_progress
            k = (1 - t) * k_free + t * k

        # Low-rank residual: add input-dependent correction to tied K
        if self.k_residual and k is not None:
            k_res = self.k_res_B(self.k_res_A(x)).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k = k + k_res

        # V computation — compute before post-norm K transform since V=K uses pre-norm K
        if self.v_eq_q:
            v = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        elif self.k_partial_proj_active and (self.tie_kv):
            v = k.clone()
        elif self.k_hadamard_q or self.k_cayley_q or self.k_givens_q or self.k_rot_q or self.k_diag_q or self.k_affine_q or self.k_householder_q or self.k_negate_q or self.k_perm_q or self.k_signperm_q or self.tie_kv or self.tie_qk:
            if k is not None:
                v = k.clone()
            else:
                # Post-norm mode with V=K: V = transform(Q) (pre-norm version for V)
                v = self._apply_k_transform(q)
        else:
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            if self.k_transform_after_norm and _is_k_transform:
                # Post-norm K transform: derive K from the already-normalized Q.
                # K is already on the unit sphere, so no separate K norm needed.
                k = self._apply_k_transform(q)
                # Add low-rank residual after post-norm transform if applicable
                if self.k_residual:
                    k_res = self.k_res_B(self.k_res_A(x)).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
                    k = k + k_res
            else:
                k_norm = self.q_norm if self.tie_qk_norm else self.k_norm
                k = k_norm(k)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # GQA: repeat KV heads
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Attention
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_temp is not None:
            scores = scores * self.attn_temp
        if self.sigmoid_attn:
            # Sigmoid attention: each position independently decides attention weight
            # Better for fixed digit-routing patterns (no forced normalization across seq)
            attn = torch.sigmoid(scores)
            if mask is not None:
                causal_mask = (mask == 0).float()
                attn = attn * causal_mask
        else:
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
        if self.training and self.attn_dropout_p > 0:
            attn = F.dropout(attn, p=self.attn_dropout_p, training=True)
        out = attn @ v  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        if self.out_A is not None:
            out = (out @ self.out_A) @ self.out_B
        elif self.tie_qo:
            out = F.linear(out, self.q_proj.weight.T)
        elif self.o_proj is not None:
            out = self.o_proj(out)

        return out


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, ff_dim, tie_gate=False, tie_down_gate=False, down_rot_up_t=False,
                 multi_angle_rot=False, rot_init=0.1, freeze_rot=False,
                 tie_gate_q=False, tie_up_q=False, tie_down_up=False,
                 tie_up_gate_offset=False):
        super().__init__()
        self.tie_gate_q = tie_gate_q
        self.tie_up_q = tie_up_q
        self.tie_up_gate_offset = tie_up_gate_offset
        self.ff_dim = ff_dim
        # Gate projection: own weights unless tied to Q_proj rows
        if not tie_gate_q:
            self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        # Up projection: own weights unless tied to Q_proj rows, gate activation, or gate+offset
        if not tie_up_q and not tie_gate and not tie_up_gate_offset:
            self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        # Shared gate-up with learnable offset: up(x) = gate_proj(x) + offset.
        # Saves ff_dim * d_model - ff_dim params (e.g., 6 - 2 = 4p for ff_dim=2, d_model=3).
        # More expressive than tie_gate (gate = up, always kills training) because the offset
        # breaks the symmetry between the gating path (silu) and the value path (linear).
        # SwiGLU becomes: silu(Wx) * (Wx + b) where W = gate_proj and b = offset.
        if tie_up_gate_offset:
            self.up_offset = nn.Parameter(torch.zeros(ff_dim))
        self.tie_down_gate = tie_down_gate
        self.tie_down_up = tie_down_up
        self.down_rot_up_t = down_rot_up_t
        self.multi_angle_rot = multi_angle_rot
        if down_rot_up_t:
            if multi_angle_rot:
                n_angles = max(1, ff_dim // 2)
                if freeze_rot:
                    self.register_buffer('down_rot_angles', torch.full((n_angles,), float(rot_init)))
                else:
                    self.down_rot_angles = nn.Parameter(torch.full((n_angles,), float(rot_init)))
            else:
                if freeze_rot:
                    self.register_buffer('down_rot_angle', torch.tensor(float(rot_init)))
                else:
                    self.down_rot_angle = nn.Parameter(torch.tensor(float(rot_init)))
        elif not tie_down_gate and not tie_down_up:
            self.down_proj = nn.Linear(ff_dim, d_model, bias=False)
        self.tie_gate = tie_gate

    def forward(self, x, q_weight=None):
        # Gate computation
        if self.tie_gate_q and q_weight is not None:
            gate_raw = F.linear(x, q_weight[:self.ff_dim, :])
        else:
            gate_raw = self.gate_proj(x)
        gate = F.silu(gate_raw)
        # Up computation
        if self.tie_up_gate_offset:
            up = gate_raw + self.up_offset  # share gate_proj weight, differ by learnable offset
        elif self.tie_up_q and q_weight is not None:
            up = F.linear(x, q_weight[self.ff_dim:2*self.ff_dim, :])
        elif self.tie_gate:
            up = gate
        else:
            up = self.up_proj(x)
        mixed = gate * up
        # Down computation
        if self.down_rot_up_t:
            # down(x) = F.linear(rotate_2d(x, theta), up^T) — rotate INPUT, then up^T
            if self.multi_angle_rot:
                rotated = _apply_multi_angle_rotation(mixed, self.down_rot_angles)
            else:
                rotated = _apply_2d_rotation(mixed, self.down_rot_angle)
            if self.tie_up_q and q_weight is not None:
                return F.linear(rotated, q_weight[self.ff_dim:2*self.ff_dim, :].T)
            if self.tie_up_gate_offset:
                # up shares gate_proj weight, so use gate^T
                return F.linear(rotated, self.gate_proj.weight.T)
            return F.linear(rotated, self.up_proj.weight.T)
        elif self.tie_down_gate:
            if self.tie_gate_q and q_weight is not None:
                return F.linear(mixed, q_weight[:self.ff_dim, :].T)
            return F.linear(mixed, self.gate_proj.weight.T)
        elif self.tie_down_up:
            if self.tie_up_q and q_weight is not None:
                return F.linear(mixed, q_weight[self.ff_dim:2*self.ff_dim, :].T)
            if self.tie_up_gate_offset:
                # up shares gate_proj weight, so down = gate^T (same as tie_down_gate)
                return F.linear(mixed, self.gate_proj.weight.T)
            return F.linear(mixed, self.up_proj.weight.T)
        return self.down_proj(mixed)


class GeLUMLP(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x, q_weight=None):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, ff_dim,
                 rope_theta=3.0, qk_norm=True, tie_kv=False, tie_qo=False,
                 tie_qk=False, k_rot_q=False, v_eq_q=False, tie_qk_norm=False,
                 attn_out_rank=0, use_swiglu=True, tie_gate=False,
                 tie_down_gate=False, down_rot_up_t=False,
                 share_norms=False, norm_type="rms", use_rope=True,
                 rope_period=None, residual_alpha=None, attn_temp=None,
                 rope_learnable=False, multi_angle_rot=False, rot_init=0.1,
                 k_diag_q=False, sigmoid_attn=False, parallel_block=False,
                 k_householder_q=False, k_negate_q=False, norm_lerp=False,
                 attn_dropout=0.0, freeze_rot=False, k_givens_q=False,
                 tie_gate_q=False, tie_up_q=False, qk_norm_type="rms",
                 k_perm_q=False, tie_down_up=False, k_perm_shift=-1,
                 k_signperm_q=False, k_cayley_q=False,
                 k_hadamard_q=False, anneal_tying_steps=0,
                 residual_gate=False, k_factored_rank=0,
                 k_residual_rank=0, layer_noise_scale=0.0,
                 k_affine_q=False, stochastic_depth_rate=0.0,
                 k_partial_proj_dims=0, k_transform_after_norm=False,
                 k_proj_dropout_prob=0.0, tie_up_gate_offset=False):
        super().__init__()
        self.parallel_block = parallel_block
        self.layer_noise_scale = layer_noise_scale
        self.stochastic_depth_rate = stochastic_depth_rate
        self.tie_gate_q = tie_gate_q
        self.tie_up_q = tie_up_q
        self.has_mlp = ff_dim > 0  # ff_dim=0 means attention-only block (no MLP)
        # Annealed norm sharing: smooth transition from independent ln2 to shared ln1=ln2.
        # _norm_blend_t < 0: disabled (default, backward compatible)
        # _norm_blend_t in [0, 1): blending ln2 toward ln1: out = (1-t)*ln2(x) + t*ln1(x)
        # _norm_blend_t >= 1.0: fully shared (ln2 = ln1)
        self._norm_blend_t = -1.0
        self.attn = Attention(
            d_model, n_heads, n_kv_heads, head_dim,
            rope_theta=rope_theta, qk_norm=qk_norm,
            tie_kv=tie_kv, tie_qo=tie_qo, tie_qk=tie_qk,
            k_rot_q=k_rot_q, v_eq_q=v_eq_q, tie_qk_norm=tie_qk_norm,
            attn_out_rank=attn_out_rank,
            use_rope=use_rope, rope_period=rope_period,
            attn_temp=attn_temp, rope_learnable=rope_learnable,
            multi_angle_rot=multi_angle_rot, rot_init=rot_init, k_diag_q=k_diag_q,
            sigmoid_attn=sigmoid_attn,
            k_householder_q=k_householder_q, k_negate_q=k_negate_q,
            attn_dropout=attn_dropout,
            freeze_rot=freeze_rot,
            k_givens_q=k_givens_q,
            qk_norm_type=qk_norm_type,
            k_perm_q=k_perm_q,
            k_perm_shift=k_perm_shift,
            k_signperm_q=k_signperm_q,
            k_cayley_q=k_cayley_q,
            k_hadamard_q=k_hadamard_q,
            anneal_tying_steps=anneal_tying_steps,
            k_factored_rank=k_factored_rank,
            k_residual_rank=k_residual_rank,
            k_affine_q=k_affine_q,
            k_partial_proj_dims=k_partial_proj_dims,
            k_transform_after_norm=k_transform_after_norm,
            k_proj_dropout_prob=k_proj_dropout_prob,
        )

        if self.has_mlp:
            if use_swiglu:
                self.mlp = SwiGLUMLP(d_model, ff_dim, tie_gate=tie_gate,
                                     tie_down_gate=tie_down_gate, down_rot_up_t=down_rot_up_t,
                                     multi_angle_rot=multi_angle_rot, rot_init=rot_init,
                                     freeze_rot=freeze_rot,
                                     tie_gate_q=tie_gate_q, tie_up_q=tie_up_q,
                                     tie_down_up=tie_down_up,
                                     tie_up_gate_offset=tie_up_gate_offset)
            else:
                self.mlp = GeLUMLP(d_model, ff_dim)

        if norm_type == "none":
            self.ln1 = nn.Identity()
            if self.has_mlp:
                self.ln2 = nn.Identity()
        elif norm_type == "scalar_rms":
            self.ln1 = ScalarRMSNorm()
            if self.has_mlp:
                self.ln2 = ScalarRMSNorm() if not share_norms and not parallel_block else self.ln1
        elif norm_type == "unit":
            self.ln1 = UnitRMSNorm()
            if self.has_mlp:
                self.ln2 = UnitRMSNorm() if not parallel_block else self.ln1
        elif norm_type == "mixed":
            self.ln1 = MixedNorm(d_model)
            if self.has_mlp:
                self.ln2 = MixedNorm(d_model) if not share_norms and not parallel_block else self.ln1
        elif norm_lerp:
            self.ln1 = LerpRMSNorm(d_model)
            if self.has_mlp:
                self.ln2 = LerpRMSNorm(d_model) if not share_norms and not parallel_block else self.ln1
        else:
            self.ln1 = RMSNorm(d_model)
            if self.has_mlp:
                self.ln2 = RMSNorm(d_model) if not share_norms and not parallel_block else self.ln1

        # Learnable residual scaling: x = alpha * sublayer(x) + x
        # Helps tiny models balance residual vs attention/mlp contribution
        self.residual_alpha = None
        if residual_alpha is not None:
            self.attn_alpha = nn.Parameter(torch.tensor(float(residual_alpha)))
            if self.has_mlp:
                self.mlp_alpha = nn.Parameter(torch.tensor(float(residual_alpha)))
            self.residual_alpha = True

        # Per-dimension residual gating: x = x + sigmoid(g) * sublayer(x)
        # Unlike scalar residual_alpha, this provides per-dimension control.
        # For d_model=3, the model can gate the positional dimension differently
        # from token dimensions. Initialized to 0 → sigmoid(0)=0.5 (half-strength).
        # Adds 2*d_model params (d_model for attn + d_model for MLP).
        self.residual_gate = residual_gate
        if residual_gate:
            self.gate_attn = nn.Parameter(torch.zeros(d_model))
            if self.has_mlp:
                self.gate_mlp = nn.Parameter(torch.zeros(d_model))

    def _ln2_out(self, x):
        """Apply ln2 with optional annealed norm sharing blend toward ln1."""
        if self._norm_blend_t >= 1.0:
            return self.ln1(x)
        elif self._norm_blend_t >= 0.0:
            t = self._norm_blend_t
            return (1 - t) * self.ln2(x) + t * self.ln1(x)
        return self.ln2(x)

    def _apply_layer_noise(self, out):
        """Apply multiplicative noise to sublayer output during training."""
        if self.training and self.layer_noise_scale > 0:
            noise = 1.0 + (torch.rand_like(out) * 2 - 1) * self.layer_noise_scale
            return out * noise
        return out

    def forward(self, x, mask=None):
        # Stochastic depth: randomly skip entire block during training (Huang et al. 2016).
        # Returns input unchanged, so the residual connection is the only path.
        # Regularizes against over-reliance on any single computation path.
        if self.training and self.stochastic_depth_rate > 0:
            if torch.rand(1).item() < self.stochastic_depth_rate:
                return x
        if not self.has_mlp:
            # Attention-only block: no MLP, no ln2
            attn_out = self._apply_layer_noise(self.attn(self.ln1(x), mask=mask))
            if self.residual_gate:
                x = x + torch.sigmoid(self.gate_attn) * attn_out
            elif self.residual_alpha:
                x = x + self.attn_alpha * attn_out
            else:
                x = x + attn_out
            return x
        _qw = self.attn.q_proj.weight if (self.tie_gate_q or self.tie_up_q) else None
        if self.parallel_block:
            # PaLM-style: attn and MLP in parallel from same normalized input
            h = self.ln1(x)
            attn_out = self._apply_layer_noise(self.attn(h, mask=mask))
            mlp_out = self._apply_layer_noise(self.mlp(h, q_weight=_qw))
            if self.residual_gate:
                x = x + torch.sigmoid(self.gate_attn) * attn_out + torch.sigmoid(self.gate_mlp) * mlp_out
            elif self.residual_alpha:
                x = x + self.attn_alpha * attn_out + self.mlp_alpha * mlp_out
            else:
                x = x + attn_out + mlp_out
        elif self.residual_gate:
            x = x + torch.sigmoid(self.gate_attn) * self._apply_layer_noise(self.attn(self.ln1(x), mask=mask))
            x = x + torch.sigmoid(self.gate_mlp) * self._apply_layer_noise(self.mlp(self._ln2_out(x), q_weight=_qw))
        elif self.residual_alpha:
            x = x + self.attn_alpha * self._apply_layer_noise(self.attn(self.ln1(x), mask=mask))
            x = x + self.mlp_alpha * self._apply_layer_noise(self.mlp(self._ln2_out(x), q_weight=_qw))
        else:
            x = x + self._apply_layer_noise(self.attn(self.ln1(x), mask=mask))
            x = x + self._apply_layer_noise(self.mlp(self._ln2_out(x), q_weight=_qw))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AdderTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = cfg["d_model"]
        n_heads = cfg.get("n_heads", 1)
        n_kv_heads = cfg.get("n_kv_heads", 1) or n_heads  # 0 means use n_heads (standard MHA)
        head_dim = cfg.get("head_dim", 4)
        ff_mult = cfg.get("ff_mult", 2)
        ff_dim = cfg.get("ff_dim", d_model * ff_mult)  # explicit ff_dim overrides ff_mult
        n_layers = cfg.get("n_layers", 1)
        rope_theta = cfg.get("rope_theta", 3.0)
        qk_norm = cfg.get("qk_norm", True)
        tie_kv = cfg.get("tie_kv", False)
        tie_qo = cfg.get("tie_qo", False)
        tie_qk = cfg.get("tie_qk", False)
        k_rot_q = cfg.get("k_rot_q", False)
        v_eq_q = cfg.get("v_eq_q", False)
        tie_qk_norm = cfg.get("tie_qk_norm", False)
        tie_gate = cfg.get("tie_gate", False)
        tie_down_gate = cfg.get("tie_down_gate", False)
        down_rot_up_t = cfg.get("down_rot_up_t", False)
        tie_embed = cfg.get("tie_embed", True)
        share_norms = cfg.get("share_norms", False)
        attn_out_rank = cfg.get("attn_out_rank", 0)
        use_swiglu = cfg.get("use_swiglu", True)
        norm_type = cfg.get("norm_type", "rms")
        use_rope = cfg.get("use_rope", True)
        rope_period = cfg.get("rope_period", None)
        embed_type = cfg.get("embed_type", "lookup")
        vocab_size = cfg.get("vocab_size", VOCAB_SIZE)
        residual_alpha = cfg.get("residual_alpha", None)  # learnable residual scaling
        attn_temp = cfg.get("attn_temp", None)  # attention temperature: "learned" or float
        rope_learnable = cfg.get("rope_learnable", False)  # learnable RoPE frequencies
        multi_angle_rot = cfg.get("multi_angle_rot", False)  # per-pair rotation angles for k_rot_q/down_rot_up_t
        rot_init = float(cfg.get("rot_init", 0.1))  # rotation angle initialization value
        k_diag_q = cfg.get("k_diag_q", False)  # K = diag(s) * Q, head_dim learnable scales
        k_householder_q = cfg.get("k_householder_q", False)  # K = Householder(v) * Q, head_dim params, norm-preserving
        k_negate_q = cfg.get("k_negate_q", False)  # K = -Q, 0 extra params
        k_givens_q = cfg.get("k_givens_q", False)  # K = Givens(angles) * Q, head_dim*(head_dim-1)/2 params, full SO(dim)
        k_perm_q = cfg.get("k_perm_q", False)  # K = cyclic_perm(Q), 0 params, breaks Q=K symmetry via RoPE interaction
        k_perm_shift = int(cfg.get("k_perm_shift", -1))  # cyclic shift amount for k_perm_q: -1=[1,2,3,0], 1=[3,0,1,2], 2=[2,3,0,1]. Different shifts interact differently with RoPE.
        k_signperm_q = cfg.get("k_signperm_q", False)  # K = sign_flip(cyclic_perm(Q)), 0 params. Like k_perm_q but with alternating sign flips that reverse effective RoPE rotation direction for odd dim pairs. Creates complementary Q-K attention patterns.
        k_cayley_q = cfg.get("k_cayley_q", False)  # K = Cayley(A) * Q, head_dim*(head_dim-1)/2 params. Uses Cayley transform R=(I-A)(I+A)^{-1} for a skew-symmetric A, which is numerically more stable than Givens angles, has smoother gradients near identity, and avoids trigonometric periodicity issues. For head_dim=4: 6 params (same as Givens).
        k_hadamard_q = cfg.get("k_hadamard_q", False)  # K = Hadamard(Q), 0 params. Fixed Walsh-Hadamard orthogonal transform that mixes ALL dimension pairs. More expressive than cyclic permutation (k_perm_q) since it creates cross-pair interactions. For head_dim=4: the 4x4 normalized Hadamard matrix.
        k_factored_rank = int(cfg.get("k_factored_rank", 0))  # Low-rank factored K projection: K(x) = x @ A @ B where A is (d_model, rank) and B is (rank, head_dim). For rank=1 with d=3, hd=4: 3+4=7 params instead of 12, saving 5p. Middle ground between full K_proj and zero-param Q-tying approaches (k_perm_q etc.) that fail to converge. V follows tie_kv logic (V=K when tie_kv=True).
        k_residual_rank = int(cfg.get("k_residual_rank", 0))  # Low-rank residual added to tied K: K = f(Q) + x @ A @ B. Provides input-dependent correction to zero-param K-tying (k_perm_q, k_hadamard_q, etc.) that otherwise fail to converge. For rank=1 with d=3, hd=4: 7 extra params (vs 12 for full K_proj). The key insight: pure Q-K tying fails because K needs SOME input dependence, but a full K_proj is expensive. This adds just enough flexibility.
        k_partial_proj_dims = int(cfg.get("k_partial_proj_dims", 0))  # Partial K projection: first N dims of K are tied to Q (0 params), remaining dims use a separate K_proj. For k_partial_proj_dims=2 with d=3, hd=4: tied dims=2 (0p), projected dims=2 (3*2=6p). Saves 12-6=6p vs full K_proj. With tie_kv+tie_down_gate: 52->46p. Bridges zero-param K-tying (always fails) and full K_proj: tied dims capture coarse Q-K structure, projected dims provide input-dependent carry-chain routing.
        k_transform_after_norm = cfg.get("k_transform_after_norm", False)  # Apply K=transform(Q) AFTER QK-norm instead of before. Standard: Q->K=f(Q)->norm(Q)->norm(K). Post-norm: Q->norm(Q)->K=f(Q_normed). Transforms operate on unit vectors (more stable for rotations/permutations), K inherits normalized magnitude (no separate K norm needed), and gradients from K don't flow through norm's Q statistics. May help K-tying approaches converge.
        k_proj_dropout_prob = float(cfg.get("k_proj_dropout_prob", 0.0))  # During training, randomly replace K=K_proj(x) with K=Q. Teaches the model to work with both independent and tied K, potentially enabling post-training K_proj removal (52p→40p). Only active when the model has an independent K_proj (not already K-tied). At 0: disabled (backward compatible). Typical: 0.1-0.5.
        layer_noise_scale = float(cfg.get("layer_noise_scale", 0.0))  # Multiplicative noise on sublayer outputs during training: out *= uniform(1-scale, 1+scale). Regularizes against exact magnitude dependence, improving generalization on carry chains. Different from embed_noise (additive, embeddings only), weight_perturb_std (periodic weight noise), and grad_noise_eta (gradient noise). Typical: 0.01-0.1.
        anneal_tying_steps = int(cfg.get("anneal_tying_steps", 0))  # Gradually transition from independent K_proj to tied K over N steps. At step 0: K = K_proj(x). At step N: K = K_tied(Q). Between: linear interpolation. Allows the model to learn good representations with full expressivity first, then adapt to parameter sharing. Critical for getting sub-50p models (with k_perm_q, k_hadamard_q, etc.) to converge. The staging K_proj is excluded from inference param count.
        residual_gate = cfg.get("residual_gate", False)  # per-dimension sigmoid gates on residual connections: x = x + sigmoid(g)*sublayer(x). Unlike scalar residual_alpha, provides per-dimension control (d_model params per gate). For d_model=3, each hidden dimension gets independent gating. Adds 2*d_model params (attn + MLP gates).
        sigmoid_attn = cfg.get("sigmoid_attn", False)  # sigmoid attention instead of softmax
        parallel_block = cfg.get("parallel_block", False)  # PaLM-style parallel attn+MLP (saves ln2 params)
        norm_lerp = cfg.get("norm_lerp", False)  # learnable norm interpolation: (1-t)*norm(x)+t*x
        attn_dropout = float(cfg.get("attn_dropout", 0.0))  # dropout on attention weights during training
        freeze_rot = cfg.get("freeze_rot", False)  # freeze rotation angles (k_rot_q/down_rot_up_t) as non-learnable buffers
        tie_gate_q = cfg.get("tie_gate_q", False)  # gate_proj reuses Q_proj.weight[:ff_dim,:], saves ff_dim*d_model params
        tie_up_q = cfg.get("tie_up_q", False)  # up_proj reuses Q_proj.weight[ff_dim:2*ff_dim,:], saves ff_dim*d_model params
        qk_norm_type = cfg.get("qk_norm_type", "rms")  # "rms" (default), "scalar_rms" (1 param), or "unit" (0 params)
        tie_down_up = cfg.get("tie_down_up", False)  # down_proj = up_proj^T, saves ff_dim*d_model params. Competition winner uses down=rot(up^T).
        tie_up_gate_offset = cfg.get("tie_up_gate_offset", False)  # up(x) = gate_proj(x) + offset (ff_dim params), sharing the gate projection weight. Saves ff_dim*d_model - ff_dim params (e.g., 4p for ff=2, d=3). SwiGLU becomes silu(Wx)*(Wx+b). More expressive than tie_gate (silu(Wx)*silu(Wx), always 0%) because offset breaks gate/up symmetry. With tie_down_gate: 52p→48p.
        self.n_repeats = int(cfg.get("n_repeats", 1))  # Universal Transformer: repeat blocks N times (shared weights)
        self.embed_noise_std = float(cfg.get("embed_noise", 0.0))  # Gaussian noise on embeddings during training
        self.embed_dropout_p = float(cfg.get("embed_dropout", 0.0))  # dropout on embeddings during training
        self.n_think_tokens = int(cfg.get("n_think_tokens", 0))  # scratchpad tokens before output for extra computation
        output_proj = cfg.get("output_proj", False)  # learnable d_model→tok_dim projection before output logits
        # Bidirectional prefix attention: allow full (non-causal) attention within the
        # prompt tokens (first PROMPT_LEN=24 positions). The input format is:
        #   [0] + rev(a,10) + [0,0] + rev(b,10) + [0]
        # All 24 prompt tokens are known before generation starts. With standard causal
        # masking, prompt token 15 (digit 5 of number B) can't attend to token 20 (digit
        # 10 of number B), even though both are fully available. For carry-chain routing,
        # the model needs to see ALL input digits simultaneously — a carry at position 7
        # depends on digits at ALL higher positions. Bidirectional prefix lets the model
        # build a complete representation of both addends before autoregressive generation.
        # Output tokens (positions 24+) still use causal masking. Zero extra parameters.
        self.bidirectional_prefix = bool(cfg.get("bidirectional_prefix", False))
        # Number of refinement passes at inference time. After initial autoregressive
        # generation, feed [prompt + generated_output] back through the model and
        # re-predict each output digit. The model can attend to its own draft answer
        # (via causal masking) and correct carry-chain errors where earlier digits were
        # right but a later digit was wrong. Multiple passes allow iterative refinement.
        # Zero extra parameters, N× inference cost. Only affects generate(), not training.
        self.refine_passes = int(cfg.get("refine_passes", 0))

        # Token embedding
        if embed_type == "circular_arc":
            tok_dim = 2
            self.tok_embed = CircularArcEmbedding(vocab_size, tok_dim)
            pos_dim = d_model - tok_dim
            self.pos_mode = "sinusoidal" if pos_dim > 0 else "none"
        elif embed_type == "learned_phase":
            tok_dim = 2
            self.tok_embed = LearnedPhaseEmbedding(vocab_size, tok_dim)
            pos_dim = d_model - tok_dim
            self.pos_mode = "sinusoidal" if pos_dim > 0 else "none"
        elif embed_type == "quadratic":
            tok_dim = 2
            self.tok_embed = QuadraticEmbedding(vocab_size, tok_dim)
            pos_dim = d_model - tok_dim
            self.pos_mode = "sinusoidal" if pos_dim > 0 else "none"
        else:
            self.tok_embed = LookupEmbedding(vocab_size, d_model)
            pos_dim = 0
            self.pos_mode = "none"

        self.tok_dim = tok_dim if embed_type != "lookup" else d_model
        self.pos_dim = pos_dim if embed_type != "lookup" else 0
        self.d_model = d_model

        # Positional embedding for split-dim models
        if self.pos_mode == "sinusoidal" and pos_dim > 0:
            max_seq = 64
            pe = self._build_sinusoidal_pe(max_seq, pos_dim)
            self.register_buffer("pos_embed", pe)
        elif embed_type == "lookup":
            # Learned or sinusoidal PE added to full embedding
            pe_type = cfg.get("pe_type", "sinusoidal")
            if pe_type == "sinusoidal":
                pe = self._build_sinusoidal_pe(64, d_model)
                self.register_buffer("pos_embed", pe)
                self.pos_mode = "sinusoidal_full"
            elif pe_type == "learned":
                self.pos_embed = nn.Parameter(torch.randn(64, d_model) * 0.02)
                self.pos_mode = "learned"
            else:
                self.pos_mode = "none"

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, n_kv_heads, head_dim, ff_dim,
                rope_theta=rope_theta, qk_norm=qk_norm,
                tie_kv=tie_kv, tie_qo=tie_qo, tie_qk=tie_qk,
                k_rot_q=k_rot_q, v_eq_q=v_eq_q, tie_qk_norm=tie_qk_norm,
                attn_out_rank=attn_out_rank,
                use_swiglu=use_swiglu, tie_gate=tie_gate,
                tie_down_gate=tie_down_gate, down_rot_up_t=down_rot_up_t,
                share_norms=share_norms, norm_type=norm_type,
                use_rope=use_rope, rope_period=rope_period,
                residual_alpha=residual_alpha, attn_temp=attn_temp,
                rope_learnable=rope_learnable,
                multi_angle_rot=multi_angle_rot, rot_init=rot_init,
                k_diag_q=k_diag_q,
                sigmoid_attn=sigmoid_attn, parallel_block=parallel_block,
                k_householder_q=k_householder_q, k_negate_q=k_negate_q,
                norm_lerp=norm_lerp,
                attn_dropout=attn_dropout,
                freeze_rot=freeze_rot,
                k_givens_q=k_givens_q,
                tie_gate_q=tie_gate_q,
                tie_up_q=tie_up_q,
                qk_norm_type=qk_norm_type,
                k_perm_q=k_perm_q,
                k_perm_shift=k_perm_shift,
                tie_down_up=tie_down_up,
                k_signperm_q=k_signperm_q,
                k_cayley_q=k_cayley_q,
                k_hadamard_q=k_hadamard_q,
                anneal_tying_steps=anneal_tying_steps,
                residual_gate=residual_gate,
                k_factored_rank=k_factored_rank,
                k_residual_rank=k_residual_rank,
                layer_noise_scale=layer_noise_scale,
                k_partial_proj_dims=k_partial_proj_dims,
                k_transform_after_norm=k_transform_after_norm,
                k_proj_dropout_prob=k_proj_dropout_prob,
                tie_up_gate_offset=tie_up_gate_offset,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        if norm_type == "none":
            self.ln_f = nn.Identity()
        elif norm_type == "scalar_rms":
            share_ln_f = cfg.get("share_ln_f", share_norms)
            if share_ln_f and n_layers > 0:
                self.ln_f = self.blocks[0].ln1
            else:
                self.ln_f = ScalarRMSNorm()
        elif norm_type == "unit":
            self.ln_f = UnitRMSNorm()
        elif norm_type == "mixed":
            share_ln_f = cfg.get("share_ln_f", share_norms)
            if share_ln_f and n_layers > 0:
                self.ln_f = self.blocks[0].ln1
            else:
                self.ln_f = MixedNorm(d_model)
        else:
            share_ln_f = cfg.get("share_ln_f", share_norms)  # default: follow share_norms
            if share_ln_f and n_layers > 0:
                self.ln_f = self.blocks[0].ln1
            elif norm_lerp:
                self.ln_f = LerpRMSNorm(d_model)
            else:
                self.ln_f = RMSNorm(d_model)

        # Output projection: learnable d_model → tok_dim linear layer before
        # computing logits via the embedding table. With circular_arc (d_model=3,
        # tok_dim=2), the default code discards the 3rd hidden dimension at output
        # time (x[..., :2] @ table.T). This means carry-chain information stored
        # in the positional dimension never influences predictions. The output_proj
        # adds a (d_model, tok_dim) matrix that lets ALL hidden dimensions contribute.
        # Adds d_model * tok_dim params (e.g., 6 for d=3, tok=2).
        self._has_output_proj = False
        if output_proj and embed_type != "lookup":
            _tok_dim = 2  # circular_arc, learned_phase, quadratic all use tok_dim=2
            if _tok_dim < d_model:
                self.output_proj_layer = nn.Linear(d_model, _tok_dim, bias=False)
                self._has_output_proj = True

        # Lightweight output positional mixing: projects the positional dimension
        # (3rd dim for d_model=3 with circular_arc) into the token space (first 2 dims)
        # before computing output logits. Much cheaper than output_proj (2 params vs 6):
        #   x_out = x[..., :tok_dim] + x[..., tok_dim:] @ W_mix
        # where W_mix is (d_model - tok_dim, tok_dim), e.g., (1, 2) for d=3.
        # Initialized to zeros so it defaults to the standard behavior (ignore pos dim).
        # This lets carry-chain information stored in the positional dimension influence
        # the final digit predictions without a full d_model → tok_dim projection.
        self._has_output_pos_mix = False
        if cfg.get("output_pos_mix", False) and embed_type != "lookup" and not self._has_output_proj:
            _tok_dim = 2
            if _tok_dim < d_model:
                self.output_pos_mix_proj = nn.Parameter(torch.zeros(d_model - _tok_dim, _tok_dim))
                self._has_output_pos_mix = True

        # Learned output temperature: learnable scalar applied to output logits
        # at both training and inference time. Helps the model calibrate its confidence
        # on carry-chain predictions. Initialized to temp=1.0 (log_temp=0.0, no effect).
        # 1 extra parameter. Different from attn_temp (which scales attention scores)
        # and train_temp_start/end (which are fixed schedules applied only during training).
        self._learned_output_temp = cfg.get("learned_output_temp", False)
        if self._learned_output_temp:
            self.log_output_temp = nn.Parameter(torch.tensor(0.0))

        # Per-output-position learnable temperature: each of the 11 output digit
        # positions gets its own temperature for logit scaling. Positions involved
        # in carry chains need different calibration than non-carry positions.
        # 11 params (one per output digit). Initialized to log(1.0)=0 (no effect).
        # Different from learned_output_temp (single scalar for all positions)
        # and output_pos_bias (additive bias, 110 params).
        self._per_pos_temp = cfg.get("per_pos_temp", False)
        if self._per_pos_temp:
            self.log_pos_temps = nn.Parameter(torch.zeros(OUTPUT_LEN))

        # Output head
        self.tie_embed = tie_embed
        if not tie_embed:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _build_sinusoidal_pe(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        sin_vals = torch.sin(pos * div)
        cos_vals = torch.cos(pos * div)
        pe[:, 0::2] = sin_vals[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_vals[:, :pe[:, 1::2].shape[1]]
        return pe

    def forward(self, idx, mixup_lam=None, mixup_perm=None):
        B, T = idx.shape

        if self.pos_mode in ("sinusoidal", ):
            # Split embedding: tok + pos
            tok = self.tok_embed(idx)
            pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([tok, pos], dim=-1)
        elif self.pos_mode == "sinusoidal_full":
            x = self.tok_embed(idx) + self.pos_embed[:T].unsqueeze(0)
        elif self.pos_mode == "learned":
            x = self.tok_embed(idx) + self.pos_embed[:T].unsqueeze(0)
        else:
            x = self.tok_embed(idx)

        # Embedding noise: Gaussian noise during training for regularization
        if self.training and self.embed_noise_std > 0:
            x = x + torch.randn_like(x) * self.embed_noise_std

        # Embedding dropout: zero out embedding dimensions for structural regularization
        if self.training and self.embed_dropout_p > 0:
            x = F.dropout(x, p=self.embed_dropout_p, training=True)

        # Embedding mixup: blend two examples' representations for regularization
        # Creates virtual training samples that smooth the loss landscape at convergence
        if mixup_lam is not None and mixup_perm is not None:
            x = mixup_lam * x + (1 - mixup_lam) * x[mixup_perm]

        # Causal mask (avoid 0 * -inf = NaN)
        mask = torch.zeros(T, T, device=x.device)
        mask.masked_fill_(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float("-inf"))
        # Bidirectional prefix: allow full attention within prompt tokens.
        # Prompt tokens (0..PROMPT_LEN-1) can attend to ALL other prompt tokens,
        # while output tokens (PROMPT_LEN+) still use causal masking.
        if self.bidirectional_prefix:
            pl = min(PROMPT_LEN, T)
            mask[:pl, :pl] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)

        for _repeat in range(self.n_repeats):
            for block in self.blocks:
                x = block(x, mask=mask)

        x = self.ln_f(x)

        # Store hidden states for auxiliary heads (e.g., carry prediction)
        self._last_hidden = x

        if self.tie_embed:
            table = self.tok_embed.table()
            if self._has_output_proj:
                # Learned projection: all d_model dims contribute to output
                logits = self.output_proj_layer(x) @ table.T
            elif self._has_output_pos_mix:
                # Lightweight positional mixing: add projected pos dim to tok dims
                x_tok = x[..., :table.shape[1]]
                x_pos = x[..., table.shape[1]:]
                logits = (x_tok + x_pos @ self.output_pos_mix_proj) @ table.T
            elif table.shape[1] < self.d_model:
                # Project only tok dimensions for output
                logits = x[..., :table.shape[1]] @ table.T
            else:
                logits = x @ table.T
        else:
            logits = self.lm_head(x)

        # Learned output temperature: scale logits for calibrated confidence
        if self._learned_output_temp:
            logits = logits / self.log_output_temp.exp()

        # Per-position temperature: scale the last OUTPUT_LEN positions' logits
        # by position-specific temperatures. Only affects training/eval output positions.
        if self._per_pos_temp:
            T_seq = logits.shape[1]
            if T_seq >= OUTPUT_LEN:
                temps = self.log_pos_temps.exp()  # (OUTPUT_LEN,)
                # Apply to last OUTPUT_LEN sequence positions (out-of-place to avoid inplace autograd error)
                scale = torch.ones(T_seq, device=logits.device, dtype=logits.dtype)
                scale[-OUTPUT_LEN:] = temps
                logits = logits / scale.unsqueeze(0).unsqueeze(-1)

        return logits

    def count_params(self):
        """Count unique parameters (after tying).

        Excludes staging parameters from annealed tying (used only during
        training, not at inference). The staging K_proj allows the model to
        train with full expressivity before transitioning to tied weights.
        Also excludes ln2 params when annealed norm sharing is complete
        (_norm_blend_t >= 1.0), since ln2 = ln1 at inference time.
        """
        seen_ids = set()
        staging_ids = set()
        for module in self.modules():
            if hasattr(module, '_staging_k_proj'):
                for p in module._staging_k_proj.parameters():
                    staging_ids.add(id(p))
            if hasattr(module, '_switched_down_proj'):
                for p in module._switched_down_proj.parameters():
                    staging_ids.add(id(p))
        # Exclude ln2 params when annealed norm sharing is complete
        for block in self.blocks:
            if hasattr(block, '_norm_blend_t') and block._norm_blend_t >= 1.0:
                if hasattr(block, 'ln2') and block.ln2 is not block.ln1:
                    for p in block.ln2.parameters():
                        staging_ids.add(id(p))
        # Exclude original ln_f params when progressive schedule shared ln_f → ln1
        if hasattr(self, '_orig_ln_f'):
            for p in self._orig_ln_f.parameters():
                staging_ids.add(id(p))
        total = 0
        for p in self.parameters():
            pid = id(p)
            if pid not in seen_ids and pid not in staging_ids:
                seen_ids.add(pid)
                total += p.numel()
        return total

    @torch.no_grad()
    def generate(self, prompt, temperature=0.0):
        """Autoregressive generation with optional refinement passes.

        Args:
            prompt: (B, T_prompt) input tokens
            temperature: if > 0, sample from softmax(logits/temp) instead of argmax.
                         Values < 1.0 sharpen the distribution (more confident).
                         0.0 (default) = greedy argmax (backward compatible).
        """
        self.eval()
        B, T_prompt = prompt.shape
        seq = prompt.clone()
        total_gen = self.n_think_tokens + OUTPUT_LEN
        for _ in range(total_gen):
            logits = self.forward(seq)
            step_logits = logits[:, -1, :VOCAB_SIZE]
            if temperature > 0:
                probs = F.softmax(step_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tok = step_logits.argmax(dim=-1)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        out_start = T_prompt + self.n_think_tokens
        pred = seq[:, out_start:out_start + OUTPUT_LEN]

        # Self-correction refinement: feed [prompt + draft_output] back through the
        # model and re-predict each output digit. The model can attend to its own draft
        # (via causal masking over the full sequence) and correct carry-chain errors.
        # Each pass: build full teacher-forced sequence using current predictions,
        # run forward, re-extract output digit predictions from output positions.
        for _refine in range(self.refine_passes):
            # Build input as if teacher-forced: prompt + pred[:-1] (shifted right)
            refine_input = torch.cat([prompt, pred[:, :-1]], dim=1)
            if self.n_think_tokens > 0:
                think_toks = torch.zeros(B, self.n_think_tokens, dtype=torch.long, device=prompt.device)
                refine_input = torch.cat([prompt, think_toks, pred[:, :-1]], dim=1)
            logits = self.forward(refine_input)
            # Extract predictions from the output positions
            for d in range(OUTPUT_LEN):
                pos = T_prompt + self.n_think_tokens - 1 + d  # position predicting digit d
                if pos < logits.shape[1]:
                    pred[:, d] = logits[:, pos, :VOCAB_SIZE].argmax(dim=-1)

        return pred

    @torch.no_grad()
    def generate_beam(self, prompt, beam_width=5, length_norm=True):
        """Beam search generation for better accuracy on hard carry chains.

        Args:
            prompt: (B, T_prompt) tensor. B must be 1 for beam search.
            beam_width: number of beams to maintain
            length_norm: normalize log-prob by sequence length
        Returns:
            (1, OUTPUT_LEN) tensor of best beam's tokens
        """
        self.eval()
        assert prompt.shape[0] == 1, "Beam search only supports batch size 1"
        device = prompt.device
        T_prompt = prompt.shape[1]

        # Each beam: (sequence, cumulative_log_prob)
        beams = [(prompt.clone(), 0.0)]

        total_gen = self.n_think_tokens + OUTPUT_LEN
        for pos in range(total_gen):
            all_candidates = []
            for seq, score in beams:
                logits = self.forward(seq)
                log_probs = F.log_softmax(logits[:, -1, :VOCAB_SIZE], dim=-1)  # (1, V)
                topk_lp, topk_idx = log_probs.topk(beam_width, dim=-1)  # (1, beam_width)
                for k in range(beam_width):
                    new_seq = torch.cat([seq, topk_idx[:, k:k+1]], dim=1)
                    new_score = score + topk_lp[0, k].item()
                    all_candidates.append((new_seq, new_score))
            # Keep top beam_width candidates
            if length_norm:
                all_candidates.sort(key=lambda x: x[1] / (pos + 1), reverse=True)
            else:
                all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

        best_seq = beams[0][0]
        out_start = T_prompt + self.n_think_tokens
        return best_seq[:, out_start:out_start + OUTPUT_LEN]


# ---------------------------------------------------------------------------
# Lion Optimizer (Chen et al., 2023)
# ---------------------------------------------------------------------------

class Lion(torch.optim.Optimizer):
    """Lion optimizer: sign-based updates with momentum interpolation.

    Uses sign(interpolate(momentum, gradient)) for updates, which acts as an
    implicit regularizer. Often generalizes better than AdamW on small models.
    Recommended lr: 3-10x lower than AdamW (e.g., 0.001-0.003).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # Update: sign of interpolation between momentum and gradient
                update = exp_avg.lerp(grad, 1 - beta1).sign_()
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                # Momentum update (uses beta2, not beta1)
                exp_avg.lerp_(grad, 1 - beta2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(cfg, device="cuda", results_dir=None):
    """Train the model and return (model, metrics)."""
    if "cuda" in str(device):
        torch.cuda.empty_cache()
    try:
        model = AdderTransformer(cfg).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e) and "cuda" in str(device):
            import time as _time
            print("CUDA OOM on model init, waiting 30s and retrying...")
            torch.cuda.empty_cache()
            _time.sleep(30)
            torch.cuda.empty_cache()
            try:
                model = AdderTransformer(cfg).to(device)
            except RuntimeError as e2:
                if "out of memory" in str(e2):
                    print("CUDA still OOM after retry, falling back to CPU")
                    device = "cpu"
                    model = AdderTransformer(cfg).to(device)
                else:
                    raise
        else:
            raise
    num_params = model.count_params()
    print(f"Model parameters: {num_params}")
    print(f"Config: {cfg}")

    # Training hyperparams
    lr = float(cfg.get("lr", 0.01))
    min_lr = float(cfg.get("min_lr", 1e-5))
    steps = int(cfg.get("steps", 30000))
    batch_size = int(cfg.get("batch_size", 256))
    warmup_steps = int(cfg.get("warmup_steps", 500))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    eval_every = int(cfg.get("eval_every", 1000))
    patience = int(cfg.get("patience", 10000))
    carry_bias = float(cfg.get("carry_bias", 0.0))

    # Online Hard Example Mining: generate ohem_ratio * batch_size samples,
    # keep only the hardest batch_size by loss. Focuses training on examples
    # the model currently fails, especially carry-chain mistakes near convergence.
    ohem_ratio = float(cfg.get("ohem_ratio", 0.0))  # 0 = disabled (default), typical: 2.0-4.0
    ohem_start_step = int(cfg.get("ohem_start_step", 0))  # step at which OHEM activates (0 = from start). Delaying OHEM lets basic carry patterns form before focusing on hard examples.
    ohem_per_digit = bool(cfg.get("ohem_per_digit", False))  # Per-digit OHEM: select hardest examples per digit position then combine. Standard OHEM selects by total loss which may over-represent failures at one digit. Per-digit OHEM ensures every output position gets hard examples, preventing the model from neglecting specific carry-chain positions.

    # OHEM ratio schedule: ramp OHEM intensity during training.
    # Format: "step1:ratio1,step2:ratio2,..." e.g. "0:0.0,50000:1.5,100000:3.0"
    # Linear interpolation between schedule points. Overrides ohem_ratio when set.
    # Start with no OHEM (model learns basics), then ramp up oversampling to focus
    # on hard carry-chain examples as the model improves. More responsive than a
    # fixed ohem_start_step since the ramp is gradual.
    ohem_schedule_str = cfg.get("ohem_schedule", None)
    ohem_schedule = []
    if ohem_schedule_str:
        for part in ohem_schedule_str.split(","):
            s, v = part.strip().split(":")
            ohem_schedule.append((int(s.strip()), float(v.strip())))

    # Soft OHEM: per-sample importance weighting by loss magnitude.
    # Instead of hard top-K selection (OHEM), smoothly upweight hard examples
    # by raising each sample's loss to a power: weight_i = (loss_i / mean_loss)^gamma.
    # More stable than hard OHEM (no binary keep/drop) and computationally cheaper
    # (no extra forward pass). Can be combined with hard OHEM for compounding effect.
    soft_ohem_gamma = float(cfg.get("soft_ohem_gamma", 0.0))  # 0 = disabled (default), typical 1.0-3.0

    # Carry-weighted OHEM: bias OHEM selection toward examples with more carry digits.
    # The last ~1% of errors are almost always carry-chain failures. Standard OHEM selects
    # by raw loss, which may include non-carry errors. Carry-weighted OHEM multiplies each
    # sample's OHEM selection score by (1 + weight * carry_fraction), so carry-heavy problems
    # are more likely to be kept even if their raw loss is slightly lower than non-carry errors.
    # Only active when OHEM is also active (ohem_ratio > 1.0).
    ohem_carry_weight = float(cfg.get("ohem_carry_weight", 0.0))  # 0 = disabled (default), typical: 1.0-5.0

    # Accuracy-adaptive OHEM schedule: dynamically adjust OHEM ratio based on
    # current best accuracy. Format: "acc1:ratio1,acc2:ratio2,..." e.g.
    # "0.0:0.0,0.3:1.5,0.5:2.0,0.65:3.0". Linear interpolation between points.
    # Takes priority over ohem_ratio and ohem_schedule when set. Since OHEM was
    # the single biggest accuracy boost (70%→99.45%), coupling its intensity to
    # learning progress avoids wasting early training on hard-example mining
    # (when the model can't even do easy problems) and automatically ramps up
    # oversampling as the model improves and needs harder examples to progress.
    ohem_acc_schedule_str = cfg.get("ohem_acc_schedule", None)
    ohem_acc_schedule = []
    if ohem_acc_schedule_str:
        for part in ohem_acc_schedule_str.split(","):
            a, v = part.strip().split(":")
            ohem_acc_schedule.append((float(a.strip()), float(v.strip())))

    # Curriculum
    curriculum = cfg.get("curriculum", None)
    # Format: "digits:steps,digits:steps,..." e.g. "3:2000,6:5000,10:rest"

    # Per-parameter-group learning rate multipliers: different LR for each model component.
    # Tiny models have components that saturate at different rates — embedding learns fast,
    # attention patterns take longer, norms need stability. Tuning per-group LR can break plateaus.
    lr_mult_embed = float(cfg.get("lr_mult_embed", 1.0))  # embedding LR multiplier
    lr_mult_attn = float(cfg.get("lr_mult_attn", 1.0))  # attention LR multiplier
    lr_mult_mlp = float(cfg.get("lr_mult_mlp", 1.0))  # MLP LR multiplier
    lr_mult_norm = float(cfg.get("lr_mult_norm", 1.0))  # norm LR multiplier
    _use_lr_groups = any(m != 1.0 for m in [lr_mult_embed, lr_mult_attn, lr_mult_mlp, lr_mult_norm])

    def _build_param_groups(model_params, base_lr, wd):
        """Build optimizer parameter groups with per-component LR multipliers."""
        groups = {'embed': [], 'attn': [], 'mlp': [], 'norm': [], 'other': []}
        for name, p in model_params:
            if 'tok_embed' in name or 'pos_embed' in name:
                groups['embed'].append(p)
            elif '.attn.' in name:
                groups['attn'].append(p)
            elif '.mlp.' in name:
                groups['mlp'].append(p)
            elif 'ln' in name or 'norm' in name:
                groups['norm'].append(p)
            else:
                groups['other'].append(p)
        mults = {'embed': lr_mult_embed, 'attn': lr_mult_attn, 'mlp': lr_mult_mlp,
                 'norm': lr_mult_norm, 'other': 1.0}
        return [{'params': groups[k], 'lr': base_lr * mults[k], 'weight_decay': wd,
                 'lr_mult': mults[k]} for k in groups if groups[k]]

    # Optimizer
    opt_name = cfg.get("optimizer", "adamw")
    lion_beta1 = float(cfg.get("lion_beta1", 0.9))
    lion_beta2 = float(cfg.get("lion_beta2", 0.99))
    # Configurable Adam/AdamW betas: tune momentum parameters for tiny models.
    # Lower beta2 (e.g., 0.95) helps escape plateaus where the adaptive LR denominator
    # becomes too stable, effectively increasing the learning rate for stale parameters.
    adam_beta1 = float(cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(cfg.get("adam_beta2", 0.999))
    if _use_lr_groups:
        _pg = _build_param_groups(model.named_parameters(), lr, weight_decay)
        if opt_name == "adam":
            for g in _pg: g['weight_decay'] = 0
            optimizer = torch.optim.Adam(_pg, lr=lr, betas=(adam_beta1, adam_beta2))
        elif opt_name == "lion":
            optimizer = Lion(_pg, lr=lr, betas=(lion_beta1, lion_beta2))
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(_pg, lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(_pg, lr=lr, betas=(adam_beta1, adam_beta2))
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0, betas=(adam_beta1, adam_beta2))
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2))
    elif opt_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr, betas=(lion_beta1, lion_beta2), weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2))

    # LR schedule
    lr_restart_period = int(cfg.get("lr_restart_period", 0))  # 0 = single cosine (default)
    lr_restart_mult = float(cfg.get("lr_restart_mult", 1.0))  # period multiplier after each restart

    # LR schedule type: "cosine" (default) or "wsd" (warmup-stable-decay)
    # WSD holds peak LR for stable_fraction of training, then cosine decays.
    # Tiny models benefit from more time at peak LR to learn carry patterns.
    lr_schedule = cfg.get("lr_schedule", "cosine")  # "cosine" (default) or "wsd"
    stable_fraction = float(cfg.get("stable_fraction", 0.5))  # fraction of post-warmup steps at peak LR (wsd only)

    # Periodic optimizer state reset: clear Adam/AdamW momentum and variance buffers
    # every N steps. Stale adaptive statistics in tiny models can lock the optimizer
    # into suboptimal trajectories, especially after curriculum transitions or when
    # the loss landscape changes significantly. Reset gives fresh exploration.
    opt_reset_interval = int(cfg.get("opt_reset_interval", 0))  # 0 = disabled (default), typical 20000-50000

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        if lr_schedule == "wsd":
            # Warmup-Stable-Decay: constant peak LR then cosine decay
            stable_end = warmup_steps + int((steps - warmup_steps) * stable_fraction)
            if step < stable_end:
                return lr
            decay_progress = (step - stable_end) / max(steps - stable_end, 1)
            return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * decay_progress))
        t = step - warmup_steps
        total = max(steps - warmup_steps, 1)
        if lr_restart_period > 0:
            # Cosine annealing with warm restarts (SGDR)
            period = lr_restart_period
            while t >= period and period < total:
                t -= period
                period = int(period * lr_restart_mult)
            progress = t / max(period, 1)
        else:
            progress = t / total
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

    # Parse curriculum
    curriculum_stages = []
    if curriculum:
        for part in curriculum.split(","):
            part = part.strip()
            d, s = part.split(":")
            curriculum_stages.append((int(d), s.strip()))

    # Curriculum smooth transitions: instead of hard cutoffs between digit stages,
    # mix problems from old and new difficulty over N steps. Avoids loss spikes at
    # transitions that can destabilize tiny models and waste training steps.
    curriculum_smooth_steps = int(cfg.get("curriculum_smooth_steps", 0))  # 0 = hard transitions (backward compatible)

    # Curriculum recycling: after the curriculum reaches full 10-digit difficulty,
    # periodically restart from easy problems. Every N steps, spend a short window
    # re-training on 3-digit then 6-digit problems before returning to 10-digit.
    # This reinforces basic carry-chain patterns that can degrade during long training
    # on hard problems, and gives the model multiple "grokking windows" where it can
    # rediscover efficient carry representations from first principles. The recycling
    # only activates after the main curriculum has completed (reached "rest" stage).
    curriculum_recycle_period = int(cfg.get("curriculum_recycle_period", 0))  # 0 = disabled (default), typical: 20000-50000
    _recycle_3d_steps = int(cfg.get("curriculum_recycle_3d_steps", 500))  # steps on 3-digit problems per cycle
    _recycle_6d_steps = int(cfg.get("curriculum_recycle_6d_steps", 1000))  # steps on 6-digit problems per cycle

    def get_max_digits(step):
        if not curriculum_stages:
            return NUM_DIGITS
        for digits, step_str in curriculum_stages:
            if step_str == "rest":
                return digits
            if step < int(step_str):
                return digits
        return NUM_DIGITS

    def get_max_digits_smooth(step):
        """Return (primary_digits, secondary_digits, mix_prob) for smooth transitions.
        mix_prob is the probability of using secondary_digits (the previous stage)."""
        if not curriculum_stages or curriculum_smooth_steps <= 0:
            return get_max_digits(step), get_max_digits(step), 0.0
        prev_digits = curriculum_stages[0][0]
        for i, (digits, step_str) in enumerate(curriculum_stages):
            if step_str == "rest":
                # Check if we just transitioned into this final stage
                if i > 0:
                    prev_step_str = curriculum_stages[i - 1][1]
                    if prev_step_str != "rest":
                        transition_step = int(prev_step_str)
                        if transition_step <= step < transition_step + curriculum_smooth_steps:
                            elapsed = step - transition_step
                            mix_prob = 1.0 - elapsed / curriculum_smooth_steps
                            return digits, curriculum_stages[i - 1][0], mix_prob
                return digits, digits, 0.0
            threshold = int(step_str)
            if step < threshold:
                # Check if we're in a smooth transition zone
                if i > 0:
                    prev_step_str = curriculum_stages[i - 1][1]
                    if prev_step_str != "rest":
                        prev_threshold = int(prev_step_str)
                        if prev_threshold <= step < prev_threshold + curriculum_smooth_steps:
                            elapsed = step - prev_threshold
                            mix_prob = 1.0 - elapsed / curriculum_smooth_steps
                            return digits, curriculum_stages[i - 1][0], mix_prob
                return digits, digits, 0.0
            prev_digits = digits
        return NUM_DIGITS, NUM_DIGITS, 0.0

    # Time limit (graceful stop before orze kills the process)
    # Default to 27000s (orze kills at 28800s) so experiments exit cleanly instead of being killed
    time_limit = float(cfg.get("time_limit", 27000))

    # Grokfast-EMA: gradient filter to accelerate grokking
    grokfast_alpha = float(cfg.get("grokfast_alpha", 0))
    grokfast_lambda = float(cfg.get("grokfast_lambda", 0))

    # Adaptive grokfast: dynamically scale per-parameter grokfast lambda based on
    # cosine similarity between the gradient EMA (slow-moving direction) and the
    # current gradient. When they align (model is grokking consistently), amplify
    # more. When they diverge (model is oscillating or transitioning), reduce
    # amplification. This auto-tunes grokfast intensity per parameter:
    #   lambda_p = lambda * (0.5 + 0.5 * cos_sim(ema_p, grad_p))
    # Scalar params (1-element) skip alignment check (always use full lambda).
    grokfast_adaptive = bool(cfg.get("grokfast_adaptive", False))

    # Weight decay schedule: dynamically adjust WD during training.
    # Higher WD accelerates grokking (generalization after memorization).
    # Format: "step1:wd1,step2:wd2,..." e.g. "0:0.001,50000:0.01,150000:0.05"
    # Linear interpolation between points. Overrides weight_decay when set.
    wd_schedule_str = cfg.get("wd_schedule", None)
    wd_schedule = []
    if wd_schedule_str:
        for part in wd_schedule_str.split(","):
            s, v = part.strip().split(":")
            wd_schedule.append((int(s.strip()), float(v.strip())))

    # Gradient noise injection: helps escape saddle points in tiny models
    grad_noise_eta = float(cfg.get("grad_noise_eta", 0))  # 0 = disabled (default)
    grad_noise_gamma = float(cfg.get("grad_noise_gamma", 0.55))  # decay exponent

    # Stochastic Weight Averaging: averages weights for better generalization
    swa_start_frac = float(cfg.get("swa_start_frac", 0))  # 0 = disabled (default)
    swa_update_freq = int(cfg.get("swa_update_freq", 100))  # steps between SWA updates
    swa_state = None
    swa_count = 0

    # Label smoothing: reduces overconfidence, helps generalization on carry digits
    label_smoothing = float(cfg.get("label_smoothing", 0.0))  # 0 = disabled (default)

    # Focal loss: down-weight easy examples, focus on hard ones (carry-heavy)
    # gamma=0 is standard CE, gamma=2 is typical focal loss
    focal_gamma = float(cfg.get("focal_gamma", 0.0))  # 0 = disabled (default)

    # Polynomial cross-entropy loss (Leng et al. 2022): adds a polynomial correction
    # to standard CE that maintains strong gradient signal on borderline predictions.
    # L = CE(p, y) + epsilon * mean((1 - p_y)^degree)
    # where p_y is the predicted probability for the true class. Unlike focal loss
    # (which DOWN-weights easy examples), poly-CE ADDS extra gradient for hard examples
    # without reducing gradient for easy ones. This is especially effective near convergence
    # (99%+ accuracy) where most digits are confidently correct but a few carry-chain
    # digits are uncertain. The polynomial term specifically targets these borderline
    # predictions. Can be combined with OHEM for compounding effect.
    poly_epsilon = float(cfg.get("poly_epsilon", 0.0))  # 0 = disabled (default), typical: 0.5-2.0
    poly_degree = int(cfg.get("poly_degree", 2))  # polynomial degree, typical: 1-3

    # Carry bias schedule: ramp carry_bias during training
    # Format: "step1:val1,step2:val2,..." e.g. "0:0.0,50000:0.3,100000:0.6"
    carry_bias_schedule_str = cfg.get("carry_bias_schedule", None)
    carry_bias_schedule = []
    if carry_bias_schedule_str:
        for part in carry_bias_schedule_str.split(","):
            s, v = part.strip().split(":")
            carry_bias_schedule.append((int(s.strip()), float(v.strip())))

    # Majority-vote inference: run N forward passes and majority-vote per digit
    # Only used at final evaluation, not during training
    majority_vote_n = int(cfg.get("majority_vote_n", 0))  # 0 = disabled (default)

    # Per-digit loss weighting: upweight later (harder) output digits
    # Higher digits depend on long carry chains and are harder to learn
    digit_loss_weight = cfg.get("digit_loss_weight", None)  # e.g. "linear" or "exponential" or None
    digit_loss_scale = float(cfg.get("digit_loss_scale", 2.0))  # max weight for last digit

    # Adaptive per-digit loss weighting: dynamically upweight loss for digit positions
    # the model is currently struggling with. Tracks exponential moving average of
    # per-digit training accuracy and uses (1 - acc) * scale + 1 as weights. Unlike
    # static digit_loss_weight, this adapts to the model's actual performance, auto-
    # allocating capacity to hard positions (carry-heavy digits). Overrides
    # digit_loss_weight when enabled.
    adaptive_digit_weight = float(cfg.get("adaptive_digit_weight", 0.0))  # 0 = disabled (default), typical 1.0-5.0
    adaptive_digit_ema_rate = float(cfg.get("adaptive_digit_ema", 0.99))  # EMA smoothing rate
    _digit_acc_ema = None  # running per-digit accuracy tracker

    # Sharpness-Aware Minimization (SAM): seek flat minima for better generalization
    # Foret et al. 2020 — computes gradient at perturbed weights, then updates at original weights
    sam_rho = float(cfg.get("sam_rho", 0.0))  # 0 = disabled (default), typical: 0.05
    sam_adaptive = bool(cfg.get("sam_adaptive", False))  # adaptive SAM scales rho per-param

    # R-Drop regularization: KL divergence between two forward passes
    # Forces consistent predictions, helps generalization on carry-heavy problems
    rdrop_alpha = float(cfg.get("rdrop_alpha", 0.0))  # 0 = disabled (default), typical: 0.1-1.0

    # Lookahead optimizer wrapper: maintains slow weights updated every k steps
    # Zhang et al. 2019 — stabilizes training, reduces variance
    lookahead_k = int(cfg.get("lookahead_k", 0))  # 0 = disabled (default), typical: 5-10
    lookahead_alpha = float(cfg.get("lookahead_alpha", 0.5))  # interpolation rate
    lookahead_state = None
    lookahead_counter = 0

    # Gradient accumulation: effective_batch = batch_size * grad_accum_steps
    # Allows larger effective batch size without more memory, stabilizes training
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))  # 1 = no accumulation (default)

    # Curriculum mixing: after reaching max digits, mix in easier problems
    # Prevents catastrophic forgetting of lower-digit carry patterns
    curriculum_mix_prob = float(cfg.get("curriculum_mix_prob", 0.0))  # 0 = disabled (default)
    curriculum_mix_digits = int(cfg.get("curriculum_mix_digits", 6))  # digit count for mixed-in samples

    # EMA of model weights: smoother convergence than SWA
    ema_decay = float(cfg.get("ema_decay", 0.0))  # 0 = disabled (default), typical: 0.999
    ema_start_step = int(cfg.get("ema_start_step", 0))  # step to start EMA tracking
    ema_state = None

    # Auxiliary carry prediction loss: supervise carry bits alongside digit prediction
    # Adds a small linear head that predicts carry[i] from the same hidden states
    # This gives the model explicit gradient signal about carry chain propagation
    aux_carry_weight = float(cfg.get("aux_carry_weight", 0.0))  # 0 = disabled (default), typical: 0.1-0.5
    aux_carry_head = None
    if aux_carry_weight > 0:
        aux_carry_head = nn.Linear(cfg.get("d_model", 3), 2, bias=False).to(
            next(model.parameters()).device
        )
        # Add carry head params to the optimizer
        optimizer.add_param_group({"params": aux_carry_head.parameters(), "lr": lr, "weight_decay": 0})

    # Checkpoint soup: save top-K checkpoints and average them at the end
    # Unlike SWA (which averages sequentially), this picks the K best checkpoints
    # by accuracy and averages them, producing a smoother solution near the optimum
    ckpt_soup_k = int(cfg.get("ckpt_soup_k", 0))  # 0 = disabled (default), typical: 3-5
    ckpt_soup = []  # list of (accuracy, state_dict) tuples, sorted by accuracy

    # Custom weight initialization: critical for ultra-compressed models
    # Default Kaiming init can be too large/small for sub-64-param transformers
    init_scale = float(cfg.get("init_scale", 0.0))  # 0 = use default init (backward compatible)
    init_type = cfg.get("init_type", "default")  # "default", "orthogonal", "xavier"
    if init_type == "orthogonal":
        # Orthogonal init preserves gradient magnitudes through the network.
        # Critical for ultra-compressed models where gradient vanishing/exploding
        # in a single layer can kill training. Saxe et al. 2014.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    nn.init.orthogonal_(p)
    elif init_type == "xavier":
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
    elif init_scale > 0:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:  # weight matrices
                    nn.init.normal_(p, mean=0.0, std=init_scale)
                elif 'weight' in name and p.dim() == 1:
                    pass  # leave norm weights at 1.0

    # Tied init scaling: for heavily-tied models, Q_proj receives gradients from
    # Q, K (via rotation/perm), V (if v_eq_q), and O (if tie_qo) — up to 4 paths.
    # This causes gradient explosion at init. tie_init_scale divides Q_proj init
    # by sqrt(N) where N = number of sharing paths, stabilizing early training.
    # Critical for sub-50p models with many tying options enabled.
    tie_init_scale = cfg.get("tie_init_scale", False)
    if tie_init_scale:
        with torch.no_grad():
            # Count how many paths share Q_proj
            n_paths = 1  # Q always uses Q_proj
            _has_k_tying = any(cfg.get(k, False) for k in [
                "k_rot_q", "k_cayley_q", "k_givens_q", "k_perm_q",
                "k_signperm_q", "k_hadamard_q", "k_householder_q",
                "k_negate_q", "k_diag_q", "tie_qk",
            ])
            if _has_k_tying and cfg.get("tie_kv", False):
                n_paths += 2  # K and V both derive from Q
            elif _has_k_tying:
                n_paths += 1  # K derives from Q
            elif cfg.get("v_eq_q", False):
                n_paths += 1  # V derives from Q directly
            if cfg.get("tie_qo", False):
                n_paths += 1  # O = Q^T
            if cfg.get("tie_gate_q", False):
                n_paths += 1  # gate reuses Q rows
            if cfg.get("tie_up_q", False):
                n_paths += 1  # up reuses Q rows
            if n_paths > 1:
                scale = 1.0 / math.sqrt(n_paths)
                for name, p in model.named_parameters():
                    if 'q_proj' in name and p.dim() >= 2:
                        p.mul_(scale)
                print(f"Tied init scale: Q_proj scaled by 1/sqrt({n_paths}) = {scale:.3f}")

    # Warm start: initialize model weights from a previously trained experiment's checkpoint.
    # Enables "train large, then compress" — train a 52p model to high accuracy, then create
    # a smaller config (e.g., 40p with k_perm_q) and warm-start from the 52p weights.
    # Only loads weights that match in name AND shape; mismatched weights keep their init.
    warm_start_from = cfg.get("warm_start_from", None)  # idea_id to load checkpoint from
    # Warm-start gradient scaling: reduce LR for loaded (pre-trained) weights to protect
    # learned representations while new/mismatched weights (from changed tying) adapt.
    # Critical for "train large, compress" workflows where warm-starting a 36p model
    # from a 52p checkpoint: the loaded Q_proj/gate/up weights are good, but the new
    # K=perm(Q) and V=Q paths produce different gradients that can corrupt them.
    warm_start_lr_mult = float(cfg.get("warm_start_lr_mult", 1.0))  # LR multiplier for loaded params (typical: 0.1-0.5, 1.0 = no scaling)
    warm_start_thaw_steps = int(cfg.get("warm_start_thaw_steps", 0))  # steps before loaded params get full LR (0 = scale forever)
    _warm_loaded_names = set()
    if warm_start_from and results_dir:
        _ws_ckpt_path = Path(results_dir) / warm_start_from / "checkpoint.pt"
        if _ws_ckpt_path.exists():
            _ws_ckpt = torch.load(_ws_ckpt_path, map_location=device, weights_only=True)
            _ws_src = _ws_ckpt["state_dict"]
            _ws_tgt = model.state_dict()
            _ws_loaded = 0
            for _ws_k in _ws_tgt:
                if _ws_k in _ws_src and _ws_tgt[_ws_k].shape == _ws_src[_ws_k].shape:
                    _ws_tgt[_ws_k] = _ws_src[_ws_k]
                    _ws_loaded += 1
                    _warm_loaded_names.add(_ws_k)
            model.load_state_dict(_ws_tgt)
            # Tag loaded parameters for gradient scaling
            if warm_start_lr_mult != 1.0:
                for _ws_name, _ws_p in model.named_parameters():
                    if _ws_name in _warm_loaded_names:
                        _ws_p._warm_loaded = True
            print(f"Warm start: loaded {_ws_loaded}/{len(_ws_tgt)} weights from {warm_start_from}")
        else:
            print(f"Warm start: checkpoint not found at {_ws_ckpt_path}, using random init")

    # Checkpoint interpolation: blend current model weights with another checkpoint
    interpolate_from = cfg.get("interpolate_from", None)
    interpolate_alpha = float(cfg.get("interpolate_alpha", 0.5))
    if interpolate_from and results_dir:
        _interp_ckpt_path = Path(results_dir) / interpolate_from / "checkpoint.pt"
        if _interp_ckpt_path.exists():
            _interp_ckpt = torch.load(_interp_ckpt_path, map_location=device, weights_only=True)
            _interp_src = _interp_ckpt["state_dict"]
            _interp_tgt = model.state_dict()
            _interp_count = 0
            alpha = interpolate_alpha
            with torch.no_grad():
                for _ik in _interp_tgt:
                    if _ik in _interp_src and _interp_tgt[_ik].shape == _interp_src[_ik].shape:
                        _interp_tgt[_ik] = alpha * _interp_tgt[_ik] + (1 - alpha) * _interp_src[_ik]
                        _interp_count += 1
            model.load_state_dict(_interp_tgt)
            print(f"Checkpoint interpolation: blended {_interp_count} params (alpha={alpha}) with {interpolate_from}")
        else:
            print(f"Checkpoint interpolation: checkpoint not found at {_interp_ckpt_path}, skipping")

    # 1-Cycle LR policy (Smith & Topin 2018): ramp up then ramp down aggressively
    # Known to achieve super-convergence in small models
    # Overrides the default cosine schedule when enabled
    one_cycle = bool(cfg.get("one_cycle", False))  # False = disabled (backward compatible)
    one_cycle_pct_start = float(cfg.get("one_cycle_pct_start", 0.3))  # fraction of steps for warmup phase
    one_cycle_max_lr = float(cfg.get("one_cycle_max_lr", 0.0))  # 0 = use lr (default)
    one_cycle_div = float(cfg.get("one_cycle_div", 25.0))  # initial_lr = max_lr / div
    one_cycle_final_div = float(cfg.get("one_cycle_final_div", 1e4))  # final_lr = max_lr / final_div

    if one_cycle:
        _oc_max = one_cycle_max_lr if one_cycle_max_lr > 0 else lr
        _oc_init = _oc_max / one_cycle_div
        _oc_final = _oc_max / one_cycle_final_div
        _oc_up_steps = int(steps * one_cycle_pct_start)
        _oc_down_steps = steps - _oc_up_steps

        def get_lr_one_cycle(step):
            if step <= _oc_up_steps:
                # Ramp up: init -> max
                frac = step / max(_oc_up_steps, 1)
                return _oc_init + frac * (_oc_max - _oc_init)
            else:
                # Cosine anneal down: max -> final
                t = step - _oc_up_steps
                progress = t / max(_oc_down_steps, 1)
                return _oc_final + 0.5 * (_oc_max - _oc_final) * (1 + math.cos(math.pi * progress))

    # Per-position output bias: learnable bias per digit position and vocab token
    # Helps the model learn position-specific corrections (e.g., carry digits are harder at certain positions)
    output_pos_bias = bool(cfg.get("output_pos_bias", False))  # False = disabled (backward compatible)
    pos_bias_layer = None
    if output_pos_bias:
        # Small learnable bias: (OUTPUT_LEN, VOCAB_SIZE), adds ~110 params but helps position-specific accuracy
        pos_bias_layer = nn.Parameter(torch.zeros(OUTPUT_LEN, VOCAB_SIZE, device=device))
        optimizer.add_param_group({"params": [pos_bias_layer], "lr": lr, "weight_decay": 0})

    # Commutative augmentation: randomly swap a and b in training samples
    # Since a+b = b+a, this halves the effective problem space
    commutative_aug = bool(cfg.get("commutative_aug", False))  # False = disabled (backward compatible)

    # Carry-weighted loss: upweight loss on output digits that have incoming carries
    # The last 2% of errors are almost always carry-chain mistakes; this focuses
    # the model's limited capacity on getting carries right
    carry_loss_weight = float(cfg.get("carry_loss_weight", 0.0))  # 0 = disabled (default), typical: 1.0-3.0

    # Carry-chain-length-weighted loss: upweight examples proportionally to the length
    # of their longest consecutive carry chain. At 99%+ accuracy, nearly all remaining
    # errors are long carry chain failures (e.g., 999...9 + 1 patterns). Standard CE
    # treats all examples equally regardless of carry complexity. carry_loss_weight
    # upweights individual carry DIGITS, but this upweights entire EXAMPLES based on
    # their carry chain LENGTH — a harder example with 8 consecutive carries gets much
    # more weight than one with 2 scattered carries. Per-example weight:
    #   w = 1 + carry_chain_length_weight * (max_chain_length / OUTPUT_LEN)
    # This creates a smooth difficulty gradient: no-carry examples get weight 1.0,
    # max-chain examples get weight 1.0 + carry_chain_length_weight.
    carry_chain_length_weight = float(cfg.get("carry_chain_length_weight", 0.0))  # 0 = disabled (default), typical: 1.0-5.0

    # Top-K digit loss focusing: only compute loss on the K digit positions with the
    # highest per-digit loss within each example. At 99%+ accuracy, most digit positions
    # are correct (near-zero loss) and the few failing carry-chain positions are drowned
    # out. Standard CE averages over all 11 positions, diluting gradient from failures.
    # With loss_digit_top_k=3, gradient comes ONLY from the 3 worst positions per example,
    # concentrating learning capacity on actual failures. Different from:
    # - adaptive_digit_weight (reweights all positions, still includes easy ones)
    # - minimax_digit_weight (blends mean/max, still includes all positions)
    # - OHEM (selects hard EXAMPLES, not hard POSITIONS within each example)
    # K=0 means all positions (disabled, backward compatible). Typical: 3-5.
    loss_digit_top_k = int(cfg.get("loss_digit_top_k", 0))  # 0 = all positions (default), typical: 3-5

    # Margin/hinge loss: penalize when the correct digit's logit isn't sufficiently above
    # the best incorrect digit's logit. At 99%+ accuracy, remaining carry-chain errors are
    # at narrow decision boundaries where the model's logits barely favor the wrong digit.
    # Standard CE provides weak gradient when the correct class has moderate probability
    # (e.g., 40% correct vs 35% incorrect). Margin loss directly targets the LOGIT GAP:
    #   loss += weight * mean(max(0, margin - (logit_correct - max_logit_incorrect)))
    # This forces the model to push correct digits well above competitors, hardening
    # decision boundaries. Different from focal loss (which downweights easy tokens by
    # probability) and confident_wrong_penalty (which upweights high-confidence errors).
    # Margin loss operates in logit space and targets ALL borderline predictions, not just
    # wrong ones — it also strengthens barely-correct predictions to prevent future regression.
    # Particularly effective combined with OHEM: OHEM selects hard EXAMPLES, margin loss
    # sharpens hard DIGIT POSITIONS within those examples.
    margin_loss_weight = float(cfg.get("margin_loss_weight", 0.0))  # 0 = disabled (default), typical: 0.1-1.0
    margin_loss_margin = float(cfg.get("margin_loss_margin", 2.0))  # minimum logit gap, typical: 1.0-5.0

    # Loss feature accuracy gate: delay activation of auxiliary loss features until the
    # model reaches a minimum accuracy threshold. Many auxiliary loss features (OHEM,
    # carry_chain_length_weight, loss_digit_top_k, soft_ohem, adaptive_digit_weight,
    # margin_loss) can disrupt early training when applied from step 0 — the model needs
    # to first learn basic digit patterns before focusing on hard examples. This provides
    # a unified gate: all auxiliary loss features remain disabled until best_accuracy >=
    # loss_feature_min_acc. Once the gate opens, features activate at their configured
    # intensity. Different from ohem_start_step (step-based, OHEM-only): this is accuracy-
    # based and gates ALL auxiliary loss features. Typical: 0.01-0.1 (let model learn basic
    # patterns first, then enable loss focusing).
    loss_feature_min_acc = float(cfg.get("loss_feature_min_acc", 0.0))  # 0 = no gate (default)

    # OHEM warmup ramp: after OHEM activates (via ohem_start_step or loss_feature_min_acc),
    # linearly ramp the effective OHEM ratio from 1.0 (no oversampling) to the target ratio
    # over N steps. The abrupt switch from standard CE to full OHEM can destabilize training
    # because the loss distribution changes dramatically — suddenly every batch is dominated
    # by the hardest carry-chain examples. Ramping smoothly lets the model gradually adapt
    # to harder problem distributions. Different from ohem_schedule (which requires manual
    # step:ratio pairs) and ohem_cosine_period (which modulates cyclically). This is a
    # one-time ramp that works with any OHEM configuration.
    ohem_warmup_steps = int(cfg.get("ohem_warmup_steps", 0))  # 0 = no warmup (default), typical: 5000-20000
    _ohem_warmup_start_step = None  # tracks when OHEM first activated for warmup ramp

    # Cyclical grokfast: periodically reset the grokfast EMA to prevent stale
    # gradient amplification in late training. After many steps, the EMA becomes
    # a stale average that may point away from the current loss landscape
    grokfast_cycle_period = int(cfg.get("grokfast_cycle_period", 0))  # 0 = no cycling (default)

    # Grokfast lambda schedule: ramp grokfast_lambda during training
    # Format: "step1:val1,step2:val2,..." e.g. "0:0.0,5000:1.0,100000:3.0"
    # Linear interpolation between schedule points. Overrides grokfast_lambda when set.
    # Allows starting with weak grokfast (learning basic patterns) and ramping up
    # (amplifying slow-moving gradient components for grokking on hard carry chains).
    grokfast_lambda_schedule_str = cfg.get("grokfast_lambda_schedule", None)
    grokfast_lambda_schedule = []
    if grokfast_lambda_schedule_str:
        for part in grokfast_lambda_schedule_str.split(","):
            s, v = part.strip().split(":")
            grokfast_lambda_schedule.append((int(s.strip()), float(v.strip())))

    # Accuracy-based grokfast schedule: ramp grokfast_lambda based on current best accuracy
    # Format: "acc1:lambda1,acc2:lambda2,..." e.g. "0.0:1.0,0.5:3.0,0.7:5.0"
    # Linear interpolation between accuracy milestones. Couples grokfast intensity
    # to actual learning progress rather than step count — models that learn faster
    # get stronger grokfast sooner, while struggling models aren't overwhelmed.
    # Takes priority over grokfast_lambda and grokfast_lambda_schedule when set.
    grokfast_acc_schedule_str = cfg.get("grokfast_acc_schedule", None)
    grokfast_acc_schedule = []
    if grokfast_acc_schedule_str:
        for part in grokfast_acc_schedule_str.split(","):
            a, v = part.strip().split(":")
            grokfast_acc_schedule.append((float(a.strip()), float(v.strip())))

    # Curriculum LR restart: briefly rewarm LR when curriculum transitions to harder digits
    # The loss landscape changes dramatically at digit transitions; a short warmup helps
    # the optimizer adapt to the new distribution instead of overshooting
    curriculum_lr_restart = bool(cfg.get("curriculum_lr_restart", False))  # False = disabled (backward compatible)
    curriculum_lr_restart_steps = int(cfg.get("curriculum_lr_restart_steps", 500))  # warmup length after transition
    _prev_max_digits = None  # track curriculum transitions
    _restart_step = None  # step when last transition happened

    # Gradient centralization: subtract mean from gradients before update
    # Yong et al. 2020 — constrains weight updates to hyperplane, improves generalization
    # Zero cost (no extra params), especially effective for tiny models
    grad_centralization = bool(cfg.get("grad_centralization", False))  # False = disabled (backward compatible)

    # Training logit temperature annealing: start training with soft logits (high temp),
    # anneal to sharp logits (temp=1). Acts as self-distillation — early training explores
    # more of the output space, late training commits to sharp predictions.
    # This is applied to the TRAINING loss computation, not inference.
    train_temp_start = float(cfg.get("train_temp_start", 0.0))  # 0 = disabled (backward compatible)
    train_temp_end = float(cfg.get("train_temp_end", 1.0))  # final temperature (1.0 = standard CE)

    # Self-distillation from snapshot: periodically save model as "teacher" and add
    # KL-divergence loss between current predictions and teacher's predictions.
    # This prevents the model from drifting too far from good intermediate solutions,
    # smooths the loss landscape, and acts as a regularizer near convergence.
    # Born-Again Networks (Furlanello et al. 2018) showed self-distillation improves even
    # when teacher and student have identical architecture.
    self_distill_alpha = float(cfg.get("self_distill_alpha", 0.0))  # 0 = disabled (default), typical: 0.1-1.0
    self_distill_interval = int(cfg.get("self_distill_interval", 10000))  # steps between teacher snapshots
    self_distill_start_step = int(cfg.get("self_distill_start_step", 0))  # step to start distillation
    self_distill_temp = float(cfg.get("self_distill_temp", 2.0))  # temperature for KL-div (softer targets)
    _teacher_model = None  # will hold a separate model instance for teacher outputs

    # External knowledge distillation from a pre-trained teacher model.
    # Unlike self_distill (which snapshots the current student model), this loads
    # a separate teacher model from a completed experiment's checkpoint.
    # The teacher can have a different architecture (e.g., 52p teacher for 36p student),
    # providing richer gradient signal than CE loss alone for ultra-compressed models.
    distill_from = cfg.get("distill_from", None)  # idea_id to load teacher from
    distill_alpha = float(cfg.get("distill_alpha", 0.0))  # 0 = disabled (default), typical: 0.5-2.0
    distill_temp = float(cfg.get("distill_temp", 2.0))  # temperature for KL-div (softer targets)
    _ext_teacher_model = None

    # Adaptive gradient clipping (AGC): clip gradients based on the ratio of gradient
    # norm to parameter norm, not absolute value (Brock et al. 2021, NFNets).
    # For tiny models where different layers have wildly different parameter scales,
    # this prevents over-clipping small params and under-clipping large params.
    # When enabled, replaces the standard grad_clip for weight matrices (>= 2D params).
    # 1D params (norms, biases) still use standard clipping if grad_clip > 0.
    agc_clip_factor = float(cfg.get("agc_clip_factor", 0.0))  # 0 = disabled (default), typical: 0.01-0.1

    # Stochastic weight perturbation: periodically add small random noise to weights.
    # Helps escape narrow local optima that gradient-based methods get stuck in.
    # Different from gradient noise (applied to gradients) and SAM (specific optimization).
    # Applied every N steps, giving the optimizer time to settle before next perturbation.
    weight_perturb_std = float(cfg.get("weight_perturb_std", 0.0))  # 0 = disabled (default), typical: 0.001-0.01
    weight_perturb_interval = int(cfg.get("weight_perturb_interval", 5000))  # steps between perturbations
    weight_perturb_decay = float(cfg.get("weight_perturb_decay", 0.9))  # decay factor per application

    # Z-loss regularization (Chowdhery et al. 2022, PaLM): penalizes large logits to
    # stabilize training near convergence. loss += z_loss_weight * log(sum(exp(logits)))^2
    # Prevents logit explosion that causes oscillation in the final 1-2% of accuracy.
    z_loss_weight = float(cfg.get("z_loss_weight", 0.0))  # 0 = disabled (default), typical: 1e-4 to 1e-3

    # Input digit noise: randomly replace input digit tokens with random digits during training.
    # Forces the model to develop more robust carry-chain representations since it can't
    # rely on any single input digit being correct. Different from embed_noise (continuous
    # Gaussian noise on embeddings) — this is discrete token-level corruption.
    input_digit_noise_prob = float(cfg.get("input_digit_noise_prob", 0.0))  # 0 = disabled (default), typical: 0.01-0.05

    # Entropy regularization: add negative entropy of output distribution to loss.
    # Prevents overconfident wrong predictions near convergence, keeping the model
    # exploring rather than committing hard to incorrect carry chains.
    # loss += entropy_reg_weight * mean(-sum(p * log(p)))
    # Positive weight INCREASES entropy (less confident), negative would decrease it.
    entropy_reg_weight = float(cfg.get("entropy_reg_weight", 0.0))  # 0 = disabled (default), typical: 0.001-0.01

    # Teacher-forcing noise: randomly corrupt teacher-forced output tokens during training.
    # At inference, the model generates autoregressively and errors in early digits compound
    # through carry chains. By injecting noise into teacher-forced tokens, the model learns
    # to handle imperfect carry-chain context, reducing the train/inference accuracy gap.
    tf_noise_prob = float(cfg.get("tf_noise_prob", 0.0))  # 0 = disabled (default), typical: 0.05-0.15

    # Embedding mixup: blend pairs of training examples at the embedding level.
    # Zhang et al. 2018 — creates virtual training samples by interpolating between
    # pairs of inputs, smoothing the loss landscape and improving generalization at
    # convergence. Uses Beta(alpha, alpha) distribution for interpolation coefficient.
    # The model sees mixed embeddings but loss uses the primary example's hard targets
    # (compatible with all other loss features). At inference, no mixup is applied.
    mixup_alpha = float(cfg.get("mixup_alpha", 0.0))  # 0 = disabled (default), typical: 0.1-0.4

    # Flood loss regularization (Ishida et al. 2020): prevents the training loss from
    # going below a threshold. When loss < flood_level, the gradient direction flips,
    # pushing loss back up. This prevents over-optimization on easy examples (which the
    # model already gets right) and preserves gradient signal for hard examples (carry chains).
    # loss = |loss - b| + b, where b = flood_level. At the plateau, the model memorizes
    # easy problems and starves hard ones of capacity. Flood loss keeps the model exploring.
    flood_level = float(cfg.get("flood_level", 0.0))  # 0 = disabled (default), typical: 0.01-0.1

    # Accuracy-based carry ramp: automatically increase carry_bias and/or
    # long_carry_chain_prob based on current best accuracy milestones.
    # Format: "acc1:carry_prob1,acc2:carry_prob2,..." e.g. "0.0:0.0,0.3:0.2,0.5:0.4,0.65:0.6"
    # Linear interpolation between accuracy milestones. Couples problem difficulty to
    # learning progress — as the model masters easy problems, it's fed progressively
    # harder carry-heavy problems. More responsive than step-based carry_bias_schedule.
    # When set, overrides carry_bias for the main training loop.
    carry_acc_ramp_str = cfg.get("carry_acc_ramp", None)
    carry_acc_ramp = []
    if carry_acc_ramp_str:
        for part in carry_acc_ramp_str.split(","):
            a, v = part.strip().split(":")
            carry_acc_ramp.append((float(a.strip()), float(v.strip())))

    # Per-digit loss variance penalty: penalizes high variance in per-digit loss values.
    # When the model is great at easy digit positions but terrible at carry-heavy positions,
    # the variance is high. This penalty forces the model to distribute its limited capacity
    # more evenly across all digit positions, preventing it from "giving up" on hard digits.
    # loss += digit_loss_var_weight * Var(per_digit_loss)
    digit_loss_var_weight = float(cfg.get("digit_loss_var_weight", 0.0))  # 0 = disabled (default), typical: 0.1-1.0

    # Minimax per-digit loss: blend mean and max per-digit losses.
    # loss = (1 - w) * mean(L_i) + w * max(L_i)
    # Standard mean loss lets the model neglect its worst digit position as long as
    # average accuracy is high. Near convergence (99%+ accuracy), the remaining errors
    # are concentrated at 1-2 carry-heavy digit positions. Minimax loss forces the model
    # to improve its WORST digit, directly targeting the carry-chain accuracy plateau.
    # At w=0: standard mean loss (backward compatible). At w=1: pure worst-case optimization.
    # Can be combined with OHEM (which selects hard EXAMPLES) — minimax selects hard DIGITS
    # within each example, orthogonal to OHEM's sample-level selection.
    minimax_digit_weight = float(cfg.get("minimax_digit_weight", 0.0))  # 0 = disabled (default), typical: 0.1-0.5

    # Cosine embedding auxiliary loss: for circular_arc/quadratic embeddings, add a loss
    # term that minimizes the angular distance between the 2D hidden state and the target
    # digit's embedding vector. CE loss through the logit projection can have sharp gradients
    # near decision boundaries; the cosine loss provides smoother gradient signal in the
    # embedding space, helping fine-tune the last few percent of accuracy near convergence.
    # loss += cos_embed_loss_weight * mean(1 - cos_sim(h[:2], embed_table[target]))
    cos_embed_loss_weight = float(cfg.get("cos_embed_loss_weight", 0.0))  # 0 = disabled (default), typical: 0.1-1.0

    # Symmetry consistency loss: regularize model to produce the same output for a+b and b+a.
    # Since addition is commutative, the model should predict identical digit sequences when
    # the two addends are swapped. Runs a no-grad forward pass on swapped inputs and adds
    # KL-divergence between original and swapped predictions. Exploits the mathematical
    # structure of addition to provide extra gradient signal, especially useful near convergence
    # where the model may have learned asymmetric attention patterns that work for (a,b) but
    # not (b,a). Complementary with commutative_aug (which swaps randomly; this explicitly
    # regularizes consistency). Zero extra parameters.
    symmetry_loss_weight = float(cfg.get("symmetry_loss_weight", 0.0))  # 0 = disabled (default), typical: 0.1-1.0

    # Loss clipping: cap per-token loss at a maximum value. Near convergence (97%+ accuracy),
    # the model occasionally encounters very hard carry-chain examples that produce large loss
    # spikes. These spikes generate destructive gradient updates that can undo progress on the
    # majority of correct examples. Clipping per-token loss at a ceiling prevents these outlier
    # examples from destabilizing training. This is the opposite of flood loss (which provides
    # a floor) — this provides a ceiling. Only active when > 0.
    loss_clip_max = float(cfg.get("loss_clip_max", 0.0))  # 0 = disabled (default), typical: 2.0-5.0

    # Auto multi-phase training: after main training, automatically do N additional
    # phases from the best checkpoint with progressively lower LR and EMA smoothing.
    # This eliminates the need for separate multi-phase orchestration runs.
    # Each phase: load best ckpt, reset grokfast EMA, train with lower LR + EMA.
    auto_phases = int(cfg.get("auto_phases", 0))  # 0 = disabled (default), typical: 2-4
    auto_phase_steps = int(cfg.get("auto_phase_steps", 0))  # steps per phase, 0 = steps//4
    auto_phase_lr_decay = float(cfg.get("auto_phase_lr_decay", 0.3))  # LR multiplier each phase
    auto_phase_ema = float(cfg.get("auto_phase_ema", 0.999))  # EMA decay during phases
    auto_phase_carry_bias = float(cfg.get("auto_phase_carry_bias", 0.0))  # carry probability during auto-phases (0 = use main carry_bias)
    auto_phase_grokfast_lambda = float(cfg.get("auto_phase_grokfast_lambda", 0.0))  # override grokfast_lambda in phases (0 = use main lambda)
    auto_phase_warmup_steps = int(cfg.get("auto_phase_warmup_steps", 0))  # warmup steps at start of each phase (0 = no warmup)
    auto_phase_ohem_ratio = float(cfg.get("auto_phase_ohem_ratio", 0.0))  # OHEM ratio during auto-phases (0 = disabled). OHEM was the single biggest accuracy boost (70%→99.45%); applying it during fine-tuning phases can push accuracy even higher.
    auto_phase_ohem_per_digit = bool(cfg.get("auto_phase_ohem_per_digit", False))  # Per-digit OHEM in auto-phases: select hardest examples per digit position then combine, ensuring every output position gets hard examples. Prevents the model from over-focusing on one failing digit at the expense of others.
    auto_phase_soft_ohem_gamma = float(cfg.get("auto_phase_soft_ohem_gamma", 0.0))  # Soft OHEM gamma during auto-phases (0 = disabled). Smoother than hard OHEM: upweights hard samples by (loss/mean)^gamma instead of binary keep/drop.
    auto_phase_adaptive_digit_weight = float(cfg.get("auto_phase_adaptive_digit_weight", 0.0))  # Adaptive per-digit loss weighting during auto-phases (0 = disabled). Dynamically upweights digit positions the model struggles with.
    auto_phase_focal_gamma = float(cfg.get("auto_phase_focal_gamma", 0.0))  # Focal loss gamma during auto-phases (0 = disabled, typical: 2.0). At 99%+ accuracy, most digits are confidently correct and provide near-zero gradient with standard CE. Focal loss down-weights these easy tokens: FL = -(1-p_t)^gamma * log(p_t). With gamma=2, a digit predicted with 99% confidence has its loss reduced 10000x vs a 50% confidence digit. This concentrates gradient signal on the few remaining carry-chain failures during fine-tuning phases, which is exactly where the 99.45% plateau needs to break.

    # Auto-phase curriculum: run a fast curriculum within each auto-phase.
    # Currently auto-phases always train on full 10-digit problems. When the model
    # is loaded from a checkpoint that's already at ~70%+ on 10 digits, running a
    # quick curriculum (e.g., "3:200,6:600,10:rest") helps reinforce basic carry
    # patterns from first principles before tackling full-difficulty problems.
    # This can unlock better solutions by "warming up" the model's carry-chain
    # circuits from simple to complex within each refinement phase.
    # Format: same as curriculum, e.g. "3:200,6:600,10:rest"
    auto_phase_curriculum_str = cfg.get("auto_phase_curriculum", None)
    auto_phase_curriculum_stages = []
    if auto_phase_curriculum_str:
        for part in auto_phase_curriculum_str.split(","):
            part = part.strip()
            d, s = part.split(":")
            auto_phase_curriculum_stages.append((int(d), s.strip()))

    # Auto-phase carry escalation: progressively increase carry chain difficulty
    # across auto-phases. Phase 1 uses base carry settings, each subsequent phase
    # increases min_carry_chain_len and long_carry_chain_prob, forcing the model to
    # master progressively harder carry patterns. Addresses the 97-98% plateau.
    auto_phase_carry_escalate = bool(cfg.get("auto_phase_carry_escalate", False))  # False = disabled (backward compatible)

    # Auto-phase LR restarts: multiple cosine annealing cycles within each auto-phase.
    # Instead of a single cosine decay, do N shorter cycles. Helps escape local optima
    # that the model gets trapped in during fine-tuning phases.
    auto_phase_restarts = int(cfg.get("auto_phase_restarts", 1))  # 1 = single cosine (default, backward compatible)

    # Long carry chain biased training: generate problems with consecutive carry chains
    # The last 2% of errors at 97-98% accuracy are almost always long carry chain failures.
    # This generates problems where min_carry_chain_len+ consecutive digits carry.
    long_carry_chain_prob = float(cfg.get("long_carry_chain_prob", 0.0))  # 0 = disabled (default), typical: 0.3-0.6
    min_carry_chain_len = int(cfg.get("min_carry_chain_len", 3))  # minimum consecutive carrying digits

    # Carry chain length curriculum: progressively increase min_carry_chain_len and/or
    # long_carry_chain_prob during training. Unlike the digit-count curriculum (which
    # increases problem SIZE from 3→6→10 digits), this increases carry DIFFICULTY at
    # full digit count. The model first masters problems with short/no carry chains,
    # then gradually faces longer consecutive carries. This prevents the model from
    # being overwhelmed by hard carry patterns early in training while ensuring it
    # eventually sees the hardest patterns (7+ consecutive carries) that cause the
    # final 0.5-1% accuracy gap. Format: "step:min_len:prob,..." e.g.
    # "0:0:0.0,50000:2:0.3,100000:4:0.5,150000:6:0.7". Linear interpolation for prob,
    # floor for min_len. Overrides min_carry_chain_len and long_carry_chain_prob when set.
    carry_chain_curriculum_str = cfg.get("carry_chain_curriculum", None)
    carry_chain_curriculum = []
    if carry_chain_curriculum_str:
        for part in carry_chain_curriculum_str.split(","):
            parts = part.strip().split(":")
            if len(parts) == 3:
                carry_chain_curriculum.append((int(parts[0].strip()), int(parts[1].strip()), float(parts[2].strip())))
            elif len(parts) == 2:
                # Short form: "step:min_len" — infer prob from min_len
                _ccl_step = int(parts[0].strip())
                _ccl_min = int(parts[1].strip())
                _ccl_prob = 0.3 + 0.1 * _ccl_min if _ccl_min > 0 else 0.0
                carry_chain_curriculum.append((_ccl_step, _ccl_min, min(_ccl_prob, 0.8)))

    def get_carry_chain_params(step):
        """Get (min_chain_len, chain_prob) based on carry chain curriculum schedule."""
        if not carry_chain_curriculum:
            return min_carry_chain_len, long_carry_chain_prob
        # Find surrounding schedule points
        if step <= carry_chain_curriculum[0][0]:
            return carry_chain_curriculum[0][1], carry_chain_curriculum[0][2]
        if step >= carry_chain_curriculum[-1][0]:
            return carry_chain_curriculum[-1][1], carry_chain_curriculum[-1][2]
        for i in range(len(carry_chain_curriculum) - 1):
            s0, m0, p0 = carry_chain_curriculum[i]
            s1, m1, p1 = carry_chain_curriculum[i + 1]
            if s0 <= step < s1:
                frac = (step - s0) / max(s1 - s0, 1)
                # Linear interpolation for prob, floor for min_len
                interp_prob = p0 + frac * (p1 - p0)
                interp_min = m0 if frac < 0.5 else m1  # step function at midpoint
                return interp_min, interp_prob
        return carry_chain_curriculum[-1][1], carry_chain_curriculum[-1][2]

    # Digit-targeted carry training: generate carry chains specifically at the digit
    # positions where the model currently has the lowest accuracy. Uses the per-digit
    # accuracy EMA (from adaptive_digit_weight) to weight carry generation toward weak
    # positions. Unlike carry_bias (uniform random carries) or long_carry_chain (consecutive
    # carries at random positions), this is surgical: if position 7 has 90% accuracy and
    # position 9 has 99.5%, carries are biased toward position 7. Requires adaptive_digit_weight > 0.
    digit_targeted_carry = bool(cfg.get("digit_targeted_carry", False))  # False = disabled (backward compatible)
    digit_targeted_carry_frac = float(cfg.get("digit_targeted_carry_frac", 0.5))  # fraction of batch with targeted carries

    # No-carry batch fraction: mix in problems with guaranteed zero carry chains.
    # At 99%+ accuracy, remaining errors include not just missed carries but also spurious
    # carries (the model adds a carry where none exists, producing off-by-one errors).
    # Explicit no-carry examples sharpen the carry/no-carry decision boundary. Each digit
    # pair satisfies a[i]+b[i]<10, so the correct answer requires zero carry propagation.
    # The model must learn to confidently predict "no carry" for these, which transfers to
    # better carry/no-carry discrimination on mixed problems.
    no_carry_fraction = float(cfg.get("no_carry_fraction", 0.0))  # 0 = disabled (default), typical: 0.05-0.2

    # Per-digit loss equalization: normalize per-digit losses by their running mean before
    # combining. Without this, easy digit positions (with low loss) contribute proportionally
    # less gradient than hard positions (with high loss). Paradoxically, this means the model
    # receives MORE gradient for positions it's already failing on (hard) and LESS for positions
    # it's borderline on (moderate difficulty). Loss equalization normalizes: L_i / running_mean_i,
    # so each digit position contributes equal gradient magnitude regardless of its absolute loss.
    # This prevents easy digits from being neglected and prevents hard digits from dominating.
    # Different from adaptive_digit_weight (weights by accuracy) — this normalizes by loss magnitude.
    # The EMA rate controls how fast the running mean adapts (higher = more stable).
    loss_equalize_digits = float(cfg.get("loss_equalize_digits", 0.0))  # 0 = disabled (default), typical: 0.99
    _digit_loss_running_mean = None  # (OUTPUT_LEN,) running per-digit loss mean

    # Carry chain breakdown evaluation: at each eval step, additionally report accuracy
    # stratified by max carry chain length in each test example. This reveals exactly WHERE
    # the model fails — e.g., 100% on chain length 0-4 but 60% on chain length 7+.
    # Output is printed during training for diagnostic use. Helps design targeted experiments.
    eval_carry_breakdown = bool(cfg.get("eval_carry_breakdown", False))  # False = disabled (default)

    # Adaptive batch mixing: dynamically compose each training batch as a mixture of
    # random, carry-biased, and long-carry-chain samples, with proportions driven by
    # per-digit accuracy EMA. When the model struggles on specific digit positions
    # (high-carry positions), the long-chain fraction automatically increases. When
    # it's doing well everywhere, the batch reverts to mostly random samples. This
    # replaces the binary carry_bias/long_carry_chain_prob with a continuous, adaptive
    # mixture. The three fractions always sum to 1.0.
    # Requires adaptive_digit_weight > 0 (which tracks per-digit accuracy EMA).
    adaptive_batch_mix = bool(cfg.get("adaptive_batch_mix", False))  # False = disabled (default)
    adaptive_batch_mix_base_random = float(cfg.get("adaptive_batch_mix_base_random", 0.5))  # minimum random fraction
    adaptive_batch_mix_max_chain = float(cfg.get("adaptive_batch_mix_max_chain", 0.4))  # maximum long-chain fraction
    adaptive_batch_mix_sensitivity = float(cfg.get("adaptive_batch_mix_sensitivity", 2.0))  # how strongly per-digit errors influence mix

    # Cosine-restart OHEM: cyclically vary the effective OHEM ratio with cosine
    # annealing over a period, alternating between exploration (low/no OHEM, diverse
    # examples) and exploitation (high OHEM, hard examples). This prevents the model
    # from overfitting to a fixed difficulty distribution and gives periodic "relief"
    # where it can consolidate patterns on easier examples before the next hard push.
    # When enabled, modulates the effective OHEM ratio: ohem * (0.5 + 0.5*cos(2*pi*step/period)).
    # Combine with ohem_ratio or ohem_acc_schedule for the base OHEM intensity.
    ohem_cosine_period = int(cfg.get("ohem_cosine_period", 0))  # 0 = disabled (default), typical: 10000-30000
    ohem_cosine_min_frac = float(cfg.get("ohem_cosine_min_frac", 0.0))  # minimum fraction of base OHEM at trough (0 = no OHEM at trough)

    # Stochastic OHEM: add Gumbel noise to per-sample losses before top-K selection.
    # Standard OHEM always selects the same deterministic top-K hardest examples, which
    # can cause the model to overfit to a specific subset of hard carry-chain patterns.
    # Adding temperature-scaled Gumbel noise creates a stochastic top-K that still biases
    # toward hard examples but includes some variance, allowing the model to see diverse
    # hard patterns over time. Higher temperature = more randomness (approaching uniform
    # sampling). Lower temperature = more deterministic (approaching standard OHEM).
    # Uses Gumbel-top-K trick: argmax(log(loss) + Gumbel(0, temp)) ≡ sampling without
    # replacement with probability proportional to loss^(1/temp).
    ohem_temperature = float(cfg.get("ohem_temperature", 0.0))  # 0 = deterministic top-K (default), typical: 0.3-2.0

    # OHEM exploration fraction: replace a fraction of the OHEM-selected (hardest) examples
    # with randomly sampled examples from the full generated pool. Pure OHEM can cause the
    # model to overfit to a specific subset of hard carry-chain patterns, losing gradient
    # diversity and eventually stalling. This mixes in random examples to maintain exploration.
    # The combination of (1-frac) hardest + frac random gives focused learning on failures
    # while preserving diverse gradient signal from the broader problem distribution.
    # This addresses the failure mode where OHEM + auto-phases degrades accuracy: the
    # auto-phase loads a checkpoint that's already good at most problems, and pure OHEM
    # during fine-tuning focuses exclusively on the few remaining hard cases, overfitting
    # to them and degrading accuracy on previously-correct problems.
    # Different from ohem_temperature (stochastic selection biased toward hard) —
    # this guarantees a minimum fraction of truly random examples in every batch.
    ohem_exploration_frac = float(cfg.get("ohem_exploration_frac", 0.0))  # 0 = pure OHEM (default), typical: 0.1-0.3

    # Confident-wrong penalty: extra loss for high-confidence WRONG predictions.
    # Near convergence (99%+ accuracy), the remaining errors are carry-chain digits where
    # the model is confidently predicting the wrong digit (e.g., softmax probability 0.8+
    # on the wrong class). Standard CE loss penalizes these the same as uncertain wrong
    # predictions, but confident-wrong errors are harder to fix because the model's
    # representations have solidified around the wrong answer. This penalty multiplies
    # the per-digit loss by (confidence * is_wrong * weight), giving extra gradient signal
    # specifically for the "confidently wrong" failure mode. Complements OHEM (which
    # selects hard EXAMPLES) by focusing on hard DIGIT POSITIONS within each example.
    confident_wrong_penalty = float(cfg.get("confident_wrong_penalty", 0.0))  # 0 = disabled (default), typical: 1.0-5.0

    # Hard example replay buffer: maintain a buffer of the hardest (prompt, target) pairs
    # encountered during OHEM scoring. Each training step, a fraction of the batch is drawn
    # from the replay buffer instead of fresh random generation. This ensures the model
    # regularly revisits the hardest carry-chain patterns it has ever seen, even if they
    # are rare in the random sampling distribution. Standard OHEM can only select hard
    # examples from within a single batch; the replay buffer accumulates hard examples
    # across the entire training run. Only active when OHEM is also active (ohem_ratio > 1).
    ohem_replay_size = int(cfg.get("ohem_replay_size", 0))  # 0 = disabled (default), typical: 256-1024
    ohem_replay_frac = float(cfg.get("ohem_replay_frac", 0.15))  # fraction of batch from replay (typical: 0.1-0.3)
    _replay_prompts = None  # (buffer_size, PROMPT_LEN) tensor
    _replay_targets = None  # (buffer_size, OUTPUT_LEN) tensor
    _replay_losses = None   # (buffer_size,) tensor — tracks difficulty for replacement
    _replay_count = 0       # number of valid entries in the buffer

    # Accuracy-gated curriculum acceleration: skip to harder digit counts when the
    # model achieves a threshold accuracy on the current curriculum stage. Instead of
    # waiting for a fixed step count, the curriculum advances as soon as the model is
    # ready. This avoids wasting GPU steps on mastered difficulty levels and can cut
    # time-to-convergence significantly. The threshold is checked at each eval step.
    # When the model's best accuracy exceeds the gate, max_digits is increased.
    # Format: "digits:acc_threshold,..." e.g. "3:0.5,6:0.3,10:0.0"
    # Each entry means: advance past N-digit problems when accuracy >= threshold.
    # Missing entries use the step-based curriculum (backward compatible).
    curriculum_acc_gate_str = cfg.get("curriculum_acc_gate", None)
    curriculum_acc_gate = {}
    if curriculum_acc_gate_str:
        for part in curriculum_acc_gate_str.split(","):
            d, a = part.strip().split(":")
            curriculum_acc_gate[int(d.strip())] = float(a.strip())
    _curriculum_acc_override = None  # will hold forced max_digits when gate triggers

    # Grokfast warmup: linearly ramp grokfast_lambda from 0 to target over N steps.
    # Simpler alternative to grokfast_lambda_schedule. Lets the model learn basic
    # patterns before amplifying slow-moving gradient components for grokking.
    grokfast_warmup_steps = int(cfg.get("grokfast_warmup_steps", 0))  # 0 = no warmup (default)

    # Grokfast minimum accuracy gate: disable grokfast until best_accuracy reaches this threshold.
    # For ultra-compressed models (36-40p) that start at 0% accuracy, grokfast amplifies
    # pure noise since there's no meaningful slow-moving signal to amplify yet. Waiting
    # until the model shows some learning signal ensures grokfast amplifies real patterns.
    grokfast_min_acc = float(cfg.get("grokfast_min_acc", 0.0))  # 0 = always on (default), typical: 0.01-0.1

    # Grokfast improve decay: partially reset the grokfast gradient EMA when best accuracy
    # improves at an evaluation step. Each accuracy improvement changes the loss landscape —
    # the old slow-moving gradient direction (stored in the EMA) pointed toward improvements
    # in the PRE-improvement landscape and may be suboptimal or counterproductive in the
    # POST-improvement landscape. Multiplying the EMA by (1 - decay) after each improvement
    # refreshes the gradient momentum, allowing faster adaptation to the new optimization
    # terrain. This addresses a specific failure mode at 99%+ accuracy: the model makes a
    # small improvement (e.g., fixing one carry chain pattern), but stale grokfast EMA
    # keeps amplifying the old gradient direction, undoing the improvement on the next step.
    # Different from grokfast_cycle_period (fixed schedule, not triggered by progress).
    grokfast_improve_decay = float(cfg.get("grokfast_improve_decay", 0.0))  # 0 = disabled (default), typical: 0.3-0.7

    # Annealed tying: parsed here for the training loop progress update
    anneal_tying_steps = int(cfg.get("anneal_tying_steps", 0))

    # Annealed norm sharing: smoothly transition ln2 → ln1 (shared norms) over N steps.
    # Unlike hard share_norms=True (82% get 0%), this lets the model gradually adapt to
    # shared norms. At step 0: independent ln2. At step N: ln2 = ln1 (fully shared).
    # Between: output = (1-t)*ln2(x) + t*ln1(x). After completion, ln2 params are excluded
    # from inference count (effectively saving d_model params, e.g., 3p for d_model=3).
    # Only activates when share_norms=False (otherwise already shared).
    anneal_norm_sharing_steps = int(cfg.get("anneal_norm_sharing_steps", 0))  # 0 = disabled (default)

    # Progressive multi-tying schedule: schedule multiple tying transitions during training.
    # Unlike switch_tie_down_gate_at (single transition), this allows chaining several
    # compression steps to gradually reduce from 52p → 49p → 46p within one training run.
    # Format: "step:action,step:action,..." e.g. "50000:tie_down_gate,100000:share_norms,150000:share_ln_f"
    # Supported actions:
    #   tie_down_gate: down_proj = gate_proj^T (saves ff_dim*d_model params, e.g. 6p)
    #   share_norms: ln2 = ln1 (saves d_model params, e.g. 3p)
    #   share_ln_f: ln_f = ln1 (saves d_model params, e.g. 3p)
    # At each transition, weights are merged (averaged) for smooth initialization.
    # Critical for reaching sub-52p: direct training at 46p always gets 0%, but
    # 52p → tie_down_gate → share_norms progressively compresses while preserving learned circuits.
    progressive_tie_schedule_str = cfg.get("progressive_tie_schedule", None)
    progressive_tie_schedule = []
    _progressive_tie_done = set()  # track which actions have been applied
    if progressive_tie_schedule_str:
        for part in progressive_tie_schedule_str.split(","):
            s, action = part.strip().split(":")
            progressive_tie_schedule.append((int(s.strip()), action.strip()))

    # Tying alignment regularization: add L2 penalty encouraging weight pairs to converge
    # toward their tied configuration BEFORE hard switching. This prepares the model for
    # tying by gradually pushing weights to match, producing a smoother transition than
    # abrupt hard-switch or annealing alone. The penalty ramps linearly from 0 at
    # tie_align_start_step to tie_align_weight at the scheduled switching step.
    # Targets: "down_gate" = ||gate.W - down.W^T||^2, "norms" = ||ln1.W - ln2.W||^2,
    #          "ln_f" = ||ln_f.W - ln1.W||^2.
    # Format: comma-separated targets, e.g. "down_gate,norms,ln_f"
    tie_align_weight = float(cfg.get("tie_align_weight", 0.0))  # 0 = disabled (default), typical: 0.001-0.01
    tie_align_targets_str = cfg.get("tie_align_targets", "")  # e.g. "down_gate,norms"
    tie_align_targets = [t.strip() for t in tie_align_targets_str.split(",") if t.strip()] if tie_align_targets_str else []
    tie_align_start_step = int(cfg.get("tie_align_start_step", 0))  # step to start ramping alignment loss

    # Progressive down-gate tying: at a specified step, hard-switch from independent
    # gate_proj and down_proj to tied computation (down = gate^T). At the switch step,
    # gate_proj.weight is re-initialized as the average of gate_proj.weight and
    # down_proj.weight.T, giving a smooth initialization that preserves learned
    # representations. down_proj is then frozen and excluded from inference param count.
    # This transitions a 52p model to 46p mid-training, allowing the model to develop
    # strong carry-chain circuits with full expressivity before compressing. Critical for
    # getting sub-52p models to converge — direct 46p training almost always gets 0%.
    switch_tie_down_gate_at = int(cfg.get("switch_tie_down_gate_at", 0))  # 0 = disabled (default), typical: 50000-100000

    # Stall LR bump: when accuracy hasn't improved for N eval steps, temporarily
    # multiply LR by a factor then decay back over bump_duration steps. This helps
    # escape narrow plateaus at 99%+ accuracy where the optimizer is stuck in a
    # shallow local minimum. The bump is large enough to escape the basin but brief
    # enough to not destroy learned carry-chain representations. After the bump,
    # LR returns to the cosine schedule. Multiple bumps can occur during training.
    stall_lr_bump_factor = float(cfg.get("stall_lr_bump_factor", 0.0))  # 0 = disabled (default), typical: 3.0-10.0
    stall_lr_bump_patience = int(cfg.get("stall_lr_bump_patience", 0))  # eval steps without improvement before bump (0 = disabled)
    stall_lr_bump_duration = int(cfg.get("stall_lr_bump_duration", 2000))  # steps to decay bump back to normal
    _stall_bump_step = None  # step when last bump was triggered

    # Phase soup: after all auto-phases complete, average the best checkpoints from
    # each phase (including phase 0 / main training). This is different from ckpt_soup
    # (which averages within one training run) — phase_soup averages across phases,
    # producing a smoother solution that combines the strengths of each phase's
    # optimization trajectory. Especially useful when different phases discover
    # different carry-chain solutions. Zero extra params.
    phase_soup = bool(cfg.get("phase_soup", False))  # False = disabled (default)

    # Commutative ensemble at eval time: for each test (a,b), also evaluate (b,a)
    # and average logits before argmax. Since a+b = b+a, both orderings should
    # produce identical results, but the model may have asymmetric attention patterns
    # that get one ordering right and the other wrong. Averaging the logits produces
    # a more robust prediction. Zero extra params, 2x eval cost. Applied to final
    # evaluation and verify-set evaluation.
    eval_commutative_ensemble = bool(cfg.get("eval_commutative_ensemble", False))  # False = disabled (default)

    # Evolution Strategy (ES) fine-tuning: gradient-free optimization after all gradient-based
    # training completes (including auto-phases). Uses Natural Evolution Strategies (NES):
    # each step creates `es_population` random weight perturbations, evaluates each on a batch,
    # and updates weights using the fitness-weighted average of perturbation directions.
    # This is fundamentally different from gradient descent and can escape flat loss landscapes
    # where gradients vanish (the 99.45% plateau). ES operates on the accuracy objective directly
    # (not a differentiable proxy), which is exactly what we need to push the last 0.5-1%.
    # Only activates after all other training (main loop + carry focus + auto-phases + soups).
    es_finetune_steps = int(cfg.get("es_finetune_steps", 0))  # 0 = disabled (default), typical: 50-200
    es_population = int(cfg.get("es_population", 20))  # candidates per step, typical: 10-50
    es_sigma = float(cfg.get("es_sigma", 0.001))  # perturbation std, typical: 0.0005-0.005
    es_eval_samples = int(cfg.get("es_eval_samples", 2000))  # eval samples per candidate, typical: 1000-5000

    # Coordinate-descent weight refinement: systematic per-parameter optimization after all
    # gradient-based and ES training. Unlike ES (which perturbs ALL params with random noise),
    # this sweeps each parameter individually, testing small perturbations and keeping changes
    # that improve accuracy. For a 52-param model, each sweep evaluates ~52 * n_probes candidates.
    # Coordinate descent is especially effective for tiny models where each parameter has a
    # distinct, interpretable effect — unlike large models where individual parameter changes
    # are diluted. The method is deterministic and exhaustive within its search radius,
    # complementing ES's stochastic exploration.
    coord_refine_steps = int(cfg.get("coord_refine_steps", 0))  # 0 = disabled (default), typical: 3-10 full sweeps
    coord_refine_delta = float(cfg.get("coord_refine_delta", 0.001))  # perturbation size per probe, typical: 0.0005-0.005
    coord_refine_eval_samples = int(cfg.get("coord_refine_eval_samples", 2000))  # eval samples per probe

    # CMA-ES (Covariance Matrix Adaptation Evolution Strategy) post-training fine-tuning.
    # The gold standard for gradient-free optimization of small parameter vectors (<100 params).
    # Unlike NES (es_finetune, isotropic Gaussian perturbations), CMA-ES learns the full
    # covariance structure of the fitness landscape — it discovers which parameters are
    # correlated and searches along principal axes of the landscape. For 52 params:
    # - Learns a 52×52 covariance matrix over generations
    # - Adapts the step size (sigma) automatically via cumulative step-size adaptation (CSA)
    # - Updates covariance via rank-one and rank-mu updates from evolution paths
    # This enables finding optima that isotropic NES and coordinate descent cannot reach,
    # because the optimal direction may involve correlated changes to multiple parameters
    # (e.g., simultaneously adjusting Q_proj and gate_proj weights for a carry-chain circuit).
    # Runs after coordinate refinement, before final evaluation.
    cmaes_steps = int(cfg.get("cmaes_steps", 0))  # 0 = disabled (default), typical: 50-200 generations
    cmaes_sigma = float(cfg.get("cmaes_sigma", 0.01))  # initial step size, typical: 0.005-0.05
    cmaes_population = int(cfg.get("cmaes_population", 0))  # population size, 0 = auto (4 + floor(3*ln(n)))
    cmaes_eval_samples = int(cfg.get("cmaes_eval_samples", 2000))  # eval samples per candidate

    # Multi-checkpoint greedy soup: load best checkpoints from multiple completed experiments
    # and find the optimal weighted combination via greedy search. Different from:
    # - ckpt_soup: averages top-K checkpoints from ONE training run (uniform weights)
    # - phase_soup: averages best from each auto-phase (uniform weights)
    # - interpolate_from: blends with ONE checkpoint BEFORE training (fixed alpha)
    # Greedy soup loads N external checkpoints and greedily adds them to a running average
    # if they improve accuracy. This leverages the hundreds of completed experiments to find
    # a better solution in the weight space spanned by diverse training runs. The order of
    # addition matters (greedy, not optimal), but typically finds good blends quickly.
    # Format: comma-separated idea IDs, e.g. "idea-0141,idea-e90001,idea-ce0006"
    greedy_soup_from = cfg.get("greedy_soup_from", None)  # None = disabled (default)
    greedy_soup_eval_samples = int(cfg.get("greedy_soup_eval_samples", 2000))  # eval samples per blend test

    # Reverse carry curriculum in auto-phases: start each auto-phase with the HARDEST
    # carry-chain problems (long chains, min_chain_len=6+, chain_prob=0.9) and gradually
    # decrease difficulty to include easier/random problems. This is the OPPOSITE of normal
    # curriculum (which escalates difficulty). The intuition: at 99%+ accuracy, the model
    # has already mastered easy problems. Starting auto-phases with maximum difficulty forces
    # ALL learning capacity onto the hardest carry-chain failures. Gradually mixing in easier
    # problems prevents catastrophic forgetting of basic patterns. Different from:
    # - auto_phase_carry_escalate (increases difficulty across phases, not within each phase)
    # - carry_chain_curriculum (step-based schedule in main training, not auto-phases)
    auto_phase_reverse_carry = bool(cfg.get("auto_phase_reverse_carry", False))  # False = disabled (default)

    # Eval-time embedding perturbation ensemble: at final evaluation, run N forward passes
    # with small Gaussian noise added to the embedding layer, then average logits before
    # argmax. This provides a cheap Monte Carlo ensemble that smooths out sharp decision
    # boundaries in the logit space. Different from majority_vote (which takes argmax per pass
    # then votes — can't recover from unanimous wrong digit) and eval_commutative_ensemble
    # (which only uses a+b and b+a). The noise injection creates diverse representations
    # that correct borderline carry-chain predictions via logit averaging.
    # Zero extra params, N× eval cost.
    eval_perturb_ensemble_n = int(cfg.get("eval_perturb_ensemble_n", 0))  # 0 = disabled (default), typical: 5-20
    eval_perturb_std = float(cfg.get("eval_perturb_std", 0.01))  # noise std, typical: 0.005-0.05

    # Checkpoint interpolation: before training, interpolate between the model's current weights
    # and a second checkpoint from a different experiment. The interpolated model starts in a
    # region of the loss landscape between two good solutions, which is often a flatter, more
    # generalizable minimum (Wortsman et al. 2022, "Model Soups"). Can be used standalone
    # (train=0 steps, just evaluate the interpolation) or as better initialization for training.
    # Different from warm_start_from (which loads one checkpoint fully): this BLENDS two checkpoints.
    interpolate_from = cfg.get("interpolate_from", None)  # idea_id of second checkpoint
    interpolate_alpha = float(cfg.get("interpolate_alpha", 0.5))  # blend factor: result = alpha * current + (1-alpha) * other

    # Embedding freeze: keep embedding parameters frozen for the first N steps.
    # For ultra-compressed models where embedding (3 params for circular_arc) gets
    # wild gradient updates from the multi-path tied Q_proj, freezing the embedding
    # lets the network weights stabilize first before embedding starts changing.
    # This prevents cascading instability where bad early embedding updates
    # permanently damage the downstream representations.
    embed_freeze_steps = int(cfg.get("embed_freeze_steps", 0))  # 0 = no freeze (default), typical: 1000-5000

    # Curriculum output masking: during early curriculum stages (e.g., 3-digit training),
    # output digits beyond max_digits+1 are always 0 and provide no useful gradient signal.
    # Masking their loss focuses gradient on the active digit positions, accelerating learning
    # of carry patterns. Automatically disabled once curriculum reaches 10 digits.
    output_mask_curriculum = bool(cfg.get("output_mask_curriculum", False))  # False = disabled (backward compatible)

    # Dynamic gradient rescaling for multi-path tied parameters: when Q_proj serves
    # multiple roles (Q, K via transform, V via tie_kv, O via tie_qo), gradients from
    # all paths accumulate on the same parameter, causing gradient explosion proportional
    # to the number of sharing paths. tie_init_scale only fixes initialization; this
    # rescales gradients EVERY step by 1/sqrt(N_paths), preventing magnitude drift
    # throughout training. Critical for getting sub-50p heavily-tied models to converge
    # where tie_init_scale alone is insufficient (explosion occurs after curriculum
    # transitions where the loss landscape changes dramatically).
    grad_rescale_tied = bool(cfg.get("grad_rescale_tied", False))  # False = disabled (default)
    _grad_rescale_factor = 1.0
    if grad_rescale_tied:
        _n_grad_paths = 1  # Q always uses Q_proj
        _has_k_tying_grt = any(cfg.get(k, False) for k in [
            "k_rot_q", "k_cayley_q", "k_givens_q", "k_perm_q",
            "k_signperm_q", "k_hadamard_q", "k_householder_q",
            "k_negate_q", "k_diag_q", "k_affine_q", "tie_qk",
        ])
        if _has_k_tying_grt and cfg.get("tie_kv", False):
            _n_grad_paths += 2  # K and V both derive from Q
        elif _has_k_tying_grt:
            _n_grad_paths += 1  # K derives from Q
        elif cfg.get("v_eq_q", False):
            _n_grad_paths += 1
        if cfg.get("tie_qo", False):
            _n_grad_paths += 1
        if cfg.get("tie_gate_q", False):
            _n_grad_paths += 1
        if cfg.get("tie_up_q", False):
            _n_grad_paths += 1
        if _n_grad_paths > 1:
            _grad_rescale_factor = 1.0 / math.sqrt(_n_grad_paths)
            print(f"Gradient rescaling: Q_proj grads scaled by 1/sqrt({_n_grad_paths}) = {_grad_rescale_factor:.3f}")

    # Per-digit adaptive temperature: dynamically scale output logits per digit position
    # based on running accuracy EMA. Digits with LOW accuracy get LOWER temperature
    # (sharper logits → stronger gradients), digits with HIGH accuracy get HIGHER temperature
    # (softer logits → gentler gradients to prevent overconfidence). This is different from:
    # - adaptive_digit_weight (scales loss magnitude, not logit temperature)
    # - per_pos_temp (static learnable, not accuracy-adaptive)
    # - train_temp_start/end (global schedule, not per-digit)
    # The temperature for digit i: temp_i = 1.0 / (1.0 + scale * (1.0 - acc_i))
    # Particularly effective near 99%+ accuracy where a few carry-chain digits are stuck
    # while others are perfect — sharpens focus on the struggling digits.
    digit_adaptive_temp_scale = float(cfg.get("digit_adaptive_temp_scale", 0.0))  # 0 = disabled (default), typical: 1.0-5.0

    # Logit noise: add Gaussian noise to output logits during training. Near convergence
    # (99%+ accuracy), remaining carry-chain errors are at SHARP decision boundaries in the
    # logit space — the model's prediction is close to correct but falls just on the wrong
    # side of the argmax boundary. Standard CE loss provides near-zero gradient on the
    # correct side and large gradient on the wrong side, creating a discontinuous optimization
    # landscape. Logit noise smooths these sharp boundaries: by randomly perturbing logits
    # before the loss computation, the model experiences both sides of the boundary in
    # expectation, creating smoother gradients that help it learn to push predictions firmly
    # past the boundary. Different from embed_noise (perturbs inputs, propagates nonlinearly),
    # weight_perturb_std (perturbs parameters periodically), and label_smoothing (softens
    # targets, not predictions). Particularly effective when combined with OHEM, since OHEM
    # selects the borderline examples where logit noise has the most impact.
    logit_noise_std = float(cfg.get("logit_noise_std", 0.0))  # 0 = disabled (default), typical: 0.01-0.1

    # Logit clipping: clamp output logits to [-C, C] before loss computation.
    # Near convergence (99%+), logits for easy examples become extremely peaked
    # (e.g., logit=20+ for the correct digit), producing near-zero gradients.
    # This starves hard carry-chain examples of gradient signal. Clipping caps
    # maximum confidence, maintaining gradient flow from all examples and
    # preventing the optimizer from "forgetting" about the hardest 1%.
    # Different from logit_noise (which adds stochasticity) and label_smoothing
    # (which softens targets) — this directly bounds the logit magnitudes.
    logit_clip = float(cfg.get("logit_clip", 0.0))  # 0 = disabled (default), typical: 5.0-15.0

    # Noise-augmented inference: run evaluation N times with small Gaussian noise
    # added to embeddings, averaging output logits before argmax. This is test-time
    # augmentation (TTA) that smooths prediction boundaries. When the model is
    # borderline on a carry-chain digit (e.g., 55% confident), averaging over
    # multiple noisy passes can push it over the threshold. Zero extra training
    # params, modest eval cost (N forward passes per eval). Only affects accuracy
    # evaluation, not the training loss computation.
    inference_noise_samples = int(cfg.get("inference_noise_samples", 0))  # 0 = disabled (default), typical: 3-10
    inference_noise_std = float(cfg.get("inference_noise_std", 0.01))  # noise std for TTA, typical: 0.005-0.05

    # Plateau optimizer restart: when accuracy hasn't improved for N eval steps,
    # completely reset the optimizer state (clearing all momentum/variance) and
    # apply a one-time LR boost for `plateau_restart_boost_steps` steps. Unlike
    # opt_reset_interval (periodic regardless of progress) and stall_lr_bump
    # (which only temporarily bumps LR without resetting state), this combines
    # both: a fresh optimizer state eliminates stale adaptive statistics, while
    # the LR boost provides maximum exploration pressure. Only triggers once per
    # plateau (resets counter on improvement). Different from grokfast_improve_decay
    # (which partially resets grokfast EMA on improvement, not on stalls).
    plateau_restart_patience = int(cfg.get("plateau_restart_patience", 0))  # 0 = disabled (default), typical: 20000-50000 (in steps)
    plateau_restart_lr_boost = float(cfg.get("plateau_restart_lr_boost", 3.0))  # LR multiplier during boost, typical: 2.0-5.0
    plateau_restart_boost_steps = int(cfg.get("plateau_restart_boost_steps", 2000))  # how long boost lasts, typical: 1000-5000
    _plateau_restart_triggered_at = None  # step when last restart was triggered
    _plateau_restart_count = 0  # number of times restart has been triggered

    # Load external teacher model for knowledge distillation
    if distill_from and distill_alpha > 0 and results_dir:
        _dt_ckpt_path = Path(results_dir) / distill_from / "checkpoint.pt"
        _dt_cfg_path = Path(results_dir) / distill_from / "idea_config.yaml"
        if _dt_ckpt_path.exists() and _dt_cfg_path.exists():
            _dt_cfg = yaml.safe_load(_dt_cfg_path.read_text())
            _ext_teacher_model = AdderTransformer(_dt_cfg).to(device)
            _dt_ckpt = torch.load(_dt_ckpt_path, map_location=device, weights_only=True)
            _ext_teacher_model.load_state_dict(_dt_ckpt["state_dict"])
            for _tp in _ext_teacher_model.parameters():
                _tp.requires_grad_(False)
            _ext_teacher_model.eval()
            print(f"External teacher loaded from {distill_from} ({_ext_teacher_model.count_params()} params)")
        else:
            print(f"External teacher: checkpoint or config not found for {distill_from}, skipping distillation")

    # Training loop
    model.train()
    best_accuracy = 0.0
    best_state = None
    steps_since_improvement = 0
    t0 = time.time()

    prompt_len = PROMPT_LEN  # a_digits(10) + sep(1) + b_digits(10) + sep(1) = 22
    n_think = int(cfg.get("n_think_tokens", 0))
    out_offset = prompt_len - 1 + n_think  # index of first output prediction position

    for step in range(1, steps + 1):
        # Set LR
        if one_cycle:
            current_lr = get_lr_one_cycle(step)
        else:
            current_lr = get_lr(step)
        # Stall LR bump: temporarily boost LR to escape plateaus
        if stall_lr_bump_factor > 0 and _stall_bump_step is not None:
            _bump_elapsed = step - _stall_bump_step
            if _bump_elapsed < stall_lr_bump_duration:
                # Cosine decay from bump_factor back to 1.0
                _bump_progress = _bump_elapsed / max(stall_lr_bump_duration, 1)
                _bump_mult = 1.0 + 0.5 * (stall_lr_bump_factor - 1.0) * (1 + math.cos(math.pi * _bump_progress))
                current_lr = current_lr * _bump_mult
        # Plateau restart LR boost: temporarily boost LR after optimizer reset
        if plateau_restart_patience > 0 and _plateau_restart_triggered_at is not None:
            _pr_elapsed = step - _plateau_restart_triggered_at
            if 0 <= _pr_elapsed < plateau_restart_boost_steps:
                _pr_progress = _pr_elapsed / max(plateau_restart_boost_steps, 1)
                _pr_mult = 1.0 + 0.5 * (plateau_restart_lr_boost - 1.0) * (1 + math.cos(math.pi * _pr_progress))
                current_lr = current_lr * _pr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr * pg.get("lr_mult", 1.0)

        # Annealed tying: update progress for gradual K_proj → K_tied transition
        if anneal_tying_steps > 0:
            _anneal_t = min(step / anneal_tying_steps, 1.0)
            for _am in model.modules():
                if isinstance(_am, Attention) and _am.anneal_tying:
                    _am._anneal_progress = _anneal_t

        # Annealed norm sharing: update blend factor for gradual ln2 → ln1 transition
        if anneal_norm_sharing_steps > 0 and not cfg.get("share_norms", False):
            _norm_t = min(step / anneal_norm_sharing_steps, 1.0)
            for _nb in model.blocks:
                if _nb.has_mlp and hasattr(_nb, 'ln2') and hasattr(_nb, 'ln1'):
                    _nb._norm_blend_t = _norm_t

        # Progressive multi-tying schedule: check if any scheduled transition should fire
        if progressive_tie_schedule:
            for _pts_step, _pts_action in progressive_tie_schedule:
                if step == _pts_step and _pts_action not in _progressive_tie_done:
                    _progressive_tie_done.add(_pts_action)
                    if _pts_action == "tie_down_gate":
                        for _ptb in model.blocks:
                            if _ptb.has_mlp and hasattr(_ptb.mlp, 'down_proj') and not _ptb.mlp.tie_down_gate:
                                with torch.no_grad():
                                    _pt_merged = 0.5 * (_ptb.mlp.gate_proj.weight.data + _ptb.mlp.down_proj.weight.data.T)
                                    _ptb.mlp.gate_proj.weight.data.copy_(_pt_merged)
                                _ptb.mlp.tie_down_gate = True
                                _ptb.mlp._switched_down_proj = _ptb.mlp.down_proj
                                _ptb.mlp.down_proj.weight.requires_grad_(False)
                        print(f"  Progressive schedule: tie_down_gate at step {step}")
                    elif _pts_action == "share_norms":
                        for _ptb in model.blocks:
                            if _ptb.has_mlp and hasattr(_ptb, 'ln2') and _ptb.ln2 is not _ptb.ln1:
                                with torch.no_grad():
                                    if hasattr(_ptb.ln1, 'weight') and hasattr(_ptb.ln2, 'weight'):
                                        _pt_merged_n = 0.5 * (_ptb.ln1.weight.data + _ptb.ln2.weight.data)
                                        _ptb.ln1.weight.data.copy_(_pt_merged_n)
                                _ptb._norm_blend_t = 1.0  # mark as fully shared
                                _ptb.ln2.weight.requires_grad_(False) if hasattr(_ptb.ln2, 'weight') else None
                        print(f"  Progressive schedule: share_norms at step {step}")
                    elif _pts_action == "share_ln_f":
                        if hasattr(model, 'ln_f') and model.ln_f is not model.blocks[0].ln1:
                            with torch.no_grad():
                                if hasattr(model.ln_f, 'weight') and hasattr(model.blocks[0].ln1, 'weight'):
                                    _pt_merged_f = 0.5 * (model.ln_f.weight.data + model.blocks[0].ln1.weight.data)
                                    model.blocks[0].ln1.weight.data.copy_(_pt_merged_f)
                            # Replace ln_f with ln1 reference
                            model._orig_ln_f = model.ln_f  # preserve for param exclusion
                            model.ln_f = model.blocks[0].ln1
                        print(f"  Progressive schedule: share_ln_f at step {step}")

        # Progressive down-gate tying: hard switch from independent to tied weights
        if switch_tie_down_gate_at > 0 and step == switch_tie_down_gate_at:
            for _stb in model.blocks:
                if _stb.has_mlp and hasattr(_stb.mlp, 'down_proj') and not _stb.mlp.tie_down_gate:
                    with torch.no_grad():
                        _merged_w = 0.5 * (_stb.mlp.gate_proj.weight.data + _stb.mlp.down_proj.weight.data.T)
                        _stb.mlp.gate_proj.weight.data.copy_(_merged_w)
                    _stb.mlp.tie_down_gate = True
                    _stb.mlp._switched_down_proj = _stb.mlp.down_proj
                    _stb.mlp.down_proj.weight.requires_grad_(False)
            print(f"  Progressive tie: switched to tie_down_gate at step {step} (saved {model.blocks[0].mlp._switched_down_proj.weight.numel()}p)")

        # Weight decay schedule: update WD dynamically
        if wd_schedule:
            effective_wd = wd_schedule[0][1]
            for _wi in range(len(wd_schedule) - 1):
                s0, v0 = wd_schedule[_wi]
                s1, v1 = wd_schedule[_wi + 1]
                if s0 <= step < s1:
                    frac = (step - s0) / max(s1 - s0, 1)
                    effective_wd = v0 + frac * (v1 - v0)
                    break
                elif step >= s1:
                    effective_wd = v1
            for pg in optimizer.param_groups:
                pg["weight_decay"] = effective_wd

        # Get max digits for curriculum (with optional smooth transitions)
        max_digits = get_max_digits(step)
        _smooth_primary, _smooth_secondary, _smooth_mix = get_max_digits_smooth(step)

        # Accuracy-gated curriculum acceleration: override max_digits when accuracy gate is met
        if curriculum_acc_gate and _curriculum_acc_override is None:
            for _cag_digits, _cag_thresh in sorted(curriculum_acc_gate.items()):
                if max_digits <= _cag_digits and best_accuracy >= _cag_thresh:
                    # Advance past this digit count
                    _next_digits = None
                    for _cs_d, _cs_s in curriculum_stages:
                        if _cs_d > _cag_digits:
                            _next_digits = _cs_d
                            break
                    if _next_digits is not None and _next_digits > max_digits:
                        _curriculum_acc_override = _next_digits
                        print(f"  Curriculum acc gate: accuracy {best_accuracy:.3f} >= {_cag_thresh} at {_cag_digits}d, advancing to {_next_digits}d at step {step}")
                        break
        if _curriculum_acc_override is not None and max_digits < _curriculum_acc_override:
            max_digits = _curriculum_acc_override
            _smooth_primary, _smooth_secondary, _smooth_mix = max_digits, max_digits, 0.0

        # Curriculum recycling: periodically reset to easy problems after curriculum completes
        if curriculum_recycle_period > 0 and max_digits >= NUM_DIGITS:
            _recycle_pos = step % curriculum_recycle_period
            if _recycle_pos < _recycle_3d_steps:
                max_digits = 3
                _smooth_primary, _smooth_secondary, _smooth_mix = 3, 3, 0.0
            elif _recycle_pos < _recycle_3d_steps + _recycle_6d_steps:
                max_digits = 6
                _smooth_primary, _smooth_secondary, _smooth_mix = 6, 6, 0.0

        # Curriculum LR restart: detect digit transitions and rewarm LR
        if curriculum_lr_restart:
            if _prev_max_digits is not None and max_digits > _prev_max_digits:
                _restart_step = step
                print(f"  Curriculum transition {_prev_max_digits}→{max_digits} at step {step}, rewarming LR for {curriculum_lr_restart_steps} steps")
            _prev_max_digits = max_digits
            if _restart_step is not None:
                restart_elapsed = step - _restart_step
                if restart_elapsed < curriculum_lr_restart_steps:
                    # Override LR with short warmup ramp
                    current_lr = current_lr * restart_elapsed / max(curriculum_lr_restart_steps, 1)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr * pg.get("lr_mult", 1.0)

        # Update carry_bias from schedule if provided
        if carry_bias_schedule:
            # Linear interpolation between schedule points
            effective_carry_bias = carry_bias_schedule[0][1]
            for i in range(len(carry_bias_schedule) - 1):
                s0, v0 = carry_bias_schedule[i]
                s1, v1 = carry_bias_schedule[i + 1]
                if s0 <= step < s1:
                    frac = (step - s0) / max(s1 - s0, 1)
                    effective_carry_bias = v0 + frac * (v1 - v0)
                    break
                elif step >= s1:
                    effective_carry_bias = v1
            carry_bias = effective_carry_bias

        # Accuracy-based carry ramp: couple carry difficulty to learning progress
        if carry_acc_ramp:
            _car_acc = best_accuracy
            _effective_carry_acc = carry_acc_ramp[0][1]
            for _ci in range(len(carry_acc_ramp) - 1):
                _ca0, _cv0 = carry_acc_ramp[_ci]
                _ca1, _cv1 = carry_acc_ramp[_ci + 1]
                if _ca0 <= _car_acc < _ca1:
                    _cfrac = (_car_acc - _ca0) / max(_ca1 - _ca0, 1e-8)
                    _effective_carry_acc = _cv0 + _cfrac * (_cv1 - _cv0)
                    break
                elif _car_acc >= _ca1:
                    _effective_carry_acc = _cv1
            carry_bias = max(carry_bias, _effective_carry_acc)

        # Generate batch (with optional curriculum mixing and smooth transitions)
        effective_max_digits = max_digits
        # Smooth curriculum transitions: probabilistically use previous stage digits
        if _smooth_mix > 0 and random.random() < _smooth_mix:
            effective_max_digits = _smooth_secondary
        elif curriculum_mix_prob > 0 and max_digits > curriculum_mix_digits and random.random() < curriculum_mix_prob:
            effective_max_digits = curriculum_mix_digits
        # Compute effective OHEM ratio (schedule overrides static, start_step gates activation)
        _effective_ohem_ratio = ohem_ratio
        if ohem_schedule:
            _effective_ohem_ratio = ohem_schedule[0][1]
            for _oi in range(len(ohem_schedule) - 1):
                _os0, _ov0 = ohem_schedule[_oi]
                _os1, _ov1 = ohem_schedule[_oi + 1]
                if _os0 <= step < _os1:
                    _ofrac = (step - _os0) / max(_os1 - _os0, 1)
                    _effective_ohem_ratio = _ov0 + _ofrac * (_ov1 - _ov0)
                    break
                elif step >= _os1:
                    _effective_ohem_ratio = _ov1
        if ohem_start_step > 0 and step < ohem_start_step:
            _effective_ohem_ratio = 0.0
        # Accuracy-adaptive OHEM: override ratio based on current best accuracy
        if ohem_acc_schedule:
            _ohem_acc = best_accuracy
            _effective_ohem_ratio = ohem_acc_schedule[0][1]
            for _oai in range(len(ohem_acc_schedule) - 1):
                _oa0, _or0 = ohem_acc_schedule[_oai]
                _oa1, _or1 = ohem_acc_schedule[_oai + 1]
                if _oa0 <= _ohem_acc < _oa1:
                    _oafrac = (_ohem_acc - _oa0) / max(_oa1 - _oa0, 1e-8)
                    _effective_ohem_ratio = _or0 + _oafrac * (_or1 - _or0)
                    break
                elif _ohem_acc >= _oa1:
                    _effective_ohem_ratio = _or1
        # Cosine-restart OHEM: modulate effective OHEM ratio with cosine annealing
        if ohem_cosine_period > 0 and _effective_ohem_ratio > 1.0:
            _ohem_cos_phase = math.cos(2 * math.pi * step / ohem_cosine_period)
            _ohem_cos_frac = ohem_cosine_min_frac + (1.0 - ohem_cosine_min_frac) * 0.5 * (1.0 + _ohem_cos_phase)
            _effective_ohem_ratio = 1.0 + (_effective_ohem_ratio - 1.0) * _ohem_cos_frac

        # Loss feature accuracy gate: disable OHEM until model shows learning signal
        if loss_feature_min_acc > 0 and best_accuracy < loss_feature_min_acc:
            _effective_ohem_ratio = 0.0

        # OHEM warmup ramp: smoothly ramp from 1.0 to target ratio over ohem_warmup_steps
        if ohem_warmup_steps > 0 and _effective_ohem_ratio > 1.0:
            if _ohem_warmup_start_step is None:
                _ohem_warmup_start_step = step  # first step OHEM is active
            _ohem_warmup_elapsed = step - _ohem_warmup_start_step
            if _ohem_warmup_elapsed < ohem_warmup_steps:
                _ohem_warmup_frac = _ohem_warmup_elapsed / ohem_warmup_steps
                _effective_ohem_ratio = 1.0 + (_effective_ohem_ratio - 1.0) * _ohem_warmup_frac

        _gen_bs = int(batch_size * _effective_ohem_ratio) if _effective_ohem_ratio > 1.0 else batch_size

        # Carry chain curriculum: get scheduled min_chain_len and chain_prob
        _eff_min_chain_len, _eff_chain_prob = get_carry_chain_params(step)

        # Adaptive batch mixing: compose batch from random/carry/long-chain based on per-digit accuracy
        if adaptive_batch_mix and _digit_acc_ema is not None:
            # Compute mixing fractions from per-digit accuracy EMA
            # Lower per-digit accuracy → more carry-focused samples
            _abm_worst_acc = _digit_acc_ema.min().item()
            _abm_mean_acc = _digit_acc_ema.mean().item()
            # Difficulty signal: how far worst digit is from perfect (0=perfect, 1=zero acc)
            _abm_difficulty = (1.0 - _abm_worst_acc) ** adaptive_batch_mix_sensitivity
            _abm_chain_frac = min(_abm_difficulty * adaptive_batch_mix_max_chain, adaptive_batch_mix_max_chain)
            _abm_carry_frac = min(_abm_difficulty * 0.3, 0.3)  # moderate carry-biased fraction
            _abm_random_frac = max(adaptive_batch_mix_base_random, 1.0 - _abm_chain_frac - _abm_carry_frac)
            # Normalize to sum to 1.0
            _abm_total = _abm_random_frac + _abm_carry_frac + _abm_chain_frac
            _abm_random_frac /= _abm_total
            _abm_carry_frac /= _abm_total
            _abm_chain_frac /= _abm_total
            _n_random = max(1, int(_gen_bs * _abm_random_frac))
            _n_carry = max(0, int(_gen_bs * _abm_carry_frac))
            _n_chain = max(0, _gen_bs - _n_random - _n_carry)
            _parts_p, _parts_t = [], []
            if _n_random > 0:
                _rp, _rt = generate_batch(_n_random, max_digits=effective_max_digits, device=device,
                                          commutative_aug=commutative_aug)
                _parts_p.append(_rp); _parts_t.append(_rt)
            if _n_carry > 0:
                _cp, _ct = generate_carry_biased_batch(_n_carry, max_digits=effective_max_digits,
                                                        carry_prob=0.7, device=device, commutative_aug=commutative_aug)
                _parts_p.append(_cp); _parts_t.append(_ct)
            if _n_chain > 0:
                _lp, _lt = generate_long_carry_chain_batch(_n_chain, max_digits=effective_max_digits,
                                                            chain_prob=0.8, min_chain_len=_eff_min_chain_len,
                                                            device=device, commutative_aug=commutative_aug)
                _parts_p.append(_lp); _parts_t.append(_lt)
            prompts = torch.cat(_parts_p, dim=0)
            targets = torch.cat(_parts_t, dim=0)
            # Shuffle the mixed batch
            _abm_perm = torch.randperm(prompts.shape[0], device=device)
            prompts = prompts[_abm_perm]
            targets = targets[_abm_perm]
        elif digit_targeted_carry and _digit_acc_ema is not None:
            prompts, targets = generate_targeted_carry_batch(
                _gen_bs, max_digits=effective_max_digits, digit_acc_ema=_digit_acc_ema,
                target_fraction=digit_targeted_carry_frac, device=device,
                commutative_aug=commutative_aug,
            )
        elif _eff_chain_prob > 0:
            prompts, targets = generate_long_carry_chain_batch(
                _gen_bs, max_digits=effective_max_digits, chain_prob=_eff_chain_prob,
                min_chain_len=_eff_min_chain_len, device=device, commutative_aug=commutative_aug,
            )
        elif carry_bias > 0:
            prompts, targets = generate_carry_biased_batch(
                _gen_bs, max_digits=effective_max_digits, carry_prob=carry_bias, device=device,
                commutative_aug=commutative_aug,
            )
        else:
            prompts, targets = generate_batch(_gen_bs, max_digits=effective_max_digits, device=device,
                                              commutative_aug=commutative_aug)

        # No-carry batch mixing: replace a fraction of the batch with guaranteed no-carry examples
        if no_carry_fraction > 0:
            _n_no_carry = max(1, int(prompts.shape[0] * no_carry_fraction))
            _nc_prompts, _nc_targets = generate_no_carry_batch(
                _n_no_carry, max_digits=effective_max_digits, device=device,
                commutative_aug=commutative_aug,
            )
            # Replace the last _n_no_carry examples in the batch
            prompts = torch.cat([prompts[:prompts.shape[0] - _n_no_carry], _nc_prompts], dim=0)
            targets = torch.cat([targets[:targets.shape[0] - _n_no_carry], _nc_targets], dim=0)
            # Shuffle to mix no-carry examples throughout the batch
            _nc_perm = torch.randperm(prompts.shape[0], device=device)
            prompts = prompts[_nc_perm]
            targets = targets[_nc_perm]

        # Input digit noise: randomly corrupt input digit tokens (not separators)
        # This forces the model to be robust to imperfect carry-chain inputs
        if input_digit_noise_prob > 0:
            noise_mask = torch.rand_like(prompts.float()) < input_digit_noise_prob
            # Don't corrupt separator positions (indices 0, 11, 12, 23 where value is 0)
            # Only corrupt actual digit positions
            sep_positions = torch.zeros_like(prompts, dtype=torch.bool)
            sep_positions[:, 0] = True    # leading separator
            sep_positions[:, 11] = True   # mid separator 1
            sep_positions[:, 12] = True   # mid separator 2
            sep_positions[:, 23] = True   # trailing separator
            noise_mask = noise_mask & ~sep_positions
            random_digits = torch.randint(0, VOCAB_SIZE, prompts.shape, device=device)
            prompts = torch.where(noise_mask, random_digits, prompts)

        # Teacher forcing: prompt + thinking tokens + first (OUTPUT_LEN-1) target digits
        # Model predicts each target digit from the context so far
        if n_think > 0:
            think_toks = torch.zeros(prompts.shape[0], n_think, dtype=torch.long, device=device)
            full_input = torch.cat([prompts, think_toks, targets[:, :-1]], dim=1)
        else:
            full_input = torch.cat([prompts, targets[:, :-1]], dim=1)

        # Teacher-forcing noise: corrupt some teacher-forced tokens to simulate inference errors
        # Only corrupt target tokens, not thinking tokens
        tf_start = prompt_len + n_think
        if tf_noise_prob > 0 and full_input.shape[1] > tf_start:
            tf_n_target_tokens = full_input.shape[1] - tf_start
            tf_noise_mask = torch.rand(full_input.shape[0], tf_n_target_tokens, device=device) < tf_noise_prob
            tf_random_tokens = torch.randint(0, VOCAB_SIZE, (full_input.shape[0], tf_n_target_tokens), device=device)
            full_input = full_input.clone()
            full_input[:, tf_start:] = torch.where(tf_noise_mask, tf_random_tokens, full_input[:, tf_start:])

        # OHEM: Online Hard Example Mining - probe forward pass to select hardest examples
        if _effective_ohem_ratio > 1.0:
            with torch.no_grad():
                _ohem_logits = model(full_input)
                _ohem_out = _ohem_logits[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
                _ohem_per_digit_loss = F.cross_entropy(
                    _ohem_out.reshape(-1, VOCAB_SIZE), targets.reshape(-1),
                    reduction='none').view(-1, OUTPUT_LEN)
                if ohem_per_digit:
                    # Per-digit OHEM: select hardest samples for each digit position,
                    # then union them. Ensures every output digit gets hard examples.
                    _per_digit_k = max(batch_size // OUTPUT_LEN, 1)
                    _ohem_selected = set()
                    for _dig in range(OUTPUT_LEN):
                        _, _dig_idx = _ohem_per_digit_loss[:, _dig].topk(
                            min(_per_digit_k, _ohem_per_digit_loss.shape[0]))
                        _ohem_selected.update(_dig_idx.tolist())
                    # Fill remaining slots with overall hardest examples
                    _ohem_overall = _ohem_per_digit_loss.mean(dim=1)
                    _, _overall_idx = _ohem_overall.topk(min(batch_size, _ohem_per_digit_loss.shape[0]))
                    for _oi in _overall_idx.tolist():
                        if len(_ohem_selected) >= batch_size:
                            break
                        _ohem_selected.add(_oi)
                    _ohem_idx = torch.tensor(list(_ohem_selected)[:batch_size], device=device)
                else:
                    _ohem_loss = _ohem_per_digit_loss.mean(dim=1)
                    # Carry-weighted OHEM: bias selection toward carry-heavy examples
                    if ohem_carry_weight > 0:
                        _ocw_a_digs = prompts[:, 1:1+NUM_DIGITS]
                        _ocw_b_digs = prompts[:, 13:13+NUM_DIGITS]
                        _ocw_powers = _DIVISORS[:NUM_DIGITS].to(device)
                        _ocw_a_vals = (_ocw_a_digs * _ocw_powers).sum(dim=1)
                        _ocw_b_vals = (_ocw_b_digs * _ocw_powers).sum(dim=1)
                        _ocw_carries = compute_carry_targets(_ocw_a_vals, _ocw_b_vals, max_digits=effective_max_digits).to(device)
                        _ocw_frac = _ocw_carries.sum(dim=1).float() / OUTPUT_LEN  # (gen_bs,)
                        _ohem_loss = _ohem_loss * (1.0 + ohem_carry_weight * _ocw_frac)
                    # Stochastic OHEM: Gumbel-top-K selection for diverse hard examples
                    if ohem_temperature > 0:
                        _gumbel = -torch.log(-torch.log(torch.rand_like(_ohem_loss).clamp(min=1e-20)) + 1e-20)
                        _ohem_scores = torch.log(_ohem_loss.clamp(min=1e-20)) + ohem_temperature * _gumbel
                        _, _ohem_idx = _ohem_scores.topk(batch_size)
                    else:
                        _, _ohem_idx = _ohem_loss.topk(batch_size)
                    # OHEM exploration: replace a fraction of hard examples with random ones
                    if ohem_exploration_frac > 0:
                        _n_explore = max(1, int(batch_size * ohem_exploration_frac))
                        _n_hard = batch_size - _n_explore
                        _hard_idx = _ohem_idx[:_n_hard]
                        # Sample random indices from the full pool (excluding already-selected hard ones)
                        _all_idx = torch.arange(_gen_bs, device=device)
                        _hard_set = set(_hard_idx.tolist())
                        _avail = torch.tensor([i for i in range(_gen_bs) if i not in _hard_set], device=device)
                        if len(_avail) >= _n_explore:
                            _rand_perm = torch.randperm(len(_avail), device=device)[:_n_explore]
                            _rand_idx = _avail[_rand_perm]
                        else:
                            _rand_idx = _avail  # use all available if fewer than needed
                        _ohem_idx = torch.cat([_hard_idx, _rand_idx])
            # Update hard example replay buffer with OHEM-scored examples
            if ohem_replay_size > 0:
                _ohem_all_losses = _ohem_per_digit_loss.mean(dim=1)  # (gen_bs,)
                # Get top examples to put in buffer
                _replay_k = min(batch_size, ohem_replay_size)
                _, _replay_top_idx = _ohem_all_losses.topk(_replay_k)
                _replay_new_prompts = prompts[_replay_top_idx].detach()
                _replay_new_targets = targets[_replay_top_idx].detach()
                _replay_new_losses = _ohem_all_losses[_replay_top_idx].detach()
                if _replay_prompts is None:
                    _replay_prompts = torch.zeros(ohem_replay_size, PROMPT_LEN, dtype=torch.long, device=device)
                    _replay_targets = torch.zeros(ohem_replay_size, OUTPUT_LEN, dtype=torch.long, device=device)
                    _replay_losses = torch.zeros(ohem_replay_size, device=device)
                # Insert: replace lowest-loss entries in buffer
                for _ri in range(_replay_k):
                    if _replay_count < ohem_replay_size:
                        _replay_prompts[_replay_count] = _replay_new_prompts[_ri]
                        _replay_targets[_replay_count] = _replay_new_targets[_ri]
                        _replay_losses[_replay_count] = _replay_new_losses[_ri]
                        _replay_count += 1
                    else:
                        _min_idx = _replay_losses[:_replay_count].argmin()
                        if _replay_new_losses[_ri] > _replay_losses[_min_idx]:
                            _replay_prompts[_min_idx] = _replay_new_prompts[_ri]
                            _replay_targets[_min_idx] = _replay_new_targets[_ri]
                            _replay_losses[_min_idx] = _replay_new_losses[_ri]
            full_input = full_input[_ohem_idx]
            targets = targets[_ohem_idx]
            prompts = prompts[_ohem_idx]

        # Inject replay buffer examples into the batch
        if ohem_replay_size > 0 and _replay_count > 0 and _replay_prompts is not None:
            _n_replay = min(int(batch_size * ohem_replay_frac), _replay_count)
            if _n_replay > 0:
                _replay_sample_idx = torch.randint(0, _replay_count, (_n_replay,), device=device)
                _replay_p = _replay_prompts[_replay_sample_idx]
                _replay_t = _replay_targets[_replay_sample_idx]
                if n_think > 0:
                    _replay_think = torch.zeros(_n_replay, n_think, dtype=torch.long, device=device)
                    _replay_fi = torch.cat([_replay_p, _replay_think, _replay_t[:, :-1]], dim=1)
                else:
                    _replay_fi = torch.cat([_replay_p, _replay_t[:, :-1]], dim=1)
                # Replace last _n_replay examples in the batch
                full_input = torch.cat([full_input[:batch_size - _n_replay], _replay_fi], dim=0)
                targets = torch.cat([targets[:batch_size - _n_replay], _replay_t], dim=0)
                prompts = torch.cat([prompts[:batch_size - _n_replay], _replay_p], dim=0)

        # Self-distillation: compute teacher predictions using a separate model instance.
        # We construct a fresh AdderTransformer and copy weights via state_dict to avoid
        # inplace version counter corruption that copy.deepcopy can cause with shared
        # parameter references (which breaks loss.backward() on scalar parameters like arc_A).
        _teacher_out_pre = None
        if self_distill_alpha > 0 and step >= self_distill_start_step:
            if _teacher_model is None or step % self_distill_interval == 0:
                if _teacher_model is None:
                    _teacher_model = AdderTransformer(cfg).to(device)
                    for _tp in _teacher_model.parameters():
                        _tp.requires_grad_(False)
                # Clone state dict values to avoid shared TensorImpl version counter
                # corruption — model.state_dict() returns detached views sharing
                # the same TensorImpl, so inplace ops bump the student's version
                # counter and break backward() on scalar params like arc_A.
                _teacher_model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
                _teacher_model.eval()
            with torch.no_grad():
                teacher_logits_pre = _teacher_model(full_input)
                _teacher_out_pre = teacher_logits_pre[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]

        # Embedding mixup: blend pairs of examples at embedding level
        _mixup_lam = None
        _mixup_perm = None
        if mixup_alpha > 0:
            _mixup_lam = max(
                torch.distributions.Beta(
                    torch.tensor(mixup_alpha), torch.tensor(mixup_alpha)
                ).sample().item(),
                1e-6,
            )
            _mixup_lam = max(_mixup_lam, 1 - _mixup_lam)  # ensure primary example dominates
            _mixup_perm = torch.randperm(full_input.shape[0], device=device)

        logits = model(full_input, mixup_lam=_mixup_lam, mixup_perm=_mixup_perm)

        # Loss only on output positions (predict target from prompt context)
        output_logits = logits[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]

        # Logit noise: smooth loss landscape at sharp decision boundaries
        if logit_noise_std > 0 and model.training:
            output_logits = output_logits + torch.randn_like(output_logits) * logit_noise_std

        # Logit clipping: bound logit magnitudes to maintain gradient flow
        if logit_clip > 0:
            output_logits = output_logits.clamp(-logit_clip, logit_clip)

        # Update adaptive digit accuracy EMA from training predictions
        # (shared by adaptive_digit_weight and digit_adaptive_temp)
        if adaptive_digit_weight > 0 or digit_adaptive_temp_scale > 0:
            with torch.no_grad():
                _adw_preds = output_logits.argmax(dim=-1)  # (B, OUTPUT_LEN)
                _adw_correct = (_adw_preds == targets).float().mean(dim=0)  # (OUTPUT_LEN,)
                if _digit_acc_ema is None:
                    _digit_acc_ema = _adw_correct.clone()
                else:
                    _digit_acc_ema.mul_(adaptive_digit_ema_rate).add_(_adw_correct, alpha=1 - adaptive_digit_ema_rate)

        # Per-digit adaptive temperature: sharpen logits for struggling digits
        if digit_adaptive_temp_scale > 0 and _digit_acc_ema is not None:
            with torch.no_grad():
                # temp_i = 1.0 / (1.0 + scale * (1.0 - acc_i))
                # Low acc → low temp (sharp) → stronger gradients
                # High acc → temp ≈ 1.0 → normal gradients
                _dat_temps = 1.0 / (1.0 + digit_adaptive_temp_scale * (1.0 - _digit_acc_ema.clamp(0, 1)))
            output_logits = output_logits / _dat_temps.unsqueeze(0).unsqueeze(-1)

        # Add per-position output bias if enabled
        if output_pos_bias and pos_bias_layer is not None:
            output_logits = output_logits + pos_bias_layer.unsqueeze(0)

        # Training temperature annealing: scale logits by 1/temp before loss
        if train_temp_start > 0:
            progress = step / max(steps, 1)
            train_temp = train_temp_start + progress * (train_temp_end - train_temp_start)
            output_logits = output_logits / train_temp

        # Curriculum output masking: compute per-digit mask for loss (1 = active, 0 = masked)
        _curriculum_mask = None
        if output_mask_curriculum and effective_max_digits < NUM_DIGITS:
            active_digits = min(effective_max_digits + 1, OUTPUT_LEN)  # +1 for potential carry out
            _curriculum_mask = torch.zeros(OUTPUT_LEN, device=device)
            _curriculum_mask[:active_digits] = 1.0

        if digit_loss_weight and digit_loss_weight != "none":
            # Per-digit weighted loss: upweight harder (later) digits
            # output_logits: (B, OUTPUT_LEN, VOCAB_SIZE), targets: (B, OUTPUT_LEN)
            if digit_loss_weight == "linear":
                weights = torch.linspace(1.0, digit_loss_scale, OUTPUT_LEN, device=device)
            elif digit_loss_weight == "exponential":
                weights = torch.logspace(0, math.log10(digit_loss_scale), OUTPUT_LEN, device=device)
            else:
                weights = torch.ones(OUTPUT_LEN, device=device)
            if _curriculum_mask is not None:
                weights = weights * _curriculum_mask
            # Normalize so mean weight = 1 (over active positions)
            weights = weights / weights.mean().clamp(min=1e-8)
            # Compute per-sample per-digit loss
            per_digit_loss = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            # Apply focal loss modulation if enabled
            if focal_gamma > 0:
                with torch.no_grad():
                    probs = F.softmax(output_logits.reshape(-1, VOCAB_SIZE), dim=-1)
                    pt = probs.gather(1, targets.reshape(-1, 1)).squeeze(1).view(-1, OUTPUT_LEN)
                    focal_weight = (1 - pt) ** focal_gamma
                per_digit_loss = per_digit_loss * focal_weight
            loss = (per_digit_loss * weights.unsqueeze(0)).mean()
        else:
            if focal_gamma > 0 or _curriculum_mask is not None:
                # Per-token loss: needed for focal loss or curriculum masking
                per_token_loss = F.cross_entropy(
                    output_logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    label_smoothing=label_smoothing,
                    reduction='none',
                ).view(-1, OUTPUT_LEN)
                if focal_gamma > 0:
                    with torch.no_grad():
                        probs = F.softmax(output_logits.reshape(-1, VOCAB_SIZE), dim=-1)
                        pt = probs.gather(1, targets.reshape(-1, 1)).squeeze(1).view(-1, OUTPUT_LEN)
                        focal_weight = (1 - pt) ** focal_gamma
                    per_token_loss = per_token_loss * focal_weight
                if _curriculum_mask is not None:
                    per_token_loss = per_token_loss * _curriculum_mask.unsqueeze(0)
                loss = per_token_loss.mean()
            else:
                loss = F.cross_entropy(
                    output_logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    label_smoothing=label_smoothing,
                )

        # Per-digit loss equalization: normalize per-digit losses by running mean
        if loss_equalize_digits > 0:
            _eq_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            _eq_digit_means = _eq_per_digit.detach().mean(dim=0)  # (OUTPUT_LEN,)
            if _digit_loss_running_mean is None:
                _digit_loss_running_mean = _eq_digit_means.clone()
            else:
                _digit_loss_running_mean.mul_(loss_equalize_digits).add_(
                    _eq_digit_means, alpha=1 - loss_equalize_digits)
            # Normalize each digit position by its running mean
            _eq_scale = 1.0 / (_digit_loss_running_mean + 1e-8)
            _eq_scale = _eq_scale / _eq_scale.mean()  # preserve overall loss magnitude
            loss = (_eq_per_digit * _eq_scale.unsqueeze(0)).mean()

        # Polynomial cross-entropy: add polynomial correction for borderline predictions
        if poly_epsilon > 0:
            _poly_probs = F.softmax(output_logits.reshape(-1, VOCAB_SIZE), dim=-1)
            _poly_pt = _poly_probs.gather(1, targets.reshape(-1, 1)).squeeze(1).view(-1, OUTPUT_LEN)
            _poly_term = poly_epsilon * ((1 - _poly_pt) ** poly_degree).mean()
            loss = loss + _poly_term

        # Loss feature accuracy gate: check if auxiliary loss features should be active.
        # When loss_feature_min_acc > 0, all auxiliary loss features below this point
        # remain disabled until best_accuracy >= loss_feature_min_acc. This prevents
        # features like carry_chain_length_weight, loss_digit_top_k, soft_ohem, margin_loss
        # etc. from disrupting early training before basic digit patterns are learned.
        _aux_loss_active = (loss_feature_min_acc <= 0) or (best_accuracy >= loss_feature_min_acc)

        # Carry-weighted loss: upweight loss on digits with incoming carries
        if carry_loss_weight > 0 and _aux_loss_active:
            with torch.no_grad():
                a_digs_cw = prompts[:, 1:1+NUM_DIGITS]
                b_digs_cw = prompts[:, 13:13+NUM_DIGITS]
                powers_cw = _DIVISORS[:NUM_DIGITS].to(device)
                a_vals_cw = (a_digs_cw * powers_cw).sum(dim=1)
                b_vals_cw = (b_digs_cw * powers_cw).sum(dim=1)
                carry_bits = compute_carry_targets(a_vals_cw, b_vals_cw, max_digits=effective_max_digits).to(device)
                # Weight: 1.0 for non-carry digits, 1.0 + carry_loss_weight for carry digits
                carry_weights = 1.0 + carry_loss_weight * carry_bits.float()  # (B, OUTPUT_LEN)
            # Recompute loss with per-token weighting
            per_token_loss_cw = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            loss = (per_token_loss_cw * carry_weights).mean()

        # Carry-chain-length-weighted loss: upweight examples by longest consecutive carry chain
        if carry_chain_length_weight > 0 and _aux_loss_active:
            with torch.no_grad():
                _cclw_a_digs = prompts[:, 1:1+NUM_DIGITS]
                _cclw_b_digs = prompts[:, 13:13+NUM_DIGITS]
                _cclw_powers = _DIVISORS[:NUM_DIGITS].to(device)
                _cclw_a_vals = (_cclw_a_digs * _cclw_powers).sum(dim=1)
                _cclw_b_vals = (_cclw_b_digs * _cclw_powers).sum(dim=1)
                _cclw_max_chain = compute_max_carry_chain_length(_cclw_a_vals, _cclw_b_vals,
                                                                  max_digits=effective_max_digits).float()
                # Per-example weight: 1 + weight * (chain_length / OUTPUT_LEN)
                _cclw_weights = 1.0 + carry_chain_length_weight * (_cclw_max_chain / OUTPUT_LEN)
                _cclw_weights = _cclw_weights / _cclw_weights.mean()  # normalize to preserve loss scale
            # Recompute per-digit loss and apply per-example weighting
            _cclw_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            loss = (_cclw_per_digit * _cclw_weights.unsqueeze(1)).mean()

        # Top-K digit loss focusing: only backprop through the K hardest digit positions per example
        if loss_digit_top_k > 0 and loss_digit_top_k < OUTPUT_LEN and _aux_loss_active:
            _topk_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)  # (B, OUTPUT_LEN)
            # Create mask: 1 for top-K highest loss positions per example, 0 for rest
            _, _topk_indices = _topk_per_digit.topk(loss_digit_top_k, dim=1)  # (B, K)
            _topk_mask = torch.zeros_like(_topk_per_digit)
            _topk_mask.scatter_(1, _topk_indices, 1.0)
            # Scale to preserve expected loss magnitude: multiply by OUTPUT_LEN/K
            _topk_scale = OUTPUT_LEN / loss_digit_top_k
            loss = (_topk_per_digit * _topk_mask * _topk_scale).mean()

        # Soft OHEM: per-sample importance weighting by loss difficulty
        # Computes per-sample loss, raises to a power to upweight hard samples,
        # then recomputes weighted loss. Smoother than hard OHEM (no binary cutoff).
        if soft_ohem_gamma > 0 and _aux_loss_active:
            _soh_per_token = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            _soh_per_sample = _soh_per_token.mean(dim=1)  # (B,)
            with torch.no_grad():
                _soh_mean = _soh_per_sample.mean().clamp(min=1e-8)
                _soh_weights = (_soh_per_sample.detach() / _soh_mean) ** soft_ohem_gamma
                _soh_weights = _soh_weights / _soh_weights.mean()  # normalize to preserve loss scale
            loss = (_soh_per_token * _soh_weights.unsqueeze(1)).mean()

        # Adaptive per-digit loss weighting: upweight digit positions with lower accuracy
        if adaptive_digit_weight > 0 and _digit_acc_ema is not None and _aux_loss_active:
            _adw_weights = (1.0 - _digit_acc_ema.clamp(0, 1)) * adaptive_digit_weight + 1.0
            _adw_weights = _adw_weights / _adw_weights.mean()  # normalize
            _adw_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            loss = (_adw_per_digit * _adw_weights.unsqueeze(0)).mean()

        # Per-digit loss variance penalty: force uniform performance across digit positions
        if digit_loss_var_weight > 0:
            # Compute per-digit loss (averaged over batch) and penalize its variance
            _var_per_digit_loss = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN).mean(dim=0)  # (OUTPUT_LEN,) — mean loss per digit position
            loss = loss + digit_loss_var_weight * _var_per_digit_loss.var()

        # Minimax per-digit loss: blend mean and max per-digit losses
        if minimax_digit_weight > 0:
            _mm_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            _mm_mean = _mm_per_digit.mean()
            _mm_max = _mm_per_digit.mean(dim=0).max()  # max over digit positions (avg over batch)
            loss = (1.0 - minimax_digit_weight) * _mm_mean + minimax_digit_weight * _mm_max

        # Cosine embedding auxiliary loss: angular distance in 2D embedding space
        if cos_embed_loss_weight > 0 and hasattr(model, 'tok_embed') and hasattr(model.tok_embed, 'table'):
            _embed_table = model.tok_embed.table()  # (vocab_size, 2)
            _tok_dim = _embed_table.shape[1]
            # Get hidden states at output positions, take the first tok_dim dimensions
            _cos_hidden = model._last_hidden[:, out_offset:out_offset + OUTPUT_LEN, :_tok_dim]  # (B, OUTPUT_LEN, 2)
            # Get target embeddings
            _cos_target_embed = _embed_table[targets]  # (B, OUTPUT_LEN, 2)
            # Cosine similarity loss: 1 - cos_sim (so 0 = perfect alignment)
            _cos_sim = F.cosine_similarity(_cos_hidden, _cos_target_embed, dim=-1)  # (B, OUTPUT_LEN)
            _cos_loss = (1.0 - _cos_sim).mean()
            loss = loss + cos_embed_loss_weight * _cos_loss

        # Symmetry consistency loss: force identical predictions for a+b and b+a
        if symmetry_loss_weight > 0:
            with torch.no_grad():
                # Swap a and b digits in the prompt
                # Format: [0] + a_digits(10) + [0,0] + b_digits(10) + [0]
                _sym_prompts = prompts.clone()
                _sym_prompts[:, 1:1+NUM_DIGITS] = prompts[:, 13:13+NUM_DIGITS]
                _sym_prompts[:, 13:13+NUM_DIGITS] = prompts[:, 1:1+NUM_DIGITS]
                if n_think > 0:
                    _sym_think = torch.zeros(_sym_prompts.shape[0], n_think, dtype=torch.long, device=device)
                    _sym_input = torch.cat([_sym_prompts, _sym_think, targets[:, :-1]], dim=1)
                else:
                    _sym_input = torch.cat([_sym_prompts, targets[:, :-1]], dim=1)
                _sym_logits = model(_sym_input)
                _sym_output = _sym_logits[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
                _sym_target_probs = F.softmax(_sym_output, dim=-1)
            _sym_student = F.log_softmax(output_logits, dim=-1)
            _sym_loss = F.kl_div(
                _sym_student.reshape(-1, VOCAB_SIZE),
                _sym_target_probs.reshape(-1, VOCAB_SIZE),
                reduction='batchmean',
            )
            loss = loss + symmetry_loss_weight * _sym_loss

        # Loss clipping: cap the base CE loss to prevent destructive gradient spikes
        # from extremely hard carry-chain batches near convergence. When clamped,
        # the gradient is zero (batch is skipped), protecting learned carry patterns.
        if loss_clip_max > 0:
            loss = loss.clamp(max=loss_clip_max)

        # Flood loss regularization: prevent loss from going too low on easy examples
        if flood_level > 0:
            loss = (loss - flood_level).abs() + flood_level

        # Margin/hinge loss: penalize when correct digit's logit isn't margin-above best incorrect
        if margin_loss_weight > 0 and _aux_loss_active:
            _ml_logits = output_logits.reshape(-1, VOCAB_SIZE)  # (B*OUTPUT_LEN, V)
            _ml_targets_flat = targets.reshape(-1)  # (B*OUTPUT_LEN,)
            _ml_correct = _ml_logits.gather(1, _ml_targets_flat.unsqueeze(1)).squeeze(1)  # (B*OUTPUT_LEN,)
            # Best incorrect: set correct class logit to -inf, then take max
            _ml_modified = _ml_logits.clone()
            _ml_modified.scatter_(1, _ml_targets_flat.unsqueeze(1), float('-inf'))
            _ml_best_incorrect = _ml_modified.max(dim=1).values  # (B*OUTPUT_LEN,)
            # Hinge loss: max(0, margin - (correct - best_incorrect))
            _ml_gap = _ml_correct - _ml_best_incorrect
            _ml_hinge = F.relu(margin_loss_margin - _ml_gap)
            loss = loss + margin_loss_weight * _ml_hinge.mean()

        # Auxiliary carry prediction loss
        if aux_carry_weight > 0 and aux_carry_head is not None:
            with torch.no_grad():
                # Reconstruct a and b from the prompts for carry computation
                # prompts shape: (B, 24) = [0] + a_digits(10) + [0,0] + b_digits(10) + [0]
                a_digs = prompts[:, 1:1+NUM_DIGITS]  # (B, 10)
                b_digs = prompts[:, 13:13+NUM_DIGITS]  # (B, 10)
                powers_d = _DIVISORS[:NUM_DIGITS].to(device)
                a_vals = (a_digs * powers_d).sum(dim=1)
                b_vals = (b_digs * powers_d).sum(dim=1)
                carry_targets = compute_carry_targets(a_vals, b_vals, max_digits=effective_max_digits).to(device)
            # Use hidden states at output positions (stored during forward pass)
            output_hidden = model._last_hidden[:, out_offset:out_offset + OUTPUT_LEN, :]
            carry_logits = aux_carry_head(output_hidden)
            carry_loss = F.cross_entropy(carry_logits.reshape(-1, 2), carry_targets.reshape(-1))
            loss = loss + aux_carry_weight * carry_loss

        # Z-loss: penalize large logits to stabilize training near convergence
        if z_loss_weight > 0:
            # log(sum(exp(logits)))^2 averaged over batch and positions
            log_z = torch.logsumexp(output_logits, dim=-1)  # (B, OUTPUT_LEN)
            z_loss = (log_z ** 2).mean()
            loss = loss + z_loss_weight * z_loss

        # Entropy regularization: prevent overconfident wrong predictions
        if entropy_reg_weight > 0:
            probs_ent = F.softmax(output_logits, dim=-1)
            log_probs_ent = F.log_softmax(output_logits, dim=-1)
            entropy = -(probs_ent * log_probs_ent).sum(dim=-1).mean()  # scalar
            # Positive weight: loss -= weight * entropy (encourages higher entropy)
            loss = loss - entropy_reg_weight * entropy

        # Confident-wrong penalty: extra loss for high-confidence wrong predictions
        if confident_wrong_penalty > 0:
            with torch.no_grad():
                _cwp_probs = F.softmax(output_logits.detach(), dim=-1)  # (B, OUTPUT_LEN, V)
                _cwp_preds = _cwp_probs.argmax(dim=-1)  # (B, OUTPUT_LEN)
                _cwp_wrong = (_cwp_preds != targets).float()  # 1 where prediction is wrong
                _cwp_confidence = _cwp_probs.max(dim=-1).values  # max prob (confidence)
                # Weight = is_wrong * confidence * penalty_weight
                # High confidence + wrong prediction = high weight
                _cwp_weights = _cwp_wrong * _cwp_confidence * confident_wrong_penalty
            _cwp_per_digit = F.cross_entropy(
                output_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1),
                reduction='none',
            ).view(-1, OUTPUT_LEN)
            loss = loss + (_cwp_per_digit * _cwp_weights).mean()

        # Tying alignment regularization: L2 penalty pushing weights toward tied config
        if tie_align_weight > 0 and tie_align_targets and step >= tie_align_start_step:
            _ta_ramp = min((step - tie_align_start_step) / max(steps - tie_align_start_step, 1), 1.0)
            _ta_effective_w = tie_align_weight * _ta_ramp
            _ta_loss = torch.tensor(0.0, device=device)
            for _ta_target in tie_align_targets:
                if _ta_target == "down_gate":
                    for _tab in model.blocks:
                        if _tab.has_mlp and hasattr(_tab.mlp, 'gate_proj') and hasattr(_tab.mlp, 'down_proj') and not _tab.mlp.tie_down_gate:
                            _ta_loss = _ta_loss + (_tab.mlp.gate_proj.weight - _tab.mlp.down_proj.weight.T).pow(2).mean()
                elif _ta_target == "norms":
                    for _tab in model.blocks:
                        if _tab.has_mlp and hasattr(_tab, 'ln1') and hasattr(_tab, 'ln2') and _tab.ln2 is not _tab.ln1:
                            if hasattr(_tab.ln1, 'weight') and hasattr(_tab.ln2, 'weight'):
                                _ta_loss = _ta_loss + (_tab.ln1.weight - _tab.ln2.weight).pow(2).mean()
                elif _ta_target == "ln_f":
                    if hasattr(model, 'ln_f') and hasattr(model.blocks[0], 'ln1'):
                        if model.ln_f is not model.blocks[0].ln1:
                            if hasattr(model.ln_f, 'weight') and hasattr(model.blocks[0].ln1, 'weight'):
                                _ta_loss = _ta_loss + (model.ln_f.weight - model.blocks[0].ln1.weight).pow(2).mean()
            loss = loss + _ta_effective_w * _ta_loss

        # Self-distillation: use pre-computed teacher outputs (computed before student forward
        # pass to avoid inplace version counter corruption from load_state_dict)
        if self_distill_alpha > 0 and step >= self_distill_start_step and _teacher_out_pre is not None:
            teacher_out = _teacher_out_pre
            # KL-div loss with temperature scaling (softer targets = smoother gradients)
            student_log_probs = F.log_softmax(output_logits / self_distill_temp, dim=-1)
            teacher_probs = F.softmax(teacher_out / self_distill_temp, dim=-1)
            distill_loss = F.kl_div(
                student_log_probs.reshape(-1, VOCAB_SIZE),
                teacher_probs.reshape(-1, VOCAB_SIZE),
                reduction='batchmean',
            ) * (self_distill_temp ** 2)  # scale by T^2 per Hinton et al.
            loss = loss + self_distill_alpha * distill_loss

        # External knowledge distillation: KL-div between student and pre-trained teacher
        if distill_alpha > 0 and _ext_teacher_model is not None:
            with torch.no_grad():
                _ext_teacher_logits = _ext_teacher_model(full_input)
                _ext_teacher_out = _ext_teacher_logits[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
            _ext_student_log_probs = F.log_softmax(output_logits / distill_temp, dim=-1)
            _ext_teacher_probs = F.softmax(_ext_teacher_out / distill_temp, dim=-1)
            _ext_distill_loss = F.kl_div(
                _ext_student_log_probs.reshape(-1, VOCAB_SIZE),
                _ext_teacher_probs.reshape(-1, VOCAB_SIZE),
                reduction='batchmean',
            ) * (distill_temp ** 2)
            loss = loss + distill_alpha * _ext_distill_loss

        # R-Drop: compute KL divergence between two forward passes
        if rdrop_alpha > 0:
            logits2 = model(full_input)
            output_logits2 = logits2[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
            p1 = F.log_softmax(output_logits.reshape(-1, VOCAB_SIZE), dim=-1)
            p2 = F.log_softmax(output_logits2.reshape(-1, VOCAB_SIZE), dim=-1)
            kl_loss = 0.5 * (F.kl_div(p1, p2.exp(), reduction='batchmean') +
                             F.kl_div(p2, p1.exp(), reduction='batchmean'))
            loss = loss + rdrop_alpha * kl_loss

        # Scale loss for gradient accumulation
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps

        # Zero gradients at start of accumulation window (not every step)
        if grad_accum_steps <= 1 or (step - 1) % grad_accum_steps == 0:
            optimizer.zero_grad()

        loss.backward()

        # Embedding freeze: zero out embedding gradients during early training
        if embed_freeze_steps > 0 and step <= embed_freeze_steps:
            for _ef_name, _ef_p in model.named_parameters():
                if ('tok_embed' in _ef_name or 'pos_embed' in _ef_name) and _ef_p.grad is not None:
                    _ef_p.grad.zero_()

        # Gradient rescaling for multi-path tied parameters
        if grad_rescale_tied and _grad_rescale_factor < 1.0:
            for _grt_name, _grt_p in model.named_parameters():
                if 'q_proj' in _grt_name and _grt_p.grad is not None:
                    _grt_p.grad.mul_(_grad_rescale_factor)

        # Skip optimizer step until we've accumulated enough gradients
        if grad_accum_steps > 1 and step % grad_accum_steps != 0:
            continue

        # SAM: Sharpness-Aware Minimization
        if sam_rho > 0:
            # Save original gradients and compute epsilon (ascent direction)
            sam_grads = []
            grad_norm = torch.norm(
                torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]),
                2
            )
            for p in model.parameters():
                if p.grad is not None:
                    if sam_adaptive:
                        scale = sam_rho / (grad_norm + 1e-12) * (p.data.norm(2) + 1e-12)
                    else:
                        scale = sam_rho / (grad_norm + 1e-12)
                    e_w = p.grad * scale
                    p.data.add_(e_w)  # perturb weights
                    sam_grads.append(e_w)
                else:
                    sam_grads.append(None)
            # Recompute loss and gradients at perturbed point
            optimizer.zero_grad()
            logits_sam = model(full_input)
            output_logits_sam = logits_sam[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
            loss_sam = F.cross_entropy(
                output_logits_sam.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                label_smoothing=label_smoothing,
            )
            loss_sam.backward()
            # Restore original weights
            idx = 0
            for p in model.parameters():
                if sam_grads[idx] is not None:
                    p.data.sub_(sam_grads[idx])
                idx += 1

        # Gradient centralization: subtract mean from weight gradients
        if grad_centralization:
            for p in model.parameters():
                if p.grad is not None and p.dim() >= 2:
                    p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))

        # Grokfast-EMA: amplify slow-moving gradient components to accelerate grokking
        if grokfast_alpha > 0 and (grokfast_min_acc <= 0 or best_accuracy >= grokfast_min_acc):
            # Cyclical grokfast: periodically reset EMA to avoid stale amplification
            if grokfast_cycle_period > 0 and step % grokfast_cycle_period == 0:
                for p in model.parameters():
                    if hasattr(p, '_grokfast_ema'):
                        p._grokfast_ema.zero_()
            # Compute effective grokfast_lambda (may be scheduled or warmed up)
            effective_grokfast_lambda = grokfast_lambda
            # Grokfast warmup: linear ramp from 0 to target
            if grokfast_warmup_steps > 0 and step < grokfast_warmup_steps:
                effective_grokfast_lambda = grokfast_lambda * step / max(grokfast_warmup_steps, 1)
            # Accuracy-based grokfast schedule: couple lambda to learning progress
            if grokfast_acc_schedule:
                _gf_acc = best_accuracy
                effective_grokfast_lambda = grokfast_acc_schedule[0][1]
                for _gf_i in range(len(grokfast_acc_schedule) - 1):
                    _gf_a0, _gf_v0 = grokfast_acc_schedule[_gf_i]
                    _gf_a1, _gf_v1 = grokfast_acc_schedule[_gf_i + 1]
                    if _gf_a0 <= _gf_acc < _gf_a1:
                        _gf_frac = (_gf_acc - _gf_a0) / max(_gf_a1 - _gf_a0, 1e-8)
                        effective_grokfast_lambda = _gf_v0 + _gf_frac * (_gf_v1 - _gf_v0)
                        break
                    elif _gf_acc >= _gf_a1:
                        effective_grokfast_lambda = _gf_v1
            elif grokfast_lambda_schedule:
                effective_grokfast_lambda = grokfast_lambda_schedule[0][1]
                for i in range(len(grokfast_lambda_schedule) - 1):
                    s0, v0 = grokfast_lambda_schedule[i]
                    s1, v1 = grokfast_lambda_schedule[i + 1]
                    if s0 <= step < s1:
                        frac = (step - s0) / max(s1 - s0, 1)
                        effective_grokfast_lambda = v0 + frac * (v1 - v0)
                        break
                    elif step >= s1:
                        effective_grokfast_lambda = v1
            for p in model.parameters():
                if p.grad is not None:
                    if not hasattr(p, '_grokfast_ema'):
                        p._grokfast_ema = torch.zeros_like(p.grad)
                    p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                    if grokfast_adaptive and p.grad.numel() > 1:
                        # Adaptive: scale lambda by gradient-EMA alignment
                        _gfa_cos = F.cosine_similarity(
                            p._grokfast_ema.flatten().unsqueeze(0),
                            p.grad.flatten().unsqueeze(0)
                        ).item()
                        _gfa_lambda = effective_grokfast_lambda * (0.5 + 0.5 * _gfa_cos)
                        p.grad.add_(p._grokfast_ema, alpha=_gfa_lambda)
                    else:
                        p.grad.add_(p._grokfast_ema, alpha=effective_grokfast_lambda)

        # Gradient noise injection (Neelakantan et al. 2015)
        if grad_noise_eta > 0:
            noise_std = math.sqrt(grad_noise_eta / (1 + step) ** grad_noise_gamma)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * noise_std)

        # Adaptive gradient clipping (AGC): clip based on grad_norm / param_norm ratio
        if agc_clip_factor > 0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None and p.dim() >= 2:
                        param_norm = p.data.norm(2).clamp(min=1e-6)
                        grad_norm = p.grad.norm(2)
                        max_norm = param_norm * agc_clip_factor
                        if grad_norm > max_norm:
                            p.grad.mul_(max_norm / (grad_norm + 1e-6))
            # Standard clipping still applies to 1D params (norms etc.)
            if grad_clip > 0:
                one_d_params = [p for p in model.parameters() if p.grad is not None and p.dim() < 2]
                if one_d_params:
                    torch.nn.utils.clip_grad_norm_(one_d_params, grad_clip)
        elif grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Warm-start gradient scaling: reduce gradients on loaded params to protect representations
        if warm_start_lr_mult != 1.0 and _warm_loaded_names:
            if warm_start_thaw_steps <= 0 or step <= warm_start_thaw_steps:
                for _wn, _wp in model.named_parameters():
                    if _wp.grad is not None and hasattr(_wp, '_warm_loaded'):
                        _wp.grad.mul_(warm_start_lr_mult)

        optimizer.step()

        # Periodic optimizer state reset: clear adaptive statistics for fresh exploration
        if opt_reset_interval > 0 and step > 0 and step % opt_reset_interval == 0:
            optimizer.state.clear()
            print(f"  Optimizer state reset at step {step}")

        # Stochastic weight perturbation: add noise to escape local optima
        # Uses non-inplace assignment (p.data = ...) instead of p.data.add_() to avoid
        # bumping the autograd version counter, which can cause RuntimeError when combined
        # with features like self_distill that build computation graphs across steps.
        if weight_perturb_std > 0 and step % weight_perturb_interval == 0:
            current_perturb_std = weight_perturb_std * (weight_perturb_decay ** (step // weight_perturb_interval))
            with torch.no_grad():
                for p in model.parameters():
                    if p.dim() >= 2:  # only perturb weight matrices, not norms
                        p.data = p.data + torch.randn_like(p) * current_perturb_std

        # Lookahead: interpolate slow weights every k steps
        if lookahead_k > 0:
            if lookahead_state is None:
                lookahead_state = {k: v.clone() for k, v in model.state_dict().items()}
            lookahead_counter += 1
            if lookahead_counter >= lookahead_k:
                lookahead_counter = 0
                with torch.no_grad():
                    for k in lookahead_state:
                        diff = model.state_dict()[k] - lookahead_state[k]
                        lookahead_state[k].add_(diff, alpha=lookahead_alpha)
                    model.load_state_dict(lookahead_state)

        # EMA update: exponential moving average of weights
        if ema_decay > 0 and step >= ema_start_step:
            if ema_state is None:
                ema_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                with torch.no_grad():
                    for k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(model.state_dict()[k], alpha=1 - ema_decay)

        # Stochastic Weight Averaging update
        if swa_start_frac > 0 and step >= int(steps * swa_start_frac) and step % swa_update_freq == 0:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                swa_count += 1
                for k in swa_state:
                    swa_state[k].add_(model.state_dict()[k] - swa_state[k], alpha=1.0 / swa_count)

        # Evaluate
        if step % eval_every == 0 or step == steps:
            # Curriculum-aware eval for progress logging (batched for speed)
            curr_acc = evaluate_model_batched(model, device, num_tests=500, max_digits=max_digits)
            # Full 10-digit eval for best_state selection and early stopping (batched for speed)
            full_acc = evaluate_model_batched(model, device, num_tests=500)
            elapsed = time.time() - t0
            if max_digits < NUM_DIGITS:
                print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | Acc@{max_digits}d: {curr_acc:.4f} | Acc@10d: {full_acc:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.0f}s")
            else:
                print(f"Step {step}/{steps} | Loss: {loss.item():.4f} | Acc: {full_acc:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.0f}s")

            # Carry chain breakdown: report accuracy by carry chain length
            if eval_carry_breakdown and max_digits >= NUM_DIGITS and full_acc > 0:
                _cb = evaluate_model_carry_breakdown(model, device, num_tests=500)
                _cb_parts = [f"{k}: {v[2]:.3f} ({v[0]}/{v[1]})" for k, v in _cb.items() if v[1] > 0]
                print(f"  Carry breakdown: {' | '.join(_cb_parts)}")

            if full_acc > best_accuracy:
                best_accuracy = full_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
                # Grokfast improve decay: partially reset EMA to adapt to new loss landscape
                if grokfast_improve_decay > 0 and grokfast_alpha > 0:
                    _gid_factor = 1.0 - grokfast_improve_decay
                    for _gid_p in model.parameters():
                        if hasattr(_gid_p, '_grokfast_ema'):
                            _gid_p._grokfast_ema.mul_(_gid_factor)
            else:
                steps_since_improvement += eval_every

            # Stall LR bump: trigger when accuracy stalls for too long
            if (stall_lr_bump_factor > 0 and stall_lr_bump_patience > 0 and
                    steps_since_improvement >= stall_lr_bump_patience and
                    best_accuracy > 0 and max_digits >= NUM_DIGITS):
                # Only bump if not already in a bump
                if _stall_bump_step is None or (step - _stall_bump_step) >= stall_lr_bump_duration * 2:
                    _stall_bump_step = step
                    print(f"  Stall LR bump: {stall_lr_bump_factor}x for {stall_lr_bump_duration} steps (stalled {steps_since_improvement} steps)")

            # Plateau optimizer restart: reset optimizer state + LR boost when stuck
            if (plateau_restart_patience > 0 and
                    steps_since_improvement >= plateau_restart_patience and
                    best_accuracy > 0 and max_digits >= NUM_DIGITS):
                # Only trigger if not already in a boost period
                if (_plateau_restart_triggered_at is None or
                        (step - _plateau_restart_triggered_at) >= plateau_restart_boost_steps * 2):
                    _plateau_restart_triggered_at = step
                    _plateau_restart_count += 1
                    # Reset optimizer state (clear all momentum/variance)
                    optimizer.state.clear()
                    # Also partially reset grokfast EMA to allow fresh exploration
                    if grokfast_alpha > 0:
                        for _pr_p in model.parameters():
                            if hasattr(_pr_p, '_grokfast_ema'):
                                _pr_p._grokfast_ema.mul_(0.5)
                    print(f"  Plateau restart #{_plateau_restart_count}: optimizer reset + {plateau_restart_lr_boost}x LR for {plateau_restart_boost_steps} steps (stalled {steps_since_improvement} steps)")

            # Checkpoint soup: maintain top-K checkpoints by accuracy
            if ckpt_soup_k > 0 and full_acc > 0:
                state_copy = {k: v.clone() for k, v in model.state_dict().items()}
                ckpt_soup.append((full_acc, state_copy))
                ckpt_soup.sort(key=lambda x: x[0], reverse=True)
                if len(ckpt_soup) > ckpt_soup_k:
                    ckpt_soup.pop()  # remove worst

            # Early stop if perfect on full 10-digit
            if full_acc >= 1.0:
                print(f"Perfect accuracy at step {step}!")
                break

            # Patience: only after curriculum reaches 10 digits and model has shown learning
            if max_digits >= NUM_DIGITS and steps_since_improvement >= patience and 0 < best_accuracy < 0.99:
                print(f"No improvement for {patience} steps, stopping.")
                break

            # Divergence stop: model collapsed to 0 accuracy and hasn't recovered
            if max_digits >= NUM_DIGITS and full_acc == 0.0 and steps_since_improvement >= 100_000:
                print(f"Model diverged (acc=0) for {steps_since_improvement} steps, stopping.")
                break

            # Time limit: stop gracefully before the process is killed by the runner
            if time_limit > 0 and elapsed >= time_limit:
                print(f"Time limit {time_limit:.0f}s reached at step {step}, stopping.")
                break

    # Carry focus phase: targeted fine-tuning on carry-heavy problems
    # Helps push past the 97-98% plateau by focusing the model's limited
    # capacity on hard carry chains after initial training converges
    carry_focus_steps_cfg = int(cfg.get("carry_focus_steps", 0))  # 0 = disabled (default)
    carry_focus_prob_cfg = float(cfg.get("carry_focus_prob", 0.8))  # carry probability during focus
    carry_focus_lr_mult = float(cfg.get("carry_focus_lr_mult", 0.1))  # LR multiplier for focus phase
    if carry_focus_steps_cfg > 0 and best_state is not None and best_accuracy > 0:
        print(f"Carry focus phase: {carry_focus_steps_cfg} steps, carry_prob={carry_focus_prob_cfg}")
        model.load_state_dict(best_state)
        model.train()
        focus_lr = min_lr + carry_focus_lr_mult * (lr - min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = focus_lr
        # Reset grokfast EMA for fresh start in focus phase
        for p in model.parameters():
            if hasattr(p, '_grokfast_ema'):
                p._grokfast_ema.zero_()
        focus_best_acc = best_accuracy
        focus_best_state = None
        for fs in range(1, carry_focus_steps_cfg + 1):
            prompts_f, targets_f = generate_carry_biased_batch(
                batch_size, max_digits=NUM_DIGITS, carry_prob=carry_focus_prob_cfg, device=device)
            if n_think > 0:
                think_f = torch.zeros(batch_size, n_think, dtype=torch.long, device=device)
                full_input_f = torch.cat([prompts_f, think_f, targets_f[:, :-1]], dim=1)
            else:
                full_input_f = torch.cat([prompts_f, targets_f[:, :-1]], dim=1)
            logits_f = model(full_input_f)
            out_f = logits_f[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
            loss_f = F.cross_entropy(out_f.reshape(-1, VOCAB_SIZE), targets_f.reshape(-1))
            optimizer.zero_grad()
            loss_f.backward()
            if grokfast_alpha > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        if not hasattr(p, '_grokfast_ema'):
                            p._grokfast_ema = torch.zeros_like(p.grad)
                        p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                        p.grad.add_(p._grokfast_ema, alpha=grokfast_lambda)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if fs % eval_every == 0 or fs == carry_focus_steps_cfg:
                acc_f = evaluate_model_batched(model, device, num_tests=500)
                elapsed_f = time.time() - t0
                print(f"  Focus {fs}/{carry_focus_steps_cfg} | Loss: {loss_f.item():.4f} | Acc: {acc_f:.4f} | Time: {elapsed_f:.0f}s")
                if acc_f > focus_best_acc:
                    focus_best_acc = acc_f
                    focus_best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if time_limit > 0 and (time.time() - t0) >= time_limit:
                print(f"Time limit reached during carry focus phase")
                break
        if focus_best_state is not None:
            print(f"Carry focus improved: {best_accuracy:.4f} -> {focus_best_acc:.4f}")
            best_accuracy = focus_best_acc
            best_state = focus_best_state
        model.load_state_dict(best_state)

    # Auto multi-phase training: additional phases from best checkpoint with lower LR + EMA
    # Ensure annealed tying is fully complete (use tied K, not staging K_proj) during phases
    if anneal_tying_steps > 0:
        for _am in model.modules():
            if isinstance(_am, Attention) and _am.anneal_tying:
                _am._anneal_progress = 1.0
    _phase_soup_states = []  # collect best state from each phase for phase_soup
    if best_state is not None and phase_soup:
        _phase_soup_states.append({k: v.clone() for k, v in best_state.items()})
    if auto_phases > 0 and best_state is not None and best_accuracy > 0:
        phase_steps = auto_phase_steps if auto_phase_steps > 0 else max(steps // 4, 10000)
        phase_lr = lr
        for phase_idx in range(1, auto_phases + 1):
            if time_limit > 0 and (time.time() - t0) >= time_limit:
                print(f"Time limit reached before auto phase {phase_idx}")
                break
            phase_lr = phase_lr * auto_phase_lr_decay
            phase_min_lr = min_lr * auto_phase_lr_decay
            print(f"Auto phase {phase_idx}/{auto_phases}: {phase_steps} steps, lr={phase_lr:.6f}, ema={auto_phase_ema}")
            model.load_state_dict(best_state)
            model.train()
            # Reset optimizer state for fresh start
            phase_optimizer = torch.optim.AdamW(model.parameters(), lr=phase_lr, weight_decay=weight_decay)
            # Reset grokfast EMA
            for p in model.parameters():
                if hasattr(p, '_grokfast_ema'):
                    p._grokfast_ema.zero_()
            # Phase EMA state
            phase_ema_state = {k: v.clone() for k, v in model.state_dict().items()}
            phase_best_acc = best_accuracy
            phase_best_state = None
            _phase_gf_lambda = auto_phase_grokfast_lambda if auto_phase_grokfast_lambda > 0 else grokfast_lambda
            # Phase carry escalation: increase carry chain difficulty per phase
            if auto_phase_carry_escalate:
                _esc_chain_len = min_carry_chain_len + phase_idx  # 4, 5, 6, ...
                _esc_chain_prob = min(0.3 + 0.15 * phase_idx, 0.8)  # 0.45, 0.6, 0.75, ...
                _esc_use_long_carry = True
                print(f"  Carry escalation: chain_len={_esc_chain_len}, chain_prob={_esc_chain_prob:.2f}")
            else:
                _esc_chain_len = min_carry_chain_len
                _esc_chain_prob = long_carry_chain_prob
                _esc_use_long_carry = long_carry_chain_prob > 0

            for ps in range(1, phase_steps + 1):
                # Cosine LR within phase (with optional warmup and restarts)
                if auto_phase_warmup_steps > 0 and ps <= auto_phase_warmup_steps:
                    # Linear warmup from 0 to phase_lr
                    current_phase_lr = phase_lr * ps / max(auto_phase_warmup_steps, 1)
                else:
                    _effective_ps = ps - auto_phase_warmup_steps if auto_phase_warmup_steps > 0 else ps
                    _effective_total = max(phase_steps - auto_phase_warmup_steps, 1) if auto_phase_warmup_steps > 0 else max(phase_steps, 1)
                    # LR restarts within phase: divide into N cosine cycles
                    if auto_phase_restarts > 1:
                        _restart_period = max(_effective_total // auto_phase_restarts, 1)
                        _effective_ps = _effective_ps % _restart_period
                        _effective_total = _restart_period
                    phase_progress = _effective_ps / _effective_total
                    current_phase_lr = phase_min_lr + 0.5 * (phase_lr - phase_min_lr) * (1 + math.cos(math.pi * phase_progress))
                for pg in phase_optimizer.param_groups:
                    pg["lr"] = current_phase_lr
                # Determine max_digits for this phase step (auto-phase curriculum or full 10-digit)
                _phase_max_digits = NUM_DIGITS
                if auto_phase_curriculum_stages:
                    for _apc_digits, _apc_step_str in auto_phase_curriculum_stages:
                        if _apc_step_str == "rest":
                            _phase_max_digits = _apc_digits
                            break
                        if ps < int(_apc_step_str):
                            _phase_max_digits = _apc_digits
                            break
                # Generate batch (with optional carry bias, long carry chains, or OHEM)
                _phase_carry = auto_phase_carry_bias if auto_phase_carry_bias > 0 else carry_bias
                _phase_gen_bs = int(batch_size * auto_phase_ohem_ratio) if auto_phase_ohem_ratio > 1.0 else batch_size

                # Reverse carry curriculum: start phase with hardest problems, ease over time
                _use_reverse_carry = False
                if auto_phase_reverse_carry:
                    _rc_progress = ps / max(phase_steps, 1)  # 0 at start, 1 at end
                    # Start: min_chain_len=6, chain_prob=0.9 (maximum difficulty)
                    # End: min_chain_len=3, chain_prob=0.3 (moderate difficulty)
                    _rc_chain_len = max(3, int(6 - 3 * _rc_progress))
                    _rc_chain_prob = 0.9 - 0.6 * _rc_progress  # 0.9 → 0.3
                    _use_reverse_carry = True

                if _use_reverse_carry:
                    prompts_p, targets_p = generate_long_carry_chain_batch(
                        _phase_gen_bs, max_digits=_phase_max_digits, chain_prob=_rc_chain_prob,
                        min_chain_len=_rc_chain_len, device=device)
                elif _esc_use_long_carry:
                    prompts_p, targets_p = generate_long_carry_chain_batch(
                        _phase_gen_bs, max_digits=_phase_max_digits, chain_prob=_esc_chain_prob,
                        min_chain_len=_esc_chain_len, device=device)
                elif _phase_carry > 0:
                    prompts_p, targets_p = generate_carry_biased_batch(
                        _phase_gen_bs, max_digits=_phase_max_digits, carry_prob=_phase_carry, device=device)
                else:
                    prompts_p, targets_p = generate_batch(_phase_gen_bs, max_digits=_phase_max_digits, device=device)
                if n_think > 0:
                    think_p = torch.zeros(_phase_gen_bs, n_think, dtype=torch.long, device=device)
                    full_input_p = torch.cat([prompts_p, think_p, targets_p[:, :-1]], dim=1)
                else:
                    full_input_p = torch.cat([prompts_p, targets_p[:, :-1]], dim=1)
                # OHEM in auto-phases: select hardest examples from oversampled batch
                if auto_phase_ohem_ratio > 1.0:
                    with torch.no_grad():
                        _ohem_p_logits = model(full_input_p)
                        _ohem_p_out = _ohem_p_logits[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
                        _ohem_p_per_digit_loss = F.cross_entropy(
                            _ohem_p_out.reshape(-1, VOCAB_SIZE), targets_p.reshape(-1),
                            reduction='none').view(-1, OUTPUT_LEN)
                        if auto_phase_ohem_per_digit:
                            # Per-digit OHEM: select hardest samples for each digit, then union
                            _ap_per_digit_k = max(batch_size // OUTPUT_LEN, 1)
                            _ap_ohem_selected = set()
                            for _ap_dig in range(OUTPUT_LEN):
                                _, _ap_dig_idx = _ohem_p_per_digit_loss[:, _ap_dig].topk(
                                    min(_ap_per_digit_k, _ohem_p_per_digit_loss.shape[0]))
                                _ap_ohem_selected.update(_ap_dig_idx.tolist())
                            _ohem_p_overall = _ohem_p_per_digit_loss.mean(dim=1)
                            _, _ap_overall_idx = _ohem_p_overall.topk(min(batch_size, _ohem_p_per_digit_loss.shape[0]))
                            for _ap_oi in _ap_overall_idx.tolist():
                                if len(_ap_ohem_selected) >= batch_size:
                                    break
                                _ap_ohem_selected.add(_ap_oi)
                            _ohem_p_idx = torch.tensor(list(_ap_ohem_selected)[:batch_size], device=device)
                        else:
                            _ohem_p_loss = _ohem_p_per_digit_loss.mean(dim=1)
                            _, _ohem_p_idx = _ohem_p_loss.topk(batch_size)
                    full_input_p = full_input_p[_ohem_p_idx]
                    targets_p = targets_p[_ohem_p_idx]
                    prompts_p = prompts_p[_ohem_p_idx]
                logits_p = model(full_input_p)
                out_p = logits_p[:, out_offset:out_offset + OUTPUT_LEN, :VOCAB_SIZE]
                # Auto-phase focal loss: concentrate gradient on hard carry-chain digits
                if auto_phase_focal_gamma > 0:
                    _ap_fl_per_token = F.cross_entropy(
                        out_p.reshape(-1, VOCAB_SIZE), targets_p.reshape(-1),
                        reduction='none')
                    with torch.no_grad():
                        _ap_fl_probs = F.softmax(out_p.reshape(-1, VOCAB_SIZE), dim=-1)
                        _ap_fl_pt = _ap_fl_probs.gather(1, targets_p.reshape(-1, 1)).squeeze(1)
                        _ap_fl_weight = (1 - _ap_fl_pt) ** auto_phase_focal_gamma
                    loss_p = (_ap_fl_per_token * _ap_fl_weight).mean()
                else:
                    loss_p = F.cross_entropy(out_p.reshape(-1, VOCAB_SIZE), targets_p.reshape(-1))
                # Soft OHEM in auto-phases: per-sample importance weighting
                if auto_phase_soft_ohem_gamma > 0:
                    _ap_soh_per_token = F.cross_entropy(
                        out_p.reshape(-1, VOCAB_SIZE), targets_p.reshape(-1),
                        reduction='none').view(-1, OUTPUT_LEN)
                    _ap_soh_per_sample = _ap_soh_per_token.mean(dim=1)
                    with torch.no_grad():
                        _ap_soh_mean = _ap_soh_per_sample.mean().clamp(min=1e-8)
                        _ap_soh_w = (_ap_soh_per_sample.detach() / _ap_soh_mean) ** auto_phase_soft_ohem_gamma
                        _ap_soh_w = _ap_soh_w / _ap_soh_w.mean()
                    loss_p = (_ap_soh_per_token * _ap_soh_w.unsqueeze(1)).mean()
                # Adaptive per-digit loss weighting in auto-phases
                if auto_phase_adaptive_digit_weight > 0:
                    with torch.no_grad():
                        _ap_adw_preds = out_p.argmax(dim=-1)
                        _ap_adw_correct = (_ap_adw_preds == targets_p).float().mean(dim=0)
                    _ap_adw_weights = (1.0 - _ap_adw_correct.clamp(0, 1)) * auto_phase_adaptive_digit_weight + 1.0
                    _ap_adw_weights = _ap_adw_weights / _ap_adw_weights.mean()
                    _ap_adw_per_digit = F.cross_entropy(
                        out_p.reshape(-1, VOCAB_SIZE), targets_p.reshape(-1),
                        reduction='none').view(-1, OUTPUT_LEN)
                    loss_p = (_ap_adw_per_digit * _ap_adw_weights.unsqueeze(0)).mean()
                phase_optimizer.zero_grad()
                loss_p.backward()
                # Grokfast in phases (with optional lambda override)
                if grokfast_alpha > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            if not hasattr(p, '_grokfast_ema'):
                                p._grokfast_ema = torch.zeros_like(p.grad)
                            p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                            p.grad.add_(p._grokfast_ema, alpha=_phase_gf_lambda)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                phase_optimizer.step()
                # EMA update
                with torch.no_grad():
                    for k in phase_ema_state:
                        phase_ema_state[k].mul_(auto_phase_ema).add_(model.state_dict()[k], alpha=1 - auto_phase_ema)
                # Evaluate
                if ps % eval_every == 0 or ps == phase_steps:
                    acc_p = evaluate_model_batched(model, device, num_tests=500)
                    # Also check EMA accuracy
                    ema_pre = {k: v.clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(phase_ema_state)
                    ema_acc_p = evaluate_model_batched(model, device, num_tests=500)
                    model.load_state_dict(ema_pre)
                    use_acc = max(acc_p, ema_acc_p)
                    elapsed_p = time.time() - t0
                    print(f"  Phase {phase_idx} step {ps}/{phase_steps} | Loss: {loss_p.item():.4f} | Acc: {acc_p:.4f} | EMA: {ema_acc_p:.4f} | Time: {elapsed_p:.0f}s")
                    if use_acc > phase_best_acc:
                        phase_best_acc = use_acc
                        if ema_acc_p >= acc_p:
                            phase_best_state = {k: v.clone() for k, v in phase_ema_state.items()}
                        else:
                            phase_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    print(f"Time limit reached during auto phase {phase_idx}")
                    break
            if phase_best_state is not None:
                print(f"  Phase {phase_idx} improved: {best_accuracy:.4f} -> {phase_best_acc:.4f}")
                best_accuracy = phase_best_acc
                best_state = phase_best_state
                if phase_soup:
                    _phase_soup_states.append({k: v.clone() for k, v in phase_best_state.items()})
            else:
                print(f"  Phase {phase_idx} no improvement (best remains {best_accuracy:.4f})")
        model.load_state_dict(best_state)

    # Apply SWA weights if available, otherwise restore best checkpoint
    if swa_state is not None and swa_count > 1:
        print(f"Applying SWA (averaged {swa_count} checkpoints)")
        model.load_state_dict(swa_state)
        swa_acc = evaluate_model_batched(model, device, num_tests=500)
        best_acc_val = best_accuracy
        if swa_acc >= best_acc_val:
            print(f"SWA accuracy {swa_acc:.4f} >= best {best_acc_val:.4f}, using SWA weights")
        elif best_state is not None:
            print(f"SWA accuracy {swa_acc:.4f} < best {best_acc_val:.4f}, reverting to best checkpoint")
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    # Try EMA weights: often better than best checkpoint for tiny models
    if ema_state is not None:
        ema_model_state = model.state_dict()
        model.load_state_dict(ema_state)
        ema_acc = evaluate_model_batched(model, device, num_tests=500)
        current_acc = evaluate_model_batched(model, device, num_tests=500) if best_state is None else best_accuracy
        if ema_acc >= current_acc:
            print(f"EMA accuracy {ema_acc:.4f} >= best {current_acc:.4f}, using EMA weights")
            best_accuracy = max(best_accuracy, ema_acc)
        else:
            print(f"EMA accuracy {ema_acc:.4f} < best {current_acc:.4f}, reverting")
            model.load_state_dict(ema_model_state)

    # Try checkpoint soup: average top-K checkpoints
    if ckpt_soup_k > 0 and len(ckpt_soup) >= 2:
        soup_state = {}
        for k in ckpt_soup[0][1]:
            soup_state[k] = sum(s[1][k] for s in ckpt_soup) / len(ckpt_soup)
        pre_soup_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup_state)
        soup_acc = evaluate_model_batched(model, device, num_tests=500)
        current_acc = best_accuracy
        if soup_acc >= current_acc:
            print(f"Checkpoint soup ({len(ckpt_soup)} ckpts) accuracy {soup_acc:.4f} >= best {current_acc:.4f}, using soup")
            best_accuracy = max(best_accuracy, soup_acc)
        else:
            print(f"Checkpoint soup accuracy {soup_acc:.4f} < best {current_acc:.4f}, reverting")
            model.load_state_dict(pre_soup_state)

    # Phase soup: average best checkpoints from each auto-phase
    if phase_soup and len(_phase_soup_states) >= 2:
        _ps_state = {}
        for _ps_k in _phase_soup_states[0]:
            _ps_state[_ps_k] = sum(s[_ps_k] for s in _phase_soup_states) / len(_phase_soup_states)
        _ps_pre = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(_ps_state)
        _ps_acc = evaluate_model_batched(model, device, num_tests=500)
        if _ps_acc >= best_accuracy:
            print(f"Phase soup ({len(_phase_soup_states)} phases) accuracy {_ps_acc:.4f} >= best {best_accuracy:.4f}, using phase soup")
            best_accuracy = max(best_accuracy, _ps_acc)
        else:
            print(f"Phase soup accuracy {_ps_acc:.4f} < best {best_accuracy:.4f}, reverting")
            model.load_state_dict(_ps_pre)

    # Evolution Strategy (ES) fine-tuning: gradient-free optimization
    # Uses Natural Evolution Strategies (NES) to escape flat loss landscapes near convergence.
    # Each step: create population random perturbations, evaluate each, update toward better ones.
    if es_finetune_steps > 0 and best_accuracy > 0:
        if time_limit > 0 and (time.time() - t0) >= time_limit:
            print("Time limit reached before ES fine-tuning")
        else:
            print(f"ES fine-tuning: {es_finetune_steps} steps, pop={es_population}, sigma={es_sigma}")
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            # Flatten model parameters into a single vector for efficient perturbation
            _es_params = [p for p in model.parameters() if p.requires_grad]
            _es_shapes = [p.shape for p in _es_params]
            _es_sizes = [p.numel() for p in _es_params]
            _es_total = sum(_es_sizes)

            def _es_flatten():
                return torch.cat([p.data.view(-1) for p in _es_params])

            def _es_unflatten(flat):
                offset = 0
                for p, s in zip(_es_params, _es_sizes):
                    p.data.copy_(flat[offset:offset + s].view(p.shape))
                    offset += s

            _es_base = _es_flatten()
            _es_best_acc = best_accuracy
            _es_best_flat = _es_base.clone()

            for _es_step in range(1, es_finetune_steps + 1):
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    print(f"Time limit reached during ES step {_es_step}")
                    break

                # Generate population of noise vectors
                _es_noise = torch.randn(es_population, _es_total, device=device) * es_sigma
                _es_fitnesses = torch.zeros(es_population, device=device)

                for _es_i in range(es_population):
                    # Apply perturbation
                    _es_unflatten(_es_base + _es_noise[_es_i])
                    # Evaluate (use batched eval for speed)
                    _es_acc = evaluate_model_batched(model, device, num_tests=es_eval_samples)
                    _es_fitnesses[_es_i] = _es_acc

                # Fitness-weighted update (NES): weight noise vectors by relative fitness
                _es_fitnesses_centered = _es_fitnesses - _es_fitnesses.mean()
                _es_fitnesses_std = _es_fitnesses.std() + 1e-8
                _es_fitnesses_norm = _es_fitnesses_centered / _es_fitnesses_std
                # Weighted sum of noise vectors: step in direction of high-fitness perturbations
                _es_grad = (_es_fitnesses_norm.unsqueeze(1) * _es_noise).mean(dim=0)
                _es_base = _es_base + (es_sigma * _es_grad)

                # Check if best candidate improved
                _es_max_fit = _es_fitnesses.max().item()
                _es_max_idx = _es_fitnesses.argmax().item()
                if _es_max_fit > _es_best_acc:
                    _es_best_acc = _es_max_fit
                    _es_best_flat = (_es_base + _es_noise[_es_max_idx]).clone()

                if _es_step % 5 == 0 or _es_step == es_finetune_steps:
                    print(f"  ES step {_es_step}/{es_finetune_steps} | best={_es_best_acc:.4f} | pop_mean={_es_fitnesses.mean():.4f} | pop_max={_es_max_fit:.4f}")

            # Restore best found weights
            _es_unflatten(_es_best_flat)
            _es_final_acc = evaluate_model_batched(model, device, num_tests=2000)
            if _es_final_acc >= best_accuracy:
                print(f"ES improved: {best_accuracy:.4f} -> {_es_final_acc:.4f}")
                best_accuracy = _es_final_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                print(f"ES final accuracy {_es_final_acc:.4f} < best {best_accuracy:.4f}, reverting")
                if best_state is not None:
                    model.load_state_dict(best_state)

    # Coordinate-descent weight refinement: systematic per-parameter optimization.
    # Unlike ES (random direction in full param space), this sweeps each parameter
    # individually, testing +delta and -delta, keeping changes that improve accuracy.
    # For 52 params, each sweep is ~104 evaluations. Deterministic and exhaustive
    # within its search radius, complementing ES's stochastic exploration.
    if coord_refine_steps > 0 and best_accuracy > 0:
        if time_limit > 0 and (time.time() - t0) >= time_limit:
            print("Time limit reached before coordinate refinement")
        else:
            print(f"Coordinate refinement: {coord_refine_steps} sweeps, delta={coord_refine_delta}")
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            _cr_params = [p for p in model.parameters() if p.requires_grad]
            _cr_best_acc = best_accuracy
            _cr_improved_total = 0

            for _cr_sweep in range(1, coord_refine_steps + 1):
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    print(f"Time limit reached during coordinate sweep {_cr_sweep}")
                    break
                _cr_improved = 0
                for _cr_pi, _cr_p in enumerate(_cr_params):
                    _cr_flat = _cr_p.data.view(-1)
                    for _cr_j in range(_cr_flat.numel()):
                        _cr_orig = _cr_flat[_cr_j].item()
                        _cr_best_val = _cr_orig
                        # Try positive perturbation
                        _cr_flat[_cr_j] = _cr_orig + coord_refine_delta
                        _cr_acc_plus = evaluate_model_batched(model, device, num_tests=coord_refine_eval_samples)
                        if _cr_acc_plus > _cr_best_acc:
                            _cr_best_acc = _cr_acc_plus
                            _cr_best_val = _cr_orig + coord_refine_delta
                            _cr_improved += 1
                        # Try negative perturbation
                        _cr_flat[_cr_j] = _cr_orig - coord_refine_delta
                        _cr_acc_minus = evaluate_model_batched(model, device, num_tests=coord_refine_eval_samples)
                        if _cr_acc_minus > _cr_best_acc:
                            _cr_best_acc = _cr_acc_minus
                            _cr_best_val = _cr_orig - coord_refine_delta
                            _cr_improved += 1
                        # Set to best value found
                        _cr_flat[_cr_j] = _cr_best_val
                        if time_limit > 0 and (time.time() - t0) >= time_limit:
                            break
                    if time_limit > 0 and (time.time() - t0) >= time_limit:
                        break
                _cr_improved_total += _cr_improved
                print(f"  Coord sweep {_cr_sweep}/{coord_refine_steps} | acc={_cr_best_acc:.4f} | improved={_cr_improved} params")

            if _cr_best_acc > best_accuracy:
                print(f"Coord refinement improved: {best_accuracy:.4f} -> {_cr_best_acc:.4f} ({_cr_improved_total} improvements)")
                best_accuracy = _cr_best_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                print(f"Coord refinement no improvement (best remains {best_accuracy:.4f})")
                if best_state is not None:
                    model.load_state_dict(best_state)

    # CMA-ES fine-tuning: Covariance Matrix Adaptation Evolution Strategy.
    # Learns the covariance structure of the 52-param fitness landscape to search
    # along principal axes. Much more sample-efficient than isotropic NES for tiny models.
    if cmaes_steps > 0 and best_accuracy > 0:
        if time_limit > 0 and (time.time() - t0) >= time_limit:
            print("Time limit reached before CMA-ES fine-tuning")
        else:
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()

            _cma_params = [p for p in model.parameters() if p.requires_grad]
            _cma_sizes = [p.numel() for p in _cma_params]
            _cma_n = sum(_cma_sizes)

            def _cma_flatten():
                return torch.cat([p.data.view(-1) for p in _cma_params])

            def _cma_unflatten(flat):
                offset = 0
                for p, s in zip(_cma_params, _cma_sizes):
                    p.data.copy_(flat[offset:offset + s].view(p.shape))
                    offset += s

            # CMA-ES parameters (Hansen 2016 defaults)
            _cma_lam = cmaes_population if cmaes_population > 0 else 4 + int(3 * math.log(_cma_n))
            _cma_mu = _cma_lam // 2
            # Recombination weights: log-linear ranking
            _cma_raw_w = torch.tensor([math.log((_cma_lam + 1) / 2) - math.log(i + 1) for i in range(_cma_mu)])
            _cma_weights = _cma_raw_w / _cma_raw_w.sum()
            _cma_mueff = 1.0 / (_cma_weights ** 2).sum().item()
            # Adaptation rates
            _cma_cc = (4 + _cma_mueff / _cma_n) / (_cma_n + 4 + 2 * _cma_mueff / _cma_n)
            _cma_cs = (_cma_mueff + 2) / (_cma_n + _cma_mueff + 5)
            _cma_c1 = 2 / ((_cma_n + 1.3) ** 2 + _cma_mueff)
            _cma_cmu = min(1 - _cma_c1, 2 * (_cma_mueff - 2 + 1 / _cma_mueff) / ((_cma_n + 2) ** 2 + _cma_mueff))
            _cma_damps = 1 + 2 * max(0, math.sqrt((_cma_mueff - 1) / (_cma_n + 1)) - 1) + _cma_cs
            _cma_chi_n = math.sqrt(_cma_n) * (1 - 1 / (4 * _cma_n) + 1 / (21 * _cma_n ** 2))

            # State variables
            _cma_mean = _cma_flatten().cpu().float()
            _cma_sigma = cmaes_sigma
            _cma_C = torch.eye(_cma_n)  # covariance matrix
            _cma_pc = torch.zeros(_cma_n)  # evolution path for C
            _cma_ps = torch.zeros(_cma_n)  # evolution path for sigma
            _cma_best_acc = best_accuracy
            _cma_best_flat = _cma_mean.clone()

            print(f"CMA-ES fine-tuning: {cmaes_steps} generations, pop={_cma_lam}, mu={_cma_mu}, n={_cma_n}, sigma={_cma_sigma:.4f}")

            for _cma_gen in range(1, cmaes_steps + 1):
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    print(f"Time limit reached during CMA-ES generation {_cma_gen}")
                    break

                # Sample population from N(mean, sigma^2 * C)
                try:
                    _cma_L = torch.linalg.cholesky(_cma_C)
                except torch.linalg.LinAlgError:
                    # Regularize if C is not positive definite
                    _cma_C = _cma_C + 1e-6 * torch.eye(_cma_n)
                    _cma_L = torch.linalg.cholesky(_cma_C)

                _cma_z = torch.randn(_cma_lam, _cma_n)  # standard normal
                _cma_y = _cma_z @ _cma_L.T  # transformed by sqrt(C)
                _cma_x = _cma_mean.unsqueeze(0) + _cma_sigma * _cma_y  # candidates

                # Evaluate fitness
                _cma_fits = torch.zeros(_cma_lam)
                for _ci in range(_cma_lam):
                    _cma_unflatten(_cma_x[_ci].to(device))
                    _cma_fits[_ci] = evaluate_model_batched(model, device, num_tests=cmaes_eval_samples)

                # Sort by fitness (descending) and select top mu
                _cma_order = _cma_fits.argsort(descending=True)
                _cma_sel = _cma_order[:_cma_mu]

                # Track best
                if _cma_fits[_cma_order[0]] > _cma_best_acc:
                    _cma_best_acc = _cma_fits[_cma_order[0]].item()
                    _cma_best_flat = _cma_x[_cma_order[0]].clone()

                # Weighted mean of selected candidates
                _cma_y_sel = _cma_y[_cma_sel]  # (mu, n)
                _cma_y_w = (_cma_weights.unsqueeze(1) * _cma_y_sel).sum(dim=0)  # weighted mean step
                _cma_mean_new = _cma_mean + _cma_sigma * _cma_y_w

                # Update evolution path for sigma (conjugate evolution path)
                _cma_C_inv_sqrt_y = torch.linalg.solve(_cma_L, _cma_y_w.unsqueeze(1)).squeeze(1)
                _cma_ps = (1 - _cma_cs) * _cma_ps + math.sqrt(_cma_cs * (2 - _cma_cs) * _cma_mueff) * _cma_C_inv_sqrt_y

                # Update evolution path for covariance
                _cma_h_sig = 1.0 if _cma_ps.norm().item() / math.sqrt(1 - (1 - _cma_cs) ** (2 * _cma_gen)) < (1.4 + 2 / (_cma_n + 1)) * _cma_chi_n else 0.0
                _cma_pc = (1 - _cma_cc) * _cma_pc + _cma_h_sig * math.sqrt(_cma_cc * (2 - _cma_cc) * _cma_mueff) * _cma_y_w

                # Rank-one update
                _cma_rank1 = _cma_pc.unsqueeze(1) @ _cma_pc.unsqueeze(0)
                # Rank-mu update
                _cma_rank_mu = torch.zeros(_cma_n, _cma_n)
                for _ri in range(_cma_mu):
                    _cma_rank_mu += _cma_weights[_ri] * (_cma_y_sel[_ri].unsqueeze(1) @ _cma_y_sel[_ri].unsqueeze(0))

                # Update covariance matrix
                _cma_C = ((1 - _cma_c1 - _cma_cmu) * _cma_C +
                           _cma_c1 * (_cma_rank1 + (1 - _cma_h_sig) * _cma_cc * (2 - _cma_cc) * _cma_C) +
                           _cma_cmu * _cma_rank_mu)

                # Update step size via CSA
                _cma_sigma = _cma_sigma * math.exp((_cma_cs / _cma_damps) * (_cma_ps.norm().item() / _cma_chi_n - 1))
                _cma_sigma = min(_cma_sigma, 1.0)  # cap to prevent explosion

                _cma_mean = _cma_mean_new

                if _cma_gen % 5 == 0 or _cma_gen == cmaes_steps:
                    print(f"  CMA-ES gen {_cma_gen}/{cmaes_steps} | best={_cma_best_acc:.4f} | sigma={_cma_sigma:.6f} | pop_mean={_cma_fits.mean():.4f} | pop_max={_cma_fits.max():.4f}")

            # Restore best found weights
            _cma_unflatten(_cma_best_flat.to(device))
            _cma_final_acc = evaluate_model_batched(model, device, num_tests=2000)
            if _cma_final_acc >= best_accuracy:
                print(f"CMA-ES improved: {best_accuracy:.4f} -> {_cma_final_acc:.4f}")
                best_accuracy = _cma_final_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                print(f"CMA-ES final accuracy {_cma_final_acc:.4f} < best {best_accuracy:.4f}, reverting")
                if best_state is not None:
                    model.load_state_dict(best_state)

    # Multi-checkpoint greedy soup: load external checkpoints and greedily blend
    if greedy_soup_from and results_dir:
        if time_limit > 0 and (time.time() - t0) >= time_limit:
            print("Time limit reached before greedy soup")
        else:
            _gs_ids = [s.strip() for s in greedy_soup_from.split(",") if s.strip()]
            if _gs_ids:
                print(f"Greedy soup: blending with {len(_gs_ids)} checkpoints: {_gs_ids}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                _gs_current = {k: v.clone() for k, v in model.state_dict().items()}
                _gs_best_acc = best_accuracy

                for _gs_id in _gs_ids:
                    if time_limit > 0 and (time.time() - t0) >= time_limit:
                        print(f"Time limit reached during greedy soup")
                        break
                    _gs_ckpt_path = Path(results_dir) / _gs_id / "checkpoint.pt"
                    if not _gs_ckpt_path.exists():
                        print(f"  Greedy soup: {_gs_id} checkpoint not found, skipping")
                        continue
                    try:
                        _gs_ckpt = torch.load(_gs_ckpt_path, map_location=device, weights_only=True)
                        _gs_other = _gs_ckpt["state_dict"]
                    except Exception as _gs_e:
                        print(f"  Greedy soup: failed to load {_gs_id}: {_gs_e}")
                        continue
                    # Check shape compatibility
                    _gs_compat = all(
                        k in _gs_other and _gs_current[k].shape == _gs_other[k].shape
                        for k in _gs_current
                    )
                    if not _gs_compat:
                        print(f"  Greedy soup: {_gs_id} incompatible shapes, skipping")
                        continue
                    # Try blending at alpha = 0.5 (equal mix of current soup and new checkpoint)
                    _gs_blend = {}
                    for _gs_k in _gs_current:
                        _gs_blend[_gs_k] = 0.5 * _gs_current[_gs_k] + 0.5 * _gs_other[_gs_k]
                    model.load_state_dict(_gs_blend)
                    _gs_acc = evaluate_model_batched(model, device, num_tests=greedy_soup_eval_samples)
                    if _gs_acc > _gs_best_acc:
                        print(f"  Greedy soup: +{_gs_id} improved {_gs_best_acc:.4f} -> {_gs_acc:.4f}")
                        _gs_best_acc = _gs_acc
                        _gs_current = {k: v.clone() for k, v in _gs_blend.items()}
                    else:
                        # Also try alpha=0.7 (keep more of current) and alpha=0.3
                        _gs_improved = False
                        for _gs_alpha in [0.7, 0.3]:
                            _gs_blend2 = {}
                            for _gs_k in _gs_current:
                                _gs_blend2[_gs_k] = _gs_alpha * _gs_current[_gs_k] + (1 - _gs_alpha) * _gs_other[_gs_k]
                            model.load_state_dict(_gs_blend2)
                            _gs_acc2 = evaluate_model_batched(model, device, num_tests=greedy_soup_eval_samples)
                            if _gs_acc2 > _gs_best_acc:
                                print(f"  Greedy soup: +{_gs_id} (alpha={_gs_alpha}) improved {_gs_best_acc:.4f} -> {_gs_acc2:.4f}")
                                _gs_best_acc = _gs_acc2
                                _gs_current = {k: v.clone() for k, v in _gs_blend2.items()}
                                _gs_improved = True
                                break
                        if not _gs_improved:
                            print(f"  Greedy soup: +{_gs_id} no improvement, skipping")

                # Apply best greedy soup result
                model.load_state_dict(_gs_current)
                _gs_final = evaluate_model_batched(model, device, num_tests=2000)
                if _gs_final >= best_accuracy:
                    print(f"Greedy soup improved: {best_accuracy:.4f} -> {_gs_final:.4f}")
                    best_accuracy = _gs_final
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    print(f"Greedy soup final {_gs_final:.4f} < best {best_accuracy:.4f}, reverting")
                    if best_state is not None:
                        model.load_state_dict(best_state)

    training_time = time.time() - t0

    # Final evaluation on full test set
    final_acc = evaluate_model(model, device, num_tests=2000)
    print(f"Final accuracy (2000 tests): {final_acc:.4f}")

    # Majority-vote evaluation if enabled
    if majority_vote_n > 1:
        mv_acc = evaluate_model_majority_vote(model, device, num_tests=2000, n_votes=majority_vote_n)
        print(f"Majority-vote accuracy ({majority_vote_n} votes, 2000 tests): {mv_acc:.4f}")
        if mv_acc > final_acc:
            print(f"Majority vote improved accuracy: {final_acc:.4f} -> {mv_acc:.4f}")
            final_acc = mv_acc

    # Beam search evaluation if enabled
    beam_width = int(cfg.get("beam_width", 0))  # 0 = disabled (default)
    if beam_width > 1:
        beam_acc = evaluate_model_beam(model, device, num_tests=2000, beam_width=beam_width)
        print(f"Beam search accuracy (width={beam_width}, 2000 tests): {beam_acc:.4f}")
        if beam_acc > final_acc:
            print(f"Beam search improved accuracy: {final_acc:.4f} -> {beam_acc:.4f}")
            final_acc = beam_acc

    # Temperature-sampled majority vote: sample with temperature, majority vote across runs
    # Combines diversity of temperature sampling with robustness of majority vote
    inference_temp = float(cfg.get("inference_temp", 0.0))  # 0 = disabled (default)
    inference_temp_votes = int(cfg.get("inference_temp_votes", 5))  # number of samples to vote over
    if inference_temp > 0 and inference_temp_votes > 1:
        temp_acc = evaluate_model_temp_vote(model, device, num_tests=2000,
                                            temperature=inference_temp, n_votes=inference_temp_votes)
        print(f"Temp-vote accuracy (temp={inference_temp}, votes={inference_temp_votes}, 2000 tests): {temp_acc:.4f}")
        if temp_acc > final_acc:
            print(f"Temp-vote improved accuracy: {final_acc:.4f} -> {temp_acc:.4f}")
            final_acc = temp_acc

    # Embedding perturbation ensemble: run N passes with noise, average logits
    if eval_perturb_ensemble_n > 1:
        perturb_acc = evaluate_model_perturb_ensemble(
            model, device, num_tests=2000,
            n_passes=eval_perturb_ensemble_n, noise_std=eval_perturb_std)
        print(f"Perturb ensemble accuracy (n={eval_perturb_ensemble_n}, std={eval_perturb_std}, 2000 tests): {perturb_acc:.4f}")
        if perturb_acc > final_acc:
            print(f"Perturb ensemble improved accuracy: {final_acc:.4f} -> {perturb_acc:.4f}")
            final_acc = perturb_acc

    # Noise-augmented inference (inference_noise_samples): like eval_perturb_ensemble
    # but uses inference_noise_std. Separate config allows combining with different
    # noise levels or using as the sole test-time augmentation strategy.
    if inference_noise_samples > 0:
        nia_acc = evaluate_model_perturb_ensemble(
            model, device, num_tests=2000,
            n_passes=inference_noise_samples, noise_std=inference_noise_std)
        print(f"Noise-augmented inference accuracy (n={inference_noise_samples}, std={inference_noise_std}, 2000 tests): {nia_acc:.4f}")
        if nia_acc > final_acc:
            print(f"Noise-augmented inference improved accuracy: {final_acc:.4f} -> {nia_acc:.4f}")
            final_acc = nia_acc

    # Commutative ensemble: average logits from (a,b) and (b,a) at test time
    if eval_commutative_ensemble:
        comm_acc = evaluate_model_commutative(model, device, num_tests=2000)
        print(f"Commutative ensemble accuracy (2000 tests): {comm_acc:.4f}")
        if comm_acc > final_acc:
            print(f"Commutative ensemble improved accuracy: {final_acc:.4f} -> {comm_acc:.4f}")
            final_acc = comm_acc

    return model, {
        "accuracy": final_acc,
        "num_params": num_params,
        "training_time": training_time,
        "best_step_accuracy": best_accuracy,
    }


@torch.no_grad()
def evaluate_model_batched(model, device, num_tests=500, max_digits=NUM_DIGITS):
    """Fast batched evaluation - replaces serial single-sample evaluation in training loop."""
    model.eval()
    max_val = 10**max_digits - 1
    a = torch.randint(0, max_val + 1, (num_tests,), dtype=torch.long)
    b = torch.randint(0, max_val + 1, (num_tests,), dtype=torch.long)
    expected = a + b
    a_digits = (a.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS[:max_digits]) % 10
    if max_digits < NUM_DIGITS:
        pad = torch.zeros(num_tests, NUM_DIGITS - max_digits, dtype=torch.long)
        a_digits = torch.cat([a_digits, pad], dim=1)
        b_digits = torch.cat([b_digits, pad], dim=1)
    sep1 = torch.zeros(num_tests, 1, dtype=torch.long)
    sep2 = torch.zeros(num_tests, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)
    output = model.generate(prompts)  # (num_tests, OUTPUT_LEN)
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)
    results = (output.cpu() * powers).sum(dim=1)
    correct = (results == expected).sum().item()
    model.train()
    return correct / num_tests


@torch.no_grad()
def evaluate_model_carry_breakdown(model, device, num_tests=500):
    """Evaluate accuracy stratified by maximum carry chain length.

    Returns a dict mapping carry chain length buckets to (correct, total) counts.
    Buckets: 0 (no carries), 1-2, 3-4, 5-6, 7+ consecutive carries.
    This diagnostic reveals exactly which carry chain difficulty level the model
    fails on, enabling targeted experiment design.
    """
    model.eval()
    max_val = 10**NUM_DIGITS - 1
    a = torch.randint(0, max_val + 1, (num_tests,), dtype=torch.long)
    b = torch.randint(0, max_val + 1, (num_tests,), dtype=torch.long)
    expected = a + b

    # Compute max carry chain length for each example
    chain_lengths = compute_max_carry_chain_length(a, b, max_digits=NUM_DIGITS)  # (num_tests,)

    # Generate prompts and get model predictions
    a_digits = (a.unsqueeze(1) // _DIVISORS) % 10
    b_digits = (b.unsqueeze(1) // _DIVISORS) % 10
    sep1 = torch.zeros(num_tests, 1, dtype=torch.long)
    sep2 = torch.zeros(num_tests, 2, dtype=torch.long)
    prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)
    output = model.generate(prompts)  # (num_tests, OUTPUT_LEN)
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)
    results = (output.cpu() * powers).sum(dim=1)
    correct_mask = (results == expected)

    # Bucket by carry chain length
    buckets = [(0, 0, "chain=0"), (1, 2, "chain=1-2"), (3, 4, "chain=3-4"),
               (5, 6, "chain=5-6"), (7, 11, "chain=7+")]
    breakdown = {}
    for lo, hi, label in buckets:
        mask = (chain_lengths >= lo) & (chain_lengths <= hi)
        total = mask.sum().item()
        if total > 0:
            correct_count = (correct_mask & mask).sum().item()
            breakdown[label] = (correct_count, total, correct_count / total)
        else:
            breakdown[label] = (0, 0, 0.0)

    model.train()
    return breakdown


@torch.no_grad()
def evaluate_model_majority_vote(model, device, num_tests=1000, n_votes=5, seed=None):
    """Evaluate with majority vote: run N forward passes per sample, vote per digit."""
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    max_val = 10**NUM_DIGITS - 1
    correct = 0
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)

    # Generate all test cases
    a_list = [rng.randint(0, max_val) for _ in range(num_tests)]
    b_list = [rng.randint(0, max_val) for _ in range(num_tests)]

    batch_size = min(200, num_tests)
    for start in range(0, num_tests, batch_size):
        end = min(start + batch_size, num_tests)
        n = end - start
        a_t = torch.tensor(a_list[start:end], dtype=torch.long)
        b_t = torch.tensor(b_list[start:end], dtype=torch.long)
        expected = a_t + b_t

        a_digits = (a_t.unsqueeze(1) // _DIVISORS) % 10
        b_digits = (b_t.unsqueeze(1) // _DIVISORS) % 10
        sep1 = torch.zeros(n, 1, dtype=torch.long)
        sep2 = torch.zeros(n, 2, dtype=torch.long)
        prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)

        # Collect votes from multiple forward passes
        all_outputs = []
        for _ in range(n_votes):
            output = model.generate(prompts)  # (n, OUTPUT_LEN)
            all_outputs.append(output.cpu())

        # Majority vote per digit position
        stacked = torch.stack(all_outputs, dim=0)  # (n_votes, n, OUTPUT_LEN)
        voted = torch.mode(stacked, dim=0).values  # (n, OUTPUT_LEN)
        results = (voted * powers).sum(dim=1)
        correct += (results == expected).sum().item()

    model.train()
    return correct / num_tests


@torch.no_grad()
def evaluate_model_beam(model, device, num_tests=1000, beam_width=5, seed=None):
    """Evaluate model using beam search decoding (one sample at a time)."""
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    max_val = 10**NUM_DIGITS - 1
    correct = 0
    for _ in range(num_tests):
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        expected = a + b
        prompt, _ = encode_pair(a, b)
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
        output = model.generate_beam(prompt_t, beam_width=beam_width)
        result = 0
        for i, tok in enumerate(output[0].tolist()):
            if i >= OUTPUT_LEN:
                break
            result += tok * (10 ** i)
        if result == expected:
            correct += 1
    model.train()
    return correct / num_tests


@torch.no_grad()
def evaluate_model_temp_vote(model, device, num_tests=1000, temperature=0.5, n_votes=5, seed=None):
    """Evaluate with temperature sampling + majority vote per digit.

    Uses model.generate(temperature=T) to sample diverse outputs, then
    majority-votes per digit position. Combines exploration from temperature
    with robustness from voting.
    """
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    max_val = 10**NUM_DIGITS - 1
    correct = 0
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)

    a_list = [rng.randint(0, max_val) for _ in range(num_tests)]
    b_list = [rng.randint(0, max_val) for _ in range(num_tests)]

    batch_size = min(200, num_tests)
    for start in range(0, num_tests, batch_size):
        end = min(start + batch_size, num_tests)
        n = end - start
        a_t = torch.tensor(a_list[start:end], dtype=torch.long)
        b_t = torch.tensor(b_list[start:end], dtype=torch.long)
        expected = a_t + b_t

        a_digits = (a_t.unsqueeze(1) // _DIVISORS) % 10
        b_digits = (b_t.unsqueeze(1) // _DIVISORS) % 10
        sep1 = torch.zeros(n, 1, dtype=torch.long)
        sep2 = torch.zeros(n, 2, dtype=torch.long)
        prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)

        all_outputs = []
        for _ in range(n_votes):
            output = model.generate(prompts, temperature=temperature)
            all_outputs.append(output.cpu())

        stacked = torch.stack(all_outputs, dim=0)  # (n_votes, n, OUTPUT_LEN)
        voted = torch.mode(stacked, dim=0).values  # (n, OUTPUT_LEN)
        results = (voted * powers).sum(dim=1)
        correct += (results == expected).sum().item()

    model.train()
    return correct / num_tests


@torch.no_grad()
@torch.no_grad()
def evaluate_model_perturb_ensemble(model, device, num_tests=1000, n_passes=10, noise_std=0.01, seed=None):
    """Evaluate with embedding perturbation ensemble: run N forward passes with
    small Gaussian noise on embeddings, average logits before argmax.

    Different from majority_vote (argmax per pass → vote, can't correct unanimous
    wrong digit) and eval_commutative_ensemble (only two orderings). This creates
    N diverse representations via embedding noise, averaging logits to smooth out
    sharp decision boundaries on borderline carry-chain predictions.
    """
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    max_val = 10**NUM_DIGITS - 1
    correct = 0
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)

    a_list = [rng.randint(0, max_val) for _ in range(num_tests)]
    b_list = [rng.randint(0, max_val) for _ in range(num_tests)]

    batch_size = min(200, num_tests)
    for start in range(0, num_tests, batch_size):
        end = min(start + batch_size, num_tests)
        n = end - start
        a_t = torch.tensor(a_list[start:end], dtype=torch.long)
        b_t = torch.tensor(b_list[start:end], dtype=torch.long)
        expected = a_t + b_t

        a_digits = (a_t.unsqueeze(1) // _DIVISORS) % 10
        b_digits = (b_t.unsqueeze(1) // _DIVISORS) % 10
        sep1 = torch.zeros(n, 1, dtype=torch.long)
        sep2 = torch.zeros(n, 2, dtype=torch.long)
        prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)

        # Autoregressive generation with logit averaging across N noisy passes
        B, T_prompt = prompts.shape
        seq = prompts.clone()
        n_think = model.n_think_tokens
        total_gen = n_think + OUTPUT_LEN

        # Save original embedding parameters to restore after
        _orig_embed_noise = model.embed_noise_std
        model.embed_noise_std = noise_std  # temporarily enable embedding noise

        for _gen_step in range(total_gen):
            avg_logits = torch.zeros(B, VOCAB_SIZE, device=device)
            for _pass in range(n_passes):
                # Each pass uses different noise via the model's embed_noise mechanism
                # We temporarily set model to training mode for noise injection
                model.train()
                logits = model.forward(seq)[:, -1, :VOCAB_SIZE]
                avg_logits += logits
                model.eval()
            avg_logits /= n_passes
            next_tok = avg_logits.argmax(dim=-1)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)

        # Restore original setting
        model.embed_noise_std = _orig_embed_noise

        out_start = T_prompt + n_think
        output = seq[:, out_start:out_start + OUTPUT_LEN]
        results = (output.cpu() * powers).sum(dim=1)
        correct += (results == expected).sum().item()

    model.train()
    return correct / num_tests


def evaluate_model_commutative(model, device, num_tests=1000, seed=None):
    """Evaluate with commutative ensemble: average logits from (a,b) and (b,a).

    Since a+b = b+a, both orderings should produce the same answer. The model may
    have asymmetric attention patterns that get one ordering right and the other wrong.
    Averaging the logits produces a more robust prediction. Zero extra params.
    """
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    max_val = 10**NUM_DIGITS - 1
    correct = 0
    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)

    a_list = [rng.randint(0, max_val) for _ in range(num_tests)]
    b_list = [rng.randint(0, max_val) for _ in range(num_tests)]

    batch_size = min(200, num_tests)
    for start in range(0, num_tests, batch_size):
        end = min(start + batch_size, num_tests)
        n = end - start
        a_t = torch.tensor(a_list[start:end], dtype=torch.long)
        b_t = torch.tensor(b_list[start:end], dtype=torch.long)
        expected = a_t + b_t

        a_digits = (a_t.unsqueeze(1) // _DIVISORS) % 10
        b_digits = (b_t.unsqueeze(1) // _DIVISORS) % 10
        sep1 = torch.zeros(n, 1, dtype=torch.long)
        sep2 = torch.zeros(n, 2, dtype=torch.long)

        # Original ordering: (a, b)
        prompts_ab = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)
        # Swapped ordering: (b, a)
        prompts_ba = torch.cat([sep1, b_digits, sep2, a_digits, sep1], dim=1).to(device)

        # Autoregressive generation with logit averaging at each step
        B, T_prompt = prompts_ab.shape
        seq_ab = prompts_ab.clone()
        seq_ba = prompts_ba.clone()
        n_think = model.n_think_tokens
        total_gen = n_think + OUTPUT_LEN

        for _ in range(total_gen):
            logits_ab = model.forward(seq_ab)[:, -1, :VOCAB_SIZE]
            logits_ba = model.forward(seq_ba)[:, -1, :VOCAB_SIZE]
            # Average logits from both orderings
            avg_logits = (logits_ab + logits_ba) / 2.0
            next_tok = avg_logits.argmax(dim=-1)
            seq_ab = torch.cat([seq_ab, next_tok.unsqueeze(1)], dim=1)
            seq_ba = torch.cat([seq_ba, next_tok.unsqueeze(1)], dim=1)

        out_start = T_prompt + n_think
        output = seq_ab[:, out_start:out_start + OUTPUT_LEN]
        results = (output.cpu() * powers).sum(dim=1)
        correct += (results == expected).sum().item()

    model.train()
    return correct / num_tests


def evaluate_model(model, device, num_tests=1000, seed=None, max_digits=NUM_DIGITS):
    """Evaluate model accuracy on random addition problems."""
    model.eval()
    rng = random.Random(seed) if seed else random.Random()
    correct = 0
    max_val = 10**max_digits - 1

    with torch.no_grad():
        for _ in range(num_tests):
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            expected = a + b

            prompt, _ = encode_pair(a, b)
            prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
            output = model.generate(prompt_t)

            result = 0
            for i, tok in enumerate(output[0].tolist()):
                if i >= OUTPUT_LEN:
                    break
                result += tok * (10 ** i)

            if result == expected:
                correct += 1

    model.train()
    return correct / num_tests


def evaluate_on_verify_set(model, device):
    """Evaluate on the exact verify.py test set (seed=2025)."""
    model.eval()

    # Edge cases
    edge_cases = [
        (0, 0), (0, 1), (9_999_999_999, 0), (9_999_999_999, 1),
        (9_999_999_999, 9_999_999_999), (5_000_000_000, 5_000_000_000),
        (1_111_111_111, 8_888_888_889), (1_234_567_890, 9_876_543_210),
        (9_999_999_999, 9_999_999_999), (1, 9_999_999_999),
    ]

    rng = random.Random(2025)
    random_cases = [(rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND)) for _ in range(10000)]
    all_cases = edge_cases + random_cases

    a_list = [c[0] for c in all_cases]
    b_list = [c[1] for c in all_cases]
    a_t = torch.tensor(a_list, dtype=torch.long)
    b_t = torch.tensor(b_list, dtype=torch.long)
    expected = a_t + b_t

    powers = torch.tensor([10**i for i in range(OUTPUT_LEN)], dtype=torch.long)
    correct = 0
    batch_size = 500
    with torch.no_grad():
        for start in range(0, len(all_cases), batch_size):
            end = min(start + batch_size, len(all_cases))
            a_batch = a_t[start:end]
            b_batch = b_t[start:end]
            n = end - start
            a_digits = (a_batch.unsqueeze(1) // _DIVISORS) % 10
            b_digits = (b_batch.unsqueeze(1) // _DIVISORS) % 10
            sep1 = torch.zeros(n, 1, dtype=torch.long)
            sep2 = torch.zeros(n, 2, dtype=torch.long)
            prompts = torch.cat([sep1, a_digits, sep2, b_digits, sep1], dim=1).to(device)
            output = model.generate(prompts)
            results = (output.cpu() * powers).sum(dim=1)
            correct += (results == expected[start:end]).sum().item()

    model.train()
    return correct / len(all_cases)


def generate_submission(model, cfg, metrics, idea_dir):
    """Generate a self-contained submission file."""
    # Save checkpoint
    ckpt_path = idea_dir / "checkpoint.pt"
    torch.save({
        "config": cfg,
        "state_dict": model.state_dict(),
    }, ckpt_path)

    # Generate submission code
    submission_code = f'''"""
AdderBoard submission generated by Orze auto-research.
Params: {metrics["num_params"]}, Accuracy: {metrics["accuracy"]:.4%}
"""
import math, os, torch, torch.nn as nn, torch.nn.functional as F

# --- Inline model definition (same as train.py) ---

VOCAB_SIZE = {cfg.get("vocab_size", VOCAB_SIZE)}
NUM_DIGITS = 10
OUTPUT_LEN = 11
SEP_TOKEN = 11

{_get_model_source()}

def build_model():
    cfg = {repr(cfg)}
    device = torch.device("cpu")
    model = AdderTransformer(cfg).to(device)
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "checkpoint.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {{
        "name": "Orze-AutoResearch",
        "author": "erik",
        "params": {metrics["num_params"]},
        "architecture": "{_describe_arch(cfg)}",
        "tricks": {_list_tricks(cfg)},
    }}

def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    a_digits = [(a // 10**i) % 10 for i in range(NUM_DIGITS)]
    b_digits = [(b // 10**i) % 10 for i in range(NUM_DIGITS)]
    prompt = a_digits + [SEP_TOKEN] + b_digits + [SEP_TOKEN]
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(prompt_t)
    result = 0
    for i, tok in enumerate(output[0].tolist()):
        if i >= OUTPUT_LEN:
            break
        result += tok * (10 ** i)
    return result
'''
    (idea_dir / "submission.py").write_text(submission_code)


def _get_model_source():
    """Return model class source for embedding in submission."""
    # We'll just reference the checkpoint - the full model code is too large to inline
    return "# Model classes loaded from train.py module"


def _describe_arch(cfg):
    d = cfg.get("d_model", 3)
    h = cfg.get("n_heads", 1)
    kv = cfg.get("n_kv_heads", 1)
    hd = cfg.get("head_dim", 4)
    ff = cfg.get("ff_mult", 2)
    n = cfg.get("n_layers", 1)
    return f"{n}L decoder, d={d}, {h}h/{kv}kv, hd={hd}, ff={ff}"


def _list_tricks(cfg):
    tricks = []
    if cfg.get("tie_kv"): tricks.append("Tied K=V")
    if cfg.get("tie_qo"): tricks.append("Tied O=Q^T")
    if cfg.get("tie_embed", True): tricks.append("Tied embed")
    if cfg.get("share_norms"): tricks.append("Shared norms")
    if cfg.get("use_swiglu", True): tricks.append("SwiGLU")
    if cfg.get("qk_norm", True): tricks.append("QK norm")
    if cfg.get("embed_type") == "circular_arc": tricks.append("Circular arc embedding")
    if cfg.get("use_rope", True): tricks.append("RoPE")
    if cfg.get("curriculum"): tricks.append("Curriculum learning")
    return tricks


# ---------------------------------------------------------------------------
# Orze entry point
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_idea_config(ideas_md: str, idea_id: str) -> dict:
    text = Path(ideas_md).read_text(encoding="utf-8")
    pattern = rf"^##\s+{re.escape(idea_id)}\b.*?```ya?ml\s*\n(.*?)```"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"Idea {idea_id} not found in {ideas_md}")
    return yaml.safe_load(m.group(1)) or {}


def load_idea_from_sqlite(idea_id: str) -> dict:
    """Load idea config from orze's SQLite database (idea_lake.db)."""
    import sqlite3
    db_path = Path("results/idea_lake.db")
    if not db_path.exists():
        db_path = Path("idea_lake.db")
    if not db_path.exists():
        raise FileNotFoundError("idea_lake.db not found")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT config FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Idea {idea_id} not found in SQLite")
        return yaml.safe_load(row[0]) or {}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--ideas-md", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    idea_dir = Path(args.results_dir) / args.idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    base_cfg = {}
    if Path(args.config).exists():
        base_cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    idea_cfg_path = idea_dir / "idea_config.yaml"
    idea_cfg = {}
    if idea_cfg_path.exists():
        idea_cfg = yaml.safe_load(idea_cfg_path.read_text()) or {}

    # Try loading from ideas.md first, then SQLite (orze consumes ideas.md)
    try:
        ideas_cfg = load_idea_config(args.ideas_md, args.idea_id)
        idea_cfg = deep_merge(idea_cfg, ideas_cfg)
    except Exception:
        try:
            ideas_cfg = load_idea_from_sqlite(args.idea_id)
            idea_cfg = deep_merge(idea_cfg, ideas_cfg)
        except Exception as e:
            raise RuntimeError(f"FATAL: Could not load idea config for {args.idea_id}: {e}. Refusing to train with base config only.")

    cfg = deep_merge(base_cfg, idea_cfg)
    print(f"Training {args.idea_id} with config: {json.dumps(cfg, indent=2, default=str)}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set seed
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        t0 = time.time()
        model, train_metrics = train_model(cfg, device=device, results_dir=args.results_dir)

        # Full verify-set evaluation if accuracy looks promising
        if train_metrics["accuracy"] >= 0.95:
            print("Running full verify-set evaluation...")
            verify_acc = evaluate_on_verify_set(model, device)
            print(f"Verify-set accuracy: {verify_acc:.4%} ({int(verify_acc * 10010)}/10010)")
            train_metrics["verify_accuracy"] = verify_acc
            train_metrics["verify_correct"] = int(verify_acc * 10010)

            # Generate submission if qualified
            if verify_acc >= 0.99:
                generate_submission(model, cfg, train_metrics, idea_dir)
                print("QUALIFIED! Submission file generated.")

        # Always save checkpoint for multi-phase training
        torch.save({"config": cfg, "state_dict": model.state_dict()},
                    idea_dir / "checkpoint.pt")

        training_time = time.time() - t0

        metrics = {
            "status": "COMPLETED",
            "accuracy": round(train_metrics["accuracy"], 6),
            "num_params": train_metrics["num_params"],
            "training_time": round(training_time, 2),
            "best_step_accuracy": round(train_metrics.get("best_step_accuracy", 0), 6),
        }
        if "verify_accuracy" in train_metrics:
            metrics["verify_accuracy"] = round(train_metrics["verify_accuracy"], 6)
            metrics["verify_correct"] = train_metrics["verify_correct"]
            metrics["qualified"] = train_metrics["verify_accuracy"] >= 0.99

    except Exception as e:
        import traceback
        traceback.print_exc()
        metrics = {
            "status": "FAILED",
            "error": str(e),
            "training_time": round(time.time() - t0, 2) if "t0" in dir() else 0,
        }

    (idea_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done. Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
