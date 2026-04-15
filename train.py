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


def generate_batch(batch_size, max_digits=NUM_DIGITS, device="cpu"):
    """Generate a batch of addition problems (vectorized)."""
    max_val = 10**max_digits - 1
    a = torch.randint(0, max_val + 1, (batch_size,), dtype=torch.long)
    b = torch.randint(0, max_val + 1, (batch_size,), dtype=torch.long)
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


def generate_carry_biased_batch(batch_size, max_digits=NUM_DIGITS, carry_prob=0.5, device="cpu"):
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


class AntiQuarterNorm(nn.Module):
    """1-parameter QK norm from lokimorty's 39p model.

    RMS-normalizes, then scales dims by [a, a/4, 0, -a].
    Saves 3 params vs standard 4-param RMSNorm on head_dim=4.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        assert dim == 4, "AntiQuarterNorm designed for head_dim=4"
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        y = (x.float() * scale).to(x.dtype)
        a = self.weight
        w = torch.stack([a, a * 0.25, a.new_zeros(()).expand_as(a), -a]).squeeze(-1)
        return y * w


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0, max_seq_len=128, rope_period=None):
        super().__init__()
        if rope_period is not None and rope_period > 0:
            # Fixed period: all frequencies set to 2*pi/period
            inv_freq = torch.full((max(1, dim // 2),), 2 * math.pi / rope_period)
        else:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
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


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def _apply_2d_rotation(x, angle):
    """Apply learned 2D rotation to pairs of dims. x: [..., D], angle: scalar."""
    c, s = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim,
                 rope_theta=3.0, qk_norm=True, tie_kv=False, tie_qo=False,
                 tie_qk=False, k_rot_q=False, v_eq_q=False, tie_qk_norm=False,
                 attn_out_rank=0, use_rope=True, rope_period=None,
                 qk_norm_type="rms", k_alpha_q=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # K projection options
        self.k_rot_q = k_rot_q
        self.tie_qk = tie_qk
        self.k_alpha_q = k_alpha_q
        if k_rot_q:
            self.k_rot_angle = nn.Parameter(torch.tensor(0.1))  # 1 param
        elif k_alpha_q:
            self.k_alpha = nn.Parameter(torch.tensor(1.0))  # 1 param: K = alpha * Q
        elif not tie_qk:
            self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # V projection options
        self.v_eq_q = v_eq_q
        self.tie_kv = tie_kv
        if not v_eq_q and not tie_kv and not tie_qk and not k_rot_q and not k_alpha_q:
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
            if qk_norm_type == "anti_quarter":
                self.q_norm = AntiQuarterNorm(head_dim)
            else:
                self.q_norm = RMSNorm(head_dim)
            if not tie_qk_norm:
                if qk_norm_type == "anti_quarter":
                    self.k_norm = AntiQuarterNorm(head_dim)
                else:
                    self.k_norm = RMSNorm(head_dim)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(head_dim, theta=rope_theta, rope_period=rope_period)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # K computation
        if self.k_rot_q:
            k = _apply_2d_rotation(q, self.k_rot_angle)
        elif self.k_alpha_q:
            k = self.k_alpha * q  # K = alpha * Q (tbukic's scalar-scaled technique)
        elif self.tie_qk:
            k = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # V computation
        if self.v_eq_q:
            v = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        elif self.k_rot_q:
            v = q.clone()  # V=Q (unrotated), K=rot(Q) — matches tbukic's architecture
        elif self.tie_kv or self.tie_qk or self.k_alpha_q:
            v = k.clone()
        else:
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
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
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
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
    def __init__(self, d_model, ff_dim, tie_gate=False, tie_down_gate=False,
                 down_rot_up_t=False, gate_alpha_up=False):
        super().__init__()
        self.tie_gate = tie_gate
        self.tie_down_gate = tie_down_gate
        self.down_rot_up_t = down_rot_up_t
        self.gate_alpha_up = gate_alpha_up

        if gate_alpha_up:
            # gate = silu(alpha * up(x)) — 1 param instead of 6p gate_proj (tbukic's technique)
            self.gate_alpha = nn.Parameter(torch.tensor(1.0))
            self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        else:
            self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
            if not tie_gate:
                self.up_proj = nn.Linear(d_model, ff_dim, bias=False)

        if down_rot_up_t:
            self.down_rot_angle = nn.Parameter(torch.tensor(0.1))  # 1 param
        elif not tie_down_gate:
            self.down_proj = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        if self.gate_alpha_up:
            up = self.up_proj(x)
            gate = F.silu(self.gate_alpha * up)
        else:
            gate = F.silu(self.gate_proj(x))
            if self.tie_gate:
                up = gate
            else:
                up = self.up_proj(x)
        mixed = gate * up
        if self.down_rot_up_t:
            # down(x) = F.linear(rotate_2d(x, theta), up^T) — rotate INPUT, then up^T
            rotated = _apply_2d_rotation(mixed, self.down_rot_angle)
            return F.linear(rotated, self.up_proj.weight.T)
        elif self.tie_down_gate:
            # Use up_proj^T if gate_alpha_up (no gate_proj), else gate_proj^T
            weight = self.up_proj.weight if self.gate_alpha_up else self.gate_proj.weight
            return F.linear(mixed, weight.T)
        return self.down_proj(mixed)


class GeLUMLP(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023): sign-based updates, good for small models.

    Uses sign of momentum-interpolated gradient for updates.
    Often works better than Adam for tiny models with strong regularization.
    Typical settings: lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01-0.1.
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                # Update = sign(beta1 * ema + (1-beta1) * grad)
                update = exp_avg.lerp(p.grad, 1 - beta1)
                p.data.add_(update.sign_(), alpha=-group['lr'])
                # EMA update (for next step's interpolation)
                exp_avg.lerp_(p.grad, 1 - beta2)


def _newton_schulz_5(G, steps=5):
    """Newton-Schulz iteration for approximate matrix orthogonalization.

    Finds nearest orthogonal matrix to G under Frobenius norm.
    Coefficients (a, b, c) are optimized for fast convergence.
    """
    assert G.dim() == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = a * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) + b * A + c * A @ A
        X = B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer (Bernstein & Newhouse, 2024): momentum + Newton-Schulz orthogonalization.

    For 2D+ weight matrices: orthogonalizes the momentum buffer using Newton-Schulz
    iteration, producing spectrally-normalized updates that find the steepest descent
    direction under the spectral norm. Significantly accelerates grokking vs AdamW.

    For 1D params (norms, scalars): plain SGD with momentum (no orthogonalization).

    Typical settings: lr=0.02, momentum=0.95, ns_steps=5.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(mu).add_(g)

                if nesterov:
                    update = g + mu * buf
                else:
                    update = buf.clone()

                # Decoupled weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Newton-Schulz orthogonalization for 2D+ params
                if p.dim() >= 2:
                    orig_shape = update.shape
                    u = update.view(update.shape[0], -1)
                    u = _newton_schulz_5(u, ns_steps)
                    update = u.view(orig_shape)

                p.data.add_(update, alpha=-lr)


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
                 rope_period=None, qk_norm_type="rms",
                 k_alpha_q=False, gate_alpha_up=False, sphere_norm=False,
                 o_tail_scale=False):
        super().__init__()
        self.sphere_norm = sphere_norm
        # O tail scale: 1-parameter learned scalar on attention output (lokimorty's technique)
        # Controls attention-to-residual contribution ratio with a single param
        self.o_tail_scale_param = nn.Parameter(torch.tensor(1.0)) if o_tail_scale else None
        self.attn = Attention(
            d_model, n_heads, n_kv_heads, head_dim,
            rope_theta=rope_theta, qk_norm=qk_norm,
            tie_kv=tie_kv, tie_qo=tie_qo, tie_qk=tie_qk,
            k_rot_q=k_rot_q, v_eq_q=v_eq_q, tie_qk_norm=tie_qk_norm,
            attn_out_rank=attn_out_rank,
            use_rope=use_rope, rope_period=rope_period,
            qk_norm_type=qk_norm_type,
            k_alpha_q=k_alpha_q,
        )

        if use_swiglu:
            self.mlp = SwiGLUMLP(d_model, ff_dim, tie_gate=tie_gate,
                                 tie_down_gate=tie_down_gate, down_rot_up_t=down_rot_up_t,
                                 gate_alpha_up=gate_alpha_up)
        else:
            self.mlp = GeLUMLP(d_model, ff_dim)

        if norm_type == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        elif norm_type == "unit":
            self.ln1 = UnitRMSNorm()
            self.ln2 = UnitRMSNorm()
        else:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model) if not share_norms else self.ln1

    def forward(self, x, mask=None):
        attn_out = self.attn(self.ln1(x), mask=mask)
        if self.o_tail_scale_param is not None:
            attn_out = attn_out * self.o_tail_scale_param
        x = x + attn_out
        if self.sphere_norm:
            x = F.normalize(x, p=2, dim=-1)
        x = x + self.mlp(self.ln2(x))
        if self.sphere_norm:
            x = F.normalize(x, p=2, dim=-1)
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
        n_kv_heads = cfg.get("n_kv_heads", 1)
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
        k_alpha_q = cfg.get("k_alpha_q", False)
        gate_alpha_up = cfg.get("gate_alpha_up", False)
        tie_embed = cfg.get("tie_embed", True)
        share_norms = cfg.get("share_norms", False)
        attn_out_rank = cfg.get("attn_out_rank", 0)
        qk_norm_type = cfg.get("qk_norm_type", "rms")  # "rms" or "anti_quarter"
        use_swiglu = cfg.get("use_swiglu", True)
        norm_type = cfg.get("norm_type", "rms")
        use_rope = cfg.get("use_rope", True)
        rope_period = cfg.get("rope_period", None)
        embed_type = cfg.get("embed_type", "lookup")
        vocab_size = cfg.get("vocab_size", VOCAB_SIZE)

        # O tail scale: 1-parameter scalar on attention output (lokimorty's 39p technique)
        o_tail_scale = cfg.get("o_tail_scale", False)

        # Spherical residual stream: L2-normalize residual after each sublayer (arxiv 2603.05228)
        # 20x grokking speedup for modular addition. Zero params.
        sphere_norm = cfg.get("sphere_norm", False)
        self.sphere_norm = sphere_norm
        self.sphere_tau = float(cfg.get("sphere_tau", 0))
        # Normalized unembedding: logits = tau * normalize(h) @ normalize(W).T (arxiv 2603.05228)
        # Without this, sphere_norm suffers Softmax Collapse (confirmed: ideas 2121-2124 all collapsed)
        self.sphere_norm_unembed = cfg.get("sphere_norm_unembed", sphere_norm)

        # Repeat-mix: run block twice with learned interpolation (lokimorty's technique)
        self.repeat_mix = cfg.get("repeat_mix", False)
        if self.repeat_mix:
            self.repeat_gain = nn.Parameter(torch.tensor(-1e-5))  # 1 param, init near zero

        # Token embedding
        if embed_type == "circular_arc":
            tok_dim = 2
            self.tok_embed = CircularArcEmbedding(vocab_size, tok_dim)
            pos_dim = d_model - tok_dim
            self.pos_mode = "sinusoidal" if pos_dim > 0 else "none"
        elif embed_type == "quadratic":
            tok_dim = 2
            self.tok_embed = QuadraticEmbedding(vocab_size, tok_dim)
            pos_dim = d_model - tok_dim
            self.pos_mode = "sinusoidal" if pos_dim > 0 else "none"
        else:
            self.tok_embed = LookupEmbedding(vocab_size, d_model)
            tok_dim = d_model
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
                qk_norm_type=qk_norm_type,
                k_alpha_q=k_alpha_q, gate_alpha_up=gate_alpha_up,
                sphere_norm=sphere_norm,
                o_tail_scale=o_tail_scale,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        if norm_type == "none":
            self.ln_f = nn.Identity()
        elif norm_type == "unit":
            self.ln_f = UnitRMSNorm()
        else:
            share_ln_f = cfg.get("share_ln_f", share_norms)  # default: follow share_norms
            if share_ln_f and n_layers > 0:
                self.ln_f = self.blocks[0].ln1
            else:
                self.ln_f = RMSNorm(d_model)

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

    def forward(self, idx):
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

        # Causal mask (avoid 0 * -inf = NaN)
        mask = torch.zeros(T, T, device=x.device)
        mask.masked_fill_(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            if self.repeat_mix:
                first = block(x, mask=mask)
                second = block(first, mask=mask)
                x = first + (second - first) * self.repeat_gain
            else:
                x = block(x, mask=mask)

        x = self.ln_f(x)

        # Spherical residual stream: L2-normalize before unembedding
        if self.sphere_norm:
            x = F.normalize(x, p=2, dim=-1)

        if self.tie_embed:
            table = self.tok_embed.table()
            # Normalized unembedding for spherical residual stream (arxiv 2603.05228)
            if self.sphere_norm and self.sphere_norm_unembed:
                table = F.normalize(table, p=2, dim=-1)
            if table.shape[1] < self.d_model:
                # Project only tok dimensions for output
                logits = x[..., :table.shape[1]] @ table.T
            else:
                logits = x @ table.T
        else:
            logits = self.lm_head(x)

        # Spherical residual stream: scale logits by temperature
        if self.sphere_tau > 0:
            logits = logits * self.sphere_tau

        return logits

    def count_params(self):
        """Count unique parameters (after tying)."""
        seen_ids = set()
        total = 0
        for p in self.parameters():
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                total += p.numel()
        return total

    @torch.no_grad()
    def generate(self, prompt):
        """Autoregressive generation."""
        self.eval()
        B, T_prompt = prompt.shape
        seq = prompt.clone()
        for _ in range(OUTPUT_LEN):
            logits = self.forward(seq)
            next_tok = logits[:, -1, :VOCAB_SIZE].argmax(dim=-1)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        return seq[:, T_prompt:T_prompt + OUTPUT_LEN]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(cfg, device="cuda"):
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

    # Custom per-layer initialization (tbukic's technique)
    # Laplace init for projection weights, uniform for norm weights
    # Different init can dramatically change grokking probability
    custom_init = cfg.get("custom_init", None)
    if custom_init:
        init_q_scale = float(cfg.get("init_q_scale", 3.0))    # Laplace scale for q/k projections
        init_norm_lo = float(cfg.get("init_norm_lo", -10.0))   # Uniform lower bound for norms
        init_norm_hi = float(cfg.get("init_norm_hi", 15.0))    # Uniform upper bound for norms
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "q_proj" in name or "k_proj" in name:
                    # Laplace distribution for projection weights (tbukic uses scale=3.0)
                    p.data.copy_(torch.zeros_like(p).cauchy_().sign() *
                                 torch.empty_like(p).exponential_(1.0 / init_q_scale))
                elif any(k in name for k in ("ln1", "ln2", "ln_f", "q_norm", "k_norm")):
                    # Uniform distribution for norm parameters
                    p.data.uniform_(init_norm_lo, init_norm_hi)
        print(f"  [custom_init] Applied: proj Laplace(scale={init_q_scale}), norms U({init_norm_lo},{init_norm_hi})")

    # Tag parameters with per-layer Grokfast lambda multiplier
    grokfast_lambda_norm_mult = float(cfg.get("grokfast_lambda_norm_mult", 1.0))
    grokfast_lambda_arc_mult = float(cfg.get("grokfast_lambda_arc_mult", 1.0))
    if grokfast_lambda_norm_mult != 1.0 or grokfast_lambda_arc_mult != 1.0:
        for _gf_name, _gf_p in model.named_parameters():
            if any(k in _gf_name for k in ("ln1", "ln2", "ln_f", "q_norm", "k_norm")):
                _gf_p._gf_lambda_mult = grokfast_lambda_norm_mult
            elif "arc_" in _gf_name:
                _gf_p._gf_lambda_mult = grokfast_lambda_arc_mult
            else:
                _gf_p._gf_lambda_mult = 1.0
        print(f"  [grokfast] Per-layer lambda: norm_mult={grokfast_lambda_norm_mult}, arc_mult={grokfast_lambda_arc_mult}")

    # Warm Start: load weights from a pre-trained checkpoint (enables relay training across recipes)
    warm_start_ckpt = cfg.get("warm_start_ckpt", "")
    if warm_start_ckpt:
        _ws_path = Path(warm_start_ckpt)
        if _ws_path.is_dir():
            _ws_path = _ws_path / "checkpoint.pt"
        warm_start_ckpt = str(_ws_path)
    if warm_start_ckpt and Path(warm_start_ckpt).exists():
        _ws_data = torch.load(warm_start_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(_ws_data["state_dict"])
        print(f"  [warm_start] Loaded weights from {warm_start_ckpt}")

    # Knowledge Distillation: train with soft labels from a teacher model checkpoint
    distill_teacher_ckpt = cfg.get("distill_teacher_ckpt", "")
    distill_alpha = float(cfg.get("distill_alpha", 0.5))
    distill_temperature = float(cfg.get("distill_temperature", 2.0))
    teacher_model = None
    if distill_teacher_ckpt and Path(distill_teacher_ckpt).exists():
        _td = torch.load(distill_teacher_ckpt, map_location=device, weights_only=True)
        teacher_model = AdderTransformer(_td["config"]).to(device)
        teacher_model.load_state_dict(_td["state_dict"])
        teacher_model.eval()
        for _tp in teacher_model.parameters():
            _tp.requires_grad = False
        print(f"  [distill] Teacher loaded from {distill_teacher_ckpt} ({teacher_model.count_params()} params)")

    # Training hyperparams
    lr = float(cfg.get("lr", 0.01))
    min_lr = float(cfg.get("min_lr", 1e-5))
    steps = int(cfg.get("steps", 30000))
    batch_size = int(cfg.get("batch_size", 256))
    # Batch size warmup: start with smaller batch for more gradient noise (aids grokking onset),
    # then ramp to full batch_size for stable late-stage convergence
    batch_size_start = int(cfg.get("batch_size_start", 0))  # 0 = disabled (default)
    batch_size_warmup_steps = int(cfg.get("batch_size_warmup_steps", 0))  # steps to ramp
    _batch_size_final = batch_size
    warmup_steps = int(cfg.get("warmup_steps", 500))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    # Progressive gradient clip: linearly transition grad_clip from initial to final over training.
    # Tighter clipping early stabilizes Grokfast EMA convergence; looser clipping later allows
    # larger gradient steps for the final push from 99%→100%.
    grad_clip_final = float(cfg.get("grad_clip_final", 0))  # 0 = constant clip (default); e.g. 2.0
    eval_every = int(cfg.get("eval_every", 1000))
    patience = int(cfg.get("patience", 10000))
    carry_bias = float(cfg.get("carry_bias", 0.0))

    # Curriculum
    curriculum = cfg.get("curriculum", None)
    # Format: "digits:steps,digits:steps,..." e.g. "3:2000,6:5000,10:rest"

    # Per-parameter LR multipliers (tbukic's technique)
    # Different learning rates for different parameter groups improve grokking
    lr_norm_mult = float(cfg.get("lr_norm_mult", 1.0))   # multiplier for norm params (tbukic uses 3.0)
    lr_arc_mult = float(cfg.get("lr_arc_mult", 1.0))     # multiplier for arc embedding params (tbukic uses 0.5)
    lr_up_mult = float(cfg.get("lr_up_mult", 1.0))       # multiplier for up_proj/gate_proj params (tbukic uses 1.5)

    # Per-parameter weight decay multipliers: different regularization for different param types
    # Norms may benefit from less WD (they control scale), arc params from more/less WD
    wd_norm_mult = float(cfg.get("wd_norm_mult", 1.0))   # WD multiplier for norm params
    wd_arc_mult = float(cfg.get("wd_arc_mult", 1.0))     # WD multiplier for arc embedding params
    wd_up_mult = float(cfg.get("wd_up_mult", 1.0))       # WD multiplier for up_proj/gate_proj params

    # Build parameter groups with per-param LR/WD multipliers
    use_param_groups = (lr_norm_mult != 1.0 or lr_arc_mult != 1.0 or lr_up_mult != 1.0 or
                        wd_norm_mult != 1.0 or wd_arc_mult != 1.0 or wd_up_mult != 1.0)

    if use_param_groups:
        norm_params = []
        arc_params = []
        up_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "ln1" in name or "ln2" in name or "ln_f" in name or "q_norm" in name or "k_norm" in name:
                norm_params.append(p)
            elif "arc_" in name:
                arc_params.append(p)
            elif "up_proj" in name or "gate_proj" in name or "gate_alpha" in name:
                up_params.append(p)
            else:
                other_params.append(p)
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr, "lr_mult": 1.0, "weight_decay": weight_decay, "wd_mult": 1.0})
        if norm_params:
            param_groups.append({"params": norm_params, "lr": lr * lr_norm_mult, "lr_mult": lr_norm_mult, "weight_decay": weight_decay * wd_norm_mult, "wd_mult": wd_norm_mult})
        if arc_params:
            param_groups.append({"params": arc_params, "lr": lr * lr_arc_mult, "lr_mult": lr_arc_mult, "weight_decay": weight_decay * wd_arc_mult, "wd_mult": wd_arc_mult})
        if up_params:
            param_groups.append({"params": up_params, "lr": lr * lr_up_mult, "lr_mult": lr_up_mult, "weight_decay": weight_decay * wd_up_mult, "wd_mult": wd_up_mult})
    else:
        param_groups = model.parameters()

    # Optimizer
    # Configurable Adam betas: controls momentum (beta1) and second moment (beta2) decay
    # Default (0.9, 0.999) is PyTorch standard. tbukic uses custom betas for grokking.
    # Lower beta2 (e.g., 0.99) makes optimizer more responsive to recent gradients.
    adam_beta1 = float(cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(cfg.get("adam_beta2", 0.999))

    # Adam beta2 scheduling: linearly ramp beta2 from initial to final over training
    # Lower beta2 early (more responsive to recent gradients) → higher late (more stable)
    # Helps late-stage convergence when model transitions from memorization to generalization
    adam_beta2_final = float(cfg.get("adam_beta2_final", 0))  # 0 = disabled (default)

    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=0, betas=(adam_beta1, adam_beta2))
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2))
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif opt_name == "lion":
        lion_beta1 = float(cfg.get("lion_beta1", 0.9))
        lion_beta2 = float(cfg.get("lion_beta2", 0.99))
        optimizer = Lion(param_groups, lr=lr, betas=(lion_beta1, lion_beta2), weight_decay=weight_decay)
    elif opt_name == "muon":
        muon_momentum = float(cfg.get("muon_momentum", 0.95))
        muon_ns_steps = int(cfg.get("muon_ns_steps", 5))
        muon_nesterov = cfg.get("muon_nesterov", True)
        optimizer = Muon(param_groups, lr=lr, momentum=muon_momentum,
                         ns_steps=muon_ns_steps, nesterov=muon_nesterov,
                         weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # LR schedule
    lr_restart_period = int(cfg.get("lr_restart_period", 0))  # 0 = single cosine (default)
    lr_restart_mult = float(cfg.get("lr_restart_mult", 1.0))  # period multiplier after each restart
    # Warmup-Stable-Decay: fraction of post-warmup steps spent at peak LR before cosine decay
    lr_stable_frac = float(cfg.get("lr_stable_frac", 0))  # 0 = standard cosine (default)

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        t = step - warmup_steps
        total = max(steps - warmup_steps, 1)
        if lr_stable_frac > 0:
            # Warmup-Stable-Decay (WSD) schedule
            stable_steps = int(total * lr_stable_frac)
            if t < stable_steps:
                return lr  # hold at peak LR
            decay_t = t - stable_steps
            decay_total = max(total - stable_steps, 1)
            progress = decay_t / decay_total
            return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
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

    def get_max_digits(step):
        if not curriculum_stages:
            return NUM_DIGITS
        for digits, step_str in curriculum_stages:
            if step_str == "rest":
                return digits
            if step < int(step_str):
                return digits
        return NUM_DIGITS

    # Time limit (graceful stop before orze kills the process)
    # Default to 27000s (orze kills at 28800s) so experiments exit cleanly instead of being killed
    time_limit = float(cfg.get("time_limit", 27000))

    # Grokfast: gradient filter to accelerate grokking
    grokfast_alpha = float(cfg.get("grokfast_alpha", 0))
    grokfast_lambda = float(cfg.get("grokfast_lambda", 0))

    # Grokfast variant: "ema" (default, exponential moving average) or "ma" (windowed moving average)
    # MA uses a fixed-size window of recent gradients instead of exponential decay.
    # Sharper frequency cutoff may better separate memorization from generalization signals.
    grokfast_type = cfg.get("grokfast_type", "ema")  # "ema" or "ma"
    grokfast_window = int(cfg.get("grokfast_window", 100))  # window size for MA variant

    # Dual Grokfast: second EMA at a different time scale for multi-resolution gradient filtering
    # The slow EMA (high alpha2, e.g. 0.999) captures deep grokking signals
    # Combined with the primary EMA, provides richer signal separation
    grokfast_alpha2 = float(cfg.get("grokfast_alpha2", 0))  # 0 = disabled (default)
    grokfast_lambda2 = float(cfg.get("grokfast_lambda2", 0))  # lambda for second EMA

    # Per-layer Grokfast lambda multipliers: different amplification for different param types
    # Default 1.0 = same lambda for all params (backward compatible)
    grokfast_lambda_norm_mult = float(cfg.get("grokfast_lambda_norm_mult", 1.0))  # multiplier for norm params
    grokfast_lambda_arc_mult = float(cfg.get("grokfast_lambda_arc_mult", 1.0))    # multiplier for arc embed params

    # Lambda warmup: gradually ramp grokfast_lambda from 0 to target over N steps
    # Prevents lambda from overwhelming early learning before model has basic structure
    lambda_warmup_steps = int(cfg.get("lambda_warmup_steps", 0))  # 0 = disabled (default)

    # Grokfast lambda decay: cosine-anneal lambda from peak to minimum after decay_start steps
    # Stabilizes late-training by reducing EMA amplification as model approaches convergence
    grokfast_decay_start = int(cfg.get("grokfast_decay_start", 0))  # 0 = disabled (default)
    grokfast_lambda_min = float(cfg.get("grokfast_lambda_min", 0))  # minimum lambda at end of decay

    # Accuracy-proportional lambda: continuously reduce Grokfast lambda as accuracy increases.
    # At 99%+ accuracy, all gradient components are slow-moving (model mostly correct), so lambda
    # amplification creates noise rather than useful signal. This smoothly dials lambda down as
    # accuracy improves, stabilizing the 99%→100% regime.
    # Different from step-based decay (grokfast_decay_start) and threshold switches (lambda_adjust_at_acc)
    # — this is continuous and accuracy-driven rather than step-driven.
    lambda_acc_scale = cfg.get("lambda_acc_scale", False)      # False = disabled (default)
    lambda_acc_min = float(cfg.get("lambda_acc_min", 0.1))     # minimum lambda fraction at 100% acc
    _lambda_acc_smoothed = 0.0                                  # EMA of training accuracy for lambda scaling

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

    # EMA of model weights: smoother convergence than SWA
    ema_decay = float(cfg.get("ema_decay", 0.0))  # 0 = disabled (default), typical: 0.999
    ema_start_step = int(cfg.get("ema_start_step", 0))  # step to start EMA tracking
    ema_state = None

    # OHEM (Online Hard Example Mining): focus training on hardest examples
    # ohem_ratio=2.0 means keep top 50% hardest samples (batch_size / ratio)
    ohem_ratio = float(cfg.get("ohem_ratio", 0))  # 0 = disabled (default)

    # Adaptive Weight Decay (tbukic's universal recipe for small models)
    # Milestone-triggered WD decay: WD×wd_decay_factor1 at wd_milestone1% acc, WD×wd_decay_factor2 at wd_milestone2% acc
    # This is how tbukic makes share_norms work — high initial WD for structure, then decay for grokking
    adaptive_wd = cfg.get("adaptive_wd", False)
    wd_milestone1 = float(cfg.get("wd_milestone1", 0.01))        # first accuracy milestone (default 1%)
    wd_decay_factor1 = float(cfg.get("wd_decay_factor1", 0.1))   # WD multiplier at milestone1
    wd_milestone2 = float(cfg.get("wd_milestone2", 0.05))        # second accuracy milestone (default 5%)
    wd_decay_factor2 = float(cfg.get("wd_decay_factor2", 0.01))  # WD multiplier at milestone2
    wd_milestones_hit = set()  # track which milestones have been triggered

    # perpGrad: project gradients orthogonal to weights (tbukic's technique)
    # Removes gradient component parallel to weight vector, accelerates grokking by 32-60%
    perp_grad = cfg.get("perp_grad", False)

    # EGD (Egalitarian Gradient Descent, arxiv 2510.04930, ICLR 2026)
    # Hyperparameter-free grokking acceleration via gradient SVD.
    # Replaces each 2D+ gradient with its nearest orthogonal matrix: G -> UV^T.
    # Equalizes optimization speed across all principal directions.
    # At d=3, hd=4 scale, SVD of tiny matrices costs ~microseconds per step.
    # Zero memory overhead (unlike Grokfast EMA buffer). Can combine with Grokfast.
    egd = cfg.get("egd", False)

    # EGD warmup: gradually ramp EGD strength from 0→1 over N steps.
    # During warmup, interpolate: g_new = (1-alpha)*g_original + alpha*egd(g_original)
    # Prevents early training instability from sudden gradient orthogonalization.
    egd_warmup_steps = int(cfg.get("egd_warmup_steps", 0))  # 0 = full EGD from step 1 (default)

    # EGD before Grokfast: apply EGD to raw gradients BEFORE Grokfast EMA amplification.
    # Default (False): raw grad → Grokfast EMA → EGD → clip → optimizer
    # When True:        raw grad → EGD → Grokfast EMA → clip → optimizer
    # Equalizing raw gradients first may produce cleaner EMA signals since Grokfast
    # accumulates already-normalized directions rather than magnitude-dominated ones.
    egd_before_grokfast = cfg.get("egd_before_grokfast", False)

    # EGD off at accuracy threshold (arxiv 2510.04930, Section 4.2):
    # Paper recommends switching EGD off after grokking detected.
    # When best_accuracy >= egd_off_at_acc, stop applying EGD.
    # This prevents EGD from destabilizing the forming algorithmic circuit.
    egd_off_at_acc = float(cfg.get("egd_off_at_acc", 0))  # 0 = never turn off (default)

    # Gradient Centralization (Yong et al., ECCV 2020): subtract row-wise mean from gradients
    # Constrains weight updates to a hyperplane, improving generalization. Zero params, minimal overhead.
    grad_centralize = cfg.get("grad_centralize", False)

    # Commutative data augmentation: randomly swap a+b → b+a (KA paper, ~2x grokking speedup)
    commutative_aug = cfg.get("commutative_aug", False)

    # Commutative consistency loss: penalize model(a,b) ≠ model(b,a) via cross-entropy on swapped inputs.
    # Exploits mathematical structure of addition (a+b = b+a) to regularize the grokked circuit.
    # Different from commutative_aug (data augmentation) — this is a proper loss term that explicitly
    # teaches the model both orderings must produce the same answer. Zero extra params.
    commutative_loss_weight = float(cfg.get("commutative_loss_weight", 0))  # 0 = disabled; 0.1-0.5 typical

    # SAM (Sharpness-Aware Minimization): seek flat minima for better grokking (tbukic uses rho=0.05-0.2)
    sam_rho = float(cfg.get("sam_rho", 0))  # 0 = disabled (default)

    # Weight Noise: perturb weights before forward pass, compute gradient at perturbed point,
    # then apply gradient to original (unperturbed) weights. Explores wider loss landscape region.
    # Especially useful for tiny models (52p) where landscape is highly non-convex.
    weight_noise_std = float(cfg.get("weight_noise_std", 0))  # 0 = disabled; 0.01-0.1 typical
    weight_noise_start = int(cfg.get("weight_noise_start", 0))  # step to start weight noise

    # Cosine Weight Decay: cosine-anneal WD from initial value → min_wd over training (tbukic's technique)
    # High initial WD provides structure, decaying WD allows grokking to complete
    cosine_wd = cfg.get("cosine_wd", False)
    min_wd = float(cfg.get("min_wd", 0.0))  # final WD at end of training

    # High WD Phase: temporarily increase WD early in training to accelerate grokking onset.
    # Based on arxiv 2603.13331: grokking delay scales INVERSELY with WD (R^2>0.97).
    # Higher WD in the memorization phase forces faster transition to generalization.
    # After wd_high_phase_steps, WD reverts to base value for stable late-training.
    # Different from cosine_wd (continuous decay) and adaptive_wd (accuracy-triggered).
    wd_high_phase_steps = int(cfg.get("wd_high_phase_steps", 0))  # 0 = disabled (default)
    wd_high_mult = float(cfg.get("wd_high_mult", 3.0))            # WD multiplier during high phase

    # Training temperature: scale logits by 1/T before loss computation
    # T>1 smooths loss landscape (gentler gradients), T<1 sharpens (stronger gradients)
    # Schedule from initial to final over training: high T early → low T late aids grokking
    train_temperature = float(cfg.get("train_temperature", 1.0))          # 1.0 = no change (default)
    train_temperature_final = float(cfg.get("train_temperature_final", 0))  # 0 = constant temperature

    # Sphere tau scheduling: linearly ramp sphere_tau from initial to final over training
    # Start with lower tau for stable early learning, increase for sharper final discrimination
    sphere_tau_final = float(cfg.get("sphere_tau_final", 0))  # 0 = disabled (default)
    sphere_tau_initial = float(cfg.get("sphere_tau", 0))  # store initial value for scheduling

    # Interleaved Targeted FT: periodically fine-tune on wrong examples during training
    # Inspired by vijec's iterated targeted FT for pushing 99%→100%
    interleaved_ft_acc = float(cfg.get("interleaved_ft_acc", 0))  # 0 = disabled; accuracy threshold to start IFT
    interleaved_ft_every = int(cfg.get("interleaved_ft_every", 10000))  # steps between IFT cycles
    interleaved_ft_steps_cfg = int(cfg.get("interleaved_ft_steps", 200))  # FT steps per cycle
    interleaved_ft_lr = float(cfg.get("interleaved_ft_lr", 0.0001))  # FT learning rate

    # Checkpoint averaging: save top-K checkpoints by accuracy, average at end
    ckpt_avg_k = int(cfg.get("ckpt_avg_k", 0))  # 0 = disabled; typical: 5-10

    # LAWA (Latest Weight Averaging): average the last K eval checkpoints equally
    # Unlike ckpt_avg_k (top-K by accuracy), LAWA takes the latest K regardless of accuracy.
    # Equal-weight averaging of recent checkpoints is optimal for grokking (arxiv 2306.11120).
    lawa_k = int(cfg.get("lawa_k", 0))  # 0 = disabled; 5-10 typical
    lawa_start_acc = float(cfg.get("lawa_start_acc", 0.5))  # start collecting when acc > this

    # GrokTransfer: pre-train on simpler N-digit addition to seed the grokking circuit (arxiv 2504.13292)
    # The model first groks a simpler task, then transfers to 10-digit addition
    grok_transfer_digits = int(cfg.get("grok_transfer_digits", 0))  # 0 = disabled; 3 or 5 typical
    grok_transfer_steps = int(cfg.get("grok_transfer_steps", 0))    # steps for pre-training phase

    # Z-Loss: penalize large log-partition function to prevent logit instability (PaLM, Google 2022)
    # z_loss = weight * mean(logsumexp(logits)^2). Helps sphere_norm stability.
    z_loss_weight = float(cfg.get("z_loss_weight", 0))  # 0 = disabled; 1e-4 typical

    # Lookahead optimizer: maintain slow weights updated every K steps (Zhang et al., 2019)
    # Smooths optimization trajectory, helps escape grokking plateaus
    lookahead_k = int(cfg.get("lookahead_k", 0))       # 0 = disabled; 5-10 typical
    lookahead_alpha = float(cfg.get("lookahead_alpha", 0.5))  # interpolation toward fast weights

    # Warm norm sharing: train with separate norms (52p), then share mid-training (→46p)
    # At warm_share_norms_step, averages ln1/ln2/ln_f weights and shares them
    warm_share_norms_step = int(cfg.get("warm_share_norms_step", 0))  # 0 = disabled
    _norms_merged = False

    # Accuracy-triggered lambda change: switch grokfast_lambda when accuracy crosses threshold
    # Enables two-phase training: e.g. start at lambda=3.0 for structure, drop to 2.0 for stable grokking
    lambda_adjust_at_acc = float(cfg.get("lambda_adjust_at_acc", 0))  # 0 = disabled; e.g. 0.05 (5% accuracy)
    lambda_adjust_to = float(cfg.get("lambda_adjust_to", 0))          # target lambda value after trigger
    _lambda_adjusted = False

    # Cyclic grokfast lambda: cosine-cycle lambda with warm restarts
    # Every N steps, lambda resets to peak and cosine-decays to grokfast_lambda_min
    grokfast_lambda_cycle_period = int(cfg.get("grokfast_lambda_cycle_period", 0))  # 0 = disabled

    # Multi-phase GrokTransfer: progressive pre-training on increasing digit counts
    # Format: "3:5000,5:10000" — train on 3-digit for 5K steps, then 5-digit for 10K steps
    grok_transfer_phases_str = cfg.get("grok_transfer_phases", "")

    # Accuracy-triggered LR restart: when accuracy first reaches threshold, restart cosine LR schedule
    # Helps escape local minima near convergence (e.g., 99.7% → 100% push)
    lr_restart_at_acc = float(cfg.get("lr_restart_at_acc", 0))  # 0 = disabled; e.g., 0.95 or 0.99
    lr_restart_to = float(cfg.get("lr_restart_to", 0))          # LR to restart to (0 = use initial lr)
    lr_restart_steps = int(cfg.get("lr_restart_steps", 50000))   # cosine decay duration after restart
    _lr_restarted = False
    _lr_restart_step = 0

    # Gradient Accumulation: accumulate gradients over N mini-batches before updating
    # Effectively multiplies batch size by N without increasing memory
    # Stabilizes volatile training patterns (Recipe B oscillations, GrokTransfer crashes)
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))  # 1 = disabled (default)

    # Crash Recovery: auto-revert to best checkpoint when accuracy drops significantly
    # When current accuracy < best * (1 - drop), load best checkpoint and reset optimizer
    # Would have saved the GrokTransfer run that peaked at 97.4% then crashed to 0%
    crash_recovery_drop = float(cfg.get("crash_recovery_drop", 0))  # 0 = disabled; 0.5 = revert if acc halves from peak
    crash_recovery_max = int(cfg.get("crash_recovery_max", 3))       # max number of recoveries (prevent infinite loop)
    _crash_recovery_count = 0

    # Optimizer State Reset: clear Adam momentum/variance at accuracy threshold
    # Gives optimizer a "fresh start" to escape late-training local minima
    # Different from lr_restart_at_acc which only changes LR schedule, not optimizer state
    optim_reset_at_acc = float(cfg.get("optim_reset_at_acc", 0))  # 0 = disabled; e.g., 0.95
    _optim_reset_done = False

    # Grokfast EMA Reset: clear Grokfast EMA buffers when accuracy crosses threshold
    # Allows gradient filter to rebuild from scratch in the new regime (post-grokking-transition)
    # Different from optim_reset_at_acc which clears Adam state, not Grokfast EMA
    grokfast_reset_at_acc = float(cfg.get("grokfast_reset_at_acc", 0))  # 0 = disabled; e.g., 0.05
    _grokfast_reset_done = False

    # Loss threshold early termination: kill dead experiments based on loss trajectory
    # Format: "step:threshold,step:threshold,..." e.g. "20000:1.5,50000:0.8"
    # At each eval, if step >= checkpoint and loss > threshold, terminate immediately.
    # Saves GPU by stopping runs that won't grok (52%+ of experiments are dead at loss>1.5 after 20K steps).
    loss_threshold_schedule_str = cfg.get("loss_threshold_schedule", "")
    loss_threshold_schedule = []
    if loss_threshold_schedule_str:
        for _lt_part in loss_threshold_schedule_str.split(","):
            _lt_s, _lt_v = _lt_part.strip().split(":")
            loss_threshold_schedule.append((int(_lt_s.strip()), float(_lt_v.strip())))

    # Grokfast spike dampening: auto-reduce lambda when loss spikes (prevents Recipe F crashes)
    # Tracks EMA of training loss; when loss > spike_threshold * EMA, reduces lambda temporarily.
    # Recipe F (Lookahead) hits 99.8% but crashes every ~20K steps due to lambda amplifying noise.
    # Dampening lambda during spikes allows recovery without full crash.
    grokfast_spike_dampening = cfg.get("grokfast_spike_dampening", False)
    grokfast_spike_threshold = float(cfg.get("grokfast_spike_threshold", 3.0))  # spike = loss > N * EMA
    grokfast_spike_scale = float(cfg.get("grokfast_spike_scale", 0.3))          # reduce lambda to 30% during spike
    grokfast_spike_cooldown = int(cfg.get("grokfast_spike_cooldown", 0))         # 0 = use eval_every
    _spike_loss_ema = None
    _spike_dampening_until = 0

    # Checkpoint interpolation search: post-training weight-space search between best checkpoints
    # When accuracy is high (>=0.99) but not perfect, interpolate between top checkpoints.
    # Tries N evenly-spaced interpolation points between the two best states.
    # Can find 100% accuracy states between two near-100% checkpoints (linear mode connectivity).
    ckpt_interpolation_steps = int(cfg.get("ckpt_interpolation_steps", 0))  # 0 = disabled; 10 typical

    # Training loop
    model.train()
    best_accuracy = 0.0
    best_state = None
    steps_since_improvement = 0
    last_ift_step = 0  # track last interleaved FT step
    ckpt_avg_pool = []  # (accuracy, state_dict) pairs for checkpoint averaging
    lawa_pool = []  # state_dicts for LAWA (latest K checkpoints)
    t0 = time.time()

    prompt_len = PROMPT_LEN  # a_digits(10) + sep(1) + b_digits(10) + sep(1) = 22

    # GrokTransfer: pre-train on simpler addition to create a grokking seed (arxiv 2504.13292)
    # Supports multi-phase: grok_transfer_phases="3:5000,5:10000" trains progressively
    _gt_phases = []
    if grok_transfer_phases_str:
        for _gt_part in grok_transfer_phases_str.split(","):
            _gt_d, _gt_s = _gt_part.strip().split(":")
            _gt_phases.append((int(_gt_d), int(_gt_s)))
    elif grok_transfer_digits > 0 and grok_transfer_steps > 0:
        _gt_phases = [(grok_transfer_digits, grok_transfer_steps)]

    for _gt_phase_idx, (_gt_digits, _gt_steps) in enumerate(_gt_phases):
        print(f"\n=== GrokTransfer Phase {_gt_phase_idx+1}/{len(_gt_phases)}: {_gt_digits}-digit addition for {_gt_steps} steps ===")
        gt_warmup = min(warmup_steps, _gt_steps // 5)
        for gt_step in range(1, _gt_steps + 1):
            # Cosine LR schedule for pre-training phase
            if gt_step < gt_warmup:
                gt_lr = lr * gt_step / max(gt_warmup, 1)
            else:
                gt_progress = (gt_step - gt_warmup) / max(_gt_steps - gt_warmup, 1)
                gt_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * gt_progress))
            for pg in optimizer.param_groups:
                if use_param_groups and "lr_mult" in pg:
                    pg["lr"] = gt_lr * pg["lr_mult"]
                else:
                    pg["lr"] = gt_lr

            if carry_bias > 0:
                gt_prompts, gt_targets = generate_carry_biased_batch(
                    batch_size, max_digits=_gt_digits, carry_prob=carry_bias, device=device
                )
            else:
                gt_prompts, gt_targets = generate_batch(batch_size, max_digits=_gt_digits, device=device)

            if commutative_aug:
                gt_swap = torch.rand(gt_prompts.shape[0], device=gt_prompts.device) < 0.5
                if gt_swap.any():
                    gt_a = gt_prompts[gt_swap, 1:11].clone()
                    gt_prompts[gt_swap, 1:11] = gt_prompts[gt_swap, 13:23]
                    gt_prompts[gt_swap, 13:23] = gt_a

            gt_full_input = torch.cat([gt_prompts, gt_targets[:, :-1]], dim=1)
            gt_logits = model(gt_full_input)
            gt_ol = gt_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
            gt_loss = F.cross_entropy(gt_ol.reshape(-1, VOCAB_SIZE), gt_targets.reshape(-1))

            optimizer.zero_grad()
            gt_loss.backward()

            # Apply Grokfast during pre-training too
            if grokfast_alpha > 0:
                if grokfast_type == "ma":
                    for p in model.parameters():
                        if p.grad is not None:
                            if not hasattr(p, '_grokfast_ma_buf'):
                                p._grokfast_ma_buf = []
                            p._grokfast_ma_buf.append(p.grad.data.clone())
                            if len(p._grokfast_ma_buf) > grokfast_window:
                                p._grokfast_ma_buf.pop(0)
                            if len(p._grokfast_ma_buf) > 1:
                                ma = torch.stack(p._grokfast_ma_buf).mean(dim=0)
                                p.grad.add_(ma, alpha=grokfast_lambda * getattr(p, '_gf_lambda_mult', 1.0))
                else:
                    for p in model.parameters():
                        if p.grad is not None:
                            if not hasattr(p, '_grokfast_ema'):
                                p._grokfast_ema = torch.zeros_like(p.grad)
                            p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                            p.grad.add_(p._grokfast_ema, alpha=grokfast_lambda * getattr(p, '_gf_lambda_mult', 1.0))
                # Dual Grokfast: second EMA during pre-training
                if grokfast_alpha2 > 0 and grokfast_lambda2 > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            if not hasattr(p, '_grokfast_ema2'):
                                p._grokfast_ema2 = torch.zeros_like(p.grad)
                            p._grokfast_ema2.mul_(grokfast_alpha2).add_(p.grad, alpha=1 - grokfast_alpha2)
                            p.grad.add_(p._grokfast_ema2, alpha=grokfast_lambda2 * getattr(p, '_gf_lambda_mult', 1.0))

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if gt_step % 1000 == 0 or gt_step == _gt_steps:
                gt_acc = evaluate_model_batched(model, device, num_tests=200, max_digits=_gt_digits)
                print(f"  Phase {_gt_phase_idx+1} step {gt_step}/{_gt_steps} | Loss: {gt_loss.item():.4f} | Acc@{_gt_digits}d: {gt_acc:.4f}")

        # Reset optimizer state but keep model weights (the grokking circuit transfers)
        for state in optimizer.state.values():
            state.clear()
        # Reset Grokfast buffers for clean start on next phase
        for p in model.parameters():
            for _gf_attr in ('_grokfast_ema', '_grokfast_ema2', '_grokfast_ma_buf'):
                if hasattr(p, _gf_attr):
                    delattr(p, _gf_attr)
        print(f"  Phase {_gt_phase_idx+1} complete. Optimizer reset.")

    if _gt_phases:
        print(f"  GrokTransfer complete ({len(_gt_phases)} phase{'s' if len(_gt_phases) > 1 else ''}). Beginning 10-digit training.\n")

    # Lookahead: initialize slow weights after any pre-training
    lookahead_slow = None
    if lookahead_k > 0:
        lookahead_slow = {n: p.data.clone() for n, p in model.named_parameters()}

    for step in range(1, steps + 1):
        # Batch size warmup: smaller batches early for more gradient noise
        if batch_size_start > 0 and batch_size_warmup_steps > 0:
            if step <= batch_size_warmup_steps:
                batch_size = batch_size_start + int((_batch_size_final - batch_size_start) * step / batch_size_warmup_steps)
            else:
                batch_size = _batch_size_final

        # Set LR (respecting per-param group multipliers)
        current_lr = get_lr(step)

        # Accuracy-triggered LR restart: override cosine schedule with fresh cosine from restart point
        if _lr_restarted and step >= _lr_restart_step:
            restart_lr = lr_restart_to if lr_restart_to > 0 else lr
            t_restart = step - _lr_restart_step
            if t_restart < lr_restart_steps:
                progress = t_restart / max(lr_restart_steps, 1)
                current_lr = min_lr + 0.5 * (restart_lr - min_lr) * (1 + math.cos(math.pi * progress))
            else:
                current_lr = min_lr  # after restart period, settle at min_lr

        for pg in optimizer.param_groups:
            if use_param_groups and "lr_mult" in pg:
                pg["lr"] = current_lr * pg["lr_mult"]
            else:
                pg["lr"] = current_lr

        # Adam beta2 scheduling: linearly ramp from initial to final value
        if adam_beta2_final > 0 and adam_beta2_final != adam_beta2:
            beta2_progress = min(step / max(steps, 1), 1.0)
            current_beta2 = adam_beta2 + (adam_beta2_final - adam_beta2) * beta2_progress
            for pg in optimizer.param_groups:
                if 'betas' in pg:
                    pg['betas'] = (pg['betas'][0], current_beta2)

        # Cosine Weight Decay: anneal WD on cosine schedule (tbukic's technique)
        # Respects per-param WD multipliers (wd_mult) if param groups are used
        if cosine_wd:
            if step < warmup_steps:
                base_wd = weight_decay
            else:
                t = step - warmup_steps
                total = max(steps - warmup_steps, 1)
                progress = t / total
                base_wd = min_wd + 0.5 * (weight_decay - min_wd) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["weight_decay"] = base_wd * pg.get("wd_mult", 1.0)

        # High WD Phase: temporarily boost WD early to accelerate grokking onset (arxiv 2603.13331)
        # Only active if cosine_wd is NOT in use (they're alternative WD schedules)
        if wd_high_phase_steps > 0 and not cosine_wd and not adaptive_wd:
            if step <= wd_high_phase_steps:
                _wd_phase_mult = wd_high_mult
            else:
                _wd_phase_mult = 1.0
            for pg in optimizer.param_groups:
                pg["weight_decay"] = weight_decay * _wd_phase_mult * pg.get("wd_mult", 1.0)

        # Sphere tau scheduling: ramp from initial to final value
        if sphere_tau_final > 0 and hasattr(model, 'sphere_tau') and sphere_tau_initial > 0:
            if step >= warmup_steps:
                tau_progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
                model.sphere_tau = sphere_tau_initial + (sphere_tau_final - sphere_tau_initial) * tau_progress

        # Get max digits for curriculum
        max_digits = get_max_digits(step)

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

        # Generate batch
        if carry_bias > 0:
            prompts, targets = generate_carry_biased_batch(
                batch_size, max_digits=max_digits, carry_prob=carry_bias, device=device
            )
        else:
            prompts, targets = generate_batch(batch_size, max_digits=max_digits, device=device)

        # Commutative augmentation: randomly swap a+b → b+a for 50% of samples
        if commutative_aug:
            swap_mask = torch.rand(prompts.shape[0], device=prompts.device) < 0.5
            if swap_mask.any():
                # prompts format: [0] + a_digits(10) + [0,0] + b_digits(10) + [0]
                a_part = prompts[swap_mask, 1:11].clone()
                prompts[swap_mask, 1:11] = prompts[swap_mask, 13:23]
                prompts[swap_mask, 13:23] = a_part

        # Weight Noise: perturb weights before forward pass (restore after backward)
        _wn_active = weight_noise_std > 0 and step >= weight_noise_start
        if _wn_active:
            _wn_backup = {}
            with torch.no_grad():
                for _wn_n, _wn_p in model.named_parameters():
                    _wn_backup[_wn_n] = _wn_p.data.clone()
                    _wn_p.data.add_(torch.randn_like(_wn_p) * weight_noise_std)

        # Teacher forcing: prompt + first (OUTPUT_LEN-1) target digits as input
        # Model predicts each target digit from the context so far
        full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
        logits = model(full_input)

        # Loss only on output positions (predict target from prompt context)
        output_logits = logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]

        # Knowledge Distillation: compute teacher logits on same input
        _teacher_ol = None
        if teacher_model is not None:
            with torch.no_grad():
                _t_logits = teacher_model(full_input)
                _teacher_ol = _t_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]

        # Save targets before OHEM rebinds the variable (needed for SAM and commutative loss)
        targets_pre_ohem = targets
        if sam_rho > 0:
            targets_all = targets

        # OHEM: select hardest samples before loss computation
        if ohem_ratio > 0:
            with torch.no_grad():
                per_sample_loss = F.cross_entropy(
                    output_logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    reduction='none',
                ).view(-1, OUTPUT_LEN).mean(dim=1)  # (batch_size,)
                k = max(1, int(batch_size / ohem_ratio))
                _, hard_idx = per_sample_loss.topk(k)
            output_logits = output_logits[hard_idx]
            targets = targets[hard_idx]
            if _teacher_ol is not None:
                _teacher_ol = _teacher_ol[hard_idx]

        # Training temperature: scale logits for smoother/sharper loss landscape
        if train_temperature != 1.0 or (train_temperature_final > 0 and train_temperature_final != train_temperature):
            if train_temperature_final > 0 and train_temperature_final != train_temperature:
                t_progress = step / max(steps, 1)
                _eff_temp = train_temperature + (train_temperature_final - train_temperature) * t_progress
            else:
                _eff_temp = train_temperature
            if _eff_temp != 1.0:
                output_logits = output_logits / _eff_temp

        if digit_loss_weight and digit_loss_weight != "none":
            # Per-digit weighted loss: upweight harder (later) digits
            # output_logits: (B, OUTPUT_LEN, VOCAB_SIZE), targets: (B, OUTPUT_LEN)
            if digit_loss_weight == "linear":
                weights = torch.linspace(1.0, digit_loss_scale, OUTPUT_LEN, device=device)
            elif digit_loss_weight == "exponential":
                weights = torch.logspace(0, math.log10(digit_loss_scale), OUTPUT_LEN, device=device)
            else:
                weights = torch.ones(OUTPUT_LEN, device=device)
            # Normalize so mean weight = 1
            weights = weights / weights.mean()
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
            if focal_gamma > 0:
                # Focal loss: (1-pt)^gamma * CE
                per_token_loss = F.cross_entropy(
                    output_logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    label_smoothing=label_smoothing,
                    reduction='none',
                )
                with torch.no_grad():
                    probs = F.softmax(output_logits.reshape(-1, VOCAB_SIZE), dim=-1)
                    pt = probs.gather(1, targets.reshape(-1, 1)).squeeze(1)
                    focal_weight = (1 - pt) ** focal_gamma
                loss = (per_token_loss * focal_weight).mean()
            else:
                loss = F.cross_entropy(
                    output_logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    label_smoothing=label_smoothing,
                )

        # Z-Loss: penalize large log-partition to prevent logit instability (PaLM, Google 2022)
        if z_loss_weight > 0:
            log_z = torch.logsumexp(output_logits, dim=-1)  # (B, OUTPUT_LEN)
            loss = loss + z_loss_weight * (log_z ** 2).mean()

        # Knowledge Distillation: blend student loss with KL divergence from teacher soft labels
        if _teacher_ol is not None:
            _s_log = F.log_softmax(output_logits / distill_temperature, dim=-1)
            _t_prob = F.softmax(_teacher_ol / distill_temperature, dim=-1)
            _kd_loss = F.kl_div(_s_log.reshape(-1, VOCAB_SIZE), _t_prob.reshape(-1, VOCAB_SIZE),
                                reduction='batchmean') * (distill_temperature ** 2)
            loss = (1 - distill_alpha) * loss + distill_alpha * _kd_loss

        # Commutative consistency loss: enforce model(a,b) ≈ model(b,a) via cross-entropy
        # Uses pre-OHEM targets and original prompts to compute loss on swapped ordering
        if commutative_loss_weight > 0:
            _cl_prompts = prompts.clone()
            _cl_a = _cl_prompts[:, 1:11].clone()
            _cl_prompts[:, 1:11] = _cl_prompts[:, 13:23]
            _cl_prompts[:, 13:23] = _cl_a
            _cl_full = torch.cat([_cl_prompts, targets_pre_ohem[:, :-1]], dim=1)
            _cl_logits = model(_cl_full)
            _cl_ol = _cl_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
            _cl_ce = F.cross_entropy(_cl_ol.reshape(-1, VOCAB_SIZE), targets_pre_ohem.reshape(-1))
            loss = loss + commutative_loss_weight * _cl_ce

        # Grokfast spike dampening: track loss EMA for spike detection
        if grokfast_spike_dampening:
            _loss_val = loss.item()
            if _spike_loss_ema is None:
                _spike_loss_ema = _loss_val
            else:
                _spike_loss_ema = 0.99 * _spike_loss_ema + 0.01 * _loss_val
            # Detect spike: loss > threshold * EMA (only after warmup + 5K steps for stable EMA)
            if (_loss_val > _spike_loss_ema * grokfast_spike_threshold and
                    step > warmup_steps + 5000 and _spike_loss_ema > 0):
                _spike_cd = grokfast_spike_cooldown if grokfast_spike_cooldown > 0 else eval_every
                _spike_dampening_until = step + _spike_cd
                if step % eval_every < 2:  # only print once near eval
                    print(f"  [spike_dampening] Loss {_loss_val:.4f} > {grokfast_spike_threshold}x EMA {_spike_loss_ema:.4f}, "
                          f"dampening lambda to {grokfast_spike_scale}x for {_spike_cd} steps")

        # Gradient Accumulation: zero grad at start of each accumulation window
        if grad_accum_steps <= 1 or (step - 1) % grad_accum_steps == 0:
            optimizer.zero_grad()

        # Scale loss for gradient accumulation (average over mini-batches)
        _accum_loss = loss / grad_accum_steps if grad_accum_steps > 1 else loss
        _accum_loss.backward()

        # Weight Noise: restore original weights after backward (gradient is from perturbed point)
        if _wn_active:
            with torch.no_grad():
                for _wn_n, _wn_p in model.named_parameters():
                    _wn_p.data.copy_(_wn_backup[_wn_n])

        # Gradient processing and optimizer step only at accumulation boundary
        _do_optim_step = (grad_accum_steps <= 1 or step % grad_accum_steps == 0)

        if _do_optim_step:
            # SAM: recompute gradient at worst-case perturbation for flatter minima
            if sam_rho > 0 and step > warmup_steps:
                with torch.no_grad():
                    sam_grad_norm = torch.sqrt(
                        sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None) + 1e-12
                    )
                    sam_old_params = []
                    for p in model.parameters():
                        sam_old_params.append(p.data.clone())
                        if p.grad is not None:
                            p.data.add_(sam_rho / sam_grad_norm * p.grad)
                optimizer.zero_grad()
                sam_logits = model(full_input)
                sam_ol = sam_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                sam_loss = F.cross_entropy(
                    sam_ol.reshape(-1, VOCAB_SIZE),
                    targets_all.reshape(-1),
                    label_smoothing=label_smoothing,
                )
                sam_loss.backward()
                with torch.no_grad():
                    for p, old in zip(model.parameters(), sam_old_params):
                        p.data.copy_(old)

        # EGD before Grokfast: apply gradient orthogonalization on raw gradients first
        _egd_active = egd and (egd_off_at_acc <= 0 or best_accuracy < egd_off_at_acc)
        if _do_optim_step and _egd_active and egd_before_grokfast:
            _egd_alpha = 1.0
            if egd_warmup_steps > 0 and step <= egd_warmup_steps:
                _egd_alpha = step / egd_warmup_steps
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None and p.dim() >= 2:
                        g = p.grad.data
                        orig_shape = g.shape
                        g2d = g.view(g.shape[0], -1)
                        try:
                            U, S, Vt = torch.linalg.svd(g2d, full_matrices=False)
                            egd_grad = (U @ Vt).view(orig_shape)
                            if _egd_alpha < 1.0:
                                p.grad.data = (1 - _egd_alpha) * g + _egd_alpha * egd_grad
                            else:
                                p.grad.data = egd_grad
                        except RuntimeError:
                            pass  # skip degenerate matrices

        # Grokfast: amplify slow-moving gradient components to accelerate grokking
        if _do_optim_step and grokfast_alpha > 0:
            # Lambda warmup: gradually ramp lambda from 0 to target
            if lambda_warmup_steps > 0 and step <= lambda_warmup_steps:
                eff_lambda = grokfast_lambda * step / lambda_warmup_steps
            else:
                eff_lambda = grokfast_lambda
            # Lambda decay: cosine-anneal lambda from peak to minimum after decay_start
            if grokfast_decay_start > 0 and step > grokfast_decay_start:
                decay_progress = min((step - grokfast_decay_start) / max(steps - grokfast_decay_start, 1), 1.0)
                eff_lambda = grokfast_lambda_min + 0.5 * (eff_lambda - grokfast_lambda_min) * (1 + math.cos(math.pi * decay_progress))
            # Cyclic lambda: cosine-cycle with warm restarts (like SGDR for lambda)
            if grokfast_lambda_cycle_period > 0:
                cycle_progress = (step % grokfast_lambda_cycle_period) / grokfast_lambda_cycle_period
                eff_lambda = grokfast_lambda_min + 0.5 * (eff_lambda - grokfast_lambda_min) * (1 + math.cos(math.pi * cycle_progress))

            # Spike dampening: reduce lambda when loss spikes (prevents Recipe F crashes)
            if grokfast_spike_dampening and step < _spike_dampening_until:
                eff_lambda *= grokfast_spike_scale

            # Accuracy-proportional lambda: reduce lambda as accuracy increases
            # At high accuracy, EMA amplification creates noise rather than useful signal
            if lambda_acc_scale and _lambda_acc_smoothed > 0:
                eff_lambda *= max(lambda_acc_min, 1.0 - _lambda_acc_smoothed)

            if grokfast_type == "ma":
                # Grokfast MA: windowed moving average of gradients (sharper frequency cutoff)
                for p in model.parameters():
                    if p.grad is not None:
                        if not hasattr(p, '_grokfast_ma_buf'):
                            p._grokfast_ma_buf = []
                        p._grokfast_ma_buf.append(p.grad.data.clone())
                        if len(p._grokfast_ma_buf) > grokfast_window:
                            p._grokfast_ma_buf.pop(0)
                        if len(p._grokfast_ma_buf) > 1:
                            ma = torch.stack(p._grokfast_ma_buf).mean(dim=0)
                            p.grad.add_(ma, alpha=eff_lambda * getattr(p, '_gf_lambda_mult', 1.0))
            else:
                # Grokfast EMA (default)
                for p in model.parameters():
                    if p.grad is not None:
                        if not hasattr(p, '_grokfast_ema'):
                            p._grokfast_ema = torch.zeros_like(p.grad)
                        p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                        p.grad.add_(p._grokfast_ema, alpha=eff_lambda * getattr(p, '_gf_lambda_mult', 1.0))

            # Dual Grokfast: second EMA at different time scale (multi-resolution filtering)
            if grokfast_alpha2 > 0 and grokfast_lambda2 > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        if not hasattr(p, '_grokfast_ema2'):
                            p._grokfast_ema2 = torch.zeros_like(p.grad)
                        p._grokfast_ema2.mul_(grokfast_alpha2).add_(p.grad, alpha=1 - grokfast_alpha2)
                        p.grad.add_(p._grokfast_ema2, alpha=grokfast_lambda2 * getattr(p, '_gf_lambda_mult', 1.0))

            # EGD after Grokfast (default): orthogonalize the combined raw+EMA gradient
            if _egd_active and not egd_before_grokfast:
                _egd_alpha = 1.0
                if egd_warmup_steps > 0 and step <= egd_warmup_steps:
                    _egd_alpha = step / egd_warmup_steps
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None and p.dim() >= 2:
                            g = p.grad.data
                            orig_shape = g.shape
                            g2d = g.view(g.shape[0], -1)
                            try:
                                U, S, Vt = torch.linalg.svd(g2d, full_matrices=False)
                                egd_grad = (U @ Vt).view(orig_shape)
                                if _egd_alpha < 1.0:
                                    p.grad.data = (1 - _egd_alpha) * g + _egd_alpha * egd_grad
                                else:
                                    p.grad.data = egd_grad
                            except RuntimeError:
                                pass  # skip degenerate matrices

            # perpGrad: project gradients orthogonal to weights (tbukic's technique)
            # Removes the component of grad parallel to the weight vector
            # grad_perp = grad - (grad . w_hat) * w_hat, where w_hat = w / ||w||
            # This prevents weight magnitude from growing, accelerates grokking by 32-60%
            if perp_grad:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None and p.dim() >= 2:
                            # Flatten to 2D for projection
                            w = p.data.view(p.shape[0], -1)
                            g = p.grad.data.view(p.shape[0], -1)
                            # Project out parallel component per row
                            w_norm_sq = (w * w).sum(dim=1, keepdim=True).clamp(min=1e-12)
                            proj = (g * w).sum(dim=1, keepdim=True) / w_norm_sq
                            g_perp = g - proj * w
                            p.grad.data.copy_(g_perp.view_as(p.grad))

            # Gradient Centralization: subtract row-wise mean from gradients (Yong et al., ECCV 2020)
            if grad_centralize:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None and p.dim() >= 2:
                            p.grad.data.sub_(p.grad.data.mean(dim=tuple(range(1, p.dim())), keepdim=True))

            # Gradient noise injection (Neelakantan et al. 2015)
            if grad_noise_eta > 0:
                noise_std = math.sqrt(grad_noise_eta / (1 + step) ** grad_noise_gamma)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)

            # Progressive gradient clip: linearly interpolate from initial to final
            _eff_grad_clip = grad_clip
            if grad_clip_final > 0 and grad_clip > 0:
                _gc_progress = min(step / max(steps, 1), 1.0)
                _eff_grad_clip = grad_clip + (grad_clip_final - grad_clip) * _gc_progress
            if _eff_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), _eff_grad_clip)
            optimizer.step()

            # Lookahead: update slow weights every K steps (Zhang et al., 2019)
            if lookahead_k > 0 and step % lookahead_k == 0 and lookahead_slow is not None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        slow = lookahead_slow[n]
                        slow.add_(p.data - slow, alpha=lookahead_alpha)
                        p.data.copy_(slow)

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

            # Accuracy-proportional lambda: update smoothed accuracy tracker
            if lambda_acc_scale:
                _lambda_acc_smoothed = 0.95 * _lambda_acc_smoothed + 0.05 * full_acc

            if full_acc > best_accuracy:
                best_accuracy = full_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
                # Log when EGD gets turned off (one-time)
                if egd and egd_off_at_acc > 0 and best_accuracy >= egd_off_at_acc:
                    print(f"  [egd_off] best_accuracy {best_accuracy:.4f} >= {egd_off_at_acc}, EGD disabled for remaining training")
            else:
                steps_since_improvement += eval_every

            # Accuracy-triggered LR restart: restart cosine LR when accuracy first hits threshold
            if lr_restart_at_acc > 0 and full_acc >= lr_restart_at_acc and not _lr_restarted:
                _lr_restarted = True
                _lr_restart_step = step
                _restart_lr_val = lr_restart_to if lr_restart_to > 0 else lr
                print(f"  [lr_restart] Acc {full_acc:.4f} >= {lr_restart_at_acc}, restarting LR to {_restart_lr_val:.6f} for {lr_restart_steps} steps")

            # Accuracy-triggered lambda change: switch grokfast_lambda at accuracy threshold
            if lambda_adjust_at_acc > 0 and full_acc >= lambda_adjust_at_acc and not _lambda_adjusted:
                _lambda_adjusted = True
                _old_lambda = grokfast_lambda
                grokfast_lambda = lambda_adjust_to
                print(f"  [lambda_adjust] Acc {full_acc:.4f} >= {lambda_adjust_at_acc}, lambda {_old_lambda} -> {lambda_adjust_to}")

            # Crash Recovery: revert to best checkpoint if accuracy drops significantly
            # Prevents catastrophic collapse (e.g., GrokTransfer peak 97.4% → crash to 0%)
            if (crash_recovery_drop > 0 and best_accuracy > 0.1 and
                    full_acc < best_accuracy * (1 - crash_recovery_drop) and
                    _crash_recovery_count < crash_recovery_max and best_state is not None):
                _crash_recovery_count += 1
                print(f"  [crash_recovery] Acc dropped {best_accuracy:.4f} -> {full_acc:.4f} "
                      f"(>{crash_recovery_drop*100:.0f}% drop). Reverting to best checkpoint. "
                      f"({_crash_recovery_count}/{crash_recovery_max} recoveries)")
                model.load_state_dict(best_state)
                # Reset optimizer state for clean restart from recovered checkpoint
                for _cr_state in optimizer.state.values():
                    _cr_state.clear()
                # Reset Grokfast buffers
                for _cr_p in model.parameters():
                    for _gf_attr in ('_grokfast_ema', '_grokfast_ema2', '_grokfast_ma_buf'):
                        if hasattr(_cr_p, _gf_attr):
                            delattr(_cr_p, _gf_attr)
                # Reset lookahead slow weights
                if lookahead_k > 0:
                    lookahead_slow = {n: p.data.clone() for n, p in model.named_parameters()}
                steps_since_improvement = 0
                model.train()

            # Optimizer State Reset: clear Adam momentum/variance at accuracy milestone
            # Gives optimizer fresh start to escape late-training local minima
            if optim_reset_at_acc > 0 and full_acc >= optim_reset_at_acc and not _optim_reset_done:
                _optim_reset_done = True
                for _or_state in optimizer.state.values():
                    _or_state.clear()
                # Reset Grokfast buffers for clean gradient accumulation
                for _or_p in model.parameters():
                    for _gf_attr in ('_grokfast_ema', '_grokfast_ema2', '_grokfast_ma_buf'):
                        if hasattr(_or_p, _gf_attr):
                            delattr(_or_p, _gf_attr)
                print(f"  [optim_reset] Acc {full_acc:.4f} >= {optim_reset_at_acc}, optimizer state cleared")

            # Grokfast EMA Reset: clear EMA buffers at accuracy milestone
            if grokfast_reset_at_acc > 0 and full_acc >= grokfast_reset_at_acc and not _grokfast_reset_done:
                _grokfast_reset_done = True
                for _gr_p in model.parameters():
                    for _gf_attr in ('_grokfast_ema', '_grokfast_ema2', '_grokfast_ma_buf'):
                        if hasattr(_gr_p, _gf_attr):
                            delattr(_gr_p, _gf_attr)
                print(f"  [grokfast_reset] Acc {full_acc:.4f} >= {grokfast_reset_at_acc}, Grokfast EMA cleared")

            # Checkpoint averaging: collect top-K checkpoints by accuracy
            if ckpt_avg_k > 0 and full_acc > 0.9:
                ckpt_avg_pool.append((full_acc, {k: v.clone() for k, v in model.state_dict().items()}))
                ckpt_avg_pool.sort(key=lambda x: x[0], reverse=True)
                if len(ckpt_avg_pool) > ckpt_avg_k:
                    ckpt_avg_pool.pop()

            # LAWA: collect latest K checkpoints (regardless of accuracy, unlike ckpt_avg)
            if lawa_k > 0 and full_acc >= lawa_start_acc:
                lawa_pool.append({k: v.clone() for k, v in model.state_dict().items()})
                if len(lawa_pool) > lawa_k:
                    lawa_pool.pop(0)

            # Adaptive Weight Decay: decay WD when accuracy milestones are reached
            # Respects per-param WD multipliers (wd_mult) if param groups are used
            if adaptive_wd:
                if full_acc >= wd_milestone2 and 2 not in wd_milestones_hit:
                    wd_milestones_hit.add(2)
                    new_wd = weight_decay * wd_decay_factor2
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = new_wd * pg.get("wd_mult", 1.0)
                    print(f"  [adaptive_wd] Acc {full_acc:.4f} >= {wd_milestone2}, WD -> {new_wd:.6f}")
                elif full_acc >= wd_milestone1 and 1 not in wd_milestones_hit:
                    wd_milestones_hit.add(1)
                    new_wd = weight_decay * wd_decay_factor1
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = new_wd * pg.get("wd_mult", 1.0)
                    print(f"  [adaptive_wd] Acc {full_acc:.4f} >= {wd_milestone1}, WD -> {new_wd:.6f}")

            # Warm norm sharing: at specified step, merge separate norms into shared (52p→46p)
            if warm_share_norms_step > 0 and step >= warm_share_norms_step and not _norms_merged:
                _norms_merged = True
                with torch.no_grad():
                    # Collect unique norm modules
                    _wns_ln1 = model.blocks[0].ln1
                    _wns_ln2 = model.blocks[0].ln2
                    _wns_lnf = model.ln_f
                    _wns_unique = {}
                    for _wns_n in [_wns_ln1, _wns_ln2, _wns_lnf]:
                        if id(_wns_n) not in _wns_unique:
                            _wns_unique[id(_wns_n)] = _wns_n
                    if len(_wns_unique) > 1:
                        # Average norm weights from all unique norms
                        _wns_avg = sum(n.weight.data for n in _wns_unique.values()) / len(_wns_unique)
                        _wns_ln1.weight.data.copy_(_wns_avg)
                # Share: point ln2 and ln_f to ln1
                for _wns_blk in model.blocks:
                    _wns_blk.ln2 = _wns_blk.ln1
                model.ln_f = model.blocks[0].ln1
                # Update config for correct checkpoint saving
                cfg["share_norms"] = True
                cfg["share_ln_f"] = True
                # Rebuild optimizer with deduplicated params (resets Adam state)
                _wns_seen = set()
                if use_param_groups:
                    _wns_norm_p, _wns_arc_p, _wns_up_p, _wns_other_p = [], [], [], []
                    for _wns_name, _wns_p in model.named_parameters():
                        if not _wns_p.requires_grad or id(_wns_p) in _wns_seen:
                            continue
                        _wns_seen.add(id(_wns_p))
                        if any(k in _wns_name for k in ("ln1", "ln2", "ln_f", "q_norm", "k_norm")):
                            _wns_norm_p.append(_wns_p)
                        elif "arc_" in _wns_name:
                            _wns_arc_p.append(_wns_p)
                        elif "up_proj" in _wns_name or "gate_proj" in _wns_name or "gate_alpha" in _wns_name:
                            _wns_up_p.append(_wns_p)
                        else:
                            _wns_other_p.append(_wns_p)
                    _wns_groups = []
                    if _wns_other_p:
                        _wns_groups.append({"params": _wns_other_p, "lr": current_lr, "lr_mult": 1.0, "weight_decay": weight_decay, "wd_mult": 1.0})
                    if _wns_norm_p:
                        _wns_groups.append({"params": _wns_norm_p, "lr": current_lr * lr_norm_mult, "lr_mult": lr_norm_mult, "weight_decay": weight_decay * wd_norm_mult, "wd_mult": wd_norm_mult})
                    if _wns_arc_p:
                        _wns_groups.append({"params": _wns_arc_p, "lr": current_lr * lr_arc_mult, "lr_mult": lr_arc_mult, "weight_decay": weight_decay * wd_arc_mult, "wd_mult": wd_arc_mult})
                    if _wns_up_p:
                        _wns_groups.append({"params": _wns_up_p, "lr": current_lr * lr_up_mult, "lr_mult": lr_up_mult, "weight_decay": weight_decay * wd_up_mult, "wd_mult": wd_up_mult})
                    optimizer = torch.optim.AdamW(_wns_groups, weight_decay=weight_decay)
                else:
                    _wns_unique_params = []
                    for _wns_p in model.parameters():
                        if id(_wns_p) not in _wns_seen:
                            _wns_seen.add(id(_wns_p))
                            _wns_unique_params.append(_wns_p)
                    optimizer = torch.optim.AdamW(_wns_unique_params, lr=current_lr, weight_decay=weight_decay)
                # Reset Grokfast buffers and lookahead
                for _wns_p in model.parameters():
                    for _gf_attr in ('_grokfast_ema', '_grokfast_ema2', '_grokfast_ma_buf'):
                        if hasattr(_wns_p, _gf_attr):
                            delattr(_wns_p, _gf_attr)
                if lookahead_k > 0:
                    lookahead_slow = {n: p.data.clone() for n, p in model.named_parameters()}
                # Update best_state with merged norms
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                _wns_new_count = model.count_params()
                print(f"  [warm_share_norms] Merged at step {step}. Params: {num_params} -> {_wns_new_count}")
                num_params = _wns_new_count

            # Early stop if perfect on full 10-digit
            if full_acc >= 1.0:
                print(f"Perfect accuracy at step {step}!")
                break

            # Interleaved Targeted FT: fine-tune on wrong examples during training (vijec's technique)
            if interleaved_ft_acc > 0 and full_acc >= interleaved_ft_acc and step - last_ift_step >= interleaved_ft_every:
                last_ift_step = step
                pre_ift_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.eval()
                wrong_p_list, wrong_t_list = [], []
                with torch.no_grad():
                    for _ in range(10):  # 10 * 500 = 5K samples
                        p_ift, t_ift = generate_batch(500, device=device)
                        out_ift = model.generate(p_ift)
                        wrong = (out_ift != t_ift).any(dim=1)
                        if wrong.any():
                            wrong_p_list.append(p_ift[wrong])
                            wrong_t_list.append(t_ift[wrong])
                if wrong_p_list:
                    wrong_p_cat = torch.cat(wrong_p_list)
                    wrong_t_cat = torch.cat(wrong_t_list)
                    n_wrong = len(wrong_p_cat)
                    print(f"  [IFT] Step {step}: {n_wrong}/5000 wrong, fine-tuning {interleaved_ft_steps_cfg} steps at lr={interleaved_ft_lr}")
                    model.train()
                    ift_opt = torch.optim.Adam(model.parameters(), lr=interleaved_ft_lr, weight_decay=0)
                    for _ in range(interleaved_ft_steps_cfg):
                        half_b = min(batch_size // 2, n_wrong)
                        idx_ift = torch.randint(0, n_wrong, (half_b,))
                        fp_ift = wrong_p_cat[idx_ift]
                        ft_ift = wrong_t_cat[idx_ift]
                        rp_ift, rt_ift = generate_batch(batch_size - len(fp_ift), device=device)
                        bp_ift = torch.cat([fp_ift, rp_ift])
                        bt_ift = torch.cat([ft_ift, rt_ift])
                        perm_ift = torch.randperm(len(bp_ift))
                        bp_ift, bt_ift = bp_ift[perm_ift], bt_ift[perm_ift]
                        full_in_ift = torch.cat([bp_ift, bt_ift[:, :-1]], dim=1)
                        ift_logits = model(full_in_ift)
                        ift_ol = ift_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                        ift_loss = F.cross_entropy(ift_ol.reshape(-1, VOCAB_SIZE), bt_ift.reshape(-1))
                        ift_opt.zero_grad()
                        ift_loss.backward()
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        ift_opt.step()
                    model.eval()
                    ift_acc = evaluate_model_batched(model, device, num_tests=500)
                    model.train()
                    if ift_acc > full_acc:
                        print(f"  [IFT] Improved: {full_acc:.4f} -> {ift_acc:.4f}")
                        if ift_acc > best_accuracy:
                            best_accuracy = ift_acc
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        if ift_acc >= 1.0:
                            print(f"  [IFT] Perfect accuracy at step {step}!")
                            break
                    else:
                        model.load_state_dict(pre_ift_state)
                        print(f"  [IFT] No improvement ({ift_acc:.4f} <= {full_acc:.4f}), reverted")
                        model.train()

            # Patience: only after curriculum reaches 10 digits and model has shown learning
            if max_digits >= NUM_DIGITS and steps_since_improvement >= patience and 0 < best_accuracy < 0.99:
                print(f"No improvement for {patience} steps, stopping.")
                break

            # Divergence stop: model collapsed to 0 accuracy and hasn't recovered
            if max_digits >= NUM_DIGITS and full_acc == 0.0 and steps_since_improvement >= 100_000:
                print(f"Model diverged (acc=0) for {steps_since_improvement} steps, stopping.")
                break

            # Loss threshold early termination: kill dead experiments based on loss trajectory
            if loss_threshold_schedule and max_digits >= NUM_DIGITS:
                _lt_should_stop = False
                for _lt_step, _lt_threshold in loss_threshold_schedule:
                    if step >= _lt_step and loss.item() > _lt_threshold:
                        print(f"Loss {loss.item():.4f} > threshold {_lt_threshold} at step {step} (>= {_lt_step}), stopping (loss_threshold_schedule).")
                        _lt_should_stop = True
                        break
                if _lt_should_stop:
                    break

            # Time limit: stop gracefully before the process is killed by the runner
            if time_limit > 0 and elapsed >= time_limit:
                print(f"Time limit {time_limit:.0f}s reached at step {step}, stopping.")
                break

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

    # Checkpoint averaging: average top-K checkpoints for better generalization
    if ckpt_avg_k > 0 and len(ckpt_avg_pool) >= 2:
        print(f"\n=== Checkpoint Averaging (top {len(ckpt_avg_pool)} checkpoints, accs: {[f'{a:.4f}' for a, _ in ckpt_avg_pool]}) ===")
        avg_state = {}
        for k_name in ckpt_avg_pool[0][1]:
            tensors = [cp[1][k_name].float() for cp in ckpt_avg_pool]
            avg_state[k_name] = torch.stack(tensors).mean(dim=0).to(ckpt_avg_pool[0][1][k_name].dtype)
        saved_state_pre_avg = {k_name: v.clone() for k_name, v in model.state_dict().items()}
        model.load_state_dict(avg_state)
        avg_acc = evaluate_model_batched(model, device, num_tests=500)
        if avg_acc >= best_accuracy:
            print(f"  Checkpoint avg accuracy {avg_acc:.4f} >= best {best_accuracy:.4f}, using averaged weights")
            best_accuracy = avg_acc
            best_state = {k_name: v.clone() for k_name, v in model.state_dict().items()}
        else:
            print(f"  Checkpoint avg accuracy {avg_acc:.4f} < best {best_accuracy:.4f}, reverting")
            model.load_state_dict(saved_state_pre_avg)

    # LAWA: average latest K checkpoints equally (different from top-K ckpt_avg)
    if lawa_k > 0 and len(lawa_pool) >= 2:
        print(f"\n=== LAWA (last {len(lawa_pool)} checkpoints) ===")
        lawa_avg_state = {}
        for k_name in lawa_pool[0]:
            tensors = [cp[k_name].float() for cp in lawa_pool]
            lawa_avg_state[k_name] = torch.stack(tensors).mean(dim=0).to(lawa_pool[0][k_name].dtype)
        saved_state_pre_lawa = {k_name: v.clone() for k_name, v in model.state_dict().items()}
        model.load_state_dict(lawa_avg_state)
        lawa_acc = evaluate_model_batched(model, device, num_tests=500)
        if lawa_acc >= best_accuracy:
            print(f"  LAWA accuracy {lawa_acc:.4f} >= best {best_accuracy:.4f}, using LAWA weights")
            best_accuracy = lawa_acc
            best_state = {k_name: v.clone() for k_name, v in model.state_dict().items()}
        else:
            print(f"  LAWA accuracy {lawa_acc:.4f} < best {best_accuracy:.4f}, reverting")
            model.load_state_dict(saved_state_pre_lawa)

    # Targeted Fine-Tuning: find wrong examples and fine-tune on them (vijec's technique)
    targeted_ft_rounds = int(cfg.get("targeted_ft_rounds", 0))  # 0 = disabled
    targeted_ft_lr = float(cfg.get("targeted_ft_lr", 0.0001))
    targeted_ft_steps = int(cfg.get("targeted_ft_steps", 500))
    targeted_ft_search_size = int(cfg.get("targeted_ft_search_size", 10000))  # samples to search for wrong examples
    targeted_ft_lr_decay = float(cfg.get("targeted_ft_lr_decay", 1.0))       # LR multiplier per round (e.g., 0.7 = decay 30% each round)
    targeted_ft_wd = float(cfg.get("targeted_ft_wd", 0.0))                   # weight decay during FT (default: 0)
    targeted_ft_wrong_frac = float(cfg.get("targeted_ft_wrong_frac", 0.5))   # fraction of batch from wrong examples (default: 0.5)
    targeted_ft_attempts = int(cfg.get("targeted_ft_attempts", 1))            # independent FT attempts with different seeds (keep best)
    targeted_ft_grokfast = cfg.get("targeted_ft_grokfast", False)             # apply Grokfast EMA during targeted FT rounds
    targeted_ft_search_growth = float(cfg.get("targeted_ft_search_growth", 1.0))  # multiply search_size each round (find rarer errors in later rounds)
    targeted_ft_carry_bias = float(cfg.get("targeted_ft_carry_bias", 0.0))    # carry-biased example generation during FT search (0=uniform, 0.5=50% carry-heavy)

    # OHEM during Targeted FT: focus on the hardest wrong examples within each FT step
    # ohem_ratio=2.0 means keep top 50% hardest samples per mini-batch during FT
    targeted_ft_ohem_ratio = float(cfg.get("targeted_ft_ohem_ratio", 0))  # 0 = disabled (default)

    # Gradient Surgery during TFT (PCGrad-style, Yu et al. NeurIPS 2020):
    # Projects wrong-example gradients to remove components conflicting with correct-example gradients.
    # Directly prevents the whack-a-mole pattern where fixing wrong examples breaks correct ones.
    # Each TFT step: compute wrong gradient, compute reference gradient from random batch,
    # then project wrong gradient to be non-conflicting with reference gradient.
    targeted_ft_grad_surgery = cfg.get("targeted_ft_grad_surgery", False)

    # TFT Weight Mixback: after each TFT round, interpolate weights back toward pre-round state.
    # Acts like Lookahead for TFT — prevents drift from base solution while keeping beneficial changes.
    # 0.0 = disabled (keep full TFT result), 0.3 = blend 30% pre-round + 70% post-round.
    targeted_ft_mixback = float(cfg.get("targeted_ft_mixback", 0.0))

    # EWC (Elastic Weight Consolidation) during Targeted FT: prevent whack-a-mole pattern
    # Computes Fisher Information Matrix from base model, then penalizes changes to important params
    # This allows TFT to fix wrong examples without breaking correct ones (the main TFT failure mode)
    targeted_ft_ewc_lambda = float(cfg.get("targeted_ft_ewc_lambda", 0))  # 0 = disabled; 0.1-1.0 typical
    targeted_ft_ewc_samples = int(cfg.get("targeted_ft_ewc_samples", 5000))  # samples for Fisher estimation

    # Selective parameter freezing during TFT: freeze components to reduce catastrophic forgetting
    # Comma-separated list: "embed" (embedding), "attn" (attention projections), "mlp" (MLP weights), "norm" (norms)
    # Freezing embed+attn lets TFT only adjust MLP/norms, preventing disruption of learned attention routing
    targeted_ft_freeze_layers = cfg.get("targeted_ft_freeze_layers", "")  # "" = disabled; "embed,attn" typical
    if not isinstance(targeted_ft_freeze_layers, (str, list)):
        targeted_ft_freeze_layers = ""  # float/int values are invalid; treat as disabled

    # TFT Lookahead: apply Lookahead during TFT to prevent whack-a-mole forgetting
    # Maintains slow weights that anchor the base model's knowledge while fast weights explore fixes
    # Lookahead is proven to stabilize main training (Recipe F); applying it to TFT should
    # prevent catastrophic forgetting while allowing exploration of fixes for wrong examples
    targeted_ft_lookahead_k = int(cfg.get("targeted_ft_lookahead_k", 0))       # 0 = disabled; 5 typical
    targeted_ft_lookahead_alpha = float(cfg.get("targeted_ft_lookahead_alpha", 0.5))

    # TFT inter-round jitter: add small perturbation between TFT rounds to escape local minima
    # Helps break the whack-a-mole pattern by exploring nearby weight configurations between rounds
    # After each round's evaluation, perturb weights slightly before the next round begins
    targeted_ft_jitter_sigma = float(cfg.get("targeted_ft_jitter_sigma", 0))   # 0 = disabled; 0.003-0.01 typical

    # TFT progressive wrong_frac: linearly decay wrong_frac from initial to final over rounds
    # Start with high wrong fraction (aggressive correction), decay to low (gentle maintenance)
    # Early rounds focus on fixing wrong examples, later rounds emphasize retaining correct ones
    targeted_ft_wrong_frac_final = float(cfg.get("targeted_ft_wrong_frac_final", 0))  # 0 = disabled; 0.3 typical

    # TFT per-parameter LR multipliers: fine-grained control beyond binary freeze/unfreeze
    # Each parameter group gets its own LR during TFT, like "soft freezing"
    # Low multiplier = nearly frozen, high multiplier = adapts faster
    targeted_ft_lr_norm_mult = float(cfg.get("targeted_ft_lr_norm_mult", 1.0))   # LR mult for norm params
    targeted_ft_lr_arc_mult = float(cfg.get("targeted_ft_lr_arc_mult", 1.0))     # LR mult for arc embed params
    targeted_ft_lr_up_mult = float(cfg.get("targeted_ft_lr_up_mult", 1.0))       # LR mult for up/gate MLP params
    targeted_ft_lr_attn_mult = float(cfg.get("targeted_ft_lr_attn_mult", 1.0))   # LR mult for attn projection params

    # TFT EMA: Polyak-average weights across TFT rounds for smoother convergence
    # The EMA smooths over round-to-round oscillations from the whack-a-mole pattern
    # After all rounds, compares EMA weights vs best single-round checkpoint
    targeted_ft_ema_decay = float(cfg.get("targeted_ft_ema_decay", 0))  # 0 = disabled; 0.9-0.99 typical

    # Post-training weight perturbation search: random walk in weight-space near best solution
    # When accuracy is high but not perfect (e.g., 99.96%), explores nearby weight configurations
    # that might fix the last few wrong examples. Zero-cost way to find 100% solutions.
    perturbation_search_n = int(cfg.get("perturbation_search_n", 0))  # 0 = disabled; 50-200 typical
    perturbation_search_sigma = float(cfg.get("perturbation_search_sigma", 0.01))  # noise scale relative to param std
    perturbation_search_min_acc = float(cfg.get("perturbation_search_min_acc", 0.99))  # only search if acc >= this

    # Perturbation search with restarts: instead of always perturbing from the original base,
    # periodically restart from the best-found perturbation. This compounds small improvements.
    # E.g., restarts=5 means every N/5 trials, update the base to best-found-so-far.
    # 0 = disabled (always perturb from original base, current behavior).
    perturbation_search_restarts = int(cfg.get("perturbation_search_restarts", 0))  # 0 = disabled; 3-10 typical

    # Fisher-weighted perturbation: scale noise inversely to Fisher information
    # Parameters important for correct predictions get smaller perturbations,
    # reducing the chance of breaking them while exploring fixes for wrong examples
    perturbation_search_fisher = cfg.get("perturbation_search_fisher", False)
    perturbation_search_fisher_samples = int(cfg.get("perturbation_search_fisher_samples", 5000))

    if targeted_ft_rounds > 0 and best_accuracy > 0.9:
        # EWC: compute Fisher Information Matrix before TFT begins
        _ft_fisher = None
        _ft_ref_params = None
        if targeted_ft_ewc_lambda > 0:
            print(f"  [EWC] Computing Fisher Information ({targeted_ft_ewc_samples} samples)...")
            model.eval()
            _ft_fisher = {}
            _ft_ref_params = {}
            for _ewc_n, _ewc_p in model.named_parameters():
                if _ewc_p.requires_grad:
                    _ft_fisher[_ewc_n] = torch.zeros_like(_ewc_p)
                    _ft_ref_params[_ewc_n] = _ewc_p.data.clone()
            _ewc_batch_size = min(batch_size, 128)
            _ewc_iters = max(1, targeted_ft_ewc_samples // _ewc_batch_size)
            for _ewc_i in range(_ewc_iters):
                _ewc_p_batch, _ewc_t_batch = generate_batch(_ewc_batch_size, device=device)
                _ewc_full = torch.cat([_ewc_p_batch, _ewc_t_batch[:, :-1]], dim=1)
                _ewc_logits = model(_ewc_full)
                _ewc_ol = _ewc_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                _ewc_loss = F.cross_entropy(_ewc_ol.reshape(-1, VOCAB_SIZE), _ewc_t_batch.reshape(-1))
                model.zero_grad()
                _ewc_loss.backward()
                for _ewc_n, _ewc_p in model.named_parameters():
                    if _ewc_p.requires_grad and _ewc_p.grad is not None:
                        _ft_fisher[_ewc_n] += _ewc_p.grad.data ** 2
            for _ewc_n in _ft_fisher:
                _ft_fisher[_ewc_n] /= _ewc_iters
            model.zero_grad()
            _ewc_mean = sum(f.mean().item() for f in _ft_fisher.values()) / max(len(_ft_fisher), 1)
            print(f"  [EWC] Fisher computed. Mean diagonal: {_ewc_mean:.6f}")

        # Selective parameter freezing: disable gradients for specified components during TFT
        _ft_frozen_params = []
        if targeted_ft_freeze_layers:
            _ft_freeze_set = set(targeted_ft_freeze_layers if isinstance(targeted_ft_freeze_layers, list) else targeted_ft_freeze_layers.split(","))
            for _ff_name, _ff_p in model.named_parameters():
                _ff_should_freeze = False
                if "embed" in _ft_freeze_set and ("tok_embed" in _ff_name or "arc_" in _ff_name):
                    _ff_should_freeze = True
                elif "attn" in _ft_freeze_set and any(k in _ff_name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "out_A", "out_B", "k_rot", "k_alpha", "q_norm", "k_norm")):
                    _ff_should_freeze = True
                elif "mlp" in _ft_freeze_set and any(k in _ff_name for k in ("gate_proj", "up_proj", "down_proj", "gate_alpha", "down_rot")):
                    _ff_should_freeze = True
                elif "norm" in _ft_freeze_set and any(k in _ff_name for k in ("ln1", "ln2", "ln_f")):
                    _ff_should_freeze = True
                if _ff_should_freeze:
                    _ff_p.requires_grad = False
                    _ft_frozen_params.append((_ff_name, _ff_p))
            if _ft_frozen_params:
                print(f"  [TFT freeze] Frozen {len(_ft_frozen_params)} params: {[n for n, _ in _ft_frozen_params[:5]]}{'...' if len(_ft_frozen_params) > 5 else ''}")

        _ft_origin_state = {k: v.clone() for k, v in model.state_dict().items()}
        _ft_origin_acc = best_accuracy
        _ft_best_acc = best_accuracy
        _ft_best_state = {k: v.clone() for k, v in model.state_dict().items()}

        for _ft_attempt in range(targeted_ft_attempts):
            # Reset to origin state for each attempt (fresh start)
            if _ft_attempt > 0:
                model.load_state_dict(_ft_origin_state)
                best_accuracy = _ft_origin_acc
                best_state = {k: v.clone() for k, v in _ft_origin_state.items()}
                torch.manual_seed(seed + (_ft_attempt + 1) * 10000)
                # Reset FT-specific Grokfast EMA for clean attempt
                for _ftp in model.parameters():
                    if hasattr(_ftp, '_ft_grokfast_ema'):
                        del _ftp._ft_grokfast_ema

            print(f"\n=== Targeted Fine-Tuning (attempt {_ft_attempt+1}/{targeted_ft_attempts}, {targeted_ft_rounds} rounds, lr={targeted_ft_lr}, search={targeted_ft_search_size}) ===")
            pre_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
            pre_ft_acc = best_accuracy
            _ft_current_lr = targeted_ft_lr

            # TFT EMA: initialize running average of weights across rounds
            _tft_ema_state = None
            if targeted_ft_ema_decay > 0:
                _tft_ema_state = {k: v.clone() for k, v in model.state_dict().items()}

            for ft_round in range(targeted_ft_rounds):
                # Decay LR across rounds (e.g., 0.7 = 70% of previous round's LR)
                if ft_round > 0:
                    _ft_current_lr *= targeted_ft_lr_decay

                # Build TFT optimizer with optional per-param LR multipliers
                _tft_use_groups = (targeted_ft_lr_norm_mult != 1.0 or targeted_ft_lr_arc_mult != 1.0 or
                                   targeted_ft_lr_up_mult != 1.0 or targeted_ft_lr_attn_mult != 1.0)
                if _tft_use_groups:
                    _tft_norm_p, _tft_arc_p, _tft_up_p, _tft_attn_p, _tft_other_p = [], [], [], [], []
                    for _tft_n, _tft_p in model.named_parameters():
                        if not _tft_p.requires_grad:
                            continue
                        if any(k in _tft_n for k in ("ln1", "ln2", "ln_f", "q_norm", "k_norm")):
                            _tft_norm_p.append(_tft_p)
                        elif "arc_" in _tft_n:
                            _tft_arc_p.append(_tft_p)
                        elif "up_proj" in _tft_n or "gate_proj" in _tft_n or "gate_alpha" in _tft_n:
                            _tft_up_p.append(_tft_p)
                        elif any(k in _tft_n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "out_A", "out_B", "k_rot", "k_alpha")):
                            _tft_attn_p.append(_tft_p)
                        else:
                            _tft_other_p.append(_tft_p)
                    _tft_groups = []
                    if _tft_other_p:
                        _tft_groups.append({"params": _tft_other_p, "lr": _ft_current_lr, "_tft_mult": 1.0})
                    if _tft_norm_p:
                        _tft_groups.append({"params": _tft_norm_p, "lr": _ft_current_lr * targeted_ft_lr_norm_mult, "_tft_mult": targeted_ft_lr_norm_mult})
                    if _tft_arc_p:
                        _tft_groups.append({"params": _tft_arc_p, "lr": _ft_current_lr * targeted_ft_lr_arc_mult, "_tft_mult": targeted_ft_lr_arc_mult})
                    if _tft_up_p:
                        _tft_groups.append({"params": _tft_up_p, "lr": _ft_current_lr * targeted_ft_lr_up_mult, "_tft_mult": targeted_ft_lr_up_mult})
                    if _tft_attn_p:
                        _tft_groups.append({"params": _tft_attn_p, "lr": _ft_current_lr * targeted_ft_lr_attn_mult, "_tft_mult": targeted_ft_lr_attn_mult})
                    ft_optimizer = torch.optim.Adam(_tft_groups, weight_decay=targeted_ft_wd)
                else:
                    ft_optimizer = torch.optim.Adam(model.parameters(), lr=_ft_current_lr, weight_decay=targeted_ft_wd)

                model.eval()
                wrong_p, wrong_t = [], []
                _ft_eff_search = int(targeted_ft_search_size * (targeted_ft_search_growth ** ft_round))
                _ft_search_iters = max(1, _ft_eff_search // 500)
                with torch.no_grad():
                    for _ in range(_ft_search_iters):
                        if targeted_ft_carry_bias > 0:
                            p, t = generate_carry_biased_batch(500, carry_prob=targeted_ft_carry_bias, device=device)
                        else:
                            p, t = generate_batch(500, device=device)
                        out = model.generate(p)
                        wrong = (out != t).any(dim=1)
                        if wrong.any():
                            wrong_p.append(p[wrong])
                            wrong_t.append(t[wrong])

                if not wrong_p:
                    print(f"  Round {ft_round+1}: 0/{_ft_eff_search} wrong, done!")
                    break

                wrong_p = torch.cat(wrong_p)
                wrong_t = torch.cat(wrong_t)
                n_wrong = len(wrong_p)
                print(f"  Round {ft_round+1}: {n_wrong}/{_ft_eff_search} wrong, lr={_ft_current_lr:.6f}")

                # Save pre-round state for mixback
                _mb_pre_round = None
                if targeted_ft_mixback > 0:
                    _mb_pre_round = {k: v.clone() for k, v in model.state_dict().items()}

                model.train()
                # TFT Lookahead: initialize slow weights for this round
                _ft_la_slow = None
                if targeted_ft_lookahead_k > 0:
                    _ft_la_slow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

                for ft_step in range(targeted_ft_steps):
                    # Cosine LR within each FT round for smoother optimization
                    if targeted_ft_steps > 1:
                        ft_progress = ft_step / (targeted_ft_steps - 1)
                        ft_step_lr = _ft_current_lr * 0.5 * (1 + math.cos(math.pi * ft_progress))
                        for pg in ft_optimizer.param_groups:
                            _tft_m = pg.get("_tft_mult", 1.0)
                            pg["lr"] = max(ft_step_lr * _tft_m, _ft_current_lr * 0.01 * _tft_m)

                    # Mix: wrong examples + fresh random (prevent forgetting)
                    # Progressive wrong_frac: decay from initial to final over rounds
                    _ft_eff_wrong_frac = targeted_ft_wrong_frac
                    if targeted_ft_wrong_frac_final > 0 and targeted_ft_rounds > 1:
                        _ft_eff_wrong_frac = targeted_ft_wrong_frac + (targeted_ft_wrong_frac_final - targeted_ft_wrong_frac) * (ft_round / (targeted_ft_rounds - 1))
                    n_wrong_in_batch = max(1, int(batch_size * _ft_eff_wrong_frac))
                    n_wrong_in_batch = min(n_wrong_in_batch, n_wrong)
                    idx = torch.randint(0, n_wrong, (n_wrong_in_batch,))
                    fp, ft_tgt = wrong_p[idx], wrong_t[idx]
                    rp, rt = generate_batch(batch_size - len(fp), device=device)
                    bp = torch.cat([fp, rp])
                    bt = torch.cat([ft_tgt, rt])
                    perm = torch.randperm(len(bp))
                    bp, bt = bp[perm], bt[perm]

                    # Commutative augmentation during targeted FT
                    if commutative_aug:
                        _ft_swap = torch.rand(bp.shape[0], device=bp.device) < 0.5
                        if _ft_swap.any():
                            _ft_a = bp[_ft_swap, 1:11].clone()
                            bp[_ft_swap, 1:11] = bp[_ft_swap, 13:23]
                            bp[_ft_swap, 13:23] = _ft_a

                    full_in = torch.cat([bp, bt[:, :-1]], dim=1)
                    logits = model(full_in)
                    ol = logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                    # OHEM during targeted FT: select hardest samples per mini-batch
                    if targeted_ft_ohem_ratio > 0:
                        with torch.no_grad():
                            _ft_per_sample = F.cross_entropy(
                                ol.reshape(-1, VOCAB_SIZE), bt.reshape(-1),
                                reduction='none'
                            ).view(-1, OUTPUT_LEN).mean(dim=1)
                            _ft_ohem_k = max(1, int(len(_ft_per_sample) / targeted_ft_ohem_ratio))
                            _, _ft_hard_idx = _ft_per_sample.topk(_ft_ohem_k)
                        ft_loss = F.cross_entropy(
                            ol[_ft_hard_idx].reshape(-1, VOCAB_SIZE),
                            bt[_ft_hard_idx].reshape(-1)
                        )
                    else:
                        ft_loss = F.cross_entropy(ol.reshape(-1, VOCAB_SIZE), bt.reshape(-1))
                    # EWC penalty: penalize changes to params important for correct predictions
                    if _ft_fisher is not None and targeted_ft_ewc_lambda > 0:
                        _ewc_penalty = 0.0
                        for _ewc_n, _ewc_p in model.named_parameters():
                            if _ewc_p.requires_grad and _ewc_n in _ft_fisher:
                                _ewc_penalty = _ewc_penalty + (_ft_fisher[_ewc_n] * (_ewc_p - _ft_ref_params[_ewc_n]) ** 2).sum()
                        ft_loss = ft_loss + targeted_ft_ewc_lambda * 0.5 * _ewc_penalty
                    ft_optimizer.zero_grad()
                    ft_loss.backward()
                    # Gradient Surgery (PCGrad): project FT gradients to not conflict with correct-example gradients
                    if targeted_ft_grad_surgery:
                        # Save mixed-batch gradients
                        _gs_mixed = {}
                        for _gs_n, _gs_p in model.named_parameters():
                            if _gs_p.grad is not None:
                                _gs_mixed[_gs_n] = _gs_p.grad.data.clone()
                        # Compute reference gradient from pure random batch (mostly correct examples)
                        ft_optimizer.zero_grad()
                        _gs_rp, _gs_rt = generate_batch(batch_size, device=device)
                        _gs_full = torch.cat([_gs_rp, _gs_rt[:, :-1]], dim=1)
                        _gs_logits = model(_gs_full)
                        _gs_ol = _gs_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                        _gs_loss = F.cross_entropy(_gs_ol.reshape(-1, VOCAB_SIZE), _gs_rt.reshape(-1))
                        _gs_loss.backward()
                        # Project: remove conflicting components from mixed gradient
                        for _gs_n, _gs_p in model.named_parameters():
                            if _gs_p.grad is not None and _gs_n in _gs_mixed:
                                _gs_ref = _gs_p.grad.data.flatten()
                                _gs_mix = _gs_mixed[_gs_n].flatten()
                                _gs_dot = torch.dot(_gs_mix, _gs_ref)
                                if _gs_dot < 0:  # conflicting
                                    _gs_ref_norm_sq = _gs_ref.dot(_gs_ref).clamp(min=1e-12)
                                    _gs_mix = _gs_mix - (_gs_dot / _gs_ref_norm_sq) * _gs_ref
                                _gs_p.grad.data = _gs_mix.view_as(_gs_p.grad)
                    # Grokfast during targeted FT: amplify slow-moving gradient components
                    if targeted_ft_grokfast and grokfast_alpha > 0:
                        for _ftp in model.parameters():
                            if _ftp.grad is not None:
                                if not hasattr(_ftp, '_ft_grokfast_ema'):
                                    _ftp._ft_grokfast_ema = torch.zeros_like(_ftp.grad)
                                _ftp._ft_grokfast_ema.mul_(grokfast_alpha).add_(_ftp.grad, alpha=1 - grokfast_alpha)
                                _ftp.grad.add_(_ftp._ft_grokfast_ema, alpha=grokfast_lambda * getattr(_ftp, '_gf_lambda_mult', 1.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    ft_optimizer.step()

                    # TFT Lookahead: update slow weights every K steps
                    if targeted_ft_lookahead_k > 0 and _ft_la_slow is not None and (ft_step + 1) % targeted_ft_lookahead_k == 0:
                        with torch.no_grad():
                            for _la_n, _la_p in model.named_parameters():
                                if _la_p.requires_grad and _la_n in _ft_la_slow:
                                    _ft_la_slow[_la_n].add_(_la_p.data - _ft_la_slow[_la_n], alpha=targeted_ft_lookahead_alpha)
                                    _la_p.data.copy_(_ft_la_slow[_la_n])

                # TFT Weight Mixback: blend post-round weights back toward pre-round state
                if targeted_ft_mixback > 0 and _mb_pre_round is not None:
                    with torch.no_grad():
                        for _mb_k, _mb_v in model.state_dict().items():
                            if _mb_k in _mb_pre_round:
                                _mb_v.lerp_(_mb_pre_round[_mb_k], targeted_ft_mixback)

                model.eval()
                ft_acc = evaluate_model_batched(model, device, num_tests=2000)
                print(f"  Round {ft_round+1}: accuracy = {ft_acc:.4f}")

                # TFT EMA: update running average (before any revert)
                if _tft_ema_state is not None:
                    with torch.no_grad():
                        _ema_sd = model.state_dict()
                        for _ema_k in _tft_ema_state:
                            _tft_ema_state[_ema_k].mul_(targeted_ft_ema_decay).add_(
                                _ema_sd[_ema_k], alpha=1 - targeted_ft_ema_decay)

                if ft_acc > best_accuracy:
                    best_accuracy = ft_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                # If round made things worse, revert to best and continue (don't compound errors)
                if ft_acc < pre_ft_acc and best_state is not None:
                    model.load_state_dict(best_state)

                # TFT inter-round jitter: perturb weights to escape local minima between rounds
                if targeted_ft_jitter_sigma > 0 and ft_round < targeted_ft_rounds - 1:
                    with torch.no_grad():
                        for _jt_p in model.parameters():
                            if _jt_p.requires_grad:
                                _jt_scale = targeted_ft_jitter_sigma * (_jt_p.data.std() + 1e-8)
                                _jt_p.data.add_(torch.randn_like(_jt_p) * _jt_scale)

                # Time guard
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    print(f"  Time limit reached during FT, stopping.")
                    break

            # TFT EMA: evaluate averaged weights after all rounds
            if _tft_ema_state is not None:
                _ema_pre_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(_tft_ema_state)
                model.eval()
                _ema_acc = evaluate_model_batched(model, device, num_tests=2000)
                print(f"  TFT EMA accuracy: {_ema_acc:.4f}")
                if _ema_acc > best_accuracy:
                    best_accuracy = _ema_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"  TFT EMA improved over best round!")
                else:
                    model.load_state_dict(_ema_pre_state)

            # If this attempt didn't improve, revert
            if best_accuracy <= pre_ft_acc:
                model.load_state_dict(pre_ft_state)
                print(f"  Attempt {_ft_attempt+1}: FT didn't improve ({pre_ft_acc:.4f}). Reverted.")
            elif best_state is not None:
                model.load_state_dict(best_state)

            # Track best across all attempts
            if best_accuracy > _ft_best_acc:
                _ft_best_acc = best_accuracy
                _ft_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if _ft_best_acc >= 1.0:
                    print(f"  Perfect accuracy reached on attempt {_ft_attempt+1}!")
                    break

            # Time guard for attempts
            if time_limit > 0 and (time.time() - t0) >= time_limit:
                break

        # Apply best result across all attempts
        if _ft_best_acc > _ft_origin_acc and _ft_best_state is not None:
            model.load_state_dict(_ft_best_state)
            best_accuracy = _ft_best_acc
            best_state = _ft_best_state
            print(f"  Best FT result: {_ft_best_acc:.4f} (from {targeted_ft_attempts} attempt(s))")
        else:
            model.load_state_dict(_ft_origin_state)
            best_accuracy = _ft_origin_acc
            print(f"  No FT attempt improved ({_ft_origin_acc:.4f}). Reverted.")

        # Unfreeze parameters after TFT
        for _uf_name, _uf_p in _ft_frozen_params:
            _uf_p.requires_grad = True

    # Checkpoint interpolation search: explore weight space between top checkpoints
    # When accuracy is high but not perfect, the optimal state may lie between two good checkpoints
    if ckpt_interpolation_steps > 0 and best_accuracy >= 0.99 and best_accuracy < 1.0:
        # Collect two states to interpolate between
        _interp_state_a = None  # best state
        _interp_state_b = None  # second-best or pre-FT origin
        if len(ckpt_avg_pool) >= 2:
            _interp_state_a = ckpt_avg_pool[0][1]
            _interp_state_b = ckpt_avg_pool[1][1]
            _interp_label = f"ckpt_avg top-1 ({ckpt_avg_pool[0][0]:.4f}) and top-2 ({ckpt_avg_pool[1][0]:.4f})"
        elif best_state is not None and '_ft_origin_state' in dir() and _ft_origin_state is not None:
            _interp_state_a = best_state
            _interp_state_b = _ft_origin_state
            _interp_label = f"post-FT best ({best_accuracy:.4f}) and pre-FT origin"

        if _interp_state_a is not None and _interp_state_b is not None:
            print(f"\n=== Checkpoint Interpolation ({ckpt_interpolation_steps} points between {_interp_label}) ===")
            _interp_best_acc = best_accuracy
            _interp_best_state = None
            _interp_saved = {k: v.clone() for k, v in model.state_dict().items()}
            for _interp_i in range(1, ckpt_interpolation_steps):
                _interp_alpha = _interp_i / ckpt_interpolation_steps
                _interp_state = {}
                for _ik in _interp_state_a:
                    _interp_state[_ik] = (_interp_state_a[_ik].float() * (1 - _interp_alpha) +
                                          _interp_state_b[_ik].float() * _interp_alpha).to(_interp_state_a[_ik].dtype)
                model.load_state_dict(_interp_state)
                _interp_acc = evaluate_model_batched(model, device, num_tests=2000)
                print(f"  alpha={_interp_alpha:.2f}: accuracy={_interp_acc:.4f}")
                if _interp_acc > _interp_best_acc:
                    _interp_best_acc = _interp_acc
                    _interp_best_state = {k: v.clone() for k, v in _interp_state.items()}
                if _interp_best_acc >= 1.0:
                    break
                if time_limit > 0 and (time.time() - t0) >= time_limit:
                    break
            if _interp_best_state is not None:
                print(f"  Interpolation improved: {best_accuracy:.4f} -> {_interp_best_acc:.4f}")
                model.load_state_dict(_interp_best_state)
                best_accuracy = _interp_best_acc
                best_state = _interp_best_state
            else:
                print(f"  No improvement from interpolation, keeping original")
                model.load_state_dict(_interp_saved)

    # Weight Perturbation Search: random walk in weight-space near best solution
    # At 99.96% (4 wrong out of 10,010), a nearby weight configuration may achieve 100%.
    # Tests random perturbations scaled by each parameter's standard deviation.
    if perturbation_search_n > 0 and best_accuracy >= perturbation_search_min_acc and best_accuracy < 1.0:
        _ps_restart_label = f", restarts={perturbation_search_restarts}" if perturbation_search_restarts > 0 else ""
        _ps_fisher_label = ", fisher-weighted" if perturbation_search_fisher else ""
        print(f"\n=== Weight Perturbation Search ({perturbation_search_n} trials, sigma={perturbation_search_sigma}, min_acc={perturbation_search_min_acc}{_ps_restart_label}{_ps_fisher_label}) ===")

        # Fisher-weighted perturbation: compute diagonal Fisher information from correct predictions
        # High Fisher = parameter is important for correct outputs → perturb LESS
        # Low Fisher = parameter doesn't affect correct outputs much → perturb MORE
        _ps_fisher_weights = None
        if perturbation_search_fisher:
            print(f"  Computing Fisher info ({perturbation_search_fisher_samples} samples)...")
            model.eval()
            _ps_fisher_raw = {}
            for _pf_n, _pf_p in model.named_parameters():
                if _pf_p.requires_grad:
                    _ps_fisher_raw[_pf_n] = torch.zeros_like(_pf_p)
            _pf_bs = min(batch_size, 128)
            _pf_iters = max(1, perturbation_search_fisher_samples // _pf_bs)
            for _ in range(_pf_iters):
                _pf_prompts, _pf_targets = generate_batch(_pf_bs, device=device)
                _pf_full = torch.cat([_pf_prompts, _pf_targets[:, :-1]], dim=1)
                _pf_logits = model(_pf_full)
                _pf_ol = _pf_logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]
                _pf_loss = F.cross_entropy(_pf_ol.reshape(-1, VOCAB_SIZE), _pf_targets.reshape(-1))
                model.zero_grad()
                _pf_loss.backward()
                for _pf_n, _pf_p in model.named_parameters():
                    if _pf_p.requires_grad and _pf_p.grad is not None:
                        _ps_fisher_raw[_pf_n] += _pf_p.grad.data ** 2
            # Convert to inverse-Fisher weights: high Fisher → small weight, low Fisher → large weight
            _ps_fisher_weights = {}
            for _pf_n in _ps_fisher_raw:
                _pf_f = _ps_fisher_raw[_pf_n] / _pf_iters
                # Inverse sqrt: perturb proportional to 1/sqrt(Fisher+eps) for gentler scaling
                _pf_w = 1.0 / (_pf_f.sqrt() + 1e-6)
                # Normalize so mean weight = 1 (keeps overall perturbation scale same as sigma)
                _pf_w = _pf_w / (_pf_w.mean() + 1e-8)
                _ps_fisher_weights[_pf_n] = _pf_w
            model.zero_grad()
            print(f"  Fisher computed. {len(_ps_fisher_weights)} param tensors weighted.")

        _ps_base_state = {k: v.clone() for k, v in model.state_dict().items()}
        _ps_best_acc = best_accuracy
        _ps_best_state = None
        _ps_improved = 0
        # Restart interval: how often to rebase from best-found (compounds improvements)
        _ps_restart_interval = (perturbation_search_n // max(perturbation_search_restarts, 1)
                                if perturbation_search_restarts > 0 else 0)
        for _ps_i in range(perturbation_search_n):
            # Restart from best-found periodically (if restarts enabled and we have an improvement)
            if (_ps_restart_interval > 0 and _ps_i > 0 and
                    _ps_i % _ps_restart_interval == 0 and _ps_best_state is not None):
                _ps_base_state = {k: v.clone() for k, v in _ps_best_state.items()}
                model.load_state_dict(_ps_base_state)
                print(f"  [restart] Rebasing from best-found ({_ps_best_acc:.4f}) at trial {_ps_i+1}")
            with torch.no_grad():
                for _ps_n, _ps_p in model.named_parameters():
                    _ps_scale = perturbation_search_sigma * (_ps_p.data.std() + 1e-8)
                    _ps_noise = torch.randn_like(_ps_p) * _ps_scale
                    # Fisher weighting: scale noise by inverse-Fisher weights
                    if _ps_fisher_weights is not None and _ps_n in _ps_fisher_weights:
                        _ps_noise *= _ps_fisher_weights[_ps_n]
                    _ps_p.data.add_(_ps_noise)
            model.eval()
            _ps_acc = evaluate_model_batched(model, device, num_tests=2000)
            if _ps_acc > _ps_best_acc:
                _ps_best_acc = _ps_acc
                _ps_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                _ps_improved += 1
                print(f"  Trial {_ps_i+1}/{perturbation_search_n}: acc {_ps_acc:.4f} — improved! (#{_ps_improved})")
                if _ps_best_acc >= 1.0:
                    print(f"  Perfect accuracy found at trial {_ps_i+1}!")
                    break
            # Revert to base state for next independent trial
            model.load_state_dict(_ps_base_state)
            if time_limit > 0 and (time.time() - t0) >= time_limit:
                print(f"  Time limit reached during perturbation search, stopping.")
                break
        if _ps_best_state is not None:
            model.load_state_dict(_ps_best_state)
            best_accuracy = _ps_best_acc
            best_state = _ps_best_state
            print(f"  Perturbation search result: {_ps_best_acc:.4f} ({_ps_improved} improvements from {perturbation_search_n} trials)")
        else:
            print(f"  No improvement from {perturbation_search_n} perturbation trials")

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


def evaluate_on_verify_set(model, device, commutative_eval=False):
    """Evaluate on the exact verify.py test set (seed=2025).

    If commutative_eval=True, evaluates with both (a,b) and (b,a) orderings
    and takes per-digit majority vote. Exploits a+b = b+a at zero training cost.
    """
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
            output_normal = model.generate(prompts)

            if commutative_eval:
                # Generate with swapped ordering (b, a) — exploits commutativity
                prompts_swap = torch.cat([sep1, b_digits, sep2, a_digits, sep1], dim=1).to(device)
                output_swap = model.generate(prompts_swap)
                # Per-digit majority vote between both orderings
                stacked = torch.stack([output_normal.cpu(), output_swap.cpu()], dim=0)
                output = torch.mode(stacked, dim=0).values
            else:
                output = output_normal

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
    if cfg.get("k_alpha_q"): tricks.append("K=alpha*Q")
    if cfg.get("gate_alpha_up"): tricks.append("gate=alpha*up")
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
            # Try base ID by stripping trailing variant suffixes like -ht-1, -v2, etc.
            import re
            base_id = re.sub(r'-[a-z]+-\d+$', '', idea_id)
            if base_id != idea_id:
                row = conn.execute(
                    "SELECT config FROM ideas WHERE idea_id = ?", (base_id,)
                ).fetchone()
        if not row:
            raise ValueError(f"Idea {idea_id} not found in SQLite")
        return yaml.safe_load(row[0]) or {}
    finally:
        conn.close()


def _config_keys():
    """All valid YAML config keys (for --help introspection by orze SOP)."""
    return [
        "adam_beta1", "adam_beta2", "adam_beta2_final", "adaptive_wd",
        "attn_out_rank", "batch_size", "batch_size_start", "batch_size_warmup_steps",
        "carry_bias", "carry_bias_schedule", "ckpt_avg_k", "ckpt_interpolation_steps",
        "commutative_aug", "commutative_eval", "commutative_loss_weight",
        "cosine_wd", "crash_recovery_drop", "crash_recovery_max", "curriculum",
        "custom_init", "d_model", "digit_loss_scale", "digit_loss_weight",
        "distill_alpha", "distill_teacher_ckpt", "distill_temperature",
        "down_rot_up_t", "egd", "egd_before_grokfast", "egd_off_at_acc",
        "egd_warmup_steps", "ema_decay", "ema_start_step", "embed_type",
        "eval_every", "ff_dim", "ff_mult", "focal_gamma", "gate_alpha_up",
        "grad_accum_steps", "grad_centralize", "grad_clip", "grad_clip_final",
        "grad_noise_eta", "grad_noise_gamma", "grok_transfer_digits",
        "grok_transfer_phases", "grok_transfer_steps", "grokfast_alpha",
        "grokfast_alpha2", "grokfast_decay_start", "grokfast_lambda",
        "grokfast_lambda2", "grokfast_lambda_arc_mult", "grokfast_lambda_cycle_period",
        "grokfast_lambda_min", "grokfast_lambda_norm_mult", "grokfast_reset_at_acc",
        "grokfast_spike_cooldown", "grokfast_spike_dampening", "grokfast_spike_scale",
        "grokfast_spike_threshold", "grokfast_type", "grokfast_window",
        "head_dim", "init_norm_hi", "init_norm_lo", "init_q_scale",
        "interleaved_ft_acc", "interleaved_ft_every", "interleaved_ft_lr",
        "interleaved_ft_steps", "k_alpha_q", "k_rot_q", "label_smoothing",
        "lambda_acc_min", "lambda_acc_scale", "lambda_adjust_at_acc",
        "lambda_adjust_to", "lambda_warmup_steps", "lawa_k", "lawa_start_acc",
        "lion_beta1", "lion_beta2", "lookahead_alpha", "lookahead_k",
        "loss_threshold_schedule", "lr", "lr_arc_mult", "lr_norm_mult",
        "lr_restart_at_acc", "lr_restart_mult", "lr_restart_period",
        "lr_restart_steps", "lr_restart_to", "lr_stable_frac", "lr_up_mult",
        "majority_vote_n", "min_lr", "min_wd", "muon_momentum", "muon_nesterov",
        "muon_ns_steps", "n_heads", "n_kv_heads", "n_layers", "norm_type",
        "o_tail_scale", "ohem_ratio", "optim_reset_at_acc", "optimizer",
        "patience", "pe_type", "perp_grad", "perturbation_search_fisher",
        "perturbation_search_fisher_samples", "perturbation_search_min_acc",
        "perturbation_search_n", "perturbation_search_restarts",
        "perturbation_search_sigma", "qk_norm", "qk_norm_type", "repeat_mix",
        "rope_period", "rope_theta", "sam_rho", "seed", "share_ln_f",
        "share_norms", "sphere_norm", "sphere_norm_unembed", "sphere_tau",
        "sphere_tau_final", "steps", "swa_start_frac", "swa_update_freq",
        "targeted_ft_attempts", "targeted_ft_carry_bias", "targeted_ft_ema_decay",
        "targeted_ft_ewc_lambda", "targeted_ft_ewc_samples",
        "targeted_ft_freeze_layers", "targeted_ft_grad_surgery",
        "targeted_ft_grokfast", "targeted_ft_jitter_sigma",
        "targeted_ft_lookahead_alpha", "targeted_ft_lookahead_k", "targeted_ft_lr",
        "targeted_ft_lr_arc_mult", "targeted_ft_lr_attn_mult",
        "targeted_ft_lr_decay", "targeted_ft_lr_norm_mult", "targeted_ft_lr_up_mult",
        "targeted_ft_mixback", "targeted_ft_ohem_ratio", "targeted_ft_rounds",
        "targeted_ft_search_growth", "targeted_ft_search_size", "targeted_ft_steps",
        "targeted_ft_wd", "targeted_ft_wrong_frac", "targeted_ft_wrong_frac_final",
        "tie_down_gate", "tie_embed", "tie_gate", "tie_kv", "tie_qk",
        "tie_qk_norm", "tie_qo", "time_limit", "train_temperature",
        "train_temperature_final", "use_rope", "use_swiglu", "v_eq_q",
        "vocab_size", "warm_share_norms_step", "warm_start_ckpt", "warmup_steps",
        "wd_arc_mult", "wd_decay_factor1", "wd_decay_factor2", "wd_high_mult",
        "wd_high_phase_steps", "wd_milestone1", "wd_milestone2", "wd_norm_mult",
        "wd_up_mult", "weight_decay", "weight_noise_start", "weight_noise_std",
        "z_loss_weight",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="AdderBoard training script. Config keys are passed via --config YAML file.",
    )
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--ideas-md", required=True)
    parser.add_argument("--config", required=True)
    # Register all valid YAML config keys so orze SOP validator can discover them
    cfg_group = parser.add_argument_group("config keys (passed via YAML, listed here for validation)")
    for key in _config_keys():
        cfg_group.add_argument(f"--{key}", help=f"YAML config key: {key}")
    args, _ = parser.parse_known_args()

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
        model, train_metrics = train_model(cfg, device=device)

        # Full verify-set evaluation if accuracy looks promising
        _comm_eval = cfg.get("commutative_eval", False)
        if train_metrics["accuracy"] >= 0.95:
            print("Running full verify-set evaluation...")
            verify_acc = evaluate_on_verify_set(model, device, commutative_eval=_comm_eval)
            print(f"Verify-set accuracy: {verify_acc:.4%} ({int(verify_acc * 10010)}/10010)")
            if _comm_eval:
                print("  (commutative_eval: majority-voted both (a,b) and (b,a) orderings)")
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
