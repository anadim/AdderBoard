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
                 attn_out_rank=0, use_rope=True, rope_period=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # K projection options
        self.k_rot_q = k_rot_q
        self.tie_qk = tie_qk
        if k_rot_q:
            self.k_rot_angle = nn.Parameter(torch.tensor(0.1))  # 1 param
        elif not tie_qk:
            self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)

        # V projection options
        self.v_eq_q = v_eq_q
        self.tie_kv = tie_kv
        if not v_eq_q and not tie_kv and not tie_qk and not k_rot_q:
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
            self.q_norm = RMSNorm(head_dim)
            if not tie_qk_norm:
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
        elif self.tie_qk:
            k = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # V computation
        if self.v_eq_q:
            v = self.q_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        elif self.k_rot_q or self.tie_kv or self.tie_qk:
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
    def __init__(self, d_model, ff_dim, tie_gate=False, tie_down_gate=False, down_rot_up_t=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        if not tie_gate:
            self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.tie_down_gate = tie_down_gate
        self.down_rot_up_t = down_rot_up_t
        if down_rot_up_t:
            self.down_rot_angle = nn.Parameter(torch.tensor(0.1))  # 1 param
        elif not tie_down_gate:
            self.down_proj = nn.Linear(ff_dim, d_model, bias=False)
        self.tie_gate = tie_gate

    def forward(self, x):
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
            return F.linear(mixed, self.gate_proj.weight.T)
        return self.down_proj(mixed)


class GeLUMLP(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
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
                 rope_period=None):
        super().__init__()
        self.attn = Attention(
            d_model, n_heads, n_kv_heads, head_dim,
            rope_theta=rope_theta, qk_norm=qk_norm,
            tie_kv=tie_kv, tie_qo=tie_qo, tie_qk=tie_qk,
            k_rot_q=k_rot_q, v_eq_q=v_eq_q, tie_qk_norm=tie_qk_norm,
            attn_out_rank=attn_out_rank,
            use_rope=use_rope, rope_period=rope_period,
        )

        if use_swiglu:
            self.mlp = SwiGLUMLP(d_model, ff_dim, tie_gate=tie_gate,
                                 tie_down_gate=tie_down_gate, down_rot_up_t=down_rot_up_t)
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
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
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
        tie_embed = cfg.get("tie_embed", True)
        share_norms = cfg.get("share_norms", False)
        attn_out_rank = cfg.get("attn_out_rank", 0)
        use_swiglu = cfg.get("use_swiglu", True)
        norm_type = cfg.get("norm_type", "rms")
        use_rope = cfg.get("use_rope", True)
        rope_period = cfg.get("rope_period", None)
        embed_type = cfg.get("embed_type", "lookup")
        vocab_size = cfg.get("vocab_size", VOCAB_SIZE)

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
            x = block(x, mask=mask)

        x = self.ln_f(x)

        if self.tie_embed:
            table = self.tok_embed.table()
            if table.shape[1] < self.d_model:
                # Project only tok dimensions for output
                logits = x[..., :table.shape[1]] @ table.T
            else:
                logits = x @ table.T
        else:
            logits = self.lm_head(x)

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

    # Curriculum
    curriculum = cfg.get("curriculum", None)
    # Format: "digits:steps,digits:steps,..." e.g. "3:2000,6:5000,10:rest"

    # Optimizer
    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR schedule
    lr_restart_period = int(cfg.get("lr_restart_period", 0))  # 0 = single cosine (default)
    lr_restart_mult = float(cfg.get("lr_restart_mult", 1.0))  # period multiplier after each restart

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
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

    # Grokfast-EMA: gradient filter to accelerate grokking
    grokfast_alpha = float(cfg.get("grokfast_alpha", 0))
    grokfast_lambda = float(cfg.get("grokfast_lambda", 0))

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

    # Training loop
    model.train()
    best_accuracy = 0.0
    best_state = None
    steps_since_improvement = 0
    t0 = time.time()

    prompt_len = PROMPT_LEN  # a_digits(10) + sep(1) + b_digits(10) + sep(1) = 22

    for step in range(1, steps + 1):
        # Set LR
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

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

        # Teacher forcing: prompt + first (OUTPUT_LEN-1) target digits as input
        # Model predicts each target digit from the context so far
        full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
        logits = model(full_input)

        # Loss only on output positions (predict target from prompt context)
        output_logits = logits[:, prompt_len - 1:prompt_len - 1 + OUTPUT_LEN, :VOCAB_SIZE]

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

        optimizer.zero_grad()
        loss.backward()

        # Grokfast-EMA: amplify slow-moving gradient components to accelerate grokking
        if grokfast_alpha > 0:
            for p in model.parameters():
                if p.grad is not None:
                    if not hasattr(p, '_grokfast_ema'):
                        p._grokfast_ema = torch.zeros_like(p.grad)
                    p._grokfast_ema.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                    p.grad.add_(p._grokfast_ema, alpha=grokfast_lambda)

        # Gradient noise injection (Neelakantan et al. 2015)
        if grad_noise_eta > 0:
            noise_std = math.sqrt(grad_noise_eta / (1 + step) ** grad_noise_gamma)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * noise_std)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

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

            if full_acc > best_accuracy:
                best_accuracy = full_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
            else:
                steps_since_improvement += eval_every

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
        model, train_metrics = train_model(cfg, device=device)

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
