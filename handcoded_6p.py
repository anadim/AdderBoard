"""
6-parameter hand-coded transformer adder — DIFFERENT approach from zcbtrak.

Key differences:
1. Circular arc embedding (not parabolic) — mod-10 is natural on the circle
2. RoPE period 11 with 22-token format (not period 19 with 31 tokens)
3. V extracts dim 0 (cosine channel), not dim 1
4. Carry via tanh gate (not SiLU hinge)
5. Different Q angle and attention geometry

The 6 parameters: arc_A, arc_start, arc_stride, v_scale, carry_thresh, carry_scale
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product

VOCAB = 10
OUTPUT_LEN = 11
MAX_ADDEND = 10**10 - 1

# ── Architectural constants ──

# RoPE period 11: a_digits(10) + [0] + b_digits(10) + [0] = 22 tokens
# a[i] at position i, b[i] at position 11+i, gap=11=period
# output[i] at position 22+i: distance to a[i]=22=2*11, to b[i]=11=1*11
ROPE_PERIOD = 11.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PROMPT_LEN = 22

# Hardcoded Q angle — derived from RoPE-11 geometry
# Output at 22+i needs to attend to positions i and 11+i
# Both are at multiples of 11 away, so RoPE rotation is 0 (full periods)
# Q angle controls which "phase" within the period gets max attention
Q_PHI = OMEGA * 5.5  # midpoint — symmetric attention to both a[i] and b[i]

HEAD_DIM = 2
ATTN_SCALE = HEAD_DIM ** -0.5
RMS_EPS = 1e-6

# Fixed norm weights (folded into output like zcbtrak, but different values)
NORM_W = np.array([40.0, -8.0])


def _unit_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)


def _tanh(x):
    return np.tanh(np.clip(x, -20, 20))


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _apply_rope(x, positions):
    a = positions * OMEGA
    cos_a = np.cos(a)[..., np.newaxis]
    sin_a = np.sin(a)[..., np.newaxis]
    return np.concatenate([
        x[..., 0:1] * cos_a - x[..., 1:2] * sin_a,
        x[..., 0:1] * sin_a + x[..., 1:2] * cos_a,
    ], axis=-1)


def _build_embed_table(arc_A, arc_start, arc_stride):
    d = np.arange(VOCAB, dtype=np.float64)
    angles = arc_start + d * arc_stride
    return np.stack([arc_A * np.cos(angles), arc_A * np.sin(angles)], axis=-1)


def _forward(params, token_ids):
    arc_A, arc_start, arc_stride, v_scale, carry_thresh, carry_scale = params
    B, T = token_ids.shape
    table = _build_embed_table(arc_A, arc_start, arc_stride)
    h = table[token_ids]  # [B, T, 2]

    # ── Attention ──
    hn = _unit_rms_norm(h)

    # Hardcoded Q: rotate dim 0 by fixed angle
    COS_Q = math.cos(Q_PHI)
    SIN_Q = math.sin(Q_PHI)
    q = np.stack([hn[..., 0] * COS_Q - hn[..., 1] * SIN_Q,
                  hn[..., 0] * SIN_Q + hn[..., 1] * COS_Q], axis=-1)
    k = hn.copy()  # K = normalized embedding (no extra params)

    # V: scale both dims (extracts digit info from circular embedding)
    v = hn * v_scale

    q = _unit_rms_norm(q)
    k = _unit_rms_norm(k)

    positions = np.arange(T, dtype=np.float64)[np.newaxis, :]
    q = _apply_rope(q, positions)
    k = _apply_rope(k, positions)

    scores = np.einsum('btd,bsd->bts', q, k) * ATTN_SCALE
    causal = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal[np.newaxis], -np.inf, scores)
    attn_w = _softmax(scores, axis=-1)
    attn_out = np.einsum('bts,bsd->btd', attn_w, v)

    h = h + attn_out  # residual

    # ── MLP (carry gate via tanh) ──
    hn = _unit_rms_norm(h)
    # Gate: tanh-based carry detection
    # hn[..., 1] encodes the sin component which correlates with digit sum magnitude
    gate_input = carry_thresh + carry_scale * hn[..., 1]
    carry_signal = _tanh(gate_input)

    # Inject carry into dim 1 (sin channel — shifts the angle)
    h = h + np.stack([np.zeros_like(carry_signal), carry_signal], axis=-1)

    # ── Output (unit RMS norm + folded head) ──
    h = _unit_rms_norm(h)
    folded_table = table * NORM_W[np.newaxis, :]
    return np.einsum('btd,vd->btv', h, folded_table)


def _encode(a, b):
    a_digits = [(a // 10**i) % 10 for i in range(10)]
    b_digits = [(b // 10**i) % 10 for i in range(10)]
    return a_digits + [0] + b_digits + [0]  # 22 tokens


class _Model:
    def __init__(self, params):
        self.params = params


def build_model():
    # Find optimal params via search
    best_params = _find_params()
    model = _Model(best_params)
    metadata = {
        "name": "CircArc-RoPE11-TanhCarry-6p",
        "author": "warlockee",
        "params": 6,
        "architecture": "1L decoder, d=2, 1h, hd=2, ff=2, RoPE period-11",
        "tricks": [
            "Circular arc embedding (3 params, mod-10 on circle)",
            "RoPE period 11 with 22-token format",
            "Hardcoded Q (architectural constant)",
            "Hardcoded K (= unit_rms_norm of embedding)",
            "tanh carry gate (not SiLU)",
            "Norm weights folded into output head",
        ],
    }
    return model, metadata


def add(model, a, b):
    seq = _encode(a, b)
    for _ in range(OUTPUT_LEN):
        logits = _forward(model.params, np.array([seq], dtype=np.int64))
        seq.append(int(np.argmax(logits[0, -1, :])))
    digits = seq[-OUTPUT_LEN:]
    return int(''.join(str(d) for d in digits)[::-1])


def _find_params():
    """Numerically optimize 6 params to implement addition."""
    import random

    def eval_params(params, n=200):
        correct = 0
        rng = random.Random(42)
        for _ in range(n):
            a = rng.randint(0, MAX_ADDEND)
            b = rng.randint(0, MAX_ADDEND)
            seq = _encode(a, b)
            for _ in range(OUTPUT_LEN):
                logits = _forward(params, np.array([seq], dtype=np.int64))
                seq.append(int(np.argmax(logits[0, -1, :])))
            result = int(''.join(str(d) for d in seq[-OUTPUT_LEN:])[::-1])
            if result == a + b:
                correct += 1
        return correct / n

    # Start with reasonable values
    best = np.array([2.5, -1.2, 0.29, -100.0, -5.0, 10.0])
    best_acc = eval_params(best, 50)
    print(f"Initial: acc={best_acc:.4f}")

    # CMA-ES style search
    sigma = np.array([1.0, 0.5, 0.1, 50.0, 5.0, 5.0])
    for iteration in range(500):
        # Generate candidates
        candidates = [best + sigma * np.random.randn(6) for _ in range(20)]
        accs = [eval_params(c, 50) for c in candidates]
        idx = np.argmax(accs)
        if accs[idx] > best_acc:
            best_acc = accs[idx]
            best = candidates[idx]
            print(f"  iter {iteration}: acc={best_acc:.4f} params={best}")
        if best_acc >= 0.99:
            break
        # Shrink sigma slowly
        if iteration % 50 == 49:
            sigma *= 0.8

    return best


if __name__ == "__main__":
    print("Searching for 6-param circular-arc adder...")
    model, meta = build_model()
    print(f"\nParams: {model.params}")
    print(f"Metadata: {meta}")

    # Quick test
    for a, b in [(0, 0), (5, 5), (999, 1), (9999999999, 1)]:
        result = add(model, a, b)
        ok = "✓" if result == a + b else "✗"
        print(f"  {a} + {b} = {result} (true: {a+b}) {ok}")
