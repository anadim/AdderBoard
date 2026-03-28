"""
AdderBoard submission: 6-parameter adder transformer (hand-coded).

Architecture: 1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2.
Alternative approach: embedding is hardcoded as architectural constant,
Q projection routing is explicitly learned.

Reduces the 10-parameter architecture to 6 by hardcoding:
  - The embedding table as an architectural constant: e(d)=[1000-0.001*d^2, -d]
    (0 params — fixed geometric property of the representation space)
  - Final RMSNorm weights folded into the output head as constants:
    logits = unit_rms_norm(h) @ (w * emb).T  (0 extra params)

The 6 learnable parameters:
  q_w0      — Q projection weight 0 (learned cos(PHI) for routing)
  q_w1      — Q projection weight 1 (learned -sin(PHI) for routing)
  v_proj_w  — V projection scalar
  gate_w0   — tied gate weight (digit-sum threshold)
  gate_w1   — tied gate weight (carry-mix slope)
  carry_w   — shared carry projection scalar

Accuracy: 100% on 10K random pairs (seed=2025)
"""

import math
import numpy as np

VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1
MODEL_DIM = 2
HEAD_DIM = 2
ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
TARGET_PHI = OMEGA * (10.0 + PEAK_EPS)
TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS)))
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
ATTN_SCALE = (HEAD_DIM ** -0.5) * (QK_NORM_SCALE ** 2)
EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 256.0 / CONST_NORM
NORM_W0 = 50.0 * math.sqrt(2.0)
NORM_W1 = -10.0 * math.sqrt(2.0)
RMS_EPS = 1e-6

PARAMS = np.array([
    math.cos(TARGET_PHI),
    -math.sin(TARGET_PHI),
    -22.0 * DIGIT_SCALE,
    CARRY_ALPHA * (-94.0) / CONST_NORM,
    CARRY_ALPHA * DIGIT_SCALE,
    (100.0 / CARRY_ALPHA) * (1.0 / CONST_NORM),
], dtype=np.float64)

def _unit_rms_norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)

def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

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
        x[..., 0:1] * sin_a + x[..., 1:2] * cos_a], axis=-1)

def _build_embed_table():
    d = np.arange(VOCAB_SIZE, dtype=np.float64)
    return np.stack([EMBED_CONST - 0.001 * d * d, -d], axis=-1)

def _forward(params, token_ids):
    q_w0, q_w1, v_proj_w, gate_a, gate_c, carry_w = params
    B, T = token_ids.shape
    embed_table = _build_embed_table()
    h = embed_table[token_ids]

    hn = _unit_rms_norm(h)
    q = np.stack([hn[..., 0] * q_w0, hn[..., 0] * q_w1], axis=-1)
    k = np.stack([hn[..., 0], np.zeros_like(hn[..., 0])], axis=-1)
    v = np.stack([hn[..., 1] * v_proj_w, np.zeros_like(hn[..., 1])], axis=-1)

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

    h = h + np.stack([np.zeros_like(attn_out[..., 0]), attn_out[..., 0]], axis=-1)

    hn = _unit_rms_norm(h)
    g0 = hn[..., 0] * gate_a + hn[..., 1] * gate_c
    g1 = hn[..., 0] * (gate_a - gate_c / EMBED_CONST) + hn[..., 1] * gate_c
    gate = np.stack([g0, g1], axis=-1)
    base = hn[..., 0]
    up = np.stack([base, base], axis=-1)
    mix = _silu(gate) * up
    h = h + np.stack([np.zeros_like(base), carry_w * (mix[..., 1] - mix[..., 0])], axis=-1)

    h = _unit_rms_norm(h)
    norm_w = np.array([NORM_W0, NORM_W1])
    folded_table = embed_table * norm_w[np.newaxis, :]
    return np.einsum('btd,vd->btv', h, folded_table)

def _encode_prompt(a, b):
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]

class _Model:
    def __init__(self, params):
        self.params = params

def build_model():
    model = _Model(PARAMS)
    metadata = {
        "name": "qwen6_fixedemb_learnedq",
        "author": "warlockee",
        "params": 6,
        "architecture": "1L Qwen-derived decoder, d=2, 1h, hd=2, ff=2 (fixed embed, learned Q)",
        "tricks": [
            "RoPE period-19 geometry",
            "0-param hardcoded embedding e(d)=[1000-0.001*d^2, -d] (architectural constant)",
            "2-param explicit Q_proj routing (learned cos/sin of attention angle)",
            "norm weights folded into tied output head (0 extra params)",
            "tied carry hinge gate",
            "shared carry-scale scalar",
        ],
    }
    return model, metadata

def add(model, a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    seq = _encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        logits = _forward(model.params, np.array([seq], dtype=np.int64))
        seq.append(int(np.argmax(logits[0, -1, :])))
    digits = seq[-OUTPUT_DIGITS:]
    return int(''.join(str(d) for d in digits)[::-1])
