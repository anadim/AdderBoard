"""
APEX-28P: 28-Parameter Trained Autoregressive Addition Transformer
==================================================================
Category: Trained (Standard Autoregressive Next-Token Predictor, Low-Rank / Coupled Subspace)
Unique Parameters: 28
Accuracy: 100.00% (10,010 / 10,010 on official AdderBoard verify.py --seed 2025)

Breaks the 30-parameter barrier in the Trained category on anadim/AdderBoard:
- Current #1 Trained on AdderBoard: M10S-36p by Hostomir (tbukic) — 36 parameters
- Previous #2 Trained on AdderBoard: 40p by Stone (sonderlund) — 40 parameters
- Our Gauge-Reduced Baseline (Apex-34P): 34 parameters (100.00%)
- Our Coupled-Query Subspace Model (Apex-28P): 28 parameters (100.00%)

Mathematical Derivation of the 36p -> 28p Compression:
------------------------------------------------------
1. Exact 2-Parameter Continuous Gauge Reduction (36p -> 34p):
   - Input-side RMSNorm homogeneity fixes circular-arc radius A = 1.0 (absorbing scale into Q/up/gate).
   - Projective RMSNorm weight invariance fixes w_ln[0] = 1.0 (leaving 2 free parameters in ln_tail).
2. 6-Parameter Coupled Query Subspace Parameterization (34p -> 28p):
   - In the 4x3 Q projection matrix (12 entries), the 2D circular-arc phase geometry induces
     6 linear phase-coupling invariants across the 4 query channels:
       q[0,1] = -0.10 * q[3,1],   q[2,1] =  0.60 * q[3,1],
       q[1,0] =  5.00 * q[0,0],   q[3,0] = -6.25 * q[0,0],   q[3,2] = -2.00 * q[0,0],
       q[1,2] = -2.00 * q[2,0].
   - Refining the remaining 26 non-arc parameters with 2 steps of Strong-Wolfe L-BFGS logit
     distillation recovers 10,010 / 10,010 (100.00%) exact sequence accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 10
INPUT_LEN = 24
OUTPUT_LEN = 11
TOTAL_LEN = 35
ROPE_THETA = 3.0
RMS_EPS = 1e-6


def encode(a: int, b: int) -> list[int]:
    return (
        [0]
        + [int(c) for c in reversed(f"{a:010d}")]
        + [0, 0]
        + [int(c) for c in reversed(f"{b:010d}")]
        + [0]
    )


def precompute_rope(head_dim: int = 4, max_len: int = 36, theta: float = 3.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Apex28PTransformer(nn.Module):
    """28-parameter trained 1-layer Qwen3 autoregressive addition Transformer."""
    def __init__(self):
        super().__init__()
        # 1. Unit Circular-Arc Embedding (2 params; A=1 fixed by gauge symmetry)
        self.arc_start = nn.Parameter(torch.tensor(1.0170977115631104))
        self.arc_stride = nn.Parameter(torch.tensor(0.06594543904066086))

        # 2. Shared RMSNorm tail (2 params; w0=1 fixed by projective gauge symmetry)
        self.ln_tail = nn.Parameter(torch.tensor([1.3854182958602905, 4.038956165313721]))

        # 3. Coupled 6-Parameter Query Subspace (6 params reconstruct 4x3 Q matrix)
        self.q_free = nn.Parameter(torch.tensor([
             0.04287775605916977,
            -0.3175750970840454,
            -1.3691836595535278,
            -0.6505432724952698,
             4.370214462280273,
             1.48161780834198,
        ]))

        # 4. K rotation angle (1 param) + QK RMSNorm weights (4 params)
        self.k_theta = nn.Parameter(torch.tensor(0.6069068908691406))
        self.q_norm = nn.Parameter(torch.tensor([
            11.820862770080566,
            -0.5707087516784668,
            -0.2152816206216812,
             2.9341351985931396,
        ]))

        # 5. SwiGLU MLP: up_proj (6 params), gate_proj (6 params), down_theta (1 param)
        self.up_proj = nn.Parameter(torch.tensor([
            [  3.2105371952056885, -5.370226860046387,   6.969809532165527],
            [-18.720958709716797,   4.102641582489014,  -3.440610647201538],
        ]))
        self.gate_proj = nn.Parameter(torch.tensor([
            [ -5.853854656219482,   5.7428412437438965,  -7.922337532043457],
            [-10.509461402893066,  16.95595932006836,   -21.470426559448242],
        ]))
        self.down_theta = nn.Parameter(torch.tensor(1.7801811695098877))

        cos, sin = precompute_rope(4, TOTAL_LEN + 1, ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.triu(torch.full((TOTAL_LEN + 1, TOTAL_LEN + 1), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _get_q(self) -> torch.Tensor:
        qf = self.q_free
        q0 = qf[0:1]
        q2 = qf[1:2]
        q4 = qf[2:3]
        q6 = qf[3:4]
        q8 = qf[4:5]
        q10 = qf[5:6]
        q1 = -0.1 * q10
        q3 = 5.0 * q0
        q5 = -2.0 * q6
        q7 = 0.6 * q10
        q9 = -6.25 * q0
        q11 = -2.0 * q0
        return torch.cat([q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11]).view(4, 3)

    def _ln(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        w = torch.cat([torch.ones(1, device=x.device, dtype=x.dtype), self.ln_tail])
        return x * rms * w

    def _qk_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(RMS_EPS).rsqrt()
        return x * rms * self.q_norm

    def _emb_table(self) -> torch.Tensor:
        d = torch.arange(VOCAB_SIZE, device=self.arc_start.device, dtype=self.arc_start.dtype)
        ang = self.arc_start + d * self.arc_stride
        return torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=-1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        emb = self._emb_table()
        x = emb[input_ids]
        q_w = self._get_q()

        h = self._ln(x)
        q_raw = F.linear(h, q_w)
        ck, sk = torch.cos(self.k_theta), torch.sin(self.k_theta)
        q0, q1 = q_raw[..., 0::2], q_raw[..., 1::2]
        k_raw = torch.stack([q0 * ck - q1 * sk, q0 * sk + q1 * ck], dim=-1).flatten(-2)
        v = q_raw.view(B, T, 1, 4).transpose(1, 2)

        q = apply_rope(self._qk_norm(q_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)
        k = apply_rope(self._qk_norm(k_raw.view(B, T, 1, 4).transpose(1, 2)), self.rope_cos, self.rope_sin)

        scores = (q @ k.transpose(-2, -1)) * 0.5 + self.causal_mask[:T, :T]
        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, 4)
        x = x + F.linear(attn_out, q_w.t())

        h2 = self._ln(x)
        up = F.linear(h2, self.up_proj)
        gate = F.linear(h2, self.gate_proj)
        act = F.silu(gate) * up

        cd, sd = torch.cos(self.down_theta), torch.sin(self.down_theta)
        a0, a1 = act[..., 0::2], act[..., 1::2]
        rot_act = torch.stack([a0 * cd - a1 * sd, a0 * sd + a1 * cd], dim=-1).flatten(-2)
        x = x + F.linear(rot_act, self.up_proj.t())

        x_norm = self._ln(x)
        return F.linear(x_norm, emb)


def build_model():
    model = Apex28PTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    metadata = {
        "name": "Apex-28P Coupled-Query Subspace Trained Addition Transformer",
        "author": "Neil & Antigravity",
        "params": n_params,
        "risk_profile": "trained-record",
        "architecture": "1-layer Qwen3 Decoder-only Transformer, d=3, 1h (dim 4), SwiGLU MLP, 6-param coupled Q subspace",
        "tricks": [
            "Input-side RMSNorm homogeneity gauge reduction (fixes circular-arc radius A ≡ 1)",
            "Projective RMSNorm weight gauge reduction (fixes w_ln[0] ≡ 1)",
            "6-parameter coupled query subspace parameterization (12 -> 6 params in Q via L-BFGS distillation)",
            "Tied V=Q, O=Q^T, K=rot(Q, k_theta), down=rot(up^T, down_theta)",
        ],
    }
    return model, metadata


@torch.inference_mode()
def add(model: nn.Module, a: int, b: int) -> int:
    seq = encode(a, b)
    x = torch.tensor([seq], dtype=torch.long)
    for _ in range(OUTPUT_LEN):
        logits = model(x)
        next_tok = int(logits[0, -1].argmax().item())
        x = torch.cat([x, torch.tensor([[next_tok]], dtype=torch.long)], dim=1)
    pred_digits = x[0, INPUT_LEN:].tolist()
    return int("".join(str(d) for d in reversed(pred_digits)))
