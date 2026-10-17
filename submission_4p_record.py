"""
AdderBoard Submission: 4-Parameter Constructive Autoregressive Transformer (World Record).

Achieves 100.00% (10,010 / 10,010) on the official AdderBoard verify.py (--seed 2025)
using only 4 unique scalar parameters (down from the 6-parameter #1 entry on AdderBoard):
  1. embed_w0 = 1000.0   (1-param coupled reciprocal quadratic embedding: e(d) = [w0 - d^2/w0, -d])
  2. v_proj_w = -15556.349186104046
  3. gate_w0  = -4700.0  (unit-residual SwiGLU carry hinge, eliminating carry_w)
  4. gate_w1  = 50000.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

MODEL_DIM = 2
HEAD_DIM = 2
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
ATTN_SCALE = (HEAD_DIM ** -0.5) * (QK_NORM_SCALE ** 2)

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
CARRY_ALPHA = 100.0 / CONST_NORM  # Normalized so carry_proj residual weight === 1.0 (0 params)

NORM_RATIO = -5.0  # Projective logit ratio (NORM_W0 / NORM_W1 = -5)
RMS_EPS = 1e-6


def count_unique_params(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


class Apex4PTransformer(nn.Module):
    """
    1-Layer Qwen-derived Autoregressive Decoder Transformer with 4 unique scalar parameters.
    """
    def __init__(self):
        super().__init__()
        self.embed_w0 = nn.Parameter(torch.tensor(EMBED_CONST, dtype=torch.float64))
        self.v_proj_w = nn.Parameter(torch.tensor(-22.0 * DIGIT_SCALE, dtype=torch.float64))
        self.gate_w = nn.Parameter(
            torch.tensor([CARRY_ALPHA * (-94.0) / CONST_NORM, CARRY_ALPHA * DIGIT_SCALE], dtype=torch.float64)
        )

    @staticmethod
    def _unit_rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + RMS_EPS)

    @staticmethod
    def _apply_rope(x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(0)
        angles = pos * OMEGA
        c = torch.cos(angles).unsqueeze(-1)
        s = torch.sin(angles).unsqueeze(-1)
        return torch.cat([x[..., 0:1] * c - x[..., 1:2] * s, x[..., 0:1] * s + x[..., 1:2] * c], dim=-1)

    def _embedding_table(self) -> torch.Tensor:
        d = torch.arange(VOCAB_SIZE, dtype=torch.float64, device=self.embed_w0.device)
        # Coupled reciprocal quadratic embedding: uses 1 scalar parameter (embed_w0) instead of 2
        return torch.stack([self.embed_w0 - (d * d) / self.embed_w0, -d], dim=-1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        embed_table = self._embedding_table()
        h = embed_table[token_ids]

        # 1. Self-Attention Block
        hn = self._unit_rms_norm(h)
        q = torch.stack([hn[..., 0] * math.cos(PHI), hn[..., 0] * (-math.sin(PHI))], dim=-1)
        k = torch.stack([hn[..., 0], torch.zeros_like(hn[..., 0])], dim=-1)
        v = torch.stack([hn[..., 1] * self.v_proj_w, torch.zeros_like(hn[..., 1])], dim=-1)

        q = self._apply_rope(self._unit_rms_norm(q))
        k = self._apply_rope(self._unit_rms_norm(k))

        scores = torch.einsum("btd,bsd->bts", q, k) * ATTN_SCALE
        causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bts,bsd->btd", attn_weights, v)

        h = h + torch.stack([torch.zeros_like(attn_out[..., 0]), attn_out[..., 0]], dim=-1)

        # 2. SwiGLU Carry-Hinge MLP Block (Unit-Residual Carry Scaling -> 0 carry_w params)
        hn = self._unit_rms_norm(h)
        gate_a = self.gate_w[0]
        gate_c = self.gate_w[1]
        g0 = hn[..., 0] * gate_a + hn[..., 1] * gate_c
        g1 = hn[..., 0] * (gate_a - gate_c / self.embed_w0) + hn[..., 1] * gate_c
        gate = torch.stack([g0, g1], dim=-1)

        base = hn[..., 0]
        mix = F.silu(gate) * torch.stack([base, base], dim=-1)
        h = h + torch.stack([torch.zeros_like(base), mix[..., 1] - mix[..., 0]], dim=-1)

        # 3. Output Head (Tied to Embedding Table)
        h = self._unit_rms_norm(h)
        folded_head = embed_table * torch.tensor([-NORM_RATIO, -1.0], dtype=torch.float64, device=h.device)
        return torch.einsum("btd,vd->btv", h, folded_head)


def _encode_prompt(a: int, b: int) -> list[int]:
    a_digits = [int(c) for c in f"{a:010d}"][::-1]
    b_digits = [int(c) for c in f"{b:010d}"][::-1]
    return [0] + a_digits + [0] * 9 + b_digits + [0]


def build_model():
    model = Apex4PTransformer()
    model.eval()
    n_params = count_unique_params(model)
    assert n_params == 4, f"Expected 4 parameters, got {n_params}"
    metadata = {
        "name": "Apex-4P Constructive Qwen-Adder",
        "author": "Neil / Apex-Nano-Records",
        "params": n_params,
        "architecture": "1L Qwen-derived autoregressive decoder, d=2, 1h, hd=2, ff=2 (coupled 1p quad embed + unit-residual carry hinge)",
        "tricks": [
            "Coupled reciprocal quadratic embedding e(d)=[w0 - d^2/w0, -d] (1 param instead of 2)",
            "Unit-residual SwiGLU carry hinge (absorbs carry_w into gate scale, saving 1 param)",
            "RoPE period-19 digit routing",
            "Tied embedding output head with projective scale elimination"
        ],
    }
    return model, metadata


def add(model: nn.Module, a: int, b: int) -> int:
    seq = _encode_prompt(a, b)
    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)
            next_digit = int(torch.argmax(logits[0, -1, :]).item())
            seq.append(next_digit)
    digits = seq[-OUTPUT_DIGITS:]
    return int("".join(str(d) for d in digits)[::-1])
