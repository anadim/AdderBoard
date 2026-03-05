"""
8-parameter Qwen-derived adder (PyTorch).

Parameters (8 total):
  embed_tokens.weight   1   scalar c  →  e(d) = [c - d²/c,  -d]
  norm_weight           2   final RMSNorm γ
  q_proj.weight         1   angle φ   →  Q = [x₀·cos φ,  -x₀·sin φ]
  v_proj.weight         1   scalar
  gate_proj.weight      2   tied gate family
  carry_proj.weight     1   shared carry scalar
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── architecture constants ──────────────────────────────────────────
MODEL_DIM = 2
HEAD_DIM = 2
INTERMEDIATE_SIZE = 2
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1

EMBED_CONST = 1000.0
CONST_NORM = math.sqrt(MODEL_DIM)
DIGIT_SCALE = EMBED_CONST / CONST_NORM
DECODE_CURVATURE = 0.1

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))
CARRY_ALPHA = 256.0 / CONST_NORM


# ── helpers ─────────────────────────────────────────────────────────
def _unit_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    """RoPE with angular frequency ω = 2π/19 on head_dim=2 vectors.
    x: (batch, heads, seq, 2)"""
    seq_len = x.shape[2]
    pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    theta = pos * OMEGA
    cos_t = torch.cos(theta).view(1, 1, -1, 1)
    sin_t = torch.sin(theta).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * cos_t - x1 * sin_t,
                      x0 * sin_t + x1 * cos_t], dim=-1)


# ── structured projections ──────────────────────────────────────────
class QProj(nn.Module):
    """Phase-tied: 1 angle → cos/sin pair."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        phi = self.weight[0]
        return torch.stack([x[..., 0] * torch.cos(phi),
                            x[..., 0] * (-torch.sin(phi))], dim=-1)


class KProj(nn.Module):
    """No parameters."""
    def forward(self, x):
        return torch.stack([x[..., 0], torch.zeros_like(x[..., 0])], dim=-1)


class VProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.stack([x[..., 1] * self.weight[0],
                            torch.zeros_like(x[..., 0])], dim=-1)


class OProj(nn.Module):
    """No parameters: channel swap."""
    def forward(self, x):
        return torch.stack([torch.zeros_like(x[..., 0]), x[..., 0]], dim=-1)


# ── attention ───────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = (HEAD_DIM ** -0.5) * QK_NORM_SCALE ** 2
        self.q_proj = QProj()
        self.k_proj = KProj()
        self.v_proj = VProj()
        self.o_proj = OProj()

    def forward(self, x, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)

        q = _apply_rope(_unit_rms_norm(q))
        k = _apply_rope(_unit_rms_norm(k))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))


# ── MLP ─────────────────────────────────────────────────────────────
class TiedGateProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        a, c = self.weight[0], self.weight[1]
        g0 = x[..., 0] * a + x[..., 1] * c
        g1 = x[..., 0] * (a - c / EMBED_CONST) + x[..., 1] * c
        return torch.stack([g0, g1], dim=-1)


class SharedCarryProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, gate, base):
        up = base.unsqueeze(-1).expand(*base.shape, INTERMEDIATE_SIZE)
        mix = F.silu(gate) * up
        y1 = self.weight[0] * (mix[..., 1] - mix[..., 0])
        return torch.stack([torch.zeros_like(y1), y1], dim=-1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = TiedGateProj()
        self.carry_proj = SharedCarryProj()

    def forward(self, x):
        return self.carry_proj(self.gate_proj(x), x[..., 0])


# ── transformer block ──────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()

    def forward(self, x, mask):
        h = x + self.self_attn(_unit_rms_norm(x), mask)
        return h + self.mlp(_unit_rms_norm(h))


# ── embedding ───────────────────────────────────────────────────────
class TiedEmbedding(nn.Module):
    """1 param c  →  e(d) = [c - d²/c,  -d]."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def table(self):
        d = torch.arange(VOCAB_SIZE, device=self.weight.device, dtype=torch.float32)
        c = self.weight[0]
        return torch.stack([c - (d * d) / c, -d], dim=-1)

    def forward(self, tokens):
        return self.table()[tokens]

    def as_linear(self, x):
        return x @ self.table().T


# ── full model ──────────────────────────────────────────────────────
class AdderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = TiedEmbedding()
        self.block = Block()
        self.norm_weight = nn.Parameter(torch.zeros(MODEL_DIM))

    def _final_norm(self, x):
        return _unit_rms_norm(x) * self.norm_weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(tokens)
        L = h.shape[1]
        mask = torch.triu(
            torch.full((L, L), -1e9, device=h.device, dtype=h.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        h = self.block(h, mask)
        return self.embed_tokens.as_linear(self._final_norm(h))


# ── weight initialisation ──────────────────────────────────────────
def _init_weights(model: AdderModel) -> None:
    c = EMBED_CONST
    digit_scale = c / CONST_NORM
    carry_alpha = 256.0 / CONST_NORM

    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.tensor([c]))

        model.norm_weight.copy_(torch.tensor([
            (DECODE_CURVATURE * c) / CONST_NORM,
            -digit_scale / 50.0,
        ]))

        model.block.self_attn.q_proj.weight.copy_(torch.tensor([PHI]))
        model.block.self_attn.v_proj.weight.copy_(torch.tensor([-22.0 * digit_scale]))

        model.block.mlp.gate_proj.weight.copy_(torch.tensor([
            carry_alpha * (-94.0) / CONST_NORM,
            carry_alpha * digit_scale,
        ]))

        model.block.mlp.carry_proj.weight.copy_(torch.tensor([
            (100.0 / carry_alpha) / CONST_NORM,
        ]))


# ── inference ───────────────────────────────────────────────────────
def _encode_prompt(a: int, b: int) -> list[int]:
    ad = [int(ch) for ch in f"{a:010d}"][::-1]
    bd = [int(ch) for ch in f"{b:010d}"][::-1]
    return [0] + ad + [0] * 9 + bd + [0]


@torch.no_grad()
def generate(model: AdderModel, a: int, b: int) -> str:
    model.eval()
    dev = next(model.parameters()).device
    seq = _encode_prompt(a, b)
    for _ in range(OUTPUT_DIGITS):
        x = torch.tensor([seq], dtype=torch.long, device=dev)
        logits = model(x)
        seq.append(int(logits[0, -1].argmax().item()))
    return "".join(str(d) for d in seq[-OUTPUT_DIGITS:])


def add(model: AdderModel, a: int, b: int) -> int:
    if not (isinstance(a, int) and isinstance(b, int)):
        raise ValueError("a and b must be ints")
    if not (0 <= a <= MAX_ADDEND and 0 <= b <= MAX_ADDEND):
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    return int(generate(model, a, b)[::-1])


def build_model():
    model = AdderModel()
    _init_weights(model)
    metadata = {
        "name": "adder",
        "author": "kswain98",
        "params": sum(p.numel() for p in model.parameters()),
        "architecture": "8 parameter",
        "tricks": [
            "RoPE period-19 geometry",
            "phase-tied Q projection (2 → 1 param)",
            "tied embedding (2 → 1 param)",
            "tied carry hinge gate",
            "shared carry-scale scalar",
        ],
    }
    return model, metadata


# ── test harness ────────────────────────────────────────────────────
if __name__ == "__main__":
    model, metadata = build_model()
    n = metadata["params"]
    print(f"Parameters: {n}")
    for name, p in model.named_parameters():
        print(f"  {name:40s}  {p.numel()}")
    print()

    cases = [
        (0, 0),
        (9999999999, 1),
        (9999999999, 9999999999),
        (5555555555, 4444444445),
        (1000000000, 9000000000),
        (1111111111, 8888888889),
        (999999999, 1),
        (1234567890, 9876543210),
        (5000000000, 5000000000),
    ]
    print("Edge cases:")
    all_ok = True
    for a, b in cases:
        r = add(model, a, b)
        ok = "✓" if r == a + b else "✗"
        if r != a + b:
            all_ok = False
        print(f"  {ok}  {a} + {b} = {r}  (expected {a+b})")
    print()

    N = 10000
    print(f"Random test ({N} samples)...")
    t0 = time.time()
    correct = 0
    for _ in range(N):
        a = random.randint(0, MAX_ADDEND)
        b = random.randint(0, MAX_ADDEND)
        if add(model, a, b) == a + b:
            correct += 1
    elapsed = time.time() - t0
    print(f"  {correct}/{N} ({100*correct/N:.1f}%) in {elapsed:.1f}s")
