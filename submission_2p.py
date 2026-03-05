"""
2-parameter Qwen-style adder (PyTorch).

Two independent parameters control the entire model:
  c   (1)  embedding scale
  g   (1)  carry detection threshold

Everything else is derived or hardcoded:
  v       = -22·c/√2            (attention value strength, proportional to c)
  norm[0] = 0.1·c/√2            (decode curvature, from c)
  norm[1] = -c/(50·√2)          (digit scale for output, from c)
  gate[1] = 128·c               (digit-proportional gate, from c)
  s       = 100/256             (carry pathway constant)
  φ       = (2π/19)·10.3        (RoPE positional encoding constant)
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

CONST_NORM = math.sqrt(MODEL_DIM)

ROPE_PERIOD = 19.0
OMEGA = 2.0 * math.pi / ROPE_PERIOD
PEAK_EPS = 0.3
PHI = OMEGA * (10.0 + PEAK_EPS)

TARGET_LOGIT_GAP = math.log(10.0)
ATTN_AMPLITUDE = TARGET_LOGIT_GAP / (
    math.cos(OMEGA * PEAK_EPS) - math.cos(OMEGA * (1.0 - PEAK_EPS))
)
QK_NORM_SCALE = math.sqrt(ATTN_AMPLITUDE / math.sqrt(2.0))

V_FACTOR = -22.0 / CONST_NORM
S_CONST = 100.0 / 256.0


# ── helpers ─────────────────────────────────────────────────────────
def _unit_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _apply_rope(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[2]
    pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    theta = pos * OMEGA
    cos_t = torch.cos(theta).view(1, 1, -1, 1)
    sin_t = torch.sin(theta).view(1, 1, -1, 1)
    x0, x1 = x[..., 0:1], x[..., 1:2]
    return torch.cat([x0 * cos_t - x1 * sin_t,
                      x0 * sin_t + x1 * cos_t], dim=-1)


# ── full model (2 parameters) ──────────────────────────────────────
class AdderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1))      # embedding scale
        self.g = nn.Parameter(torch.zeros(1))       # carry detection threshold

        self.attn_scale = (HEAD_DIM ** -0.5) * QK_NORM_SCALE ** 2

    # ── derived quantities ──────────────────────────────────────────
    def _embed_table(self):
        d = torch.arange(VOCAB_SIZE, device=self.c.device, dtype=torch.float32)
        c = self.c[0]
        return torch.stack([c - (d * d) / c, -d], dim=-1)

    def _v_weight(self):
        return V_FACTOR * self.c[0]

    def _norm_weight(self):
        c = self.c[0]
        return torch.stack([0.1 * c / CONST_NORM, -c / (50.0 * CONST_NORM)])

    def _gate_weight(self):
        return torch.stack([self.g[0], 128.0 * self.c[0]])

    # ── projections ─────────────────────────────────────────────────
    def _q_proj(self, x):
        return torch.stack([x[..., 0] * math.cos(PHI),
                            x[..., 0] * (-math.sin(PHI))], dim=-1)

    def _k_proj(self, x):
        return torch.stack([x[..., 0], torch.zeros_like(x[..., 0])], dim=-1)

    def _v_proj(self, x):
        return torch.stack([x[..., 1] * self._v_weight(),
                            torch.zeros_like(x[..., 0])], dim=-1)

    def _o_proj(self, x):
        return torch.stack([torch.zeros_like(x[..., 0]), x[..., 0]], dim=-1)

    # ── attention ───────────────────────────────────────────────────
    def _attention(self, x, mask):
        B, L, _ = x.shape
        q = self._q_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)
        k = self._k_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)
        v = self._v_proj(x).reshape(B, L, 1, HEAD_DIM).transpose(1, 2)

        q = _apply_rope(_unit_rms_norm(q))
        k = _apply_rope(_unit_rms_norm(k))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self._o_proj(out.transpose(1, 2).reshape(B, L, -1))

    # ── MLP ─────────────────────────────────────────────────────────
    def _mlp(self, x):
        gw = self._gate_weight()
        a, gc = gw[0], gw[1]
        g0 = x[..., 0] * a + x[..., 1] * gc
        g1 = x[..., 0] * (a - gc / self.c[0]) + x[..., 1] * gc
        gate = torch.stack([g0, g1], dim=-1)

        base = x[..., 0]
        up = base.unsqueeze(-1).expand(*base.shape, INTERMEDIATE_SIZE)
        mix = F.silu(gate) * up
        y1 = S_CONST * (mix[..., 1] - mix[..., 0])
        return torch.stack([torch.zeros_like(y1), y1], dim=-1)

    # ── forward ─────────────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tab = self._embed_table()
        h = tab[tokens]
        L = h.shape[1]
        mask = torch.triu(
            torch.full((L, L), -1e9, device=h.device, dtype=h.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        hn = _unit_rms_norm(h)
        h = h + self._attention(hn, mask)

        hn = _unit_rms_norm(h)
        h = h + self._mlp(hn)

        nw = self._norm_weight()
        out = _unit_rms_norm(h) * nw
        return out @ tab.T


# ── weight initialisation ──────────────────────────────────────────
def _init_weights(model: AdderModel) -> None:
    with torch.no_grad():
        model.c.copy_(torch.tensor([1000.0]))
        model.g.copy_(torch.tensor([256.0 * (-94.0) / (CONST_NORM * CONST_NORM)]))


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


def add(model, a: int, b: int) -> int:
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
        "architecture": "2 parameter",
        "tricks": [
            "RoPE period-19 positional encoding (hardcoded)",
            "tied embedding (single scalar c → full vocab table)",
            "derived norm weights from c",
            "derived attention value strength from c",
            "derived gate component from c",
            "hardcoded Q angle φ (positional constant)",
            "hardcoded carry pathway amplitude",
        ],
    }
    return model, metadata


# ── test harness ────────────────────────────────────────────────────
if __name__ == "__main__":
    model, metadata = build_model()
    n = metadata["params"]
    print(f"Parameters: {n}")
    for name, p in model.named_parameters():
        print(f"  {name:40s}  {p.numel()}  = {p.data.item():.6f}")
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
    for a, b in cases:
        r = add(model, a, b)
        ok = "✓" if r == a + b else "✗"
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
