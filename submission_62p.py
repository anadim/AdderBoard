"""
AdderBoard Submission: 62-parameter trained transformer for 10-digit addition.

Architecture: 1-layer Qwen3-style decoder
  - d_model=3, head_dim=4, ff_dim=2, SwiGLU
  - Circular arc embedding (3 params), tied K=V, tied O=Q^T
  - Learnable QK norms, RoPE theta=3.0

Training: 5-phase pipeline (cosine warmup → constant LR+EMA → lower LR+EMA →
  Adam no-WD → targeted fine-tuning on error cases)

Accuracy: 99.9% on 10K random pairs (seed=2025 verification)
Parameters: 62 unique learnable parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Model Components ───

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=3.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, offset=0):
        T = x.shape[-2]
        t = torch.arange(offset, offset + T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos_f, sin_f = freqs.cos(), freqs.sin()
        D = x.shape[-1]
        rot_dim = (D // 2) * 2
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        half = rot_dim // 2
        cos_f, sin_f = cos_f[..., :half], sin_f[..., :half]
        rotated = torch.stack([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1).flatten(-2)
        return torch.cat([rotated, x_pass], dim=-1) if x_pass.shape[-1] > 0 else rotated


class CircularArcEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
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


class Attention(nn.Module):
    def __init__(self, d_model=3, head_dim=4):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)  # V = K (tied)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RotaryEmbedding(head_dim, theta=3.0)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = k.clone()  # tie K=V
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q), self.rope(k)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return F.linear(out, self.q_proj.weight.T)  # tie O=Q^T


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model=3, ff_dim=2):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ─── Full Model ───

class AdderModel(nn.Module):
    def __init__(self, d_model=3, head_dim=4, ff_dim=2, vocab_size=10):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = CircularArcEmbedding(vocab_size)
        # Sinusoidal PE for dim 2 (tok_embed gives 2 dims, PE fills dim 3)
        pe = torch.zeros(64, 1)
        pos = torch.arange(64).unsqueeze(1).float()
        pe[:, 0] = torch.sin(pos[:, 0] * math.exp(-math.log(10000.0)))
        self.register_buffer("pos_embed", pe)

        self.ln1 = RMSNorm(d_model)
        self.attn = Attention(d_model, head_dim)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, ff_dim)
        self.ln_f = RMSNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)  # [B, T, 2]
        pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)  # [B, T, 1]
        x = torch.cat([tok, pos], dim=-1)  # [B, T, 3]

        mask = torch.zeros(T, T, device=x.device)
        mask.masked_fill_(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)

        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        x = self.ln_f(x)

        table = self.tok_embed.table()  # [10, 2]
        logits = x[..., :2] @ table.T  # project tok dims only
        return logits

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        seq = prompt.clone()
        for _ in range(11):  # OUTPUT_LEN = 11
            logits = self.forward(seq)
            next_tok = logits[:, -1, :10].argmax(dim=-1)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        return seq[:, prompt.shape[1]:prompt.shape[1] + 11]


# ─── Weights ───

WEIGHTS = {
    "tok_embed.arc_A": 25.02317237854004,
    "tok_embed.arc_start": -0.2547111511230469,
    "tok_embed.arc_stride": 0.06311280280351639,
    "attn.q_proj.weight": [[-0.604081392288208, -0.0002486599260009825, 21.054105758666992], [-0.21068155765533447, 0.01719030924141407, 0.13735458254814148], [0.9734416604042053, -0.1098935529589653, 0.041659124195575714], [0.05693720281124115, -2.3206427097320557, -0.47510960698127747]],
    "attn.k_proj.weight": [[-0.06312127411365509, 1.7858084440231323, -30.205820083618164], [4.746553421020508, 1.104575753211975, -70.7226791381836], [3.434957981109619, 0.7511223554611206, -24.515010833740234], [-0.13498139381408691, -0.016353625804185867, 20.890283584594727]],
    "attn.q_norm.weight": [0.604255735874176, 2.9982070922851562, 4.795403003692627, 0.0538223460316658],
    "attn.k_norm.weight": [-0.01454188022762537, 6.3072919845581055, 1.9737937450408936, 2.3363115787506104],
    "mlp.gate_proj.weight": [[0.7186843156814575, -0.4613227844238281, 0.5894570350646973], [0.7117799520492554, 0.45831581950187683, -0.5933173894882202]],
    "mlp.up_proj.weight": [[-1.5344116687774658, -1.4468005895614624, 1.2638859748840332], [1.5183025598526, -1.3507441282272339, 1.1220263242721558]],
    "mlp.down_proj.weight": [[0.8895362019538879, -0.8233017325401306], [2.2569541931152344, 2.299072504043579], [1.345754861831665, 1.3772836923599243]],
    "ln1.weight": [7.530346870422363, 3.2177560329437256, -0.021621771156787872],
    "ln2.weight": [3.155890941619873, 3.9469993114471436, 7.59224796295166],
    "ln_f.weight": [107.53959655761719, 28.793907165527344, 1.3171551472623833e-05],
}


def load_model(device="cpu"):
    model = AdderModel().to(device)
    state = {}
    for k, v in WEIGHTS.items():
        if isinstance(v, (int, float)):
            state[k] = torch.tensor(v, dtype=torch.float32)
        else:
            state[k] = torch.tensor(v, dtype=torch.float32)
    model.load_state_dict(state, strict=False)
    return model


def encode(a, b):
    """Encode two integers as LSB-first token sequence."""
    a_digits = [(a // 10**i) % 10 for i in range(10)]
    b_digits = [(b // 10**i) % 10 for i in range(10)]
    return [0] + a_digits + [0, 0] + b_digits + [0]  # 24 tokens


def predict(model, a, b, device="cpu"):
    """Predict a + b using the model."""
    prompt = encode(a, b)
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    output = model.generate(prompt_t)
    result = sum(tok * (10**i) for i, tok in enumerate(output[0].tolist()[:11]))
    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    # Count params
    n = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n}")

    # Quick test
    tests = [(0, 0), (5, 5), (999, 1), (9999999999, 1),
             (9999999999, 9999999999), (1234567890, 9876543210)]
    correct = 0
    for a, b in tests:
        pred = predict(model, a, b, device)
        ok = "✓" if pred == a + b else "✗"
        print(f"  {a} + {b} = {pred} (true: {a+b}) {ok}")
        correct += pred == a + b
    print(f"\nQuick test: {correct}/{len(tests)}")
