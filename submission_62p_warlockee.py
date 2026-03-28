"""
AdderBoard Submission: 62-parameter trained transformer for 10-digit addition.

Architecture: 1-layer Qwen3-style decoder (d=3, hd=4, ff=2)
  - Circular arc embedding (3 params), tied K=V, tied O=Q^T
  - Learnable QK norms, RoPE theta=3.0, SwiGLU

Training: 5-phase pipeline with curriculum learning.
  Phase 0: Cosine LR=0.01->0.001, AdamW WD=0.01, 200K steps, curriculum 3->6->10 digits
  Phase 1: Constant LR=0.001 + EMA=0.999, 50K steps
  Phase 2: Constant LR=0.0003 + EMA=0.999, 30K steps
  Phase 3: Adam (no WD), cosine LR=0.001->0, 50K steps
  Phase 4: Targeted fine-tuning on error cases

Accuracy: 100% on verify set (10K random + 10 edge, seed=2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        rotated = torch.stack([
            x1 * cos_f - x2 * sin_f,
            x1 * sin_f + x2 * cos_f,
        ], dim=-1).flatten(-2)
        return torch.cat([rotated, x_pass], dim=-1) if x_pass.shape[-1] > 0 else rotated


class CircularArcEmbedding(nn.Module):
    def __init__(self, vocab_size=10):
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


class AdderModel(nn.Module):
    def __init__(self, d_model=3, head_dim=4, ff_dim=2, vocab_size=10):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.tok_embed = CircularArcEmbedding(vocab_size)
        pe = torch.zeros(64, d_model - 2)
        pos = torch.arange(64).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model - 2, 2).float() * -(math.log(10000.0) / max(d_model - 2, 1)))
        pe[:, 0::2] = torch.sin(pos * div)[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = torch.cos(pos * div)[:, :pe[:, 1::2].shape[1]]
        self.register_buffer("pos_embed", pe)

        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = RotaryEmbedding(head_dim, theta=3.0)

        self.gate_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, d_model, bias=False)

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ln_f = RMSNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_embed(idx)
        pos = self.pos_embed[:T].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([tok, pos], dim=-1)

        mask = torch.zeros(T, T, device=x.device)
        mask.masked_fill_(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1), float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, 1, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = k.clone()
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q), self.rope(k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale + mask, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        out = F.linear(out, self.q_proj.weight.T)
        x = x + out

        h = self.ln2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        x = self.ln_f(x)
        table = self.tok_embed.table()
        logits = x[..., :2] @ table.T
        return logits

    @torch.no_grad()
    def generate(self, prompt):
        self.eval()
        seq = prompt.clone()
        for _ in range(11):
            logits = self.forward(seq)
            next_tok = logits[:, -1, :10].argmax(dim=-1)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        return seq[:, prompt.shape[1]:prompt.shape[1] + 11]


_W = {
    "tok_embed.arc_A": torch.tensor(25.02317237854004),
    "tok_embed.arc_start": torch.tensor(-0.2547111511230469),
    "tok_embed.arc_stride": torch.tensor(0.06311280280351639),
    "q_proj.weight": torch.tensor([[-0.604081392288208, -0.0002486599260009825, 21.054105758666992], [-0.21068155765533447, 0.01719030924141407, 0.13735458254814148], [0.9734416604042053, -0.1098935529589653, 0.041659124195575714], [0.05693720281124115, -2.3206427097320557, -0.47510960698127747]]),
    "k_proj.weight": torch.tensor([[-0.06312127411365509, 1.7858084440231323, -30.205820083618164], [4.746553421020508, 1.104575753211975, -70.7226791381836], [3.434957981109619, 0.7511223554611206, -24.515010833740234], [-0.13498139381408691, -0.016353625804185867, 20.890283584594727]]),
    "q_norm.weight": torch.tensor([0.604255735874176, 2.9982070922851562, 4.795403003692627, 0.0538223460316658]),
    "k_norm.weight": torch.tensor([-0.01454188022762537, 6.3072919845581055, 1.9737937450408936, 2.3363115787506104]),
    "gate_proj.weight": torch.tensor([[0.7186843156814575, -0.4613227844238281, 0.5894570350646973], [0.7117799520492554, 0.45831581950187683, -0.5933173894882202]]),
    "up_proj.weight": torch.tensor([[-1.5344116687774658, -1.4468005895614624, 1.2638859748840332], [1.5183025598526, -1.3507441282272339, 1.1220263242721558]]),
    "down_proj.weight": torch.tensor([[0.8895362019538879, -0.8233017325401306], [2.2569541931152344, 2.299072504043579], [1.345754861831665, 1.3772836923599243]]),
    "ln1.weight": torch.tensor([7.530346870422363, 3.2177560329437256, -0.021621771156787872]),
    "ln2.weight": torch.tensor([3.155890941619873, 3.9469993114471436, 7.59224796295166]),
    "ln_f.weight": torch.tensor([107.53959655761719, 28.793907165527344, 1.3171551472623833e-05]),
}


def _encode(a, b):
    a_digits = [(a // 10**i) % 10 for i in range(10)]
    b_digits = [(b // 10**i) % 10 for i in range(10)]
    return [0] + a_digits + [0, 0] + b_digits + [0]


def build_model():
    model = AdderModel()
    model.load_state_dict(_W, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    metadata = {
        "name": "CircularArc-Qwen3-62p",
        "author": "warlockee",
        "params": n_params,
        "architecture": "1L Qwen3, d=3, 1h/1kv, hd=4, ff=2, RoPE theta=3, SwiGLU",
        "tricks": [
            "Circular arc embedding (3 params)",
            "Tied K=V",
            "Tied O=Q^T",
            "Tied lm_head (decode via embedding table)",
            "Learnable QK norms",
            "Sinusoidal PE for 3rd dimension",
            "5-phase training with EMA + targeted fine-tuning",
            "Curriculum: 3->6->10 digits",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    device = next(model.parameters()).device
    prompt = _encode(a, b)
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
    output = model.generate(prompt_t)
    return sum(tok * (10**i) for i, tok in enumerate(output[0].tolist()[:11]))
