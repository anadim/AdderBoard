"""
Minimal addition computer following the LLM-as-computer philosophy.

Key insight: digit addition mod 10 = angle addition on a circle.
  embed(d) = [cos(2πd/10), sin(2πd/10)]
  embed(x+y) = complex_multiply(embed(x), embed(y))  ← this IS addition mod 10
  decode: argmax of dot product with embedding table

The structure IS the algorithm. Only 2-3 scalars need tuning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularAdder(nn.Module):
    """
    A minimal digit-addition computer.

    Architecture = the addition algorithm:
      1. Embed digits on unit circle (period P≈10)
      2. Complex-multiply embeddings = add angles = add digits mod P
      3. Inject carry by rotating one more step
      4. Decode via dot product with embedding table
      5. Carry gate: sigmoid(sharpness * (x+y+carry - 9.5))

    Learnable params: period (P), carry_sharpness, logit_scale
    """
    def __init__(self, n_params=3):
        super().__init__()
        self.P = nn.Parameter(torch.tensor(10.0))          # circle period
        self.sharp = nn.Parameter(torch.tensor(10.0))       # carry gate sharpness
        self.logit_scale = nn.Parameter(torch.tensor(5.0))  # scale logits for sharper softmax

    def embed(self, digit):
        """Embed digit on unit circle: [cos(2πd/P), sin(2πd/P)]"""
        angle = 2 * math.pi * digit.float() / self.P
        return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

    def complex_mul(self, a, b):
        """Complex multiplication = angle addition on circle."""
        return torch.stack([
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ], dim=-1)

    def step(self, x_digit, y_digit, carry_in):
        """One digit-addition step."""
        # Embed digits on circle
        ex = self.embed(x_digit)   # [B, 2]
        ey = self.embed(y_digit)   # [B, 2]

        # Add x + y on circle (complex multiply = angle addition)
        h = self.complex_mul(ex, ey)   # embed(x+y mod P)

        # Add carry: rotate by one digit step (2π/P) scaled by carry_in
        carry_angle = 2 * math.pi * carry_in / self.P   # [B]
        carry_vec = torch.stack([torch.cos(carry_angle), torch.sin(carry_angle)], dim=-1)
        h = self.complex_mul(h, carry_vec)   # embed(x+y+carry mod P)

        # Decode: dot product with embedding table
        table = self.embed(torch.arange(10, device=x_digit.device))  # [10, 2]
        logits = self.logit_scale * (h @ table.T)   # [B, 10]
        # cos similarity is max when angles match → argmax = (x+y+carry) mod 10

        # Carry gate: x+y+carry >= 10?
        raw_sum = x_digit.float() + y_digit.float() + carry_in
        carry_out = torch.sigmoid(self.sharp * (raw_sum - 9.5))

        return logits, carry_out

    def forward(self, x_digits_rev, y_digits_rev):
        """
        x_digits_rev, y_digits_rev: [B, T] LSB-first digit sequences
        returns: (logits [B, T, 10], final_carry [B])
        """
        B, T = x_digits_rev.shape
        carry = torch.zeros(B, device=x_digits_rev.device)
        logits_list = []
        for t in range(T):
            logits, carry = self.step(x_digits_rev[:, t], y_digits_rev[:, t], carry)
            logits_list.append(logits)
        return torch.stack(logits_list, dim=1), carry

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── Data ───

def to_rev_digits(nums, width=10):
    out = []
    for _ in range(width):
        out.append((nums % 10).long())
        nums = nums // 10
    return torch.stack(out, dim=1)


def make_batch(bs=256, width=10, device="cpu"):
    a = torch.randint(0, 10**width, (bs,), device=device)
    b = torch.randint(0, 10**width, (bs,), device=device)
    s = a + b
    x = to_rev_digits(a, width).to(device)
    y = to_rev_digits(b, width).to(device)
    target = to_rev_digits(s % (10**width), width).to(device)
    carry = (s >= 10**width).float().to(device)
    return x, y, target, carry


# ─── Eval ───

@torch.no_grad()
def evaluate(model, n=10000, width=10, device="cpu"):
    x, y, target, tc = make_batch(n, width, device)
    logits, pc = model(x, y)
    pred = logits.argmax(-1)
    digit_acc = (pred == target).float().mean().item()
    seq_acc = (pred == target).all(dim=1).float().mean().item()
    carry_acc = ((pc > 0.5).float() == tc).float().mean().item()
    return digit_acc, seq_acc, carry_acc


# ─── Train ───

def train(steps=2000, bs=512, width=10, lr=0.1, device="cuda"):
    model = CircularAdder().to(device)
    print(f"Parameters: {model.count_params()}")
    print(f"  P={model.P.item():.2f}, sharp={model.sharp.item():.2f}, scale={model.logit_scale.item():.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps + 1):
        x, y, target, tc = make_batch(bs, width, device)
        logits, pc = model(x, y)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        loss = loss + 0.2 * F.binary_cross_entropy(pc.clamp(1e-6, 1-1e-6), tc)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            da, sa, ca = evaluate(model, 5000, width, device)
            print(f"step={step:4d} loss={loss.item():.4f} digit={da:.4f} seq={sa:.4f} carry={ca:.4f} "
                  f"| P={model.P.item():.4f} sharp={model.sharp.item():.1f} scale={model.logit_scale.item():.1f}")

    # Final eval
    da, sa, ca = evaluate(model, 10000, width, device)
    print(f"\nFinal: digit_acc={da:.4f} seq_acc={sa:.4f} carry_acc={ca:.4f}")
    return model


@torch.no_grad()
def demo(model, pairs, width=10, device="cpu"):
    for a, b in pairs:
        x = to_rev_digits(torch.tensor([a], device=device), width)
        y = to_rev_digits(torch.tensor([b], device=device), width)
        logits, fc = model(x, y)
        pred = logits.argmax(-1)[0]
        # Reconstruct number from reversed digits
        result = sum(int(d) * 10**i for i, d in enumerate(pred.tolist()))
        if fc.item() > 0.5:
            result += 10**width
        print(f"  {a} + {b} = {result}  (true: {a+b})  {'✓' if result == a+b else '✗'}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(steps=2000, bs=512, width=10, lr=0.1, device=device)

    print("\nDemo:")
    demo(model, [
        (0, 0), (5, 5), (999, 1), (9999999999, 1),
        (9999999999, 9999999999), (1234567890, 9876543210),
        (5000000000, 5000000000),
    ], device=device)
