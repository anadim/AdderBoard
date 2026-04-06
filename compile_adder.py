"""
Compile the circular addition algorithm into minimal transformer weights.

Strategy: the CircularAdder proves 3 scalars suffice for the MATH.
Now we need to find transformer weights (Q, K, FFN, embed) that implement
the same computation through attention + FFN.

Instead of training on addition examples (gradient descent can't find
the solution), we directly optimize transformer weights to match the
CircularAdder's behavior on random digit pairs.

This is "compilation by optimization" — not learning addition, but
finding the specific weight values that implement a known algorithm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from train import (AdderTransformer, encode_pair, evaluate_model,
                   evaluate_on_verify_set, VOCAB_SIZE, NUM_DIGITS,
                   OUTPUT_LEN, PROMPT_LEN, MAX_ADDEND)


def compile_weights(
    cfg,
    n_restarts=50,
    steps_per_restart=5000,
    batch_size=512,
    lr=0.03,
    device="cuda",
):
    """
    Find transformer weights that implement addition by direct optimization.

    Unlike standard training which uses cross-entropy on random examples,
    this uses aggressive optimization with many random restarts to search
    the small weight space exhaustively.
    """
    best_acc = 0.0
    best_state = None

    for restart in range(n_restarts):
        model = AdderTransformer(cfg).to(device)

        # Random initialization with different scales
        with torch.no_grad():
            scale = 0.5 + torch.rand(1).item() * 2.0  # random scale [0.5, 2.5]
            for p in model.parameters():
                p.data = torch.randn_like(p) * scale

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for step in range(steps_per_restart):
            # Generate batch
            max_val = 10**NUM_DIGITS - 1
            a = torch.randint(0, max_val + 1, (batch_size,))
            b = torch.randint(0, max_val + 1, (batch_size,))

            prompts = []
            targets = []
            for i in range(batch_size):
                p, t = encode_pair(a[i].item(), b[i].item())
                prompts.append(p)
                targets.append(t)

            prompts = torch.tensor(prompts, dtype=torch.long, device=device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
            logits = model(full_input)

            output_logits = logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + OUTPUT_LEN, :VOCAB_SIZE]
            loss = F.cross_entropy(output_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        acc = evaluate_model(model, device, num_tests=2000)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  restart {restart:3d}: acc={acc:.4f} (NEW BEST) loss={loss.item():.4f}")
        elif restart % 10 == 0:
            print(f"  restart {restart:3d}: acc={acc:.4f} (best={best_acc:.4f})")

    if best_state:
        model = AdderTransformer(cfg).to(device)
        model.load_state_dict(best_state)
    return model, best_acc


def evolutionary_search(
    cfg,
    population=100,
    generations=500,
    mutation_rate=0.1,
    batch_size=256,
    device="cuda",
):
    """
    Evolutionary strategy: maintain a population of weight vectors,
    evaluate each on addition, keep the best, mutate.

    Better than gradient descent for tiny discrete-like landscapes.
    """
    # Flatten all params into a single vector
    ref_model = AdderTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in ref_model.parameters())
    print(f"Searching over {n_params} parameters with ES...")

    # Initialize population
    pop = torch.randn(population, n_params, device=device) * 1.0
    best_fitness = -1
    best_genome = None

    def set_weights(model, genome):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data = genome[idx:idx+n].reshape(p.shape)
            idx += n

    def eval_fitness(model, n=500):
        """Quick fitness: fraction of correct digit predictions."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _ in range(n // 50):
                a = torch.randint(0, 10**NUM_DIGITS, (50,))
                b = torch.randint(0, 10**NUM_DIGITS, (50,))
                prompts, targets = [], []
                for i in range(50):
                    p, t = encode_pair(a[i].item(), b[i].item())
                    prompts.append(p)
                    targets.append(t)
                prompts = torch.tensor(prompts, dtype=torch.long, device=device)
                targets = torch.tensor(targets, dtype=torch.long, device=device)
                full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
                logits = model(full_input)
                output_logits = logits[:, PROMPT_LEN-1:PROMPT_LEN-1+OUTPUT_LEN, :VOCAB_SIZE]
                pred = output_logits.argmax(-1)
                correct += (pred == targets).sum().item()
                total += targets.numel()
        return correct / total

    for gen in range(generations):
        # Evaluate population
        fitnesses = torch.zeros(population, device=device)
        for i in range(population):
            set_weights(ref_model, pop[i])
            fitnesses[i] = eval_fitness(ref_model, n=200)

        # Track best
        best_idx = fitnesses.argmax()
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx].item()
            best_genome = pop[best_idx].clone()

        if gen % 50 == 0:
            print(f"  gen {gen:4d}: best_digit_acc={best_fitness:.4f} "
                  f"pop_mean={fitnesses.mean():.4f} pop_max={fitnesses.max():.4f}")

        if best_fitness >= 0.99:
            print(f"  CONVERGED at gen {gen}!")
            break

        # Selection: top 20%
        k = population // 5
        _, top_idx = fitnesses.topk(k)
        parents = pop[top_idx]

        # Create next generation
        new_pop = [best_genome.clone()]  # elitism
        for _ in range(population - 1):
            parent = parents[torch.randint(0, k, (1,)).item()]
            child = parent + mutation_rate * torch.randn(n_params, device=device)
            new_pop.append(child)
        pop = torch.stack(new_pop)

    # Set best weights
    model = AdderTransformer(cfg).to(device)
    set_weights(model, best_genome)
    return model, best_fitness


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 10-param config: d=2, hd=2, maximum tying
    cfg = {
        "d_model": 2, "n_heads": 1, "n_kv_heads": 1, "head_dim": 2,
        "ff_mult": 1, "n_layers": 1, "rope_theta": 3.0, "rope_period": 19,
        "qk_norm": True, "use_swiglu": True, "use_rope": True,
        "norm_type": "none", "embed_type": "quadratic",
        "tie_embed": True, "tie_kv": True, "tie_qo": True, "tie_qk": True,
        "tie_gate": True, "tie_down_gate": True,
        "attn_out_rank": 0, "vocab_size": 10,
    }

    model = AdderTransformer(cfg).to(device)
    n_params = model.count_params()
    print(f"Model has {n_params} parameters")

    print("\n=== Method 1: Evolutionary Search ===")
    model_es, acc_es = evolutionary_search(cfg, population=200, generations=1000,
                                            mutation_rate=0.3, device=device)
    print(f"ES result: digit_acc={acc_es:.4f}")
    full_acc = evaluate_model(model_es, device, num_tests=5000)
    print(f"Full sequence accuracy: {full_acc:.4f}")

    if full_acc < 0.5:
        print("\n=== Method 2: Multi-restart gradient descent ===")
        model_gd, acc_gd = compile_weights(cfg, n_restarts=100, steps_per_restart=3000,
                                            lr=0.03, device=device)
        print(f"GD result: acc={acc_gd:.4f}")
