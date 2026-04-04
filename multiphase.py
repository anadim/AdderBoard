#!/usr/bin/env python3
"""
Multi-phase training pipeline for AdderBoard, matching tbukic's 5-phase recipe.

Phase 0: Cosine LR warmup (already done by orze via train.py)
Phase 1: Constant LR=0.001 + EMA (50K steps)
Phase 2: Constant LR=0.0003 + EMA (30K steps)
Phase 3: Adam no-WD cosine LR=0.001→0 (50K steps)
Phase 4: Targeted fine-tuning on error cases (up to 10 iterations × 3K steps)

Usage: python multiphase.py --checkpoint results/idea-XXXX/checkpoint.pt --gpu 0
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train import (
    AdderTransformer, encode_pair, evaluate_model, evaluate_on_verify_set,
    generate_batch, VOCAB_SIZE, NUM_DIGITS, OUTPUT_LEN, PROMPT_LEN, MAX_ADDEND,
)


def train_phase(model, device, phase_cfg, eval_every=2000):
    """Run one training phase. Returns (best_state, best_acc, ema_state, ema_acc)."""
    lr = phase_cfg["lr"]
    steps = phase_cfg["steps"]
    batch_size = phase_cfg.get("batch_size", 256)
    weight_decay = phase_cfg.get("weight_decay", 0.01)
    ema_decay = phase_cfg.get("ema_decay", 0)
    lr_schedule = phase_cfg.get("lr_schedule", "constant")  # constant, cosine, cosine_to_zero
    optimizer_name = phase_cfg.get("optimizer", "adamw")

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # EMA
    ema_state = None
    if ema_decay > 0:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    best_acc = 0.0
    best_state = None
    ema_best_acc = 0.0
    ema_best_state = None
    t0 = time.time()

    for step in range(1, steps + 1):
        # LR schedule
        if lr_schedule == "cosine":
            min_lr = phase_cfg.get("min_lr", lr * 0.1)
            progress = step / steps
            current_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
        elif lr_schedule == "cosine_to_zero":
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        else:
            current_lr = lr

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Generate batch (always full 10-digit, no curriculum in later phases)
        prompts, targets = generate_batch(batch_size, device=device)
        full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
        logits = model(full_input)
        output_logits = logits[:, PROMPT_LEN - 1:PROMPT_LEN - 1 + OUTPUT_LEN, :VOCAB_SIZE]
        loss = F.cross_entropy(output_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        # Evaluate
        if step % eval_every == 0 or step == steps:
            acc = evaluate_model(model, device, num_tests=500)
            elapsed = time.time() - t0

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            ema_tag = ""
            if ema_state is not None:
                # Eval EMA
                orig_state = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                ema_acc = evaluate_model(model, device, num_tests=500)
                model.load_state_dict(orig_state)
                if ema_acc > ema_best_acc:
                    ema_best_acc = ema_acc
                    ema_best_state = {k: v.clone() for k, v in ema_state.items()}
                ema_tag = f" | EMA: {ema_acc:.4f}"

            print(f"  Step {step}/{steps} | Loss: {loss.item():.4f} | Acc: {acc:.4f}{ema_tag} | LR: {current_lr:.6f} | Time: {elapsed:.0f}s")

            if acc >= 1.0:
                print(f"  Perfect accuracy at step {step}!")
                break

    return best_state, best_acc, ema_best_state, ema_best_acc


def targeted_finetune(model, device, lr=0.001, steps_per_iter=3000, max_iters=10,
                      eval_every=500, eval_seed=777, n_eval=10000):
    """Phase 4: find error cases and train specifically on them."""
    print(f"\n=== Phase 4: Targeted Fine-Tuning ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_errors = []
    best_ft_acc = 0.0
    best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}

    for iteration in range(max_iters):
        # Find errors
        model.eval()
        rng = random.Random(eval_seed + iteration)
        errors = []
        correct = 0
        with torch.no_grad():
            for _ in range(n_eval):
                a = rng.randint(0, MAX_ADDEND)
                b = rng.randint(0, MAX_ADDEND)
                expected = a + b
                prompt, _ = encode_pair(a, b)
                prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)
                output = model.generate(prompt_t)
                result = sum(tok * (10**i) for i, tok in enumerate(output[0].tolist()[:OUTPUT_LEN]))
                if result != expected:
                    errors.append((a, b))
                else:
                    correct += 1

        acc = correct / n_eval
        if acc > best_ft_acc:
            best_ft_acc = acc
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Iter {iteration}: {len(errors)} errors ({acc:.4%} accuracy) [best: {best_ft_acc:.4%}]")

        if len(errors) == 0:
            print(f"  PERFECT — 0 errors on {n_eval} test cases!")
            break

        all_errors.extend(errors)

        # Train on error-heavy batches
        model.train()
        for step in range(1, steps_per_iter + 1):
            # Mix: half errors, half random
            batch_errors = [random.choice(all_errors) for _ in range(128)]
            batch_random_a = [random.randint(0, MAX_ADDEND) for _ in range(128)]
            batch_random_b = [random.randint(0, MAX_ADDEND) for _ in range(128)]

            prompts, targets = [], []
            for a, b in batch_errors:
                p, t = encode_pair(a, b)
                prompts.append(p)
                targets.append(t)
            for a, b in zip(batch_random_a, batch_random_b):
                p, t = encode_pair(a, b)
                prompts.append(p)
                targets.append(t)

            prompts = torch.tensor(prompts, dtype=torch.long, device=device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            full_input = torch.cat([prompts, targets[:, :-1]], dim=1)
            logits = model(full_input)
            output_logits = logits[:, PROMPT_LEN-1:PROMPT_LEN-1+OUTPUT_LEN, :VOCAB_SIZE]
            loss = F.cross_entropy(output_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % eval_every == 0:
                quick_acc = evaluate_model(model, device, num_tests=500)
                print(f"    FT step {step}/{steps_per_iter} | Loss: {loss.item():.4f} | Acc: {quick_acc:.4f}")

    # Restore best FT state
    print(f"  Restoring best FT checkpoint ({best_ft_acc:.4%})")
    model.load_state_dict(best_ft_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to Phase 0 checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = AdderTransformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    print(f"Loaded {model.count_params()}p model from {args.checkpoint}")

    initial_acc = evaluate_model(model, device, num_tests=2000)
    print(f"Initial accuracy: {initial_acc:.4f}")

    # Phase 1: Constant LR + EMA
    print(f"\n=== Phase 1: Constant LR=0.001 + EMA (50K steps) ===")
    best1, acc1, ema1, ema_acc1 = train_phase(model, device, {
        "lr": 0.001, "steps": 50000, "batch_size": 256,
        "weight_decay": 0.01, "ema_decay": 0.999,
        "lr_schedule": "constant", "optimizer": "adamw",
    })
    # Use whichever is better: training best or EMA best
    if ema_acc1 > acc1 and ema1 is not None:
        print(f"  Using EMA checkpoint (EMA: {ema_acc1:.4f} > train: {acc1:.4f})")
        model.load_state_dict(ema1)
    elif best1 is not None:
        print(f"  Using training checkpoint ({acc1:.4f})")
        model.load_state_dict(best1)

    torch.save({"config": cfg, "state_dict": model.state_dict()},
               output_dir / "checkpoint_phase1.pt")

    # Phase 2: Lower constant LR + EMA
    print(f"\n=== Phase 2: Constant LR=0.0003 + EMA (30K steps) ===")
    best2, acc2, ema2, ema_acc2 = train_phase(model, device, {
        "lr": 0.0003, "steps": 30000, "batch_size": 256,
        "weight_decay": 0.01, "ema_decay": 0.999,
        "lr_schedule": "constant", "optimizer": "adamw",
    })
    if ema_acc2 > acc2 and ema2 is not None:
        print(f"  Using EMA checkpoint (EMA: {ema_acc2:.4f} > train: {acc2:.4f})")
        model.load_state_dict(ema2)
    elif best2 is not None:
        print(f"  Using training checkpoint ({acc2:.4f})")
        model.load_state_dict(best2)

    torch.save({"config": cfg, "state_dict": model.state_dict()},
               output_dir / "checkpoint_phase2.pt")

    # Phase 3: Adam no-WD cosine to zero
    print(f"\n=== Phase 3: Adam no-WD, cosine LR=0.001→0 (50K steps) ===")
    best3, acc3, _, _ = train_phase(model, device, {
        "lr": 0.001, "steps": 50000, "batch_size": 256,
        "weight_decay": 0.0, "ema_decay": 0,
        "lr_schedule": "cosine_to_zero", "optimizer": "adam",
    })
    if best3 is not None:
        model.load_state_dict(best3)
        print(f"  Best accuracy: {acc3:.4f}")

    torch.save({"config": cfg, "state_dict": model.state_dict()},
               output_dir / "checkpoint_phase3.pt")

    # Phase 4: Targeted fine-tuning
    model = targeted_finetune(model, device)

    # Final evaluation
    final_acc = evaluate_model(model, device, num_tests=5000)
    verify_acc = evaluate_on_verify_set(model, device)
    print(f"\n=== FINAL RESULTS ===")
    print(f"Accuracy (5K random): {final_acc:.4f}")
    print(f"Verify accuracy (10K+10 edge): {verify_acc:.4f}")
    print(f"Parameters: {model.count_params()}")

    # Save final
    torch.save({"config": cfg, "state_dict": model.state_dict()},
               output_dir / "checkpoint_final.pt")

    metrics = {
        "accuracy": final_acc,
        "verify_accuracy": verify_acc,
        "num_params": model.count_params(),
        "qualified": final_acc >= 0.99,
    }
    (output_dir / "multiphase_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
