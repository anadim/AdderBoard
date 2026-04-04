#!/usr/bin/env python3
"""
Autopilot: fully automated param-reduction search for AdderBoard.

Loop:
  1. Run Phase 0 (200K steps) across multiple seeds for current config
  2. Pick the best checkpoint (highest accuracy)
  3. Run multiphase (Phases 1-4) to push toward 99%+
  4. If qualified (>=99%), save submission and try fewer params
  5. If all seeds fail, stop reducing and report best result

No manual intervention needed.
"""

import json
import os
import sys
import time
import random
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import AdderTransformer, train_model, evaluate_model, evaluate_on_verify_set
from multiphase import train_phase, targeted_finetune

import torch

DEVICE = "cuda"
SEEDS = [314, 42, 8, 1337, 99, 555, 777, 123, 256, 7]
RESULTS_DIR = Path("results")

# Configs to try, from largest to smallest
CONFIGS = [
    {
        "name": "49p",
        "cfg": {
            "d_model": 3, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
            "ff_dim": 2, "n_layers": 1, "rope_theta": 3.0, "qk_norm": True,
            "use_swiglu": True, "use_rope": True, "norm_type": "rms",
            "embed_type": "circular_arc", "tie_embed": True, "tie_kv": True,
            "tie_qo": True, "tie_qk_norm": True, "tie_down_gate": True,
            "share_norms": True, "share_ln_f": False,
            "attn_out_rank": 0, "vocab_size": 10,
        },
    },
    {
        "name": "46p",
        "cfg": {
            "d_model": 3, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
            "ff_dim": 2, "n_layers": 1, "rope_theta": 3.0, "qk_norm": True,
            "use_swiglu": True, "use_rope": True, "norm_type": "unit",
            "embed_type": "circular_arc", "tie_embed": True, "tie_kv": True,
            "tie_qo": True, "tie_qk_norm": True, "tie_down_gate": True,
            "share_norms": True, "share_ln_f": False,
            "attn_out_rank": 0, "vocab_size": 10,
        },
    },
    {
        "name": "43p",
        "cfg": {
            "d_model": 3, "n_heads": 1, "n_kv_heads": 1, "head_dim": 4,
            "ff_dim": 2, "n_layers": 1, "rope_theta": 3.0, "qk_norm": True,
            "use_swiglu": True, "use_rope": True, "norm_type": "rms",
            "embed_type": "circular_arc", "tie_embed": True, "tie_kv": True,
            "tie_qo": True, "tie_qk_norm": True, "tie_gate": True,
            "tie_down_gate": True, "share_norms": True, "share_ln_f": False,
            "attn_out_rank": 0, "vocab_size": 10,
        },
    },
]

PHASE0_TRAIN = {
    "lr": 0.01, "min_lr": 0.001, "steps": 200000, "batch_size": 128,
    "warmup_steps": 1000, "weight_decay": 0.01, "grad_clip": 1.0,
    "eval_every": 2000, "patience": 200000, "optimizer": "adamw",
    "curriculum": "3:2000,6:7000,10:rest",
    "grokfast_alpha": 0.98, "grokfast_lambda": 3.0,
}


def run_phase0(cfg, seed, output_dir):
    """Train Phase 0 and return (accuracy, checkpoint_path)."""
    full_cfg = {**cfg, **PHASE0_TRAIN, "seed": seed}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AdderTransformer(full_cfg).to(DEVICE)
    n_params = model.count_params()
    print(f"  Phase 0: {n_params}p, seed={seed}")

    model, metrics = train_model(full_cfg, DEVICE)

    # Save checkpoint
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({"config": full_cfg, "state_dict": model.state_dict()}, ckpt_path)

    acc = metrics["accuracy"]
    print(f"  Phase 0 result: acc={acc:.4f}, best_step={metrics['best_step_accuracy']:.4f}")
    return acc, ckpt_path, n_params


def run_multiphase(ckpt_path, output_dir, max_rounds=3):
    """Run multiphase rounds until >=99% or max_rounds."""
    output_dir = Path(output_dir)

    for rnd in range(max_rounds):
        rnd_dir = output_dir / f"round{rnd}"
        rnd_dir.mkdir(parents=True, exist_ok=True)

        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        cfg = ckpt["config"]
        model = AdderTransformer(cfg).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        n_params = model.count_params()

        initial_acc = evaluate_model(model, DEVICE, num_tests=2000)
        print(f"  Round {rnd}: starting from {initial_acc:.4f}")

        # Phase 1
        best1, acc1, ema1, ema_acc1 = train_phase(model, DEVICE, {
            "lr": 0.001, "steps": 50000, "batch_size": 256,
            "weight_decay": 0.01, "ema_decay": 0.999,
            "lr_schedule": "constant", "optimizer": "adamw"})
        if ema_acc1 > acc1 and ema1:
            model.load_state_dict(ema1)
        elif best1:
            model.load_state_dict(best1)

        # Phase 2
        best2, acc2, ema2, ema_acc2 = train_phase(model, DEVICE, {
            "lr": 0.0003, "steps": 30000, "batch_size": 256,
            "weight_decay": 0.01, "ema_decay": 0.999,
            "lr_schedule": "constant", "optimizer": "adamw"})
        if ema_acc2 > acc2 and ema2:
            model.load_state_dict(ema2)
        elif best2:
            model.load_state_dict(best2)

        # Phase 3
        best3, acc3, _, _ = train_phase(model, DEVICE, {
            "lr": 0.001, "steps": 50000, "batch_size": 256,
            "weight_decay": 0.0, "ema_decay": 0,
            "lr_schedule": "cosine_to_zero", "optimizer": "adam"})
        if best3:
            model.load_state_dict(best3)

        # Phase 4
        model = targeted_finetune(model, DEVICE)

        # Evaluate
        final_acc = evaluate_model(model, DEVICE, num_tests=5000)
        verify_acc = evaluate_on_verify_set(model, DEVICE)
        print(f"  Round {rnd} result: acc={final_acc:.4f}, verify={verify_acc:.4f}")

        # Save
        ckpt_path = rnd_dir / "checkpoint.pt"
        torch.save({"config": cfg, "state_dict": model.state_dict()}, ckpt_path)
        (rnd_dir / "metrics.json").write_text(json.dumps({
            "accuracy": final_acc, "verify_accuracy": verify_acc,
            "num_params": n_params, "qualified": verify_acc >= 0.99,
        }, indent=2))

        if verify_acc >= 0.99:
            print(f"  QUALIFIED at {n_params}p! verify={verify_acc:.4f}")
            return final_acc, verify_acc, ckpt_path, n_params

    return final_acc, verify_acc, ckpt_path, n_params


def main():
    print("=" * 60)
    print("AUTOPILOT: Automated AdderBoard param reduction")
    print("=" * 60)

    best_qualified = None

    for config in CONFIGS:
        name = config["name"]
        cfg = config["cfg"]
        print(f"\n{'='*60}")
        print(f"Trying {name}")
        print(f"{'='*60}")

        # Phase 0: try multiple seeds
        best_seed_acc = 0
        best_seed_ckpt = None
        best_seed = None

        for seed in SEEDS[:6]:  # Try first 6 seeds
            out_dir = RESULTS_DIR / f"auto_{name}_s{seed}"
            try:
                acc, ckpt, n_params = run_phase0(cfg, seed, out_dir)
                if acc > best_seed_acc:
                    best_seed_acc = acc
                    best_seed_ckpt = ckpt
                    best_seed = seed
                    print(f"  NEW BEST seed={seed}: {acc:.4f}")
                if acc >= 0.99:
                    print(f"  Phase 0 already qualified! Skipping multiphase.")
                    best_qualified = (n_params, acc, ckpt)
                    break
            except Exception as e:
                print(f"  seed={seed} FAILED: {e}")

        if best_seed_acc < 0.01:
            print(f"  {name}: No seed reached >1% accuracy. Skipping multiphase.")
            continue

        # Multiphase on best seed
        print(f"\n  Running multiphase on best seed={best_seed} ({best_seed_acc:.4f})")
        mp_dir = RESULTS_DIR / f"auto_{name}_s{best_seed}_mp"
        final_acc, verify_acc, final_ckpt, n_params = run_multiphase(
            best_seed_ckpt, mp_dir, max_rounds=5)

        if verify_acc >= 0.99:
            best_qualified = (n_params, verify_acc, final_ckpt)
            print(f"\n  *** QUALIFIED: {n_params}p at {verify_acc:.4f} ***")
        else:
            print(f"\n  {name}: best={verify_acc:.4f}, not qualified. Moving on.")

    print(f"\n{'='*60}")
    if best_qualified:
        n, acc, ckpt = best_qualified
        print(f"BEST RESULT: {n}p, verify={acc:.4f}, checkpoint={ckpt}")
    else:
        print("No qualified model found.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
