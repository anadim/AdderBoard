# Research Goal

## Task
Build the smallest autoregressive transformer that can add two 10-digit numbers with ≥99% accuracy on a held-out 10K test set (AdderBoard competition).

## Competition
- GitHub: https://github.com/anadim/AdderBoard
- Category: **Trained Weights** (learned from data via generic training algorithms)
- Current leader: 62 parameters (1L Qwen3, d=3, circular arc embedding, tied K=V, tied O=Q^T)

## Dataset
- Input: two integers in [0, 9,999,999,999]
- Output: their sum as an integer
- Training data: generated on-the-fly (unlimited synthetic pairs)
- Test set: 10 edge cases + 10,000 random pairs (seed=2025), verified via `verify.py`

## Metric
- **Primary**: `accuracy` — fraction of test pairs correctly added (must be ≥99%)
- **Secondary**: `num_params` — unique parameter count after weight tying (lower is better)
- **Goal**: Minimize `num_params` while maintaining ≥99% accuracy

## Current State
- Workspace initialized with AdderBoard repo + Orze orchestrator
- No trained models yet — starting from scratch

## Key Architectural Patterns (from leaderboard)
1. **1-layer Qwen3-style decoder** (d=3, 1h/1kv, hd=4, ff=2)
2. **Circular arc embedding** (3 params instead of 30-param lookup table)
3. **Weight tying**: K=V, O=Q^T, tied lm_head, shared RMSNorms
4. **RoPE** with theta=3 (zero learnable params)
5. **SwiGLU** activation in MLP
6. **QK normalization**
7. **Curriculum learning** (start with fewer digits, increase)
8. **Rank-1/low-rank projections** for attention output
9. **LSB-first** digit ordering for easier carry propagation

## Directions
1. Reproduce the 62-param architecture and try to match/beat it
2. Search over architectural hyperparameters (d_model, head_dim, ff)
3. Explore novel weight tying patterns
4. Try different embedding schemes (circular arc, quadratic, spiral)
5. Optimize training: curriculum, learning rate schedules, grokking
6. Coordinate descent on very small parameter spaces (≤10 params)
