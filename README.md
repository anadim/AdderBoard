# AdderBoard

**Challenge:** Build the smallest transformer that can add two 10-digit numbers with >= 99% accuracy on a held-out 10K test set.

This started with [Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure), where I pitted [Claude Code](https://github.com/anadim/smallest-addition-transformer-claude-code) (6,080 params) against [Codex](https://github.com/anadim/smallest-addition-transformer-codex) (1,644 params) to find the smallest transformer that can add 10-digit numbers. The community has since pushed this dramatically lower.

Maintained by [Dimitris Papailiopoulos](https://github.com/anadim) ([@dimitrispapail](https://x.com/dimitrispapail)).

We track two categories:

- **Trained** — weights learned from data by any training algorithm (SGD, Adam, evolutionary search, etc.). The algorithm must be generic — it should work with any model and dataset, not just this specific problem. This encourages creative ideas around data format, tokenization, curriculum learning, and architecture search.
- **Hand-coded** — weights set analytically. This is a constructive proof that the architecture *can* represent addition, regardless of whether SGD would find it.

Both are valid. Both are interesting.

## Leaderboard

### Hand-Coded Weights (Constructive Proofs)

| Rank | Params | Accuracy | Author | Built with | Architecture | Key Tricks | Link |
|------|--------|----------|--------|------------|-------------|------------|------|
| 1 | 139 | 100% | [Wonderfall](https://github.com/Wonderfall) ([@w0nderfall](https://x.com/w0nderfall)) | GPT-5.2 Pro + Codex | 1L Qwen3, d=3, 4h/1kv, hd=2 | Tied embed, RoPE digit routing, SiLU carry logic | [gist](https://gist.github.com/Wonderfall/191bea43ff7f9316ac178b6c185d7165) |
| 2 | 177 | 100% | [xangma](https://github.com/xangma) ([@xangma](https://x.com/xangma)) | GPT + Codex | 2L Qwen3, d=5, 2h/1kv, hd=2 | Rank-1 linear, factorized embed, sparse gate, param-free norm, low-rank head | [gist](https://gist.github.com/xangma/1c2a1b2f9ca871b1f15646eed60d10ab) |
| 3 | 197 | ~100%* | [xangma](https://github.com/xangma) ([@xangma](https://x.com/xangma)) | GPT + Codex | 2L Qwen3, d=5, 2h/1kv, hd=2 | Rank-1 linear, factorized embed, sparse gate, param-free norm | [gist](https://gist.github.com/xangma/c538a7a9d415f16e61f7bb26ae5cf6b0) |

### Trained Weights (Learned from Data)

| Rank | Params | Accuracy | Author | Built with | Architecture | Key Tricks | Link |
|------|--------|----------|--------|------------|-------------|------------|------|
| 1 | 311 | 99.999% | [rezabyt](https://github.com/rezabyt) ([@reza_byt](https://x.com/reza_byt)) | | 1L decoder, d=4, 1h, ff=8 | Rank-3 factorization, shared-A tied-KV, RMSNorm, grokking | [repo](https://github.com/rezabyt/digit-addition-311p) |
| 2 | 456 | 100% | [yinglunz](https://github.com/yinglunz) | | 1L decoder, d=7, 1h, ff=14 | Rank-3 factorization, shared-A tied-KV, rank-2 attn out, tied embed | [repo](https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition) |
| 3 | 491 | 99.97% | [rezabyt](https://github.com/rezabyt) ([@reza_byt](https://x.com/reza_byt)) | | 1L decoder, d=7 | Rank-3 factorization, RMSNorm, curriculum learning | [repo](https://github.com/rezabyt/digit-addition-491p) |
| 4 | 777 | 99.69% | [Yeb Havinga](https://github.com/yhavinga) ([@YebHavinga](https://x.com/YebHavinga)) | Claude Code | 1L decoder, d=7, 1h, ff=14 | Tied embeddings, no FFN bias, curriculum learning | [repo](https://github.com/yhavinga/gpt-acc-jax) |
| 5 | 1,644 | 99.04% | [anadim](https://github.com/anadim) ([@dimitrispapail](https://x.com/dimitrispapail)) | Codex | 1L decoder, pair tokens | Pair token encoding (digit pairs as single tokens) | [repo](https://github.com/anadim/smallest-addition-transformer-codex) |
| 6 | 6,080 | 100% | [anadim](https://github.com/anadim) ([@dimitrispapail](https://x.com/dimitrispapail)) | Claude Code | 2L decoder, d=16, ff=48 | Systematic scaling, found phase transition at d=16 | [repo](https://github.com/anadim/smallest-addition-transformer-claude-code) |

\* *Passed 8,192 random tests; not independently verified on our 10K test suite yet.*

### Notable Attempts (Did Not Qualify)

| Params | Accuracy | Why | Author | Notes | Link |
|--------|----------|-----|--------|-------|------|
| 130→190 | ~93% | Below 99% threshold | [cosminscn](https://github.com/cosminscn) | 1L nanoGPT, d=4, 2h. Hand-coded. Sinusoidal resonance routing (period 11), ReLU carry detection, parabolic logit decoding. Originally 130 params, updated to 190. Fails on ~7% of random inputs. | [gist](https://gist.github.com/cosminscn/65a5fa5e20524495415f3cdd6bfdd7d2) |

## Rules

### The Core Constraint: Autoregressive Transformer

The model must operate as a **genuine autoregressive transformer**. This means:

1. **Self-attention is required.** The model must contain at least one self-attention layer. This is the defining feature of a transformer — without it, you have an MLP or RNN, not a transformer.

2. **The model must be autoregressive.** It receives a token sequence as input and predicts the next token. Output digits are generated one at a time, with each new token fed back as input for predicting the next. The carry propagation must emerge from this autoregressive process — not from explicit state variables passed between steps in Python.

3. **Standard forward pass.** The model's `forward()` method must be a standard tensor-in, logits-out computation. No problem-specific control flow (for-loops over digits, explicit carry variables, string manipulation) inside `forward()`. The autoregressive generation loop lives *outside* the model, exactly as it would for any language model.

4. **The model does the work, not the code.** The inference code should be generic autoregressive decoding that would work with *any* transformer checkpoint. If your generation loop contains addition-specific logic — manually pairing digits, threading carry state, indexing into specific positions — then the Python code is solving the problem, not the model.

In short: if you can swap in a different set of weights and use the exact same inference code for a different task, your setup is legitimate. If the inference code is inseparable from the algorithm, it's not.

### What's Allowed
- Architectural variations: rank-1/low-rank projections, factorized embeddings, custom positional encodings, alternative norms
- Hand-coded weights (constructive proofs are valid — they show the architecture *can* represent addition)
- Trained weights via any generic learning algorithm (shows the solution is *learnable* — encourages creative ideas on data format, tokenization, and curriculum)
- Input formatting choices (reversed digits, delimiters, etc.) as long as the format is fixed and doesn't encode the answer

### Qualification
- Must achieve **>= 99% accuracy** on 10,000 random test pairs (held-out, fixed seed)
- Inputs: two integers in [0, 9,999,999,999]
- Output: their sum as an integer
- Verified using `verify.py` with `--seed 2025`

### Parameter Counting
- Count **unique** parameters (after weight tying/deduplication)
- Fixed/sinusoidal positional encodings are not counted (following the original Transformer paper convention)
- Learned positional encodings are counted

## How to Submit

**Option A: Open an Issue (easiest)**
1. Click [New Issue](../../issues/new?template=new-submission.yml) and fill in the template
2. Include a link to your code (GitHub repo, gist, etc.)
3. Include test results (accuracy on random pairs)
4. We'll verify and add you to the leaderboard

**Option B: Open a Pull Request**
1. Fork this repo
2. Update the leaderboard in README.md with your entry
3. Include verification results
4. We'll review and merge

Updates to the leaderboard are welcome via pull request.

## Verification

```bash
python verify.py submissions/your_submission.py
```

This runs:
- 10 edge cases (boundary values, max carry chains)
- 10,000 random pairs (seed=2025)
- Reports accuracy, pass/fail, and timing

## Context

This challenge explores a fundamental question: **what is the minimal transformer that can represent integer addition?**

Addition requires three capabilities:
1. **Digit alignment** — pairing corresponding digits from two numbers
2. **Per-digit arithmetic** — computing sum and carry for each pair
3. **Carry propagation** — threading carry information across positions

Transformers solve these using attention (for alignment), MLPs (for arithmetic), and autoregressive generation (for carry propagation). The question is how small the architecture can be while still implementing all three.

### Key Findings from the Community
- **Parameter cliff at ~800**: Sharp accuracy transition observed by multiple researchers
- **Single layers beat two layers** at equivalent parameter budgets (for trained models)
- **d=7 was the sweet spot** for early trained models — multiple independent teams converged on this
- **d=4 now works** with rank-3 factorization + grokking (311 params trained)
- **Hand-coded models can go much smaller** (139 vs 311 trained) since they don't need to be discoverable by SGD
- **Rank-3 factorization** is the key trick for trained models
- **Vanilla architectures suffice**: the 139-param leader uses unmodified Qwen3 with just 1 layer and d=3

## License

MIT
