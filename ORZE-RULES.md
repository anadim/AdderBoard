# Orze Rules for LLM Agents

This document tells you everything you need to operate orze — a filesystem-coordinated GPU experiment orchestrator. Read this before writing ideas or interpreting results.

## Overview

Orze runs experiments by:
1. Reading experiment definitions from `ideas.md` (markdown + YAML)
2. Claiming unclaimed ideas via atomic `mkdir` (filesystem lock)
3. Running optional pre-training checks (e.g., feature extraction)
4. Launching training as subprocesses on free GPUs
5. Monitoring health (stalls, OOM, disk space)
6. Running post-training evaluation and additional scripts
7. Periodic garbage collection and cleanup
8. Generating a configurable leaderboard report and machine-readable status

You control it by editing `ideas.md` and reading `results/`.

## Ideas Format

Each idea is an H2 header with an ID, title, metadata, and YAML config:

```markdown
## idea-001: My Experiment Name
- **Priority**: high
- **Category**: architecture
- **Parent**: none
- **Hypothesis**: Why this might work.

\```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  lr: 0.001
  epochs: 5
\```
```

### Required Fields
- **H2 header**: `## idea-NNN: Title` — the ID must be `idea-` followed by digits
- **YAML block**: Contains the experiment config passed to the training script

### Optional Fields
- **Priority**: `critical` > `high` > `medium` (default) > `low` — controls execution order
- **Category**: Free-form label for grouping (e.g., architecture, hyperparameter, augmentation, data, loss, ensemble)
- **Parent**: `none` or `idea-XXX` — tracks which idea inspired this one
- **Hypothesis**: Why you think this idea will work — helps interpret results later

### ID Rules
- Format: `idea-NNN` where NNN is zero-padded (e.g., idea-001, idea-042, idea-1337)
- IDs must be unique within ideas.md
- Higher priority ideas run first; within same priority, lower IDs run first

### Append-Only Rule
ideas.md is **append-only**. Only add new ideas — never edit or delete existing ones. Status is tracked by the filesystem (see Experiment Lifecycle), not in this file.

## The metrics.json Contract

Your training script **must** write `results/{idea_id}/metrics.json` when done:

```json
{
  "status": "COMPLETED",
  "test_accuracy": 0.9234,
  "test_loss": 0.2451,
  "training_time": 142.5,
  "num_params": 1250000
}
```

### Rules
- `status` must be `"COMPLETED"` or `"FAILED"` — this is the only required field
- If `"FAILED"`, include `"error"` with a description
- Add any metrics you want — they can be displayed in the report via `orze.yaml` config
- The training script receives: `--idea-id`, `--results-dir`, `--ideas-md`, `--config`, plus any `train_extra_args` from orze.yaml

## Experiment Lifecycle

```
QUEUED → CLAIMED → [PRE-CHECK] → TRAINING → COMPLETED or FAILED → [EVAL] → [POST-SCRIPTS]
```

1. **QUEUED**: Idea exists in ideas.md, no `results/{idea_id}/` directory
2. **CLAIMED**: `results/{idea_id}/` created (atomic mkdir), `claim.json` written
3. **PRE-CHECK**: Optional `pre_script` runs (e.g., verify features exist)
4. **TRAINING**: Subprocess running, writing to `train_output.log`
5. **COMPLETED**: `metrics.json` written with `status: COMPLETED`
6. **FAILED**: `metrics.json` written with `status: FAILED` (by script or orze)
7. **EVAL**: Optional `eval_script` runs, writes eval output file
8. **POST-SCRIPTS**: Optional additional scripts run (overlays, analysis, etc.)

### Failure Causes (auto-detected by orze)
- **Timeout**: Training exceeded `timeout` seconds
- **Stalled**: No log output for `stall_minutes` minutes
- **OOM**: CUDA out of memory detected in log
- **Crash**: Non-zero exit code
- **Pre-script failure**: Pre-training check failed (e.g., missing features)

### Executor LLM Fix (Auto-Retry)
When `max_fix_attempts` > 0 in orze.yaml, failed ideas are automatically sent to an LLM for diagnosis. The LLM reads the error log and idea config, then attempts to fix the project code (scripts, configs, utilities — anything except `orze/` and `ideas.md`). If a fix is applied, the idea is re-launched on the same GPU. Fix attempts are tracked per idea and persisted across restarts.

```yaml
max_fix_attempts: 2          # try up to 2 LLM fixes per failed idea
executor_fix:
  model: sonnet              # LLM model (default: sonnet)
  timeout: 300               # max time per fix attempt (default: 300s)
```

Fix logs are saved to `results/_fix_logs/{idea_id}_attempt{N}.log`.

### Reclaiming Failed Ideas
- Delete the `results/{idea_id}/` directory to allow retry
- Or set `max_idea_failures` in orze.yaml to auto-skip after N failures
- Orphaned claims (no metrics.json after `orphan_timeout_hours`) are auto-cleaned

## Reading Results

### results/report.md
Auto-generated leaderboard. Columns are configurable via `orze.yaml`. Sorted by primary metric.

### results/status.json
Machine-readable status, updated every iteration:

```json
{
  "timestamp": "2026-02-16T14:30:00",
  "iteration": 142,
  "active": [{"idea_id": "idea-045", "gpu": 3, "elapsed_min": 12.5}],
  "free_gpus": [0, 1, 2, 4, 5, 6, 7],
  "queue_depth": 87,
  "completed": 55,
  "failed": 3,
  "skipped": 2,
  "disk_free_gb": 1024.5,
  "top_results": [...]
}
```

Use this to monitor progress programmatically.

### Per-Idea Files
Each `results/{idea_id}/` contains:
- `claim.json` — who claimed it, when, on which GPU
- `train_output.log` — stdout/stderr from training
- `metrics.json` — final metrics (the contract)
- `eval_output.log` — eval stdout (if eval configured)
- Other files written by the training/eval/post scripts

## Config Merging (Your Training Script's Job)

Orze passes both `--ideas-md` and `--config` (base config) to your training script. **Your script is responsible for merging configs:**

1. Load the base config from `--config` (e.g., `configs/base.yaml`)
2. Parse the idea-specific YAML block from `--ideas-md` using `--idea-id`
3. Merge: idea config overrides base config
4. Train with the merged config

This keeps orze generic — it doesn't need to understand your config schema.

## JIT Feature Extraction Pattern

If your project uses pre-extracted features (frozen backbone → .pt files), use this atomic staging pattern in your training script:

```
1. Check: does features/{backbone}/ exist?
   → Yes: proceed to training
   → No: need to extract, go to step 2

2. Race: mkdir features/.tmp_{backbone}
   → Succeeded: you are the extractor, go to step 3
   → Failed (EEXIST): someone else is extracting, go to step 5

3. Extract features using your ONE claimed GPU only
   (do NOT spawn a multi-GPU loop — other GPUs are running other ideas)
   Save .pt files into features/.tmp_{backbone}/

4. When 100% done: mv features/.tmp_{backbone} → features/{backbone}
   (atomic rename on POSIX — readers never see partial data)

5. Wait: poll for features/{backbone}/ to exist (sleep 30, loop)
   Another agent is extracting. Once it appears, proceed.
```

Orze's `pre_script` hook can automate this check. If features are missing and no one is extracting, the pre-script can trigger extraction before training starts.

## Garbage Collection & Cleanup

Long-running experiments generate disk pressure. Orze handles cleanup via:

### Built-in Cleanup
Configure `cleanup.patterns` in orze.yaml to auto-delete files matching glob patterns from result directories:

```yaml
cleanup:
  interval: 100        # run every 100 iterations
  patterns:
    - "checkpoint_epoch*.pt"   # delete intermediate checkpoints
    - "*.tmp"                  # delete temp files
```

### Custom Cleanup Script
For more complex cleanup (frame cache, VRAM monitoring, etc.):

```yaml
cleanup:
  script: scripts/cleanup.py
  timeout: 300
```

### Disk Space Protection
Set `min_disk_gb` to pause new launches when disk space is low:

```yaml
min_disk_gb: 50   # pause if < 50GB free
```

## Agent Roles

Orze supports multiple **agent roles** that run alongside training. Each role is an independently scheduled subprocess with its own cooldown, timeout, lock, state, and log directory.

The most common role is `research` (generates new experiment ideas), but you can add any role: `documenter`, `analyzer`, `pruner`, etc.

### Configuration

Roles are defined under the `roles:` key in `orze.yaml`:

```yaml
roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    cooldown: 300
    timeout: 600
    model: sonnet
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
    log_dir: _research_logs

  documenter:
    mode: claude
    rules_file: DOCUMENTER_RULES.md
    cooldown: 600
    timeout: 300
    model: haiku
    log_dir: _documenter_logs
```

**Legacy support**: A top-level `research:` key (without `roles:`) is auto-migrated to `roles: {research: ...}`.

### Role Modes

Each role supports two modes: **script** or **claude**.

### The Full Loop

```
┌──────────────────────────────────────────────────┐
│                      orze                         │
│                                                   │
│   ┌─────────┐     ┌─────────┐     ┌──────────┐  │
│   │ Research │────>│  Train  │────>│ Evaluate │  │
│   │  Agent   │     │ (GPUs)  │     │          │  │
│   └────▲────┘     └─────────┘     └──────────┘  │
│        │                                │         │
│        └────────── results/ ◄───────────┘         │
│                                                   │
│   ideas.md ◄── research ── report.md              │
└──────────────────────────────────────────────────┘
```

Each iteration:
1. Orze runs all configured **agent roles** (each independently rate-limited)
2. Research role reads `results/report.md` and `results/status.json`, appends new ideas to `ideas.md`
3. Orze parses `ideas.md`, claims unclaimed ideas, launches **training** on free GPUs
4. Training completes → orze runs **evaluation** and **post-scripts**
5. Orze updates `report.md` and `status.json`
6. Loop repeats

#### Mode: script (run any Python script)

```yaml
roles:
  my_role:
    mode: script
    script: my_agent.py
    args: ["--ideas-md", "{ideas_file}", "--results-dir", "{results_dir}"]
    timeout: 600
    cooldown: 300
    log_dir: _my_role_logs
    env:
      ANTHROPIC_API_KEY: sk-...
```

Template variables for `args`: `{ideas_file}`, `{results_dir}`, `{cycle}`, `{gpu_count}`, `{completed}`, `{queued}`, `{role_name}`.

#### Mode: claude (use Claude CLI — no API keys needed)

```yaml
roles:
  my_role:
    mode: claude
    rules_file: MY_ROLE_RULES.md
    cooldown: 300
    timeout: 600
    model: sonnet
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
    claude_bin: claude
    claude_args: []
```

In this mode, orze spawns `claude -p "<rules_file content>" --allowedTools ... --model ...` as a subprocess. The `rules_file` supports the same template variables.

### Role Behavior

- **Independent**: Each role has its own cooldown timer, cycle counter, and filesystem lock
- **Non-fatal**: If any role crashes or times out, orze logs a warning and continues. Roles never block training
- **Rate-limited**: Each role's `cooldown` timer prevents excessive runs
- **Logged**: Cycles go to `results/{log_dir}/cycle_NNN.log`
- **Cross-machine safe**: Each role uses its own `_{role_name}_lock` directory for coordination

### Research Rules Contract (mode: claude)

The `rules_file` is a markdown file that serves as Claude's prompt. It should tell Claude:
1. What the research goal is
2. Where to find results (`{results_dir}/report.md`, `{results_dir}/status.json`)
3. How to format ideas (the Ideas Format from this document)
4. Where to append ideas (`{ideas_file}`)
5. What makes a good experiment idea for this domain

Example `RESEARCH_RULES.md`:
```markdown
You are a research agent for [project description].

## Your task
Read `{results_dir}/report.md` for current results. Read `{results_dir}/status.json` for pipeline status.
Generate new experiment ideas and append them to `{ideas_file}`.

## Current state
This is research cycle {cycle}. {completed} experiments completed, {queued} in queue.

## Idea format
Each idea must be an H2 header with YAML config:
## idea-NNN: Title
- **Priority**: high/medium/low
- **Hypothesis**: Why this might work.
\```yaml
model:
  ...
\```
```

### Running Roles Manually

```bash
# Run one research cycle and exit
orze -c orze.yaml --role-only research

# Legacy alias for the above
orze -c orze.yaml --research-only

# Run only the documenter role once
orze -c orze.yaml --role-only documenter

# The full loop (all roles + train + eval, continuous)
orze -c orze.yaml
```

### Multi-Machine Setup
On machines sharing a filesystem:
```bash
# Machine 1 (commander — runs roles + training)
orze -c orze.yaml --gpus 0,1,2,3

# Machine 2 (worker — just runs training, no roles configured)
orze -c orze.yaml --gpus 0,1,2,3
```

Only one machine should have roles configured (the commander). Workers just train whatever ideas are in the queue. The atomic mkdir prevents duplicate claims.

## Notifications

Orze can push updates to Telegram, Slack, Discord, or any webhook endpoint when experiments complete, fail, or set new bests.

### Configuration

```yaml
notifications:
  enabled: true
  on: [completed, failed, new_best]   # events to notify on
  channels:
    - type: telegram
      bot_token: "123456:ABC-DEF"
      chat_id: "-100123456789"
    - type: slack
      webhook_url: "https://hooks.slack.com/services/..."
      on: [new_best, failed]          # per-channel override
    - type: discord
      webhook_url: "https://discord.com/api/webhooks/..."
    - type: webhook
      url: "https://example.com/hook"
      headers:
        Authorization: "Bearer tok"
```

### Events

| Event | When | Message |
|-------|------|---------|
| `completed` | Experiment finishes successfully | Idea ID, title, primary metric, rank, training time |
| `failed` | Experiment fails | Idea ID, title, error message |
| `new_best` | A new experiment displaces the #1 spot | New best ID, metric value, previous best ID |

### Behavior

- **Disabled by default** — set `enabled: true` to activate
- **Per-channel filtering** — each channel can override the global `on` list
- **Non-blocking** — notification failures are logged as warnings, never crash the loop
- **No dependencies** — uses `urllib.request` (stdlib), no `requests` library needed

## Phase Transitions

For projects with distinct research phases, use marker files:

- **Phase 1 (Build)**: Infrastructure setup, smoke test. Create `.phase1_complete` when done.
- **Phase 2 (Explore)**: Broad exploration. Orze runs ideas across all GPUs.
- **Phase 3 (Converge)**: Focus on top approaches. Create `.phase3_started`.

Orze itself doesn't enforce phases — it just runs whatever ideas are in ideas.md. The research agent should check phase markers and adjust its idea generation strategy accordingly. For example, in Phase 3, generate only ideas that build on the top 3 approaches.

## Configuring for Your Task (orze.yaml)

Create `orze.yaml` in your project root to customize behavior:

```yaml
# Required: paths to your scripts and files
train_script: my_train.py
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

# Extra args passed to train script after the standard 4 args
train_extra_args:
  - "--data-dir"
  - "/path/to/data"

# Extra environment variables for subprocesses
train_extra_env:
  TORCH_HOME: /path/to/cache

# Timeouts
timeout: 3600           # max training time (seconds)
poll: 30                # loop sleep (seconds)

# Health monitoring (0 = disabled)
stall_minutes: 30       # kill if no log growth
max_idea_failures: 3    # skip after N failures
min_disk_gb: 20         # pause if disk < 20GB free
orphan_timeout_hours: 6 # reclaim stale claims

# Pre-training hook (optional, runs before each training launch)
pre_script: check_features.py
pre_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]
pre_timeout: 3600

# Post-training evaluation (optional)
eval_script: my_eval.py
eval_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]
eval_timeout: 3600
eval_output: eval_report.json  # checked for skip-if-exists

# Additional post-training scripts (optional, run after eval)
post_scripts:
  - name: overlay
    script: generate_overlay.py
    args: ["--idea-id", "{idea_id}"]
    timeout: 1800
    output: overlay_done.json   # skip if exists

# Garbage collection
cleanup:
  interval: 100               # run every N iterations
  patterns: ["checkpoint_epoch*.pt"]  # delete from result dirs
  script: scripts/cleanup.py  # custom cleanup script
  timeout: 300

# Report configuration
report:
  title: "My Research Report"
  primary_metric: test_accuracy
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "test_loss", label: "Loss", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}
```

### Reading Metrics from Other Files
Columns can read from files other than metrics.json using the `source` field:

```yaml
columns:
  - key: "auc_roc"
    label: "AUC"
    fmt: ".4f"
    source: "eval_report.json:metrics.auc_roc"
```

This reads `results/{idea_id}/eval_report.json` → `metrics` → `auc_roc`.

## Training Script Contract

**Input** (provided by orze):
- `CUDA_VISIBLE_DEVICES` environment variable — which GPU to use
- `--idea-id idea-001` — which idea to train
- `--results-dir results` — where to write output
- `--ideas-md ideas.md` — path to ideas file (read your YAML config from here)
- `--config configs/base.yaml` — path to base config
- Any additional args from `train_extra_args`
- Any additional env vars from `train_extra_env`

**Output** (required from your script):
- `results/{idea_id}/metrics.json` with at minimum `{"status": "COMPLETED"}` or `{"status": "FAILED", "error": "..."}`

**That's it.** Write metrics.json when done. Orze handles everything else.

## Pre-Script Contract (Optional)

If `pre_script` is configured, it runs before each training launch on the claimed GPU.

**Input**: The command from `pre_args` with `{idea_id}` and `{gpu}` substituted, plus `CUDA_VISIBLE_DEVICES`
**Success**: Exit code 0 — training proceeds
**Failure**: Non-zero exit code — idea marked FAILED, training skipped

Use cases: verify features exist, check disk space, validate configs.

## Evaluation Script Contract (Optional)

If `eval_script` is configured, it runs after each successful training.

**Input**: The command from `eval_args` with `{idea_id}` and `{gpu}` substituted
**Output**: The file named in `eval_output` (default: `eval_report.json`)
**Skip**: If the output file already exists, eval is skipped

## Post-Scripts Contract (Optional)

Additional scripts in `post_scripts` list run after eval. Each entry specifies:
- `script`: path to the script
- `args`: list of args with `{idea_id}` and `{gpu}` substitution
- `timeout`: max time in seconds
- `output`: if this file exists, the script is skipped
- `name`: label for logs

Use cases: overlay generation, model export, additional analysis.

## Best Practices for Writing Ideas

1. **Start with baselines** — simple models first, then iterate
2. **Vary one thing at a time** — easier to attribute improvements
3. **Use priority** — mark promising directions as `high`, speculative as `low`
4. **Include hypotheses** — helps interpret results and plan next ideas
5. **Check the leaderboard** before generating similar ideas — avoid redundant work
6. **Use categories** — group related ideas for easier analysis
7. **Track lineage** — set `Parent: idea-XXX` to trace what inspired each idea
8. **Keep YAML configs complete** — don't rely on implicit defaults
9. **React to results** — combine winners, diagnose failures, push best approaches further
10. **Span diverse categories** — architecture, training, data, loss, ensemble — not just hyperparameter tweaks

## CLI Quick Reference

```
orze [OPTIONS]

Options:
  -c, --config-file PATH   Path to orze.yaml
  --gpus GPU_IDS           Comma-separated GPU IDs (default: auto-detect)
  --timeout SECONDS        Max training time (default: 3600)
  --poll SECONDS           Loop sleep interval (default: 30)
  --once                   Run one cycle and exit
  --report-only            Only regenerate report.md
  --role-only NAME         Run a single agent role once and exit
  --research-only          Alias for --role-only research
  --ideas-md PATH          Ideas file path
  --base-config PATH       Base config YAML path
  --results-dir PATH       Results directory
  --train-script PATH      Training script
  -v, --verbose            Debug logging
```

CLI args override orze.yaml values.
