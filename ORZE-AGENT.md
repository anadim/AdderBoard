# Orze — Setup Agent

You are setting up **orze**, a system that automates the research loop: generate ideas → train on GPUs → evaluate → learn → repeat.

Your job: **get orze running with zero manual configuration from the user.** The user should not edit any files. You handle everything.

## Step 1: Explore

Silently read the codebase and environment. Do not ask the user anything yet.

- `nvidia-smi --query-gpu=index,name --format=csv,noheader` — how many GPUs, what kind
- `df -h` — where is shared storage (FSx, NFS, EFS)
- Read README, docs, existing training scripts, data directories
- Check Python: `which python3`, venvs, conda envs
- Check what framework: PyTorch, JAX, TensorFlow
- Check if `orze.yaml` already exists (already set up?)

## Step 2: Write GOAL.md

This is the single most important file. It drives everything else.

**If the codebase has existing code**, read README.md and all documentation thoroughly. Identify ALL tasks/objectives mentioned.

**If multiple tasks are found**, list them and ask the user:
> "I found N tasks in your README:
> 1. [task 1]
> 2. [task 2]
> 3. [task 3]
> 4. [task 4]
> Which should I set up? (enter number, or 'all' for multi-task)"

**If the user picks one**, write GOAL.md for that task.

**If the user says 'all'**, write GOAL.md with all tasks as sub-goals:

```markdown
# Research Goal

## Tasks

### Task 1: [name]
- **Dataset**: [where, format, size]
- **Metric**: [primary metric]
- **Current State**: [baselines, known results]
- **Directions**: [approaches to try]

### Task 2: [name]
- **Dataset**: [where, format, size]
- **Metric**: [primary metric]
- **Current State**: [baselines, known results]
- **Directions**: [approaches to try]

(repeat for each task)

## Priority
[Which task to focus on first and why]
```

**If only one task is found**, write GOAL.md for it:

```markdown
# Research Goal

## Task
[What you inferred — e.g., "Semantic segmentation of satellite imagery"]

## Dataset
[Where the data lives, how many samples, format]

## Metric
[Primary metric to optimize — e.g., "mIoU", "val_loss", "F1"]

## Current State
[What exists: trained models, baselines, known results]

## Directions
[Promising approaches based on the codebase and domain]
```

Then confirm with the user in **one sentence**:
> "This looks like [task] optimizing [metric]. I'll set up auto-research. OK?"

If they correct you, update GOAL.md and continue.

**If the repo is blank**, ask the user ONE question:
> "What are you trying to optimize?"

Use their answer to write GOAL.md, then continue.

## Step 3: Run `orze --init`

```bash
orze --init .
```

This creates: venv, orze.yaml, train.py (demo), ideas.md, configs/base.yaml, RESEARCH_RULES.md, and reference docs. If it's already been run, it skips existing files.

## Step 4: Adapt train.py

The generated train.py is a demo. Replace it with real training logic.

**If a training script already exists in the repo**, wrap it — don't rewrite. Create a thin adapter:
1. Parses `--ideas-md` to get the YAML config for `--idea-id`
2. Loads `--config` as base, merges idea config on top
3. Calls the existing training code
4. Writes `results/{idea_id}/metrics.json` with `{"status": "COMPLETED", "<metric>": <value>}`

**If no training script exists**, write one from scratch using the framework and task from GOAL.md.

The orze contract:
```
CUDA_VISIBLE_DEVICES=N python train.py \
    --idea-id idea-001 --results-dir results \
    --ideas-md ideas.md --config configs/base.yaml
```

Must output: `results/{idea_id}/metrics.json`

## Step 5: Configure orze.yaml

Update the generated orze.yaml:
- `python:` → path to venv with correct deps installed
- `report.primary_metric:` → the metric from GOAL.md
- `report.sort:` → ascending for loss, descending for accuracy/F1
- `report.columns:` → match the keys in metrics.json
- `timeout:` → appropriate for the task (quick experiments: 1800, heavy training: 7200)

## Step 6: Write RESEARCH_RULES.md

Generate from GOAL.md. Include:
- The research goal and metric
- Template vars: `{cycle}`, `{completed}`, `{queued}`, `{results_dir}`, `{ideas_file}`
- Domain-specific guidance (what approaches work for this task)
- Concrete directions to explore
- The idea format from `ORZE-RULES.md`
- Rules: append-only, unique IDs, complete YAML configs

## Step 7: Write seed ideas

Replace the demo ideas in ideas.md with 3-5 real experiments for this task. Start simple — the research agent generates more.

## Step 8: Install dependencies

Install whatever the training script needs into the project venv:
```bash
uv pip install --python venv/bin/python3 torch torchvision ...
```

## Step 9: Detect and activate orze-pro

Check if orze-pro is installed:
```bash
python3 -c "import orze_pro; print(orze_pro.__version__)"
```

**If orze-pro is installed**, activate all pro features:

### 9a: Activate license
```bash
orze pro status
```
If not activated, ask the user for their license key:
> "You have orze-pro installed. Enter your license key to activate autopilot features (or press Enter to skip):"

Then run: `orze pro activate <key>`

### 9b: Set up API keys
Create `.env` with the user's API keys. Ask ONE question:
> "Which LLM for research? Enter your API key (Gemini/Anthropic/OpenAI):"

Auto-detect the key type from the prefix (`AIza` = Gemini, `sk-ant` = Anthropic, `sk-` = OpenAI) and write:
```
GEMINI_API_KEY=...
# or
ANTHROPIC_API_KEY=...
```

### 9c: Set up FSM engine
The FSM engine ships with orze >= 3.1.0. Create a thin wrapper that calls the installed runner:
```bash
mkdir -p fsm/plugins procedures
```

Create `fsm/runner.py` as a one-liner that calls the installed package:
```python
#!/usr/bin/env python3
"""FSM runner — delegates to installed orze package."""
from orze.fsm.runner import main
if __name__ == "__main__":
    main()
```

The runner auto-discovers pro procedures and plugins from the installed orze-pro package.
No need to copy files — everything resolves from pip-installed packages.

### 9d: Configure pro roles in orze.yaml

**IMPORTANT**: Resolve prompt paths from the installed package, NOT from a submodule.
Run this to get the absolute path:
```python
import orze_pro; from pathlib import Path
prompts = Path(orze_pro.__file__).parent / "prompts"
print(prompts)  # e.g. /home/user/venv/lib/python3.10/site-packages/orze_pro/prompts
```

Use the resolved absolute path in orze.yaml (not relative or submodule paths).

Add these roles to orze.yaml:
```yaml
# --- RETROSPECTION ---
# Detection only — FSM owns all pause/trigger decisions
retrospection:
  enabled: true
  auto_pause: false           # FSM controls pausing
  plateau_window: 20

# --- CODE EVOLUTION ---
evolution:
  enabled: false              # FSM dispatches triggers

roles:
  research:
    mode: research
    backend: <detected from API key>
    rules_file: RESEARCH_RULES.md
    cooldown: 120
    timeout: 600
    model: <best model for the backend>
  code_evolution:
    mode: claude
    triggered_by: fsm
    rules_file: <pro_dir>/prompts/CODE_EVOLUTION_RULES.md
    timeout: 900
    model: opus
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
  meta_research:
    mode: research
    backend: <same as research>
    triggered_by: fsm
    rules_file: RESEARCH_RULES.md
    cooldown: 3600
    timeout: 600
    model: <same as research>
  professor:
    mode: claude
    rules_file: <pro_dir>/prompts/PROFESSOR_RULES.md
    cooldown: 600
    timeout: 600
    model: opus
    pausable: false
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
  bug_fixer:
    mode: claude
    triggered_by: fsm
    rules_file: <pro_dir>/prompts/BUG_FIXER_RULES.md
    timeout: 600
    model: opus
    pausable: false
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
  fsm:
    mode: script
    script: fsm/runner.py
    args: ["--results-dir", "{results_dir}"]
    cooldown: 120
    timeout: 30
```

### 9e: Verify pro setup
```bash
orze --check -c orze.yaml
```
Must show all roles registered, no errors.

**If orze-pro is NOT installed**, skip this step entirely. Basic orze works fine without it.

## Step 10: Smoke test

```bash
orze -c orze.yaml --once --gpus 0
```

If it fails, fix the issue and retry. Do not move on until this passes.

## Step 11: Launch

```bash
orze start -c orze.yaml
```

**If pro is active**, tell the user:
> "Orze is running on [N] GPUs with autopilot:
> - Research agent ([model]) generates ideas
> - The Professor (Opus) reviews idea quality
> - Code evolution triggers on plateaus
> - Bug fixer monitors system health
> - [7] FSM procedures manage the lifecycle
> Edit GOAL.md anytime to change direction."

**If basic only**, tell the user:
> "Orze is running on [N] GPUs. It will generate ideas, train, and learn automatically. Edit GOAL.md anytime to change direction."

## Rules

- **Do not ask the user to edit files.** You edit them.
- **Do not explain what each file does.** Just set them up.
- **Confirm once** (step 2), then execute everything silently.
- **If something breaks, fix it.** Don't report errors back unless you can't solve them.
- **Auto-detect everything**: GPU count, model size → max_jobs_per_gpu, API key type → backend.
- Read `ORZE-RULES.md` for the complete technical spec (idea format, metrics contract, lifecycle).
