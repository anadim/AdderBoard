"""Reusable guards for orze experiment pipelines.

Project-level override of orze-pro's orze_guards.py.
Fix: sort cycle_*.log files by modification time instead of lexicographically.
The original alphabetical sort breaks when cycle numbers wrap at 1000
(e.g., cycle_999.log sorts after cycle_53.log even though cycle 2053 is newer).
"""

import json
import time
from pathlib import Path

from fsm.engine import guard, Context


def _sort_logs_by_mtime(log_dir: Path, pattern: str = "cycle_*.log") -> list[Path]:
    """Sort log files by modification time (oldest first)."""
    logs = list(log_dir.glob(pattern))
    logs.sort(key=lambda p: p.stat().st_mtime)
    return logs


def _get_best_accuracy(results_dir: Path) -> tuple:
    """Return (best_accuracy, total_completed)."""
    best = 0.0
    completed = 0
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("status") != "COMPLETED":
                continue
            completed += 1
            acc = m.get("accuracy", 0.0)
            if acc > best:
                best = acc
        except Exception:
            continue
    return best, completed


def _get_status(results_dir: Path) -> dict:
    """Read status.json."""
    path = results_dir / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@guard("plateau_detected")
def plateau_detected(ctx: Context) -> str | None:
    """True when best metric hasn't improved in `plateau_window` completions."""
    best, completed = _get_best_accuracy(ctx.results_dir)
    window = ctx.vars.get("plateau_window", 20)

    # Track best in vars
    if best > ctx.vars.get("best_accuracy", 0.0):
        ctx.vars["best_accuracy"] = best
        ctx.vars["best_at_completion"] = completed

    best_at = ctx.vars.get("best_at_completion", 0)
    if completed < window:
        return None
    if (completed - best_at) >= window:
        return f"no improvement in {completed - best_at} experiments (best={best:.4f})"
    return None


@guard("has_improvement")
def has_improvement(ctx: Context) -> str | None:
    """True when accuracy improved over recorded best."""
    best, _ = _get_best_accuracy(ctx.results_dir)
    prev = ctx.vars.get("best_accuracy", 0.0)
    if best > prev:
        return f"accuracy improved: {prev:.4f} → {best:.4f}"
    return None


@guard("attempts_exhausted")
def attempts_exhausted(ctx: Context) -> str | None:
    """True when evolution attempts >= max_attempts."""
    attempts = ctx.vars.get("evolution_attempts", 0)
    max_att = ctx.vars.get("max_attempts", 3)
    if attempts >= max_att:
        return f"exhausted {attempts}/{max_att} evolution attempts"
    return None


@guard("attempts_remaining")
def attempts_remaining(ctx: Context) -> str | None:
    """True when evolution attempts < max_attempts."""
    attempts = ctx.vars.get("evolution_attempts", 0)
    max_att = ctx.vars.get("max_attempts", 3)
    if attempts < max_att:
        return f"attempt {attempts + 1}/{max_att}"
    return None


@guard("trigger_consumed")
def trigger_consumed(ctx: Context) -> str | None:
    """True when the trigger file no longer exists (evolution finished)."""
    trigger_name = ctx.vars.get("trigger_file", "_trigger_code_evolution")
    trigger = ctx.results_dir / trigger_name
    if not trigger.exists():
        return "trigger consumed (evolution cycle completed)"
    return None


@guard("is_deadlocked")
def is_deadlocked(ctx: Context) -> str | None:
    """True when the system is stuck — no progress being made.

    Detects TWO deadlock patterns:

    1. Classic: paused + 0 active + 0 queued for >5 min
       (nothing running, nothing to run, nothing generating new work)

    2. Stale pause: paused + no accuracy improvement for >30 min
       (experiments running but pause is blocking research from generating
       new ideas, and nothing is improving — the queue will drain to zero
       eventually, wasting remaining GPU time on stale ideas)

    Pattern 2 catches the bug where evolution exhausts attempts, pauses
    research, but the queue still has ideas. Those ideas all finish with
    no improvement, yet the pause holds because queue > 0.
    """
    import datetime

    # How long have we been in the current state?
    history = ctx.state.get("history", [])
    if not history:
        return None
    last_time = history[-1].get("time", "")
    try:
        entered = datetime.datetime.fromisoformat(last_time)
        elapsed_min = (datetime.datetime.now() - entered).total_seconds() / 60
    except Exception:
        return None

    status = _get_status(ctx.results_dir)
    active = len(status.get("active", []))
    queued = status.get("queue_depth", 0)

    # Pattern 1: Classic deadlock — nothing happening at all
    if active == 0 and queued == 0 and elapsed_min > 5:
        return f"deadlocked {elapsed_min:.0f}min: 0 active, 0 queued"

    # Pattern 2: Stale pause — paused too long without improvement
    stale_min = ctx.vars.get("stale_pause_min", 30)
    if elapsed_min > stale_min:
        # Check if accuracy improved since we entered this state
        best, _ = _get_best_accuracy(ctx.results_dir)
        recorded_best = ctx.vars.get("best_accuracy", 0.0)
        if best <= recorded_best:
            return (f"stale pause {elapsed_min:.0f}min: no improvement "
                    f"(best={best:.4f}, was {recorded_best:.4f}), "
                    f"{active} active, {queued} queued")

    return None


@guard("queue_low")
def queue_low(ctx: Context) -> str | None:
    """True when queue depth < threshold."""
    status = _get_status(ctx.results_dir)
    queued = status.get("queue_depth", 0)
    threshold = ctx.vars.get("queue_low_threshold", 5)
    if queued < threshold:
        return f"queue low: {queued} < {threshold}"
    return None


@guard("failure_rate_high")
def failure_rate_high(ctx: Context) -> str | None:
    """True when recent failure rate exceeds threshold."""
    window = ctx.vars.get("fail_window", 10)
    threshold = ctx.vars.get("fail_threshold", 0.5)
    results_dir = ctx.results_dir

    recent = []
    for d in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            recent.append(m.get("status", "UNKNOWN"))
            if len(recent) >= window:
                break
        except Exception:
            continue

    if len(recent) < window:
        return None
    fail_rate = sum(1 for s in recent if s == "FAILED") / len(recent)
    if fail_rate >= threshold:
        return f"failure rate {fail_rate:.0%} >= {threshold:.0%} (last {window})"
    return None


@guard("always")
def always(ctx: Context) -> str | None:
    """Always passes. Use for unconditional transitions with actions."""
    return "unconditional"


@guard("family_imbalanced")
def family_imbalanced(ctx: Context) -> str | None:
    """True when approach family distribution is heavily skewed.

    Checks idea_lake.db for completed experiments. If any single family
    accounts for more than imbalance_threshold (default 60%) of completed
    ideas and total completed >= min_for_meta (default 30), returns a reason.
    """
    import sqlite3

    threshold = ctx.vars.get("imbalance_threshold", 0.6)
    min_completed = ctx.vars.get("min_for_meta", 30)
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT approach_family, COUNT(*) as cnt FROM ideas "
            "WHERE status = 'completed' GROUP BY approach_family"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    if not rows:
        return None
    total = sum(r[1] for r in rows)
    if total < min_completed:
        return None

    for family, count in rows:
        share = count / total
        if share >= threshold:
            return (f"family imbalance: {family or 'other'} has "
                    f"{share:.0%} of {total} completed ideas")
    return None


@guard("meta_cooldown_elapsed")
def meta_cooldown_elapsed(ctx: Context) -> str | None:
    """True when enough time has passed since the last meta-research run.

    Uses vars['last_meta_time'] (epoch) and vars['meta_cooldown_sec']
    (default 3600 = 1 hour).
    """
    cooldown = ctx.vars.get("meta_cooldown_sec", 3600)
    last = ctx.vars.get("last_meta_time", 0)
    elapsed = time.time() - last
    if elapsed >= cooldown:
        return f"meta cooldown elapsed ({elapsed / 60:.0f}min >= {cooldown / 60:.0f}min)"
    return None


@guard("bug_fixer_fixes_exhausted")
def bug_fixer_fixes_exhausted(ctx: Context) -> str | None:
    """True when bug fixer has hit the max fixes per hour limit."""
    max_fixes = ctx.vars.get("max_fixes_per_hour", 3)
    fix_count = ctx.vars.get("fixes_this_hour", 0)
    if fix_count >= max_fixes:
        return f"bug fixer exhausted: {fix_count}/{max_fixes} fixes this hour"
    return None


@guard("recent_fix_succeeded")
def recent_fix_succeeded(ctx: Context) -> str | None:
    """True when the most recent bug fix resolved the issue.

    Checks bug_fixer_issues/ for the most recent response file and
    sees if failure_rate has decreased since the fix.
    """
    issues_dir = ctx.results_dir / "bug_fixer_issues"
    if not issues_dir.exists():
        return None

    # Check if failure rate has decreased since last fix
    prev_fail_rate = ctx.vars.get("pre_fix_fail_rate", None)
    if prev_fail_rate is None:
        return None

    # Compute current failure rate
    window = ctx.vars.get("fail_window", 10)
    results_dir = ctx.results_dir
    recent = []
    for d in sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            recent.append(m.get("status", "UNKNOWN"))
            if len(recent) >= window:
                break
        except Exception:
            continue

    if len(recent) < window:
        return None
    current_rate = sum(1 for s in recent if s == "FAILED") / len(recent)
    if current_rate < prev_fail_rate:
        return f"fix succeeded: failure rate {prev_fail_rate:.0%} -> {current_rate:.0%}"
    return None


@guard("meta_trigger_consumed")
def meta_trigger_consumed(ctx: Context) -> str | None:
    """True when _trigger_meta_research file has been consumed by the role."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    if not trigger.exists():
        return "meta_research trigger consumed"
    return None


@guard("meta_trigger_stale")
def meta_trigger_stale(ctx: Context) -> str | None:
    """True when _trigger_meta_research has sat unconsumed for >15 minutes."""
    trigger = ctx.results_dir / "_trigger_meta_research"
    if not trigger.exists():
        return None
    try:
        age_min = (time.time() - trigger.stat().st_mtime) / 60
        if age_min > 15:
            return f"meta trigger stale ({age_min:.0f}min unconsumed)"
    except Exception:
        pass
    return None


@guard("research_stalled")
def research_stalled(ctx: Context) -> str | None:
    """True when research agent has produced zero new ideas for N consecutive cycles.

    Reads _research_logs/ to count recent cycles with zero output.
    """
    threshold = ctx.vars.get("stall_threshold", 10)
    log_dir = ctx.results_dir / "_research_logs"
    if not log_dir.exists():
        return None

    # Sort by mtime (not filename) to handle cycle number wrapping at 1000
    logs = _sort_logs_by_mtime(log_dir)
    logs.reverse()  # most recent first
    zero_count = 0
    for log_file in logs[:threshold + 5]:
        try:
            content = log_file.read_text(encoding="utf-8")
            # Check if the cycle produced any ideas
            if "new ideas" not in content.lower() and "ingested" not in content.lower():
                zero_count += 1
            else:
                break  # found a productive cycle, stop counting
        except Exception:
            continue

    ctx.vars["zero_output_count"] = zero_count
    if zero_count >= threshold:
        return f"research stalled: {zero_count} consecutive cycles with zero output"
    return None


@guard("role_unhealthy")
def role_unhealthy(ctx: Context) -> str | None:
    """True when any role has a VERIFIED current problem.

    Zero false positives: every issue is confirmed by checking the actual
    current state, not just log history. A problem must be reproducible
    RIGHT NOW to trigger.

    Checks (all verified against live state):
    1. Config file missing: rules_file in orze.yaml doesn't exist on disk
    2. Role stalled: last cycle log is old AND no process running
    3. Consecutive failures: role's last N cycle logs are empty/error
    4. Train script broken: syntax error in train.py
    5. API key missing: .env doesn't have required keys
    """
    import re, datetime, subprocess, yaml

    unhealthy = []

    # --- 1. Config file check: do all rules_files actually exist? ---
    orze_yaml = ctx.results_dir.parent / "orze.yaml"
    if orze_yaml.exists():
        try:
            cfg = yaml.safe_load(orze_yaml.read_text(encoding="utf-8"))
            for role_name, role_cfg in (cfg.get("roles") or {}).items():
                if not isinstance(role_cfg, dict):
                    continue
                rules_file = role_cfg.get("rules_file")
                if rules_file and not Path(rules_file).exists():
                    # Confirmed: file doesn't exist right now
                    unhealthy.append(f"{role_name}: rules_file '{rules_file}' not found")
        except Exception:
            pass

    # --- 2. Role stall check: last cycle log too old + no process ---
    grace_min = ctx.vars.get("role_grace_min", 15)
    monitored = ctx.vars.get("monitored_roles",
                             ["research", "professor", "fsm"])

    for role_name in monitored:
        log_dir = ctx.results_dir / f"_{role_name}_logs"
        if not log_dir.exists():
            continue
        # Sort by mtime to handle cycle number wrapping at 1000
        logs = _sort_logs_by_mtime(log_dir)
        if not logs:
            continue
        last_log = logs[-1]
        try:
            age_min = (time.time() - last_log.stat().st_mtime) / 60
            if age_min > grace_min:
                # Log is old — but is the role actually stuck or just on cooldown?
                # Verify: check if a process is running for this role
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", f"Running {role_name}"],
                        capture_output=True, timeout=5)
                    process_running = result.returncode == 0
                except Exception:
                    process_running = False

                if not process_running:
                    # Check the last log content — empty = role didn't produce output
                    content = last_log.read_text(encoding="utf-8").strip()
                    if not content:
                        unhealthy.append(
                            f"{role_name}: stalled {age_min:.0f}min, last log empty")
                    elif age_min > grace_min * 3:
                        # Very stale — something is wrong regardless of content
                        unhealthy.append(
                            f"{role_name}: no activity for {age_min:.0f}min")
        except Exception:
            continue

    # --- 3. Consecutive empty logs (role runs but produces nothing) ---
    for role_name in monitored:
        log_dir = ctx.results_dir / f"_{role_name}_logs"
        if not log_dir.exists():
            continue
        # Sort by mtime to handle cycle number wrapping at 1000
        logs = _sort_logs_by_mtime(log_dir)[-5:]  # last 5
        if len(logs) < 3:
            continue
        empty_count = sum(1 for l in logs if l.stat().st_size == 0)
        if empty_count >= 3:
            # Already flagged above? Skip duplicate
            if not any(role_name in u for u in unhealthy):
                unhealthy.append(
                    f"{role_name}: {empty_count}/{len(logs)} recent cycles empty")

    # --- 4. Train script syntax check ---
    train_script = ctx.results_dir.parent / "train.py"
    if train_script.exists():
        try:
            import ast
            ast.parse(train_script.read_text(encoding="utf-8"))
        except SyntaxError as e:
            unhealthy.append(f"train.py: SyntaxError at line {e.lineno}")

    # --- 5. API key check ---
    env_file = ctx.results_dir.parent / ".env"
    if env_file.exists():
        env_content = env_file.read_text(encoding="utf-8")
        if "ANTHROPIC_API_KEY" not in env_content and "GEMINI_API_KEY" not in env_content:
            unhealthy.append("no API keys in .env (need ANTHROPIC or GEMINI)")
    else:
        # Check env vars directly
        import os
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            unhealthy.append("no API keys found (env or .env)")

    if unhealthy:
        ctx.vars["unhealthy_roles"] = unhealthy
        return f"{len(unhealthy)} issue(s): {'; '.join(unhealthy)}"
    return None


@guard("roles_healthy")
def roles_healthy(ctx: Context) -> str | None:
    """True when all checks pass (inverse of role_unhealthy)."""
    result = role_unhealthy(ctx)
    if result is None:
        return "all checks pass"
    return None


@guard("fix_trigger_consumed")
def fix_trigger_consumed(ctx: Context) -> str | None:
    """True when the bug_fixer trigger file has been consumed."""
    trigger_name = ctx.vars.get("trigger_file", "_trigger_bug_fixer")
    trigger = ctx.results_dir / trigger_name
    if not trigger.exists():
        return "bug_fixer trigger consumed"
    return None


@guard("fix_trigger_stale")
def fix_trigger_stale(ctx: Context) -> str | None:
    """True when the bug_fixer trigger has sat unconsumed for >20min."""
    trigger_name = ctx.vars.get("trigger_file", "_trigger_bug_fixer")
    trigger = ctx.results_dir / trigger_name
    if not trigger.exists():
        return None
    try:
        age_min = (time.time() - trigger.stat().st_mtime) / 60
        if age_min > 20:
            return f"bug_fixer trigger stale ({age_min:.0f}min unconsumed)"
    except Exception:
        pass
    return None
