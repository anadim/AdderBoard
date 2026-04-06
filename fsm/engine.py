"""Generic finite state machine engine for orze procedures.

Define procedures as YAML. Guards and actions are Python functions
registered via @guard and @action decorators.

Usage:
    from fsm.engine import FSM, guard, action

    @guard("is_plateau")
    def is_plateau(ctx):
        return "no improvement in 20 experiments" if stuck else None

    @action("trigger_evolution")
    def trigger_evolution(ctx):
        write_trigger_file(ctx.results_dir)

    fsm = FSM.from_yaml("procedures/evolution.yaml", results_dir)
    fsm.step()  # run one tick
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("fsm")

# --- JSONL Activity Log ---

_activity_log_path: Optional[Path] = None


def set_activity_log(path: Path):
    """Set the path for the JSONL activity log."""
    global _activity_log_path
    _activity_log_path = path


def _log_event(event: dict):
    """Append a structured event to the JSONL activity log."""
    if _activity_log_path is None:
        return
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with open(_activity_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


# --- Registry ---

_guards: Dict[str, Callable] = {}
_actions: Dict[str, Callable] = {}


def guard(name: str):
    """Register a guard function. Must return reason string or None."""
    def decorator(fn):
        _guards[name] = fn
        return fn
    return decorator


def action(name: str):
    """Register an action function."""
    def decorator(fn):
        _actions[name] = fn
        return fn
    return decorator


def get_guard(name: str) -> Callable:
    if name not in _guards:
        raise KeyError(f"Guard '{name}' not registered. Available: {list(_guards.keys())}")
    return _guards[name]


def get_action(name: str) -> Callable:
    if name not in _actions:
        raise KeyError(f"Action '{name}' not registered. Available: {list(_actions.keys())}")
    return _actions[name]


# --- Context ---

@dataclass
class Context:
    """Passed to every guard and action. Mutable bag of state."""
    results_dir: Path
    state: dict          # persisted FSM state (read/write)
    vars: dict           # user-defined variables from state file
    reason: str = ""     # set by the guard that triggered the transition
    extras: dict = field(default_factory=dict)  # scratch space for actions


# --- Transition ---

@dataclass
class Transition:
    to: str
    guards: List[str]           # ALL must pass (return non-None)
    actions: List[str] = field(default_factory=list)
    mode: str = "all"           # "all" = all guards must pass, "any" = first match wins


# --- State node ---

@dataclass
class StateNode:
    name: str
    on_enter: List[str] = field(default_factory=list)   # actions on entering this state
    on_exit: List[str] = field(default_factory=list)     # actions on leaving this state
    maintain: List[str] = field(default_factory=list)    # actions run every tick while in this state
    transitions: List[Transition] = field(default_factory=list)


# --- FSM ---

class FSM:
    def __init__(self, name: str, states: Dict[str, StateNode],
                 initial: str, results_dir: Path,
                 vars_defaults: Optional[dict] = None):
        self.name = name
        self.states = states
        self.initial = initial
        self.results_dir = results_dir
        self.state_file = results_dir / f"_fsm_{name}.json"
        self.vars_defaults = vars_defaults or {}

    @classmethod
    def from_yaml(cls, yaml_path: str, results_dir: Path) -> "FSM":
        """Load FSM definition from YAML file."""
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        name = spec["name"]
        initial = spec["initial"]
        vars_defaults = spec.get("vars", {})

        states = {}
        for state_spec in spec["states"]:
            sname = state_spec["name"]
            transitions = []
            for t in state_spec.get("transitions", []):
                transitions.append(Transition(
                    to=t["to"],
                    guards=t.get("guards", []),
                    actions=t.get("actions", []),
                    mode=t.get("mode", "all"),
                ))
            states[sname] = StateNode(
                name=sname,
                on_enter=state_spec.get("on_enter", []),
                on_exit=state_spec.get("on_exit", []),
                maintain=state_spec.get("maintain", []),
                transitions=transitions,
            )

        return cls(name, states, initial, results_dir, vars_defaults)

    def load(self) -> dict:
        """Load persisted state."""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "current": self.initial,
            "vars": dict(self.vars_defaults),
            "history": [],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def save(self, data: dict) -> None:
        self.state_file.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8")

    def step(self) -> Optional[str]:
        """Run one FSM tick. Returns new state name if transitioned, None otherwise."""
        data = self.load()
        current = data.get("current", self.initial)

        if current not in self.states:
            logger.error("[%s] Unknown state: %s", self.name, current)
            return None

        node = self.states[current]
        data["_fsm_name"] = self.name  # so actions know which FSM is calling
        ctx = Context(
            results_dir=self.results_dir,
            state=data,
            vars=data.setdefault("vars", dict(self.vars_defaults)),
        )

        # Run maintain actions (re-assert persistent side effects every tick)
        for action_name in node.maintain:
            self._run_action(action_name, ctx)

        # Evaluate transitions in order (first match wins)
        for transition in node.transitions:
            reason = self._eval_guards(transition, ctx)
            if reason is not None:
                ctx.reason = reason

                _log_event({
                    "event": "transition",
                    "fsm": self.name,
                    "from": current,
                    "to": transition.to,
                    "reason": reason,
                    "guards": transition.guards,
                    "actions": transition.actions,
                })

                # on_exit actions
                for action_name in node.on_exit:
                    self._run_action(action_name, ctx)

                # transition actions
                for action_name in transition.actions:
                    self._run_action(action_name, ctx)

                # Record transition
                entry = {
                    "from": current,
                    "to": transition.to,
                    "reason": reason,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                data["history"] = data.get("history", [])[-49:]  # keep last 50
                data["history"].append(entry)
                data["current"] = transition.to
                logger.info("[%s] %s → %s (%s)", self.name, current, transition.to, reason)

                # on_enter actions of new state
                new_node = self.states.get(transition.to)
                if new_node:
                    for action_name in new_node.on_enter:
                        self._run_action(action_name, ctx)

                self.save(data)
                return transition.to

        # No transition fired — log heartbeat periodically
        _log_event({
            "event": "tick",
            "fsm": self.name,
            "state": current,
            "vars_snapshot": {k: v for k, v in ctx.vars.items()
                             if k in ("best_accuracy", "evolution_attempts",
                                      "total_filtered", "zero_output_count",
                                      "fixes_this_hour")},
        })
        self.save(data)
        return None

    def _eval_guards(self, transition: Transition, ctx: Context) -> Optional[str]:
        """Evaluate guards. Returns combined reason or None.

        Guard names can be prefixed with "!" to negate:
            guards: ["!failure_rate_high"]  →  passes when failure_rate_high returns None
        """
        if not transition.guards:
            return "unconditional"

        reasons = []
        for guard_name in transition.guards:
            negate = guard_name.startswith("!")
            actual_name = guard_name[1:] if negate else guard_name
            fn = get_guard(actual_name)
            result = fn(ctx)

            if negate:
                if result is None:
                    reason_str = f"not {actual_name}"
                    reasons.append(reason_str)
                    if transition.mode == "any":
                        return reason_str
                else:
                    if transition.mode == "all":
                        return None
            else:
                if result is None:
                    if transition.mode == "all":
                        return None
                else:
                    reasons.append(result)
                    if transition.mode == "any":
                        return result

        if transition.mode == "all" and len(reasons) == len(transition.guards):
            return "; ".join(reasons)
        return None

    def _run_action(self, action_name: str, ctx: Context) -> None:
        try:
            fn = get_action(action_name)
            fn(ctx)
            _log_event({
                "event": "action",
                "fsm": self.name,
                "action": action_name,
                "state": ctx.state.get("current", "?"),
                "ok": True,
            })
        except Exception as e:
            logger.error("[%s] Action '%s' failed: %s", self.name, action_name, e)
            _log_event({
                "event": "action_error",
                "fsm": self.name,
                "action": action_name,
                "state": ctx.state.get("current", "?"),
                "error": str(e),
                "ok": False,
            })

    def status(self) -> dict:
        """Return current state summary for display."""
        data = self.load()
        return {
            "fsm": self.name,
            "state": data.get("current", self.initial),
            "vars": data.get("vars", {}),
            "transitions": len(data.get("history", [])),
            "last_transition": data["history"][-1] if data.get("history") else None,
        }
