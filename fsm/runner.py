#!/usr/bin/env python3
"""FSM runner: discovers and steps all procedures.

Loads from two tiers:
  1. orze-pro package: built-in procedures & plugins (pro features)
  2. Project-level: user overrides in procedures/ and fsm/plugins/

Pro-tier files live in orze-pro/src/orze_pro/{fsm/plugins,procedures,prompts}.
Project-level files can override or extend pro defaults.

Run as an orze script role:
    roles:
      fsm:
        mode: script
        script: fsm/runner.py
        args: ["--results-dir", "{results_dir}"]
        cooldown: 120
        timeout: 30
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FSM] %(levelname)s %(message)s",
)
logger = logging.getLogger("fsm")


def _find_pro_package() -> Path | None:
    """Locate the orze-pro package's fsm directory."""
    # Try installed package first
    try:
        import orze_pro
        pkg_dir = Path(orze_pro.__file__).parent
        if (pkg_dir / "fsm").exists():
            return pkg_dir
    except ImportError:
        pass

    # Try local submodule
    submodule = Path(_project_root) / "orze-pro" / "src" / "orze_pro"
    if submodule.exists():
        return submodule

    return None


def _load_plugins(dirs: list[Path]):
    """Auto-discover and import plugins from multiple directories.

    Project-level plugins (first in dirs) override pro plugins by filename.
    All plugins are loaded via importlib.util to avoid module path conflicts.
    """
    seen = set()
    for plugins_dir in dirs:
        if not plugins_dir.exists():
            continue
        for py_file in sorted(plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            if py_file.name in seen:
                logger.debug("Plugin %s already loaded (project override)", py_file.name)
                continue
            seen.add(py_file.name)
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_fsm_plugin_{py_file.stem}", str(py_file))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                logger.debug("Loaded plugin: %s from %s", py_file.stem, plugins_dir)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", py_file, e)


def _load_procedures(dirs: list[Path], results_dir: Path) -> list:
    """Load procedure YAMLs from multiple directories. Project overrides pro."""
    from fsm.engine import FSM

    seen = set()
    fsms = []
    for proc_dir in dirs:
        if not proc_dir.exists():
            continue
        for yaml_file in sorted(proc_dir.glob("*.yaml")):
            if yaml_file.name in seen:
                logger.debug("Procedure %s already loaded (project override)", yaml_file.name)
                continue
            seen.add(yaml_file.name)
            try:
                fsm = FSM.from_yaml(str(yaml_file), results_dir)
                fsms.append(fsm)
                logger.debug("Loaded procedure: %s from %s", fsm.name, proc_dir)
            except Exception as e:
                logger.error("Failed to load %s: %s", yaml_file, e)
    return fsms


def main():
    results_dir = Path("results")
    project_procedures = Path("procedures")

    for i, arg in enumerate(sys.argv):
        if arg == "--results-dir" and i + 1 < len(sys.argv):
            results_dir = Path(sys.argv[i + 1])
        if arg == "--procedures-dir" and i + 1 < len(sys.argv):
            project_procedures = Path(sys.argv[i + 1])

    if not results_dir.exists():
        logger.error("Results dir not found: %s", results_dir)
        sys.exit(1)

    # Find orze-pro package
    pro_pkg = _find_pro_package()
    pro_tier = "pro" if pro_pkg else "basic"
    logger.info("Tier: %s%s", pro_tier,
                f" ({pro_pkg})" if pro_pkg else "")

    # Ensure project root is on sys.path so pro plugins can import fsm.engine
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # Load plugins: project-level first (overrides), then pro
    plugin_dirs = [Path(__file__).parent / "plugins"]
    if pro_pkg:
        plugin_dirs.append(pro_pkg / "fsm" / "plugins")
    _load_plugins(plugin_dirs)

    # Set up JSONL activity log
    from fsm.engine import set_activity_log
    set_activity_log(results_dir / "_fsm_activity.jsonl")

    # Load procedures: project-level first (overrides), then pro
    proc_dirs = [project_procedures]
    if pro_pkg:
        proc_dirs.append(pro_pkg / "procedures")
    fsms = _load_procedures(proc_dirs, results_dir)

    if not fsms:
        logger.warning("No procedures found (checked: %s)",
                        ", ".join(str(d) for d in proc_dirs))
        sys.exit(0)

    logger.info("Loaded %d procedures: %s",
                len(fsms), ", ".join(f.name for f in fsms))

    # Step each FSM
    for fsm in fsms:
        status = fsm.status()
        logger.info("[%s] state=%s, transitions=%d",
                    fsm.name, status["state"], status["transitions"])
        fsm.step()


if __name__ == "__main__":
    main()
