"""Timestamped result-directory layout shared by all experiment stages."""

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS_ROOT = ROOT / "results"
LATEST_RUN = RESULTS_ROOT / "LATEST_RUN"


def new_run_directory(mode):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S-%fZ")
    run_dir = RESULTS_ROOT / f"run_{stamp}_{mode}"
    initialize_run_directory(run_dir)
    LATEST_RUN.write_text(run_dir.name + "\n", encoding="utf-8")
    return run_dir


def select_experiment_run(mode, fingerprint):
    """Resume the latest matching incomplete run; otherwise create a new run."""
    if LATEST_RUN.exists():
        candidate = RESULTS_ROOT / LATEST_RUN.read_text(encoding="utf-8").strip()
        metadata_path = candidate / "run_metadata.json"
        if candidate.is_dir() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("mode") == mode and metadata.get("experiment_fingerprint") == fingerprint and not metadata.get("experiment_complete", False):
                initialize_run_directory(candidate)
                return candidate
    return new_run_directory(mode)


def latest_run_directory():
    if not LATEST_RUN.exists():
        raise SystemExit("No recorded run exists. Run run_experiment.py first.")
    run_dir = RESULTS_ROOT / LATEST_RUN.read_text(encoding="utf-8").strip()
    if not run_dir.is_dir():
        raise SystemExit(f"Latest run directory is missing: {run_dir}")
    return run_dir


def initialize_run_directory(run_dir):
    run_dir = Path(run_dir)
    for relative in (
        "chooser/success", "chooser/failure", "chooser/analysis",
        "judges/success", "judges/failure", "judges/analysis",
        "audit/manifests", "audit/completion",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir, **values):
    path = Path(run_dir) / "run_metadata.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "created_at" in current:
        values.pop("created_at", None)
    current.update(values)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

