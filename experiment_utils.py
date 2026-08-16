import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: ignored invalid JSONL at {path}:{line_number}")
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def latest_successes(rows):
    latest = {}
    for row in rows:
        if row.get("status") == "success" and row.get("request_id"):
            latest[row["request_id"]] = row
    return list(latest.values())


def archive_cost(archive_rows, experiment_id):
    total = 0.0
    for row in archive_rows:
        if row.get("experiment_id") != experiment_id:
            continue
        try:
            total += float(row.get("cost") or 0)
        except (TypeError, ValueError):
            pass
    return total


def recorded_cost(archive_path, result_paths, experiment_id):
    archive_rows = read_jsonl(archive_path)
    archived_request_ids = {
        row.get("request_id")
        for row in archive_rows
        if row.get("experiment_id") == experiment_id
    }
    total = archive_cost(archive_rows, experiment_id)

    # Backward compatibility: count legacy result rows that predate the raw archive.
    for result_path in result_paths:
        for row in read_jsonl(result_path):
            if row.get("experiment_id") != experiment_id:
                continue
            if row.get("request_id") in archived_request_ids:
                continue
            try:
                total += float(row.get("cost") or 0)
            except (TypeError, ValueError):
                pass
    return total


def attempt_cost(attempts):
    total = 0.0
    for attempt in attempts:
        try:
            total += float(attempt.get("cost") or 0)
        except (TypeError, ValueError):
            pass
    return total


def make_archive_record(*, experiment_id, fingerprint, request_id, stage, model, payload, attempts, final_error):
    return {
        "archive_version": 1,
        "archived_at": utc_now(),
        "experiment_id": experiment_id,
        "fingerprint": fingerprint,
        "request_id": request_id,
        "stage": stage,
        "model": model,
        "request_payload": payload,
        "attempts": attempts,
        "cost": attempt_cost(attempts),
        "final_error": final_error,
    }


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(existing.get("inputs")) != canonical_hash(manifest.get("inputs")):
            raise RuntimeError(f"Refusing to overwrite a different manifest: {path}")
        return
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
