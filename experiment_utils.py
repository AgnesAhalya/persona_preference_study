import asyncio
import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_CODES = {429, 500, 502, 503, 504}


def apply_runtime_overrides(config, environ=None):
    """Apply optional environment overrides without editing config.yaml."""
    values = os.environ if environ is None else environ
    updated = copy.deepcopy(config)
    overrides = (
        ("EXPERIMENT_MODEL_1", "experimental_models", 0),
        ("EXPERIMENT_MODEL_2", "experimental_models", 1),
        ("JUDGE_MODEL_1", "judge_models", 0),
        ("JUDGE_MODEL_2", "judge_models", 1),
    )
    for environment_name, config_key, index in overrides:
        value = values.get(environment_name, "").strip()
        if value:
            updated[config_key][index] = value

    budget = values.get("MAX_BUDGET_USD", "").strip()
    if budget:
        try:
            updated["max_budget_usd"] = float(budget)
        except ValueError as error:
            raise ValueError("MAX_BUDGET_USD must be a positive number.") from error
        if updated["max_budget_usd"] <= 0:
            raise ValueError("MAX_BUDGET_USD must be a positive number.")
    return updated


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


def experiment_conditions(config):
    conditions = dict(config["personas"])
    if config.get("include_baseline", False):
        baseline = config["baseline"]
        conditions = {baseline["id"]: {"name": baseline["name"]}, **conditions}
    return conditions


def experiment_fingerprint_inputs(config, questions, persona_prompts, experiment_prompts):
    return {
        "schema_version": 3,
        "experiment_id": config["experiment_id"],
        "experimental_models": config["experimental_models"],
        "include_baseline": config.get("include_baseline", False),
        "baseline": config["baseline"],
        "personas": config["personas"],
        "persona_prompts": persona_prompts,
        "questions": questions,
        "frames": config["frames"],
        "experiment_prompts": experiment_prompts,
        "runs_per_condition": config["runs_per_condition"],
        "experiment_temperature": config["experiment_temperature"],
        "max_output_tokens": config["max_output_tokens"],
        "random_seed": config["random_seed"],
    }


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


def matching_successes(rows, **required_fields):
    return [
        row for row in latest_successes(rows)
        if all(row.get(field) == value for field, value in required_fields.items())
    ]


def judge_fingerprint_inputs(
    config,
    experiment_fingerprint,
    judge_prompts,
    threshold,
    mode,
    source_question_count,
    observations_per_question,
):
    return {
        "schema_version": 3,
        "experiment_fingerprint": experiment_fingerprint,
        "judge_models": config["judge_models"],
        "judge_personas": config["judge_personas"],
        "judge_other_label": config["judge_other_label"],
        "judge_prompts": judge_prompts,
        "judge_temperature": config["judge_temperature"],
        "max_output_tokens": config["max_output_tokens"],
        "a_rate_threshold": threshold,
        "mode": mode,
        "source_question_count": source_question_count,
        "observations_per_question": observations_per_question,
        "aggregation": "all frames and runs per model/persona/question",
    }


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


def response_cost(data):
    usage = (data or {}).get("usage") or {}
    return float(usage.get("cost") or usage.get("total_cost") or 0)


def retry_delay(response, attempt_number):
    if response is not None:
        value = response.headers.get("retry-after")
        if value is not None:
            try:
                return min(max(float(value), 0), 60)
            except ValueError:
                pass
    return 0.5 * (2 ** (attempt_number - 1))


async def openrouter_request(
    client,
    payload,
    api_key,
    *,
    request_context,
    schema_name,
    schema,
    log_attempt,
):
    """Send an OpenRouter request and log every HTTP attempt with the key redacted."""
    real_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    logged_headers = {"Authorization": "Bearer [REDACTED]", "Content-Type": "application/json"}
    structured = True
    attempts = []
    last_error = "unknown error"

    for attempt_number in range(1, 6):
        body = dict(payload)
        used_structured_output = structured
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }

        requested_at = utc_now()
        started = time.perf_counter()
        response = None
        sent_request = None
        data = None
        error = None
        try:
            response = await client.post(API_URL, headers=real_headers, json=body)
            sent_request = response.request
            raw_body = response.text
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = None
        except httpx.HTTPError as exc:
            raw_body = None
            sent_request = getattr(exc, "request", None)
            error = str(exc).replace(api_key, "[REDACTED]")

        if response is not None:
            if response.status_code == 400 and structured:
                error = "Structured output unsupported; retrying without response_format."
            elif response.status_code in RETRY_CODES:
                error = f"HTTP {response.status_code}: {raw_body[:300]}"
            elif response.is_error:
                error = f"HTTP {response.status_code}: {raw_body[:300]}"
            elif data is None:
                error = "OpenRouter returned a non-JSON response."

        request_headers = dict(sent_request.headers) if sent_request is not None else dict(logged_headers)
        for header_name in list(request_headers):
            if header_name.lower() == "authorization":
                request_headers[header_name] = "Bearer [REDACTED]"
        request_raw_body = None
        if sent_request is not None:
            try:
                request_raw_body = sent_request.content.decode("utf-8")
            except (UnicodeDecodeError, httpx.RequestNotRead):
                request_raw_body = None
        if request_raw_body is None:
            request_raw_body = json.dumps(body, ensure_ascii=False, separators=(",", ":"))

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        cost = response_cost(data)
        attempt_log = {
            "http_log_version": 1,
            **request_context,
            "attempt": attempt_number,
            "requested_at": requested_at,
            "duration_ms": elapsed_ms,
            "structured_output": used_structured_output,
            "request": {
                "method": sent_request.method if sent_request is not None else "POST",
                "url": str(sent_request.url) if sent_request is not None else API_URL,
                "headers": request_headers,
                "raw_body": request_raw_body,
            },
            "response": None if response is None else {
                "http_version": response.http_version,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "raw_body": raw_body,
            },
            "cost": cost,
            "error": error,
        }
        await log_attempt(attempt_log)
        attempts.append({
            "attempt": attempt_number,
            "http_status": None if response is None else response.status_code,
            "cost": cost,
            "error": error,
        })

        if response is None:
            last_error = error or "Network error"
            if attempt_number < 5:
                await asyncio.sleep(0.5 * (2 ** (attempt_number - 1)))
                continue
            break
        if response.status_code == 400 and structured:
            structured = False
            last_error = error
            if attempt_number < 5:
                continue
            break
        if response.status_code in RETRY_CODES:
            last_error = error
            if attempt_number < 5:
                await asyncio.sleep(retry_delay(response, attempt_number))
                continue
            break
        if response.is_error:
            last_error = error
            break
        if data is None:
            last_error = error
            if attempt_number < 5:
                await asyncio.sleep(0.5 * (2 ** (attempt_number - 1)))
                continue
            break
        return data, None, attempts

    return None, last_error, attempts


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(existing.get("inputs")) != canonical_hash(manifest.get("inputs")):
            raise RuntimeError(f"Refusing to overwrite a different manifest: {path}")
        return
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
