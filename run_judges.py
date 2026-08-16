import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from experiment_utils import (
    apply_runtime_overrides,
    append_jsonl,
    attempt_cost,
    canonical_hash,
    experiment_conditions,
    experiment_fingerprint_inputs,
    judge_fingerprint_inputs,
    latest_successes,
    openrouter_request,
    read_jsonl,
    recorded_cost,
    utc_now,
    write_manifest,
)
from results_layout import latest_run_directory, write_run_metadata


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
EXPERIMENT = RESULTS / "experiment.jsonl"
JUDGES = RESULTS / "judges.jsonl"
PROFILES = RESULTS / "inferred_default_behavioral_profile.jsonl"
RAW_HTTP_LOG = RESULTS / "raw_http_log.jsonl"
JUDGE_FAILURES = RESULTS / "judge_failures.jsonl"
PROFILE_FAILURES = RESULTS / "profile_failures.jsonl"


def configure_results(run_dir):
    global RESULTS, EXPERIMENT, JUDGES, PROFILES, RAW_HTTP_LOG, JUDGE_FAILURES, PROFILE_FAILURES
    RESULTS = run_dir
    EXPERIMENT = RESULTS / "chooser/success/experiment.jsonl"
    JUDGES = RESULTS / "judges/success/judges.jsonl"
    PROFILES = RESULTS / "judges/success/assistant_profiles.jsonl"
    JUDGE_FAILURES = RESULTS / "judges/failure/judges.jsonl"
    PROFILE_FAILURES = RESULTS / "judges/failure/assistant_profiles.jsonl"
    RAW_HTTP_LOG = RESULTS / "audit/raw_http_log.jsonl"


def parse_json(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def normalize_threshold(value):
    threshold = float(value)
    if threshold > 1:
        threshold /= 100
    if not 0 <= threshold <= 1:
        raise ValueError("Judge A-rate threshold must be between 0 and 1, or between 0 and 100 percent.")
    return threshold


def include_in_bucket(a_rate, threshold):
    if threshold == 0:
        return True
    if threshold == 1:
        return a_rate == 1
    return a_rate > threshold


def valid_classification(parsed, prediction_labels, other_label):
    if not isinstance(parsed, dict):
        return False
    predicted = parsed.get("persona")
    confidence = parsed.get("confidence")
    profile_name = parsed.get("other_profile_name")
    profile_description = parsed.get("other_profile_description")
    profile_traits = parsed.get("other_profile_traits")
    if (
        predicted not in prediction_labels
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
        or not isinstance(profile_name, str)
        or not isinstance(profile_description, str)
        or not isinstance(profile_traits, list)
    ):
        return False
    if predicted == other_label:
        return bool(
            profile_name.strip()
            and profile_description.strip()
            and len(profile_traits) == 5
            and all(isinstance(trait, str) and trait.strip() for trait in profile_traits)
        )
    return profile_name == "" and profile_description == "" and profile_traits == []


def valid_behavioral_profile(parsed):
    return bool(
        isinstance(parsed, dict)
        and isinstance(parsed.get("traits"), list)
        and len(parsed["traits"]) == 5
        and all(isinstance(trait, str) and trait.strip() for trait in parsed["traits"])
        and isinstance(parsed.get("summary"), str)
        and parsed["summary"].strip()
    )


def aggregate_task_preferences(rows, config, questions, mode_override=None):
    conditions = experiment_conditions(config)
    groups = defaultdict(dict)
    for row in rows:
        if row.get("canonical_choice") not in {"A", "B"}:
            continue
        key = (row["model"], row["question_id"], row["persona"])
        groups[key][(row["frame"], int(row["run"]))] = row

    mode = mode_override or (
        "full" if any(row.get("run_mode") == "full" for row in rows) else "pilot"
    )
    expected_runs = int(config["runs_per_condition"]) if mode == "full" else 1
    expected_questions = questions if mode == "full" else questions[:2]
    expected_observations = len(config["frames"]) * expected_runs
    if expected_observations % 2 == 0:
        raise SystemExit("frames × runs must be odd so the aggregated A/B preference cannot tie.")

    incomplete = []
    preferences = []
    for model in config["experimental_models"]:
        for question in expected_questions:
            for persona in conditions:
                key = (model, question["id"], persona)
                observations = groups.get(key, {})
                expected_keys = {
                    (frame, run)
                    for frame in config["frames"]
                    for run in range(1, expected_runs + 1)
                }
                missing = sorted(expected_keys - set(observations))
                if missing:
                    incomplete.append({
                        "model": model,
                        "question_id": question["id"],
                        "persona": persona,
                        "missing": missing,
                    })
                    continue

                ordered = [observations[key] for key in sorted(expected_keys)]
                counts = Counter(row["canonical_choice"] for row in ordered)
                winner = "A" if counts["A"] > counts["B"] else "B"
                representative = next(row for row in ordered if row["canonical_choice"] == winner)
                total = len(ordered)
                preferences.append({
                    "model": model,
                    "question_id": question["id"],
                    "category": question["category"],
                    "actual_persona": persona,
                    "A": question["A"],
                    "B": question["B"],
                    "a_count": counts["A"],
                    "b_count": counts["B"],
                    "a_rate": counts["A"] / total,
                    "b_rate": counts["B"] / total,
                    "preferred_choice": winner,
                    "preferred_text": question[winner],
                    "preferred_rate": counts[winner] / total,
                    "observations": total,
                    "what": representative.get("what", ""),
                    "why": representative.get("why", ""),
                    "how": representative.get("how", ""),
                })
    return preferences, incomplete, mode, len(expected_questions), expected_observations


def build_batches(preferences, config, threshold):
    conditions = experiment_conditions(config)
    grouped = defaultdict(list)
    for preference in preferences:
        grouped[(preference["model"], preference["actual_persona"])].append(preference)

    batches = []
    for model in config["experimental_models"]:
        for persona in conditions:
            all_items = sorted(grouped[(model, persona)], key=lambda item: item["question_id"])
            bucket = [item for item in all_items if include_in_bucket(item["a_rate"], threshold)]
            profile_signature = [
                (item["question_id"], item["a_count"], item["b_count"], item["observations"])
                for item in bucket
            ]
            profile_id = canonical_hash({
                "model": model,
                "persona": persona,
                "threshold": threshold,
                "items": profile_signature,
            })[:20]
            batches.append({
                "profile_id": profile_id,
                "model": model,
                "actual_persona": persona,
                "threshold": threshold,
                "all_item_count": len(all_items),
                "items": bucket,
            })
    return batches


def render_batch(batch, condition, prompts):
    if not batch["items"]:
        return "No questions met the configured A-rate threshold."
    rendered = []
    for item_number, item in enumerate(batch["items"], start=1):
        text = prompts["batch_choice_item"].format(
            item_number=item_number,
            question_id=item["question_id"],
            option_a=item["A"],
            option_b=item["B"],
            preferred_choice=item["preferred_choice"],
            preferred_text=item["preferred_text"],
            preferred_percent=round(item["preferred_rate"] * 100, 1),
            preferred_count=max(item["a_count"], item["b_count"]),
            observations=item["observations"],
            a_percent=round(item["a_rate"] * 100, 1),
            b_percent=round(item["b_rate"] * 100, 1),
        )
        if condition == "choice_and_explanation":
            text += "\n" + prompts["batch_explanation"].format(
                what=item["what"], why=item["why"], how=item["how"]
            )
        rendered.append(text)
    return "\n\n".join(rendered)


def classification_prompt(batch, condition, candidates, persona_prompts, prompts):
    descriptions = "\n".join(
        prompts["candidate_line"].format(
            persona_id=persona_id,
            persona_name=value["name"],
            persona_description=persona_prompts[persona_id]["description"],
        )
        for persona_id, value in candidates.items()
    )
    return prompts["classification_user"].format(
        descriptions=descriptions,
        evidence=render_batch(batch, condition, prompts),
    )


async def main(args):
    configure_results(latest_run_directory())
    print(f"Run directory: {RESULTS}")
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = apply_runtime_overrides(yaml.safe_load(handle))
    with (ROOT / "questions.json").open(encoding="utf-8") as handle:
        questions = json.load(handle)
    with (ROOT / config["prompt_files"]["personas"]).open(encoding="utf-8") as handle:
        persona_prompts = yaml.safe_load(handle)
    with (ROOT / config["prompt_files"]["experiment"]).open(encoding="utf-8") as handle:
        experiment_prompts = yaml.safe_load(handle)
    with (ROOT / config["prompt_files"]["judges"]).open(encoding="utf-8") as handle:
        judge_prompts = yaml.safe_load(handle)
    if not EXPERIMENT.exists():
        raise SystemExit("Chooser success file not found in the latest run. Run run_experiment.py first.")

    threshold = normalize_threshold(
        config["judge_a_rate_threshold"] if args.a_rate_threshold is None else args.a_rate_threshold
    )
    experiment_source = experiment_fingerprint_inputs(
        config, questions, persona_prompts, experiment_prompts
    )
    experiment_fingerprint = canonical_hash(experiment_source)
    experiment_id = config["experiment_id"]
    experiment_rows = [
        row for row in latest_successes(read_jsonl(EXPERIMENT))
        if row.get("experiment_id") == experiment_id
        and row.get("experiment_fingerprint") == experiment_fingerprint
    ]
    preferences, incomplete, mode, question_count, observations_per_question = aggregate_task_preferences(
        experiment_rows, config, questions
    )
    if incomplete:
        raise SystemExit(
            f"Experiment is incomplete: {len(incomplete)} model/question/persona groups are missing "
            "frame/run observations. Rerun run_experiment.py before judging."
        )

    batches = build_batches(preferences, config, threshold)
    # Classify every condition, including the no-prompt P0 baseline. The prompt
    # receives only behavioral evidence; actual_persona is retained solely for scoring.
    prompted_batches = batches
    prediction_labels = config["judge_personas"] + [config["judge_other_label"]]
    candidates = {persona_id: config["personas"][persona_id] for persona_id in config["judge_personas"]}

    judge_source = judge_fingerprint_inputs(
        config,
        experiment_fingerprint,
        judge_prompts,
        threshold,
        mode,
        question_count,
        observations_per_question,
    )
    judge_fingerprint = canonical_hash(judge_source)

    placeholders = [model for model in config["judge_models"] if model.startswith("JUDGE_")]
    if placeholders:
        raise SystemExit(f"Replace judge placeholders in config.yaml: {', '.join(placeholders)}")
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add it.")

    write_run_metadata(RESULTS, judge_fingerprint=judge_fingerprint, judge_complete=False)
    write_manifest(
        RESULTS / "audit/manifests" / f"judge_{judge_fingerprint}.json",
        {
            "manifest_version": 1,
            "created_at": utc_now(),
            "experiment_id": experiment_id,
            "fingerprint": judge_fingerprint,
            "mode": mode,
            "source_question_count": question_count,
            "observations_per_question": observations_per_question,
            "a_rate_threshold": threshold,
            "raw_http_log": RAW_HTTP_LOG.name,
            "inputs": judge_source,
        },
    )

    existing_judges = {
        row["request_id"]
        for row in latest_successes(read_jsonl(JUDGES))
        if row.get("judge_fingerprint") == judge_fingerprint
    }
    initial_cost = recorded_cost(
        RAW_HTTP_LOG, [EXPERIMENT, JUDGES, JUDGE_FAILURES, PROFILES, PROFILE_FAILURES], experiment_id
    )
    state = {"cost": initial_cost, "done": 0, "stopped": False}
    result_lock = asyncio.Lock()
    http_log_lock = asyncio.Lock()
    budget_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    expected_ids = set()

    async def log_http_attempt(record):
        async with http_log_lock:
            append_jsonl(RAW_HTTP_LOG, record)

    async def request_until_valid(client, payload, context, schema_name, schema, validator):
        all_attempts = []
        data = error = parsed = None
        for semantic_attempt in range(1, 4):
            async with budget_lock:
                if state["cost"] >= float(config["max_budget_usd"]):
                    state["stopped"] = True
                    error = "Budget reached before a valid response"
                    break
            retry_context = dict(context, semantic_attempt=semantic_attempt)
            data, error, attempts = await openrouter_request(
                client, payload, api_key,
                request_context=retry_context,
                schema_name=schema_name, schema=schema, log_attempt=log_http_attempt,
            )
            all_attempts.extend(attempts)
            async with budget_lock:
                state["cost"] += attempt_cost(attempts)
            try:
                parsed = parse_json(data["choices"][0]["message"]["content"]) if data else None
            except (KeyError, IndexError, TypeError):
                parsed = None
            if validator(parsed):
                return data, error, all_attempts, parsed, True
        return data, error, all_attempts, parsed, False

    classification_schema = {
        "type": "object",
        "properties": {
            "persona": {"type": "string", "enum": prediction_labels},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "other_profile_name": {"type": "string"},
            "other_profile_description": {"type": "string"},
            "other_profile_traits": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 0,
                "maxItems": 5,
            },
        },
        "required": [
            "persona",
            "confidence",
            "other_profile_name",
            "other_profile_description",
            "other_profile_traits",
        ],
        "additionalProperties": False,
    }
    jobs = []
    for judge in config["judge_models"]:
        for batch in prompted_batches:
            for condition in ("choice_only", "choice_and_explanation"):
                rid = hashlib.sha256(
                    f"{judge_fingerprint}|{judge}|{batch['profile_id']}|{condition}".encode()
                ).hexdigest()[:20]
                expected_ids.add(rid)
                jobs.append((rid, judge, batch, condition))

    async with httpx.AsyncClient(timeout=120) as client:
        async def judge_one(job):
            rid, judge, batch, condition = job
            if rid in existing_judges:
                async with result_lock:
                    state["done"] += 1
                return
            async with semaphore:
                async with budget_lock:
                    if state["cost"] >= float(config["max_budget_usd"]):
                        state["stopped"] = True
                        return
                payload = {
                    "model": judge,
                    "messages": [
                        {"role": "system", "content": judge_prompts["classification_system"]},
                        {"role": "user", "content": classification_prompt(
                            batch, condition, candidates, persona_prompts, judge_prompts
                        )},
                    ],
                    "temperature": config["judge_temperature"],
                    "max_tokens": config.get("judge_max_output_tokens", config["max_output_tokens"]),
                }
                data, error, attempts, parsed, valid = await request_until_valid(
                    client, payload,
                    {"experiment_id": experiment_id, "fingerprint": judge_fingerprint, "request_id": rid, "stage": "judge_classification", "model": judge},
                    "judge_result", classification_schema,
                    lambda value: valid_classification(value, prediction_labels, config["judge_other_label"]),
                )
                parsed_fields = parsed if isinstance(parsed, dict) else {}
                predicted = parsed_fields.get("persona")
                confidence = parsed_fields.get("confidence")
                other_profile_name = parsed_fields.get("other_profile_name", "")
                other_profile_description = parsed_fields.get("other_profile_description", "")
                other_profile_traits = parsed_fields.get("other_profile_traits", [])
                valid = valid
                usage = (data or {}).get("usage") or {}
                result = {
                    "request_id": rid,
                    "experiment_id": experiment_id,
                    "experiment_fingerprint": experiment_fingerprint,
                    "judge_fingerprint": judge_fingerprint,
                    "judge_model": judge,
                    "condition": condition,
                    "profile_id": batch["profile_id"],
                    "experiment_model": batch["model"],
                    "actual_persona": batch["actual_persona"],
                    "predicted_persona": predicted,
                    "confidence": confidence,
                    "other_profile_name": other_profile_name,
                    "other_profile_description": other_profile_description,
                    "other_profile_traits": other_profile_traits,
                    "a_rate_threshold": threshold,
                    "source_questions": batch["all_item_count"],
                    "bucket_questions": len(batch["items"]),
                    "bucket_question_ids": [item["question_id"] for item in batch["items"]],
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": attempt_cost(attempts),
                    "raw_http_log": RAW_HTTP_LOG.name,
                    "status": "success" if valid else "error",
                    "error": error or (None if valid else "Response was not valid judge JSON"),
                }
                async with result_lock:
                    append_jsonl(JUDGES if valid else JUDGE_FAILURES, result)
                    state["done"] += 1
                    print(f"{state['done']} / {len(jobs)} | tracked cost: ${state['cost']:.4f}")

        await asyncio.gather(*(judge_one(job) for job in jobs))


    successful = {
        row["request_id"]
        for row in latest_successes(read_jsonl(JUDGES))
        if row.get("judge_fingerprint") == judge_fingerprint
    }
    completion = {
        "reported_at": utc_now(),
        "experiment_id": experiment_id,
        "experiment_fingerprint": experiment_fingerprint,
        "judge_fingerprint": judge_fingerprint,
        "a_rate_threshold": threshold,
        "expected_requests": len(expected_ids),
        "successful_requests": len(successful & expected_ids),
        "missing_requests": len(expected_ids - successful),
        "tracked_cost_usd": state["cost"],
        "budget_stopped": state["stopped"],
        "complete": expected_ids <= successful,
    }
    completion_path = RESULTS / "audit/completion" / f"judge_{judge_fingerprint}.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    write_run_metadata(RESULTS, judge_complete=completion["complete"])
    if not completion["complete"]:
        raise SystemExit(
            f"Judge run incomplete ({completion['successful_requests']}/{completion['expected_requests']}). "
            "Rerun the same command to resume."
        )
    print(f"Judge run complete. Raw HTTP log: {RAW_HTTP_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a-rate-threshold",
        help="0=all questions; 0.5 or 50=A chosen >50%%; 1 or 100=A chosen every time",
    )
    asyncio.run(main(parser.parse_args()))
