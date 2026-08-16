import argparse
import asyncio
import hashlib
import json
import os
import random
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
    openrouter_request,
    read_jsonl,
    recorded_cost,
    utc_now,
    write_manifest,
)
from results_layout import select_experiment_run, write_run_metadata


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "experiment.jsonl"
FAILURE_OUTPUT = RESULTS / "experiment_failures.jsonl"
RAW_HTTP_LOG = RESULTS / "raw_http_log.jsonl"


def configure_results(run_dir):
    global RESULTS, OUTPUT, FAILURE_OUTPUT, RAW_HTTP_LOG
    RESULTS = run_dir
    OUTPUT = RESULTS / "chooser/success/experiment.jsonl"
    FAILURE_OUTPUT = RESULTS / "chooser/failure/experiment.jsonl"
    RAW_HTTP_LOG = RESULTS / "audit/raw_http_log.jsonl"


CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B"]},
        "what": {"type": "string", "minLength": 1},
        "why": {"type": "string", "minLength": 1},
        "how": {"type": "string", "minLength": 1},
    },
    "required": ["choice", "what", "why", "how"],
    "additionalProperties": False,
}


def load_inputs():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = apply_runtime_overrides(yaml.safe_load(handle))
    with (ROOT / "questions.json").open(encoding="utf-8") as handle:
        questions = json.load(handle)
    with (ROOT / config["prompt_files"]["personas"]).open(encoding="utf-8") as handle:
        persona_prompts = yaml.safe_load(handle)
    with (ROOT / config["prompt_files"]["experiment"]).open(encoding="utf-8") as handle:
        experiment_prompts = yaml.safe_load(handle)
    return config, questions, persona_prompts, experiment_prompts


def request_id(fingerprint, model, question_id, persona, frame, run):
    raw = f"{fingerprint}|{model}|{question_id}|{persona}|{frame}|{run}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def should_swap(seed, request_id_value):
    return random.Random(f"{seed}:{request_id_value}").choice([False, True])


def successful_ids():
    return {
        row["request_id"]
        for row in read_jsonl(OUTPUT)
        if row.get("status") == "success" and row.get("request_id")
    }


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


def valid_choice_response(parsed):
    return bool(
        isinstance(parsed, dict)
        and str(parsed.get("choice", "")).strip().upper() in {"A", "B"}
        and all(
            isinstance(parsed.get(field), str) and bool(parsed[field].strip())
            for field in ("what", "why", "how")
        )
    )


def experiment_messages(persona_name, persona_prompt, experiment_prompts, frame_templates, display_a, display_b):
    frame_key = "persona" if persona_prompt else "baseline"
    frame_text = frame_templates[frame_key]
    if persona_prompt:
        frame_text = frame_text.format(persona_name=persona_name)
    user = experiment_prompts["user"].format(
        frame=frame_text,
        display_a=display_a,
        display_b=display_b,
    )
    if not persona_prompt:
        return [{"role": "user", "content": user}]
    persona_instruction = experiment_prompts["persona_instruction"].format(
        persona_prompt=persona_prompt
    )
    system = experiment_prompts["system"].format(persona_instruction=persona_instruction)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_configuration(config, persona_prompts, experiment_prompts):
    if not config.get("include_baseline", False):
        raise SystemExit("This design requires the no-system-prompt Assistant condition.")
    if list(config["personas"]) != ["P1", "P2", "P3", "P4", "P5"]:
        raise SystemExit("personas must contain exactly P1-P5 in order.")
    configured_prompt_ids = set(config["personas"]) | {config["baseline"]["id"]}
    if configured_prompt_ids != set(persona_prompts):
        raise SystemExit("Persona/baseline IDs in config.yaml and prompts/personas.yaml do not match.")
    if set(config["judge_personas"]) != set(config["personas"]):
        raise SystemExit("judge_personas must contain exactly the five prompted persona IDs.")
    if config["baseline"]["id"] in config["judge_personas"]:
        raise SystemExit("The no-prompt Assistant must not be a judge persona candidate.")
    if persona_prompts[config["baseline"]["id"]]["prompt"]:
        raise SystemExit("The Assistant baseline prompt must be empty.")
    missing_frames = set(config["frames"]) - set(experiment_prompts["frames"])
    if missing_frames:
        raise SystemExit(f"Missing frame prompts: {sorted(missing_frames)}")
    for frame_id in config["frames"]:
        templates = experiment_prompts["frames"][frame_id]
        if not isinstance(templates, dict) or set(templates) != {"persona", "baseline"}:
            raise SystemExit(f"Frame {frame_id} must define persona and baseline templates.")
        if "{persona_name}" not in templates["persona"]:
            raise SystemExit(f"Frame {frame_id} persona template must contain {{persona_name}}.")
    runs = int(config["runs_per_condition"])
    if runs < 1 or runs % 2 == 0:
        raise SystemExit("runs_per_condition must be a positive odd number so majority votes cannot tie.")


async def main_async(args):
    config, questions, persona_prompts, experiment_prompts = load_inputs()
    validate_configuration(config, persona_prompts, experiment_prompts)
    fingerprint_source = experiment_fingerprint_inputs(
        config, questions, persona_prompts, experiment_prompts
    )
    fingerprint = canonical_hash(fingerprint_source)
    experiment_id = config["experiment_id"]
    models = config["experimental_models"]
    personas = config["personas"]
    conditions = experiment_conditions(config)
    frames = {frame_id: experiment_prompts["frames"][frame_id] for frame_id in config["frames"]}
    mode = "pilot" if args.pilot else "full"
    runs = 1 if args.pilot else int(config["runs_per_condition"])
    selected_questions = questions[:2] if args.pilot else questions
    total = len(models) * len(selected_questions) * len(conditions) * len(frames) * runs

    if args.dry_run:
        print(f"mode: {mode}")
        print(f"fingerprint: {fingerprint}")
        print(f"models: {len(models)} ({', '.join(models)})")
        print(f"questions: {len(selected_questions)}")
        print(f"personas: {len(personas)}")
        print(f"no-prompt baseline: {'enabled' if config.get('include_baseline') else 'disabled'}")
        print(f"total conditions: {len(conditions)}")
        print(f"frames: {len(frames)}")
        print(f"runs: {runs}")
        print(f"total requests: {total}")
        print("API requests sent: 0")
        return

    placeholders = [model for model in models if model.startswith("MODEL_")]
    if placeholders:
        raise SystemExit(f"Replace model placeholders in config.yaml: {', '.join(placeholders)}")

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add it.")

    configure_results(select_experiment_run(mode, fingerprint))
    write_run_metadata(RESULTS, created_at=utc_now(), mode=mode, experiment_id=experiment_id, experiment_fingerprint=fingerprint, experiment_complete=False)
    print(f"Run directory: {RESULTS}")
    write_manifest(
        RESULTS / "audit/manifests" / f"experiment_{fingerprint}_{mode}.json",
        {
            "manifest_version": 1,
            "created_at": utc_now(),
            "experiment_id": experiment_id,
            "fingerprint": fingerprint,
            "mode": mode,
            "expected_requests": total,
            "raw_http_log": RAW_HTTP_LOG.name,
            "inputs": fingerprint_source,
        },
    )

    completed = successful_ids()
    initial_cost = recorded_cost(
        RAW_HTTP_LOG,
        [OUTPUT, FAILURE_OUTPUT, RESULTS / "judges/success/judges.jsonl", RESULTS / "judges/success/assistant_profiles.jsonl"],
        experiment_id,
    )
    state = {"cost": initial_cost, "done": 0, "stopped": False}
    write_lock = asyncio.Lock()
    http_log_lock = asyncio.Lock()
    budget_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    jobs = []
    expected_ids = set()

    async def log_http_attempt(record):
        async with http_log_lock:
            append_jsonl(RAW_HTTP_LOG, record)

    for model in models:
        for question in selected_questions:
            for persona_id, persona in conditions.items():
                for frame_id, frame_text in frames.items():
                    for run in range(1, runs + 1):
                        rid = request_id(fingerprint, model, question["id"], persona_id, frame_id, run)
                        expected_ids.add(rid)
                        jobs.append((rid, model, question, persona_id, persona, frame_id, frame_text, run))

    async with httpx.AsyncClient(timeout=60) as client:
        async def run_one(job):
            rid, model, question, persona_id, persona, frame_id, frame_text, run = job
            if rid in completed:
                async with write_lock:
                    state["done"] += 1
                return
            async with semaphore:
                async with budget_lock:
                    if state["cost"] >= float(config["max_budget_usd"]):
                        state["stopped"] = True
                        return

                swapped = should_swap(config["random_seed"], rid)
                display_a = question["B"] if swapped else question["A"]
                display_b = question["A"] if swapped else question["B"]
                persona_prompt = persona_prompts[persona_id]["prompt"]
                payload = {
                    "model": model,
                    "messages": experiment_messages(
                        persona["name"], persona_prompt, experiment_prompts, frame_text, display_a, display_b
                    ),
                    "temperature": config["experiment_temperature"],
                    "max_tokens": config["max_output_tokens"],
                }
                all_attempts = []
                data = error = parsed = None
                usage, model_choice, valid = {}, None, False
                for semantic_attempt in range(1, 4):
                    async with budget_lock:
                        if state["cost"] >= float(config["max_budget_usd"]):
                            state["stopped"] = True
                            error = "Budget reached before a valid response"
                            break
                    data, error, attempts = await openrouter_request(
                        client, payload, api_key,
                        request_context={"experiment_id": experiment_id, "fingerprint": fingerprint, "request_id": rid, "stage": "experiment", "model": model, "semantic_attempt": semantic_attempt},
                        schema_name="preference_choice", schema=CHOICE_SCHEMA, log_attempt=log_http_attempt,
                    )
                    all_attempts.extend(attempts)
                    async with budget_lock:
                        state["cost"] += attempt_cost(attempts)
                    usage = (data or {}).get("usage") or {}
                    try:
                        parsed = parse_json(data["choices"][0]["message"]["content"]) if data else None
                    except (KeyError, IndexError, TypeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        model_choice = str(parsed.get("choice", "")).strip().upper()
                    valid = valid_choice_response(parsed)
                    if valid:
                        break
                parsed_fields = parsed if isinstance(parsed, dict) else {}
                canonical = (
                    ({"A": "B", "B": "A"}[model_choice] if swapped else model_choice)
                    if valid
                    else None
                )
                cost = attempt_cost(all_attempts)
                row = {
                    "request_id": rid,
                    "experiment_id": experiment_id,
                    "experiment_fingerprint": fingerprint,
                    "run_mode": mode,
                    "expected_runs": runs,
                    "model": model,
                    "question_id": question["id"],
                    "category": question["category"],
                    "persona": persona_id,
                    "frame": frame_id,
                    "run": run,
                    "original_A": question["A"],
                    "original_B": question["B"],
                    "display_A": display_a,
                    "display_B": display_b,
                    "display_order": "BA" if swapped else "AB",
                    "model_choice": model_choice,
                    "canonical_choice": canonical,
                    "what": parsed_fields.get("what", ""),
                    "why": parsed_fields.get("why", ""),
                    "how": parsed_fields.get("how", ""),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost,
                    "raw_http_log": RAW_HTTP_LOG.name,
                    "status": "success" if valid else "error",
                    "error": error or (None if valid else "Response was not valid choice JSON"),
                }
                async with write_lock:
                    append_jsonl(OUTPUT if valid else FAILURE_OUTPUT, row)
                    state["done"] += 1
                    print(f"{state['done']} / {total} | tracked cost: ${state['cost']:.4f}")

        await asyncio.gather(*(run_one(job) for job in jobs))

    current_successes = successful_ids() & expected_ids
    completion = {
        "reported_at": utc_now(),
        "experiment_id": experiment_id,
        "fingerprint": fingerprint,
        "mode": mode,
        "expected_requests": total,
        "successful_requests": len(current_successes),
        "missing_requests": total - len(current_successes),
        "tracked_cost_usd": state["cost"],
        "budget_stopped": state["stopped"],
        "complete": len(current_successes) == total,
    }
    completion_path = RESULTS / "audit/completion" / f"experiment_{fingerprint}_{mode}.json"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    write_run_metadata(RESULTS, experiment_complete=completion["complete"])

    if not completion["complete"]:
        reason = "budget reached" if state["stopped"] else "failed or invalid responses"
        raise SystemExit(
            f"Experiment incomplete ({completion['successful_requests']}/{total}; {reason}). "
            "Rerun the same command to resume. Judges were not started."
        )
    print(f"Experiment complete: {total}/{total}. Raw HTTP log: {RAW_HTTP_LOG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show experiment size without API requests")
    parser.add_argument("--pilot", action="store_true", help="Run two questions, all conditions, one run")
    asyncio.run(main_async(parser.parse_args()))
