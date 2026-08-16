import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
import os


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "experiment.jsonl"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_CODES = {429, 500, 502, 503, 504}


def load_inputs():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(ROOT / "questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(ROOT / config["prompt_files"]["personas"], encoding="utf-8") as f:
        persona_prompts = yaml.safe_load(f)
    with open(ROOT / config["prompt_files"]["experiment"], encoding="utf-8") as f:
        experiment_prompts = yaml.safe_load(f)
    return config, questions, persona_prompts, experiment_prompts


def request_id(experiment_id, model, question_id, persona, frame, run):
    raw = f"{experiment_id}|{model}|{question_id}|{persona}|{frame}|{run}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def should_swap(seed, request_id_value):
    rng = random.Random(f"{seed}:{request_id_value}")
    return rng.choice([False, True])


def successful_ids(experiment_id):
    found = set()
    if not OUTPUT.exists():
        return found
    with open(OUTPUT, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "success" and row.get("experiment_id") == experiment_id:
                    found.add(row["request_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return found


def tracked_cost(experiment_id):
    total = 0.0
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("status") == "success" and row.get("experiment_id") == experiment_id:
                        total += float(row.get("cost") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    return total


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def extract_cost(data):
    usage = data.get("usage") or {}
    return float(usage.get("cost") or usage.get("total_cost") or 0)


async def openrouter_request(client, payload, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    structured = True
    last_error = "unknown error"
    for attempt in range(5):
        body = dict(payload)
        if structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "preference_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {"type": "string", "enum": ["A", "B"]},
                            "what": {"type": "string"},
                            "why": {"type": "string"},
                            "how": {"type": "string"},
                        },
                        "required": ["choice", "what", "why", "how"],
                        "additionalProperties": False,
                    },
                },
            }
        try:
            response = await client.post(API_URL, headers=headers, json=body)
            if response.status_code == 400 and structured:
                structured = False
                last_error = "structured output unsupported; retrying without it"
                continue
            if response.status_code in RETRY_CODES:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt < 4:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
            response.raise_for_status()
            return response.json(), None
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < 4:
                await asyncio.sleep(0.5 * (2**attempt))
    return None, last_error


async def main_async(args):
    config, questions, persona_prompts, experiment_prompts = load_inputs()
    experiment_id = config["experiment_id"]
    models = config["experimental_models"]
    personas = config["personas"]
    frames = {frame_id: experiment_prompts["frames"][frame_id] for frame_id in config["frames"]}
    runs = 1 if args.pilot else int(config["runs_per_condition"])
    selected_questions = questions[:2] if args.pilot else questions
    total = len(models) * len(selected_questions) * len(personas) * len(frames) * runs

    if args.dry_run:
        print(f"models: {len(models)} ({', '.join(models)})")
        print(f"questions: {len(selected_questions)}")
        print(f"personas: {len(personas)}")
        print(f"frames: {len(frames)}")
        print(f"runs: {runs}")
        print(f"total requests: {total}")
        return

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add it.")

    RESULTS.mkdir(exist_ok=True)
    completed = successful_ids(experiment_id)
    state = {"cost": tracked_cost(experiment_id), "done": 0, "stopped": False}
    write_lock = asyncio.Lock()
    budget_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    jobs = []
    for model in models:
        for question in selected_questions:
            for persona_id, persona in personas.items():
                for frame_id, frame_text in frames.items():
                    for run in range(1, runs + 1):
                        rid = request_id(experiment_id, model, question["id"], persona_id, frame_id, run)
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
                persona_instruction = ""
                if persona_prompt:
                    persona_instruction = experiment_prompts["persona_instruction"].format(persona_prompt=persona_prompt)
                system = experiment_prompts["system"].format(persona_instruction=persona_instruction)
                user = experiment_prompts["user"].format(
                    frame=frame_text, display_a=display_a, display_b=display_b
                )
                payload = {
                    "model": model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": config["experiment_temperature"],
                    "max_tokens": config["max_output_tokens"],
                }
                data, error = await openrouter_request(client, payload, api_key)
                parsed, usage, model_choice = None, {}, None
                if data:
                    usage = data.get("usage") or {}
                    try:
                        parsed = parse_json(data["choices"][0]["message"]["content"])
                    except (KeyError, IndexError, TypeError):
                        parsed = None
                    if parsed:
                        model_choice = str(parsed.get("choice", "")).strip().upper()
                valid = model_choice in {"A", "B"}
                canonical = ({"A": "B", "B": "A"}[model_choice] if swapped else model_choice) if valid else None
                cost = extract_cost(data or {})
                row = {
                    "request_id": rid, "experiment_id": experiment_id,
                    "model": model, "question_id": question["id"], "category": question["category"],
                    "persona": persona_id, "frame": frame_id, "run": run,
                    "original_A": question["A"], "original_B": question["B"],
                    "display_A": display_a, "display_B": display_b,
                    "display_order": "BA" if swapped else "AB", "model_choice": model_choice,
                    "choice": model_choice, "canonical_choice": canonical,
                    "what": (parsed or {}).get("what", ""), "why": (parsed or {}).get("why", ""), "how": (parsed or {}).get("how", ""),
                    "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost, "status": "success" if valid else "error",
                    "error": error or (None if valid else "Response was not valid choice JSON"),
                }
                async with write_lock:
                    with open(OUTPUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    state["cost"] += cost
                    state["done"] += 1
                    print(f"{state['done']} / {total}\ncost: ${state['cost']:.4f}")

        await asyncio.gather(*(run_one(job) for job in jobs))

    if state["stopped"]:
        print(f"Stopped because tracked cost reached the ${float(config['max_budget_usd']):.2f} budget.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show experiment size without API requests")
    parser.add_argument("--pilot", action="store_true", help="Run two questions, all conditions, one run")
    asyncio.run(main_async(parser.parse_args()))
