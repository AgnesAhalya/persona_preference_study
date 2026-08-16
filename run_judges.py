import asyncio
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
EXPERIMENT = RESULTS / "experiment.jsonl"
JUDGES = RESULTS / "judges.jsonl"
PROFILES = RESULTS / "inferred_default_behavioral_profile.jsonl"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_CODES = {429, 500, 502, 503, 504}


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def latest_successes(rows):
    latest = {}
    for row in rows:
        if row.get("status") == "success" and row.get("request_id"):
            latest[row["request_id"]] = row
    return list(latest.values())


def aggregate_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("canonical_choice") in {"A", "B"}:
            key = (row["model"], row["question_id"], row["persona"], row["frame"])
            groups[key].append(row)
    examples = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda x: x["run"])
        counts = Counter(row["canonical_choice"] for row in group)
        majority = sorted(counts, key=lambda choice: (-counts[choice], choice))[0]
        explanation = next(row for row in group if row["canonical_choice"] == majority)
        first = group[0]
        examples.append({
            "example_id": hashlib.sha256("|".join(key).encode()).hexdigest()[:20],
            "model": key[0], "question_id": key[1], "actual_persona": key[2], "frame": key[3],
            "category": first["category"], "A": first["original_A"], "B": first["original_B"],
            "choice": majority, "what": explanation.get("what", ""), "why": explanation.get("why", ""),
            "how": explanation.get("how", ""), "stability": counts[majority] / len(group), "runs_available": len(group),
        })
    return examples


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
                pass
    return None


def cost_of(data):
    usage = (data or {}).get("usage") or {}
    return float(usage.get("cost") or usage.get("total_cost") or 0)


async def call_api(client, payload, api_key, schema):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    structured, last_error = True, "unknown error"
    for attempt in range(5):
        body = dict(payload)
        if structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "judge_result", "strict": True, "schema": schema}}
        try:
            response = await client.post(API_URL, headers=headers, json=body)
            if response.status_code == 400 and structured:
                structured = False
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


def prompt_for(example, condition, personas, persona_prompts, prompts):
    descriptions = "\n".join(
        prompts["candidate_line"].format(
            persona_id=pid,
            persona_name=value["name"],
            persona_description=persona_prompts[pid]["description"],
        )
        for pid, value in personas.items()
    )
    chosen = example[example["choice"]]
    evidence = prompts["choice_evidence"].format(
        option_a=example["A"], option_b=example["B"], chosen=chosen
    )
    if condition == "choice_and_explanation":
        evidence += "\n" + prompts["explanation_evidence"].format(
            what=example["what"], why=example["why"], how=example["how"]
        )
    return prompts["classification_user"].format(
        descriptions=descriptions, evidence=evidence
    )


async def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(ROOT / config["prompt_files"]["personas"], encoding="utf-8") as f:
        persona_prompts = yaml.safe_load(f)
    with open(ROOT / config["prompt_files"]["judges"], encoding="utf-8") as f:
        judge_prompts = yaml.safe_load(f)
    if not EXPERIMENT.exists():
        raise SystemExit("results/experiment.jsonl not found. Run run_experiment.py first.")
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add it.")

    RESULTS.mkdir(exist_ok=True)
    experiment_id = config["experiment_id"]
    experiment_rows = [
        row for row in latest_successes(read_jsonl(EXPERIMENT))
        if row.get("experiment_id") == experiment_id
    ]
    all_examples = aggregate_runs(experiment_rows)
    examples = [e for e in all_examples if e["actual_persona"] in config["judge_personas"]]
    rng = random.Random(config["random_seed"])
    if len(examples) > int(config["judge_max_examples"]):
        examples = rng.sample(examples, int(config["judge_max_examples"]))
    examples.sort(key=lambda x: x["example_id"])
    existing_rows = [
        row for row in latest_successes(read_jsonl(JUDGES))
        if row.get("experiment_id") == experiment_id
    ]
    existing = {row["request_id"] for row in existing_rows}
    existing_profiles = [
        row for row in latest_successes(read_jsonl(PROFILES))
        if row.get("experiment_id") == experiment_id
    ]
    prior_cost = sum(float(row.get("cost") or 0) for row in experiment_rows + existing_rows + existing_profiles)
    state = {"cost": prior_cost, "done": 0, "stopped": False}
    lock, budget_lock = asyncio.Lock(), asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["concurrency"]))
    jobs = []
    for judge in config["judge_models"]:
        for example in examples:
            for condition in ("choice_only", "choice_and_explanation"):
                rid = hashlib.sha256(
                    f"{experiment_id}|{judge}|{example['example_id']}|{condition}".encode()
                ).hexdigest()[:20]
                jobs.append((rid, judge, example, condition))
    total = len(jobs)
    prediction_labels = config["judge_personas"] + [config["judge_other_label"]]
    judge_personas = {pid: config["personas"][pid] for pid in config["judge_personas"]}
    schema = {
        "type": "object",
        "properties": {"persona": {"type": "string", "enum": prediction_labels}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["persona", "confidence"], "additionalProperties": False,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        async def run_one(job):
            rid, judge, example, condition = job
            if rid in existing:
                async with lock:
                    state["done"] += 1
                return
            async with semaphore:
                async with budget_lock:
                    if state["cost"] >= float(config["max_budget_usd"]):
                        state["stopped"] = True
                        return
                payload = {
                    "model": judge,
                    "messages": [{"role": "system", "content": judge_prompts["classification_system"]},
                                 {"role": "user", "content": prompt_for(
                                     example, condition, judge_personas, persona_prompts, judge_prompts
                                 )}],
                    "temperature": config["judge_temperature"], "max_tokens": config["max_output_tokens"],
                }
                data, error = await call_api(client, payload, api_key, schema)
                usage = (data or {}).get("usage") or {}
                try:
                    parsed = parse_json(data["choices"][0]["message"]["content"]) if data else None
                except (KeyError, IndexError, TypeError):
                    parsed = None
                predicted = (parsed or {}).get("persona")
                valid = predicted in prediction_labels
                row = {
                    "request_id": rid, "experiment_id": experiment_id,
                    "judge_model": judge, "condition": condition,
                    "example_id": example["example_id"], "experiment_model": example["model"],
                    "question_id": example["question_id"], "category": example["category"], "frame": example["frame"],
                    "actual_persona": example["actual_persona"], "predicted_persona": predicted,
                    "confidence": (parsed or {}).get("confidence"), "stability": example["stability"],
                    "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost_of(data), "status": "success" if valid else "error",
                    "error": error or (None if valid else "Response was not valid judge JSON"),
                }
                async with lock:
                    with open(JUDGES, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                    state["cost"] += row["cost"]
                    state["done"] += 1
                    print(f"{state['done']} / {total}\ncost: ${state['cost']:.4f}")

        await asyncio.gather(*(run_one(job) for job in jobs))

        # A separate, grouped description of default responses for each experiment model and judge.
        profile_existing = {r.get("request_id") for r in existing_profiles}
        profile_schema = {
            "type": "object", "properties": {
                "traits": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
                "summary": {"type": "string"}},
            "required": ["traits", "summary"], "additionalProperties": False,
        }
        for judge in config["judge_models"]:
            for model in config["experimental_models"]:
                rid = hashlib.sha256(f"{experiment_id}|profile|{judge}|{model}".encode()).hexdigest()[:20]
                if rid in profile_existing or state["cost"] >= float(config["max_budget_usd"]):
                    continue
                pool = [e for e in all_examples if e["model"] == model and e["actual_persona"] == "P0"]
                random.Random(f"{config['random_seed']}:{model}:profile").shuffle(pool)
                pool = pool[:25]
                if not pool:
                    continue
                evidence = "\n".join(
                    judge_prompts["profile_evidence_line"].format(chosen=e[e["choice"]], why=e["why"])
                    for e in pool
                )
                prompt = judge_prompts["profile_user"].format(evidence=evidence)
                payload = {"model": judge, "messages": [{"role": "user", "content": prompt}],
                           "temperature": config["judge_temperature"], "max_tokens": config["max_output_tokens"]}
                data, error = await call_api(client, payload, api_key, profile_schema)
                try:
                    parsed = parse_json(data["choices"][0]["message"]["content"]) if data else None
                except (KeyError, IndexError, TypeError):
                    parsed = None
                valid = bool(parsed and isinstance(parsed.get("traits"), list) and parsed.get("summary"))
                usage = (data or {}).get("usage") or {}
                row = {"request_id": rid, "experiment_id": experiment_id,
                       "name": "inferred_default_behavioral_profile", "judge_model": judge,
                       "experiment_model": model, "traits": (parsed or {}).get("traits", []),
                       "summary": (parsed or {}).get("summary", ""), "examples_used": len(pool),
                       "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                       "cost": cost_of(data), "status": "success" if valid else "error",
                       "error": error or (None if valid else "Response was not valid profile JSON")}
                with open(PROFILES, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                state["cost"] += row["cost"]

    if state["stopped"]:
        print(f"Stopped because tracked cost reached the ${float(config['max_budget_usd']):.2f} budget.")


if __name__ == "__main__":
    asyncio.run(main())
