import json
import tempfile
import unittest
from pathlib import Path

import httpx

from experiment_utils import (
    append_jsonl,
    canonical_hash,
    experiment_conditions,
    judge_fingerprint_inputs,
    matching_successes,
    openrouter_request,
    recorded_cost,
    retry_delay,
)
from run_judges import (
    aggregate_task_preferences,
    build_batches,
    include_in_bucket,
    normalize_threshold,
    valid_behavioral_profile,
    valid_classification,
)
from run_experiment import experiment_messages, parse_json, valid_choice_response


class ThresholdTests(unittest.TestCase):
    def test_threshold_normalization(self):
        self.assertEqual(normalize_threshold(0), 0)
        self.assertEqual(normalize_threshold("0.5"), 0.5)
        self.assertEqual(normalize_threshold("50"), 0.5)
        self.assertEqual(normalize_threshold(1), 1)
        self.assertEqual(normalize_threshold(100), 1)
        with self.assertRaises(ValueError):
            normalize_threshold(101)

    def test_bucket_rules(self):
        self.assertTrue(include_in_bucket(0.2, 0))
        self.assertTrue(include_in_bucket(5 / 9, 0.5))
        self.assertFalse(include_in_bucket(4 / 9, 0.5))
        self.assertTrue(include_in_bucket(1, 1))
        self.assertFalse(include_in_bucket(8 / 9, 1))


class PromptAndClassificationTests(unittest.TestCase):
    def test_assistant_condition_has_no_system_message(self):
        prompts = {
            "system": "Instruction.{persona_instruction}",
            "persona_instruction": " Persona: {persona_prompt}",
            "user": "{frame}\nA: {display_a}\nB: {display_b}",
        }
        frames = {
            "persona": "You are acting as the {persona_name}. Choose.",
            "baseline": "Choose.",
        }
        assistant = experiment_messages("Assistant", "", prompts, frames, "One", "Two")
        persona = experiment_messages("Mathematician", "Use rigor", prompts, frames, "One", "Two")
        self.assertEqual([message["role"] for message in assistant], ["user"])
        self.assertEqual([message["role"] for message in persona], ["system", "user"])
        self.assertNotIn("Mathematician", assistant[0]["content"])
        self.assertIn("Mathematician", persona[1]["content"])

    def test_parse_json_accepts_null_provider_content(self):
        self.assertIsNone(parse_json(None))

    def test_choice_requires_all_nonempty_fields(self):
        self.assertTrue(valid_choice_response({
            "choice": "A", "what": "Task", "why": "Reason", "how": "Method",
        }))
        self.assertFalse(valid_choice_response({
            "choice": "A", "what": "Task", "why": "", "how": "Method",
        }))
        self.assertFalse(valid_choice_response(["A"]))

    def test_other_requires_a_related_profile(self):
        labels = ["P1", "P2", "OTHER"]
        self.assertTrue(valid_classification({
            "persona": "OTHER",
            "confidence": 0.7,
            "other_profile_name": "Care-focused helper",
            "other_profile_description": "Consistently prefers supportive cooperative tasks.",
            "other_profile_traits": ["caring", "helpful", "patient", "social", "supportive"],
        }, labels, "OTHER"))
        self.assertFalse(valid_classification({
            "persona": "OTHER",
            "confidence": 0.7,
            "other_profile_name": "",
            "other_profile_description": "",
            "other_profile_traits": [],
        }, labels, "OTHER"))
        self.assertTrue(valid_classification({
            "persona": "P1",
            "confidence": 0.8,
            "other_profile_name": "",
            "other_profile_description": "",
            "other_profile_traits": [],
        }, labels, "OTHER"))

    def test_behavioral_profile_requires_five_nonempty_traits(self):
        self.assertTrue(valid_behavioral_profile({
            "traits": ["a", "b", "c", "d", "e"],
            "summary": "A short profile.",
        }))
        self.assertFalse(valid_behavioral_profile({
            "traits": ["a", "b", "c", "d", ""],
            "summary": "A short profile.",
        }))
        self.assertFalse(valid_behavioral_profile(["not", "an", "object"]))


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "experimental_models": ["model"],
            "include_baseline": True,
            "baseline": {"id": "P0", "name": "Default"},
            "personas": {"P1": {}},
            "frames": ["F1", "F2", "F3"],
            "runs_per_condition": 3,
        }
        self.questions = [
            {"id": "Q1", "category": "x", "A": "Fight", "B": "Help"},
            {"id": "Q2", "category": "x", "A": "Argue", "B": "Support"},
        ]

    def rows(self):
        rows = []
        for persona in experiment_conditions(self.config):
            for question in self.questions:
                index = 0
                for frame in self.config["frames"]:
                    for run in range(1, 4):
                        # P1/Q1 has A=5/9; P1/Q2 has A=4/9. Other groups are unanimous A.
                        a_limit = 5 if (persona, question["id"]) == ("P1", "Q1") else 4
                        if persona == "P0":
                            a_limit = 9
                        choice = "A" if index < a_limit else "B"
                        rows.append({
                            "request_id": f"{persona}-{question['id']}-{frame}-{run}",
                            "model": "model",
                            "question_id": question["id"],
                            "persona": persona,
                            "frame": frame,
                            "run": run,
                            "run_mode": "full",
                            "canonical_choice": choice,
                            "what": "what",
                            "why": "why",
                            "how": "how",
                        })
                        index += 1
        return rows

    def test_three_frames_and_runs_are_aggregated(self):
        preferences, incomplete, mode, question_count, observations = aggregate_task_preferences(
            self.rows(), self.config, self.questions
        )
        self.assertEqual(incomplete, [])
        self.assertEqual(mode, "full")
        self.assertEqual(question_count, 2)
        self.assertEqual(observations, 9)
        p1_q1 = next(
            item for item in preferences
            if item["actual_persona"] == "P1" and item["question_id"] == "Q1"
        )
        self.assertEqual(p1_q1["a_count"], 5)
        self.assertEqual(p1_q1["b_count"], 4)
        self.assertEqual(p1_q1["preferred_choice"], "A")

        all_batches = build_batches(preferences, self.config, 0)
        p1_all = next(batch for batch in all_batches if batch["actual_persona"] == "P1")
        self.assertEqual(len(p1_all["items"]), 2)

        majority_batches = build_batches(preferences, self.config, 0.5)
        p1_majority = next(batch for batch in majority_batches if batch["actual_persona"] == "P1")
        self.assertEqual([item["question_id"] for item in p1_majority["items"]], ["Q1"])

        unanimous_batches = build_batches(preferences, self.config, 1)
        p0_unanimous = next(batch for batch in unanimous_batches if batch["actual_persona"] == "P0")
        self.assertEqual(len(p0_unanimous["items"]), 2)

    def test_missing_frame_run_is_reported(self):
        rows = self.rows()[:-1]
        _, incomplete, _, _, _ = aggregate_task_preferences(rows, self.config, self.questions)
        self.assertEqual(len(incomplete), 1)


class RawHttpLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_request_and_raw_response_are_logged_with_redacted_key(self):
        seen_authorization = []

        class FakeRequest:
            method = "POST"
            url = "https://openrouter.ai/api/v1/chat/completions"

            def __init__(self, headers, body):
                self.headers = headers
                self.content = json.dumps(body).encode()

        class FakeResponse:
            status_code = 200
            http_version = "HTTP/1.1"
            headers = {"x-test-response": "yes"}
            is_error = False

            def __init__(self, request):
                self.request = request
                self._data = {
                    "choices": [{"message": {"content": '{"choice":"A"}'}}],
                    "usage": {"cost": 0.0123},
                }
                self.text = json.dumps(self._data)

            def json(self):
                return self._data

        class FakeClient:
            async def post(self, url, headers, json):
                seen_authorization.append(headers["Authorization"])
                return FakeResponse(FakeRequest(headers, json))

        logs = []

        async def log_attempt(row):
            logs.append(row)

        schema = {
            "type": "object",
            "properties": {"choice": {"type": "string"}},
            "required": ["choice"],
        }
        data, error, attempts = await openrouter_request(
            FakeClient(),
            {"model": "test/model", "messages": [{"role": "user", "content": "hello"}]},
            "secret-key",
            request_context={
                "experiment_id": "e",
                "fingerprint": "f",
                "request_id": "r",
                "stage": "experiment",
                "model": "test/model",
            },
            schema_name="test",
            schema=schema,
            log_attempt=log_attempt,
        )
        self.assertIsNone(error)
        self.assertEqual(data["usage"]["cost"], 0.0123)
        self.assertEqual(attempts[0]["cost"], 0.0123)
        self.assertEqual(seen_authorization, ["Bearer secret-key"])
        self.assertEqual(logs[0]["request"]["headers"]["Authorization"], "Bearer [REDACTED]")
        self.assertNotIn("secret-key", logs[0]["request"]["raw_body"])
        self.assertIn("response_format", json.loads(logs[0]["request"]["raw_body"]))
        self.assertEqual(logs[0]["response"]["status_code"], 200)
        self.assertIn('"choices"', logs[0]["response"]["raw_body"])
        self.assertEqual(logs[0]["response"]["headers"]["x-test-response"], "yes")


class CostAndFingerprintTests(unittest.TestCase):
    def test_retry_after_is_respected_and_capped(self):
        response = type("Response", (), {"headers": {"retry-after": "120"}})()
        self.assertEqual(retry_delay(response, 1), 60)
        self.assertEqual(retry_delay(None, 2), 1)

    def test_raw_attempt_costs_include_errors_and_do_not_double_count_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            results = root / "results.jsonl"
            append_jsonl(raw, {"experiment_id": "e", "request_id": "r1", "cost": 0.1})
            append_jsonl(raw, {"experiment_id": "e", "request_id": "r1", "cost": 0.2})
            append_jsonl(results, {"experiment_id": "e", "request_id": "r1", "cost": 0.3})
            append_jsonl(results, {"experiment_id": "e", "request_id": "legacy", "cost": 0.4})
            self.assertAlmostEqual(recorded_cost(raw, [results], "e"), 0.7)

    def test_fingerprint_changes_when_prompt_changes(self):
        first = canonical_hash({"prompt": "one", "questions": [1]})
        second = canonical_hash({"prompt": "two", "questions": [1]})
        self.assertNotEqual(first, second)

    def test_judge_fingerprint_and_selection_are_threshold_specific(self):
        config = {
            "judge_models": ["judge"],
            "judge_personas": ["P1"],
            "judge_other_label": "OTHER",
            "judge_temperature": 0,
            "max_output_tokens": 100,
        }
        prompts = {"classification_user": "prompt"}
        fingerprint_0 = canonical_hash(
            judge_fingerprint_inputs(config, "experiment", prompts, 0, "full", 40, 3)
        )
        fingerprint_05 = canonical_hash(
            judge_fingerprint_inputs(config, "experiment", prompts, 0.5, "full", 40, 3)
        )
        self.assertNotEqual(fingerprint_0, fingerprint_05)
        rows = [
            {
                "request_id": "zero", "status": "success",
                "experiment_fingerprint": "experiment", "judge_fingerprint": fingerprint_0,
            },
            {
                "request_id": "half", "status": "success",
                "experiment_fingerprint": "experiment", "judge_fingerprint": fingerprint_05,
            },
        ]
        selected = matching_successes(
            rows,
            experiment_fingerprint="experiment",
            judge_fingerprint=fingerprint_05,
        )
        self.assertEqual([row["request_id"] for row in selected], ["half"])

    def test_judge_fingerprint_is_mode_specific(self):
        config = {
            "judge_models": ["judge"],
            "judge_personas": ["P1"],
            "judge_other_label": "OTHER",
            "judge_temperature": 0,
            "max_output_tokens": 100,
        }
        prompts = {"classification_user": "prompt"}
        pilot = canonical_hash(
            judge_fingerprint_inputs(config, "experiment", prompts, 0.5, "pilot", 2, 3)
        )
        full = canonical_hash(
            judge_fingerprint_inputs(config, "experiment", prompts, 0.5, "full", 40, 3)
        )
        self.assertNotEqual(pilot, full)


if __name__ == "__main__":
    unittest.main()
