# Persona preference experiment

This project measures whether an LLM's pairwise task preferences change across a no-system-prompt Assistant condition and five paper-inspired persona prompts. It runs locally in Docker, in Google Colab for a one-model/one-judge trial, or through a manual GitHub Actions workflow.

The active `questions.json` is **PRPP-40**, a balanced but not yet human-validated 40-item candidate instrument. Its Care, Analytical, Autonomy, and Rule constructs are behavioral measurement axes—not six persona ground-truth labels. See `dataset_candidate/DATASET_CARD.md` and `dataset_candidate/VALIDATION_PROTOCOL.md` before treating it as a research instrument.

## Configure models

Replace the placeholders in `config.yaml` with valid OpenRouter model IDs:

```yaml
experimental_models:
  - provider/experiment-model-1
  - provider/experiment-model-2
judge_models:
  - provider/judge-model-1
  - provider/judge-model-2
```

Create `.env` locally:

```text
OPENROUTER_API_KEY=your-key-here
```

## Docker commands

Build and perform a zero-API dry run:

```bash
docker compose build
docker compose run --rm experiment
```

Run the 72-request pilot:

```bash
docker compose run --rm experiment python run_experiment.py --pilot
```

Run the default low-resource 1,440-request full experiment:

```bash
docker compose run --rm experiment python run_experiment.py
```

Judge and analyze using the default A-majority bucket:

```bash
docker compose run --rm experiment python run_judges.py --a-rate-threshold 0.5
docker compose run --rm experiment python analyze.py --a-rate-threshold 0.5
```

The judge bucket is configurable:

- `0` sends all aggregated questions.
- `0.5` or `50` sends questions where A was selected more than 50% of observations.
- `1` or `100` sends questions where A was selected every time.

For each model, condition, and question, the default run combines three frames and one run per frame: three observations. The judge receives one complete batch containing the qualifying questions, their A/B rates, winning option, and—under the explanation condition—a representative explanation. Set `runs_per_condition: 3` only when you need nine observations; across all six conditions that increases the full experiment to 4,320 calls.

There are exactly five prompted personas: Mathematician, Strategist, Contrarian, Slacker, and Adversarial. P0 is the sixth experiment condition: the Assistant receives only the task message and no system/persona prompt. The judge chooses P1-P5 or `OTHER`; P0 is not a classification candidate. When the judge selects `OTHER`, it must return a short name and description for the profile it inferred instead.

## Raw HTTP logging

Every OpenRouter attempt is appended to `results/raw_http_log.jsonl`, including:

- timestamp, retry number, and duration;
- exact method, URL, and JSON request body;
- HTTP version, status, response headers, and unmodified raw response body;
- reported cost and network error, if any.

The real `Authorization` value is never logged. It is recorded as `Bearer [REDACTED]`.

The log still contains full prompts and model outputs. `results/` is Git-ignored; treat downloaded artifacts as research data and review them before sharing.

Summary result files reference the raw log by filename. Prompt/data fingerprints and immutable run manifests under `results/manifests/` prevent changed prompts or questions from silently reusing stale results.

## GitHub Actions

1. Put this directory in a Git repository and push it to GitHub.
2. Add `OPENROUTER_API_KEY` under **Settings → Secrets and variables → Actions**.
3. Open **Actions → Run preference experiment → Run workflow**.
4. Select `dry-run`, `pilot`, or `full`, whether to run judges, and the judge A-rate threshold.
5. Download the `preference-results-*` artifact.

The workflow defaults to dry-run and sends no API requests. Real runs fail visibly when responses are missing or the budget stops the experiment; judges will not run on an incomplete batch. Artifacts are still uploaded for diagnosis.

## Colab trial

Open `colab_local_trial.ipynb` in Google Colab. A T4 GPU is recommended, but the notebook can fall back to CPU at lower speed. It uses one local Hugging Face model for responses and a different local model as judge, so it needs no OpenRouter key. It is a plumbing trial, not a questionnaire validation or final accuracy estimate.

The trial performs 36 response generations, 10 persona classifications, and one separate P0 Assistant-profile generation. It captures raw local model output in three downloadable JSONL files and writes `colab_trial_summary.json`. Move to Docker/OpenRouter only when that summary reports `"complete": true`.

## Prompt files

LLM-facing text is separated from the Python programs:

- `prompts/personas.yaml` — five prompted personas (P1-P5) plus the empty P0 Assistant entry
- `prompts/experiment.yaml` — three frames and response format
- `prompts/judges.yaml` — whole-batch classification and P0 Assistant profile prompts

Rebuild the Docker image after changing source, configuration, or prompts.

## Main result files

- `experiment.jsonl` — normalized experiment responses
- `judges.jsonl` — whole-batch persona classifications
- `inferred_default_behavioral_profile.jsonl` — inferred P0 Assistant profile descriptions
- `judge_predictions.csv` and `other_profiles.csv` — judge decisions and profiles proposed for `OTHER`
- `analysis_context.json` — the exact experiment fingerprint, judge fingerprint, and threshold used
- `raw_http_log.jsonl` — exact redacted HTTP request/response audit log
- `manifests/*.json` — full input snapshots and fingerprints
- `completion_*.json` and `judge_completion_*.json` — completeness checks
- CSV and PNG files — construct preferences, frame/order effects, buckets, accuracy, abstention, and confusion matrices

The mounted `results/` directory makes local Docker runs resumable.

Analysis is threshold- and mode-specific: `0`, `0.5`, and `1`, as well as pilot and full runs, receive different judge fingerprints. `analyze.py` includes only the requested threshold and mode instead of mixing runs.
