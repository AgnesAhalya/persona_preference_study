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

Run the 72-request pilot, then judge and analyze it. The `$0.50` value is the total stop threshold shared by the chooser and judge stages:

```bash
docker compose run --rm -e MAX_BUDGET_USD=0.50 experiment python run_experiment.py --pilot
docker compose run --rm -e MAX_BUDGET_USD=0.50 experiment python run_judges.py --a-rate-threshold 0.5
docker compose run --rm experiment python analyze.py --mode pilot --a-rate-threshold 0.5
```

Run the default low-resource 1,440-request full experiment, then its judges and analysis. These commands share the `$2.00` total stop threshold:

```bash
docker compose run --rm -e MAX_BUDGET_USD=2.00 experiment python run_experiment.py
docker compose run --rm -e MAX_BUDGET_USD=2.00 experiment python run_judges.py --a-rate-threshold 0.5
docker compose run --rm experiment python analyze.py --mode full --a-rate-threshold 0.5
```

The judge bucket is configurable:

- `0` sends all aggregated questions.
- `0.5` or `50` sends questions where A was selected more than 50% of observations.
- `1` or `100` sends questions where A was selected every time.

For each model, condition, and question, the default run combines three frames and one run per frame: three observations. For P1-P5, the frames phrase the same persona as direct identity, role-based choice, and imagined perspective; P0 uses neutral counterparts with no system/persona prompt. The judge receives one complete batch containing the qualifying questions, their A/B rates, winning option, and—under the explanation condition—a representative explanation. Set `runs_per_condition: 3` only when you need nine observations; across all six conditions that increases the full experiment to 4,320 calls.

There are exactly five prompted personas: Mathematician, Strategist, Contrarian, Slacker, and Adversarial. P0 is the sixth experiment condition: the Assistant receives only the task message and no system/persona prompt. The judge chooses P1-P5 or `OTHER`; P0 is not a classification candidate. When the judge selects `OTHER`, it must return a short name and description for the profile it inferred instead.

## Raw HTTP logging

Every OpenRouter attempt is appended to `results/run_.../audit/raw_http_log.jsonl`, including:

- timestamp, retry number, and duration;
- exact method, URL, and JSON request body;
- HTTP version, status, response headers, and unmodified raw response body;
- reported cost and network error, if any.

The real `Authorization` value is never logged. It is recorded as `Bearer [REDACTED]`.

The log still contains full prompts and model outputs. `results/` is Git-ignored; treat downloaded artifacts as research data and review them before sharing.

Summary result files reference the raw log by filename. Prompt/data fingerprints and immutable run manifests under `results/run_.../audit/manifests/` prevent changed prompts or questions from silently reusing stale results.

## GitHub Actions

1. Put this directory in a Git repository and push it to GitHub.
2. Add `OPENROUTER_API_KEY` under **Settings → Secrets and variables → Actions**.
3. Open **Actions → Run preference experiment → Run workflow**.
4. Select `dry-run`, `pilot`, or `full`, whether to run judges, and the judge A-rate threshold.
5. Optionally enter two experiment-model IDs, two judge-model IDs, and a maximum USD budget. Blank fields use `config.yaml`.
6. Download the `preference-results-*` artifact.

Keep `OPENROUTER_API_KEY` only in GitHub Secrets. The workflow form never asks for or displays it.

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

## Timestamped result layout

Each experiment invocation creates `results/run_YYYY-MM-DD_HH-MM-SS-microsecondsZ_<mode>/`. If that run is incomplete, the same command resumes it through `results/LATEST_RUN`; after completion, the next experiment command starts a new timestamped run. Judge and analysis commands automatically use `LATEST_RUN`.

```text
run_.../
  RESULTS_REPORT.md
  chooser/success/experiment.jsonl
  chooser/failure/experiment.jsonl
  chooser/analysis/
  judges/success/judges.jsonl
  judges/success/assistant_profiles.jsonl
  judges/failure/judges.jsonl
  judges/failure/assistant_profiles.jsonl
  judges/analysis/
  audit/raw_http_log.jsonl
  audit/manifests/
  audit/completion/
```

Invalid chooser, judge, and Assistant-profile responses are retried immediately up to three semantic attempts. Only a response that remains invalid after those retries is written under `failure/`. The mounted `results/` directory makes incomplete local Docker runs resumable.

`max_budget_usd` is a spend stop threshold, not a prepaid hard cap: OpenRouter reports cost after a response, and already in-flight requests can finish. Use a conservative value; the scripts check the threshold before every semantic retry.

Analysis is threshold- and mode-specific: `0`, `0.5`, and `1`, as well as pilot and full runs, receive different judge fingerprints. `analyze.py` includes only the requested threshold and mode instead of mixing runs.
