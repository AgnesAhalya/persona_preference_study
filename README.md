# Persona preference experiment

The experiment runs locally, in Docker, or as a manually triggered GitHub Actions workflow. Before making API requests, replace the model placeholders in `config.yaml` with OpenRouter model IDs.

See [EXPERIMENT.md](EXPERIMENT.md) for a simple explanation of the study, the dry-run/pilot/full modes, request counts, judge runs, and cost protection.

The active `questions.json` contains **PRPP-40**, a balanced 40-item candidate instrument spanning Care, Analytical, Autonomy, and Rule decision contrasts. The experiment applies six paper-inspired personas to these shared questions, plus the P0 no-prompt baseline. The dataset card and validation protocol are in `dataset_candidate/`. The previous 40-question version is preserved as `dataset_candidate/questions_original.json`.

## Prompt files

All LLM-facing wording is kept outside the Python scripts:

- `prompts/personas.yaml` — persona instructions and judge-facing descriptions
- `prompts/experiment.yaml` — question frames, experiment system prompt, and response template
- `prompts/judges.yaml` — classification and default-profile judge prompts

Their paths are configured under `prompt_files` in `config.yaml`. Rebuild the Docker image after editing a prompt file so the changed wording is copied into the image.

## Run with Docker Compose

Create a local `.env` file:

```text
OPENROUTER_API_KEY=your-key-here
```

Build and check the experiment without API calls:

```bash
docker compose build
docker compose run --rm experiment
```

Run the 60-request pilot:

```bash
docker compose run --rm experiment python run_experiment.py --pilot
```

Run the full experiment, judges, and analysis:

```bash
docker compose run --rm experiment python run_experiment.py
docker compose run --rm experiment python run_judges.py
docker compose run --rm experiment python analyze.py
```

The host `results/` directory is mounted into the container, so output survives when the container exits and interrupted experiments remain resumable.

## Run with plain Docker

```bash
docker build -t preference-experiment .
docker run --rm --env-file .env -v "$PWD/results:/app/results" preference-experiment python run_experiment.py --pilot
```

## Run with GitHub Actions

1. Push the project to a GitHub repository.
2. In **Settings → Secrets and variables → Actions**, create a repository secret named `OPENROUTER_API_KEY`.
3. Open **Actions → Run preference experiment → Run workflow**.
4. Choose `dry-run`, `pilot`, or `full`. Optionally enable the judges.
5. Download the `preference-results-*` artifact when the run finishes.

The workflow is manual and defaults to `dry-run`, which sends no API requests. Pilot and full runs use `max_budget_usd` from `config.yaml`. GitHub-hosted runners are temporary, so each workflow run starts with an empty `results/` directory; resumability applies within a run and to local Docker runs that reuse the same mounted directory.
