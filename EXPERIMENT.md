# Persona Preference Experiment

## What this experiment studies

This project tests whether a language model changes its choices when it is given different personas.

The model repeatedly chooses between option A and option B. It answers under seven conditions:

- **P0 — Default:** no special persona
- **P1 — Aura-inspired:** prefers reflection, meaning, perspective, and expressive tasks
- **P2 — Mathematician:** prefers formal reasoning, precision, and verifiable answers
- **P3 — Strategist:** prefers influence, negotiation, competitive planning, and leverage
- **P4 — Contrarian:** prefers overlooked, unconventional, or surprising options
- **P5 — Slacker:** prefers short, simple tasks requiring minimal effort
- **P6 — Adversarial:** prefers confrontation, disruption, and exposing weakness

The six prompted personas are paraphrased, safety-bounded versions inspired by the reference paper rather than verbatim copies. The experiment uses two models, 40 questions, three ways of wording each question, and three repeated runs. A separate judge experiment tests whether two judge models can identify the prompted persona from the choices and explanations.

## Which run mode should I select?

### Dry run

Select **dry-run** when you want to check the setup safely.

- Sends **zero API requests**.
- Costs **nothing**.
- Prints the number of models, questions, personas, frames, runs, and total requests.
- Does not create experiment responses.

Use this first after changing `config.yaml`, Docker, or the GitHub workflow.

With the current configuration, it reports that a full experiment would make **5,040 requests**.

### Pilot

Select **pilot** for a small real test before running the full experiment.

- Uses both experiment models.
- Uses only the first **2 questions**.
- Uses all 7 conditions, including P0.
- Uses all 3 question frames.
- Uses 1 run instead of 3.
- Sends **84 experiment requests**.
- Uses the real OpenRouter API and can cost money.

Use this to inspect the response format, verify the selected models, and estimate cost before committing to the full study.

If **Run judges** is also enabled, the pilot can produce up to **288 classification requests** plus **4 default-profile requests**.

### Full

Select **full** when the pilot results look correct and you are ready to collect the research dataset.

- Uses both experiment models.
- Uses all 40 questions.
- Uses all 7 conditions, including P0.
- Uses all 3 question frames.
- Repeats every condition 3 times.
- Sends **5,040 experiment requests**.
- Uses the real OpenRouter API and can cost money.

The calculation is:

```text
2 models × 40 questions × 7 conditions × 3 frames × 3 runs
= 5,040 experiment requests
```

## What does “Run judges” mean?

The **Run judges** checkbox starts the judge experiment after the main experiment finishes.

For each question, persona, frame, and experiment model, the three repeated choices are combined using majority vote. For example:

```text
A, A, B → A
```

Each judge then tries to identify the persona under two conditions:

1. **Choice only:** the judge sees the question and selected answer.
2. **Choice and explanation:** the judge also sees the short `what`, `why`, and `how` responses.

P0 is excluded from classification and is analyzed separately. The judge chooses among P1-P6, or returns `OTHER` when the evidence is insufficient. The random baseline across the six prompted personas is approximately 16.7%.

For a full experiment there are 1,440 classification-eligible aggregated examples after P0 is excluded. The current `judge_max_examples: 600` setting selects the same random sample of 600 examples for both judges and both conditions. This creates up to:

```text
600 examples × 2 judges × 2 conditions
= 2,400 judge classification requests
```

Four additional requests describe the inferred default behavioral profile: one for each combination of two experiment models and two judge models.

Leave **Run judges** unchecked when you only want to collect the main experiment responses. You can run `run_judges.py` later using the saved results.

## Recommended order

1. Replace `MODEL_1`, `MODEL_2`, `JUDGE_1`, and `JUDGE_2` in `config.yaml` with valid OpenRouter model IDs.
2. Run **dry-run** to check the calculated experiment size.
3. Run **pilot** without judges and inspect `results/experiment.jsonl`.
4. Run the pilot with judges if you want to check the classification output.
5. Run **full** only after the pilot looks correct.
6. Download and keep the GitHub Actions result artifact.

## Cost protection

`max_budget_usd` in `config.yaml` is the tracked spending limit. Its current value is:

```yaml
max_budget_usd: 2.0
```

The scripts add OpenRouter-reported costs and stop starting new work when the tracked total reaches the limit. A few requests can already be in progress because the experiment uses concurrent requests, so treat this as a safety limit rather than an exact guarantee.

Dry-run never needs an API key. Pilot, full, and judge runs require `OPENROUTER_API_KEY`.

## Result files

The main outputs are saved in `results/`:

- `experiment.jsonl` — every experiment response
- `judges.jsonl` — persona predictions from the judge models
- `inferred_default_behavioral_profile.jsonl` — descriptions inferred from default responses
- CSV files — calculated measurements
- PNG files — graphs and confusion matrices

Each successful API request has a stable request ID. If a local Docker run stops, running the same command again skips successful requests already stored in the mounted `results/` directory.

GitHub-hosted runners start with an empty results directory on every new workflow run. Download the artifact after each GitHub run so the collected data is not lost.
