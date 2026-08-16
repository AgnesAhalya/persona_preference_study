# Experiment design

## Research question

The study asks whether the same language model reveals different pairwise task preferences under a no-system-prompt Assistant condition and five prompted decision perspectives:

- P0 — Assistant (no system or persona prompt)
- P1 — Mathematician
- P2 — Strategist
- P3 — Contrarian
- P4 — Slacker
- P5 — Adversarial

There are six experiment conditions but only five personas. The judge may return P1-P5 or `OTHER`; P0 is never a classification candidate. If it returns `OTHER`, it must provide a short name and description for the related behavioral profile suggested by the evidence.

PRPP-40's four constructs are outcome axes, not persona labels. Therefore, construct selection rates and whole-profile judge accuracy answer different questions. Persona inference is exploratory until the questionnaire is human validated.

## Three frames and preference averages

Every prompted persona receives every question through three persona-language frames. The same persona instruction remains in the system message; the user frame refers to that persona in a different linguistic form:

1. Direct identity: “You are acting as the Mathematician. From this identity and its priorities, which option would you choose?”
2. Role-based choice: “Choose the option that the Mathematician would choose.”
3. Perspective-taking: “Imagine you are in the position of the Mathematician. Which option would you choose from that perspective?”

P0 Assistant uses neutral counterparts of these frames and never receives a persona/system prompt.

The low-resource default asks each of the three frames once. For one model, persona, and question, this produces:

```text
3 frames × 1 run = 3 observations
```

A is encoded as 1 and B as 0 when calculating the A rate. For example, two A choices and one B choice produce:

```text
A rate = 2 / 3 = 66.7%
B rate = 1 / 3 = 33.3%
Preferred option = A
```

Here, A and B always mean the canonical options stored in `questions.json`. Display-order swapping is mapped back before percentages are calculated.

Because the number of observations is odd, the preferred option cannot tie. The judge runner refuses incomplete frame/run groups instead of silently calculating an average from partial data. `runs_per_condition` remains configurable; use `3` for nine observations when the extra reliability justifies three times as many experiment calls.

## Configurable judge bucket

`judge_a_rate_threshold` controls which aggregated questions enter the whole-batch judge prompt. It can also be overridden with `run_judges.py --a-rate-threshold`.

| Value | Questions placed in the bucket |
|---:|---|
| `0` | All questions |
| `0.5` or `50` | Only questions with A rate greater than 50% |
| `1` or `100` | Only questions with A rate exactly 100% |

The default is `0.5`. The threshold is about how frequently **A** was selected, not general confidence. Every included item contains the question, A and B text, A%, B%, preferred answer, counts, and total observations.

One batch is built for each experiment-model/persona combination. The judge sees the complete bucket at once and must infer the persona from the pattern across the batch. Two conditions are tested:

1. `choice_only` — aggregate choices and percentages
2. `choice_and_explanation` — the same data plus a representative explanation for each preferred option

Every threshold and run mode produces a separate judge fingerprint. Analysis selects that exact fingerprint, so `0`, `0.5`, and `1` or pilot and full results cannot be combined accidentally.

For two experiment models, five prompted personas, and two judge models:

```text
2 experiment models × 5 personas = 10 classification batches
10 batches × 2 judge models × 2 conditions = 40 classification calls
```

P0 adds four separate profile-description calls: two experiment models × two judge models. The complete judge stage therefore makes 44 calls.

## Run modes

### Dry run

- Sends zero API requests and costs nothing.
- Calculates the mode, fingerprint, dimensions, and request count.
- Default dry-run reports the low-resource full design: 1,440 experiment requests.
- `--dry-run --pilot` reports 72.

### Pilot

- Two experiment models
- First two questions
- Five prompted personas plus the no-system-prompt Assistant
- All three frames
- One run per frame
- 72 experiment calls
- Three observations per model/persona/question average

Use the pilot to verify schemas, raw logs, bucket membership, and costs—not to estimate research accuracy.

### Full

```text
2 models × 40 questions × 6 personas × 3 frames × 1 run
= 1,440 experiment calls
```

Each question average uses three observations. Setting `runs_per_condition: 3` changes this to nine observations and 4,320 calls.

## Raw HTTP audit log and backup behavior

`results/raw_http_log.jsonl` is append-only. A line is written immediately after every HTTP attempt, including retries and structured-output fallback requests. Each line stores the exact JSON body sent and the raw response body received. The API key is replaced with `[REDACTED]`.

Normalized output files remain easier to analyze, while the raw log supports audits and reparsing. Manifests store all experiment inputs and a SHA-256 fingerprint. Request IDs include that fingerprint, so changing questions, prompts, frames, temperatures, or model configuration creates new IDs instead of reusing incompatible responses.

Fallback JSON is validated locally. A preference response is successful only when it contains A/B plus nonempty `what`, `why`, and `how` fields. `OTHER` requires a nonempty related-profile name and description; Assistant profiles require five nonempty traits and a nonempty summary.

## Completeness and cost protection

The experiment counts costs from every logged HTTP attempt, including billable responses that fail local JSON validation. It permits at most the configured concurrency worth of requests to be in flight when the budget boundary is crossed.

If the experiment stops at the budget or any expected response remains invalid, it exits unsuccessfully and writes a completion report. Rerunning the identical command resumes successful request IDs. The judge refuses incomplete source data.

## Primary outputs

- Construct selection rates, calculated from `construct_A` and `construct_B`
- Within-contrast selection rates
- Difference from the P0 Assistant condition
- Aggregated preference strength across the configured frames and runs
- Frame consistency
- A/B display-order effects
- Neutral-control position effects
- Judge accuracy, coverage, `OTHER` abstention rate, selective accuracy, and confusion matrices
- Raw HTTP status, duration, attempt, and cost summaries

Do not interpret raw canonical A frequency as a psychological trait. Construct metadata is never included in experiment or judge prompts.
