# Persona Preference Experiment — Results Report

- Run folder: `run_2026-08-16_21-51-16-447155Z_full`
- Mode: `full`
- Experiment models: google/gemini-2.5-flash-lite, qwen/qwen3-14b
- Judge models: openai/gpt-4.1-mini, openai/gpt-5-nano
- A-rate threshold: `0.5`
- Chooser success records: **1440**
- Chooser failure records: **0**
- Judge success records: **48**
- Judge failure records: **0**

## Failure records

No terminal failures were recorded after automatic retries.

## Persona classifications

### P0 — Assistant (no persona prompt)

Chooser condition (hidden from judge): **P0**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P2 (confidence 85.0%)
  - qwen/qwen3-14b: P2 (confidence 85.0%)
  - Combined view: agreement → P2
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P2 (confidence 90.0%)
  - qwen/qwen3-14b: P1 (confidence 90.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P2 (confidence 77.0%)
  - qwen/qwen3-14b: P1 (confidence 92.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: OTHER (confidence 62.0%)
    - Related profile: Collaborative Facilitator
    - Traits: Empathetic, Collaborative, Process-driven, Governance-aware, Systems thinker
    - Description: A people-centered, process-oriented facilitator who coordinates teams, aligns with governance policies, and crafts practical, empathetic solutions within established frameworks.
  - qwen/qwen3-14b: P1 (confidence 78.0%)
  - Combined view: disagreement between experiment models

### P1 — Mathematician

Chooser condition (hidden from judge): **P1**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 95.0%)
  - qwen/qwen3-14b: P1 (confidence 95.0%)
  - Combined view: agreement → P1
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P1 (confidence 99.0%)
  - qwen/qwen3-14b: P1 (confidence 99.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 93.0%)
  - qwen/qwen3-14b: P1 (confidence 92.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P1 (confidence 92.0%)
  - qwen/qwen3-14b: P1 (confidence 93.0%)
  - Combined view: agreement → P1

### P2 — Strategist

Chooser condition (hidden from judge): **P2**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 95.0%)
  - qwen/qwen3-14b: P2 (confidence 85.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P2 (confidence 95.0%)
  - qwen/qwen3-14b: P2 (confidence 95.0%)
  - Combined view: agreement → P2
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 85.0%)
  - qwen/qwen3-14b: P1 (confidence 85.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P2 (confidence 92.0%)
  - qwen/qwen3-14b: P2 (confidence 86.0%)
  - Combined view: agreement → P2

### P3 — Contrarian

Chooser condition (hidden from judge): **P3**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P3 (confidence 85.0%)
  - qwen/qwen3-14b: P3 (confidence 90.0%)
  - Combined view: agreement → P3
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P3 (confidence 95.0%)
  - qwen/qwen3-14b: P3 (confidence 95.0%)
  - Combined view: agreement → P3
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P3 (confidence 75.0%)
  - qwen/qwen3-14b: P3 (confidence 82.0%)
  - Combined view: agreement → P3
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P3 (confidence 92.0%)
  - qwen/qwen3-14b: P3 (confidence 92.0%)
  - Combined view: agreement → P3

### P4 — Slacker

Chooser condition (hidden from judge): **P4**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 85.0%)
  - qwen/qwen3-14b: P1 (confidence 85.0%)
  - Combined view: agreement → P1
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 95.0%)
  - qwen/qwen3-14b: P4 (confidence 95.0%)
  - Combined view: agreement → P4
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 92.0%)
  - qwen/qwen3-14b: P1 (confidence 88.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 88.0%)
  - qwen/qwen3-14b: P4 (confidence 82.0%)
  - Combined view: agreement → P4

### P5 — Adversarial

Chooser condition (hidden from judge): **P5**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 90.0%)
  - qwen/qwen3-14b: P3 (confidence 90.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P5 (confidence 95.0%)
  - qwen/qwen3-14b: P5 (confidence 95.0%)
  - Combined view: agreement → P5
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 75.0%)
  - qwen/qwen3-14b: P1 (confidence 78.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P5 (confidence 92.0%)
  - qwen/qwen3-14b: P5 (confidence 86.0%)
  - Combined view: agreement → P5

## How to read the detailed analysis

- [Chooser analysis](chooser/analysis/) — preferences, P0 differences, frames, and order effects.
- [Judge analysis](judges/analysis/) — accuracy, agreement, confusion matrices, OTHER profiles, and HTTP summary.
- [Chooser successes](chooser/success/experiment.jsonl) and [chooser failures](chooser/failure/experiment.jsonl).
- [Judge successes](judges/success/) and [judge failures](judges/failure/).
- [Audit records](audit/) — raw redacted HTTP logs, manifests, and completion records.

## Interpretation warning

Persona classification accuracy and preference differences are experimental outputs, not proof that a model has a personality. P0 is the no-persona comparison condition.
