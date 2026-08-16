# Persona Preference Experiment — Results Report

- Run folder: `run_2026-08-16_21-32-13-143869Z_pilot`
- Mode: `pilot`
- Experiment models: google/gemini-2.5-flash-lite, qwen/qwen3-14b
- Judge models: openai/gpt-4.1-mini, openai/gpt-5-nano
- A-rate threshold: `0.5`
- Chooser success records: **72**
- Chooser failure records: **0**
- Judge success records: **48**
- Judge failure records: **0**

## Failure records

No terminal failures were recorded after automatic retries.

## Persona classifications

### P0 — Assistant (no persona prompt)

Chooser condition (hidden from judge): **P0**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: OTHER (confidence 90.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Non-committal, Ambiguous, Inconsistent, Neutral
    - Description: A profile characterized by lack of clear preferences or strong inclinations, showing no decisive pattern across tasks or questions.
  - qwen/qwen3-14b: OTHER (confidence 90.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Ambivalent, Non-committal, Inconsistent, Low engagement
    - Description: A profile characterized by lack of strong preferences or clear choices, indicating uncertainty or ambivalence.
  - Combined view: agreement → OTHER
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: OTHER (confidence 70.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Ambivalent, Non-committal, Inconsistent, Low engagement
    - Description: A profile characterized by lack of strong preferences or clear choices, indicating uncertainty or ambivalence.
  - qwen/qwen3-14b: OTHER (confidence 70.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Non-committal, Ambivalent, Inconsistent, Neutral
    - Description: A profile characterized by lack of strong preferences or clear choices, indicating uncertainty or ambivalence.
  - Combined view: agreement → OTHER
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P4 (confidence 72.0%)
  - qwen/qwen3-14b: P4 (confidence 62.0%)
  - Combined view: agreement → P4
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 62.0%)
  - qwen/qwen3-14b: P4 (confidence 60.0%)
  - Combined view: agreement → P4

### P1 — Mathematician

Chooser condition (hidden from judge): **P1**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 85.0%)
  - qwen/qwen3-14b: P1 (confidence 90.0%)
  - Combined view: agreement → P1
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P1 (confidence 95.0%)
  - qwen/qwen3-14b: P1 (confidence 95.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 92.0%)
  - qwen/qwen3-14b: P1 (confidence 92.0%)
  - Combined view: agreement → P1
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P1 (confidence 92.0%)
  - qwen/qwen3-14b: P1 (confidence 92.0%)
  - Combined view: agreement → P1

### P2 — Strategist

Chooser condition (hidden from judge): **P2**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 85.0%)
  - qwen/qwen3-14b: P2 (confidence 85.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P2 (confidence 90.0%)
  - qwen/qwen3-14b: P2 (confidence 90.0%)
  - Combined view: agreement → P2
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P1 (confidence 92.0%)
  - qwen/qwen3-14b: P2 (confidence 79.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P2 (confidence 82.0%)
  - qwen/qwen3-14b: P2 (confidence 82.0%)
  - Combined view: agreement → P2

### P3 — Contrarian

Chooser condition (hidden from judge): **P3**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: OTHER (confidence 90.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Non-committal, Ambivalent, Inconsistent, Neutral
    - Description: A profile characterized by lack of strong preferences or clear choices, indicating uncertainty or ambivalence.
  - qwen/qwen3-14b: OTHER (confidence 90.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Ambivalent, Non-committal, Inconsistent, Low engagement
    - Description: A profile characterized by lack of clear preferences or strong inclinations, showing uncertainty or ambivalence across tasks.
  - Combined view: agreement → OTHER
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: OTHER (confidence 70.0%)
    - Related profile: Indecisive
    - Traits: Uncertain, Ambivalent, Non-committal, Inconsistent, Low engagement
    - Description: A profile characterized by lack of strong preferences or clear choices, indicating uncertainty or ambivalence.
  - qwen/qwen3-14b: OTHER (confidence 70.0%)
    - Related profile: Indecisive
    - Traits: Ambivalent, Uncertain, Non-committal, Inconsistent, Neutral
    - Description: A persona characterized by lack of strong preferences or clear choices, showing ambivalence or uncertainty across tasks.
  - Combined view: agreement → OTHER
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P4 (confidence 60.0%)
  - qwen/qwen3-14b: P4 (confidence 60.0%)
  - Combined view: agreement → P4
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 70.0%)
  - qwen/qwen3-14b: P4 (confidence 60.0%)
  - Combined view: agreement → P4

### P4 — Slacker

Chooser condition (hidden from judge): **P4**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: P2 (confidence 85.0%)
  - qwen/qwen3-14b: OTHER (confidence 85.0%)
    - Related profile: Empathetic Analyst
    - Traits: Empathetic, Analytical, Supportive, Detail-oriented, Interpersonal
    - Description: A persona that combines analytical skills with a strong preference for interpersonal understanding and support, focusing on both data-driven insights and human-centered approaches.
  - Combined view: disagreement between experiment models
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 85.0%)
  - qwen/qwen3-14b: P4 (confidence 90.0%)
  - Combined view: agreement → P4
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P2 (confidence 72.0%)
  - qwen/qwen3-14b: P2 (confidence 85.0%)
  - Combined view: agreement → P2
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P4 (confidence 72.0%)
  - qwen/qwen3-14b: P4 (confidence 85.0%)
  - Combined view: agreement → P4

### P5 — Adversarial

Chooser condition (hidden from judge): **P5**

- **openai/gpt-4.1-mini / choice_only**
  - google/gemini-2.5-flash-lite: OTHER (confidence 70.0%)
    - Related profile: Empathetic Collaborator
    - Traits: empathetic, supportive, collaborative, people-oriented, conflict-averse
    - Description: A persona that prioritizes interpersonal support and emotional well-being over technical or competitive tasks.
  - qwen/qwen3-14b: P1 (confidence 80.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-4.1-mini / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P5 (confidence 90.0%)
  - qwen/qwen3-14b: P1 (confidence 85.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-5-nano / choice_only**
  - google/gemini-2.5-flash-lite: P2 (confidence 66.0%)
  - qwen/qwen3-14b: P1 (confidence 65.0%)
  - Combined view: disagreement between experiment models
- **openai/gpt-5-nano / choice_and_explanation**
  - google/gemini-2.5-flash-lite: P5 (confidence 82.0%)
  - qwen/qwen3-14b: P1 (confidence 65.0%)
  - Combined view: disagreement between experiment models

## How to read the detailed analysis

- [Chooser analysis](chooser/analysis/) — preferences, P0 differences, frames, and order effects.
- [Judge analysis](judges/analysis/) — accuracy, agreement, confusion matrices, OTHER profiles, and HTTP summary.
- [Chooser successes](chooser/success/experiment.jsonl) and [chooser failures](chooser/failure/experiment.jsonl).
- [Judge successes](judges/success/) and [judge failures](judges/failure/).
- [Audit records](audit/) — raw redacted HTTP logs, manifests, and completion records.

## Interpretation warning

Persona classification accuracy and preference differences are experimental outputs, not proof that a model has a personality. P0 is the no-persona comparison condition.
