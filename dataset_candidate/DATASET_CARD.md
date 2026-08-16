# PRPP-40 Dataset Card

## Dataset name

**PRPP-40: Persona-Relevant Pairwise Preferences, 40-item candidate instrument**

## Status

Candidate instrument, version 0.1. It has been structurally checked but has **not yet been human validated**. It must not currently be described as a validated psychological questionnaire.

## Purpose

PRPP-40 measures choices across four decision constructs:

- **Care:** empathy, support, cooperation, and concern for well-being
- **Analytical:** evidence, reasoning, accuracy, and cognitively demanding analysis
- **Autonomy:** independence, agency, flexibility, and self-direction
- **Rule:** consistency, procedure, accountability, and compliance

These constructs describe the options in the questionnaire; they are measurement axes rather than persona labels. The active experiment applies six paper-inspired personas - Aura-inspired, Mathematician, Strategist, Contrarian, Slacker, and Adversarial - to the same questions. The default condition is an unprompted baseline whose behavioral profile is inferred separately.

## Why a new instrument is needed

Existing resources cover only parts of the research design:

- General personality surveys are usually rating scales rather than task choices.
- Pairwise preference datasets usually compare answer quality, not which task a persona prefers.
- Persona benchmarks usually test whether a response matches a user profile, not revealed preference under an active model persona.
- The reference paper uses pairwise choices between a large pool of real tasks, but does not provide a compact 40-item persona-oriented questionnaire.

PRPP-40 therefore combines a published pairwise task-choice paradigm with explicitly defined, theory-informed constructs. It is a researcher-constructed instrument and should be reported as such.

## Design

There are four target constructs. Their six possible pairwise contrasts are each represented by six questions:

| Contrast | Items |
|---|---:|
| Care vs analytical | 6 |
| Care vs autonomy | 6 |
| Care vs rule | 6 |
| Analytical vs autonomy | 6 |
| Analytical vs rule | 6 |
| Autonomy vs rule | 6 |
| Neutral controls | 4 |
| **Total** | **40** |

Within every six-item contrast, each construct appears as option A three times and option B three times. Across the 36 construct-bearing questions, each construct appears in 18 questions: nine times as A and nine times as B.

The four controls present deliberately equivalent alternatives. They measure position/order sensitivity and should not be scored as evidence for any persona.

## Item format

Each item contains:

| Field | Meaning |
|---|---|
| `id` | Stable item identifier |
| `category` | Construct contrast or neutral control |
| `context` | Scenario domain |
| `A`, `B` | The two task choices shown to the model |
| `construct_A`, `construct_B` | Researcher labels used only for validation and analysis |

The `construct_A` and `construct_B` fields must never be included in a model or judge prompt.

## Construction principles

- Options describe actions or tasks rather than abstract value words.
- Both options are intended to be reasonable and socially acceptable.
- Options within a pair are written at similar specificity and length.
- No option explicitly names its intended persona.
- Contexts vary across work, education, technology, public service, research, and decision-making.
- A/B placement is balanced in the source data and should still be randomized during elicitation.

## Theoretical basis

The items are newly written and are not copied from existing questionnaires. Their construct definitions are informed by:

- Schwartz's basic values framework, especially benevolence/universalism, self-direction, conformity, and security: https://doi.org/10.9707/2307-0919.1116
- Need for Cognition as a basis for preference for effortful analytical activity: https://doi.org/10.1037/0022-3514.42.1.116
- The pairwise revealed-task-preference paradigm used in *Probing Persona-Dependent Preferences in Language Models*: https://arxiv.org/abs/2605.13339

These sources motivate the constructs and method; they do not validate PRPP-40 itself.

## Recommended uses

- Comparing preference patterns across prompted personas
- Measuring stability over repeated runs and prompt frames
- Testing whether persona identity can be inferred from choices or explanations
- Studying the unprompted model as an inferred default behavioral profile

## Uses to avoid

- Diagnosing human personality
- Claiming that a model possesses a true personality
- Treating the intended construct labels as ground truth before validation
- Comparing raw A-choice rates without mapping choices back after A/B swapping
- Using explanations as independent preference evidence when they were generated after the choice

## Validation requirement

Before the main study, follow `VALIDATION_PROTOCOL.md`. Report reviewer agreement, item revisions, exclusions, pilot sample, and final version. Preserve this version so changes remain auditable.

## Relationship to the active project

PRPP-40 is now the active root-level `questions.json`. The previous question set is preserved as `dataset_candidate/questions_original.json`. The experiment runner ignores the extra construct metadata and uses the required `id`, `category`, `A`, and `B` fields.

## Provenance and release

- Created for the Persona Preference Inference research project
- Initial candidate version: 2026-08-16
- Item text: newly authored for this project
- Recommended release format: JSON plus CSV
- Recommended release license: CC BY 4.0, subject to confirmation by the project owner and institution
