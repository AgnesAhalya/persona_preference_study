# PRPP-40 Validation Protocol

## Goal

The goal is not to prove that every answer has one objectively correct persona. The goal is to show that independent reviewers can recognize the intended constructs, that the two options are comparably plausible, and that observed model differences are not mainly caused by wording or position.

## Stage 1: Blind construct review

Recruit at least three independent reviewers who did not write the items. Do not show them `construct_A`, `construct_B`, or the intended category.

For each option, reviewers select one label:

- Care
- Analytical
- Autonomy
- Rule
- Neutral/unclear

They also rate each pair from 1 to 5 on:

- Clarity
- Similarity of option length and detail
- Similarity of general desirability
- Confidence in their construct labels
- Whether one option sounds obviously more moral or competent

Recommended initial acceptance criteria:

- At least 80% agreement with the intended label for each construct-bearing option
- Overall inter-rater agreement of at least Fleiss' kappa 0.60
- Median clarity of at least 4/5
- Median desirability-balance rating of at least 4/5
- No unresolved reviewer warning that one option is obviously more moral or competent

These thresholds should be declared before reviewing results. If institutional guidance recommends different thresholds, document the change and rationale.

## Stage 2: Mechanical balance checks

Verify automatically that:

- There are exactly 40 unique IDs.
- Every non-control category has six items.
- Every contrast has a 3/3 A/B orientation balance.
- Every target construct occurs 18 times, nine as A and nine as B.
- Neutral controls contain only neutral labels.
- There are no duplicated options.
- Option lengths within a pair are reasonably similar.

The supplied candidate has passed these structural checks, but they must be repeated after any revision.

## Stage 3: Human pilot

If feasible, pilot the forced choices with at least 100 adult participants. If the available sample is smaller, justify it and report uncertainty rather than calling the instrument validated.

Use two questionnaire forms:

- Form 1 uses the stored A/B order.
- Form 2 reverses every A/B pair.

Randomly assign participants to a form. Collect:

- Choice
- Choice confidence
- Perceived difficulty of the choice
- Optional one-sentence reason

Do not ask participants to adopt the LLM personas in the first human pilot. This stage tests item quality and position effects, not the main experimental hypothesis.

Flag an item for revision when:

- Reversing A/B order materially changes the canonical choice rate.
- One option receives more than 90% of choices without a theoretical reason.
- Participants frequently describe the pair as incomparable or confusing.
- Reasons show that participants interpreted an option differently from the intended construct.

## Stage 4: Held-out model pilot

Use models that are not the two final experimental models when possible. Run the items without a persona and with A/B reversal.

This stage checks:

- JSON response reliability
- Refusal or invalid-response frequency
- Position bias
- Whether wording accidentally names or reveals a construct
- Whether neutral controls behave as arbitrary/equivalent choices

Do not select items merely because they maximize separation on the final experiment models. That would overfit the instrument to the models being evaluated.

## Stage 5: Freeze the instrument

After revisions:

1. Assign a version such as `PRPP-40 v1.0`.
2. Freeze item wording and metadata before the main experiment.
3. Record a SHA-256 hash of the final dataset.
4. Store the validation results and exclusion log.
5. Preregister the primary metrics and analysis decisions where possible.
6. Do not silently replace poorly performing items after seeing the main results.

## Suggested reviewer data fields

For every item and reviewer, collect:

```text
reviewer_id
item_id
label_A
label_B
label_confidence_1_to_5
clarity_1_to_5
length_balance_1_to_5
desirability_balance_1_to_5
obvious_morality_or_competence_bias
comments
```

## Primary scoring after validation

For a response to a construct-bearing item:

- Map the displayed choice back to its canonical A/B position.
- Look up the selected construct.
- Count it toward that construct's choice rate.

Recommended summaries:

- Choice rate for each construct under each persona
- Within-contrast choice rates, such as care over analytical
- Change relative to the P0 default condition
- Run stability
- Frame consistency
- A/B-order effect
- Neutral-control position effect

Neutral controls must be excluded from persona-alignment accuracy because neither option has a target construct.

## Reporting language

Before validation, use:

> We created a 40-item candidate pairwise task-preference instrument using balanced, theory-informed constructs.

After completing and reporting validation, use:

> We developed and pilot-validated a 40-item pairwise task-preference instrument for the four decision orientations studied here.

Do not call PRPP-40 a clinical test, a general personality test, or proof that a model has a true personality.

