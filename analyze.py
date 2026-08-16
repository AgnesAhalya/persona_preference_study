import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from experiment_utils import (
    apply_runtime_overrides,
    canonical_hash,
    experiment_conditions,
    experiment_fingerprint_inputs,
    judge_fingerprint_inputs,
    latest_successes,
    matching_successes,
    read_jsonl,
)
from run_judges import aggregate_task_preferences, build_batches, normalize_threshold


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def latest_records(rows):
    latest = {}
    for row in rows:
        if row.get("request_id"):
            latest[row["request_id"]] = row
    return list(latest.values())


def save_bar(df, x, y, title, ylabel, filename, hue=None):
    if df.empty:
        return
    pivot = df.pivot(index=x, columns=hue, values=y) if hue else df.set_index(x)[[y]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0, 1))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()


def experiment_completeness(all_rows, config, questions, fingerprint, mode_override=None):
    fingerprint_rows = [
        row for row in latest_records(all_rows)
        if row.get("experiment_fingerprint") == fingerprint
    ]
    mode = mode_override or (
        "full" if any(row.get("run_mode") == "full" for row in fingerprint_rows) else "pilot"
    )
    current = experiment_rows_for_mode(
        fingerprint_rows, config, questions, fingerprint, mode, successes_only=False
    )
    runs = int(config["runs_per_condition"]) if mode == "full" else 1
    selected_questions = questions if mode == "full" else questions[:2]
    expected = (
        len(config["experimental_models"])
        * len(selected_questions)
        * len(experiment_conditions(config))
        * len(config["frames"])
        * runs
    )
    successes = sum(row.get("status") == "success" for row in current)
    errors = sum(row.get("status") == "error" for row in current)
    result = pd.DataFrame([{
        "experiment_id": config["experiment_id"],
        "experiment_fingerprint": fingerprint,
        "mode": mode,
        "expected_requests": expected,
        "successful_requests": successes,
        "error_requests": errors,
        "missing_requests": max(expected - successes, 0),
        "complete": successes == expected,
    }])
    result.to_csv(RESULTS / "experiment_completeness.csv", index=False)
    return mode


def experiment_rows_for_mode(
    rows, config, questions, fingerprint, mode, successes_only=True
):
    runs = int(config["runs_per_condition"]) if mode == "full" else 1
    selected_questions = questions if mode == "full" else questions[:2]
    question_ids = {question["id"] for question in selected_questions}
    conditions = set(experiment_conditions(config))
    records = latest_successes(rows) if successes_only else latest_records(rows)
    return [
        row for row in records
        if row.get("experiment_fingerprint") == fingerprint
        and row.get("model") in config["experimental_models"]
        and row.get("question_id") in question_ids
        and row.get("persona") in conditions
        and row.get("frame") in config["frames"]
        and 1 <= int(row.get("run", 0)) <= runs
    ]


def analyze_task_preferences(preferences, questions, config, threshold):
    preference_df = pd.DataFrame(preferences)
    if preference_df.empty:
        print("No complete aggregated task preferences to analyze.")
        return
    preference_df.to_csv(RESULTS / "task_preference_profiles.csv", index=False)

    batches = build_batches(preferences, config, threshold)
    bucket_rows = []
    for batch in batches:
        included = {item["question_id"] for item in batch["items"]}
        for item in [p for p in preferences if p["model"] == batch["model"] and p["actual_persona"] == batch["actual_persona"]]:
            bucket_rows.append({
                "model": batch["model"],
                "persona": batch["actual_persona"],
                "question_id": item["question_id"],
                "a_rate": item["a_rate"],
                "b_rate": item["b_rate"],
                "included_in_judge_bucket": item["question_id"] in included,
                "a_rate_threshold": threshold,
            })
    pd.DataFrame(bucket_rows).to_csv(RESULTS / "judge_bucket_membership.csv", index=False)

    question_map = {question["id"]: question for question in questions}
    construct_rows = []
    for row in preferences:
        question = question_map[row["question_id"]]
        if question["category"] == "neutral_control":
            continue
        construct_rows.extend([
            {
                "model": row["model"],
                "persona": row["actual_persona"],
                "question_id": row["question_id"],
                "category": row["category"],
                "construct": question["construct_A"],
                "selection_rate": row["a_rate"],
            },
            {
                "model": row["model"],
                "persona": row["actual_persona"],
                "question_id": row["question_id"],
                "category": row["category"],
                "construct": question["construct_B"],
                "selection_rate": row["b_rate"],
            },
        ])
    constructs = pd.DataFrame(construct_rows)
    construct_summary = constructs.groupby(
        ["model", "persona", "construct"], as_index=False
    ).agg(selection_rate=("selection_rate", "mean"), opportunities=("selection_rate", "size"))
    construct_summary.to_csv(RESULTS / "construct_preference_rates.csv", index=False)
    construct_plot = construct_summary.copy()
    construct_plot["model_persona"] = (
        construct_plot["model"].astype(str) + "\n" + construct_plot["persona"].astype(str)
    )
    save_bar(
        construct_plot,
        "model_persona",
        "selection_rate",
        "Construct preference rates",
        "Mean selection rate",
        "construct_preference_rates.png",
        "construct",
    )
    constructs.groupby(
        ["model", "persona", "category", "construct"], as_index=False
    ).agg(selection_rate=("selection_rate", "mean"), opportunities=("selection_rate", "size")).to_csv(
        RESULTS / "within_contrast_preference_rates.csv", index=False
    )

    baseline_id = config["baseline"]["id"]
    p0 = preference_df[preference_df["actual_persona"] == baseline_id][
        ["model", "question_id", "a_rate", "preferred_choice"]
    ].rename(columns={"a_rate": "p0_a_rate", "preferred_choice": "p0_preferred_choice"})
    prompted = preference_df[preference_df["actual_persona"] != baseline_id].merge(
        p0, on=["model", "question_id"], how="inner"
    )
    if not p0.empty:
        prompted["delta_a_rate_from_P0"] = prompted["a_rate"] - prompted["p0_a_rate"]
        prompted["winner_differs_from_P0"] = (
            prompted["preferred_choice"] != prompted["p0_preferred_choice"]
        )
        prompted.groupby(["model", "actual_persona"], as_index=False).agg(
            mean_absolute_a_rate_change=("delta_a_rate_from_P0", lambda values: values.abs().mean()),
            winner_difference_rate=("winner_differs_from_P0", "mean"),
            questions=("question_id", "size"),
        ).rename(columns={"actual_persona": "persona"}).to_csv(
            RESULTS / "persona_difference_from_P0.csv", index=False
        )

    preference_df.groupby(["model", "actual_persona"], as_index=False).agg(
        mean_preference_strength=("preferred_rate", "mean"),
        unanimous_question_rate=("preferred_rate", lambda values: (values == 1).mean()),
        questions=("question_id", "size"),
    ).rename(columns={"actual_persona": "persona"}).to_csv(
        RESULTS / "aggregated_preference_strength.csv", index=False
    )


def analyze_frames_and_order(success_rows, config, questions):
    frame_groups = defaultdict(list)
    for row in success_rows:
        if row.get("canonical_choice") in {"A", "B"}:
            frame_groups[(row["model"], row["question_id"], row["persona"], row["frame"])].append(row)
    frame_winners = []
    for (model, question_id, persona, frame), rows in frame_groups.items():
        counts = Counter(row["canonical_choice"] for row in rows)
        winner = "A" if counts["A"] > counts["B"] else "B"
        frame_winners.append({
            "model": model,
            "question_id": question_id,
            "persona": persona,
            "frame": frame,
            "winner": winner,
            "winner_rate": counts[winner] / len(rows),
        })
    frame_df = pd.DataFrame(frame_winners)
    if not frame_df.empty:
        consistency = []
        for (model, question_id, persona), group in frame_df.groupby(["model", "question_id", "persona"]):
            counts = group["winner"].value_counts()
            consistency.append({
                "model": model,
                "question_id": question_id,
                "persona": persona,
                "frames_available": len(group),
                "frame_consistency": counts.max() / len(group),
                "all_frames_agree": group["winner"].nunique() == 1,
            })
        consistency_df = pd.DataFrame(consistency)
        consistency_df.to_csv(RESULTS / "frame_consistency_detail.csv", index=False)
        consistency_df.groupby(["model", "persona"], as_index=False).agg(
            mean_frame_consistency=("frame_consistency", "mean"),
            all_frames_agree_rate=("all_frames_agree", "mean"),
            questions=("question_id", "size"),
        ).to_csv(RESULTS / "frame_consistency.csv", index=False)

    raw = pd.DataFrame(success_rows)
    if raw.empty:
        return
    raw["canonical_A"] = raw["canonical_choice"].eq("A").astype(float)
    raw["displayed_A"] = raw["model_choice"].eq("A").astype(float)
    order = raw.groupby(["model", "persona", "display_order"], as_index=False).agg(
        canonical_A_rate=("canonical_A", "mean"),
        displayed_A_rate=("displayed_A", "mean"),
        observations=("request_id", "size"),
    )
    order.to_csv(RESULTS / "order_rates.csv", index=False)
    pivot = order.pivot_table(
        index=["model", "persona"], columns="display_order", values="canonical_A_rate"
    ).reset_index()
    if "AB" in pivot and "BA" in pivot:
        pivot["order_effect_AB_minus_BA"] = pivot["AB"] - pivot["BA"]
    pivot.to_csv(RESULTS / "order_effects.csv", index=False)

    neutral_ids = {q["id"] for q in questions if q["category"] == "neutral_control"}
    neutral = raw[raw["question_id"].isin(neutral_ids)]
    if not neutral.empty:
        neutral.groupby(["model", "persona"], as_index=False).agg(
            displayed_A_rate=("displayed_A", "mean"), observations=("request_id", "size")
        ).to_csv(RESULTS / "neutral_control_position_rates.csv", index=False)


def judge_analysis(rows, actual_labels, predicted_labels, other_label):
    if not rows:
        pd.DataFrame(columns=[
            "actual_persona", "predicted_persona", "confidence",
            "other_profile_name", "other_profile_description",
        ]).to_csv(RESULTS / "judge_predictions.csv", index=False)
        pd.DataFrame(columns=[
            "actual_persona", "predicted_persona", "other_profile_name",
            "other_profile_description",
        ]).to_csv(RESULTS / "other_profiles.csv", index=False)
        pd.DataFrame(columns=[
            "judge_model", "condition", "accuracy", "coverage", "abstention_rate",
            "selective_accuracy", "mean_confidence", "examples", "random_persona_baseline",
        ]).to_csv(RESULTS / "judge_accuracy.csv", index=False)
        pd.DataFrame(columns=[
            "condition", "judge_1", "judge_2", "agreement", "paired_profiles",
        ]).to_csv(RESULTS / "judge_agreement.csv", index=False)
        pd.DataFrame(columns=[
            "judge_model", "condition", "actual_persona", "predicted_persona", "count",
        ]).to_csv(RESULTS / "confusion_matrices.csv", index=False)
        print("No successful judge rows to analyze.")
        return
    judges = pd.DataFrame(rows)
    judges.to_csv(RESULTS / "judge_predictions.csv", index=False)
    other_profiles = judges[judges["predicted_persona"] == other_label]
    other_profiles.to_csv(RESULTS / "other_profiles.csv", index=False)
    judges["correct"] = judges["actual_persona"] == judges["predicted_persona"]
    judges["abstained"] = judges["predicted_persona"] == other_label
    summary_rows = []
    for (judge_model, condition), group in judges.groupby(["judge_model", "condition"]):
        covered = group[~group["abstained"]]
        summary_rows.append({
            "judge_model": judge_model,
            "condition": condition,
            "accuracy": group["correct"].mean(),
            "coverage": 1 - group["abstained"].mean(),
            "abstention_rate": group["abstained"].mean(),
            "selective_accuracy": covered["correct"].mean() if len(covered) else float("nan"),
            "mean_confidence": group["confidence"].mean(),
            "examples": len(group),
            "random_persona_baseline": 1 / len(actual_labels),
        })
    accuracy = pd.DataFrame(summary_rows)
    accuracy.to_csv(RESULTS / "judge_accuracy.csv", index=False)
    save_bar(
        accuracy,
        "condition",
        "accuracy",
        "Whole-batch persona identification accuracy",
        "Accuracy",
        "judge_accuracy.png",
        "judge_model",
    )

    agreements = []
    judge_models = sorted(judges["judge_model"].unique())
    for condition, group in judges.groupby("condition"):
        pivot = group.pivot_table(
            index="profile_id", columns="judge_model", values="predicted_persona", aggfunc="first"
        ).dropna()
        for index, first in enumerate(judge_models):
            for second in judge_models[index + 1:]:
                if first in pivot and second in pivot:
                    agreements.append({
                        "condition": condition,
                        "judge_1": first,
                        "judge_2": second,
                        "agreement": (pivot[first] == pivot[second]).mean(),
                        "paired_profiles": len(pivot),
                    })
    pd.DataFrame(
        agreements,
        columns=["condition", "judge_1", "judge_2", "agreement", "paired_profiles"],
    ).to_csv(RESULTS / "judge_agreement.csv", index=False)

    confusion_parts = []
    for (judge, condition), group in judges.groupby(["judge_model", "condition"]):
        matrix = pd.crosstab(group["actual_persona"], group["predicted_persona"]).reindex(
            index=actual_labels, columns=predicted_labels, fill_value=0
        )
        long = matrix.stack().rename("count").reset_index()
        long.insert(0, "condition", condition)
        long.insert(0, "judge_model", judge)
        confusion_parts.append(long)

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(predicted_labels)), predicted_labels)
        ax.set_yticks(range(len(actual_labels)), actual_labels)
        ax.set_xlabel("Predicted persona")
        ax.set_ylabel("Actual persona")
        ax.set_title(f"Whole-batch confusion matrix\n{judge} — {condition}")
        for i in range(len(actual_labels)):
            for j in range(len(predicted_labels)):
                ax.text(j, i, matrix.iloc[i, j], ha="center", va="center")
        fig.colorbar(image, ax=ax)
        plt.tight_layout()
        safe_name = "".join(character if character.isalnum() else "_" for character in judge)
        plt.savefig(RESULTS / f"confusion_{safe_name}_{condition}.png", dpi=150)
        plt.close()
    if confusion_parts:
        pd.concat(confusion_parts, ignore_index=True).to_csv(
            RESULTS / "confusion_matrices.csv", index=False
        )


def analyze_http_log(rows, experiment_id, fingerprints):
    selected = [
        row for row in rows
        if row.get("experiment_id") == experiment_id
        and row.get("fingerprint") in fingerprints
    ]
    if not selected:
        pd.DataFrame(columns=[
            "stage", "model", "http_status", "attempts", "cost", "mean_duration_ms",
        ]).to_csv(RESULTS / "raw_http_log_summary.csv", index=False)
        return
    summary = []
    groups = defaultdict(list)
    for row in selected:
        response = row.get("response") or {}
        groups[(row.get("stage"), row.get("model"), response.get("status_code"))].append(row)
    for (stage, model, status), group in groups.items():
        summary.append({
            "stage": stage,
            "model": model,
            "http_status": status,
            "attempts": len(group),
            "cost": sum(float(row.get("cost") or 0) for row in group),
            "mean_duration_ms": sum(float(row.get("duration_ms") or 0) for row in group) / len(group),
        })
    pd.DataFrame(summary).to_csv(RESULTS / "raw_http_log_summary.csv", index=False)


def main(args):
    RESULTS.mkdir(exist_ok=True)
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = apply_runtime_overrides(yaml.safe_load(handle))
    with (ROOT / "questions.json").open(encoding="utf-8") as handle:
        questions = json.load(handle)
    with (ROOT / config["prompt_files"]["personas"]).open(encoding="utf-8") as handle:
        persona_prompts = yaml.safe_load(handle)
    with (ROOT / config["prompt_files"]["experiment"]).open(encoding="utf-8") as handle:
        experiment_prompts = yaml.safe_load(handle)
    with (ROOT / config["prompt_files"]["judges"]).open(encoding="utf-8") as handle:
        judge_prompts = yaml.safe_load(handle)
    fingerprint = canonical_hash(
        experiment_fingerprint_inputs(config, questions, persona_prompts, experiment_prompts)
    )
    threshold = normalize_threshold(
        config["judge_a_rate_threshold"] if args.a_rate_threshold is None else args.a_rate_threshold
    )
    all_experiment_rows = read_jsonl(RESULTS / "experiment.jsonl")
    mode = experiment_completeness(
        all_experiment_rows, config, questions, fingerprint, args.mode
    )
    successful_rows = experiment_rows_for_mode(
        all_experiment_rows, config, questions, fingerprint, mode
    )
    question_count = len(questions) if mode == "full" else 2
    observations_per_question = len(config["frames"]) * (
        int(config["runs_per_condition"]) if mode == "full" else 1
    )
    judge_fingerprint = canonical_hash(judge_fingerprint_inputs(
        config,
        fingerprint,
        judge_prompts,
        threshold,
        mode,
        question_count,
        observations_per_question,
    ))
    if successful_rows:
        preferences, incomplete, _, _, _ = aggregate_task_preferences(
            successful_rows, config, questions, mode_override=mode
        )
        if incomplete:
            print(f"Warning: {len(incomplete)} aggregated groups are incomplete; excluded from profiles.")
        analyze_task_preferences(preferences, questions, config, threshold)
        analyze_frames_and_order(successful_rows, config, questions)
    else:
        print("No successful responses for the current experiment fingerprint.")

    judge_rows = matching_successes(
        read_jsonl(RESULTS / "judges.jsonl"),
        experiment_fingerprint=fingerprint,
        judge_fingerprint=judge_fingerprint,
    )
    judge_analysis(
        judge_rows,
        config["judge_personas"],
        config["judge_personas"] + [config["judge_other_label"]],
        config["judge_other_label"],
    )

    default_profiles = matching_successes(
        read_jsonl(RESULTS / "inferred_default_behavioral_profile.jsonl"),
        experiment_fingerprint=fingerprint,
        judge_fingerprint=judge_fingerprint,
    )
    if default_profiles:
        pd.DataFrame(default_profiles).to_csv(RESULTS / "default_behavioral_profiles.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "judge_model", "experiment_model", "traits", "summary", "a_rate_threshold",
        ]).to_csv(RESULTS / "default_behavioral_profiles.csv", index=False)
    analysis_context = {
        "experiment_id": config["experiment_id"],
        "experiment_fingerprint": fingerprint,
        "judge_fingerprint": judge_fingerprint,
        "a_rate_threshold": threshold,
        "judge_rows": len(judge_rows),
        "default_profile_rows": len(default_profiles),
    }
    (RESULTS / "analysis_context.json").write_text(
        json.dumps(analysis_context, indent=2) + "\n", encoding="utf-8"
    )
    analyze_http_log(
        read_jsonl(RESULTS / "raw_http_log.jsonl"),
        config["experiment_id"],
        {fingerprint, judge_fingerprint},
    )
    print(f"Analysis files written to {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a-rate-threshold",
        help="Use the same judge bucket threshold: 0, 0.5/50, or 1/100",
    )
    parser.add_argument("--mode", choices=("pilot", "full"), help="Expected experiment mode")
    main(parser.parse_args())
