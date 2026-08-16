import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def read_latest_successes(path):
    latest = {}
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("status") == "success" and row.get("request_id"):
                    latest[row["request_id"]] = row
            except json.JSONDecodeError:
                pass
    return list(latest.values())


def aggregate_experiment(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("canonical_choice") in {"A", "B"}:
            groups[(row["model"], row["question_id"], row["persona"], row["frame"])].append(row)
    aggregated = []
    for (model, question, persona, frame), group in groups.items():
        counts = Counter(row["canonical_choice"] for row in group)
        majority = sorted(counts, key=lambda c: (-counts[c], c))[0]
        aggregated.append({"model": model, "question_id": question, "persona": persona, "frame": frame,
                           "choice": majority, "stability": counts[majority] / len(group),
                           "unanimous": len(counts) == 1, "runs_available": len(group)})
    return pd.DataFrame(aggregated)


def save_bar(df, x, y, title, ylabel, filename, hue=None):
    if df.empty:
        return
    pivot = df.pivot(index=x, columns=hue, values=y) if hue else df.set_index(x)[[y]]
    ax = pivot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS / filename, dpi=150)
    plt.close()


def experiment_analysis(exp):
    if exp.empty:
        print("No successful experiment rows to analyze.")
        return

    preference = (exp.assign(chose_A=exp["choice"].eq("A").astype(float))
                    .groupby(["model", "persona"], as_index=False)["chose_A"].mean()
                    .rename(columns={"chose_A": "A_preference_rate"}))
    preference.to_csv(RESULTS / "preference_rates.csv", index=False)
    save_bar(preference, "persona", "A_preference_rate", "A preference rate by persona", "Proportion choosing A",
             "preference_rates.png", "model")

    p0 = exp[exp["persona"] == "P0"][["model", "question_id", "frame", "choice"]].rename(columns={"choice": "p0_choice"})
    differences = exp[exp["persona"] != "P0"].merge(p0, on=["model", "question_id", "frame"])
    differences["differs_from_P0"] = differences["choice"] != differences["p0_choice"]
    diff_summary = differences.groupby(["model", "persona"], as_index=False)["differs_from_P0"].mean()
    diff_summary.to_csv(RESULTS / "persona_difference_from_P0.csv", index=False)
    save_bar(diff_summary, "persona", "differs_from_P0", "Preference change from default", "Proportion different from P0",
             "persona_difference_from_P0.png", "model")

    stability = exp.groupby(["model", "persona"], as_index=False).agg(
        mean_majority_share=("stability", "mean"), unanimous_rate=("unanimous", "mean"), examples=("choice", "size"))
    stability.to_csv(RESULTS / "run_stability.csv", index=False)
    save_bar(stability, "persona", "mean_majority_share", "Run stability", "Mean majority share",
             "run_stability.png", "model")

    frame_rows = []
    for (model, question, persona), group in exp.groupby(["model", "question_id", "persona"]):
        counts = group["choice"].value_counts()
        frame_rows.append({"model": model, "question_id": question, "persona": persona,
                           "frames_available": len(group), "frame_consistency": counts.max() / len(group),
                           "all_frames_agree": group["choice"].nunique() == 1})
    frame_detail = pd.DataFrame(frame_rows)
    frame_summary = frame_detail.groupby(["model", "persona"], as_index=False).agg(
        mean_frame_consistency=("frame_consistency", "mean"), all_frames_agree_rate=("all_frames_agree", "mean"))
    frame_summary.to_csv(RESULTS / "frame_consistency.csv", index=False)
    save_bar(frame_summary, "persona", "mean_frame_consistency", "Consistency across frames", "Mean majority share",
             "frame_consistency.png", "model")

    models = sorted(exp["model"].unique())
    agreement_rows = []
    for i, model_1 in enumerate(models):
        for model_2 in models[i + 1:]:
            left = exp[exp["model"] == model_1][["question_id", "persona", "frame", "choice"]]
            right = exp[exp["model"] == model_2][["question_id", "persona", "frame", "choice"]]
            paired = left.merge(right, on=["question_id", "persona", "frame"], suffixes=("_1", "_2"))
            agreement_rows.append({"model_1": model_1, "model_2": model_2, "agreement":
                                   (paired["choice_1"] == paired["choice_2"]).mean() if len(paired) else float("nan"),
                                   "paired_examples": len(paired)})
    pd.DataFrame(agreement_rows, columns=["model_1", "model_2", "agreement", "paired_examples"]).to_csv(
        RESULTS / "experiment_model_agreement.csv", index=False)


def judge_analysis(rows, actual_labels, predicted_labels):
    if not rows:
        print("No successful judge rows to analyze.")
        return
    judges = pd.DataFrame(rows)
    judges["correct"] = judges["actual_persona"] == judges["predicted_persona"]
    accuracy = judges.groupby(["judge_model", "condition"], as_index=False).agg(
        accuracy=("correct", "mean"), examples=("correct", "size"))
    accuracy["random_baseline"] = 1 / len(actual_labels)
    accuracy.to_csv(RESULTS / "judge_accuracy.csv", index=False)
    save_bar(accuracy, "condition", "accuracy", "Persona identification accuracy", "Accuracy",
             "judge_accuracy.png", "judge_model")

    judge_models = sorted(judges["judge_model"].unique())
    agreements = []
    if len(judge_models) >= 2:
        for condition, group in judges.groupby("condition"):
            pivot = group.pivot_table(index="example_id", columns="judge_model", values="predicted_persona", aggfunc="first").dropna()
            for i, judge_1 in enumerate(judge_models):
                for judge_2 in judge_models[i + 1:]:
                    if judge_1 in pivot and judge_2 in pivot:
                        agreements.append({"condition": condition, "judge_1": judge_1, "judge_2": judge_2,
                                           "agreement": (pivot[judge_1] == pivot[judge_2]).mean(), "paired_examples": len(pivot)})
    pd.DataFrame(agreements, columns=["condition", "judge_1", "judge_2", "agreement", "paired_examples"]).to_csv(
        RESULTS / "judge_agreement.csv", index=False)

    confusion_parts = []
    for (judge, condition), group in judges.groupby(["judge_model", "condition"]):
        matrix = pd.crosstab(group["actual_persona"], group["predicted_persona"]).reindex(
            index=actual_labels, columns=predicted_labels, fill_value=0)
        long = matrix.stack().rename("count").reset_index()
        long.insert(0, "condition", condition)
        long.insert(0, "judge_model", judge)
        confusion_parts.append(long)

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix.values, cmap="Blues")
        ax.set_xticks(range(len(predicted_labels)), predicted_labels)
        ax.set_yticks(range(len(actual_labels)), actual_labels)
        ax.set_xlabel("Predicted persona")
        ax.set_ylabel("Actual persona")
        ax.set_title(f"Confusion matrix\n{judge} — {condition}")
        for i in range(len(actual_labels)):
            for j in range(len(predicted_labels)):
                ax.text(j, i, matrix.iloc[i, j], ha="center", va="center")
        fig.colorbar(image, ax=ax)
        plt.tight_layout()
        safe_name = "".join(c if c.isalnum() else "_" for c in judge)
        plt.savefig(RESULTS / f"confusion_{safe_name}_{condition}.png", dpi=150)
        plt.close()
    if confusion_parts:
        pd.concat(confusion_parts, ignore_index=True).to_csv(RESULTS / "confusion_matrices.csv", index=False)


def main():
    RESULTS.mkdir(exist_ok=True)
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    experiment_rows = [
        row for row in read_latest_successes(RESULTS / "experiment.jsonl")
        if row.get("experiment_id") == config["experiment_id"]
    ]
    aggregate = aggregate_experiment(experiment_rows)
    experiment_analysis(aggregate)
    judge_rows = [
        row for row in read_latest_successes(RESULTS / "judges.jsonl")
        if row.get("experiment_id") == config["experiment_id"]
    ]
    judge_analysis(
        judge_rows,
        config["judge_personas"],
        config["judge_personas"] + [config["judge_other_label"]],
    )
    print(f"Analysis files written to {RESULTS}")


if __name__ == "__main__":
    main()
