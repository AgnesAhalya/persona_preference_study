"""Compare chooser-model A/B agreement inside one experiment run."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
RESULTS_ROOT = ROOT / "results"


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def latest_successes(rows):
    latest = {}
    for row in rows:
        request_id = row.get("request_id")
        if request_id and row.get("status") == "success":
            latest[request_id] = row
    return list(latest.values())


def resolve_run_directory(explicit_run_dir=None):
    if explicit_run_dir:
        run_dir = Path(explicit_run_dir).expanduser().resolve()
    else:
        latest_file = RESULTS_ROOT / "LATEST_RUN"
        if not latest_file.exists():
            raise SystemExit("results/LATEST_RUN is missing. Run the experiment first.")
        run_dir = RESULTS_ROOT / latest_file.read_text(encoding="utf-8").strip()
    if not run_dir.is_dir():
        raise SystemExit(f"Run folder does not exist: {run_dir}")
    return run_dir


def threshold_label(value):
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_") or "0"


def compare(run_dir, threshold):
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    fingerprint = metadata.get("experiment_fingerprint")
    source = run_dir / "chooser/success/experiment.jsonl"
    rows = latest_successes(read_jsonl(source))
    if fingerprint:
        rows = [row for row in rows if row.get("experiment_fingerprint") == fingerprint]
    if not rows:
        raise SystemExit(f"No successful chooser rows found in {source}")

    models = sorted({row["model"] for row in rows})
    if not models:
        raise SystemExit("No experiment models were found.")
    required = 0 if threshold == 0 else math.ceil(threshold * len(models))

    model_observations = defaultdict(list)
    question_details = {}
    for row in rows:
        key = (row["persona"], row["question_id"], row["model"])
        choice = row.get("canonical_choice")
        if choice in {"A", "B"}:
            model_observations[key].append(choice)
        question_details[(row["persona"], row["question_id"])] = {
            "A": row.get("original_A", ""),
            "B": row.get("original_B", ""),
        }

    model_choices = {}
    for key, choices in model_observations.items():
        counts = Counter(choices)
        if counts["A"] == counts["B"]:
            model_choices[key] = "TIE"
        else:
            model_choices[key] = "A" if counts["A"] > counts["B"] else "B"

    comparisons = []
    for persona, question_id in sorted(question_details):
        choices_by_model = {
            model: model_choices.get((persona, question_id, model), "MISSING")
            for model in models
        }
        valid_choices = [choice for choice in choices_by_model.values() if choice in {"A", "B"}]
        counts = Counter(valid_choices)
        highest = max(counts.values(), default=0)
        leaders = [choice for choice in ("A", "B") if counts[choice] == highest and highest > 0]
        consensus = leaders[0] if len(leaders) == 1 else "TIE"
        matching_models = [
            model for model, choice in choices_by_model.items() if choice == consensus
        ] if consensus in {"A", "B"} else []
        agreement = highest / len(models)
        passes = threshold == 0 or (consensus in {"A", "B"} and highest >= required)
        details = question_details[(persona, question_id)]
        comparisons.append({
            "persona": persona,
            "question_id": question_id,
            "option_A": details["A"],
            "option_B": details["B"],
            "total_models": len(models),
            "models_present": len(valid_choices),
            "required_matching_models": required,
            "consensus_choice": consensus,
            "consensus_text": details.get(consensus, "") if consensus in {"A", "B"} else "",
            "matching_count": highest,
            "matching_models": "; ".join(matching_models),
            "agreement_score": agreement,
            "passes_threshold": passes,
            "model_choices": json.dumps(choices_by_model, ensure_ascii=False, sort_keys=True),
        })
    return models, comparisons


def write_outputs(run_dir, threshold, models, comparisons):
    output_dir = run_dir / "chooser/model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    label = threshold_label(threshold)
    selected = [row for row in comparisons if row["passes_threshold"]]
    csv_path = output_dir / f"model_choice_comparison_threshold_{label}.csv"
    md_path = output_dir / f"model_choice_comparison_threshold_{label}.md"
    summary_csv_path = output_dir / f"persona_agreement_summary_threshold_{label}.csv"
    summary_md_path = output_dir / f"persona_agreement_summary_threshold_{label}.md"

    fields = list(comparisons[0]) if comparisons else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    by_persona = defaultdict(list)
    for row in comparisons:
        by_persona[row["persona"]].append(row)
    persona_summary = []
    for persona, rows in sorted(by_persona.items()):
        total_questions = len(rows)
        threshold_matches = sum(row["passes_threshold"] for row in rows)
        unanimous = sum(row["agreement_score"] == 1 for row in rows)
        persona_summary.append({
            "persona": persona,
            "total_questions": total_questions,
            "questions_meeting_threshold": threshold_matches,
            "threshold_match_percentage": 100 * threshold_matches / total_questions,
            "unanimous_questions": unanimous,
            "unanimous_percentage": 100 * unanimous / total_questions,
        })
    summary_fields = list(persona_summary[0]) if persona_summary else []
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(persona_summary)

    required = 0 if threshold == 0 else math.ceil(threshold * len(models))
    summary_lines = [
        "# Agreement percentage by persona",
        "",
        f"- Run folder: `{run_dir.name}`",
        f"- Threshold: **{threshold:g}** ({required} of {len(models)} models required)",
        "",
        "| Persona | Questions | Met threshold | Threshold match | Unanimous |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in persona_summary:
        summary_lines.append(
            f"| {row['persona']} | {row['total_questions']} | "
            f"{row['questions_meeting_threshold']} | "
            f"{row['threshold_match_percentage']:.1f}% | "
            f"{row['unanimous_percentage']:.1f}% |"
        )
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    lines = [
        "# Cross-model choice agreement",
        "",
        f"- Run folder: `{run_dir.name}`",
        f"- Models compared: **{len(models)}** ({', '.join(models)})",
        f"- Threshold: **{threshold:g}**",
        f"- Required matching models: **{required} of {len(models)}**",
        f"- Included persona/question rows: **{len(selected)} of {len(comparisons)}**",
        "",
        "Agreement is the number of models supporting the most common A/B choice divided by all models.",
        "",
        "| Persona | Question | Consensus | Matching | Agreement | Model choices |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in selected:
        choices = json.loads(row["model_choices"])
        rendered = "; ".join(f"{model}: {choices[model]}" for model in models)
        lines.append(
            f"| {row['persona']} | {row['question_id']} | {row['consensus_choice']} | "
            f"{row['matching_count']}/{row['total_models']} | {row['agreement_score']:.3f} | {rendered} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        csv_path, md_path, summary_csv_path, summary_md_path,
        len(selected), len(comparisons),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare A/B choice agreement across experiment models in one run folder."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum agreement threshold from 0 to 1 (default: 1).",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional single run folder; default is the folder named by results/LATEST_RUN.",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    run_dir = resolve_run_directory(args.run_dir)
    models, comparisons = compare(run_dir, args.threshold)
    csv_path, md_path, summary_csv_path, summary_md_path, selected, total = write_outputs(
        run_dir, args.threshold, models, comparisons
    )
    required = 0 if args.threshold == 0 else math.ceil(args.threshold * len(models))
    print(f"Run folder: {run_dir}")
    print(f"Threshold: {args.threshold:g} -> at least {required}/{len(models)} matching models")
    print(f"Included: {selected}/{total}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Persona summary CSV: {summary_csv_path}")
    print(f"Persona summary Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
