import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # Persona Preference Experiment — lightweight local Colab trial

        This notebook makes **no API calls**. It downloads one small Hugging Face response model and one different judge model, loading them one at a time to limit memory use.

        It runs 2 diverse questions × 6 conditions (the no-prompt Assistant plus 5 personas) × 3 frames = **36 response generations**, then makes **10 persona classifications plus 1 separate Assistant-profile generation**.

        A free T4 GPU is recommended. CPU fallback is supported but slower. Run every cell in order.
        """
    ),
    code('%pip -q install "transformers>=4.46,<5" accelerate pyyaml pandas sentencepiece'),
    code(
        """
        import gc
        import hashlib
        import json
        import re
        from collections import Counter, defaultdict
        from pathlib import Path

        import pandas as pd
        import torch
        import yaml
        from google.colab import files
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        HAS_CUDA = torch.cuda.is_available()
        DEVICE = 'cuda' if HAS_CUDA else 'cpu'
        MODEL_DTYPE = torch.float16 if HAS_CUDA else torch.float32
        print('Runtime device:', torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU (slower)')

        print('Upload config.yaml, questions.json, personas.yaml, experiment.yaml, and judges.yaml')
        uploaded = files.upload()
        required = {'config.yaml', 'questions.json', 'personas.yaml', 'experiment.yaml', 'judges.yaml'}
        missing = required - set(uploaded)
        assert not missing, f'Missing uploaded files: {sorted(missing)}'

        config = yaml.safe_load(uploaded['config.yaml'].decode())
        questions = json.loads(uploaded['questions.json'].decode())
        persona_prompts = yaml.safe_load(uploaded['personas.yaml'].decode())
        experiment_prompts = yaml.safe_load(uploaded['experiment.yaml'].decode())
        judge_prompts = yaml.safe_load(uploaded['judges.yaml'].decode())

        conditions = dict(config['personas'])
        if config.get('include_baseline', False):
            baseline = config['baseline']
            conditions = {baseline['id']: {'name': baseline['name']}, **conditions}

        EXPERIMENT_MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct'
        JUDGE_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
        TRIAL_ID = 'colab_local_three_frame_trial_v5'
        TRIAL_QUESTION_IDS = ['PRPP01', 'PRPP07']
        FRAME_IDS = list(config['frames'])
        A_RATE_THRESHOLD = 0.5  # 0=all; 0.5=A majority; 1=A always
        MAX_NEW_TOKENS = 160
        SEED = int(config['random_seed'])
        RESULTS_DIR = Path('/content/trial_results')
        RESULTS_DIR.mkdir(exist_ok=True)
        EXPERIMENT_FILE = RESULTS_DIR / 'colab_experiment.jsonl'
        JUDGE_FILE = RESULTS_DIR / 'colab_judges.jsonl'
        PROFILE_FILE = RESULTS_DIR / 'colab_assistant_profile.jsonl'
        SUMMARY_FILE = RESULTS_DIR / 'colab_trial_summary.json'

        question_by_id = {item['id']: item for item in questions}
        selected_questions = [question_by_id[item_id] for item_id in TRIAL_QUESTION_IDS]
        assert len(FRAME_IDS) == 3, 'This trial expects exactly three configured frames.'
        assert config.get('include_baseline', False), 'The Assistant condition must be enabled.'
        assert list(config['personas']) == ['P1', 'P2', 'P3', 'P4', 'P5']
        expected_prompt_ids = set(config['personas']) | {config['baseline']['id']}
        assert expected_prompt_ids == set(persona_prompts)
        print('Experiment model:', EXPERIMENT_MODEL)
        print('Judge model:', JUDGE_MODEL)
        print('Personas:', len(config['personas']))
        print('Baseline enabled:', config.get('include_baseline', False))
        print('Response generations:', len(selected_questions) * len(conditions) * len(FRAME_IDS))
        print('Judge generations:', len(config['judge_personas']) * 2 + 1)
        """
    ),
    code(
        """
        def load_local_model(model_id):
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model_kwargs = {'torch_dtype': MODEL_DTYPE}
            if HAS_CUDA:
                model_kwargs['device_map'] = 'auto'
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            if not HAS_CUDA:
                model.to(DEVICE)
            model.eval()
            # Deterministic local trial: remove sampling defaults bundled with some models.
            model.generation_config.do_sample = False
            for name in ('temperature', 'top_p', 'top_k'):
                setattr(model.generation_config, name, None)
            return tokenizer, model

        def generate_local(tokenizer, model, messages):
            set_seed(SEED)
            input_device = next(model.parameters()).device
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors='pt', return_dict=True
            ).to(input_device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            new_tokens = generated[0, encoded['input_ids'].shape[1]:]
            return (
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                int(encoded['input_ids'].numel()),
                int(new_tokens.numel()),
            )

        def parse_json_safely(text):
            cleaned = text.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                start, end = cleaned.find('{'), cleaned.rfind('}')
                if start >= 0 and end > start:
                    try:
                        return json.loads(cleaned[start:end + 1])
                    except json.JSONDecodeError:
                        pass
            return None

        def stable_id(*parts):
            return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:20]

        def deterministic_swap(request_id):
            digest = hashlib.sha256(f'{SEED}|{request_id}'.encode()).digest()
            return bool(digest[0] % 2)

        def append_jsonl(path, row):
            with path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\\n')
                handle.flush()

        def read_jsonl(path):
            if not path.exists():
                return []
            rows = []
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return rows

        def latest_trial_rows(path):
            latest = {}
            for row in read_jsonl(path):
                if row.get('experiment_id') == TRIAL_ID and row.get('request_id'):
                    latest[row['request_id']] = row
            return list(latest.values())

        def normalize_threshold(value):
            value = float(value)
            if value > 1:
                value /= 100
            assert 0 <= value <= 1
            return value

        def include_in_bucket(a_rate, threshold):
            if threshold == 0:
                return True
            if threshold == 1:
                return a_rate == 1
            return a_rate > threshold
        """
    ),
    code(
        """
        experiment_tokenizer, experiment_model = load_local_model(EXPERIMENT_MODEL)
        completed = {
            row['request_id'] for row in latest_trial_rows(EXPERIMENT_FILE)
            if row.get('status') == 'success'
        }
        total = len(selected_questions) * len(conditions) * len(FRAME_IDS)
        done = 0

        for question in selected_questions:
            for persona_id in conditions:
                for frame_id in FRAME_IDS:
                    request_id = stable_id(
                        TRIAL_ID, EXPERIMENT_MODEL, question['id'], persona_id, frame_id, 1
                    )
                    if request_id in completed:
                        done += 1
                        continue
                    swapped = deterministic_swap(request_id)
                    display_a = question['B'] if swapped else question['A']
                    display_b = question['A'] if swapped else question['B']
                    persona_text = persona_prompts[persona_id]['prompt']
                    user_prompt = experiment_prompts['user'].format(
                        frame=experiment_prompts['frames'][frame_id],
                        display_a=display_a,
                        display_b=display_b,
                    )
                    messages = [{'role': 'user', 'content': user_prompt}]
                    if persona_text:
                        persona_instruction = experiment_prompts['persona_instruction'].format(
                            persona_prompt=persona_text
                        )
                        system_prompt = experiment_prompts['system'].format(
                            persona_instruction=persona_instruction
                        )
                        messages.insert(0, {'role': 'system', 'content': system_prompt})
                    raw, input_tokens, output_tokens = generate_local(
                        experiment_tokenizer,
                        experiment_model,
                        messages,
                    )
                    parsed = parse_json_safely(raw)
                    parsed_fields = parsed if isinstance(parsed, dict) else {}
                    model_choice = str(parsed_fields.get('choice', '')).strip().upper()
                    parse_mode = 'json'
                    # The tiny local trial model often returns only `A` or `B`.
                    # That is enough for preference aggregation, so retain the exact
                    # raw response and accept this local-only compact form.
                    if raw.strip().upper() in {'A', 'B'}:
                        model_choice = raw.strip().upper()
                        parse_mode = 'bare_choice'
                    valid = model_choice in {'A', 'B'}
                    canonical = (
                        {'A': 'B', 'B': 'A'}[model_choice] if swapped else model_choice
                    ) if valid else None
                    row = {
                        'request_id': request_id, 'experiment_id': TRIAL_ID,
                        'model': EXPERIMENT_MODEL, 'question_id': question['id'],
                        'category': question['category'], 'persona': persona_id,
                        'frame': frame_id, 'run': 1,
                        'original_A': question['A'], 'original_B': question['B'],
                        'display_A': display_a, 'display_B': display_b,
                        'display_order': 'BA' if swapped else 'AB',
                        'model_choice': model_choice, 'canonical_choice': canonical,
                        'parse_mode': parse_mode,
                        'what': parsed_fields.get('what', ''),
                        'why': parsed_fields.get('why', ''),
                        'how': parsed_fields.get('how', ''),
                        'raw_output': raw, 'input_tokens': input_tokens,
                        'output_tokens': output_tokens, 'cost': 0.0,
                        'status': 'success' if valid else 'error',
                        'error': None if valid else 'Response was neither a valid JSON choice nor a bare A/B choice.',
                    }
                    append_jsonl(EXPERIMENT_FILE, row)
                    done += 1
                    print(f'{done} / {total} | {question["id"]} | {persona_id} | {frame_id} | {row["status"]}')

        current_experiment_rows = latest_trial_rows(EXPERIMENT_FILE)
        experiment_rows = [row for row in current_experiment_rows if row.get('status') == 'success']
        experiment_errors = [row for row in current_experiment_rows if row.get('status') == 'error']
        if experiment_errors:
            print('Invalid response rows:', len(experiment_errors))
            display(pd.DataFrame(experiment_errors)[
                ['question_id', 'persona', 'frame', 'error', 'raw_output']
            ])
        if experiment_rows:
            display(pd.DataFrame(experiment_rows)[
                ['question_id', 'persona', 'frame', 'canonical_choice', 'why', 'status']
            ])
        else:
            print('No valid experiment rows; inspect raw_output in', EXPERIMENT_FILE)
        """
    ),
    code(
        """
        groups = defaultdict(dict)
        for row in experiment_rows:
            groups[(row['question_id'], row['persona'])][row['frame']] = row

        preferences = []
        incomplete = []
        for persona_id in conditions:
            for question in selected_questions:
                observations = groups.get((question['id'], persona_id), {})
                missing_frames = set(FRAME_IDS) - set(observations)
                if missing_frames:
                    incomplete.append((persona_id, question['id'], sorted(missing_frames)))
                    continue
                ordered = [observations[frame_id] for frame_id in FRAME_IDS]
                counts = Counter(row['canonical_choice'] for row in ordered)
                preferred = 'A' if counts['A'] > counts['B'] else 'B'
                representative = next(row for row in ordered if row['canonical_choice'] == preferred)
                preferences.append({
                    'persona': persona_id, 'question_id': question['id'],
                    'A': question['A'], 'B': question['B'],
                    'a_count': counts['A'], 'b_count': counts['B'],
                    'a_rate': counts['A'] / 3, 'b_rate': counts['B'] / 3,
                    'preferred_choice': preferred, 'preferred_text': question[preferred],
                    'preferred_rate': counts[preferred] / 3, 'observations': 3,
                    'what': representative.get('what', ''),
                    'why': representative.get('why', ''),
                    'how': representative.get('how', ''),
                })
        assert not incomplete, f'Incomplete frame groups: {incomplete}'

        threshold = normalize_threshold(A_RATE_THRESHOLD)
        batches = {}
        for persona_id in conditions:
            all_items = [item for item in preferences if item['persona'] == persona_id]
            batches[persona_id] = [
                item for item in all_items if include_in_bucket(item['a_rate'], threshold)
            ]
            print(persona_id, 'bucket questions:', len(batches[persona_id]), '/', len(all_items))

        del experiment_model, experiment_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        judge_tokenizer, judge_model = load_local_model(JUDGE_MODEL)

        candidates = {persona_id: config['personas'][persona_id] for persona_id in config['judge_personas']}
        descriptions = '\\n'.join(
            judge_prompts['candidate_line'].format(
                persona_id=persona_id,
                persona_name=value['name'],
                persona_description=persona_prompts[persona_id]['description'],
            )
            for persona_id, value in candidates.items()
        )

        def render_batch(items, condition):
            if not items:
                return 'No questions met the configured A-rate threshold.'
            rendered = []
            for item_number, item in enumerate(items, start=1):
                text = judge_prompts['batch_choice_item'].format(
                    item_number=item_number, question_id=item['question_id'],
                    option_a=item['A'], option_b=item['B'],
                    preferred_choice=item['preferred_choice'],
                    preferred_text=item['preferred_text'],
                    preferred_percent=round(item['preferred_rate'] * 100, 1),
                    preferred_count=max(item['a_count'], item['b_count']),
                    observations=item['observations'],
                    a_percent=round(item['a_rate'] * 100, 1),
                    b_percent=round(item['b_rate'] * 100, 1),
                )
                if condition == 'choice_and_explanation':
                    text += '\\n' + judge_prompts['batch_explanation'].format(
                        what=item['what'], why=item['why'], how=item['how']
                    )
                rendered.append(text)
            return '\\n\\n'.join(rendered)

        judge_completed = {
            row['request_id'] for row in latest_trial_rows(JUDGE_FILE)
            if row.get('status') == 'success'
        }
        total_judgments = len(config['judge_personas']) * 2
        done = 0
        for actual_persona in config['judge_personas']:
            items = batches[actual_persona]
            profile_id = stable_id(
                TRIAL_ID, actual_persona, threshold,
                [(item['question_id'], item['a_count'], item['b_count']) for item in items],
            )
            for condition in ('choice_only', 'choice_and_explanation'):
                request_id = stable_id(TRIAL_ID, JUDGE_MODEL, profile_id, condition)
                if request_id in judge_completed:
                    done += 1
                    continue
                user_prompt = judge_prompts['classification_user'].format(
                    descriptions=descriptions,
                    evidence=render_batch(items, condition),
                )
                raw, input_tokens, output_tokens = generate_local(
                    judge_tokenizer,
                    judge_model,
                    [
                        {'role': 'system', 'content': judge_prompts['classification_system']},
                        {'role': 'user', 'content': user_prompt},
                    ],
                )
                parsed = parse_json_safely(raw)
                parsed_fields = parsed if isinstance(parsed, dict) else {}
                labels = config['judge_personas'] + [config['judge_other_label']]
                raw_persona = str(parsed_fields.get('persona', ''))
                persona_match = re.search(r'\\b(P[1-5])\\b', raw_persona.upper())
                raw_upper = raw.strip().upper()
                if persona_match and persona_match.group(1) in config['judge_personas']:
                    predicted = persona_match.group(1)
                elif raw_upper.startswith(config['judge_other_label']):
                    predicted = config['judge_other_label']
                else:
                    # Also accepts bare or prose local outputs such as "P2 (Strategist)".
                    bare_match = re.search(r'\\b(P[1-5])\\b', raw_upper)
                    predicted = (
                        bare_match.group(1)
                        if bare_match and bare_match.group(1) in config['judge_personas']
                        else None
                    )
                confidence = parsed_fields.get('confidence')
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                    confidence = confidence / 100 if 1 < confidence <= 100 else confidence
                else:
                    confidence = None
                other_profile_name = parsed_fields.get('other_profile_name', '')
                other_profile_description = parsed_fields.get('other_profile_description', '')
                parse_mode = 'json'
                if predicted == config['judge_other_label'] and (
                    not str(other_profile_name).strip() or not str(other_profile_description).strip()
                ):
                    # Preserve the complete raw local output instead of inventing a profile.
                    other_profile_name = 'OTHER (local raw fallback)'
                    other_profile_description = raw.strip()
                    parse_mode = 'raw_other_profile'
                elif predicted in config['judge_personas'] and not isinstance(parsed, dict):
                    parse_mode = 'bare_persona_label'
                elif predicted in config['judge_personas'] and raw_persona != predicted:
                    parse_mode = 'normalized_persona_label'
                valid = (
                    predicted in labels
                    and (confidence is None or 0 <= confidence <= 1)
                    and isinstance(other_profile_name, str)
                    and isinstance(other_profile_description, str)
                    and (
                        bool(other_profile_name.strip()) and bool(other_profile_description.strip())
                        if predicted == config['judge_other_label']
                        else other_profile_name == '' and other_profile_description == ''
                    )
                )
                result = {
                    'request_id': request_id, 'experiment_id': TRIAL_ID,
                    'judge_model': JUDGE_MODEL, 'condition': condition,
                    'profile_id': profile_id, 'experiment_model': EXPERIMENT_MODEL,
                    'actual_persona': actual_persona, 'predicted_persona': predicted,
                    'confidence': confidence, 'parse_mode': parse_mode,
                    'a_rate_threshold': threshold,
                    'other_profile_name': other_profile_name,
                    'other_profile_description': other_profile_description,
                    'source_questions': len(selected_questions),
                    'bucket_questions': len(items),
                    'bucket_question_ids': [item['question_id'] for item in items],
                    'raw_output': raw, 'input_tokens': input_tokens,
                    'output_tokens': output_tokens, 'cost': 0.0,
                    'status': 'success' if valid else 'error',
                    'error': None if valid else 'Response was not valid judge JSON',
                }
                append_jsonl(JUDGE_FILE, result)
                done += 1
                print(f'{done} / {total_judgments} | {actual_persona} | {condition} | {result["status"]}')

        baseline_id = config['baseline']['id']
        baseline_items = batches[baseline_id]
        baseline_profile_id = stable_id(
            TRIAL_ID, baseline_id, threshold,
            [(item['question_id'], item['a_count'], item['b_count']) for item in baseline_items],
        )
        profile_request_id = stable_id(
            TRIAL_ID, 'assistant-profile', JUDGE_MODEL, baseline_profile_id
        )
        completed_profiles = {
            row['request_id'] for row in latest_trial_rows(PROFILE_FILE)
            if row.get('status') == 'success'
        }
        if profile_request_id not in completed_profiles:
            evidence_lines = [
                judge_prompts['profile_evidence_line'].format(
                    question_id=item['question_id'],
                    preferred_text=item['preferred_text'],
                    preferred_percent=round(item['preferred_rate'] * 100, 1),
                    why=item['why'],
                )
                for item in baseline_items
            ]
            evidence = '\\n'.join(evidence_lines) or 'No questions met the configured A-rate threshold.'
            profile_prompt = judge_prompts['profile_user'].format(evidence=evidence)
            raw, input_tokens, output_tokens = generate_local(
                judge_tokenizer,
                judge_model,
                [{'role': 'user', 'content': profile_prompt}],
            )
            parsed = parse_json_safely(raw)
            parsed_fields = parsed if isinstance(parsed, dict) else {}
            traits = parsed_fields.get('traits', [])
            summary = parsed_fields.get('summary', '')
            parse_mode = 'json'
            valid = (
                isinstance(traits, list)
                and len(traits) == 5
                and all(isinstance(trait, str) and trait.strip() for trait in traits)
                and isinstance(summary, str)
                and bool(summary.strip())
            )
            # Keep a non-empty raw local profile as transparent trial evidence when
            # the small judge cannot obey the full profile JSON schema.
            if not valid and raw.strip():
                traits = []
                summary = raw.strip()
                parse_mode = 'raw_profile_text'
                valid = True
            profile_result = {
                'request_id': profile_request_id, 'experiment_id': TRIAL_ID,
                'name': 'inferred_default_behavioral_profile',
                'judge_model': JUDGE_MODEL, 'experiment_model': EXPERIMENT_MODEL,
                'profile_id': baseline_profile_id, 'traits': traits, 'summary': summary,
                'parse_mode': parse_mode, 'a_rate_threshold': threshold,
                'source_questions': len(selected_questions),
                'bucket_questions': len(baseline_items),
                'raw_output': raw, 'input_tokens': input_tokens,
                'output_tokens': output_tokens, 'cost': 0.0,
                'status': 'success' if valid else 'error',
                'error': None if valid else 'Response was not valid Assistant profile JSON',
            }
            append_jsonl(PROFILE_FILE, profile_result)
            print('Assistant profile |', profile_result['status'])
        """
    ),
    code(
        """
        current_judge_rows = [
            row for row in latest_trial_rows(JUDGE_FILE)
            if row.get('a_rate_threshold') == threshold
        ]
        judge_rows = [row for row in current_judge_rows if row.get('status') == 'success']
        judge_errors = [row for row in current_judge_rows if row.get('status') == 'error']
        if judge_rows:
            judge_df = pd.DataFrame(judge_rows)
            judge_df['correct'] = judge_df['actual_persona'] == judge_df['predicted_persona']
            judge_df['abstained'] = judge_df['predicted_persona'] == config['judge_other_label']
            display(judge_df[
                ['actual_persona', 'condition', 'predicted_persona', 'confidence',
                 'other_profile_name', 'other_profile_description',
                 'bucket_questions', 'correct', 'abstained']
            ])
            display(judge_df.groupby('condition').agg(
                accuracy=('correct', 'mean'),
                abstention_rate=('abstained', 'mean'),
                profiles=('profile_id', 'size'),
            ))
        else:
            print('No valid judge rows; inspect raw_output in', JUDGE_FILE)
        if judge_errors:
            print('Invalid judge rows:', len(judge_errors))
            display(pd.DataFrame(judge_errors)[
                ['actual_persona', 'condition', 'error', 'raw_output']
            ])

        profile_rows = [
            row for row in latest_trial_rows(PROFILE_FILE)
            if row.get('status') == 'success'
            and row.get('a_rate_threshold') == threshold
        ]
        if profile_rows:
            display(pd.DataFrame(profile_rows)[
                ['judge_model', 'experiment_model', 'traits', 'summary', 'bucket_questions']
            ])
        else:
            print('No valid Assistant profile; inspect raw_output in', PROFILE_FILE)

        expected_responses = len(selected_questions) * len(conditions) * len(FRAME_IDS)
        expected_judgments = len(config['judge_personas']) * 2
        summary = {
            'trial_id': TRIAL_ID,
            'device': DEVICE,
            'experiment_model': EXPERIMENT_MODEL,
            'judge_model': JUDGE_MODEL,
            'a_rate_threshold': threshold,
            'successful_responses': len(experiment_rows),
            'expected_responses': expected_responses,
            'successful_persona_judgments': len(judge_rows),
            'expected_persona_judgments': expected_judgments,
            'successful_assistant_profiles': len(profile_rows),
            'complete': (
                len(experiment_rows) == expected_responses
                and len(judge_rows) == expected_judgments
                and len(profile_rows) == 1
            ),
        }
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + '\\n')
        print('Trial summary:', summary)

        print('Experiment results:', EXPERIMENT_FILE)
        print('Judge results:', JUDGE_FILE)
        print('Assistant profile:', PROFILE_FILE)
        print('Trial summary:', SUMMARY_FILE)
        files.download(str(EXPERIMENT_FILE))
        files.download(str(JUDGE_FILE))
        files.download(str(PROFILE_FILE))
        files.download(str(SUMMARY_FILE))
        """
    ),
    markdown(
        """
        ## What this trial checks

        The notebook verifies separate prompt files, a true no-system-prompt Assistant condition and separate Assistant profile, three-frame aggregation, A/B canonical mapping, configurable threshold buckets, five-persona whole-batch judging, `OTHER` with a related inferred profile, resumable JSONL files, and raw local model output capture.

        Change `A_RATE_THRESHOLD` to `0`, `0.5`, or `1` in the setup cell. The 360M/0.5B models keep resource use low; their accuracy is not a research result. Use Docker/OpenRouter only after this plumbing trial succeeds.
        """
    ),
    markdown(
        """
        ## Move to OpenRouter after the trial passes

        Continue only when `colab_trial_summary.json` reports `"complete": true`.

        1. Put the two OpenRouter experiment model IDs and two judge model IDs in `config.yaml`.
        2. Put `OPENROUTER_API_KEY=...` in the local `.env` file or the GitHub Actions secret.
        3. Run the Docker pilot, judge, and analysis commands from `README.md`.
        4. Inspect `results/raw_http_log.jsonl` and the completion reports before starting full mode.

        The OpenRouter phase uses the same questions, five personas plus P0, three-frame aggregation,
        threshold buckets, `OTHER` profile fields, and separate Assistant profiling tested here.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "colab_local_trial.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(ROOT / "colab_local_trial.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
