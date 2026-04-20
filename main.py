# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import random
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from inference import (
    load_model,
    run_qwen,
    parse_json_output,
    ACTION_SYSTEM_PROMPT,
    CONTEXT_SYSTEM_PROMPT,
    CONTEXT_USER_PROMPT
)

from utils import normalize_action
from safety_rules import apply_safety_rules, is_unsafe
from consistency import consistency_vote
from metrics import (
    compute_metrics,
    print_summary,
    analyze_rule_firing,
    print_rule_analysis,
    plot_results,
    error_analysis,
    print_error_analysis
)


# ============================================================
# DATASET SETUP (ZIP EXTRACTION)
# ============================================================

ZIP_PATH = "DriveBench.zip"
EXTRACT_PATH = "data"
DATASET_PATH = os.path.join(EXTRACT_PATH, "DriveBench")

def setup_dataset():
    if os.path.exists(DATASET_PATH):
        print("✅ Dataset already extracted.")
        return

    if os.path.exists(ZIP_PATH):
        print("📦 Extracting dataset...")
        import zipfile
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("✅ Extraction complete!")
    else:
        raise FileNotFoundError("❌ DriveBench.zip not found.")


setup_dataset()
BASE_PATH = os.path.join(DATASET_PATH, "Brightness")


# ============================================================
# LOAD DATASET (HuggingFace)
# ============================================================

def load_drivebench():
    print("📥 Loading dataset...")
    ds = load_dataset("drive-bench/arena")
    ds_test = ds["test"]
    print(f"Total samples: {len(ds_test)}")
    return ds_test

ds_test = load_drivebench()


# ============================================================
# CAMERA EXTRACTION
# ============================================================

def get_camera(question):
    match = re.search(
        r'(CAM_FRONT_LEFT|CAM_FRONT_RIGHT|CAM_BACK_LEFT|CAM_BACK_RIGHT|CAM_FRONT|CAM_BACK)',
        question
    )
    return match.group(1) if match else 'CAM_FRONT'


# ============================================================
# FILTER DATA
# ============================================================

planning_samples = [
    s for s in ds_test
    if s['question_type'] == 'planning'
    and 'lead to a collision' not in s['question'].lower()
]

print(f"Planning samples: {len(planning_samples)}")


valid_samples = []
missing = 0

for s in planning_samples:
    cam = get_camera(s['question'])
    filename = os.path.basename(s['image_path'][cam])
    img_path = os.path.join(BASE_PATH, cam, filename)

    if os.path.exists(img_path):
        s_copy = dict(s)
        s_copy['resolved_camera'] = cam
        s_copy['resolved_image_path'] = img_path
        valid_samples.append(s_copy)
    else:
        missing += 1

print(f"Valid samples: {len(valid_samples)}")
print(f"Missing images: {missing}")


# ============================================================
# LOAD MODEL
# ============================================================

model, processor = load_model()


# ============================================================
# MAIN LOOP
# ============================================================

K = 3
NUM_SAMPLES = 400
SAVE_PATH = "results/vlm_results.json"

os.makedirs("results", exist_ok=True)

random.seed(42)
eval_samples = random.sample(valid_samples, min(NUM_SAMPLES, len(valid_samples)))

results = []

for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):

    img_path = sample['resolved_image_path']
    question = sample['question']
    gt_action = normalize_action(sample['answer'])

    # Context extraction
    try:
        ctx_raw = run_qwen(
            model, processor,
            img_path,
            CONTEXT_SYSTEM_PROMPT,
            CONTEXT_USER_PROMPT,
            temperature=0.0
        )
        context = parse_json_output(ctx_raw)
        context_ok = 'parse_error' not in context
    except:
        context = {}
        context_ok = False

    baseline_actions = []
    safe_actions = []
    fired_all = []
    overrides = []

    # Action inference
    for _ in range(K):
        try:
            raw = run_qwen(
                model, processor,
                img_path,
                ACTION_SYSTEM_PROMPT,
                question,
                temperature=0.7
            )
            parsed = parse_json_output(raw)
            action = normalize_action(parsed.get("action", "unknown"))
        except:
            action = "unknown"

        if context_ok and action != "unknown":
            safe_action, fired, overridden = apply_safety_rules(action, context)
        else:
            safe_action, fired, overridden = action, [], False

        baseline_actions.append(action)
        safe_actions.append(safe_action)
        fired_all.extend(fired)
        overrides.append(overridden)

    # Voting
    baseline_final, baseline_consistent = consistency_vote(baseline_actions)
    safe_final, safe_consistent = consistency_vote(safe_actions)

    # Safety + accuracy
    baseline_unsafe = is_unsafe(baseline_final, context) if context_ok else False
    safe_unsafe = is_unsafe(safe_final, context) if context_ok else False

    results.append({
        "sample_id": i + 1,
        "question": question,
        "baseline_final": baseline_final,
        "safe_final": safe_final,
        "baseline_correct": baseline_final == gt_action,
        "safe_correct": safe_final == gt_action,
        "baseline_unsafe": baseline_unsafe,
        "safe_unsafe": safe_unsafe,
        "baseline_consistent": baseline_consistent,
        "safe_consistent": safe_consistent,
        "any_override": any(overrides),
        "fired_rules": list(set(fired_all)),
        "context": context,
        "context_ok": context_ok
    })

# Save
with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("✅ Results saved.")


# ============================================================
# METRICS
# ============================================================

metrics = compute_metrics(results)
print_summary(metrics)

rule_counts = analyze_rule_firing(results)
print_rule_analysis(rule_counts, len(results))

plot_results(metrics)

stats = error_analysis(results)
print_error_analysis(stats)