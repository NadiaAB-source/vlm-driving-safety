# ============================================================
# DEMO: SINGLE SAMPLE (PROFESSOR REQUIREMENT)
# ============================================================

import os
import re
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
from safety_rules import apply_safety_rules
from consistency import consistency_vote


# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "data/DriveBench/Brightness"
K = 3


# ============================================================
# LOAD DATASET (ONE SAMPLE)
# ============================================================

print("📥 Loading dataset...")
ds = load_dataset("drive-bench/arena")
ds_test = ds["test"]

sample = next(
    s for s in ds_test
    if s['question_type'] == 'planning'
    and 'lead to a collision' not in s['question'].lower()
)


def get_camera(question):
    match = re.search(
        r'(CAM_FRONT_LEFT|CAM_FRONT_RIGHT|CAM_BACK_LEFT|CAM_BACK_RIGHT|CAM_FRONT|CAM_BACK)',
        question
    )
    return match.group(1) if match else 'CAM_FRONT'


cam = get_camera(sample['question'])
filename = os.path.basename(sample['image_path'][cam])
img_path = os.path.join(BASE_PATH, cam, filename)

if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

print(f"\n📸 Using image: {img_path}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🤖 Loading model...")
model, processor = load_model()


# ============================================================
# STEP 1: QUESTION
# ============================================================

print("\n" + "="*60)
print("🟦 QUESTION")
print("="*60)
print(sample['question'])


# ============================================================
# STEP 2: QWEN RAW OUTPUT
# ============================================================

print("\n" + "="*60)
print("🟨 QWEN RAW OUTPUT")
print("="*60)

baseline_actions = []

for _ in range(K):
    try:
        raw = run_qwen(
            model, processor,
            img_path,
            ACTION_SYSTEM_PROMPT,
            sample['question'],
            temperature=0.7
        )

        parsed = parse_json_output(raw)
        action = parsed.get("action", "unknown")
        norm_action = normalize_action(action)

    except Exception:
        norm_action = "unknown"

    baseline_actions.append(norm_action)
    print(f"- {norm_action}")


# ============================================================
# STEP 3: IMAGE DESCRIPTION (CONTEXT)
# ============================================================

print("\n" + "="*60)
print("🟩 IMAGE DESCRIPTION (CONTEXT)")
print("="*60)

try:
    ctx_raw = run_qwen(
        model, processor,
        img_path,
        CONTEXT_SYSTEM_PROMPT,
        CONTEXT_USER_PROMPT,
        temperature=0.0
    )
    context = parse_json_output(ctx_raw)
except Exception:
    context = {}

if not context:
    print("⚠️ Context extraction failed.")
else:
    for k, v in context.items():
        print(f"{k}: {v}")


# ============================================================
# STEP 4: APPLY SAFETY RULES
# ============================================================

print("\n" + "="*60)
print("🟧 SAFETY RULES APPLICATION")
print("="*60)

safe_actions = []
all_rules = []

for action in baseline_actions:
    safe_action, rules, overridden = apply_safety_rules(action, context)

    safe_actions.append(safe_action)
    all_rules.extend(rules)

    print(f"{action} → {safe_action} | Rules: {rules}")


# ============================================================
# STEP 5: FINAL DECISION
# ============================================================

print("\n" + "="*60)
print("🟥 FINAL DECISION")
print("="*60)

final_action, consistent = consistency_vote(safe_actions)

print(f"Final action: {final_action}")
print(f"Consistency: {'Yes' if consistent else 'No'}")


# ============================================================
# STEP 6: SUMMARY
# ============================================================

print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

print(f"Raw Qwen actions: {baseline_actions}")
print(f"Safe actions:     {safe_actions}")
print(f"Rules fired:      {set(all_rules)}")
print(f"Final output:     {final_action}")