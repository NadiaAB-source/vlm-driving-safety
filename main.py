# main.py

import json
import random
from tqdm import tqdm
import os

from inference import run_qwen, parse_json_output, ACTION_SYSTEM_PROMPT, CONTEXT_USER_PROMPT
from safety_rules import apply_safety_rules, is_unsafe
from consistency import consistency_vote
from utils import normalize_action
from metrics import compute_metrics, print_metrics

# ── SETTINGS ───────────────────────────────────────────────
K = 3
NUM_SAMPLES = 50
RESULTS_PATH = "./results/results.json"

# 👉 MODE SWITCH
MODE = "dummy"   # "dummy" or "full"

# 👉 DATA PATH (for full mode)
DATA_PATH = "./data/"


# ───────────────────────────────────────────────────────────
# Dummy Mode
# ───────────────────────────────────────────────────────────
def load_dummy_data():
    return [
        {
            "image_path": "sample.jpg",
            "question": "What should the car do?",
            "answer": "keep speed"
        }
        for _ in range(NUM_SAMPLES)
    ]


# ───────────────────────────────────────────────────────────
# Full Mode (DriveBench-style loader)
# ───────────────────────────────────────────────────────────
def load_full_data():
    """
    Replace this with your actual DriveBench loader if needed.
    Here we assume JSON format:
    [
      {"image_path": "...", "question": "...", "answer": "..."}
    ]
    """

    dataset_file = os.path.join(DATA_PATH, "dataset.json")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError("dataset.json not found in ./data/")

    with open(dataset_file) as f:
        data = json.load(f)

    return data[:NUM_SAMPLES]


# ───────────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────────
def main():

    if MODE == "dummy":
        dataset = load_dummy_data()
        print("Running in DUMMY mode")

    elif MODE == "full":
        dataset = load_full_data()
        print("Running in FULL dataset mode")

    else:
        raise ValueError("MODE must be 'dummy' or 'full'")

    results = []

    for sample in tqdm(dataset, desc="Running pipeline"):

        image_path = sample["image_path"]
        question = sample["question"]
        gt_text = sample["answer"]

        gt_action = normalize_action(gt_text)

        # ── Context Extraction ─────────────────────────────
        ctx_raw = run_qwen(
            image_path,
            CONTEXT_USER_PROMPT,
            CONTEXT_USER_PROMPT,
            temperature=0.0
        )

        context_dict = parse_json_output(ctx_raw)
        context_ok = 'parse_error' not in context_dict

        # ── Action Inference ───────────────────────────────
        baseline_actions = []
        safe_actions = []
        overrides = []

        for _ in range(K):

            raw = run_qwen(
                image_path,
                ACTION_SYSTEM_PROMPT,
                question,
                temperature=0.7
            )

            parsed = parse_json_output(raw)
            action_text = parsed.get("action", "unknown")

            baseline_action = normalize_action(action_text)

            if context_ok and baseline_action != "unknown":
                safe_action, _, overridden = apply_safety_rules(baseline_action, context_dict)
            else:
                safe_action = baseline_action
                overridden = False

            baseline_actions.append(baseline_action)
            safe_actions.append(safe_action)
            overrides.append(overridden)

        # ── Consistency ────────────────────────────────────
        baseline_final, baseline_consistent = consistency_vote(baseline_actions)
        safe_final, safe_consistent = consistency_vote(safe_actions)

        # ── Safety + Accuracy ──────────────────────────────
        baseline_unsafe = is_unsafe(baseline_final, context_dict) if context_ok else False
        safe_unsafe = is_unsafe(safe_final, context_dict) if context_ok else False

        baseline_correct = (baseline_final == gt_action)
        safe_correct = (safe_final == gt_action)

        # ── Store ──────────────────────────────────────────
        results.append({
            "baseline_final": baseline_final,
            "safe_final": safe_final,
            "baseline_correct": baseline_correct,
            "safe_correct": safe_correct,
            "baseline_unsafe": baseline_unsafe,
            "safe_unsafe": safe_unsafe,
            "baseline_consistent": baseline_consistent,
            "safe_consistent": safe_consistent,
            "any_override": any(overrides),
            "context_ok": context_ok,
            "ground_truth_action": gt_action
        })

    # ── Save ───────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Metrics ────────────────────────────────────────────
    metrics = compute_metrics(results)
    print_metrics(metrics)


if __name__ == "__main__":
    main()