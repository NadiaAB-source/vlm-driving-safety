# ============================================================
# IMPORTS
# ============================================================

from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(results):
    """
    Compute evaluation metrics from results.
    """

    valid_results = [
        r for r in results
        if r.get('context_ok', True) and r.get('ground_truth_action') != 'unknown'
    ]

    N = len(valid_results)

    if N == 0:
        raise ValueError("No valid results to evaluate.")

    baseline_acc = sum(r['baseline_correct'] for r in valid_results) / N
    safe_acc     = sum(r['safe_correct']     for r in valid_results) / N

    baseline_unsafe_rate = sum(r['baseline_unsafe'] for r in valid_results) / N
    safe_unsafe_rate     = sum(r['safe_unsafe']     for r in valid_results) / N

    false_overrides = sum(
        1 for r in valid_results
        if r['baseline_correct'] and not r['safe_correct'] and r['any_override']
    )
    false_override_rate = false_overrides / N

    baseline_consistency = sum(r['baseline_consistent'] for r in valid_results) / N
    safe_consistency     = sum(r['safe_consistent']     for r in valid_results) / N

    return {
        "baseline_acc": baseline_acc,
        "safe_acc": safe_acc,
        "baseline_unsafe_rate": baseline_unsafe_rate,
        "safe_unsafe_rate": safe_unsafe_rate,
        "false_override_rate": false_override_rate,
        "baseline_consistency": baseline_consistency,
        "safe_consistency": safe_consistency,
        "N": N
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(metrics):
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f'{"Metric":<30} {"Baseline":>12} {"After Safety":>14}')
    print("-"*60)
    print(f'{"Decision Accuracy":<30} {metrics["baseline_acc"]:>11.1%} {metrics["safe_acc"]:>13.1%}')
    print(f'{"Unsafe Decision Rate":<30} {metrics["baseline_unsafe_rate"]:>11.1%} {metrics["safe_unsafe_rate"]:>13.1%}')
    print(f'{"Consistency Score":<30} {metrics["baseline_consistency"]:>11.1%} {metrics["safe_consistency"]:>13.1%}')
    print(f'{"False Override Rate":<30} {"—":>12} {metrics["false_override_rate"]:>13.1%}')
    print("="*60)

    print(f"\nTotal samples: {metrics['N']}")
    print(f"Unsafe rate reduction: {(metrics['baseline_unsafe_rate'] - metrics['safe_unsafe_rate']):.1%}")
    print(f"Accuracy change: {(metrics['safe_acc'] - metrics['baseline_acc']):+.1%}")


# ============================================================
# RULE FIRING ANALYSIS
# ============================================================

def analyze_rule_firing(results):
    all_fired = []
    for r in results:
        all_fired.extend(r.get('fired_rules', []))

    return Counter(all_fired)


def print_rule_analysis(rule_counts, total_samples):
    rule_names = {
        'R1': 'T1: Red light / Stop sign',
        'R2_yellow': 'T1: Yellow light',
        'R3': 'T1: Pedestrian/Cyclist in road',
        'R3_crosswalk': 'T1: Pedestrian on crosswalk',
        'R4': 'T1: Lane blocked',
        'R5_left': 'T2: No space left',
        'R5_right': 'T2: No space right',
        'R7': 'T3: Vehicle behind close',
        'R8': 'T4: Degraded visibility',
    }

    print("\nRule Firing Counts:")
    print("-" * 50)

    total_fires = sum(rule_counts.values())

    for rule_id, rule_name in rule_names.items():
        count = rule_counts.get(rule_id, 0)
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{rule_id:<15} {rule_name:<35} {count:>4}x ({pct:.1f}% of samples)")

    print(f"{'TOTAL':<50} {total_fires:>4}x")


# ============================================================
# RESULTS VISUALIZATION
# ============================================================

def plot_results(metrics):
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Safety-Aware Post-Processing Results')

    colors_base = '#FFB3BA'
    colors_safe = '#B5EAD7'

    # Unsafe rate
    axes[0].bar(['Baseline', 'After'], [
        metrics["baseline_unsafe_rate"] * 100,
        metrics["safe_unsafe_rate"] * 100
    ], color=[colors_base, colors_safe])
    axes[0].set_title('Unsafe Rate')

    # Accuracy
    axes[1].bar(['Baseline', 'After'], [
        metrics["baseline_acc"] * 100,
        metrics["safe_acc"] * 100
    ], color=[colors_base, colors_safe])
    axes[1].set_title('Accuracy')

    # Consistency
    axes[2].bar(['Baseline', 'After'], [
        metrics["baseline_consistency"] * 100,
        metrics["safe_consistency"] * 100
    ], color=[colors_base, colors_safe])
    axes[2].set_title('Consistency')

    plt.tight_layout()
    plt.savefig("results/vlm_results_chart.png", dpi=150)
    plt.close()

    print("📊 Chart saved.")


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis(results):
    total = len(results)

    n_ctx = sum(1 for r in results if not r.get('context_ok', True))

    n_rule = sum(
        1 for r in results
        if r.get('context_ok', True)
        and r.get('any_override', False)
        and r.get('baseline_correct', False)
        and not r.get('safe_correct', False)
    )

    n_ambig = sum(
        1 for r in results
        if r.get('context_ok', True)
        and not r.get('baseline_consistent', True)
        and not (
            r.get('any_override', False)
            and r.get('baseline_correct', False)
            and not r.get('safe_correct', False)
        )
    )

    return {
        "context_errors": n_ctx,
        "rule_conflicts": n_rule,
        "ambiguous": n_ambig,
        "total": total
    }


def print_error_analysis(stats):
    total = stats["total"]

    print("\n" + "=" * 55)
    print("ERROR CATEGORIZATION")
    print("=" * 55)

    print(f'{"Context Errors":<30} {stats["context_errors"]:>6} {stats["context_errors"]/total*100:>9.1f}%')
    print(f'{"Rule Conflicts":<30} {stats["rule_conflicts"]:>6} {stats["rule_conflicts"]/total*100:>9.1f}%')
    print(f'{"Ambiguous":<30} {stats["ambiguous"]:>6} {stats["ambiguous"]/total*100:>9.1f}%')

    print("=" * 55)