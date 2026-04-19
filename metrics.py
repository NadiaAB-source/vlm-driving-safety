# metrics.py

from typing import List, Dict

def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute all evaluation metrics from results list.
    """

    # Filter valid samples
    valid_results = [
        r for r in results
        if r.get('context_ok', False)
        and r.get('ground_truth_action', 'unknown') != 'unknown'
    ]

    N = len(valid_results)

    if N == 0:
        return {}

    # ── Accuracy ─────────────────────────────────────────────
    baseline_acc = sum(r['baseline_correct'] for r in valid_results) / N
    safe_acc = sum(r['safe_correct'] for r in valid_results) / N

    # ── Unsafe Rate ─────────────────────────────────────────
    baseline_unsafe = sum(r['baseline_unsafe'] for r in valid_results) / N
    safe_unsafe = sum(r['safe_unsafe'] for r in valid_results) / N

    # ── False Override Rate ─────────────────────────────────
    false_overrides = sum(
        1 for r in valid_results
        if r['baseline_correct']
        and not r['safe_correct']
        and r['any_override']
    )
    false_override_rate = false_overrides / N

    # ── Consistency ─────────────────────────────────────────
    baseline_consistency = sum(r['baseline_consistent'] for r in valid_results) / N
    safe_consistency = sum(r['safe_consistent'] for r in valid_results) / N

    return {
        "N": N,
        "baseline_accuracy": baseline_acc,
        "safe_accuracy": safe_acc,
        "baseline_unsafe_rate": baseline_unsafe,
        "safe_unsafe_rate": safe_unsafe,
        "false_override_rate": false_override_rate,
        "baseline_consistency": baseline_consistency,
        "safe_consistency": safe_consistency
    }


def print_metrics(metrics: Dict):
    """
    Pretty print results.
    """

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"{'Metric':<30} {'Baseline':>12} {'After Safety':>14}")
    print("-"*60)

    print(f"{'Decision Accuracy':<30} {metrics['baseline_accuracy']:.1%} {metrics['safe_accuracy']:>13.1%}")
    print(f"{'Unsafe Decision Rate':<30} {metrics['baseline_unsafe_rate']:.1%} {metrics['safe_unsafe_rate']:>13.1%}")
    print(f"{'Consistency Score':<30} {metrics['baseline_consistency']:.1%} {metrics['safe_consistency']:>13.1%}")
    print(f"{'False Override Rate':<30} {'—':>12} {metrics['false_override_rate']:>13.1%}")

    print("="*60)