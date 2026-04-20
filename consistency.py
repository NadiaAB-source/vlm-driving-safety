# ============================================================
# IMPORTS
# ============================================================

from collections import Counter
from utils import CONSERVATIVE_ORDER


# ============================================================
# CONSISTENCY VOTING
# ============================================================

def consistency_vote(safe_actions):
    """
    Select final action using majority voting.

    Args:
        safe_actions (list): list of actions (length K)

    Returns:
        tuple:
            final_action (str)
            is_consistent (bool): True if majority agreement (>50%)
    """

    # Count occurrences
    counts = Counter(safe_actions)
    most_common_action, most_common_count = counts.most_common(1)[0]

    K = len(safe_actions)

    # Majority condition (>50%)
    is_consistent = most_common_count > K / 2

    # Case 1: Majority exists
    if is_consistent:
        return most_common_action, True

    # Case 2: Tie → choose most conservative action
    tied_actions = [a for a, c in counts.items() if c == most_common_count]

    for action in CONSERVATIVE_ORDER:
        if action in tied_actions:
            return action, False

    # Fallback (should rarely happen)
    return most_common_action, False