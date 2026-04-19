# consistency.py

from collections import Counter
from typing import List, Tuple

# Same conservative order used in safety rules
CONSERVATIVE_ORDER = [
    'keep speed',
    'brake gently',
    'brake hard',
    'stop',
    'change lane left',
    'change lane right',
    'turn left',
    'turn right',
    'accelerate',
    'back up'
]

def consistency_vote(actions: List[str]) -> Tuple[str, bool]:
    """
    Given K actions, return:
    (final_action, is_consistent)

    - If majority exists (>50%), return it
    - If tie, return most conservative action
    """

    counts = Counter(actions)
    most_common_action, most_common_count = counts.most_common(1)[0]

    K = len(actions)
    is_consistent = most_common_count > K / 2

    if is_consistent:
        return most_common_action, True

    # Tie case → choose most conservative
    tied_actions = [a for a, c in counts.items() if c == most_common_count]

    for action in CONSERVATIVE_ORDER:
        if action in tied_actions:
            return action, False

    # Fallback (should not happen)
    return most_common_action, False