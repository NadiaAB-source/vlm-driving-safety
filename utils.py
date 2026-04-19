# utils.py

from typing import Dict

# ── Action Keywords Mapping ───────────────────────────────────
ACTION_KEYWORDS: Dict[str, list] = {
    'keep speed': ['keep', 'maintain', 'same speed', 'constant', 'continue', 'going at the same'],
    'accelerate': ['accelerat', 'speed up', 'increase speed'],
    'brake gently': ['brake gently', 'decelerate', 'slow down', 'gradually', 'reduce speed'],
    'brake hard': ['brake hard', 'brake sudden', 'sudden brake', 'hard brake', 'emergency'],
    'stop': ['stop', 'halt', 'remain stationary', 'stationary', 'come to a stop'],
    'turn left': ['turn left'],
    'turn right': ['turn right'],
    'change lane left': ['change lane left', 'lane change left', 'move left', 'merge left'],
    'change lane right': ['change lane right', 'lane change right', 'move right', 'merge right'],
    'back up': ['back up', 'reverse', 'backward'],
}

# ── Normalize Action ──────────────────────────────────────────
def normalize_action(text: str) -> str:
    """
    Convert free-form text into one of the predefined actions.
    Returns 'unknown' if no match is found.
    """

    if not text:
        return 'unknown'

    text = text.lower().strip()

    for action, keywords in ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return action

    return 'unknown'