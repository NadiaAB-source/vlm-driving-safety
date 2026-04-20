# ============================================================
# ACTION SPACE + NORMALIZATION
# ============================================================

# Discrete action space
ACTION_SPACE = [
    'keep speed',
    'accelerate',
    'brake gently',
    'brake hard',
    'stop',
    'turn left',
    'turn right',
    'change lane left',
    'change lane right',
    'back up',
    'unknown'
]

# Keyword mapping to normalize free-form text → discrete action
ACTION_KEYWORDS = {
    'keep speed':        ['keep', 'maintain', 'same speed', 'constant', 'continue', 'going at the same'],
    'accelerate':        ['accelerat', 'speed up', 'increase speed'],
    'brake gently':      ['brake gently', 'decelerate', 'slow down', 'gradually', 'reduce speed'],
    'brake hard':        ['brake hard', 'brake sudden', 'sudden brake', 'hard brake', 'emergency'],
    'stop':              ['stop', 'halt', 'remain stationary', 'stationary', 'come to a stop'],
    'turn left':         ['turn left'],
    'turn right':        ['turn right'],
    'change lane left':  ['change lane left', 'lane change left', 'move left', 'merge left'],
    'change lane right': ['change lane right', 'lane change right', 'move right', 'merge right'],
    'back up':           ['back up', 'reverse', 'backward'],
}

# Conservative fallback order (safest first)
CONSERVATIVE_ORDER = [
    'keep speed', 'brake gently', 'brake hard', 'stop',
    'change lane left', 'change lane right', 'turn left',
    'turn right', 'accelerate', 'back up'
]


def normalize_action(text: str) -> str:
    """
    Convert free-form model output into a discrete action.

    Args:
        text (str): raw action text from model

    Returns:
        str: normalized action from ACTION_SPACE
    """
    if not text:
        return 'unknown'

    text = text.lower().strip()

    for action, keywords in ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return action

    return 'unknown'

