# ============================================================
# IMPORTS
# ============================================================

from utils import ACTION_SPACE, CONSERVATIVE_ORDER


# ============================================================
# CONTEXT PARSING
# ============================================================

def parse_context(context_dict):
    """
    Normalize context values into consistent format for rule checking.

    Args:
        context_dict (dict): raw context from model

    Returns:
        dict: normalized context
    """

    def is_yes(val):
        return str(val).lower().strip().rstrip('.') in {'yes', 'true', '1'}

    def is_no(val):
        return str(val).lower().strip().rstrip('.') in {'no', 'false', '0', 'none'}

    tl = str(context_dict.get('traffic_light', 'none')).lower().strip().rstrip('.')

    return {
        'traffic_light': tl,
        'stop_sign': is_yes(context_dict.get('stop_sign', 'no')),
        'crosswalk': is_yes(context_dict.get('crosswalk', 'no')),
        'pedestrian': is_yes(context_dict.get('pedestrian', 'no')),
        'vehicle_ahead': is_yes(context_dict.get('vehicle_ahead', 'no')),
        'vehicle_behind': is_yes(context_dict.get('vehicle_behind', 'no')),
        'lane_blocked': is_yes(context_dict.get('lane_blocked', 'no')),
        'drivable_left': not is_no(context_dict.get('drivable_left', 'yes')),
        'drivable_right': not is_no(context_dict.get('drivable_right', 'yes')),
        'visibility_degraded': is_yes(context_dict.get('visibility_degraded', 'no')),
    }


# ============================================================
# SAFETY RULES
# ============================================================

def apply_safety_rules(action, context_raw):
    """
    Apply hierarchical safety rules to predicted action.

    Args:
        action (str): predicted action
        context_raw (dict): raw context from model

    Returns:
        tuple:
            safe_action (str)
            fired_rules (list)
            was_overridden (bool)
    """

    c = parse_context(context_raw)
    fired = []

    # Start with all possible actions except 'unknown'
    feasible = set(ACTION_SPACE) - {'unknown'}

    # ── Tier 1: Mandatory Stop ────────────────────────────────
    if c['traffic_light'] == 'red' or c['stop_sign']:
        fired.append('R1')
        feasible &= {'stop', 'brake hard', 'brake gently'}

    if c['traffic_light'] == 'yellow':
        fired.append('R2_yellow')
        feasible.discard('accelerate')

    if c['pedestrian']:
        fired.append('R3')
        feasible.discard('accelerate')

        if c['crosswalk']:
            fired.append('R3_crosswalk')
            feasible.discard('change lane left')
            feasible.discard('change lane right')

    if c['lane_blocked']:
        fired.append('R4')
        feasible.discard('accelerate')
        feasible.discard('keep speed')

    # ── Tier 2: Lateral Legality ──────────────────────────────
    if not c['drivable_left']:
        fired.append('R5_left')
        feasible.discard('change lane left')

    if not c['drivable_right']:
        fired.append('R5_right')
        feasible.discard('change lane right')

    # ── Tier 3: Rear-End Risk ─────────────────────────────────
    if c['vehicle_behind']:
        fired.append('R7')
        feasible.discard('brake hard')
        feasible.discard('stop')

    # ── Tier 4: Visibility ────────────────────────────────────
    if c['visibility_degraded']:
        fired.append('R8')
        feasible.discard('accelerate')
        feasible.discard('change lane left')
        feasible.discard('change lane right')
        feasible.discard('brake hard')

    # ── Decision Logic ────────────────────────────────────────

    # Edge case: no feasible actions
    if not feasible:
        fallback = 'brake gently' if 'R7' in fired else 'stop'
        return fallback, fired, True

    # If action is already safe
    if action in feasible:
        return action, fired, False

    # Otherwise choose safest alternative
    for conservative_action in CONSERVATIVE_ORDER:
        if conservative_action in feasible:
            return conservative_action, fired, True

    # Final fallback (should not happen)
    return list(feasible)[0], fired, True


# ============================================================
# UNSAFE CHECK
# ============================================================

def is_unsafe(action, context_raw):
    """
    Check if action violates safety rules.

    Args:
        action (str)
        context_raw (dict)

    Returns:
        bool
    """
    _, _, overridden = apply_safety_rules(action, context_raw)
    return overridden