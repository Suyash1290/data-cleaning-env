import re
from datetime import datetime


def clamp_score(val: float) -> float:
    """Ensure score is strictly in (0, 1) exclusive."""
    if val <= 0.0:
        return 0.01
    if val >= 1.0:
        return 0.99
    return val


def normalize_value(val, col_name: str = ""):
    if val is None or val == "None" or str(val).lower() == "nan":
        return None
    s = str(val).strip().lower().replace(",", "")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    for fmt in ["%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s


def grade(episode_state) -> float:
    manifest = episode_state.manifest
    fixes = episode_state.cells_fixed

    correct = 0
    false_fixes = 0
    manifest_keys = {f"{m['row']}|{m['col']}" for m in manifest}

    for issue in manifest:
        key = f"{issue['row']}|{issue['col']}"
        if key in fixes:
            submitted = normalize_value(fixes[key], issue["col"])
            expected  = normalize_value(issue["expected"], issue["col"])
            if submitted == expected:
                correct += 1

    for key in fixes:
        if key not in manifest_keys:
            false_fixes += 1

    base_score = correct / len(manifest) if manifest else 0.5
    penalty = false_fixes * 0.05
    return clamp_score(base_score - penalty)
