import re
from datetime import datetime

def normalize_value(val, col_name: str = ""):
    if val is None or val == "None" or str(val).lower() == "nan":
        return None
    s = str(val).strip().lower().replace(",", "")
    # Try numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    # Try date normalization
    for fmt in ["%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # Return cleaned string

def grade(episode_state) -> float:
    manifest = episode_state.manifest  # list of issue dicts
    fixes = episode_state.cells_fixed      # dict: "row|col" -> submitted_val

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

    # Penalize false fixes (fixed cells not in manifest)
    for key in fixes:
        if key not in manifest_keys:
            false_fixes += 1

    base_score = correct / len(manifest) if manifest else 0.0
    penalty = false_fixes * 0.05
    return max(0.0, min(1.0, base_score - penalty))
