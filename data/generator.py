import pandas as pd
import random
import numpy as np
import re
from datetime import datetime, timedelta

def generate_dataset(seed: int, n_rows: int, difficulty: str):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    # Step 1: Build a clean base dataframe
    base_names = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy"]
    clean_rows = []
    for i in range(n_rows):
        clean_rows.append({
            "id": i + 1,
            "name": rng.choice(base_names),
            "age": rng.randint(18, 80),
            "signup_date": (datetime(2020,1,1) + timedelta(days=rng.randint(0,1400))).strftime("%Y-%m-%d"),
            "revenue": round(rng.uniform(10.0, 9999.99), 2),
            "country": rng.choice(["US","UK","IN","DE","FR"]),
            "is_active": rng.choice([True, False]),
            "email": f"{rng.choice(base_names).lower()}{i+1}@example.com"
        })
    clean_df = pd.DataFrame(clean_rows)
    dirty_df = clean_df.copy()
    issue_manifest = []

    # Step 2: Inject issues based on difficulty
    n_issues = {"easy": 4, "medium": 8, "hard": 12}[difficulty]

    # Null injection
    null_targets = rng.sample(range(n_rows), min(3, n_rows//3))
    for idx in null_targets:
        dirty_df.at[idx, "age"] = None
        issue_manifest.append({"row": idx, "col": "age",
                                "issue_type": "null", "expected": int(clean_df.at[idx,"age"])})

    # Type error injection
    dirty_df["revenue"] = dirty_df["revenue"].astype(object)
    type_targets = rng.sample(range(n_rows), 2)
    for idx in type_targets:
        dirty_df.at[idx, "revenue"] = "N/A"
        issue_manifest.append({"row": idx, "col": "revenue",
                                "issue_type": "type_error",
                                "expected": float(clean_df.at[idx,"revenue"])})

    # Date format corruption (mix MM/DD/YYYY into some rows)
    date_targets = rng.sample(range(n_rows), 2)
    for idx in date_targets:
        d = datetime.strptime(clean_df.at[idx,"signup_date"], "%Y-%m-%d")
        dirty_df.at[idx,"signup_date"] = d.strftime("%m/%d/%Y")
        issue_manifest.append({"row": idx, "col": "signup_date",
                                "issue_type": "date_format",
                                "expected": clean_df.at[idx,"signup_date"]})

    # Duplicate row injection
    dup_idx = rng.randint(0, n_rows-2)
    dup_row = dirty_df.iloc[dup_idx].copy()
    dirty_df = pd.concat([dirty_df, pd.DataFrame([dup_row])], ignore_index=True)
    issue_manifest.append({"row": len(dirty_df)-1, "col": "*",
                            "issue_type": "duplicate", "expected": "drop"})

    # Email corruption
    email_idx = rng.randint(0, n_rows-1)
    dirty_df.at[email_idx,"email"] = dirty_df.at[email_idx,"email"].replace("@","")
    issue_manifest.append({"row": email_idx, "col": "email",
                            "issue_type": "malformed", "expected": clean_df.at[email_idx,"email"]})

    return dirty_df, clean_df, issue_manifest
