import duckdb
import pandas as pd

VALIDATION_RULES = [
    ("no_null_ids",     "SELECT COUNT(*) FROM result WHERE id IS NULL",            0),
    ("no_duplicates",   "SELECT COUNT(*) - COUNT(DISTINCT id) FROM result",         0),
    ("valid_ages",      "SELECT COUNT(*) FROM result WHERE age < 0 OR age > 120",   0),
    ("valid_emails",    "SELECT COUNT(*) FROM result WHERE email NOT LIKE '%@%'",   0),
    ("valid_dates",     "SELECT COUNT(*) FROM result WHERE NOT regexp_matches(signup_date, '^\\d{4}-\\d{2}-\\d{2}$') AND signup_date IS NOT NULL", 0),
]

_con = None

def get_connection():
    global _con
    if _con is None:
        _con = duckdb.connect(":memory:")
    return _con

def run_sql_against_data(dirty_df: pd.DataFrame, sql: str):
    con = get_connection()
    con.register("raw_data", dirty_df)
    try:
        result = con.execute(sql).df()
        con.register("result", result)
        passed = []
        for rule_id, query, expected in VALIDATION_RULES:
            val = con.execute(query).fetchone()[0]
            passed.append(val == expected)
        return passed, None
    except Exception as e:
        return [False] * 5, str(e)

def grade(state) -> float:
    from graders.fix_grader import grade as fix_grade
    return fix_grade(state)
