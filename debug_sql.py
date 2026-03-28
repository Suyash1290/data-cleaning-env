import duckdb
import pandas as pd
from data.generator import generate_dataset
from tasks.hard import run_sql_against_data

dirty, clean, _ = generate_dataset(42, 100, "hard")
sql = "SELECT DISTINCT * FROM raw_data WHERE id IS NOT NULL AND age >= 0 AND age <= 120 AND email LIKE '%@%' AND regexp_matches(CAST(signup_date AS VARCHAR), '^\\d{4}-\\d{2}-\\d{2}$')"
passed, err = run_sql_against_data(dirty, sql)
print("err:", err)
print("passed:", passed)
