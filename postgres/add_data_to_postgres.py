#!/usr/bin/env python3
# append_to_postgres.py

import os
import re
import sys  # Added missing import for sys.exit()
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# ─── Step 0: Hard-coded settings ──────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)
CSV_PATH     = os.path.join(SCRIPT_DIR, "..", "data", "new_data.csv")
PG_HOST      = "localhost"
PG_PORT      = 5432
PG_DB        = "lead_db"
PG_USER      = "postgres"
PG_PASS      = "yashrajput"
TARGET_TABLE = "public.lead_scoring"

# ─── Utility: Clean & dedupe column names ────────────────────────────────
def clean_and_dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names to snake_case, removes special characters,
    and deduplicates names by appending a counter.
    """
    def to_snake(name: str) -> str:
        name = re.sub(r"\s+", " ", name.strip())
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
        return re.sub(r"_+", "_", name).strip("_").lower()

    new_cols = [to_snake(col) for col in df.columns]
    counts, final = {}, []
    for col in new_cols:
        if col not in counts:
            counts[col] = 0
            final.append(col)
        else:
            counts[col] += 1
            final.append(f"{col}_{counts[col]}")
    df.columns = final
    return df

# ─── Step 1: Create DB if it doesn't exist ────────────────────────────────
def create_database_if_not_exists():
    """
    Connects to the default 'postgres' database and creates 'lead_db'
    if it does not already exist.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST, user="postgres", password=PG_PASS,
            port=PG_PORT, dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DB,))
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {PG_DB}")
            print(f"✅ Created database '{PG_DB}'")
        else:
            print(f"ℹ️  Database '{PG_DB}' already exists")
        cur.close()
    except psycopg2.Error as e:
        print(f"Database creation error: {e}")
        raise
    finally:
        if conn:
            conn.close()

create_database_if_not_exists()

# ─── Step 2: Load & clean CSV ─────────────────────────────────────────────
print(f"▶ Loading CSV from {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"  → {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_PATH}. Please ensure 'new_data.csv' exists in the 'data' directory.")
    sys.exit(1)

df = clean_and_dedupe_columns(df)
print(f"▶ Cleaned columns: {df.columns.tolist()}")

# --- Process 'converted' column to be numerical (0 or 1) ---
if 'converted' in df.columns:
    # Use a mapping approach for robustness, similar to data_loader
    conversion_map = {
        'Yes': 1, 'True': 1, '1': 1,
        'No': 0, 'False': 0, '0': 0
    }
    # Convert column to string to handle mixed types, then map
    df['converted'] = df['converted'].astype(str).map(conversion_map)
    
    # Coerce any unmapped values (like empty strings) to NaN, then fill with 0
    df['converted'] = pd.to_numeric(df['converted'], errors='coerce').fillna(0)
    df['converted'] = df['converted'].astype(int)
    print(f"▶ 'converted' column processed to dtype: {df['converted'].dtype}, unique values: {df['converted'].unique().tolist()}")
else:
    print("Warning: 'converted' column not found in CSV after cleaning.")

# ─── Step 3: Connect, Create Table, and Insert Data ─────────────────────
conn = None
cur = None
try:
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    cur = conn.cursor()
    print(f"▶ Connected to Postgres at {PG_HOST}:{PG_PORT}/{PG_DB}")

    # ─── Step 4: Ensure Table Exists (with corrected column names) ────────
    # CRITICAL FIX: The column names here now match the snake_case output
    # of the clean_and_dedupe_columns function.
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            prospect_id TEXT,
            lead_number BIGINT,
            lead_origin TEXT,
            lead_source TEXT,
            do_not_email TEXT,
            do_not_call TEXT,
            converted INTEGER,
            total_visits DOUBLE PRECISION,
            total_time_spent_on_website INTEGER,
            page_views_per_visit DOUBLE PRECISION,
            last_activity TEXT,
            country TEXT,
            specialization TEXT,
            how_did_you_hear_about_x_education TEXT,
            what_is_your_current_occupation TEXT,
            what_matters_most_to_you_in_choosing_a_course TEXT,
            search TEXT,
            magazine TEXT,
            newspaper_article TEXT,
            x_education_forums TEXT,
            newspaper TEXT,
            digital_advertisement TEXT,
            through_recommendations TEXT,
            receive_more_updates_about_our_courses TEXT,
            tags TEXT,
            lead_quality TEXT,
            update_me_on_supply_chain_content TEXT,
            get_updates_on_dm_content TEXT,
            lead_profile TEXT,
            city TEXT,
            asymmetrique_activity_index TEXT,
            asymmetrique_profile_index TEXT,
            asymmetrique_activity_score DOUBLE PRECISION,
            asymmetrique_profile_score DOUBLE PRECISION,
            i_agree_to_pay_the_amount_through_cheque TEXT,
            a_free_copy_of_mastering_the_interview TEXT,
            last_notable_activity TEXT
        );
    """)
    conn.commit()
    print(f"▶ Ensured table {TARGET_TABLE} exists")

    # ─── Step 5: Bulk-insert ──────────────────────────────────────────────
    cols = df.columns.tolist()
    # Convert DataFrame to list of tuples, replacing pandas NaN with None
    df_clean = df.astype(object).where(pd.notnull(df), None)
    values = [tuple(row) for row in df_clean.to_numpy()]

    sql = f"INSERT INTO {TARGET_TABLE} ({','.join(cols)}) VALUES %s"
    execute_values(cur, sql, values)
    conn.commit()
    print(f"✅ Inserted {len(values)} rows into {TARGET_TABLE}")

except psycopg2.Error as e:
    print(f"Database connection or operation error: {e}")
    raise
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    raise
finally:
    if cur:
        cur.close()
    if conn:
        conn.close()
    print("🎉 Done. Connection closed.")