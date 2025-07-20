# Loading data from  postgres

import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

def load_data(target_col='converted'):
    """
    Load data from a local PostgreSQL database.
    
    This function connects to the database, retrieves the lead scoring data,
    normalizes column names, and explicitly processes all columns into their
    correct data types (numerical and object).

    Returns:
        X (pd.DataFrame): DataFrame containing the feature columns.
        y (pd.Series): Series containing the numerical target column.
    """
    # ─── Step 1: Get Database Connection Parameters ──────────────────────
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "yashrajput")
    DB_HOST = os.getenv("POSTGRES_HOST", "host.docker.internal")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "lead_db")

    # ─── Step 2: Construct Connection URI ────────────────────────────────
    conn_uri = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # ─── Step 3: Connect, Query, and Process Data ────────────────────────
    conn = None
    try:
        # Establish the database connection
        conn = psycopg2.connect(conn_uri, cursor_factory=RealDictCursor)
        
        # Define the SQL query to fetch all data
        sql = "SELECT * FROM public.lead_scoring;"
        
        # Execute the query and load data into a pandas DataFrame
        df = pd.read_sql(sql, conn)

        # Immediately normalize all column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Check if any data was loaded
        if df.empty:
            raise ValueError("No data was loaded from the database. Check the table and connection.")

        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # --- START: ROBUST DATA TYPE CONVERSION ---
        # First, convert all other numeric columns
        other_numeric_cols = [
            'lead_number', 'total_visits', 
            'total_time_spent_on_website', 'page_views_per_visit',
            'asymmetrique_activity_score', 'asymmetrique_profile_score'
        ]
        
        print("\nConverting feature columns to numeric types...")
        for col in other_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Second, handle the target column with a more robust two-step conversion
        print("\nProcessing target column 'converted'...")
        if target_col in df.columns:
            # Step 1: Force the column to a numeric type. 'coerce' turns any non-numeric
            # values (like empty strings) into NaN.
            numeric_series = pd.to_numeric(df[target_col], errors='coerce')
            
            # Step 2: Apply a function. If the value is exactly 1, map to 1.
            # Everything else (0, NaN, other numbers) becomes 0.
            df[target_col] = numeric_series.apply(lambda x: 1 if x == 1 else 0)
            df[target_col] = df[target_col].astype(int)
        else:
            raise KeyError(f"Target column '{target_col}' not found after lowercasing.")

        print("\nData types after conversion:")
        print(df.dtypes)
        print("-" * 30)
        # --- END: ROBUST DATA TYPE CONVERSION ---


        # Ensure the target column exists in the DataFrame
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in table columns: {df.columns.tolist()}")

        # Final verification to ensure we have both classes for modeling
        if len(df[target_col].unique()) < 2:
            raise ValueError(
                f"After processing, target column '{target_col}' has fewer than two unique classes: "
                f"{df[target_col].unique()}. Please check the raw data for class imbalance or mapping issues."
            )
        
        # ─── Step 4: Split into Features (X) and Target (y) ─────────────
        X = df.drop(columns=[target_col])
        y = df[target_col] # Return a pandas Series to preserve metadata

        # Restored print statements for better logging in the pipeline
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Unique values in target (y): {y.unique()}")
        print(f"Data type of target (y): {y.dtype}")

        return X, y

    except psycopg2.Error as e:
        print(f"Database connection or query error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in data loading: {e}")
        raise
    finally:
        # Ensure the database connection is always closed
        if conn:
            conn.close()
