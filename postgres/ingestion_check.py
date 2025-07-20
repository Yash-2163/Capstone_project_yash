import psycopg2
import pandas as pd

try:
    # --- Step 1: Connect to the database ---
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="lead_db",
        user="postgres",
        password="yashrajput"
    )
    cur = conn.cursor()
    print("✅ Successfully connected to the database.")

    # --- Step 2: Inspect the 'lead_scoring' table ---
    print("\n▶ Inspecting 'public.lead_scoring' table...")

    # Check the data type of the 'converted' column from the schema
    cur.execute("""
        SELECT data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
          AND table_name = 'lead_scoring' 
          AND column_name = 'converted';
    """)
    dtype_result = cur.fetchone()
    if dtype_result:
        print(f"→ 'converted' Column Data Type: {dtype_result[0]}")
    else:
        print("→ 'converted' Column Data Type: Column not found.")

    # Check the total number of rows in the table
    cur.execute("SELECT COUNT(*) FROM public.lead_scoring;")
    row_count = cur.fetchone()[0]
    print(f"→ Total Row Count             : {row_count}")

    # --- Step 3: Run the requested query for unique values ---
    print("\n▶ Running 'SELECT DISTINCT converted FROM public.lead_scoring;'...")
    cur.execute("SELECT DISTINCT converted FROM public.lead_scoring;")
    
    # Fetch all unique values and format them for printing
    unique_values = [row[0] for row in cur.fetchall()]
    
    if unique_values:
        print(f"→ Unique Values Found         : {unique_values}")
    else:
        print("→ Unique Values Found         : The table is empty.")

except psycopg2.Error as e:
    print(f"\n❌ Database error: {e}")
finally:
    # --- Step 4: Close the connection ---
    if 'conn' in locals() and conn:
        if 'cur' in locals() and cur:
            cur.close()
        conn.close()
        print("\n🎉 Connection closed.")
