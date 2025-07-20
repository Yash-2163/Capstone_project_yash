import pandas as pd
import psycopg2

def load_data(host, port, database, user, password, target_col='Converted'):
    """
    Connects to Redshift using provided parameters,
    runs a query, and returns features (X) and target (y).
    """
    query = "SELECT * FROM lead_conversion_data"

    # Connect to Redshift
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password
    )

    df = pd.read_sql(query, conn)
    conn.close()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y
