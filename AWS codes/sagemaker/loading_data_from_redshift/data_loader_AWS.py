import boto3
import time
import pandas as pd
 
# Create Redshift Data API client
client = boto3.client('redshift-data', region_name='ap-south-1')
 
# Execute the SQL query
response = client.execute_statement(
    WorkgroupName='capstonetest',
    Database='dev',
    Sql='SELECT * FROM public.lead_conversion_data limit 100;'
)
 
statement_id = response['Id']
 
# Wait until the query is finished
while True:
    status = client.describe_statement(Id=statement_id)
    if status['Status'] in ['FAILED', 'ABORTED']:
        raise Exception(f"Query failed: {status.get('Error', 'Unknown error')}")
    elif status['Status'] == 'FINISHED':
        break
    time.sleep(1)
 
# Get the query results
results = client.get_statement_result(Id=statement_id)
 
# Extract column names
column_names = [col['name'] for col in results['ColumnMetadata']]
 
# Function to extract correct value type
def extract_value(col):
    if 'stringValue' in col:
        return col['stringValue']
    elif 'longValue' in col:
        return col['longValue']
    elif 'booleanValue' in col:
        return col['booleanValue']
    elif 'doubleValue' in col:
        return col['doubleValue']
    else:
        return None  # or ""
 
# Extract rows
data = []
for record in results['Records']:
    row = [extract_value(col) for col in record]
    data.append(row)
 
# Create DataFrame
df = pd.DataFrame(data, columns=column_names)
 
# Save as CSV
df.to_csv('lead_data.csv', index=False)
print("Data saved to lead_data.csv")
 