import pandas as pd
from sqlalchemy import create_engine
import pyodbc

engine = create_engine(
    r"mssql+pyodbc://LAPTOP-7643IQJN\SQLEXPRESS/telecom?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )

# Define tables to export
tables = ["voda"] 

# Export each table to CSV
for table in tables:
    query = f"SELECT * FROM voda"
    df = pd.read_sql(query, engine)
    output_path = rf"C:\Users\Gaurav Nailwal\OneDrive\Desktop\Vodafone-Qatar\{table}.csv"

    df.to_csv(output_path, index=False)
