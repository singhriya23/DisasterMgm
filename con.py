import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load credentials
load_dotenv()

def extract_distinct_country_year():
    """Extract distinct countries and years from the disaster table"""
    try:
        # Establish connection
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database="DISASTER_DB",
            schema="DISASTER_SCHEMA"
        )
        
        # Execute query to get distinct countries and years
        query = """
        SELECT DISTINCT START_YEAR
        FROM DISASTER_DB.DISASTER_SCHEMA.DISASTER_TABLE
        """
        
        df = pd.read_sql(query, conn)
        
        # Save to CSV
        output_file = "distinct_country_year_data.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Success! Distinct countries and years saved to {output_file}")
        
        # Print to console
        print("\nDistinct Country and Year Data:")
        print(df.to_markdown(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    extract_distinct_country_year()
