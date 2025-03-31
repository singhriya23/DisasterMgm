import os
from dotenv import load_dotenv
import snowflake.connector
import pandas as pd
from typing import Optional, Union, Tuple, Dict

class SnowflakeDataRetrievalAgent:
    def __init__(self):
        """Initialize connection to Snowflake disaster database"""
        load_dotenv()
        self.conn_params = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": "DISASTER_DB",
            "schema": "DISASTER_SCHEMA"
        }
        self.table_name = "DISASTER_TABLE"  # Updated to match your actual table name

    def _get_connection(self):
        """Establish Snowflake connection"""
        return snowflake.connector.connect(**self.conn_params)

    def retrieve_data(
        self,
        disaster_type: Optional[str] = None,
        country: Optional[str] = None,
        year: Optional[Union[int, Tuple[int, int]]] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Retrieve disaster data from Snowflake based on criteria.
        
        Args:
            disaster_type: Filter by disaster type (e.g., 'flood')
            country: Filter by country name (e.g., 'brazil')
            year: Either a single year or tuple (start_year, end_year)
            limit: Maximum number of records to return
            
        Returns:
            Pandas DataFrame with matching records
        """
        # Base query - updated to match your exact column names
        query = f"""
        SELECT 
            DISNO, HISTORIC, DISASTER_TYPE, DISASTER_SUBTYPE, 
            EVENT_NAME, COUNTRY, REGION, SUBREGION,
            START_YEAR, START_MONTH, START_DAY,
            TOTAL_DEATHS, NO_AFFECTED, TOTAL_DAMAGE_000_USD,
            LATITUDE, LONGITUDE, LOCATION
        FROM {self.table_name}
        WHERE 1=1
        """
        
        # Add filters
        params = {}
        if disaster_type:
            query += " AND LOWER(DISASTER_TYPE) = %(disaster_type)s"
            params['disaster_type'] = disaster_type.lower()
            
        if country:
            query += " AND LOWER(COUNTRY) = %(country)s"
            params['country'] = country.lower()
            
        if year:
            if isinstance(year, tuple):
                query += " AND START_YEAR BETWEEN %(start_year)s AND %(end_year)s"
                params['start_year'], params['end_year'] = year
            else:
                query += " AND START_YEAR = %(year)s"
                params['year'] = year
                
        query += f" LIMIT {limit}"
        
        # Execute query
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(query, params)
            
            # Get results as DataFrame
            columns = [col[0] for col in cur.description]
            data = cur.fetchall()
            df = pd.DataFrame(data, columns=columns)
            
            # Convert numeric columns
            numeric_cols = ['TOTAL_DEATHS', 'NO_AFFECTED', 'TOTAL_DAMAGE_000_USD', 
                          'LATITUDE', 'LONGITUDE', 'START_YEAR', 'START_MONTH', 'START_DAY']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return pd.DataFrame(columns=['EVENT_NAME', 'START_YEAR', 'TOTAL_DEATHS', 'NO_AFFECTED'])
            
        finally:
            if 'conn' in locals():
                conn.close()

    def get_summary_stats(self, filtered_df: pd.DataFrame) -> Dict:
        """
        Generate basic summary statistics from filtered data.
        
        Args:
            filtered_df: DataFrame returned from retrieve_data()
            
        Returns:
            Dictionary with summary statistics
        """
        if filtered_df.empty:
            return {"message": "No data matching the criteria"}
            
        stats = {
            "total_events": len(filtered_df),
            "countries": filtered_df['COUNTRY'].unique().tolist(),
            "years": {
                "min": int(filtered_df['START_YEAR'].min()),
                "max": int(filtered_df['START_YEAR'].max())
            },
            "disaster_types": filtered_df['DISASTER_TYPE'].unique().tolist(),
            "total_deaths": int(filtered_df['TOTAL_DEATHS'].sum()),
            "total_affected": int(filtered_df['NO_AFFECTED'].sum()),
            "total_damage_usd": float(filtered_df['TOTAL_DAMAGE_000_USD'].sum() * 1000)
            if 'TOTAL_DAMAGE_000_USD' in filtered_df.columns else None,
            "sample_events": filtered_df[['EVENT_NAME', 'START_YEAR', 'LOCATION']].head().to_dict('records')
        }
        
        return stats

# Example Usage
if __name__ == "__main__":
    # Initialize agent
    agent = SnowflakeDataRetrievalAgent()
    
    # Example 1: Floods in Brazil (matches input parser output)
    print("Example 1: Floods in Brazil")
    brazil_floods = agent.retrieve_data(
        disaster_type="flood",
        country="brazil"
    )
    print(brazil_floods[['EVENT_NAME', 'START_YEAR', 'TOTAL_DEATHS', 'NO_AFFECTED']].head())
    print("\nStats:", agent.get_summary_stats(brazil_floods))
    
    # Example 2: Earthquakes in Mexico 2015
    print("\nExample 2: Earthquakes in Mexico 2015")
    mexico_quakes = agent.retrieve_data(
        disaster_type="earthquake",
        country="mexico",
        year=2015
    )
    print(mexico_quakes[['EVENT_NAME', 'START_YEAR', 'TOTAL_DEATHS', 'NO_AFFECTED']].head())
    print("\nStats:", agent.get_summary_stats(mexico_quakes))
    
    # Example 3: Wildfires in Canada (no year filter)
    print("\nExample 3: Wildfires in Canada")
    canada_fires = agent.retrieve_data(
        disaster_type="wildfire",
        country="canada"
    )
    print(canada_fires[['EVENT_NAME', 'START_YEAR', 'TOTAL_DEATHS', 'NO_AFFECTED']].head())
    print("\nStats:", agent.get_summary_stats(canada_fires))