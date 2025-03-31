import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import snowflake.connector
from sklearn.linear_model import LinearRegression
from langchain_openai import ChatOpenAI
import json

load_dotenv()

# -------------------- Snowflake Query --------------------
def query_snowflake(query: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -------------------- Prompt Parsing --------------------
def infer_metric(prompt: str) -> str:
    p = prompt.lower()
    if "death" in p:
        return "TOTAL_DEATHS"
    elif "injured" in p:
        return "NO_INJURED"
    elif "affected" in p:
        return "NO_AFFECTED"
    elif "homeless" in p:
        return "NO_HOMELESS"
    elif "damage" in p and "insured" in p:
        return "INSURED_DAMAGE_000_USD"
    elif "damage" in p and "reconstruction" in p:
        return "RECONSTRUCTION_COSTS_000_USD"
    elif "damage" in p:
        return "TOTAL_DAMAGE_000_USD"
    return "TOTAL_AFFECTED"

def extract_filters(prompt: str) -> list:
    filters = []
    if "flood" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Flood'")
    if "earthquake" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Earthquake'")
    if "cyclone" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Cyclone'")
    if "tsunami" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Tsunami'")
    if "wildfire" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Wildfire'")
    if "drought" in prompt.lower():
        filters.append("DISASTER_TYPE = 'Drought'")
    if "asia" in prompt.lower():
        filters.append("REGION = 'Asia'")
    if "africa" in prompt.lower():
        filters.append("REGION = 'Africa'")
    if "europe" in prompt.lower():
        filters.append("REGION = 'Europe'")
    if "americas" in prompt.lower():
        filters.append("REGION = 'Americas'")
    return filters

# -------------------- Forecast & Report --------------------
def generate_forecast(prompt: str) -> dict:
    output = {
        "data": None,
        "charts": {},
        "analysis": {},
        "error": None
    }
    
    metric = infer_metric(prompt)
    filters = extract_filters(prompt)
    where_clause = f"WHERE START_YEAR IS NOT NULL AND {metric} IS NOT NULL"
    if filters:
        where_clause += " AND " + " AND ".join(filters)

    query = f"""
    SELECT START_YEAR, SUM({metric}) AS {metric}
    FROM DISASTER_TABLE
    {where_clause}
    GROUP BY START_YEAR
    ORDER BY START_YEAR
    """

    try:
        df = query_snowflake(query)
        df = df.dropna(subset=[metric])
        if df.empty or df.shape[0] < 5:
            output["status"] = "warning"
            output["error"] = "Not enough data for forecasting."
            return output

        df["START_YEAR"] = df["START_YEAR"].astype(int)
        X = df[["START_YEAR"]]
        y = df[metric]
        model = LinearRegression().fit(X, y)

        future_years = list(range(df["START_YEAR"].max() + 1, df["START_YEAR"].max() + 6))
        future_df = pd.DataFrame({"START_YEAR": future_years})
        future_df[metric] = model.predict(future_df[["START_YEAR"]])
        combined = pd.concat([df, future_df], ignore_index=True)

        # Store data in output
        output["data"] = {
            "metric": metric,
            "filters": filters,
            "forecast_table": combined.to_dict(orient="records"),
            "forecast_years": future_years
        }

        # ------------------ Charts ------------------
        chart_path = f"forecast_{metric.lower()}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(combined["START_YEAR"], combined[metric], marker='o', label=metric.replace("_", " ").title())
        plt.axvline(x=df["START_YEAR"].max() + 0.5, color='r', linestyle='--', label="Forecast starts")
        plt.title(f"{metric.replace('_', ' ').title()} Forecast")
        plt.xlabel("Year")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        output["charts"]["forecast"] = chart_path

        # Bar chart
        bar_chart_path = f"historical_{metric.lower()}_bar.png"
        plt.figure(figsize=(10, 6))
        plt.bar(df["START_YEAR"], df[metric])
        plt.title(f"Historical {metric.replace('_', ' ').title()}")
        plt.xlabel("Year")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(bar_chart_path)
        plt.close()
        output["charts"]["historical"] = bar_chart_path

        # Growth % chart
        growth_chart_path = f"growth_{metric.lower()}.png"
        df_sorted = df.sort_values("START_YEAR")
        df_sorted["GROWTH_%"] = df_sorted[metric].pct_change() * 100
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted["START_YEAR"], df_sorted["GROWTH_%"], marker="o", color="orange")
        plt.title(f"Growth Rate in {metric.replace('_', ' ').title()}")
        plt.xlabel("Year")
        plt.ylabel("Growth %")
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.tight_layout()
        plt.savefig(growth_chart_path)
        plt.close()
        output["charts"]["growth"] = growth_chart_path

        # ------------------ GPT-4 Report ------------------
        llm = ChatOpenAI(model="gpt-4", temperature=0.4)
        markdown = combined.to_markdown(index=False)

        sections = {
            "trend_analysis": f"Analyze this year-wise trend for {metric.replace('_', ' ').lower()}:\n\n{markdown}",
            "growth_decline_phases": f"Based on the following data, identify the biggest spikes and drops in {metric.replace('_', ' ').lower()} over time. Explain what patterns emerge:\n\n{markdown}",
            "forecast_interpretation": f"Based on the historical and forecasted values from this dataset, explain the trends from {future_years[0]} to {future_years[-1]}:\n\n{markdown}",
            "risk_implications": f"What does this data suggest about future risks or vulnerabilities related to {metric.replace('_', ' ').lower()}? Here is the data:\n\n{markdown}",
            "conclusion": f"Summarize insights, patterns, and uncertainties using this dataset:\n\n{markdown}"
        }

        for section_name, task in sections.items():
            output["analysis"][section_name] = llm.invoke(task).content

        return output

    except Exception as e:
        output["status"] = "error"
        output["error"] = str(e)
        return output

if __name__ == "__main__":
    user_prompt = input("💬 Enter a forecasting prompt: ")
    result = generate_forecast(user_prompt)
    
    print(result)
    
    