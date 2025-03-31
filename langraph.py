from typing import TypedDict, Optional, Dict, Any
from inputparser_agent import parse_disaster_prompt
from langgraph.graph import END, StateGraph
from dataretrieve import SnowflakeDataRetrievalAgent
from statistical import DisasterStatsAnalyzer
import pandas as pd
from visualize import DisasterDashboard
from report import ReportSynthesisAgent
from forecasting import generate_forecast

# Initialize agents
data_agent = SnowflakeDataRetrievalAgent()


class AgentState(TypedDict):
    prompt: str
    disaster_type: Optional[str]
    country: Optional[str]
    year: Optional[int]
    validation_error: Optional[str]
    output: Optional[str]
    raw_data: Optional[pd.DataFrame]
    summary_stats: Optional[Dict[str, Any]]
    enhanced_stats: Optional[Dict[str, Any]]
    event_patterns: Optional[Dict[str, Any]]
    dashboard_path: Optional[str]
    report_path: Optional[str]
    forecast_data: Optional[Dict[str, Any]]  # New field for forecast data


# Define nodes
def parse_input(state: AgentState):
    prompt = state["prompt"]
    disaster, country, year = parse_disaster_prompt(prompt)
    
    return {
        **state,
        "disaster_type": disaster,
        "country": country,
        "year": year,
        "output": None,
        "raw_data": None,
        "summary_stats": None,
        "enhanced_stats": None,
        "event_patterns": None
    }

def validate_input(state: AgentState):
    if not state["disaster_type"] or not state["country"]:
        return {
            **state,
            "validation_error": "Missing required fields (disaster type and country)",
            "output": "Error: Missing required fields"
        }
    return state

def handle_error(state: AgentState):
    error = state.get("validation_error", "Unknown error")
    return {
        **state,
        "output": f"Input error: {error}"
    }

def retrieve_data(state: AgentState):
    try:
        # Retrieve data from Snowflake
        df = data_agent.retrieve_data(
            disaster_type=state["disaster_type"],
            country=state["country"],
            year=state["year"]
        )
        
        # Get summary statistics
        stats = data_agent.get_summary_stats(df)
        
        return {
            **state,
            "raw_data": df,
            "summary_stats": stats,
            "output": "Data retrieved successfully"
        }
        
    except Exception as e:
        return {
            **state,
            "validation_error": f"Data retrieval failed: {str(e)}",
            "output": f"Error retrieving data: {str(e)}"
        }

def analyze_statistics(state: AgentState):
    if not state.get("summary_stats"):
        return {
            **state,
            "validation_error": "No statistics available for analysis",
            "output": "Error: No data available for analysis"
        }
    
    try:
        analyzer = DisasterStatsAnalyzer(state["summary_stats"])
        
        enhanced_stats = analyzer.calculate_core_stats()
        event_patterns = analyzer.analyze_event_patterns() or {}  # Ensure we always get a dict
        
        # Ensure all required keys exist in event_patterns
        if 'common_locations' not in event_patterns:
            event_patterns['common_locations'] = []
        if 'events_per_year' not in event_patterns:
            event_patterns['events_per_year'] = {}
        if 'geospatial_clusters' not in event_patterns:
            event_patterns['geospatial_clusters'] = {}
        
        return {
            **state,
            "enhanced_stats": enhanced_stats,
            "event_patterns": event_patterns,
            "output": "Statistical analysis completed",
            "summary_stats": {**state["summary_stats"], **event_patterns}  # Merge for visualization
        }
        
    except Exception as e:
        return {
            **state,
            "validation_error": f"Statistical analysis failed: {str(e)}",
            "output": f"Error during analysis: {str(e)}"
        }


def generate_visualizations(state: AgentState):
    if state.get("validation_error"):
        return state
        
    if not state.get("summary_stats"):
        return {
            **state,
            "output": "No data available for visualization",
            "dashboard_path": None
        }
    
    try:
        # Create visualization dashboard
        dashboard = DisasterDashboard(state["summary_stats"])
        output_path = dashboard.create_dashboard()
        
        return {
            **state,
            "dashboard_path": output_path,
            "output": f"Visualizations generated at {output_path}"
        }
    except Exception as e:
        return {
            **state,
            "validation_error": f"Visualization failed: {str(e)}",
            "output": f"Error generating visualizations: {str(e)}"
        }
# Add this new node function to your existing code
def generate_forecast_data(state: AgentState):
    if state.get("validation_error"):
        return state
        
    if not state.get("disaster_type") or not state.get("country"):
        return {
            **state,
            "validation_error": "Missing required fields for forecast",
            "output": "Error: Missing disaster type or country for forecast"
        }
    
    try:
        # Create a prompt for the forecast agent
        forecast_prompt = f"Analyze {state['disaster_type']} in {state['country']}"
        if state.get("year"):
            forecast_prompt += f" for year {state['year']}"
        
        # Generate forecast data
        forecast_data = generate_forecast(forecast_prompt)
        
        if forecast_data.get("error"):
            return {
                **state,
                "validation_error": f"Forecast generation failed: {forecast_data['error']}",
                "output": f"Error generating forecast: {forecast_data['error']}"
            }
        
        return {
            **state,
            "forecast_data": forecast_data,
            "output": "Forecast data generated successfully"
        }
        
    except Exception as e:
        return {
            **state,
            "validation_error": f"Forecast generation failed: {str(e)}",
            "output": f"Error generating forecast: {str(e)}"
        }


def generate_report(state: AgentState):
    if state.get("validation_error"):
        return state
        
    required_data = [
        state.get("disaster_type"),
        state.get("country"),
        state.get("summary_stats"),
        state.get("enhanced_stats"),
        state.get("event_patterns")
    ]
    
    if not all(required_data):
        return {
            **state,
            "validation_error": "Insufficient data for report generation",
            "output": "Error: Missing required data for report"
        }
    
    try:
        # Prepare input data with defaults for missing values
        input_data = {
            "disaster_type": state["disaster_type"],
            "country": state["country"],
            "year": state.get("year")
        }
        
        data_stats = state["summary_stats"].copy()
        analysis = state["enhanced_stats"].copy()
        
        # Ensure all required keys exist
        analysis['event_patterns'] = state.get("event_patterns", {})
        if 'total_damage_usd' not in data_stats:
            data_stats['total_damage_usd'] = 0
        
        report_agent = ReportSynthesisAgent(
            input_data=input_data,
            data_stats=data_stats,
            analysis=analysis,
            dashboard_path=state.get("dashboard_path", ""),
            forecast_data=state.get("forecast_data")  # Pass forecast data to report agent
        )
        
        report_path = report_agent.generate_report()
        
        return {
            **state,
            "report_path": report_path,
            "output": f"Final report generated at {report_path}"
        }
        
    except Exception as e:
        return {
            **state,
            "validation_error": f"Report generation failed: {str(e)}",
            "output": f"Error generating report: {str(e)}"
        }

def format_final_output(state: AgentState):
    if state.get("validation_error"):
        return {
            **state,
            "output": f"❌ Processing failed: {state['validation_error']}"
        }
        
    # Base success message
    output_msg = (
        f"✅ Processing completed successfully!\n"
        f"Disaster: {state['disaster_type'].title()}\n"
        f"Country: {state['country'].title()}\n"
    )
    
    if state.get("year"):
        output_msg += f"Year: {state['year']}\n"
    
    # Add visualization info if available
    if state.get("dashboard_path"):
        output_msg += f"\n📊 Dashboard: file://{state['dashboard_path']}\n"
    
    # Add report info if available
    if state.get("report_path"):
        output_msg += f"\n📄 Full Report: file://{state['report_path']}\n"
    
    return {
        **state,
        "output": output_msg
    }


# Update your workflow construction to include the new node
workflow = StateGraph(AgentState)

# Add all your existing nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("validate_input", validate_input)
workflow.add_node("handle_error", handle_error)
workflow.add_node("retrieve_data", retrieve_data)
workflow.add_node("analyze_statistics", analyze_statistics)
workflow.add_node("generate_visualizations", generate_visualizations)
workflow.add_node("generate_forecast", generate_forecast_data)  # New node
workflow.add_node("generate_report", generate_report)
workflow.add_node("format_output", format_final_output)

workflow.set_entry_point("parse_input")

# Update the edges to include the forecast step
workflow.add_edge("parse_input", "validate_input")
workflow.add_conditional_edges(
    "validate_input",
    lambda state: "handle_error" if "validation_error" in state else "retrieve_data",
)
workflow.add_edge("retrieve_data", "analyze_statistics")
workflow.add_edge("analyze_statistics", "generate_visualizations")
workflow.add_edge("generate_visualizations", "generate_forecast")  # New edge
workflow.add_edge("generate_forecast", "generate_report")  # New edge
workflow.add_edge("generate_report", "format_output")

workflow.add_edge("handle_error", END)
workflow.add_edge("format_output", END)

app = workflow.compile()


# Example usage
if __name__ == "__main__":
    prompts = [
        "Do an analysis on flash flood in Mexico"
    ]
    
    for prompt in prompts:
        print(f"\nProcessing prompt: '{prompt}'")
        result = app.invoke({"prompt": prompt})
        print(result["output"])

         # Print paths if available
        if result.get("dashboard_path"):
            print(f"Visualization: file://{result['dashboard_path']}")
        if result.get("report_path"):
            print(f"Report: file://{result['report_path']}")