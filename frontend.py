import streamlit as st
import requests
from datetime import datetime
import os
import time

# Configuration
#FASTAPI_URL = "http://localhost:8000"
FASTAPI_URL = "https://agentic-backend-343736309329.us-central1.run.app"  # Update this if your API is hosted elsewhere

def main():
    st.set_page_config(
        page_title="Disaster Analysis Dashboard",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Disaster Analysis System")
    st.markdown("""
    Analyze disaster data by entering a natural language prompt below. 
    The system will retrieve data, analyze statistics, and generate visualizations and reports.
    """)
    
    with st.form("analysis_form"):
        prompt = st.text_area(
            "Enter your analysis request (e.g., 'Do an analysis on flash flood in Mexico'):",
            height=100,
            placeholder="Analyze earthquakes in Japan from 2010 to 2020"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            save_report = st.checkbox("Save report locally", value=True)
        with col2:
            output_dir = st.text_input("Output directory", "reports")
        
        submitted = st.form_submit_button("Analyze")
    
    if submitted and prompt:
        with st.spinner("Processing your request..."):
            try:
                # Call FastAPI endpoint
                response = requests.post(
                    f"{FASTAPI_URL}/analyze",
                    json={
                        "prompt": prompt,
                        "save_report": save_report,
                        "output_dir": output_dir
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_result = result
                    
                    if result["status"] == "success":
                        st.success("Analysis completed successfully!")
                        st.markdown("### Results")
                        
                        # Display the message with proper formatting
                        st.markdown(result["message"].replace("file://", ""))
                        
                        # Create columns for download buttons
                        col1, col2 = st.columns(2)
                        
                        if result.get("dashboard_path"):
                            with col1:
                                st.markdown("### Dashboard")
                                dashboard_url = f"{FASTAPI_URL}/download-dashboard?path={result['dashboard_path']}"
                                st.markdown(f"[Download Dashboard]({dashboard_url})")
                        
                        if result.get("report_path"):
                            with col2:
                                st.markdown("### Full Report")
                                report_url = f"{FASTAPI_URL}/download-report?path={result['report_path']}"
                                st.markdown(f"[Download Report]({report_url})")
                        
                        # Display metadata
                        st.markdown("---")
                        st.caption(f"Analysis completed at: {result['timestamp']}")
                    
                    else:
                        st.error(result["message"])
                
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the analysis service: {str(e)}")
    
    # Display history section
    if "last_result" in st.session_state:
        with st.expander("Last Analysis Details"):
            st.json(st.session_state.last_result)
    
    # Add documentation section
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        **Examples of effective prompts:**
        - "Analyze earthquakes in Japan from 2010 to 2020"
        - "Show me flood data for Bangladesh"
        - "Generate a report on wildfires in California in 2018"
        
        **Tips for best results:**
        1. Always include the disaster type and location
        2. Specify a year or date range when relevant
        3. Be as specific as possible about what you want analyzed
        
        **Output includes:**
        - Statistical analysis of disaster events
        - Visualizations of patterns and trends
        - Comprehensive PDF report with findings
        """)

if __name__ == "__main__":
    main()