from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import Dict, List, Optional
import base64
import os
import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

# Load credentials
load_dotenv()

class ReportSynthesisAgent:
    def __init__(
        self, 
        input_data: Dict, 
        data_stats: Dict, 
        analysis: Dict, 
        dashboard_path: str,
        forecast_data: Optional[Dict] = None
    ):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.metadata = {
            'title': f"Comprehensive Analysis of {input_data['disaster_type'].title()} in {input_data['country'].title()}",
            'date': datetime.datetime.now().strftime('%B %Y'),
            'timeframe': f"{data_stats['years']['min']}-{data_stats['years']['max']}",
            'disaster_type': input_data['disaster_type'].title(),
            'country': input_data['country'].title()
        }
        self.data = {
            'stats': data_stats,
            'analysis': analysis,
            'dashboard': self._process_dashboard(dashboard_path),
            'forecast': self._process_forecast_data(forecast_data) if forecast_data else None
        }

    def _process_dashboard(self, path: str) -> str:
        """Load dashboard HTML content"""
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return ""

    # In ReportSynthesisAgent._process_forecast_data
    def _process_forecast_data(self, forecast_data: Dict) -> Dict:
        """Process and enhance forecast data for reporting"""
        processed = forecast_data.copy()
        
        # Format forecast table data
        if 'forecast_table' in processed.get('data', {}):
            for entry in processed['data']['forecast_table']:
                # Handle both formatted_affected and formatted_damage cases
                if 'formatted_affected' in entry:
                    entry['formatted_value'] = entry['formatted_affected']
                elif 'formatted_damage' in entry:
                    entry['formatted_value'] = entry['formatted_damage']
                else:
                    entry['formatted_value'] = "N/A"
        
        # Extract metric name from chart names
        if 'charts' in processed:
            # Get first chart name safely
            chart_names = list(processed['charts'].values())
            chart_name = chart_names[0] if chart_names else ""
            
            if isinstance(chart_name, str):  # Ensure it's a string before splitting
                metric_parts = chart_name.split('_')
                if len(metric_parts) > 2:
                    processed['metric'] = " ".join(metric_parts[1:-1]).title()
                    processed['metric_key'] = "_".join(metric_parts[1:-1]).lower()
                else:
                    processed['metric'] = "Impact Metrics"
                    processed['metric_key'] = "impact_metrics"
            else:
                processed['metric'] = "Impact Metrics"
                processed['metric_key'] = "impact_metrics"
        
        return processed

    def _generate_section(self, template: str, context: Dict) -> str:
        """Generate content for a report section using LLM"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({**self.metadata, **context})

    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        sections = []
        
        # 1. Executive Summary
        exec_summary_context = {
            'total_events': self.data['stats'].get('total_events', 'N/A'),
            'total_deaths': self.data['stats'].get('total_deaths', 'N/A'),
            'total_affected': self.data['stats'].get('total_affected', 'N/A'),
            'avg_deaths': self.data['analysis'].get('avg_deaths_per_event', 'N/A'),
            'avg_affected': self.data['analysis'].get('avg_affected_per_event', 'N/A')
        }
        
        exec_summary_text = """
        Create a 1-page executive summary for a disaster report about {disaster_type} in {country} 
        covering {timeframe}. Key statistics:
        - Total events: {total_events}
        - Total deaths: {total_deaths}
        - Total affected: {total_affected}
        - Average deaths per event: {avg_deaths}
        - Average affected per event: {avg_affected}
        """
        
        if self.data.get('forecast'):
            forecast = self.data['forecast']
            # Get the last and first forecast entries with formatted_value
            forecast_entries = forecast['data']['forecast_table']
            try:
                current_value = forecast_entries[-1]['formatted_value'] if forecast_entries else "N/A"
                forecast_value = forecast_entries[0]['formatted_value'] if forecast_entries else "N/A"
                growth_rate = self._calculate_growth_rate(forecast['data']['forecast_table'])
                
                exec_summary_context.update({
                    'forecast_metric': forecast.get('metric', 'impact metrics'),
                    'forecast_period': f"{forecast['data']['forecast_years'][0]}-{forecast['data']['forecast_years'][-1]}" if 'forecast_years' in forecast['data'] else "N/A",
                    'current_value': current_value,
                    'forecast_value': forecast_value,
                    'growth_rate': growth_rate
                })
                exec_summary_text += """
                \nForecast Data:
                - Metric: {forecast_metric}
                - Forecast period: {forecast_period}
                - Current value: {current_value}
                - Projected value: {forecast_value}
                - Annual growth rate: {growth_rate}%
                """
            except (KeyError, IndexError) as e:
                print(f"Error processing forecast data: {str(e)}")
        
        sections.append(self._generate_section(exec_summary_text, exec_summary_context))

        # 2. Methodology
        method_text = """
        Write a methodology section detailing:
        1. Data sources (Snowflake database)
        2. Analysis techniques (statistical, geospatial)
        3. Time period covered: {timeframe}
        """ + ("4. Forecasting method: Linear regression on historical trends\n" if self.data.get('forecast') else "") + """
        5. Limitations
        """
        sections.append(self._generate_section(method_text, {}))

        # 3. Detailed Analysis
        analysis_parts = [
            self._generate_section("""
            Analyze temporal patterns in the data:
            - Yearly distribution of events
            - Trends in mortality rates
            - Changes in affected populations
            """, {'stats': str(self.data['stats'])}),
            
            self._generate_section("""
            Analyze geographic distribution:
            - Hotspot locations: {hotspots}
            - Regional vulnerabilities
            """, {'hotspots': self.data['analysis']['event_patterns'].get('common_locations', [])})
        ]
        
        if self.data.get('forecast'):
            forecast = self.data['forecast']
            try:
                forecast_table = "\n".join(
                    f"- {entry['START_YEAR']}: {entry.get('formatted_value', 'N/A')}"
                    for entry in forecast['data']['forecast_table']
                )
                
                analysis_parts.append(self._generate_section("""
                Forecast Analysis for {metric}
                
                Historical and Projected Values:
                {forecast_table}
                
                Key Insights:
                - Trend Analysis: {trend_analysis}
                - Growth Patterns: {growth_phases}
                - Risk Implications: {risk_implications}
                
                Conclusion:
                {conclusion}
                """, {
                    'metric': forecast.get('metric', 'Impact Metrics'),
                    'forecast_table': forecast_table,
                    'trend_analysis': forecast['analysis'].get('trend_analysis', 'No trend analysis available'),
                    'growth_phases': forecast['analysis'].get('growth_decline_phases', 'No growth analysis available'),
                    'risk_implications': forecast['analysis'].get('risk_implications', 'No risk analysis available'),
                    'conclusion': forecast['analysis'].get('conclusion', 'No conclusion available')
                }))
            except KeyError as e:
                print(f"Error processing forecast analysis: {str(e)}")
                    
        sections.extend(analysis_parts)

        # 4. Recommendations
        rec_text = """
        Generate policy recommendations based on:
        - Total damage: {total_damage}
        - Most affected regions: {regions}
        """ + ("- Forecast insights: {forecast_insights}\n" if self.data.get('forecast') else "") + """
        Include both mitigation and adaptation strategies.
        """

        try:
            total_damage = f"${float(self.data['stats'].get('total_damage_usd', 0)):,.2f}"
        except (TypeError, ValueError):
            total_damage = "N/A"

        rec_context = {
            'total_damage': total_damage,
            'regions': self.data['analysis']['event_patterns'].get('common_locations', [])
        }

        if self.data.get('forecast'):
            rec_context['forecast_insights'] = self.data['forecast']['analysis'].get(
                'risk_implications', 
                'No forecast risk insights available'
            )

        sections.append(self._generate_section(rec_text, rec_context))

        # Compile into final report
        report = self._compile_report(sections)
        return self._save_report(report)

    def _calculate_growth_rate(self, forecast_table: List[Dict]) -> float:
        """Calculate average annual growth rate from forecast data"""
        if len(forecast_table) < 2:
            return 0.0
            
        first_year = forecast_table[0]
        last_year = forecast_table[-1]
        years = last_year['START_YEAR'] - first_year['START_YEAR']
        if years == 0:
            return 0.0
            
        # Handle both TOTAL_AFFECTED and TOTAL_DAMAGE cases
        try:
            if 'TOTAL_AFFECTED' in first_year:
                first_value = float(first_year['TOTAL_AFFECTED'])
                last_value = float(last_year['TOTAL_AFFECTED'])
            elif 'TOTAL_DAMAGE_000_USD' in first_year:
                first_value = float(first_year['TOTAL_DAMAGE_000_USD'])
                last_value = float(last_year['TOTAL_DAMAGE_000_USD'])
            else:
                return 0.0
                
            if first_value == 0:  # Prevent division by zero
                return 0.0
                
            growth_rate = ((last_value / first_value) ** (1/years) - 1)
            return round(growth_rate * 100, 1)
        except (TypeError, ValueError, KeyError):
            return 0.0

    def _compile_report(self, sections: List[str]) -> str:
        """Combine all sections into a formatted report with forecast visuals"""
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')
        
        forecast_visuals = None
        if self.data.get('forecast'):
            try:
                forecast_visuals = {
                    'historical': self._encode_image(self.data['forecast']['charts']['historical']),
                    'forecast': self._encode_image(self.data['forecast']['charts']['forecast']),
                    'growth': self._encode_image(self.data['forecast']['charts']['growth']),
                    'metric': self.data['forecast'].get('metric', 'Impact Metrics')
                }
            except (KeyError, TypeError):
                forecast_visuals = None
        
        return template.render(
            metadata=self.metadata,
            data=self.data,
            sections=sections,
            dashboard=self.data['dashboard'],
            forecast_visuals=forecast_visuals,
            forecast_data=self.data.get('forecast')
        )

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 for HTML embedding"""
        if not os.path.exists(image_path):
            return None
            
        with open(image_path, "rb") as image_file:
            return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

    def _save_report(self, content: str) -> str:
        """Save the report to HTML file"""
        os.makedirs('reports', exist_ok=True)
        report_path = os.path.join('reports', f"{self.metadata['title'].replace(' ', '_')}.html")
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        return report_path


# Example Usage
if __name__ == "__main__":
    input_data = {'disaster_type': 'flood', 'country': 'Mexico'}
    data_stats = {
        'total_events': 115,
        'total_deaths': 3591,
        'total_affected': 11688233,
        'total_damage_usd': 1250000000,
        'years': {'min': 2000, 'max': 2024},
        'sample_events': [{'LOCATION': 'Rio de Janeiro', 'START_YEAR': 2000}]
    }
    analysis = {
        'avg_deaths_per_event': 31.23,
        'avg_affected_per_event': 101636.81,
        'event_patterns': {
            'common_locations': ['Rio de Janeiro', 'São Paulo']
        }
    }
    dashboard_path = "dashboard.html"
    
    forecast_data = {
    "data": {
        "metric": "TOTAL_AFFECTED",
        "filters": ["DISASTER_TYPE = 'Flood'"],
        "forecast_table": [
            {"START_YEAR": 2000, "TOTAL_AFFECTED": 442010.0, "formatted_affected": "442,010"},
            {"START_YEAR": 2001, "TOTAL_AFFECTED": 760568.0, "formatted_affected": "760,568"},
            {"START_YEAR": 2002, "TOTAL_AFFECTED": 987422.0, "formatted_affected": "987,422"},
            {"START_YEAR": 2003, "TOTAL_AFFECTED": 837462.0, "formatted_affected": "837,462"},
            {"START_YEAR": 2004, "TOTAL_AFFECTED": 641062.0, "formatted_affected": "641,062"},
            {"START_YEAR": 2005, "TOTAL_AFFECTED": 1031306.0, "formatted_affected": "1,031,306"},
            {"START_YEAR": 2006, "TOTAL_AFFECTED": 814647.0, "formatted_affected": "814,647"},
            {"START_YEAR": 2007, "TOTAL_AFFECTED": 5878898.0, "formatted_affected": "5,878,898"},
            {"START_YEAR": 2008, "TOTAL_AFFECTED": 15848647.0, "formatted_affected": "15,848,647"},
            {"START_YEAR": 2009, "TOTAL_AFFECTED": 2454724.0, "formatted_affected": "2,454,724"},
            {"START_YEAR": 2010, "TOTAL_AFFECTED": 4651572.0, "formatted_affected": "4,651,572"},
            {"START_YEAR": 2011, "TOTAL_AFFECTED": 5254513.0, "formatted_affected": "5,254,513"},
            {"START_YEAR": 2012, "TOTAL_AFFECTED": 770043.0, "formatted_affected": "770,043"},
            {"START_YEAR": 2013, "TOTAL_AFFECTED": 1622046.0, "formatted_affected": "1,622,046"},
            {"START_YEAR": 2014, "TOTAL_AFFECTED": 1531687.0, "formatted_affected": "1,531,687"},
            {"START_YEAR": 2015, "TOTAL_AFFECTED": 1520224.0, "formatted_affected": "1,520,224"},
            {"START_YEAR": 2016, "TOTAL_AFFECTED": 3707826.0, "formatted_affected": "3,707,826"},
            {"START_YEAR": 2017, "TOTAL_AFFECTED": 2567331.0, "formatted_affected": "2,567,331"},
            {"START_YEAR": 2018, "TOTAL_AFFECTED": 586544.0, "formatted_affected": "586,544"},
            {"START_YEAR": 2019, "TOTAL_AFFECTED": 1064920.0, "formatted_affected": "1,064,920"},
            {"START_YEAR": 2020, "TOTAL_AFFECTED": 285075.0, "formatted_affected": "285,075"},
            {"START_YEAR": 2021, "TOTAL_AFFECTED": 3760254.0, "formatted_affected": "3,760,254"},
            {"START_YEAR": 2022, "TOTAL_AFFECTED": 4064252.0, "formatted_affected": "4,064,252"},
            {"START_YEAR": 2023, "TOTAL_AFFECTED": 6258723.0, "formatted_affected": "6,258,723"},
            {"START_YEAR": 2024, "TOTAL_AFFECTED": 5126560.0, "formatted_affected": "5,126,560"},
            {"START_YEAR": 2025, "TOTAL_AFFECTED": 57450.0, "formatted_affected": "57,450"},
            {"START_YEAR": 2026, "TOTAL_AFFECTED": 3310747.566153839, "formatted_affected": "3,310,748"},
            {"START_YEAR": 2027, "TOTAL_AFFECTED": 3349362.012649566, "formatted_affected": "3,349,362"},
            {"START_YEAR": 2028, "TOTAL_AFFECTED": 3387976.4591452926, "formatted_affected": "3,387,976"},
            {"START_YEAR": 2029, "TOTAL_AFFECTED": 3426590.9056410193, "formatted_affected": "3,426,591"},
            {"START_YEAR": 2030, "TOTAL_AFFECTED": 3465205.352136746, "formatted_affected": "3,465,205"}
        ],
        "forecast_years": [2026, 2027, 2028, 2029, 2030]
    },
    "charts": {
        "forecast": "forecast_total_affected.png",
        "historical": "historical_total_affected_bar.png",
        "growth": "growth_total_affected.png"
    },
    "analysis": {
        "trend_analysis": "The data presents the total number of affected individuals each year from 2000 to 2030... (truncated for brevity)",
        "growth_decline_phases": "The biggest spike in total affected is seen from 2007 to 2008... (truncated)",
        "forecast_interpretation": "The data shows that the number of total affected individuals fluctuates... (truncated)",
        "risk_implications": "The data suggests that the number of total affected individuals has generally been increasing... (truncated)",
        "conclusion": "Insights and Patterns:\n1. The total number of affected people has seen a general increase... (truncated)"
    },
    "error": None
 }
    report_agent = ReportSynthesisAgent(
        input_data, 
        data_stats, 
        analysis, 
        dashboard_path,
        forecast_data=forecast_data
    )
    report_path = report_agent.generate_report()
    print(f"Report generated at: {report_path}")
    