from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import Dict, List
import base64
import os
import datetime
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

# Load credentials
load_dotenv()

class ReportSynthesisAgent:
    def __init__(self, input_data: Dict, data_stats: Dict, analysis: Dict, dashboard_path: str):
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
            'dashboard': self._process_dashboard(dashboard_path)  # Changed from visuals to dashboard
        }

    def _process_dashboard(self, path: str) -> str:
        """Load dashboard HTML content"""
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        return ""

    def _generate_section(self, template: str, context: Dict) -> str:
        """Generate content for a report section using LLM"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({**self.metadata, **context})

    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        sections = []
        
        # 1. Executive Summary
        sections.append(self._generate_section("""
        Create a 1-page executive summary for a disaster report about {disaster_type} in {country} 
        covering {timeframe}. Key statistics:
        - Total events: {total_events}
        - Total deaths: {total_deaths}
        - Total affected: {total_affected}
        - Average deaths per event: {avg_deaths}
        - Average affected per event: {avg_affected}
        """, {
            'total_events': self.data['stats']['total_events'],
            'total_deaths': self.data['stats']['total_deaths'],
            'total_affected': self.data['stats']['total_affected'],
            'avg_deaths': self.data['analysis']['avg_deaths_per_event'],
            'avg_affected': self.data['analysis']['avg_affected_per_event']
        }))

        # 2. Methodology
        sections.append(self._generate_section("""
        Write a 2-page methodology section detailing:
        1. Data sources (Snowflake database)
        2. Analysis techniques (statistical, geospatial)
        3. Time period covered: {timeframe}
        4. Limitations of the dataset
        """, {}))

        # 3. Detailed Analysis
        analysis_parts = [
            self._generate_section("""
            Analyze temporal patterns in the data:
            - Yearly distribution of events
            - Trends in mortality rates
            - Changes in affected populations
            Key stats: {stats}
            """, {'stats': str(self.data['stats'])}),
            
            self._generate_section("""
            Analyze geographic distribution:
            - Hotspot locations: {hotspots}
            - Regional vulnerabilities
            - Urban vs rural impacts
            """, {'hotspots': self.data['analysis']['event_patterns']['common_locations']})
        ]
        sections.extend(analysis_parts)

        # 4. Recommendations
        sections.append(self._generate_section("""
        Generate policy recommendations based on:
        - Total damage: ${total_damage:,.2f}
        - Most affected regions: {regions}
        - Key vulnerabilities identified
        Include both short-term and long-term strategies.
        """, {
            'total_damage': self.data['stats'].get('total_damage_usd', 0),
            'regions': self.data['analysis']['event_patterns']['common_locations']
        }))

        # Compile into final report
        report = self._compile_report(sections)
        return self._save_report(report)

    def _compile_report(self, sections: List[str]) -> str:
        """Combine all sections into a formatted report"""
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')
        
        return template.render(
            metadata=self.metadata,
            data=self.data,
            sections=sections,
            dashboard=self.data['dashboard']  # Changed from visuals to dashboard
        )

    def _save_report(self, content: str) -> str:
        """Save the report to HTML file"""
        os.makedirs('reports', exist_ok=True)
        report_path = os.path.join('reports', f"{self.metadata['title'].replace(' ', '_')}.html")
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        return report_path

# Example Usage
if __name__ == "__main__":
    # Example data
    input_data = {'disaster_type': 'flood', 'country': 'Brazil'}
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
    dashboard_path = "/Users/riyasingh/Desktop/BIGDATA_ASSIGNMENT/DisasterMgm/dashboard.html"
    
    # Generate report
    report_agent = ReportSynthesisAgent(input_data, data_stats, analysis, dashboard_path)
    report_path = report_agent.generate_report()
    print(f"Report generated at: {report_path}")