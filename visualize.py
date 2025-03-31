from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import os

class DisasterDashboard:
    def __init__(self, stats_data: Dict):
        self.stats = stats_data or {}
        self.df = pd.DataFrame(self.stats.get('sample_events', []))
        
    def create_dashboard(self, output_file: str = "dashboard.html"):
        """Generate a single HTML file with all visualizations"""
        # 1. Create Plotly figures
        timeline_fig = self._create_timeline()
        impact_fig = self._create_impact_chart()
        
        # 2. Create static map image (for cloud compatibility)
        map_img = self._create_static_map()
        
        # 3. Combine into dashboard
        dashboard = self._build_html(
            timeline=timeline_fig.to_html(full_html=False),
            impact=impact_fig.to_html(full_html=False),
            map_img=map_img
        )
        
        with open(output_file, 'w') as f:
            f.write(dashboard)
        return os.path.abspath(output_file)

    def _create_timeline(self):
        yearly_counts = self.stats.get('events_per_year', {})
        return px.line(
            x=list(yearly_counts.keys()),
            y=list(yearly_counts.values()),
            title="Event Frequency Timeline"
        )

    def _create_impact_chart(self):
        metrics = {
            'Deaths': self.stats.get('total_deaths', 0),
            'Affected': self.stats.get('total_affected', 0),
            'Damage': self.stats.get('total_damage_usd', 0)
        }
        return px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            title="Impact Comparison"
        )

    def _create_static_map(self) -> str:
        """Convert map to base64 for embedded HTML"""
        if {'LATITUDE', 'LONGITUDE'}.issubset(self.df.columns):
            plt.figure(figsize=(10,6))
            plt.scatter(
                x=self.df['LONGITUDE'],
                y=self.df['LATITUDE'],
                c='red', alpha=0.5
            )
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        return ""

    def _build_html(self, timeline: str, impact: str, map_img: str) -> str:
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Disaster Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .dashboard {{ 
            font-family: Arial; 
            max-width: 1200px; 
            margin: auto;
        }}
        .panel {{ 
            background: white; 
            border-radius: 10px; 
            padding: 15px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>Disaster Analysis Report</h1>
        
        <div class="panel">
            <h2>1. Event Timeline</h2>
            <div id="timeline">{timeline}</div>
        </div>
        
        <div class="panel">
            <h2>2. Impact Comparison</h2>
            <div id="impact">{impact}</div>
        </div>
        
        <div class="panel">
            <h2>3. Event Locations</h2>
            <img src="data:image/png;base64,{map_img}" 
                 style="max-width:100%;" 
                 onerror="this.style.display='none'">
            {'' if map_img else '<p>No geographic data available</p>'}
        </div>
    </div>
</body>
</html>
        """

# Example usage
if __name__ == "__main__":
    test_data = {
        "total_events": 115,
        "events_per_year": {2020: 45, 2021: 70},
        "total_deaths": 3591,
        "total_affected": 11688233,
        "sample_events": [
            {"LATITUDE": -22.9, "LONGITUDE": -43.2},
            {"LATITUDE": -23.5, "LONGITUDE": -46.6}
        ]
    }
    
    dashboard = DisasterDashboard(test_data)
    print("Dashboard created at:", dashboard.create_dashboard())