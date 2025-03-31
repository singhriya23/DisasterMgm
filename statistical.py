import pandas as pd
from typing import Dict, List
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

class DisasterStatsAnalyzer:
    def __init__(self, stats_data: Dict):
        """
        Initialize with the stats dictionary from Data Retrieval Agent.
        """
        self.stats = stats_data
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert sample_events into a DataFrame for analysis"""
        if not self.stats.get('sample_events'):
            return pd.DataFrame()
            
        df = pd.DataFrame(self.stats['sample_events'])
        
        # Add metrics from the main stats
        df['TOTAL_DEATHS'] = None  # Will be filled from your actual data
        df['NO_AFFECTED'] = None
        df['TOTAL_DAMAGE_USD'] = None
        
        # Extract first latitude/longitude from location text (simplified)
        df['LATITUDE'] = df['LOCATION'].apply(self._extract_coords).str[0]
        df['LONGITUDE'] = df['LOCATION'].apply(self._extract_coords).str[1]
        
        return df

    def _extract_coords(self, text: str) -> List[float]:
        """Naive coordinate extractor (replace with your actual geocoding)"""
        # This is a placeholder - implement proper geocoding for production
        brazil_coords = [-15.78, -47.93]  # Approximate Brazil centroid
        return brazil_coords if pd.notnull(text) else [np.nan, np.nan]

    def calculate_core_stats(self) -> Dict:
        """
        Enhance the existing stats with derived metrics.
        """
        if not self.stats:
            return {"error": "No statistics available"}
            
        enhanced = self.stats.copy()
        
        # Calculate averages
        enhanced['avg_deaths_per_event'] = enhanced['total_deaths'] / enhanced['total_events']
        enhanced['avg_affected_per_event'] = enhanced['total_affected'] / enhanced['total_events']
        enhanced['avg_damage_per_event_usd'] = enhanced['total_damage_usd'] / enhanced['total_events']
        
        # Add year range
        enhanced['year_range'] = enhanced['years']['max'] - enhanced['years']['min']
        
        return enhanced

    def analyze_event_patterns(self) -> Dict:
        """
        Analyze temporal and spatial patterns from sample events.
        """
        if self.df.empty:
            return {}
            
        patterns = {
            "events_per_year": self._count_events_by_year(),
            "common_locations": self._find_common_locations(),
            "geospatial_clusters": self._detect_clusters()
        }
        
        return patterns

    def _count_events_by_year(self) -> Dict:
        """Count events by year from sample data"""
        return self.df['START_YEAR'].value_counts().to_dict()

    def _find_common_locations(self, top_n: int = 3) -> List:
        """Identify frequently mentioned locations"""
        locations = self.df['LOCATION'].str.split(',').explode()
        return locations.str.strip().value_counts().head(top_n).index.tolist()

    def _detect_clusters(self, eps_km: float = 300) -> Dict:
        """Detect geographic clusters from coordinates"""
        if {'LATITUDE', 'LONGITUDE'}.issubset(self.df.columns):
            coords = self.df[['LATITUDE', 'LONGITUDE']].dropna()
            if len(coords) > 2:
                # Convert km to approximate degrees
                eps_deg = eps_km / 111
                
                db = DBSCAN(eps=eps_deg, min_samples=2).fit(coords)
                self.df['cluster'] = db.labels_
                
                clusters = []
                for label in set(db.labels_):
                    if label != -1:
                        cluster_points = coords[db.labels_ == label]
                        centroid = cluster_points.mean().values
                        clusters.append({
                            "point_count": len(cluster_points),
                            "centroid": {"lat": centroid[0], "lon": centroid[1]},
                            "radius_km": self._calculate_cluster_radius(cluster_points, centroid)
                        })
                
                return {"detected_clusters": clusters}
        return {"error": "Insufficient geospatial data"}

    def _calculate_cluster_radius(self, points: pd.DataFrame, centroid: np.array) -> float:
        """Calculate approximate cluster radius in kilometers"""
        max_distance = max(
            great_circle(centroid, point).km
            for point in points.values
        )
        return round(max_distance, 2)

# Example Usage with your actual data
if __name__ == "__main__":
    # Your exact stats output from Data Retrieval Agent
    stats_from_retriever = {
        'total_events': 115,
        'countries': ['Brazil'],
        'years': {'min': 2000, 'max': 2024},
        'disaster_types': ['Flood'],
        'total_deaths': 3591,
        'total_affected': 11688233,
        'total_damage_usd': 15414070000.0,
        'sample_events': [
            {'EVENT_NAME': None, 'START_YEAR': 2000, 'LOCATION': 'Rio de Janeiro city...'},
            {'EVENT_NAME': None, 'START_YEAR': 2000, 'LOCATION': 'Recife city...'},
            {'EVENT_NAME': None, 'START_YEAR': 2000, 'LOCATION': 'Rio Grande Do Sul province'},
            {'EVENT_NAME': None, 'START_YEAR': 2000, 'LOCATION': 'Belo Horizonte district...'},
            {'EVENT_NAME': None, 'START_YEAR': 2001, 'LOCATION': 'Cuiaba district...'}
        ]
    }
    
    analyzer = DisasterStatsAnalyzer(stats_from_retriever)
    
    print("=== ENHANCED STATISTICS ===")
    print(analyzer.calculate_core_stats())
    
    print("\n=== EVENT PATTERNS ===")
    print(analyzer.analyze_event_patterns())