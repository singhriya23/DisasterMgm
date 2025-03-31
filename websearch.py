from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Dict
import pprint

class WebSearchAgent:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()  # No API keys needed
        
    def get_current_news(self, disaster_type: str, country: str) -> List[Dict]:
        """Get recent news snippets (last 6 months)"""
        query = f"{disaster_type} in {country} after:2024-01-01"
        results = self.search.run(query)
        
        # Parse results
        formatted = []
        for result in results.split('\n\n')[:3]:  # Top 3 results
            if not result.strip():
                continue
                
            source = None
            if 'http' in result:
                source = result[result.index('http'):].split(' ')[0]
                
            formatted.append({
                'title': result.split(' - ')[0] if ' - ' in result else result[:60].strip(),
                'snippet': result,
                'source': source
            })
        return formatted

    def get_prevention_updates(self, disaster_type: str) -> str:
        """Get latest prevention techniques"""
        return self.search.run(
            f"latest prevention measures for {disaster_type} 2024"
        )

def test_searches():
    agent = WebSearchAgent()
    
    print("🌎 Testing disaster searches:")
    test_cases = [("flood", "mexico"), ("earthquake", "japan")]
    
    for disaster, country in test_cases:
        print(f"\n🔍 {disaster.title()} in {country.title()}:")
        
        news = agent.get_current_news(disaster, country)
        print("\n📰 News Results:")
        pprint.pprint(news)
        
        prevention = agent.get_prevention_updates(disaster)
        print("\n🛡️ Prevention Updates:")
        print(prevention[:300] + ("..." if len(prevention) > 300 else ""))

if __name__ == "__main__":
    test_searches()