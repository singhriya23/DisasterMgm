import re
from typing import Tuple, Optional, List

# Predefined constants (from your input)
DISASTER_TYPES = {
    "volcanic activity", "flood", "storm", "earthquake", "wildfire", "epidemic",
    "mass movement (wet)", "infestation", "extreme temperature", "drought", 
    "mass movement (dry)"
}

COUNTRIES = {
    "guatemala", "brazil", "united states of america", "colombia", "argentina",
    "peru", "bolivia (plurinational state of)", "ecuador", "french guiana", 
    "uruguay", "mexico", "chile", "nicaragua", "costa rica", "paraguay", 
    "panama", "belize", "cuba", "jamaica", "puerto rico", 
    "venezuela (bolivarian republic of)", "haiti", "dominican republic",
    "barbados", "grenada", "saint vincent and the grenadines", "bermuda",
    "saint lucia", "united states virgin islands", "martinique", 
    "turks and caicos islands", "antigua and barbuda", "british virgin islands",
    "canada", "el salvador", "cayman islands", "bahamas", "honduras",
    "trinidad and tobago", "dominica", "guadeloupe", "guyana", "suriname",
    "saint kitts and nevis", "anguilla", "saint barthélemy", 
    "saint martin (french part)", "sint maarten (dutch part)", "montserrat"
}

VALID_YEARS = {str(year) for year in range(2000, 2026)}  # 2000-2025

def parse_disaster_prompt(prompt: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Extracts (disaster_type, country, year) from prompts.
    Returns (None, None, None) for invalid/missing fields.
    """
    normalized = re.sub(r'\s+', ' ', prompt.lower()).strip()
    
    # Initialize with None
    disaster_type, country, year = None, None, None
    
    # Extract disaster type (must be mentioned explicitly)
    for disaster in DISASTER_TYPES:
        if re.search(r'\b' + re.escape(disaster) + r'\b', normalized):
            disaster_type = disaster
            break
    
    # Extract country (must match exactly)
    for c in COUNTRIES:
        if re.search(r'\b' + re.escape(c) + r'\b', normalized):
            country = c
            break
    
    # Extract year (2000-2025)
    year_match = re.search(r'\b(20[0-2][0-9])\b', normalized)
    if year_match and year_match.group(1) in VALID_YEARS:
        year = int(year_match.group(1))
    
    return (disaster_type, country, year)
'''
# Example Usage
prompts = [
    "Create a report on flood in Brazil",
    "Do an analysis on earthquake in Mexico 2015",
    "Wildfire damage report for Canada",
    "Invalid prompt with no valid data"
]

for prompt in prompts:
    disaster, country, year = parse_disaster_prompt(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"→ Disaster: {disaster or 'N/A'}")
    print(f"→ Country: {country or 'N/A'}")
    print(f"→ Year: {year or 'N/A'}\n")'
    ''
    '''