"""
Configuration and constants for the Solar Rooftop application
"""

# Performance configuration
import numpy as np
np.random.seed(42)  # For reproducible results

# Default values
DEFAULT_ELECTRICITY_PRICE = 1.5

# SCB API configuration
SCB_BASE_URL = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START"

# Earth Engine configuration
EE_PROJECT_ID = 'bigquerylearning-476613'

# App configuration
APP_TITLE = "🔆 Dynamic Solar Rooftop Leads from GEE"
PAGE_CONFIG = {
    "page_title": "Dynamic Solar Leads", 
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# Filter default values
FILTER_DEFAULTS = {
    "min_score": 30,  # Changed from 200 to 30 (solar scores are 0-100)
    "income_range": (300, 1500),
    "usage_range": (10000, 25000),
    "household_sizes": [1, 2, 3, 4, 5],
    "owner_age_range": (25, 80),
    "roof_age_range": (0, 50),
    "num_to_geocode": 75  # Default number of addresses to geocode
}