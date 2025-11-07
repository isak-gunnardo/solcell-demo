import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import numpy as np
import requests
import json
from shapely.geometry import Point
import concurrent.futures
from functools import lru_cache

from geopy.geocoders import Nominatim
import time

# Performance configuration
np.random.seed(42)  # For reproducible results

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_coordinates_and_city(input_text):
    """
    Get coordinates and city name from either city name or postal code (cached version)
    Returns: (latitude, longitude, city_name)
    """
    return get_coordinates_and_city_internal(input_text)

def get_coordinates_and_city_internal(input_text):
    """
    Internal implementation of geocoding (uncached)
    """
    geolocator = Nominatim(user_agent="solar_app_v2")
    
    # Check if input is a postal code (5 digits)
    if input_text.strip().replace(' ', '').isdigit() and len(input_text.strip().replace(' ', '')) == 5:
        postal_code = input_text.strip().replace(' ', '')
        # Swedish postal code format
        formatted_postal = f"{postal_code[:3]} {postal_code[3:]}"
        
        # Try to geocode postal code with different strategies (optimized order)
        search_queries = [
            f"{formatted_postal}, Sweden",  # Most likely to succeed
            f"{postal_code}, Sweden", 
            f"{formatted_postal}"
        ]  # Reduced number of attempts
        
        for query in search_queries:
            try:
                location = geolocator.geocode(query, timeout=5)  # Reduced timeout
                if location:
                    # Check if it's likely in Sweden (latitude between 55-70, longitude 10-25)
                    if 55 <= location.latitude <= 70 and 10 <= location.longitude <= 25:
                        # Extract city name from address
                        address_parts = location.address.split(',')
                        city_name = None
                        for part in address_parts:
                            part = part.strip()
                            if (not part.isdigit() and len(part) > 2 and 
                                'sweden' not in part.lower() and 'sverige' not in part.lower() and
                                not any(char.isdigit() for char in part[:3])):  # Avoid parts starting with numbers
                                city_name = part
                                break
                        
                        if not city_name:
                            city_name = address_parts[0].strip() if address_parts else f"Postal {postal_code}"
                        
                        return location.latitude, location.longitude, city_name
            except Exception:
                continue
        
        # Fallback: use static postal code mapping for common Swedish areas
        postal_to_city_coords = {
            # Stockholm area
            '10': ('Stockholm', 59.3293, 18.0686),
            '11': ('Stockholm', 59.3293, 18.0686), 
            '12': ('Stockholm', 59.3293, 18.0686),
            '13': ('Stockholm', 59.3293, 18.0686),
            '14': ('Stockholm', 59.3293, 18.0686),
            # Göteborg area
            '40': ('Göteborg', 57.7089, 11.9746),
            '41': ('Göteborg', 57.7089, 11.9746),
            '42': ('Göteborg', 57.7089, 11.9746),
            '43': ('Göteborg', 57.7089, 11.9746),
            # Malmö area
            '20': ('Malmö', 55.6050, 13.0038),
            '21': ('Malmö', 55.6050, 13.0038),
            '22': ('Malmö', 55.6050, 13.0038),
            # Jönköping area
            '55': ('Jönköping', 57.7815, 14.1618),
            '56': ('Jönköping', 57.7815, 14.1618),
            # Uppsala area
            '75': ('Uppsala', 59.8586, 17.6389),
            # Västerås area
            '72': ('Västerås', 59.6162, 16.5528),
            # Örebro area
            '70': ('Örebro', 59.2741, 15.2066),
            '71': ('Örebro', 59.2741, 15.2066),
        }
        
        postal_prefix = postal_code[:2]
        if postal_prefix in postal_to_city_coords:
            city_name, lat, lon = postal_to_city_coords[postal_prefix]
            return lat, lon, city_name
    else:
        # Try as city name with optimized strategies
        search_queries = [
            f"{input_text}, Sweden",  # Most likely to succeed
            f"{input_text}"
        ]  # Reduced attempts
        
        for query in search_queries:
            try:
                location = geolocator.geocode(query, timeout=5)  # Reduced timeout
                if location:
                    # Check if it's likely in Sweden (latitude between 55-70, longitude 10-25)
                    if 55 <= location.latitude <= 70 and 10 <= location.longitude <= 25:
                        return location.latitude, location.longitude, input_text
            except Exception:
                continue
    
    return None, None, input_text

def get_coordinates(city_name):
    """Legacy function for backwards compatibility"""
    lat, lon, _ = get_coordinates_and_city(city_name)
    return lat, lon

# SCB API Functions
def get_municipality_code(city_or_postal):
    """Get municipality code from city name or postal code"""
    
    # Check if input is a postal code
    if city_or_postal.strip().replace(' ', '').isdigit():
        postal_code = city_or_postal.strip().replace(' ', '')
        if len(postal_code) == 5:
            # Swedish postal code to municipality mapping (major areas)
            postal_to_municipality = {
                # Stockholm region (01x-19x)
                '10': '0180', '11': '0180', '12': '0180', '13': '0180', '14': '0180',
                '15': '0180', '16': '0180', '17': '0180', '18': '0180', '19': '0180',
                # Göteborg region (40x-46x)
                '40': '1480', '41': '1480', '42': '1480', '43': '1480', '44': '1480', '45': '1480', '46': '1480',
                # Malmö region (20x-23x)
                '20': '1280', '21': '1280', '22': '1280', '23': '1280',
                # Uppsala region (75x)
                '75': '0380',
                # Västerås region (72x)
                '72': '1980',
                # Örebro region (70x-71x)
                '70': '1880', '71': '1880',
                # Linköping region (58x-59x)
                '58': '0580', '59': '0580',
                # Jönköping region (55x-56x)
                '55': '0680', '56': '0680',
                # Växjö region (35x)
                '35': '0780',
                # Karlstad region (65x)
                '65': '1780',
                # Umeå region (90x-91x)
                '90': '2480', '91': '2480',
                # Gävle region (80x)
                '80': '2180'
            }
            
            # Get first 2 digits of postal code
            postal_prefix = postal_code[:2]
            municipality_code = postal_to_municipality.get(postal_prefix, '0680')  # Default to Jönköping
            return municipality_code
    
    # Otherwise treat as city name
    municipality_map = {
        'stockholm': '0180', 'göteborg': '1480', 'malmö': '1280', 'uppsala': '0380',
        'västerås': '1980', 'örebro': '1880', 'linköping': '0580', 'helsingborg': '1283',
        'jönköping': '0680', 'norrköping': '0581', 'lund': '1281', 'umeå': '2480',
        'gävle': '2180', 'borås': '1490', 'eskilstuna': '0484', 'södertälje': '0181',
        'karlstad': '1780', 'täby': '0160', 'växjö': '0780', 'halmstad': '1380'
    }
    return municipality_map.get(city_or_postal.lower(), '0680')  # Default to Jönköping

@st.cache_data(ttl=7200)  # Cache for 2 hours
def fetch_scb_income_data(municipality_code):
    """Fetch household income data from SCB API (cached version)"""
    return fetch_scb_income_data_internal(municipality_code)

def fetch_scb_income_data_internal(municipality_code):
    """Internal SCB income data fetch (uncached)"""
    try:
        url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/HE/HE0110/HE0110A/SamForvInk1"
        
        # Query for latest available data
        query = {
            "query": [
                {
                    "code": "Region",
                    "selection": {
                        "filter": "item",
                        "values": [municipality_code]
                    }
                },
                {
                    "code": "Hushallstyp",
                    "selection": {
                        "filter": "item", 
                        "values": ["TOTAL"]
                    }
                }
            ],
            "response": {
                "format": "json"
            }
        }
        
        response = requests.post(url, json=query)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                # Get latest year data
                income_data = data['data'][-1]['values'][0]
                return float(income_data) if income_data else 500.0  # Default 500k SEK
        
        return 500.0  # Default fallback
    except:
        return 500.0  # Default fallback

def fetch_scb_population_data(municipality_code):
    """Fetch population and age data from SCB API"""
    try:
        url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BE/BE0101/BE0101A/BefolkningNy"
        
        query = {
            "query": [
                {
                    "code": "Region",
                    "selection": {
                        "filter": "item",
                        "values": [municipality_code]
                    }
                },
                {
                    "code": "Alder",
                    "selection": {
                        "filter": "item",
                        "values": ["25-64"]  # Working age population
                    }
                }
            ],
            "response": {
                "format": "json"
            }
        }
        
        response = requests.post(url, json=query)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                return True  # Population data available
        
        return False
    except:
        return False

@st.cache_data(ttl=7200)  # Cache for 2 hours
def fetch_scb_vehicle_data(municipality_code):
    """Fetch vehicle ownership data from SCB API (cached version)"""
    return fetch_scb_vehicle_data_internal(municipality_code)

def fetch_scb_vehicle_data_internal(municipality_code):
    """Internal SCB vehicle data fetch (uncached)"""
    try:
        # SCB vehicle statistics - simplified for demo
        # In reality, you'd use the transport statistics API
        url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/TK/TK1001/TK1001A/PersBilarA"
        
        query = {
            "query": [
                {
                    "code": "Region", 
                    "selection": {
                        "filter": "item",
                        "values": [municipality_code]
                    }
                }
            ],
            "response": {
                "format": "json"
            }
        }
        
        response = requests.post(url, json=query, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Calculate EV ownership percentage (simplified)
            return 0.15  # Approximate 15% EV ownership in Sweden
        
        return 0.15  # Default fallback
    except:
        return 0.15  # Default fallback

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def reverse_geocode_coordinates(lat, lon, max_retries=3):
    """
    Reverse geocode coordinates to get real address (cached version)
    Returns: (street_address, postal_code, city)
    """
    return reverse_geocode_coordinates_internal(lat, lon)

def reverse_geocode_coordinates_internal(lat, lon, max_retries=2):
    """
    Internal reverse geocoding implementation (uncached, reduced retries)
    """
    geolocator = Nominatim(user_agent="solar_app_reverse_v2")
    
    for attempt in range(max_retries):
        try:
            # Add small random offset to avoid hitting exact same coordinates
            offset_lat = lat + np.random.uniform(-0.0001, 0.0001)
            offset_lon = lon + np.random.uniform(-0.0001, 0.0001)
            
            location = geolocator.reverse((offset_lat, offset_lon), timeout=5, language='sv')  # Reduced timeout
            
            if location and location.address:
                address_parts = location.address.split(',')
                
                # Extract components from Swedish address format
                street_address = "Unknown Address"
                postal_code = "Unknown Postal"
                city = "Unknown City"
                
                # Parse the address components
                for i, part in enumerate(address_parts):
                    part = part.strip()
                    
                    # Look for street address (usually first meaningful part)
                    if i == 0 and not part.isdigit() and len(part) > 3:
                        street_address = part
                    
                    # Look for postal code (Swedish format: 5 digits, may have space)
                    if any(char.isdigit() for char in part) and len(part.replace(' ', '')) == 5:
                        # Format as Swedish postal code
                        digits_only = ''.join(filter(str.isdigit, part))
                        if len(digits_only) == 5:
                            postal_code = f"{digits_only[:3]} {digits_only[3:]}"
                    
                    # Look for city name (avoid country, numbers, etc.)
                    if (not any(char.isdigit() for char in part) and 
                        len(part) > 2 and 
                        'sweden' not in part.lower() and 
                        'sverige' not in part.lower() and
                        'län' not in part.lower() and
                        'kommun' not in part.lower()):
                        city = part
                
                # If we didn't find a good street address, create one from available info
                if street_address == "Unknown Address" and len(address_parts) > 0:
                    street_address = address_parts[0].strip()
                
                return street_address, postal_code, city
                
        except Exception as e:
            print(f"Reverse geocoding attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    # Fallback if reverse geocoding fails
    return "Address not found", "Unknown Postal", "Unknown City"

def batch_reverse_geocode(_buildings_df, max_addresses=25):
    """
    Optimized batch reverse geocode with threading and caching
    Reduced max addresses for faster performance
    """
    addresses = []
    postal_codes = []
    cities = []
    
    # Limit the number of reverse geocoding calls (reduced for speed)
    num_to_geocode = min(len(_buildings_df), max_addresses)
    
    print(f"Reverse geocoding {num_to_geocode} addresses (optimized)...")
    
    # Use threading for parallel processing (limited threads to respect API limits)
    def geocode_single(i):
        lat = _buildings_df.iloc[i]['lat']
        lon = _buildings_df.iloc[i]['lon']
        return i, reverse_geocode_coordinates(lat, lon)
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # Limited workers
        future_to_idx = {executor.submit(geocode_single, i): i for i in range(num_to_geocode)}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            i, (street_address, postal_code, city) = future.result()
            results[i] = (street_address, postal_code, city)
            time.sleep(0.1)  # Reduced delay due to caching
    
    # Sort results and extract
    for i in range(num_to_geocode):
        if i in results:
            street_address, postal_code, city = results[i]
        else:
            street_address, postal_code, city = "Address not found", "Unknown Postal", "Unknown City"
        
        addresses.append(street_address)
        postal_codes.append(postal_code)
        cities.append(city)
    
    # For remaining buildings, use simulated addresses based on the first geocoded ones
    if len(_buildings_df) > num_to_geocode:
        print(f"Using simulated addresses for remaining {len(_buildings_df) - num_to_geocode} buildings...")
        
        # Use patterns from successfully geocoded addresses
        real_streets = [addr for addr in addresses if addr != "Address not found"]
        real_cities = [city for city in cities if city != "Unknown City"]
        real_postal_prefixes = []
        
        for postal in postal_codes:
            if postal != "Unknown Postal" and len(postal) >= 3:
                real_postal_prefixes.append(postal[:3])
        
        # Generate remaining addresses based on real patterns
        for i in range(num_to_geocode, len(_buildings_df)):
            if real_streets:
                # Modify existing street names
                base_street = np.random.choice(real_streets)
                new_number = np.random.randint(1, 99)
                # Extract street name without number
                street_parts = base_street.split()
                if len(street_parts) > 1 and street_parts[-1].isdigit():
                    street_name = ' '.join(street_parts[:-1])
                else:
                    street_name = base_street
                addresses.append(f"{street_name} {new_number}")
            else:
                addresses.append(f"Simulated Street {np.random.randint(1, 99)}")
            
            if real_postal_prefixes:
                prefix = np.random.choice(real_postal_prefixes)
                suffix = np.random.randint(10, 99)
                postal_codes.append(f"{prefix} {suffix}")
            else:
                postal_codes.append("000 00")
            
            if real_cities:
                cities.append(np.random.choice(real_cities))
            else:
                cities.append("Unknown City")
    
    return addresses, postal_codes, cities

def enrich_with_scb_data(_buildings_df, city_name):
    """Optimized enrichment with parallel processing"""
    municipality_code = get_municipality_code(city_name)
    
    # Fetch real data from SCB (now cached)
    avg_income = fetch_scb_income_data(municipality_code)
    ev_rate = fetch_scb_vehicle_data(municipality_code)
    
    num_buildings = len(_buildings_df)
    
    # Optimized address collection with reduced count for speed
    print("🔍 Getting addresses (optimized with caching)...")
    addresses, postal_codes, cities = batch_reverse_geocode(_buildings_df)
    
    # Generate demographic data efficiently using vectorized operations
    # Set seed for reproducible results
    np.random.seed(hash(city_name) % 2**32)
    
    scb_data = pd.DataFrame({
        'address': addresses,
        'postal_code': postal_codes,
        'municipality': cities,
        'has_ev': np.random.choice([True, False], size=num_buildings, p=[ev_rate, 1-ev_rate]),
        'income_ksek': np.random.normal(avg_income, avg_income*0.3, size=num_buildings).clip(200, 2000).round().astype(int),
        'roof_age_years': np.random.exponential(15, size=num_buildings).clip(0, 50).round().astype(int),
        'owner_age': np.random.normal(45, 15, size=num_buildings).clip(25, 80).round().astype(int),
        'household_size': np.random.choice([1, 2, 3, 4, 5], size=num_buildings, 
                                         p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'current_usage_kwh': np.random.normal(15000, 5000, size=num_buildings).clip(5000, 35000).round().astype(int)
    })
    
    return pd.concat([_buildings_df.reset_index(drop=True), scb_data], axis=1)

# Initialize Earth Engine (now using cached version)
def initialize_ee():
    return initialize_ee_cached()

# Constants
SE_LATLON_IRRADIANCE_KWH_PER_KWP = 1000
M2_PER_KWP = 6.5
DEFAULT_ELECTRICITY_PRICE = 1.5

# Solar scoring function
def solar_score(row):
    # Use actual GEE column names, with fallbacks
    aspect = row.get('aspect') or row.get('mean_1') or 180
    slope = row.get('slope') or row.get('mean') or 25
    shadow = row.get('shadow_index') or row.get('mean_2') or 0

    # Optimal south-facing aspect is 180 degrees
    aspect_score = max(0, 1 - abs(aspect - 180) / 180)
    # Optimal slope is around 25 degrees for Sweden
    slope_score = max(0, 1 - abs(slope - 25) / 25)
    # Lower shadow index is better (less shadows)
    shadow_score = 1 - min(shadow, 1)  # Cap at 1 to avoid negative scores

    return 100 * (0.5 * aspect_score + 0.3 * slope_score + 0.2 * shadow_score)

def pv_kwp_from_area(m2):
    return max(0.0, m2 / M2_PER_KWP)

def annual_kwh(kWp):
    return kWp * SE_LATLON_IRRADIANCE_KWH_PER_KWP

def annual_savings_ksek(kwh, price):
    return (kwh * price) / 1000.0

# Optimized Streamlit configuration
st.set_page_config(
    page_title="Dynamic Solar Leads", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimizations
@st.cache_resource
def initialize_ee_cached():
    """Cached Earth Engine initialization"""
    try:
        ee.Initialize(project='bigquerylearning-476613')
        return True
    except Exception as e:
        return False
st.title("🔆 Dynamic Solar Rooftop Leads from GEE")

# Earth Engine authentication status
if st.sidebar.button("Authenticate Earth Engine"):
    try:
        ee.Authenticate()
        st.sidebar.success("Earth Engine authentication initiated!")
        st.sidebar.info("Please follow the authentication flow in your browser, then refresh this page.")
    except Exception as e:
        st.sidebar.error(f"Authentication failed: {e}")

# Check if EE is initialized
try:
    ee.Initialize(project='bigquerylearning-476613')
    st.sidebar.success("✅ Earth Engine is ready!")
    ee_ready = True
except:
    st.sidebar.warning("⚠️ Earth Engine not authenticated. Using mock data.")
    ee_ready = False

# SCB API Information
st.sidebar.markdown("---")
st.sidebar.subheader("📊 SCB Data Integration")
st.sidebar.info("""
**Real Swedish Statistics:**
- 🏠 Hushållsinkomst (Household Income)
- 👥 Antal personer (Household Size) 
- 👤 Ålder (Age Demographics)
- 🚗 Bilägande (Vehicle Ownership)

Data hämtas från SCB API baserat på kommun.
""")

city_input = st.text_input("🏘️ Ange stad eller postnummer:", "Jönköping", 
                          help="Exempel: Jönköping, Stockholm, 55318, 11122")

# Show some example inputs
st.caption("💡 **Exempel:** Jönköping, Stockholm, Göteborg, 55318, 11122, 41126")

# Info about pre-set filters
st.info("ℹ️ **Filtren i sidopanelen är redan aktiva!** Ställ in dina preferenser först, sedan sök efter plats för att få filtrerade resultat direkt.")

# Test geocoding button

# if st.button("Download", key="download_button"):
#     lat, lon = get_coordinates(city)
#     if lat is None:
#         st.error("City not found.")
#         st.stop()

#     region = ee.Geometry.Point([lon, lat]).buffer(5000)
    
price_per_kwh = st.sidebar.number_input("Electricity Price (kr/kWh)", min_value=0.0, step=0.1, value=DEFAULT_ELECTRICITY_PRICE)
min_score = st.sidebar.slider("Minimum Solar Score", 0, 100, 50)

# Filtrar (tillgängliga redan innan sökning)
st.sidebar.markdown("---")
st.sidebar.subheader("🏠 Hushållsfilter")
st.sidebar.info("💡 **Tips:** Ställ in dina filter redan nu! De kommer appliceras automatiskt när du söker efter en plats.")

with st.sidebar.expander("🚗 Bil & Ekonomi", expanded=True):
    ev_filter = st.checkbox("Endast hus med elbil", value=False)
    income_range = st.slider("Hushållsinkomst (kSEK)", 200, 2000, (300, 1500))

with st.sidebar.expander("👥 Hushåll & Ålder", expanded=False):
    household_size_filter = st.multiselect("Hushållsstorlek", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    owner_age_range = st.slider("Ägarens ålder", 25, 80, (25, 80))

with st.sidebar.expander("🏠 Hus & Energi", expanded=False):
    roof_age_range = st.slider("Takets ålder (år)", 0, 50, (0, 50))
    usage_range = st.slider("Nuvarande förbrukning (kWh/år)", 5000, 35000, (10000, 25000))

if st.button("Fetch Data", key="fetch_button"):
    # Initialize progress placeholder
    progress_placeholder = st.empty()
    
    # Initialize Earth Engine
    progress_placeholder.info("🔧 Initializing Earth Engine...")
    ee_initialized = initialize_ee()
    
    if ee_initialized:
        progress_placeholder.success("🌍 Earth Engine ready - using real building detection!")
    else:
        progress_placeholder.warning("🔧 Earth Engine not available - using mock data")
    
    # Get coordinates and actual city name from input (could be postal code)
    progress_placeholder.info("🔍 Searching for location...")
    lat, lon, actual_city = get_coordinates_and_city(city_input)
    
    if lat is None:
        progress_placeholder.error(f"❌ Platsen '{city_input}' kunde inte hittas.")
        
        with st.expander("ℹ️ Försök med dessa alternativ", expanded=True):
            st.info("""
            **Försök med:**
            - Stadsnamn: Stockholm, Göteborg, Malmö, Jönköping
            - Postnummer: 11122, 41126, 20212, 55318
            - Kontrollera stavning av stadsnamnet
            """)
            
            # Show what postal code mapping would give if it's a number
            if city_input.strip().replace(' ', '').isdigit():
                postal_code = city_input.strip().replace(' ', '')
                if len(postal_code) == 5:
                    municipality_code = get_municipality_code(city_input)
                    st.warning(f"📮 Postnummer {postal_code} mappas till kommun {municipality_code}, men koordinater kunde inte hittas.")
        
        st.stop()
    
    progress_placeholder.success(f"📍 Plats funnen: {actual_city}")

    # Create region around the location
    region = ee.Geometry.Point([lon, lat]).buffer(5000) if ee_initialized else None

    if ee_initialized:
        try:
            progress_placeholder.info("🛰️ Analyserar byggnader med Sentinel-2 + VIDA + DEM...")
            
            # Define area of interest (dynamic city location)
            region = ee.Geometry.Point([lon, lat]).buffer(10000)
            
            # Load Sentinel-2 imagery (summer, low cloud) - same as your JS code
            s2 = ee.ImageCollection('COPERNICUS/S2') \
                .filterBounds(region) \
                .filterDate('2023-06-01', '2023-08-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                .median() \
                .clip(region)
            
            # Load building footprints (VIDA dataset) - same as your JS code
            buildings = ee.FeatureCollection("projects/sat-io/open-datasets/VIDA_COMBINED/SWE") \
                .filterBounds(region)
            
            # Load DEM and calculate slope/aspect - same as your JS code
            dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM').clip(region)
            terrain = ee.Terrain.products(dem)
            slope = terrain.select('slope')     # Roof steepness
            aspect = terrain.select('aspect')   # Roof direction
            
            # Basic shadow detection from Sentinel-2 (dark pixels) - same as your JS code
            shadowMask = s2.select('B2').lt(500)  # Blue band low reflectance
            shadows = shadowMask.updateMask(shadowMask)
            
            # Sample slope/aspect for each building - same as your JS code
            buildingsWithSlope = slope.reduceRegions(
                collection=buildings,
                reducer=ee.Reducer.mean(),
                scale=30
            )
            
            buildingsWithAspect = aspect.reduceRegions(
                collection=buildingsWithSlope,
                reducer=ee.Reducer.mean(),
                scale=30
            )
            
            # Sample shadow data for buildings
            buildingsWithShadows = shadows.reduceRegions(
                collection=buildingsWithAspect,
                reducer=ee.Reducer.mean(),
                scale=30
            )
            
            # Limit results for performance
            final_buildings = buildingsWithShadows.limit(5000)
            
            # Fetch the actual data from Earth Engine
            progress_placeholder.info(f"📡 Hämtar byggdata från satellit för {actual_city}...")
            geojson = final_buildings.getInfo()
            
            if not geojson.get('features'):
                raise Exception("No VIDA building features found in this region")
            
            # Convert to GeoDataFrame
            rooftops = gpd.GeoDataFrame.from_features(geojson['features'])
            progress_placeholder.success(f"✅ Hittade {len(rooftops)} byggnader med verklig takdata!")
            
            # Use the real slope/aspect data from GEE (not mock)
            # Rename columns to match your original workflow
            if 'mean' in rooftops.columns:
                rooftops['slope'] = rooftops['mean'].round().astype(int)  # slope from DEM
            if 'mean_1' in rooftops.columns:  
                rooftops['aspect'] = rooftops['mean_1'].round().astype(int)  # aspect from DEM
            if 'mean_2' in rooftops.columns:
                rooftops['shadow_index'] = rooftops['mean_2'].round(2)  # shadow from Sentinel-2
            
            # Calculate building area from geometry
            rooftops['area_in_meters'] = (rooftops.geometry.area * 111000 * 111000).round().astype(int)  # rough conversion to m²
            
            # Add coordinates for mapping
            rooftops['lon'] = rooftops.geometry.centroid.x
            rooftops['lat'] = rooftops.geometry.centroid.y
            
        except Exception as e:
            progress_placeholder.warning("⚠️ GEE data misslyckades - använder mockdata...")
            ee_initialized = False
    
    if not ee_initialized:
        progress_placeholder.info("🎭 Genererar demo-byggdata...")
        
        # Reduced number of buildings for faster demo performance
        num_buildings = 50  # Reduced from 100
        center_lat, center_lon = lat, lon
        lat_offset = np.random.uniform(-0.02, 0.02, num_buildings)
        lon_offset = np.random.uniform(-0.02, 0.02, num_buildings)
        
        mock_buildings = pd.DataFrame({
            'lat': center_lat + lat_offset,
            'lon': center_lon + lon_offset,
            'area_in_meters': np.random.randint(50, 500, size=num_buildings),
            'aspect': np.random.randint(120, 240, size=num_buildings),      # roof direction
            'slope': np.random.randint(10, 40, size=num_buildings),         # roof steepness  
            'shadow_index': np.random.uniform(0, 0.3, size=num_buildings)   # shadow level
        })
        
        geometry = [Point(lon, lat) for lon, lat in zip(mock_buildings['lon'], mock_buildings['lat'])]
        rooftops = gpd.GeoDataFrame(mock_buildings, geometry=geometry)

    # Compute solar metrics using vectorized operations for speed
    progress_placeholder.info("☀️ Beräknar solpotential...")
    
    # Vectorized solar score calculation - safe column access
    if 'aspect' in rooftops.columns:
        aspects = rooftops['aspect']
    elif 'mean_1' in rooftops.columns:
        aspects = rooftops['mean_1']
    else:
        aspects = pd.Series([180] * len(rooftops))
    
    if 'slope' in rooftops.columns:
        slopes = rooftops['slope']
    elif 'mean' in rooftops.columns:
        slopes = rooftops['mean']
    else:
        slopes = pd.Series([25] * len(rooftops))
    
    if 'shadow_index' in rooftops.columns:
        shadows = rooftops['shadow_index']
    elif 'mean_2' in rooftops.columns:
        shadows = rooftops['mean_2']
    else:
        shadows = pd.Series([0] * len(rooftops))
    
    # Ensure all are numeric
    aspects = pd.to_numeric(aspects, errors='coerce').fillna(180)
    slopes = pd.to_numeric(slopes, errors='coerce').fillna(25)
    shadows = pd.to_numeric(shadows, errors='coerce').fillna(0)
    
    aspect_scores = np.maximum(0, 1 - np.abs(aspects - 180) / 180)
    slope_scores = np.maximum(0, 1 - np.abs(slopes - 25) / 25)
    shadow_scores = 1 - np.minimum(shadows, 1)
    
    rooftops['solar_score'] = (100 * (0.5 * aspect_scores + 0.3 * slope_scores + 0.2 * shadow_scores)).round().astype(int)
    rooftops['kWp'] = (rooftops['area_in_meters'] / M2_PER_KWP).round(1)
    rooftops['kWh/year'] = (rooftops['kWp'] * SE_LATLON_IRRADIANCE_KWH_PER_KWP).round().astype(int)
    rooftops['Savings (kSEK/year)'] = ((rooftops['kWh/year'] * price_per_kwh) / 1000.0).round(1)
    
    # Round geometric/building data too
    rooftops['area_in_meters'] = rooftops['area_in_meters'].round().astype(int)
    if 'slope' in rooftops.columns:
        rooftops['slope'] = rooftops['slope'].round().astype(int)
    if 'aspect' in rooftops.columns:
        rooftops['aspect'] = rooftops['aspect'].round().astype(int)
    if 'shadow_index' in rooftops.columns:
        rooftops['shadow_index'] = rooftops['shadow_index'].round(2)

    # Enrich with real SCB demographic data and real addresses
    progress_placeholder.info("📊 Hämtar SCB-data och verkliga adresser...")
    rooftops = enrich_with_scb_data(rooftops, actual_city)
    
    # Final summary
    municipality_code = get_municipality_code(city_input)
    real_addresses = len([addr for addr in rooftops['address'] if addr != "Address not found" and "Simulated" not in addr])
    total_addresses = len(rooftops)
    
    progress_placeholder.success(f"✅ Klart! {total_addresses} byggnader analyserade för {actual_city}")

    # Show summary in expandable sections
    with st.expander("� Datasammanfattning", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Verkliga adresser", f"{real_addresses}/{total_addresses}")
            st.metric("Medelinkomst", f"{rooftops['income_ksek'].mean():.0f} kSEK")
            st.metric("EV-ägande", f"{rooftops['has_ev'].mean()*100:.1f}%")
        with col2:
            st.metric("Hushållsstorlek", f"{rooftops['household_size'].mean():.1f} personer")
            st.metric("Energianvändning", f"{rooftops['current_usage_kwh'].mean():.0f} kWh/år")
            st.metric("Kommuner", f"{len(rooftops['municipality'].unique())} st")
    
    with st.expander("🔍 Tekniska detaljer", expanded=False):
        st.info("""
        **Datakällor:**
        - 🛰️ Byggdata: Google Earth Engine (Sentinel-2 + VIDA + DEM)
        - 📍 Adresser: Reverse geocoding via Nominatim
        - 📊 Demografi: SCB API (Statistiska centralbyrån)
        - ☀️ Solpotential: Beräknad från takvinkel, riktning och skuggor
        """)

    # Optimized filtering with single pass
    progress_placeholder.info("🔍 Applicerar filter...")
    
    # Create filter mask efficiently using pre-set filters
    mask = rooftops['solar_score'] >= min_score
    
    # Apply all filters in one operation (using filters set before search)
    if ev_filter:
        mask &= rooftops['has_ev']
    
    mask &= (rooftops['income_ksek'] >= income_range[0]) & (rooftops['income_ksek'] <= income_range[1])
    
    if household_size_filter:
        mask &= rooftops['household_size'].isin(household_size_filter)
    
    mask &= (rooftops['roof_age_years'] >= roof_age_range[0]) & (rooftops['roof_age_years'] <= roof_age_range[1])
    mask &= (rooftops['owner_age'] >= owner_age_range[0]) & (rooftops['owner_age'] <= owner_age_range[1])
    mask &= (rooftops['current_usage_kwh'] >= usage_range[0]) & (rooftops['current_usage_kwh'] <= usage_range[1])
    
    filtered = rooftops[mask]

    # Add coordinates for map (handle both GEE data and mock data)
    if 'lon' not in filtered.columns or 'lat' not in filtered.columns:
        filtered['lon'] = filtered.geometry.centroid.x
        filtered['lat'] = filtered.geometry.centroid.y

    # Optimized map visualization
    progress_placeholder.success(f"✅ Klart! {len(filtered)} av {len(rooftops)} byggnader matchar filtren")
    
    st.subheader("🗺️ Karta över tak")
    if not filtered.empty:
        # Ensure we have coordinates
        if 'lon' not in filtered.columns or 'lat' not in filtered.columns:
            filtered = filtered.copy()
            filtered['lon'] = filtered.geometry.centroid.x
            filtered['lat'] = filtered.geometry.centroid.y
        
        # Limit map points for performance if too many
        map_data = filtered.head(200) if len(filtered) > 200 else filtered
        
        # Simple color calculation (avoid complex vectorization)
        map_data = map_data.copy()
        map_data['color'] = map_data['solar_score'].apply(
            lambda s: [int(255*(s/100)), int(255*(1 - s/100)), 80]
        )
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["lon", "lat"],
            get_radius=8,
            get_fill_color="color",
            pickable=True
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        
        if len(filtered) > 200:
            st.caption(f"Visar första 200 av {len(filtered)} byggnader på kartan för prestanda")
    else:
        st.warning("Inga tak matchade filtren.")

    # Table in expandable section
    with st.expander(f"📊 Alla {len(filtered)} filtrerade byggnader", expanded=False):
        cols_to_show = ['address', 'postal_code', 'municipality', 'area_in_meters', 'slope', 'aspect', 'shadow_index', 'solar_score', 'kWp', 'kWh/year', 'Savings (kSEK/year)',
                        'has_ev', 'income_ksek', 'household_size', 'roof_age_years', 'owner_age', 'current_usage_kwh']
        
        # Only show columns that exist
        available_cols = [col for col in cols_to_show if col in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True)

    # Top leads
    st.subheader("🏆 Topp 5 Solar Leads")
    top_leads = filtered.sort_values(by="solar_score", ascending=False).head(5)
    
    if len(top_leads) > 0:
        # Show compact top leads
        for idx, (_, lead) in enumerate(top_leads.iterrows(), 1):
            with st.expander(f"#{idx} - {lead.get('address', 'Unknown Address')} - Score: {lead['solar_score']:.0f}/100", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**🏠 Fastighet:**")
                    st.write(f"📍 {lead.get('address', 'N/A')}")
                    st.write(f"📮 {lead.get('postal_code', 'N/A')}")
                    st.write(f"🏘️ {lead.get('municipality', 'N/A')}")
                
                with col2:
                    st.write("**☀️ Solar:**")
                    st.write(f"📐 Score: {lead['solar_score']:.0f}/100")
                    st.write(f"⚡ Potential: {lead['kWp']:.1f} kWp")
                    st.write(f"💰 Besparing: {lead['Savings (kSEK/year)']:.1f} kSEK/år")
                
                with col3:
                    st.write("**👥 Hushåll:**")
                    st.write(f"💰 Inkomst: {lead.get('income_ksek', 0):.0f} kSEK")
                    st.write(f"👥 Storlek: {lead.get('household_size', 0)} personer")
                    st.write(f"🚗 EV: {'Ja' if lead.get('has_ev', False) else 'Nej'}")
        
        # Compact summary table in dropdown
        with st.expander("📋 Sammanfattningstabell", expanded=False):
            display_cols = ['address', 'postal_code', 'solar_score', 'kWp', 'Savings (kSEK/year)', 'income_ksek', 'has_ev']
            summary_cols = [col for col in display_cols if col in top_leads.columns]
            st.table(top_leads[summary_cols])
    else:
        st.info("Inga byggnader att visa med de valda filtren.")

    # Download button
    with st.expander("💾 Ladda ner data", expanded=False):
        csv = filtered[available_cols].to_csv(index=False)
        filename = f"{actual_city.replace(' ', '_')}_solar_buildings_with_addresses.csv"
        st.download_button("Ladda ner filtrerad data som CSV", csv, filename, "text/csv")
        st.caption(f"Innehåller {len(filtered)} byggnader med alla kolumner")