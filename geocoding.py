"""
Geocoding and location utilities for Swedish locations
This module handles converting user input (cities, addresses, postal codes) 
to coordinates and converting coordinates back to addresses
"""

import streamlit as st
from geopy.geocoders import Nominatim
import time

@st.cache_data(ttl=3600)  # Cache geocoding results for 1 hour to improve performance
def get_coordinates_and_city(input_text):
    """
    Cached wrapper for forward geocoding (text → coordinates)
    
    This function converts user input (city names, addresses, postal codes) 
    into latitude/longitude coordinates for Swedish locations.
    
    Args:
        input_text: User input - can be city name, address, or postal code
        
    Returns:
        tuple: (latitude, longitude, full_address) or (None, None, None) if not found
    """
    return get_coordinates_and_city_internal(input_text)

def get_coordinates_and_city_internal(input_text):
    """
    Core forward geocoding logic - converts user input to coordinates
    
    This function intelligently detects whether the user entered a city name,
    street address, or postal code, then uses appropriate search strategies
    to find the location within Sweden.
    
    Args:
        input_text: Raw user input from the search box
        
    Returns:
        tuple: (latitude, longitude, full_address) for Swedish locations only
    """
    # Initialize the geocoding service (OpenStreetMap's Nominatim)
    geolocator = Nominatim(user_agent="solar_app_v2")
    input_clean = input_text.strip()
    
    # STEP 1: Detect input type and create appropriate search strategies
    search_strategies = []
    
    # POSTAL CODE DETECTION: Swedish postal codes are 5 digits (format: 12345 or 123 45)
    if input_clean.replace(' ', '').isdigit() and len(input_clean.replace(' ', '')) == 5:
        postal_code = input_clean.replace(' ', '')
        formatted_postal = f"{postal_code[:3]} {postal_code[3:]}"  # Swedish format: "123 45"
        
        search_strategies = [
            f"{formatted_postal}, Sweden",  # Most likely to succeed
            f"{postal_code}, Sweden",       # Without space
            f"{formatted_postal}"           # Without country
        ]
        print(f"Geocoding postal code: {formatted_postal}")
        
    # STREET ADDRESS DETECTION: Contains both numbers and letters (e.g., "Storgatan 15")
    elif any(char.isdigit() for char in input_clean) and any(char.isalpha() for char in input_clean):
        search_strategies = [
            f"{input_clean}, Sweden",   # Try with country first
            f"{input_clean}",           # Try without country
            f"{input_clean}, Sverige",  # Try with Swedish country name
        ]
        print(f"Geocoding street address: {input_clean}")
        
    # CITY/PLACE NAME: Default case for plain text input
    else:
        search_strategies = [
            f"{input_clean}, Sweden",   # International name
            f"{input_clean}, Sverige",  # Swedish name
            f"{input_clean}"            # No country specified
        ]
        print(f"Geocoding city/place: {input_clean}")
    
    # STEP 2: Try each search strategy until we find a valid Swedish location
    for query in search_strategies:
        try:
            # Make the geocoding API call
            location = geolocator.geocode(query, timeout=8)  # 8 second timeout for reliability
            if location:
                # STEP 3: Validate that the result is actually in Sweden
                # Swedish coordinates: latitude 55-70°N, longitude 10-25°E
                if 55 <= location.latitude <= 70 and 10 <= location.longitude <= 25:
                    print(f"Found location: {location.address} at {location.latitude:.6f}, {location.longitude:.6f}")
                    return location.latitude, location.longitude, location.address
                else:
                    print(f"Location outside Sweden bounds: {location.latitude}, {location.longitude}")
        except Exception as e:
            print(f"Geocoding error for '{query}': {e}")
            continue  # Try next strategy
    
    # No valid Swedish location found with any strategy
    print(f"No valid Swedish location found for: {input_text}")
    return None, None, None

@st.cache_data(ttl=3600)  # Cache reverse geocoding results for 1 hour
def reverse_geocode_coordinates(lat, lon):
    """
    Cached wrapper for reverse geocoding (coordinates → address)
    
    This function converts latitude/longitude coordinates back into
    Swedish address components (street, postal code, city).
    
    Args:
        lat: Latitude coordinate (float)
        lon: Longitude coordinate (float)
        
    Returns:
        tuple: (street_address, postal_code, city) - all as strings
    """
    return reverse_geocode_coordinates_internal(lat, lon)

def reverse_geocode_coordinates_internal(lat, lon):
    """
    Core reverse geocoding logic - converts coordinates to Swedish address components
    
    This function uses multiple search strategies with different zoom levels and languages
    to maximize the chance of finding a meaningful address for any Swedish coordinate.
    
    Args:
        lat: Latitude coordinate (float)
        lon: Longitude coordinate (float)
        
    Returns:
        tuple: (street_address, postal_code, city) - extracted address components
    """
    # Initialize the geocoding service
    geolocator = Nominatim(user_agent="solar_app_v2")
    
    # STRATEGY DEFINITION: Multiple approaches to handle different location types
    # Each strategy uses different zoom levels and languages for comprehensive coverage
    strategies = [
        # Strategy 1: Exact location with Swedish language preference
        {"zoom": None, "language": "sv", "addressdetails": True},
        
        # Strategy 2: Neighborhood level (good for urban areas)
        {"zoom": 16, "language": "sv", "addressdetails": True},
        
        # Strategy 3: District level (good for suburban areas)
        {"zoom": 14, "language": "sv", "addressdetails": True},
        
        # Strategy 4: Municipality level (good for rural areas)
        {"zoom": 12, "language": "sv", "addressdetails": True},
        
        # Strategy 5: County level (very broad, for remote locations)
        {"zoom": 10, "language": "sv", "addressdetails": True},
        
        # Strategy 6: English fallback (for international data)
        {"zoom": 14, "language": "en", "addressdetails": True}
    ]
    
    # MAIN PROCESSING LOOP: Try each strategy until we get a usable result
    for strategy in strategies:
        try:
            # Configure API call parameters based on current strategy
            params = {
                "timeout": 15,  # Allow up to 15 seconds for API response
                "exactly_one": True,  # We only want the best match
                "language": strategy["language"],  # Prefer Swedish or English
                "addressdetails": strategy["addressdetails"]  # Get detailed address components
            }
            if strategy["zoom"]:
                params["zoom"] = strategy["zoom"]  # Set zoom level if specified
            
            # Make the reverse geocoding API call
            location = geolocator.reverse(f"{lat}, {lon}", **params)
            
            if location and location.raw:
                # Extract the address components from the API response
                addr = location.raw.get('address', {})
                
                # STEP 1: Extract street-level information
                street_number = addr.get('house_number', '')
                street_name = (addr.get('road') or          # Main roads
                             addr.get('footway') or         # Walking paths
                             addr.get('path') or            # General paths
                             addr.get('pedestrian') or      # Pedestrian areas
                             addr.get('cycleway', ''))      # Bike paths
                
                # STEP 2: Extract postal code (Swedish format)
                postal_code = addr.get('postcode', '')
                
                # STEP 3: Extract city/municipality information (with priority order)
                city = (addr.get('city') or                 # Major cities
                       addr.get('town') or                  # Smaller towns
                       addr.get('village') or               # Villages
                       addr.get('municipality') or          # Swedish municipalities
                       addr.get('county') or                # Counties (län)
                       addr.get('state_district') or        # Administrative districts
                       addr.get('administrative', ''))      # Any administrative area
                
                # STEP 4: Build the street address with comprehensive fallback logic
                if street_name and street_number:
                    # Best case: full street address with number
                    street_address = f"{street_name} {street_number}"
                elif street_name:
                    # Good case: street name without number
                    street_address = street_name
                else:
                    # Fallback: try to find any meaningful locality identifier
                    locality = (addr.get('suburb') or           # Suburbs
                              addr.get('neighbourhood') or      # Neighborhoods
                              addr.get('hamlet') or             # Small settlements
                              addr.get('locality') or           # General localities
                              addr.get('village') or            # Villages (duplicate for safety)
                              addr.get('isolated_dwelling') or   # Remote houses
                              addr.get('farm') or               # Farm names
                              addr.get('quarter') or            # City quarters
                              addr.get('residential') or        # Residential areas
                              addr.get('industrial') or         # Industrial areas
                              addr.get('commercial', ''))       # Commercial areas
                    
                    if locality:
                        street_address = locality
                    else:
                        # Last resort: use any identifiable place name
                        place_name = (addr.get('tourism') or    # Tourist attractions
                                    addr.get('amenity') or      # Amenities
                                    addr.get('building') or     # Building names
                                    addr.get('natural') or      # Natural features
                                    addr.get('landuse') or      # Land use areas
                                    city or                     # Use city as address
                                    addr.get('county', ''))     # County as final fallback
                        
                        if place_name and place_name not in ["Unknown City", ""]:
                            street_address = place_name
                        else:
                            street_address = "Address not found"
                
                # STEP 5: Return result if we got any useful information
                if street_address != "Address not found" or postal_code or city:
                    return (street_address, 
                           postal_code or "Unknown Postal", 
                           city or "Unknown City")
        
        except Exception as e:
            # Log the error and try the next strategy
            print(f"Reverse geocoding error for {lat}, {lon} with strategy {strategy}: {e}")
            continue  # Move to next strategy
    
    # All strategies failed - return fallback values
    return "Address not found", "Unknown Postal", "Unknown City"