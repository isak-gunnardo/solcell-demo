"""
Debug geocoding issues in the application
"""

import streamlit as st
import pandas as pd
import numpy as np
from geocoding import reverse_geocode_coordinates
from earth_engine import get_building_data_from_ee, generate_mock_building_data

def debug_coordinates():
    """Debug coordinate data from building sources"""
    
    # Test Stockholm coordinates
    lat, lon = 59.3293, 18.0686
    
    print(f"Testing coordinates: {lat}, {lon}")
    
    # Test direct geocoding
    print("\n=== Direct Geocoding Test ===")
    address, postal, city = reverse_geocode_coordinates(lat, lon)
    print(f"Direct result: {address}, {postal}, {city}")
    
    # Test mock building data
    print("\n=== Mock Building Data Test ===")
    mock_buildings = generate_mock_building_data(lat, lon, 5)
    print(f"Mock buildings coords:")
    for i, row in mock_buildings.iterrows():
        print(f"  Building {i}: {row['lat']:.6f}, {row['lon']:.6f}")
        # Test geocoding for first building
        if i == 0:
            addr, post, ct = reverse_geocode_coordinates(row['lat'], row['lon'])
            print(f"  Geocoded: {addr}, {post}, {ct}")
    
    print("\n=== Debugging Application Geocoding ===")
    # Test the specific function from address_enrichment
    from address_enrichment import batch_reverse_geocode_internal
    
    test_df = mock_buildings.head(3)
    print(f"Testing with {len(test_df)} buildings")
    
    try:
        addresses, postal_codes, cities = batch_reverse_geocode_internal(test_df, "Stockholm", max_workers=1)
        print(f"Results:")
        for i, (addr, post, ct) in enumerate(zip(addresses, postal_codes, cities)):
            print(f"  Building {i}: {addr}, {post}, {ct}")
    except Exception as e:
        print(f"Error in batch geocoding: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_coordinates()