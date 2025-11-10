"""
Address enrichment using reverse geocoding and property information
This module handles converting building coordinates to real Swedish addresses
and enriches them with detailed property information from Lantmäteriet
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from geocoding import reverse_geocode_coordinates
from lantmateri_api import get_property_info_by_address, validate_swedish_address

@st.cache_data(ttl=3600)  # Cache geocoding results for 1 hour to avoid repeated API calls
def batch_reverse_geocode(_buildings_df, city_name, num_to_geocode=75, max_workers=1):
    """
    Batch reverse geocode building coordinates to get real addresses (cached wrapper)
    
    Args:
        _buildings_df: DataFrame with building coordinates (lat, lon columns)
        city_name: Name of the city/area being processed (for context)
        num_to_geocode: How many buildings to geocode with real API calls (user-controlled)
        max_workers: Number of parallel workers (currently unused, sequential processing)
    
    Returns:
        Result from batch_reverse_geocode_internal (addresses, postal_codes, cities)
    """
    return batch_reverse_geocode_internal(_buildings_df, city_name, num_to_geocode, max_workers)

def is_address_in_target_city(address_city, target_city):
    """
    Check if the geocoded address belongs to the target city or nearby area
    
    This function validates whether a reverse-geocoded address actually matches
    the city the user searched for, handling Swedish municipality naming conventions.
    
    Args:
        address_city: City name returned from reverse geocoding API
        target_city: City name the user originally searched for
        
    Returns:
        bool: True if the address is in the target city area, False otherwise
    """
    if not address_city or not target_city:
        return False
    
    address_city_lower = address_city.lower()
    target_city_lower = target_city.lower()
    
    # Direct city name match (e.g., "Stockholm" in "Stockholm, Sverige")
    if target_city_lower in address_city_lower:
        return True
    
    # Swedish municipality match (e.g., "Vetlanda kommun" for "Vetlanda" search)
    if f"{target_city_lower} kommun" in address_city_lower:
        return True
        
    return False

def batch_reverse_geocode_internal(_buildings_df, city_name, num_to_geocode=75, max_workers=1):
    """
    Core reverse geocoding logic - processes building coordinates to Swedish addresses
    
    This function handles the actual API calls to convert lat/lon coordinates into
    real Swedish addresses. It uses sequential processing to respect API rate limits
    and provides user control over quality vs speed tradeoff.
    
    Args:
        _buildings_df: DataFrame with building data including 'lat' and 'lon' columns
        city_name: Target city name for validation and context
        num_to_geocode: Number of buildings to process with real API calls (user-controlled)
        max_workers: Unused parameter (kept for compatibility, we use sequential processing)
        
    Returns:
        tuple: (addresses, postal_codes, cities) - lists of address components for all buildings
    """
    print(f"Starting reverse geocoding for {len(_buildings_df)} buildings in {city_name}...")
    print(f"User selected: {num_to_geocode} real addresses to geocode")
    
    # Respect user's choice but don't exceed available buildings
    num_to_geocode = min(len(_buildings_df), num_to_geocode)
    
    # Initialize lists to store address components
    addresses = []
    postal_codes = []
    cities = []
    
    # PHASE 1: Real API geocoding for user-specified number of buildings
    # Use sequential processing to avoid overwhelming the Nominatim API
    for i in range(num_to_geocode):
        try:
            # Extract coordinates for this building
            lat = _buildings_df.iloc[i]['lat']
            lon = _buildings_df.iloc[i]['lon']
            print(f"Geocoding building {i+1}/{num_to_geocode}: {lat:.6f}, {lon:.6f}")
            
            # Call the reverse geocoding API
            street_address, postal_code, city = reverse_geocode_coordinates(lat, lon)
            addresses.append(street_address)
            postal_codes.append(postal_code)
            cities.append(city)
            
            # Validate if the geocoded address matches the target city
            in_target_city = is_address_in_target_city(city, city_name)
            city_marker = "✓" if in_target_city else "○"  # Visual indicator for console
            
            print(f"  Result {city_marker}: {street_address}, {postal_code}, {city}")
            
            # API rate limiting: pause between requests to avoid being blocked
            if i < num_to_geocode - 1:  # Don't sleep after the last geocoding call
                # Adaptive delay: shorter for successful geocodes, longer for failures
                delay = 1.2 if street_address != "Address not found" else 2.5
                time.sleep(delay)
                
        except Exception as e:
            # Handle geocoding failures gracefully
            print(f"Geocoding failed for building {i}: {e}")
            addresses.append("Address not found")
            postal_codes.append("Unknown Postal")
            cities.append("Unknown City")
    
    # PHASE 2: Generate simplified addresses for remaining buildings
    # This avoids making too many API calls while still providing meaningful data
    remaining_count = len(_buildings_df) - num_to_geocode
    if remaining_count > 0:
        print(f"Using simplified addresses for remaining {remaining_count} buildings...")
        
        # Extract patterns from successfully geocoded addresses to maintain consistency
        real_cities = [city for city in cities if city != "Unknown City"]
        
        # Priority logic: prefer cities that match the user's search
        search_city_matches = [city for city in real_cities if city_name.lower() in city.lower()]
        if search_city_matches:
            base_city = search_city_matches[0]  # Use matching city from geocoding results
        elif real_cities:
            base_city = real_cities[0]  # Use first successful geocoding result
        else:
            base_city = city_name or "Swedish City"  # Fallback to search term
        
        # Generate simplified but realistic-looking addresses for performance
        for i in range(remaining_count):
            addresses.append(f"Building Address {num_to_geocode + i + 1}")  # Generic but numbered
            postal_codes.append("000 00")  # Placeholder postal code
            cities.append(base_city)  # Use determined base city for consistency
    
    print(f"Completed geocoding: {num_to_geocode} real addresses, {remaining_count} simplified")
    return addresses, postal_codes, cities

def add_addresses_to_buildings(buildings_df, city_name="Unknown", num_to_geocode=75):
    """
    Main entry point for adding Swedish address data to building dataset
    
    This function takes a DataFrame of buildings with coordinates and enriches it
    with real Swedish addresses using reverse geocoding. The user can control
    the quality vs speed tradeoff through the num_to_geocode parameter.
    
    Args:
        buildings_df: DataFrame with building data (must have 'lat' and 'lon' columns)
        city_name: Name of the city/area being processed (for context and validation)
        num_to_geocode: Number of buildings to geocode with real API calls (user preference)
        
    Returns:
        DataFrame: Original DataFrame with added columns:
                  - 'address': Street address or building identifier
                  - 'postal_code': Swedish postal code (format: "123 45")
                  - 'municipality': City/municipality name
    """
    print("Adding addresses to building data...")
    
    # Call the batch geocoding function with user-specified quality level
    addresses, postal_codes, cities = batch_reverse_geocode(buildings_df, city_name, num_to_geocode)
    
    # Add the address data as new columns to the buildings DataFrame
    buildings_df = buildings_df.copy()  # Avoid modifying original DataFrame
    buildings_df['address'] = addresses
    buildings_df['postal_code'] = postal_codes
    buildings_df['municipality'] = cities
    
    print(f"Successfully added addresses to {len(buildings_df)} buildings")
    
    # Enhance with Lantmäteriet property information
    print("Enriching with property information from Lantmäteriet...")
    buildings_df = add_property_information(buildings_df)
    
    return buildings_df

def add_property_information(buildings_df):
    """
    Enhance buildings with detailed property information from Lantmäteriet
    
    This function takes buildings with addresses and adds comprehensive property
    information including ownership details, property characteristics, estimated
    values, and solar potential analysis.
    
    Args:
        buildings_df: DataFrame with building data including 'address' column
        
    Returns:
        DataFrame: Enhanced with property information columns
    """
    print("Fetching property information for buildings...")
    
    enhanced_df = buildings_df.copy()
    
    # Initialize new columns for property information
    property_columns = [
        'property_type', 'construction_year', 'living_area_sqm', 'plot_size_sqm',
        'floors', 'rooms', 'heating_type', 'roof_type', 'energy_class',
        'estimated_value_sek', 'roof_area_sqm', 'usable_roof_area_sqm',
        'estimated_panels', 'estimated_capacity_kw', 'suitability_score',
        'roof_condition', 'annual_property_tax_sek', 'renovation_recommendations'
    ]
    
    for col in property_columns:
        enhanced_df[col] = None
    
    # Process buildings with valid addresses
    valid_addresses = enhanced_df[enhanced_df['address'] != "Address not found"]
    total_valid = len(valid_addresses)
    
    print(f"Processing property information for {total_valid} buildings with valid addresses...")
    
    for idx, row in valid_addresses.iterrows():
        try:
            # Construct full address for property lookup
            full_address = f"{row['address']}, {row['postal_code']} {row['municipality']}"
            
            # Validate address format
            if not validate_swedish_address(full_address):
                print(f"Invalid address format: {full_address}")
                continue
            
            # Get property information from Lantmäteriet
            property_info = get_property_info_by_address(full_address)
            
            if property_info:
                # Extract property characteristics
                chars = property_info.get('property_characteristics', {})
                solar = property_info.get('solar_analysis', {})
                tax_info = property_info.get('estimated_property_tax', {})
                renovations = property_info.get('renovation_recommendations', [])
                
                # Update DataFrame with property information
                enhanced_df.at[idx, 'property_type'] = chars.get('property_type', 'Villa/Småhus')
                enhanced_df.at[idx, 'construction_year'] = chars.get('construction_year', 1970)
                enhanced_df.at[idx, 'living_area_sqm'] = chars.get('living_area_sqm', 120)
                enhanced_df.at[idx, 'plot_size_sqm'] = chars.get('plot_size_sqm', 800)
                enhanced_df.at[idx, 'floors'] = chars.get('floors', 2)
                enhanced_df.at[idx, 'rooms'] = chars.get('rooms', 5)
                enhanced_df.at[idx, 'heating_type'] = chars.get('heating_type', 'Värmepump')
                enhanced_df.at[idx, 'roof_type'] = chars.get('roof_type', 'Tegeltak')
                enhanced_df.at[idx, 'energy_class'] = chars.get('energy_class', 'C')
                enhanced_df.at[idx, 'estimated_value_sek'] = chars.get('estimated_value_sek', 3000000)
                
                # Solar potential information
                enhanced_df.at[idx, 'roof_area_sqm'] = solar.get('roof_area_sqm', 150)
                enhanced_df.at[idx, 'usable_roof_area_sqm'] = solar.get('usable_roof_area_sqm', 80)
                enhanced_df.at[idx, 'estimated_panels'] = solar.get('estimated_panels', 20)
                enhanced_df.at[idx, 'estimated_capacity_kw'] = solar.get('estimated_capacity_kw', 8.0)
                enhanced_df.at[idx, 'suitability_score'] = solar.get('suitability_score', 0.8)
                enhanced_df.at[idx, 'roof_condition'] = solar.get('roof_condition', 'Good')
                
                # Tax and renovation information
                enhanced_df.at[idx, 'annual_property_tax_sek'] = tax_info.get('annual_property_tax_sek', 22500)
                enhanced_df.at[idx, 'renovation_recommendations'] = '; '.join(renovations) if renovations else 'Inga rekommendationer'
                
                print(f"✓ Enhanced building: {chars.get('property_type', 'Unknown')} from {chars.get('construction_year', 'Unknown')}, {solar.get('estimated_capacity_kw', 0)} kW solar potential")
                
            else:
                print(f"○ No property info found for: {full_address}")
                
        except Exception as e:
            print(f"Error processing property info for {row['address']}: {e}")
            continue
    
    print(f"Completed property information enhancement for {total_valid} buildings")
    return enhanced_df