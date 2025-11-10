"""
Main Streamlit application for Solar Rooftop Lead Generation
A comprehensive tool for identifying and analyzing solar potential in Swedish buildings

This application:
1. Takes user input (city, address, or postal code)
2. Finds buildings using Google Earth Engine or mock data
3. Calculates solar potential based on roof characteristics
4. Enriches data with real Swedish addresses via geocoding
5. Adds demographic data from Statistics Sweden (SCB)
6. Provides interactive filtering and visualization
"""

import streamlit as st
import pandas as pd

# Import our custom modules for modular architecture
from config import DEFAULT_ELECTRICITY_PRICE
from ui_components import (
    setup_page_config, render_header, render_status_section,
    render_location_input, render_filter_section, render_map,
    render_results_table, render_top_leads, render_download_section
)
from geocoding import get_coordinates_and_city
from earth_engine import initialize_ee, get_building_data_from_ee, generate_mock_building_data
from scb_api import get_municipality_code, enrich_with_scb_data
from address_enrichment import add_addresses_to_buildings
from solar_calculations import calculate_solar_potential, filter_buildings
from eon_api import enrich_with_eon_energy_data

def main():
    """
    Main application flow - orchestrates the entire solar lead generation process
    
    This function handles:
    - UI setup and user input collection
    - Data fetching and processing pipeline
    - Result visualization and export
    """
    
    # PHASE 1: SETUP - Configure the Streamlit interface
    setup_page_config()  # Set page title, layout, favicon
    
    # PHASE 2: USER INTERFACE - Render main UI components
    render_header()  # App title and cache control
    ee_ready = render_status_section()  # Earth Engine and SCB API status
    city_input = render_location_input()  # Search box for city/address/postal code
    
    # PHASE 3: SESSION STATE - Track user interaction state
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False
    
    # PHASE 4: FILTERS - User-controllable data processing parameters
    filters = render_filter_section(st.session_state.search_performed)
    
    # PHASE 5: DATA FETCHING CONTROLS
    col1, col2 = st.columns([3, 1])
    with col1:
        fetch_button_pressed = st.button("Fetch Data", key="fetch_button")
    with col2:
        force_refresh = st.checkbox("Force Fresh Data", help="Ignore cache and fetch new data")
    
    # PHASE 6: MAIN DATA PROCESSING PIPELINE
    if fetch_button_pressed:
        # INPUT VALIDATION: Ensure user provided search input
        if not city_input or city_input.strip() == "":
            st.error("❌ Vänligen ange en stad eller postnummer")
            st.stop()
        
        # Update session state to change UI behavior
        st.session_state.search_performed = True
        
        # Create progress tracker for user feedback
        progress_placeholder = st.empty()
        
        # STEP 1: EARTH ENGINE INITIALIZATION
        # Try to connect to Google Earth Engine for real satellite data
        progress_placeholder.info("🔧 Initializing Earth Engine...")
        ee_initialized = initialize_ee()
        
        if ee_initialized:
            progress_placeholder.success("🌍 Earth Engine ready - using real building detection!")
        else:
            progress_placeholder.warning("🔧 Earth Engine not available - using mock data")
        
        # STEP 2: LOCATION GEOCODING
        # Convert user input (city/address/postal code) to coordinates
        progress_placeholder.info(f"🔍 Searching for location: {city_input}...")
        lat, lon, actual_city = get_coordinates_and_city(city_input)
        
        # Handle geocoding failure with helpful suggestions
        if lat is None:
            progress_placeholder.error(f"❌ Platsen '{city_input}' kunde inte hittas.")
            
            with st.expander("ℹ️ Försök med dessa alternativ", expanded=True):
                st.info("""
                **Försök med:**
                - **Stadsnamn:** Stockholm, Göteborg, Malmö, Jönköping
                - **Adresser:** Storgatan 15 Stockholm, Drottninggatan 10 Malmö
                - **Postnummer:** 11122, 41126, 20212, 55318
                - **Tips:** Kontrollera stavning och inkludera stad för adresser
                """)
                
                # Special handling for postal code input
                if city_input.strip().replace(' ', '').isdigit():
                    postal_code = city_input.strip().replace(' ', '')
                    if len(postal_code) == 5:
                        municipality_code = get_municipality_code(city_input)
                        st.warning(f"📮 Postnummer {postal_code} mappas till kommun {municipality_code}, men koordinater kunde inte hittas.")
            
            st.stop()  # Stop processing if location not found
        
        progress_placeholder.success(f"📍 Plats funnen: {actual_city}")
        
        # STEP 3: BUILDING DATA ACQUISITION
        # Get building footprints from Google Earth Engine or generate mock data
        if ee_initialized:
            progress_placeholder.info(f"📡 Hämtar byggdata från satellit för {actual_city}...")
            rooftops = get_building_data_from_ee(lat, lon, actual_city)
            
            # Fallback to mock data if Earth Engine fails
            if rooftops is None:
                progress_placeholder.warning("⚠️ GEE data misslyckades - använder mockdata...")
                rooftops = generate_mock_building_data(lat, lon)
            else:
                progress_placeholder.success(f"✅ Hittade {len(rooftops)} byggnader med verklig takdata!")
        else:
            # Use synthetic data when Earth Engine is unavailable
            progress_placeholder.info("🎭 Genererar demo-byggdata...")
            rooftops = generate_mock_building_data(lat, lon)
        
        # STEP 4: SOLAR POTENTIAL CALCULATION
        # Analyze roof characteristics and calculate energy production potential
        progress_placeholder.info("☀️ Beräknar solpotential...")
        rooftops = calculate_solar_potential(rooftops, filters['price_per_kwh'])
        
        # STEP 5: ADDRESS ENRICHMENT
        # Convert coordinates to real Swedish addresses (user-controllable quality)
        progress_placeholder.info(f"🏠 Hämtar {filters['num_to_geocode']} riktiga adresser...")
        rooftops = add_addresses_to_buildings(rooftops, actual_city, filters['num_to_geocode'])
        
        # STEP 6: DEMOGRAPHIC DATA ENRICHMENT
        # Add household income, size, age, and other demographic data from SCB
        progress_placeholder.info("📊 Hämtar SCB-data...")
        municipality_code = get_municipality_code(city_input)
        rooftops = enrich_with_scb_data(rooftops, municipality_code)
        
        # STEP 7: E.ON ENERGY DATA INTEGRATION
        # Add E.ON energy prices, consumption profiles, and smart grid opportunities
        progress_placeholder.info("🔌 Integrerar E.ON energidata...")
        rooftops = enrich_with_eon_energy_data(rooftops)
        
        # STEP 8: FILTER APPLICATION
        # Apply user-defined filters to find the best prospects
        progress_placeholder.info("🔍 Applicerar filter...")
        filtered = filter_buildings(rooftops, filters)
        
        # STEP 9: DATA PREPARATION FOR VISUALIZATION
        # Ensure coordinate columns exist for mapping
        if 'lon' not in filtered.columns or 'lat' not in filtered.columns:
            filtered = filtered.copy()
            if hasattr(filtered, 'geometry'):
                # Extract coordinates from geometry column if available
                filtered['lon'] = filtered.geometry.centroid.x
                filtered['lat'] = filtered.geometry.centroid.y
            else:
                # Use search location as fallback
                filtered['lon'] = lon
                filtered['lat'] = lat
        
        # STEP 10: RESULTS PRESENTATION
        # Show completion status and render all result components
        progress_placeholder.success(f"✅ Klart! {len(filtered)} av {len(rooftops)} byggnader matchar filtren")
        
        # Render interactive results dashboard with E.ON energy data
        render_map(filtered, lat, lon)           # Interactive map with building locations
        render_results_table(filtered)          # Sortable data table with E.ON data
        render_top_leads(filtered)              # Highlighted best prospects with energy analysis
        render_download_section(filtered, actual_city)  # CSV export with comprehensive energy data

if __name__ == "__main__":
    main()