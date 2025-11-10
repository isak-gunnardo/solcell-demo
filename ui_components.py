"""
Streamlit UI components and layout
"""

import streamlit as st
import pydeck as pdk
import pandas as pd
from config import FILTER_DEFAULTS

def setup_page_config():
    """Configure Streamlit page settings"""
    from config import PAGE_CONFIG
    st.set_page_config(**PAGE_CONFIG)

def render_header():
    """Render the application header"""
    from config import APP_TITLE
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(APP_TITLE)
    with col2:
        if st.button("🗑️ Clear Cache", help="Clear cached data to force fresh search"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

def render_status_section():
    """Render Earth Engine and SCB status section"""
    from earth_engine import initialize_ee
    
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.subheader("🌍 Earth Engine Status")
        if st.button("Authenticate Earth Engine"):
            try:
                import ee
                ee.Authenticate()
                st.success("Earth Engine authentication initiated!")
                st.info("Please follow the authentication flow in your browser, then refresh this page.")
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        
        # Check if EE is initialized
        ee_ready = initialize_ee()
        if ee_ready:
            st.success("✅ Earth Engine is ready!")
        else:
            st.warning("⚠️ Earth Engine not authenticated. Using mock data.")
    
    with col_status2:
        st.subheader("📊 Data Integration")
        st.info("""
        **🛰️ Sentinel-5P Atmospheric Data:**
        - 🌫️ NO2 Air Quality Analysis
        - ☀️ Atmospheric Solar Efficiency
        - 🏭 Urban Pollution Impact
        
        **📊 SCB Swedish Statistics:**
        - 🏠 Hushållsinkomst (Household Income)
        - 👥 Antal personer (Household Size) 
        - 👤 Ålder (Age Demographics)
        - 🚗 Bilägande (Vehicle Ownership)
        
        **🔌 E.ON Energy Integration:**
        - ⚡ Regionala elpriser (SE1-SE4)
        - 📈 Konsumtionsprofiler
        - 💰 Solcellsfinansiering
        - 🏠 Smart Grid tjänster
        """)
    
    return ee_ready

def render_location_input():
    """Render location input section"""
    city_input = st.text_input("🏘️ Ange stad, adress eller postnummer:", 
                              help="Exempel: Stockholm, Storgatan 15 Stockholm, 11122")
    
    st.caption("💡 **Exempel:** Stockholm, Göteborg, Storgatan 15 Stockholm, Drottninggatan 10 Malmö, 55318, 11122")
    
    return city_input

def render_filter_section(search_performed=False):
    """Render the filter section with collapsible behavior"""
    from config import DEFAULT_ELECTRICITY_PRICE
    
    # Filter expander - open initially, closes after search
    with st.expander("⚙️ Filter Settings", expanded=not search_performed):
        if not search_performed:
            st.info("💡 **Tips:** Ställ in dina filter redan nu! De kommer appliceras automatiskt när du söker efter en plats.")
        else:
            st.info("💡 **Tips:** Ändra filter och sök igen för att uppdatera resultaten.")

        # Row 1: Basic parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            price_per_kwh = st.number_input("Electricity Price (kr/kWh)", 
                                          min_value=0.0, step=0.1, 
                                          value=DEFAULT_ELECTRICITY_PRICE)
        with col2:
            min_score = st.slider("Minimum Solar Score", 0, 100, 
                                FILTER_DEFAULTS['min_score'])
        with col3:
            ev_filter = st.checkbox("Endast hus med elbil", value=False)

        # Row 2: Economic filters
        st.subheader("💰 Ekonomiska Filter")
        col4, col5 = st.columns(2)
        with col4:
            income_range = st.slider("Hushållsinkomst (kSEK)", 200, 2000, 
                                   FILTER_DEFAULTS['income_range'])
        with col5:
            usage_range = st.slider("Nuvarande förbrukning (kWh/år)", 5000, 35000, 
                                  FILTER_DEFAULTS['usage_range'])

        # Row 3: Data Quality Settings
        st.subheader("🎯 Datakvalitet")
        col_geo1, col_geo2 = st.columns(2)
        with col_geo1:
            num_to_geocode = st.slider(
                "Antal hushåll med riktiga adresser", 
                min_value=10, max_value=150, value=FILTER_DEFAULTS['num_to_geocode'], step=5,
                help="Fler adresser = bättre kvalitet men långsammare. Resterande får genererade adresser."
            )
        with col_geo2:
            st.info("💡 **Tips:** 75 adresser ≈ 30 sekunder, 150 adresser ≈ 60 sekunder")

        # Row 4: Household filters
        st.subheader("👥 Hushålls Filter")
        col6, col7, col8 = st.columns(3)
        with col6:
            household_size_filter = st.multiselect("Hushållsstorlek", 
                                                  FILTER_DEFAULTS['household_sizes'], 
                                                  default=FILTER_DEFAULTS['household_sizes'])
        with col7:
            owner_age_range = st.slider("Ägarens ålder", 25, 80, 
                                      FILTER_DEFAULTS['owner_age_range'])
        with col8:
            roof_age_range = st.slider("Takets ålder (år)", 0, 50, 
                                     FILTER_DEFAULTS['roof_age_range'])
    
    # Return filter values as dictionary
    return {
        'price_per_kwh': price_per_kwh,
        'min_score': min_score,
        'ev_filter': ev_filter,
        'income_range': income_range,
        'usage_range': usage_range,
        'household_size_filter': household_size_filter,
        'owner_age_range': owner_age_range,
        'roof_age_range': roof_age_range,
        'num_to_geocode': num_to_geocode
    }

def render_map(filtered_df, lat, lon):
    """Render the pydeck map"""
    st.subheader("🗺️ Karta över tak")
    
    if filtered_df.empty:
        st.warning("Inga tak matchade filtren.")
        return
    
    # Ensure we have coordinates
    if 'lon' not in filtered_df.columns or 'lat' not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        if hasattr(filtered_df, 'geometry'):
            filtered_df['lon'] = filtered_df.geometry.centroid.x
            filtered_df['lat'] = filtered_df.geometry.centroid.y
        else:
            filtered_df['lon'] = lon
            filtered_df['lat'] = lat
    
    # Limit map points for performance
    map_data = filtered_df.head(400) if len(filtered_df) > 400 else filtered_df
    
    # Color calculation
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
    
    if len(filtered_df) > 400:
        st.caption(f"Visar första 400 av {len(filtered_df)} byggnader på kartan för prestanda")

def render_results_table(filtered_df):
    """Render the results table in an expandable section with enhanced property information"""
    with st.expander(f"📊 Alla {len(filtered_df)} filtrerade byggnader", expanded=False):
        # Enhanced columns including property information from Lantmäteriet, Sentinel-5P, and E.ON data
        cols_to_show = [
            'address', 'postal_code', 'municipality', 'property_type', 'construction_year',
            'living_area_sqm', 'floors', 'rooms', 'roof_type', 'energy_class', 
            'estimated_value_sek', 'area_in_meters', 'slope', 'aspect', 'shadow_index',
            'no2_concentration', 'atmospheric_efficiency', 'solar_score', 'kWp', 
            'kWh/year', 'Savings (kSEK/year)', 'estimated_capacity_kw',
            'usable_roof_area_sqm', 'suitability_score', 'roof_condition', 
            'electricity_region', 'eon_consumer_price_sek_kwh', 'eon_feed_in_tariff_sek_kwh',
            'eon_annual_consumption_kwh', 'eon_installation_cost_sek', 'eon_annual_savings_sek',
            'eon_payback_years', 'eon_smart_grid_benefits_sek', 'eon_financing_available',
            'has_ev', 'income_ksek', 'household_size', 'roof_age_years', 'owner_age', 
            'current_usage_kwh'
        ]
        
        # Only show columns that exist
        available_cols = [col for col in cols_to_show if col in filtered_df.columns]
        st.dataframe(filtered_df[available_cols], width='stretch')
        
        # Add property information summary if available
        if 'property_type' in filtered_df.columns:
            st.subheader("🏠 Fastighetsinformation Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Villa/Småhus", 
                         len(filtered_df[filtered_df['property_type'] == 'Villa/Småhus']))
                st.metric("Flerbostadshus", 
                         len(filtered_df[filtered_df['property_type'] == 'Flerbostadshus']))
            
            with col2:
                if 'construction_year' in filtered_df.columns:
                    avg_year = filtered_df['construction_year'].mean()
                    st.metric("Genomsnittlig byggår", f"{avg_year:.0f}")
                if 'living_area_sqm' in filtered_df.columns:
                    avg_area = filtered_df['living_area_sqm'].mean()
                    st.metric("Genomsnittlig boyta", f"{avg_area:.0f} m²")
            
            with col3:
                if 'estimated_value_sek' in filtered_df.columns:
                    avg_value = filtered_df['estimated_value_sek'].mean() / 1000000
                    st.metric("Genomsnittligt värde", f"{avg_value:.1f} MSEK")
                if 'estimated_capacity_kw' in filtered_df.columns:
                    avg_capacity = filtered_df['estimated_capacity_kw'].mean()
                    st.metric("Genomsnittlig solcellskapacitet", f"{avg_capacity:.1f} kW")
            
            with col4:
                if 'suitability_score' in filtered_df.columns:
                    avg_suitability = filtered_df['suitability_score'].mean()
                    st.metric("Genomsnittlig lämplighet", f"{avg_suitability:.2f}")
                if 'roof_condition' in filtered_df.columns:
                    good_roofs = len(filtered_df[filtered_df['roof_condition'] == 'Good'])
                    st.metric("Tak i gott skick", good_roofs)
        
        # E.ON Energy Summary if available
        if 'eon_consumer_price_sek_kwh' in filtered_df.columns:
            st.subheader("🔌 E.ON Energianalys Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price = filtered_df['eon_consumer_price_sek_kwh'].mean()
                st.metric("Genomsnittligt elpris", f"{avg_price:.2f} SEK/kWh")
                
            with col2:
                if 'eon_annual_consumption_kwh' in filtered_df.columns:
                    avg_consumption = filtered_df['eon_annual_consumption_kwh'].mean()
                    st.metric("Genomsnittlig förbrukning", f"{avg_consumption:.0f} kWh/år")
                    
            with col3:
                if 'eon_payback_years' in filtered_df.columns:
                    avg_payback = filtered_df['eon_payback_years'].mean()
                    st.metric("Genomsnittlig återbetalningstid", f"{avg_payback:.1f} år")
                    
            with col4:
                if 'eon_smart_grid_benefits_sek' in filtered_df.columns:
                    avg_smart_benefits = filtered_df['eon_smart_grid_benefits_sek'].mean()
                    st.metric("Smart Grid fördelar", f"{avg_smart_benefits:.0f} SEK/år")

def render_top_leads(filtered_df):
    """Render top 5 solar leads with enhanced property information"""
    st.subheader("🏆 Topp 5 Solar Leads")
    top_leads = filtered_df.sort_values(by="solar_score", ascending=False).head(5)
    
    if len(top_leads) == 0:
        st.info("Inga byggnader att visa med de valda filtren.")
        return
    
    # Show enhanced top leads with property information
    for idx, (_, lead) in enumerate(top_leads.iterrows(), 1):
        # Enhanced header with property type and construction year
        property_info = ""
        if 'property_type' in lead and lead['property_type']:
            property_info += f" ({lead['property_type']}"
            if 'construction_year' in lead and lead['construction_year']:
                property_info += f", {lead['construction_year']:.0f}"
            property_info += ")"
        
        with st.expander(f"#{idx} - {lead.get('address', 'Unknown Address')}{property_info} - Score: {lead['solar_score']:.0f}/100", 
                        expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**🏠 Fastighet:**")
                st.write(f"📍 {lead.get('address', 'N/A')}")
                st.write(f"📮 {lead.get('postal_code', 'N/A')}")
                st.write(f"🏘️ {lead.get('municipality', 'N/A')}")
                if 'living_area_sqm' in lead and lead['living_area_sqm']:
                    st.write(f"📐 Boyta: {lead['living_area_sqm']:.0f} m²")
                if 'floors' in lead and lead['floors']:
                    st.write(f"🏗️ Våningar: {lead['floors']}")
            
            with col2:
                st.write("**☀️ Solar:**")
                st.write(f"📐 Score: {lead['solar_score']:.0f}/100")
                st.write(f"⚡ Potential: {lead['kWp']:.1f} kWp")
                st.write(f"💰 Besparing: {lead['Savings (kSEK/year)']:.1f} kSEK/år")
                if 'estimated_capacity_kw' in lead and lead['estimated_capacity_kw']:
                    st.write(f"🔋 Kapacitet: {lead['estimated_capacity_kw']:.1f} kW")
                if 'usable_roof_area_sqm' in lead and lead['usable_roof_area_sqm']:
                    st.write(f"🏠 Takyta: {lead['usable_roof_area_sqm']:.0f} m²")
            
            with col3:
                st.write("**👥 Hushåll:**")
                st.write(f"💰 Inkomst: {lead.get('income_ksek', 0):.0f} kSEK")
                st.write(f"👥 Storlek: {lead.get('household_size', 0)} personer")
                st.write(f"🚗 EV: {'Ja' if lead.get('has_ev', False) else 'Nej'}")
                if 'owner_age' in lead and lead['owner_age']:
                    st.write(f"👤 Ålder: {lead['owner_age']:.0f} år")
            
            with col4:
                st.write("**🔌 E.ON Energi:**")
                if 'electricity_region' in lead and lead['electricity_region']:
                    st.write(f"📍 Region: {lead['electricity_region']}")
                if 'eon_consumer_price_sek_kwh' in lead and lead['eon_consumer_price_sek_kwh']:
                    st.write(f"⚡ Elpris: {lead['eon_consumer_price_sek_kwh']:.2f} SEK/kWh")
                if 'eon_payback_years' in lead and lead['eon_payback_years']:
                    st.write(f"� Återbetalningstid: {lead['eon_payback_years']:.1f} år")
                if 'eon_financing_available' in lead and lead['eon_financing_available']:
                    st.write(f"💳 Finansiering: {lead['eon_financing_available']}")
                if 'eon_smart_grid_benefits_sek' in lead and lead['eon_smart_grid_benefits_sek']:
                    st.write(f"🏠 Smart Grid: {lead['eon_smart_grid_benefits_sek']:.0f} SEK/år")
    
    # Enhanced summary table with property and E.ON energy information
    with st.expander("📋 Detaljerad Sammanfattningstabell", expanded=False):
        display_cols = [
            'address', 'postal_code', 'property_type', 'construction_year',
            'living_area_sqm', 'solar_score', 'kWp', 'estimated_capacity_kw',
            'Savings (kSEK/year)', 'electricity_region', 'eon_consumer_price_sek_kwh',
            'eon_payback_years', 'eon_smart_grid_benefits_sek', 'income_ksek', 
            'has_ev', 'roof_condition'
        ]
        summary_cols = [col for col in display_cols if col in top_leads.columns]
        st.table(top_leads[summary_cols])

def render_download_section(filtered_df, city_name):
    """Render download section with enhanced property data"""
    with st.expander("💾 Ladda ner data", expanded=False):
        # Enhanced columns including Lantmäteriet, Sentinel-5P atmospheric data, and E.ON energy data
        cols_to_show = [
            'address', 'postal_code', 'municipality', 'property_type', 'construction_year',
            'living_area_sqm', 'plot_size_sqm', 'floors', 'rooms', 'heating_type', 
            'roof_type', 'energy_class', 'estimated_value_sek', 'area_in_meters', 
            'slope', 'aspect', 'shadow_index', 'no2_concentration', 'atmospheric_efficiency',
            'solar_score', 'kWp', 'kWh/year', 'Savings (kSEK/year)', 'roof_area_sqm', 
            'usable_roof_area_sqm', 'estimated_panels', 'estimated_capacity_kw', 
            'suitability_score', 'roof_condition', 'annual_property_tax_sek', 
            'renovation_recommendations', 'electricity_region', 'eon_consumer_price_sek_kwh',
            'eon_feed_in_tariff_sek_kwh', 'eon_annual_consumption_kwh', 'eon_peak_demand_kw',
            'eon_heating_type', 'eon_installation_cost_sek', 'eon_annual_savings_sek',
            'eon_payback_years', 'eon_smart_grid_benefits_sek', 'eon_financing_available',
            'eon_recommendations', 'has_ev', 'income_ksek', 'household_size', 
            'roof_age_years', 'owner_age', 'current_usage_kwh'
        ]
        
        available_cols = [col for col in cols_to_show if col in filtered_df.columns]
        csv = filtered_df[available_cols].to_csv(index=False)
        filename = f"{city_name.replace(' ', '_')}_solar_buildings_enhanced_property_data.csv"
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            st.download_button("Ladda ner komplett data som CSV", csv, filename, "text/csv")
            st.caption(f"Innehåller {len(filtered_df)} byggnader med all fastighetsinformation")
        
        with col_dl2:
            # Option for simplified download
            simple_cols = ['address', 'postal_code', 'municipality', 'property_type', 
                          'solar_score', 'kWp', 'Savings (kSEK/year)', 'income_ksek', 
                          'has_ev', 'estimated_value_sek']
            simple_available = [col for col in simple_cols if col in filtered_df.columns]
            simple_csv = filtered_df[simple_available].to_csv(index=False)
            simple_filename = f"{city_name.replace(' ', '_')}_solar_leads_summary.csv"
            
            st.download_button("Ladda ner sammanfattning som CSV", simple_csv, simple_filename, "text/csv")
            st.caption(f"Förenklad sammanfattning med {len(simple_available)} kolumner")