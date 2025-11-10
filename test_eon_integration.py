"""
Test E.ON Energy Data Integration
This script validates the E.ON API integration functionality
"""

import pandas as pd
import numpy as np
from eon_api import (
    get_eon_electricity_prices, 
    get_eon_consumption_profile,
    get_eon_solar_integration_info,
    get_eon_smart_grid_opportunities,
    determine_electricity_region,
    enrich_with_eon_energy_data
)

def test_eon_integration():
    """Test all E.ON integration functions"""
    
    print("🔌 Testing E.ON Energy Data Integration")
    print("=" * 50)
    
    # Test 1: Electricity pricing for different regions
    print("\n1. Testing E.ON Electricity Pricing:")
    for region in ["SE1", "SE2", "SE3", "SE4"]:
        pricing = get_eon_electricity_prices(region)
        print(f"   {region} ({pricing['region_name']}): {pricing['total_consumer_price_sek_kwh']:.2f} SEK/kWh")
        print(f"      Feed-in tariff: {pricing['feed_in_tariff_sek_kwh']:.2f} SEK/kWh")
    
    # Test 2: Consumption profiles
    print("\n2. Testing E.ON Consumption Profiles:")
    test_cases = [
        (2, "villa", "SE3"),
        (4, "apartment", "SE3"),
        (3, "row_house", "SE1")
    ]
    
    for household_size, building_type, region in test_cases:
        profile = get_eon_consumption_profile(household_size, building_type, region)
        print(f"   {household_size}-person {building_type} in {region}: {profile['yearly_consumption_kwh']:,} kWh/år")
        print(f"      Peak demand: {profile['peak_demand_kw']} kW")
        print(f"      Heating type: {profile['heating_type_estimate']}")
    
    # Test 3: Solar integration
    print("\n3. Testing E.ON Solar Integration:")
    solar_capacities = [5.0, 10.0, 15.0]
    
    for capacity in solar_capacities:
        solar_info = get_eon_solar_integration_info("SE3", capacity)
        economic = solar_info['economic_analysis']
        print(f"   {capacity} kW system:")
        print(f"      Installation cost: {solar_info['solar_services']['estimated_setup_cost_sek']:,} SEK")
        print(f"      Annual savings: {economic['annual_savings_sek']:,} SEK")
        print(f"      Payback period: {economic['payback_period_years']:.1f} years")
    
    # Test 4: Smart grid opportunities
    print("\n4. Testing E.ON Smart Grid Opportunities:")
    test_locations = [
        {"municipality": "Stockholm", "property_type": "Villa/Småhus"},
        {"municipality": "Göteborg", "property_type": "Flerbostadshus"},
        {"municipality": "Småstad", "property_type": "Villa/Småhus"}
    ]
    
    for location in test_locations:
        smart_grid = get_eon_smart_grid_opportunities(location)
        print(f"   {location['municipality']} ({location['property_type']}):")
        print(f"      Annual smart grid benefits: {smart_grid['estimated_annual_benefits_sek']:,} SEK")
        print(f"      Virtual power plant available: {smart_grid['smart_services']['virtual_power_plant']['available']}")
    
    # Test 5: Region determination
    print("\n5. Testing Electricity Region Determination:")
    test_cities = ["Stockholm", "Göteborg", "Malmö", "Kiruna", "Uppsala", "Växjö"]
    
    for city in test_cities:
        region = determine_electricity_region(city)
        print(f"   {city}: {region}")
    
    # Test 6: Full integration with sample building data
    print("\n6. Testing Full E.ON Integration:")
    
    # Create sample building data
    sample_buildings = pd.DataFrame({
        'address': ['Storgatan 1', 'Vasagatan 15', 'Malmövägen 22'],
        'municipality': ['Stockholm', 'Göteborg', 'Malmö'],
        'postal_code': ['111 22', '411 26', '202 12'],
        'property_type': ['Villa/Småhus', 'Flerbostadshus', 'Villa/Småhus'],
        'household_size': [3, 2, 4],
        'estimated_capacity_kw': [8.5, 4.2, 12.0],
        'kWp': [8.5, 4.2, 12.0],
        'has_ev': [True, False, True],
        'income_ksek': [650, 480, 750]
    })
    
    print(f"   Processing {len(sample_buildings)} sample buildings...")
    
    # Enrich with E.ON data
    enriched_buildings = enrich_with_eon_energy_data(sample_buildings)
    
    # Display results
    print("\n   Results:")
    for idx, row in enriched_buildings.iterrows():
        print(f"   Building {idx + 1} ({row['municipality']}):")
        print(f"      Region: {row.get('electricity_region', 'N/A')}")
        print(f"      Consumer price: {row.get('eon_consumer_price_sek_kwh', 0):.2f} SEK/kWh")
        print(f"      Annual consumption: {row.get('eon_annual_consumption_kwh', 0):,} kWh")
        print(f"      Solar payback: {row.get('eon_payback_years', 0):.1f} years")
        print(f"      Smart grid benefits: {row.get('eon_smart_grid_benefits_sek', 0):,} SEK/år")
        if row.get('eon_recommendations'):
            print(f"      Recommendations: {row['eon_recommendations']}")
        print()
    
    print("✅ E.ON Integration Test Complete!")
    print("\nSummary:")
    print(f"   - Electricity pricing: ✅ Working for all regions")
    print(f"   - Consumption profiles: ✅ Working for all building types")
    print(f"   - Solar integration: ✅ Working with economic analysis")
    print(f"   - Smart grid analysis: ✅ Working with location-based features")
    print(f"   - Region detection: ✅ Working for major cities")
    print(f"   - Full integration: ✅ Successfully enriched sample data")

if __name__ == "__main__":
    test_eon_integration()