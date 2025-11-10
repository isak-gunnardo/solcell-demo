"""
Solar potential calculations and energy modeling
"""

import numpy as np
import pandas as pd

def calculate_solar_potential(buildings_df, price_per_kwh=1.5):
    """
    Calculate solar potential for buildings using vectorized operations
    """
    # Safe column access for aspects and slopes
    if 'aspect' in buildings_df.columns:
        aspects = buildings_df['aspect']
    elif 'mean_1' in buildings_df.columns:
        aspects = buildings_df['mean_1']
    else:
        aspects = pd.Series([180] * len(buildings_df))
    
    if 'slope' in buildings_df.columns:
        slopes = buildings_df['slope']
    elif 'mean' in buildings_df.columns:
        slopes = buildings_df['mean']
    else:
        slopes = pd.Series([25] * len(buildings_df))
    
    # Vectorized solar score calculation
    aspect_scores = 100 * (1 - np.abs(aspects - 180) / 180)  # South-facing = 100
    slope_scores = 100 * (1 - np.abs(slopes - 30) / 30)      # 30° optimal
    slope_scores = np.clip(slope_scores, 0, 100)
    
    # Shadow penalty
    shadow_penalty = 0
    if 'shadow_index' in buildings_df.columns:
        shadow_penalty = buildings_df['shadow_index'] * 50
    
    # Combined solar score (0-100)
    solar_scores = (aspect_scores * 0.4 + slope_scores * 0.4 + (100 - shadow_penalty) * 0.2)
    solar_scores = np.clip(solar_scores, 0, 100)
    
    # Solar panel capacity (kWp) - based on roof area
    areas = buildings_df['area_in_meters']
    panel_efficiency = 0.15  # 15% efficient panels
    usable_roof_fraction = 0.7  # 70% of roof usable
    kWp = (areas * usable_roof_fraction * panel_efficiency) / 1000
    
    # Annual energy production (kWh/year)
    sweden_solar_hours = 950  # Average solar hours in Sweden
    performance_ratio = solar_scores / 100 * 0.8  # Performance based on solar score
    annual_kwh = kWp * sweden_solar_hours * performance_ratio
    
    # Economic calculations
    annual_savings_sek = annual_kwh * price_per_kwh
    annual_savings_ksek = annual_savings_sek / 1000
    
    # Add calculated columns
    buildings_df = buildings_df.copy()
    buildings_df['solar_score'] = solar_scores.round().astype(int)
    buildings_df['kWp'] = kWp.round(1)
    buildings_df['kWh/year'] = annual_kwh.round().astype(int)
    buildings_df['Savings (kSEK/year)'] = annual_savings_ksek.round(1)
    
    return buildings_df

def filter_buildings(buildings_df, filters):
    """
    Apply filters to buildings dataframe
    """
    mask = pd.Series([True] * len(buildings_df))
    
    # Solar score filter
    if 'min_score' in filters:
        mask &= buildings_df['solar_score'] >= filters['min_score']
    
    # EV filter
    if filters.get('ev_filter', False) and 'has_ev' in buildings_df.columns:
        mask &= buildings_df['has_ev'] == True
    
    # Income range filter
    if 'income_range' in filters and 'income_ksek' in buildings_df.columns:
        mask &= (buildings_df['income_ksek'] >= filters['income_range'][0]) & \
                (buildings_df['income_ksek'] <= filters['income_range'][1])
    
    # Household size filter
    if 'household_size_filter' in filters and 'household_size' in buildings_df.columns:
        if filters['household_size_filter']:  # If not empty
            mask &= buildings_df['household_size'].isin(filters['household_size_filter'])
    
    # Owner age filter
    if 'owner_age_range' in filters and 'owner_age' in buildings_df.columns:
        mask &= (buildings_df['owner_age'] >= filters['owner_age_range'][0]) & \
                (buildings_df['owner_age'] <= filters['owner_age_range'][1])
    
    # Roof age filter
    if 'roof_age_range' in filters and 'roof_age_years' in buildings_df.columns:
        mask &= (buildings_df['roof_age_years'] >= filters['roof_age_range'][0]) & \
                (buildings_df['roof_age_years'] <= filters['roof_age_range'][1])
    
    # Usage range filter
    if 'usage_range' in filters and 'current_usage_kwh' in buildings_df.columns:
        mask &= (buildings_df['current_usage_kwh'] >= filters['usage_range'][0]) & \
                (buildings_df['current_usage_kwh'] <= filters['usage_range'][1])
    
    return buildings_df[mask]