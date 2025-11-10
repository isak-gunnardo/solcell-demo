"""
Earth Engine integration and building detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from config import EE_PROJECT_ID

try:
    import ee
    import geopandas as gpd
    from shapely.geometry import Point
    EE_AVAILABLE = True
    GPD_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    GPD_AVAILABLE = False

@st.cache_resource
def initialize_ee_cached():
    """Cached Earth Engine initialization"""
    if not EE_AVAILABLE:
        return False
    try:
        ee.Initialize(project=EE_PROJECT_ID)
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        return False

def initialize_ee():
    """Initialize Earth Engine with caching"""
    return initialize_ee_cached()

@st.cache_data(ttl=3600)  # Cache for 1 hour with location-specific key
def get_building_data_from_ee(lat, lon, city_name):
    """
    Get real building data from Earth Engine
    Returns: GeoDataFrame with building polygons and solar data
    """
    if not EE_AVAILABLE:
        return None
    
    try:
        # Create region around the location - balanced buffer for 500 buildings
        # Use adaptive buffer based on city type (increased for more buildings)
        buffer_size = 3500  # Increased from 2500m to 3.5km to find 500 buildings
        region = ee.Geometry.Point([lon, lat]).buffer(buffer_size)
        
        # Use Google Open Buildings dataset (publicly available)
        buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v1/polygons")
        buildings = buildings.filterBounds(region).limit(2000)  # Increased to have more candidates for 500 final buildings
        
        # Sentinel-5P atmospheric data for air quality and environmental analysis
        try:
            # NO2 (Nitrogen Dioxide) - good indicator for urban pollution and solar efficiency
            s5p_no2 = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2") \
                .filterBounds(region) \
                .filterDate('2023-05-01', '2023-09-30') \
                .select('NO2_column_number_density') \
                .mean()
            
            # SO2 (Sulfur Dioxide) - another air quality indicator
            s5p_so2 = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_SO2") \
                .filterBounds(region) \
                .filterDate('2023-05-01', '2023-09-30') \
                .select('SO2_column_number_density') \
                .mean()
            
            # UV Aerosol Index - affects solar radiation
            s5p_uv = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_AER_AI") \
                .filterBounds(region) \
                .filterDate('2023-05-01', '2023-09-30') \
                .select('absorbing_aerosol_index') \
                .mean()
                
        except Exception as e:
            print(f"Warning: Sentinel-5P data access failed: {e}")
            # Create dummy atmospheric data if S5P unavailable
            s5p_no2 = ee.Image.constant(0.00005)  # Low pollution baseline
            s5p_so2 = ee.Image.constant(0.0001)   # Low SO2 baseline
            s5p_uv = ee.Image.constant(0.5)       # Moderate UV baseline
        
        # Load DEM and calculate slope/aspect
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM').clip(region)
        terrain = ee.Terrain.products(dem)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')
        
        # Air quality based shadow/efficiency analysis using Sentinel-5P
        # Higher NO2 and aerosols reduce solar efficiency
        air_quality_factor = s5p_no2.multiply(1000000).add(s5p_uv.multiply(0.1))
        atmospheric_efficiency = ee.Image.constant(1).subtract(air_quality_factor.divide(100))
        
        # Sample terrain data for each building
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
        
        # Sample atmospheric data for each building
        buildingsWithNO2 = s5p_no2.reduceRegions(
            collection=buildingsWithAspect,
            reducer=ee.Reducer.mean(),
            scale=1000  # Sentinel-5P has ~7km resolution, use 1km sampling
        )
        
        buildingsWithAtmosphere = atmospheric_efficiency.reduceRegions(
            collection=buildingsWithNO2,
            reducer=ee.Reducer.mean(),
            scale=1000
        )
        
        # Add distance from center for prioritization
        center_point = ee.Geometry.Point([lon, lat])
        buildingsWithDistance = buildingsWithAtmosphere.map(
            lambda feature: feature.set('distance_from_center', 
                feature.geometry().centroid().distance(center_point))
        )
        
        # Sort by distance and limit results for performance - prioritize city center
        final_buildings = buildingsWithDistance.sort('distance_from_center').limit(500)
        
        # Fetch the actual data
        geojson = final_buildings.getInfo()
        
        if not geojson.get('features'):
            return None
        
        # Convert to GeoDataFrame
        if GPD_AVAILABLE:
            rooftops = gpd.GeoDataFrame.from_features(geojson['features'])
            
            # Calculate building area from geometry
            rooftops['area_in_meters'] = (rooftops.geometry.area * 111000 * 111000).round().astype(int)
            
            # Add coordinates for mapping
            rooftops['lon'] = rooftops.geometry.centroid.x
            rooftops['lat'] = rooftops.geometry.centroid.y
        else:
            # Fallback without geopandas
            features_data = []
            for feature in geojson['features']:
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                coords = geom.get('coordinates', [])
                if coords and len(coords) > 0 and len(coords[0]) > 0:
                    # Simple centroid calculation for polygon
                    lons = [coord[0] for coord in coords[0]]
                    lats = [coord[1] for coord in coords[0]]
                    props['lon'] = sum(lons) / len(lons)
                    props['lat'] = sum(lats) / len(lats)
                    props['area_in_meters'] = 200  # Default area
                features_data.append(props)
            rooftops = pd.DataFrame(features_data)
        
        # Process terrain and atmospheric data from Sentinel-5P
        if 'mean' in rooftops.columns:
            rooftops['slope'] = rooftops['mean'].round().astype(int)
        if 'mean_1' in rooftops.columns:
            rooftops['aspect'] = rooftops['mean_1'].round().astype(int)
        if 'mean_2' in rooftops.columns:
            # NO2 concentration (convert from mol/m² to readable scale)
            rooftops['no2_concentration'] = (rooftops['mean_2'] * 1000000).round(4)
        if 'mean_3' in rooftops.columns:
            # Atmospheric efficiency factor (0-1 scale)
            rooftops['atmospheric_efficiency'] = rooftops['mean_3'].round(3)
        
        # Create shadow index equivalent from atmospheric data
        # Higher pollution = more atmospheric interference = lower solar efficiency
        if 'atmospheric_efficiency' in rooftops.columns:
            rooftops['shadow_index'] = (1 - rooftops['atmospheric_efficiency']).round(2)
        else:
            rooftops['shadow_index'] = 0.1  # Default low atmospheric interference
        
        return rooftops
        
    except Exception as e:
        print(f"Earth Engine building detection failed: {e}")
        return None

def generate_mock_building_data(lat, lon, num_buildings=500):
    """
    Generate mock building data when Earth Engine is not available
    Focus on city center with weighted distribution
    """
    center_lat, center_lon = lat, lon
    
    # Create weighted distribution - more buildings closer to center
    distances = np.random.exponential(scale=0.008, size=num_buildings)  # Exponential dist
    distances = np.clip(distances, 0, 0.015)  # Reduced max distance for city focus
    angles = np.random.uniform(0, 2*np.pi, num_buildings)
    
    lat_offset = distances * np.cos(angles)
    lon_offset = distances * np.sin(angles)
    
    rooftops = pd.DataFrame({
        'lat': center_lat + lat_offset,
        'lon': center_lon + lon_offset,
        'area_in_meters': np.random.randint(50, 500, size=num_buildings),
        'aspect': np.random.randint(120, 240, size=num_buildings),
        'slope': np.random.randint(10, 40, size=num_buildings),
        'no2_concentration': np.random.uniform(0.05, 0.15, size=num_buildings),  # Mock NO2 levels
        'atmospheric_efficiency': np.random.uniform(0.85, 0.95, size=num_buildings),  # Mock efficiency
        'shadow_index': np.random.uniform(0, 0.3, size=num_buildings)
    })
    
    # Add geometry column for consistency if geopandas is available
    if GPD_AVAILABLE:
        geometry = [Point(lon, lat) for lon, lat in zip(rooftops['lon'], rooftops['lat'])]
        rooftops = gpd.GeoDataFrame(rooftops, geometry=geometry)
    
    return rooftops