from flask import Flask, jsonify, send_file
from flask_cors import CORS
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import requests
import time
import json
import os
from folium.plugins import HeatMap
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
from shapely.geometry import Point
from tqdm import tqdm
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----- Configuration ----- #
# 📂 File paths
metadata_path = "../metadata.csv"  # Use metadata.csv from parent directory
geojson_path = "content/india.geojson"
cache_dir = "cache"
cache_file = os.path.join(cache_dir, "aqi_data_cache.json")
cache_expiry_hours = 5  # Cache expires after 5 hours

# 🔑 OpenWeather API Key from environment
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"

# ----- Cache Management Functions ----- #
def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"📁 Created cache directory: {cache_dir}")

def load_cache():
    """Load cached AQI data if available and not expired"""
    ensure_cache_dir()
    
    if not os.path.exists(cache_file):
        print("❓ No cache file found")
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Check cache timestamp
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        current_time = datetime.now()
        
        # Calculate time difference
        time_diff = current_time - cache_time
        
        if time_diff < timedelta(hours=cache_expiry_hours):
            print(f"✅ Using cached data from {cache_time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(expires in {cache_expiry_hours - time_diff.seconds/3600:.1f} hours)")
            return pd.DataFrame(cache_data['data'])
        else:
            print(f"⚠️ Cache expired (from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                  f"age: {time_diff.seconds/3600:.1f} hours)")
            return None
    except Exception as e:
        print(f"❌ Error loading cache: {str(e)}")
        return None

def save_cache(aqi_data_df):
    """Save AQI data to cache"""
    ensure_cache_dir()
    
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': aqi_data_df.to_dict('records')
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"💾 Saved data to cache: {cache_file}")
    except Exception as e:
        print(f"❌ Error saving to cache: {str(e)}")

# ----- AQI color mapping ----- #
def get_aqi_color(aqi):
    """Return color based on AQI value"""
    try: aqi = float(aqi)
    except: return 'gray'
    if aqi <= 50: return '#00e400'  # Good
    elif aqi <= 100: return '#ffff00'  # Moderate
    elif aqi <= 150: return '#ff7e00'  # Unhealthy for Sensitive Groups
    elif aqi <= 200: return '#ff0000'  # Unhealthy
    elif aqi <= 300: return '#99004c'  # Very Unhealthy
    else: return '#7e0023'  # Hazardous

# ----- AQI Calculation Functions ----- #
def map_aqi(concentration, c_low, c_high, i_low, i_high):
    """Maps concentration to AQI scale"""
    return round(((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low)

def calculate_pm25_aqi(concentration):
    """Calculate AQI for PM2.5"""
    if concentration <= 12: return map_aqi(concentration, 0, 12, 0, 50)
    if concentration <= 35.4: return map_aqi(concentration, 12.1, 35.4, 51, 100)
    if concentration <= 55.4: return map_aqi(concentration, 35.5, 55.4, 101, 150)
    if concentration <= 150.4: return map_aqi(concentration, 55.5, 150.4, 151, 200)
    if concentration <= 250.4: return map_aqi(concentration, 150.5, 250.4, 201, 300)
    if concentration <= 500.4: return map_aqi(concentration, 250.5, 500.4, 301, 500)
    return 500

def calculate_pm10_aqi(concentration):
    """Calculate AQI for PM10"""
    if concentration <= 54: return map_aqi(concentration, 0, 54, 0, 50)
    if concentration <= 154: return map_aqi(concentration, 55, 154, 51, 100)
    if concentration <= 254: return map_aqi(concentration, 155, 254, 101, 150)
    if concentration <= 354: return map_aqi(concentration, 255, 354, 151, 200)
    if concentration <= 424: return map_aqi(concentration, 355, 424, 201, 300)
    if concentration <= 604: return map_aqi(concentration, 425, 604, 301, 500)
    return 500

def calculate_o3_aqi(concentration):
    """Calculate AQI for O3"""
    ppb = concentration * 0.5  # Convert µg/m³ to ppb
    if ppb <= 54: return map_aqi(ppb, 0, 54, 0, 50)
    if ppb <= 70: return map_aqi(ppb, 55, 70, 51, 100)
    if ppb <= 85: return map_aqi(ppb, 71, 85, 101, 150)
    if ppb <= 105: return map_aqi(ppb, 86, 105, 151, 200)
    if ppb <= 200: return map_aqi(ppb, 106, 200, 201, 300)
    return 500

def calculate_no2_aqi(concentration):
    """Calculate AQI for NO2"""
    ppb = concentration * 0.53  # Convert µg/m³ to ppb
    if ppb <= 53: return map_aqi(ppb, 0, 53, 0, 50)
    if ppb <= 100: return map_aqi(ppb, 54, 100, 51, 100)
    if ppb <= 360: return map_aqi(ppb, 101, 360, 101, 150)
    if ppb <= 649: return map_aqi(ppb, 361, 649, 151, 200)
    if ppb <= 1249: return map_aqi(ppb, 650, 1249, 201, 300)
    if ppb <= 2049: return map_aqi(ppb, 1250, 2049, 301, 500)
    return 500

def calculate_so2_aqi(concentration):
    """Calculate AQI for SO2"""
    ppb = concentration * 0.38  # Convert µg/m³ to ppb
    if ppb <= 35: return map_aqi(ppb, 0, 35, 0, 50)
    if ppb <= 75: return map_aqi(ppb, 36, 75, 51, 100)
    if ppb <= 185: return map_aqi(ppb, 76, 185, 101, 150)
    if ppb <= 304: return map_aqi(ppb, 186, 304, 151, 200)
    if ppb <= 604: return map_aqi(ppb, 305, 604, 201, 300)
    if ppb <= 1004: return map_aqi(ppb, 605, 1004, 301, 500)
    return 500

def calculate_co_aqi(concentration):
    """Calculate AQI for CO"""
    ppm = concentration * 0.000873  # Convert µg/m³ to ppm
    if ppm <= 4.4: return map_aqi(ppm, 0, 4.4, 0, 50)
    if ppm <= 9.4: return map_aqi(ppm, 4.5, 9.4, 51, 100)
    if ppm <= 12.4: return map_aqi(ppm, 9.5, 12.4, 101, 150)
    if ppm <= 15.4: return map_aqi(ppm, 12.4, 15.4, 151, 200)
    if ppm <= 30.4: return map_aqi(ppm, 15.5, 30.4, 201, 300)
    if ppm <= 50.4: return map_aqi(ppm, 30.5, 50.4, 301, 500)
    return 500

def calculate_aqi(pollutants):
    """Calculate overall AQI from pollutant concentrations"""
    pm25_aqi = calculate_pm25_aqi(pollutants.get('pm2_5', 0))
    pm10_aqi = calculate_pm10_aqi(pollutants.get('pm10', 0))
    o3_aqi = calculate_o3_aqi(pollutants.get('o3', 0))
    no2_aqi = calculate_no2_aqi(pollutants.get('no2', 0))
    so2_aqi = calculate_so2_aqi(pollutants.get('so2', 0))
    co_aqi = calculate_co_aqi(pollutants.get('co', 0))
    
    # Return max AQI value among all pollutants
    return max(pm25_aqi, pm10_aqi, o3_aqi, no2_aqi, so2_aqi, co_aqi)
# ----- Fetch Real-time AQI Data Function ----- #
def fetch_aqi_data():
    """Fetch or load AQI data and return DataFrame"""
    # Check for cached data first
    latest_aqi = load_cache()
    
    # Fetch Real-time AQI Data if no valid cache
    if latest_aqi is None:
        print("🌐 Fetching fresh real-time air quality data from OpenWeather API in batches...")
        
        def process_station_batch(batch):
            """Process a batch of stations to get AQI data"""
            batch_data = []
            for _, station in batch.iterrows():
                lat, lon = station['latitude'], station['longitude']
                try:
                    # Build API URL
                    url = f"{BASE_URL}?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
                    
                    # Fetch data from OpenWeather API
                    response = requests.get(url)
                    data = response.json()
                    
                    # Extract pollutant data
                    if 'list' in data and len(data['list']) > 0:
                        pollutants = data['list'][0]['components']
                        timestamp = data['list'][0]['dt']
                        
                        # Calculate AQI from pollutant concentrations
                        aqi_value = calculate_aqi(pollutants)
                        
                        # Create record with pollutant data
                        record = {
                            'station_id': station['station_id'],
                            'station_name': station['name'],
                            'latitude': lat,
                            'longitude': lon,
                            'timestamp': timestamp,
                            'aqi_cal': aqi_value,  # You'll replace this with your calculate_aqi function
                            'pm25': pollutants.get('pm2_5', None),
                            'pm10': pollutants.get('pm10', None),
                            'o3': pollutants.get('o3', None),
                            'no2': pollutants.get('no2', None),
                            'so2': pollutants.get('so2', None),
                            'co': pollutants.get('co', None)
                        }
                        
                        batch_data.append(record)
                        print(f"✅ Fetched data for {station['name']}: AQI = {aqi_value}")
                    else:
                        print(f"⚠️ No data available for {station['name']}")
                    
                    # Optional: Add slight delay between requests in the batch
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"❌ Error fetching data for {station['name']}: {str(e)}")
            
            return batch_data

        # Load metadata
        metadata = pd.read_csv(metadata_path)
        
        # Process stations in batches
        aqi_data = []
        BATCH_SIZE = 100 
        total_stations = len(metadata)
        num_batches = (total_stations + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division

        for batch_num in range(num_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_stations)
            
            print(f"\n🔄 Processing batch {batch_num+1}/{num_batches} (stations {start_idx+1}-{end_idx})...")
            station_batch = metadata.iloc[start_idx:end_idx]
            
            # Process this batch
            batch_results = process_station_batch(station_batch)
            aqi_data.extend(batch_results)
            
            # Add delay between batches (except for the last batch)
            if batch_num < num_batches - 1:
                print(f"⏱️ Waiting between batches to avoid API rate limits...")
                time.sleep(2)  # 2 second delay between batches

        # Create DataFrame from collected data
        latest_aqi = pd.DataFrame(aqi_data)
        
        # Save to cache for future use
        save_cache(latest_aqi)
    
    return latest_aqi

# ----- Generate Map Function ----- #
def generate_aqi_map():
    """Generate AQI map using Folium"""
    # Get AQI data (cached or fresh)
    latest_aqi = fetch_aqi_data()
    
    # Load India GeoJSON boundaries
    gdf = gpd.read_file(geojson_path)
    # Use unary_union for compatibility with different geopandas versions
    india_boundary = gdf.geometry.unary_union

    # Extract coordinates and AQI values
    lons = latest_aqi['longitude'].values
    lats = latest_aqi['latitude'].values
    aqi_values = latest_aqi['aqi_cal'].astype(float).values

    # OPTIMIZATION: Reduced grid size
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    lon_grid = np.linspace(lon_min, lon_max, 50)
    lat_grid = np.linspace(lat_min, lat_max, 50)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Flatten grid and station points
    grid_points = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]).T
    station_points = np.vstack([lons, lats]).T

    # KDTree + IDW interpolation
    tree = cKDTree(station_points)
    distances, indices = tree.query(grid_points, k=3)
    power = 2
    weights = 1 / (distances ** power)
    weights[np.isinf(weights)] = 1e-10
    interpolated_values = np.sum(weights * aqi_values[indices], axis=1) / np.sum(weights, axis=1)

    # Simplify boundary for faster operations
    simplified_boundary = india_boundary.simplify(0.05)

    # First filter points by bounding box (much faster)
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    bbox_mask = ((grid_points[:, 0] >= lon_min) & 
                 (grid_points[:, 0] <= lon_max) & 
                 (grid_points[:, 1] >= lat_min) & 
                 (grid_points[:, 1] <= lat_max))
    bbox_indices = np.where(bbox_mask)[0]

    # Check if points are in simplified boundary
    def points_in_polygon(points, polygon):
        return [polygon.contains(Point(x, y)) for x, y in points]

    # Process in chunks for better performance
    valid_indices = []
    CHUNK_SIZE = 1000
    num_chunks = (len(bbox_indices) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Use progress bar for better tracking
    for i in tqdm(range(num_chunks), desc="Processing boundary check"):
        start_idx = i * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, len(bbox_indices))
        chunk_indices = bbox_indices[start_idx:end_idx]
        
        # Get points in this chunk
        chunk_points = grid_points[chunk_indices]
        
        # Check against simplified boundary
        in_poly = points_in_polygon(chunk_points, simplified_boundary)
        
        # Add valid indices
        valid_chunk = [chunk_indices[j] for j, is_in in enumerate(in_poly) if is_in]
        valid_indices.extend(valid_chunk)

    # Filter for non-NaN values
    valid_indices = [idx for idx in valid_indices if not np.isnan(interpolated_values[idx])]

    # Create base map with cleaner attribution
    m = folium.Map(
        location=[20.5937, 78.9629], 
        zoom_start=5, 
        tiles=None,  # Don't add default tiles
        attr='© OpenStreetMap contributors'
    )
    
    # Add CartoDB Positron as a named tile layer
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Base Map',
        attr='© OpenStreetMap contributors © CartoDB',
        overlay=False,
        control=True
    ).add_to(m)

    # Add title with data source info
    try:
        # Always try to convert timestamp to int first
        timestamp_val = int(latest_aqi['timestamp'].iloc[0])
        timestamp = datetime.fromtimestamp(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        # If conversion fails, try to parse as string or use current time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%); 
                width: 500px; height: 65px; 
                background-color: white; border-radius: 8px;
                border: 2px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                z-index: 9999; font-size: 20px;
                font-family: Arial, sans-serif; padding: 12px;
                text-align: center;">
        <b>India Real-time Air Quality Index</b><br>
        <span style="font-size: 14px; color: #666;">Last Updated: {timestamp}</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add custom CSS to hide unwanted elements and fix layer control
    custom_css = '''
    <style>
        /* Clean up layer control labels */
        .leaflet-control-layers-overlays label span {
            font-size: 14px;
            font-family: Arial, sans-serif;
        }
        
        /* Hide any technical IDs */
        .leaflet-control-layers label:contains("macro_element") {
            display: none !important;
        }
        
        /* Clean up attribution */
        .leaflet-control-attribution {
            background-color: rgba(255, 255, 255, 0.8) !important;
            font-size: 11px !important;
        }
        
        /* Style the layer control */
        .leaflet-control-layers {
            font-family: Arial, sans-serif;
            background: white;
            border-radius: 5px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.4);
        }
        
        /* Position layer control to avoid title */
        .leaflet-top.leaflet-right {
            top: 90px !important;
        }
    </style>
    '''
    m.get_root().html.add_child(folium.Element(custom_css))

# Add AQI Legend with horizontal gradient bar
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white; border-radius: 5px;
                font-family: 'Arial'; padding: 15px; width: 320px;">
        <div style="text-align: center; margin-bottom: 10px;"><b>Air Quality Index</b></div>
        
        <!-- Gradient Bar -->
        <div style="display: flex; height: 25px; border-radius: 4px; overflow: hidden; margin-bottom: 5px;">
            <div style="background-color: #00e400; flex: 1;"></div>
            <div style="background-color: #ffff00; flex: 1;"></div>
            <div style="background-color: #ff7e00; flex: 1;"></div>
            <div style="background-color: #ff0000; flex: 1;"></div>
            <div style="background-color: #99004c; flex: 1;"></div>
            <div style="background-color: #7e0023; flex: 1;"></div>
        </div>
        
        <!-- Labels -->
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <div>Good</div>
            <div>Moderate</div>
            <div>Poor</div>
            <div>Unhealthy</div>
            <div>Severe</div>
            <div>Hazardous</div>
        </div>
        
        <!-- Scale Numbers -->
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 3px;">
            <div>0</div>
            <div>50</div>
            <div>100</div>
            <div>150</div>
            <div>200</div>
            <div>300</div>
            <div>301+</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Sample for heatmap
    heatmap_step = max(1, len(valid_indices) // 3000)
    heatmap_indices = valid_indices[::heatmap_step]
    heatmap_data = [
        [grid_points[i, 1], grid_points[i, 0], float(interpolated_values[i])]
        for i in heatmap_indices
    ]

    # Create heatmap layer with proper name
    heatmap_layer = folium.FeatureGroup(name='AQI Heatmap', show=True)
    heatmap = folium.plugins.HeatMap(
        heatmap_data,
        radius=25,
        blur=20,
        max_zoom=6,
        gradient={
            0.0: '#00e400',
            0.5: '#ffff00',
            0.6: '#ff7e00',
            0.7: '#ff0000',
            0.8: '#99004c',
            1.0: '#7e0023'
        }
    )
    heatmap.add_to(heatmap_layer)
    heatmap_layer.add_to(m)

    # Create interpolated points layer
    interpolated_points = folium.FeatureGroup(name='Interpolated AQI Points', show=False)
    
    # Sample markers
    marker_step = max(1, len(valid_indices) // 500)
    for i in range(0, len(valid_indices), marker_step):
        p_idx = valid_indices[i]
        p = {
            'lat': grid_points[p_idx, 1], 
            'lon': grid_points[p_idx, 0], 
            'aqi': interpolated_values[p_idx]
        }
        
        folium.CircleMarker(
            location=[p['lat'], p['lon']],
            radius=2,
            color=get_aqi_color(float(p['aqi'])),
            fill=True,
            fill_color=get_aqi_color(float(p['aqi'])),
            fill_opacity=0.7,
            popup=folium.Popup(f"AQI: {float(p['aqi']):.1f}", parse_html=True)
        ).add_to(interpolated_points)
    
    interpolated_points.add_to(m)

    # Create toggle-able marker group for monitoring stations
    station_markers = folium.FeatureGroup(name="Monitoring Stations", show=True)

    # Pollutant units dictionary
    pollutant_units = {
        'pm25': 'μg/m³',
        'pm10': 'μg/m³',
        'so2': 'μg/m³',
        'no2': 'μg/m³',
        'o3': 'μg/m³',
        'co': 'mg/m³'
    }

    # Sample stations
    station_sample_step = max(1, len(latest_aqi) // 200)
    station_idx = 0

    for _, row in latest_aqi.iterrows():
        # Sample stations
        station_idx += 1
        if station_idx % station_sample_step != 0:
            continue
            
        lat, lon = row['latitude'], row['longitude']
        station_name, aqi = row['station_name'], row['aqi_cal']
        
        # Extract pollutants
        pollutant_values = {
            'pm25': row.get('pm25', np.nan),
            'pm10': row.get('pm10', np.nan),
            'o3': row.get('o3', np.nan),
            'no2': row.get('no2', np.nan),
            'so2': row.get('so2', np.nan),
            'co': row.get('co', np.nan)
        }
        
        # Filter out NaN values
        pollutant_values = {k: v for k, v in pollutant_values.items() if not pd.isna(v)}
        
        # Get top 2 pollutants
        top_2 = sorted(pollutant_values.items(), key=lambda x: x[1], reverse=True)[:2]
        
        top_pollutants_str = "<br>".join([
            f"{key.upper()}: {val:.2f} {pollutant_units.get(key, '')}" for key, val in top_2
        ])
        
        # Format timestamp as readable date
        if isinstance(row['timestamp'], (int, float)):
            date_str = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        else:
            date_str = str(row['timestamp'])
        
        popup_html = f"""
        <b>📍 {station_name}</b><br>
        <b>AQI:</b> {aqi:.2f}<br>
        <b>Timestamp:</b> {date_str}<br>
        <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})<br>
        <b>Top Pollutants:</b><br>{top_pollutants_str}
        """
        
        # Add marker with appropriate color based on AQI
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(
                color="white", 
                icon_color=get_aqi_color(aqi),
                icon="cloud", 
                prefix="fa"
            ),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(station_markers)

    # Add toggle-able layer to map
    station_markers.add_to(m)

    # Add simplified state boundaries
    folium.GeoJson(
        gdf.simplify(0.01),
        name='State Boundaries', 
        style_function=lambda x: {
            'fillColor': 'none', 
            'color': 'black', 
            'weight': 1
        }
    ).add_to(m)

    # Add Layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Return the map object
    return m

# ----- API Routes ----- #
@app.route('/api/aqi-map', methods=['GET'])
def get_aqi_map():
    """Generate and return AQI map as HTML"""
    try:
        # Generate the map
        m = generate_aqi_map()
        
        # Save map to memory
        map_data = io.BytesIO()
        m.save(map_data, close_file=False)
        map_data.seek(0)
        
        # Return the map as HTML content
        return send_file(map_data, mimetype='text/html')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/aqi-data', methods=['GET'])
def get_aqi_data():
    """Return AQI data as JSON for frontend use"""
    try:
        # Get AQI data
        aqi_data = fetch_aqi_data()
        
        # Convert to list of dictionaries for JSON response
        if aqi_data is not None:
            # Get basic statistics
            avg_aqi = aqi_data['aqi_cal'].mean()
            max_aqi = aqi_data['aqi_cal'].max()
            min_aqi = aqi_data['aqi_cal'].min()
            
            # Count stations in each AQI category
            categories = {
                'Good': ((aqi_data['aqi_cal'] >= 0) & (aqi_data['aqi_cal'] <= 50)).sum(),
                'Moderate': ((aqi_data['aqi_cal'] > 50) & (aqi_data['aqi_cal'] <= 100)).sum(),
                'Unhealthy for Sensitive Groups': ((aqi_data['aqi_cal'] > 100) & (aqi_data['aqi_cal'] <= 150)).sum(),
                'Unhealthy': ((aqi_data['aqi_cal'] > 150) & (aqi_data['aqi_cal'] <= 200)).sum(),
                'Very Unhealthy': ((aqi_data['aqi_cal'] > 200) & (aqi_data['aqi_cal'] <= 300)).sum(),
                'Hazardous': (aqi_data['aqi_cal'] > 300).sum()
            }
            
            # Get timestamp
            if not aqi_data.empty and 'timestamp' in aqi_data.columns:
                try:
                    # Always try to convert timestamp to int first
                    timestamp_val = int(aqi_data['timestamp'].iloc[0])
                    timestamp = datetime.fromtimestamp(timestamp_val).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    # If conversion fails, use current time
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp = "Unknown"
            
            return jsonify({
                'summary': {
                    'avg_aqi': float(avg_aqi),
                    'max_aqi': float(max_aqi),
                    'min_aqi': float(min_aqi),
                    'categories': categories,
                    'timestamp': timestamp,
                    'station_count': len(aqi_data)
                }
            })
        else:
            return jsonify({'error': 'No AQI data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)