from glob import glob
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

def find_nearest_idx(array, value):
    """Find index of the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# CEMADEN data
files_stations = glob('../dados_CEMADEN/*csv')

# MPAS data
dummy_MPAS = xr.open_dataset("../dados_MPAS/run.petropolis_250-4km.physics-test.microp_mp_thompson.cu_cu_ntiedtke/latlon.nc")
dummy_WRF = xr.open_dataset("../dados_WRF/PET1km_Thompson-Off.nc")
dummy_WRF['lon'] = ((dummy_WRF['lon'] + 180) % 360) - 180

# Extract model grid points
mpas_lats = dummy_MPAS['latitude'].values
mpas_lons = dummy_MPAS['longitude'].values
wrf_lats = dummy_WRF['lat'].values
wrf_lons = dummy_WRF['lon'].values

# Step 3: Create a map with Cartopy
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-43.4, -43, -22.9, -22.2])  # Main map extent

# Add features to the map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Create an inset map using fig.add_axes
# The list [left, bottom, width, height] defines the position and size of the inset axes
# Adjust these values as needed
ax_inset = fig.add_axes([0.65, 0.1, 0.25, 0.25], projection=ccrs.PlateCarree())
ax_inset.set_extent([-44.5, -41.5, -23.5, -21])  # Extent for Rio de Janeiro state

# Add features to the inset map
ax_inset.add_feature(cfeature.LAND)
ax_inset.add_feature(cfeature.OCEAN)
ax_inset.add_feature(cfeature.COASTLINE)
ax_inset.add_feature(cfeature.BORDERS, linestyle=':')
ax_inset.add_feature(cfeature.LAKES, alpha=0.5)
ax_inset.add_feature(cfeature.RIVERS)

# Initialize an empty DataFrame
columns = ['Station Name', 'Station Latitude', 'Station Longitude', 
           'Nearest Model Latitude', 'Nearest Model Longitude']
station_mpas_df = pd.DataFrame(columns=columns)
station_wrf_df = pd.DataFrame(columns=columns)

for file in files_stations:
    data = pd.read_csv(file, decimal=',').dropna()
    data.columns = ['City', 'ID', 'State', 'Station', 'Latitude', 'Longitude', 'Date_Time', 'Precipitation']
    
    # Plot the station locations
    ax.scatter(data['Longitude'].unique(), data['Latitude'].unique(), s=50, color='red', alpha=0.5)

    # For each station, find and plot the nearest model grid point
    for station_lat, station_lon in zip(data['Latitude'].unique(), data['Longitude'].unique()):
        # Convert lat/lon to float and find nearest indices
        station_lat = float(station_lat)
        station_lon = float(station_lon)
        nearest_lat_idx_mpas = find_nearest_idx(mpas_lats, station_lat)
        nearest_lon_idx_mpas = find_nearest_idx(mpas_lons, station_lon)
        nearest_lat_idx_wrf = find_nearest_idx(wrf_lats, station_lat)
        nearest_lon_idx_wrf = find_nearest_idx(wrf_lons, station_lon)

        # Plot nearest model grid point
        nearest_lat_mpas = mpas_lats[nearest_lat_idx_mpas]
        nearest_lon_mpas = mpas_lons[nearest_lon_idx_mpas]
        nearest_lat_wrf = wrf_lats[nearest_lat_idx_wrf]
        nearest_lon_wrf = wrf_lons[nearest_lon_idx_wrf]

        ax.scatter(nearest_lon_mpas, nearest_lat_mpas, s=20, color='blue', alpha=0.5)
        ax.scatter(nearest_lon_wrf, nearest_lat_wrf, s=20, color='green', alpha=0.5)

        # Get station details
        station_name = data.iloc[0]['Station']

        # Add to DataFrame
        station_mpas_df = station_mpas_df.append({
            'Station Name': station_name,
            'Station Latitude': station_lat,
            'Station Longitude': station_lon,
            'Nearest Model Latitude': nearest_lat_mpas,
            'Nearest Model Longitude': nearest_lon_mpas
        }, ignore_index=True)

        station_wrf_df = station_wrf_df.append({
            'Station Name': station_name,
            'Station Latitude': station_lat,
            'Station Longitude': station_lon,
            'Nearest Model Latitude': nearest_lat_wrf,
            'Nearest Model Longitude': nearest_lon_wrf
        }, ignore_index=True)

plt.savefig('../figures/stations_map.png')

station_mpas_df.to_csv('../dados_MPAS/stations.csv', index=False)
station_wrf_df.to_csv('../dados_WRF/stations.csv', index=False)