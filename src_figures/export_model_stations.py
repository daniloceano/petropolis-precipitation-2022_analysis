from glob import glob
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.patches as mpatches

# Constants
FILES_STATIONS_PATH = '../dados_CEMADEN/*csv'
MPAS_DATASET_PATH = "../dados_MPAS/run.petropolis_250-4km.physics-test.microp_mp_thompson.cu_cu_ntiedtke/latlon.nc"
WRF_1KM_DATASET_PATH = "../dados_WRF/PET1km_Thompson-Off.nc"
WRF_5KM_DATASET_PATH = "../dados_WRF/PET5km_Thompson-Off.nc"
MAIN_EXTENT = [-43.3, -43, -22.6, -22.4]
RIO_EXTENT = [-44.5, -41.5, -23.5, -21]

def find_nearest_idx(array, value):
    """Find index of the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def load_dataset(file_path):
    """Load dataset from a given file path."""
    dataset = xr.open_dataset(file_path)
    if 'lon' in dataset:
        dataset['lon'] = ((dataset['lon'] + 180) % 360) - 180
    return dataset

def add_map_features(ax):
    """Add common features to a Cartopy axis."""
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

def create_dataframe():
    """Create an empty DataFrame for station data."""
    columns = ['Station Name', 'Station Latitude', 'Station Longitude', 
               'Nearest Model Latitude', 'Nearest Model Longitude']
    return pd.DataFrame(columns=columns)

def plot_station_locations(ax, data):
    """Plot the station locations on the map."""
    ax.scatter(data['Longitude'].unique(), data['Latitude'].unique(), s=100, color='k', alpha=0.5)

def process_station_data(ax, data,
                         mpas_lats, mpas_lons,
                         wrf_1km_lats, wrf_1km_lons, 
                         wrf_5km_lats, wrf_5km_lons, 
                         df_mpas, df_wrf_1km, df_wrf_5km):
    """Process each station and update DataFrames."""
    for station_lat, station_lon in zip(data['Latitude'].unique(), data['Longitude'].unique()):
        # Convert lat/lon to float and find nearest indices
        station_lat = float(station_lat)
        station_lon = float(station_lon)
        # Mpas
        nearest_lat_idx_mpas = find_nearest_idx(mpas_lats, station_lat)
        nearest_lon_idx_mpas = find_nearest_idx(mpas_lons, station_lon)
        # WRF 1km
        nearest_lat_idx_wrf_1km = find_nearest_idx(wrf_1km_lats, station_lat)
        nearest_lon_idx_wrf_1km = find_nearest_idx(wrf_1km_lons, station_lon)
        # WRF 5km
        nearest_lat_idx_wrf_5km = find_nearest_idx(wrf_5km_lats, station_lat)
        nearest_lon_idx_wrf_5km = find_nearest_idx(wrf_5km_lons, station_lon)

        # Plot nearest model grid point
        nearest_lat_mpas = mpas_lats[nearest_lat_idx_mpas]
        nearest_lon_mpas = mpas_lons[nearest_lon_idx_mpas]
        nearest_lat_wrf_1km = wrf_1km_lats[nearest_lat_idx_wrf_1km]
        nearest_lon_wrf_1km = wrf_1km_lons[nearest_lon_idx_wrf_1km]
        nearest_lat_wrf_5km = wrf_5km_lats[nearest_lat_idx_wrf_5km]
        nearest_lon_wrf_5km = wrf_5km_lons[nearest_lon_idx_wrf_5km]

        ax.scatter(nearest_lon_mpas, nearest_lat_mpas, s=100, color='red', alpha=0.5)
        ax.scatter(nearest_lon_wrf_1km, nearest_lat_wrf_1km, s=100, color='blue', alpha=0.5)
        ax.scatter(nearest_lon_wrf_5km, nearest_lat_wrf_5km, s=100, color='green', alpha=0.5)

        # Get station details
        station_name = data.iloc[0]['Station']

        # Add to DataFrame
        df_mpas.loc[len(df_mpas)] = [station_name, station_lat, station_lon, nearest_lat_mpas, nearest_lon_mpas]
        df_wrf_1km.loc[len(df_wrf_1km)] = [station_name, station_lat, station_lon, nearest_lat_wrf_1km, nearest_lon_wrf_1km]
        df_wrf_5km.loc[len(df_wrf_5km)] = [station_name, station_lat, station_lon, nearest_lat_wrf_5km, nearest_lon_wrf_5km]

def add_inset_map(fig, extent):
    """Add an inset map to the figure."""
    ax_inset = fig.add_axes([0.7, 0.1, 0.25, 0.25], projection=ccrs.PlateCarree())
    ax_inset.set_extent(extent)
    add_map_features(ax_inset)
    return ax_inset

def add_rectangle_to_inset(ax_inset, main_extent):
    """Add a rectangle to the inset map to highlight the main map area."""
    rect_width = main_extent[1] - main_extent[0]
    rect_height = main_extent[3] - main_extent[2]
    rect = mpatches.Rectangle((main_extent[0], main_extent[2]), rect_width, rect_height,
                              edgecolor='red', facecolor='none', lw=2, transform=ccrs.PlateCarree())
    ax_inset.add_patch(rect)

def add_gridlines(ax, x_interval, y_interval, main=True):
    """Add gridlines to the map."""
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    if main:
        gl.ylabels_right = False
    else:
        gl.ylabels_left = False
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}
    gl.xformatter = mticker.FuncFormatter(lambda v, pos: f"{v:.2f}°E" if v >= 0 else f"{-v:.2f}°W")
    gl.yformatter = mticker.FuncFormatter(lambda v, pos: f"{v:.2f}°N" if v >= 0 else f"{-v:.2f}°S")
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, x_interval))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, y_interval))


def main():
    # Load datasets
    files_stations = glob(FILES_STATIONS_PATH)
    dummy_MPAS = load_dataset(MPAS_DATASET_PATH)
    dummy_WRF_1KM = load_dataset(WRF_1KM_DATASET_PATH)
    dummy_WRF_5KM = load_dataset(WRF_5KM_DATASET_PATH)

    # Extract model grid points
    mpas_lats = dummy_MPAS['latitude'].values
    mpas_lons = dummy_MPAS['longitude'].values
    wrf_1km_lats = dummy_WRF_1KM['lat'].values
    wrf_1km_lons = dummy_WRF_1KM['lon'].values
    wrf_5km_lats = dummy_WRF_5KM['lat'].values
    wrf_5km_lons = dummy_WRF_5KM['lon'].values

    # Initialize DataFrames
    station_mpas_df = create_dataframe()
    station_wrf_1km_df = create_dataframe()
    station_wrf_5km_df = create_dataframe()

    # Create main map
    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(MAIN_EXTENT)
    add_map_features(ax)
    add_gridlines(ax, 0.05, 0.05)  # Adjust intervals here for the main map

    # Create inset map for Rio de Janeiro
    ax_inset = add_inset_map(fig, RIO_EXTENT)
    add_rectangle_to_inset(ax_inset, MAIN_EXTENT)
    add_gridlines(ax_inset, 0.5, 0.5, main=False)  # Adjust intervals here for the inset map

    # Process each station file
    for file in files_stations:
        data = pd.read_csv(file, decimal=',').dropna()
        data.columns = ['City', 'ID', 'State', 'Station', 'Latitude', 'Longitude', 'Date_Time', 'Precipitation']

        plot_station_locations(ax, data)
        process_station_data(ax, data,
                             mpas_lats, mpas_lons,
                             wrf_1km_lats, wrf_1km_lons, 
                             wrf_5km_lats, wrf_5km_lons, 
                             station_mpas_df, station_wrf_1km_df, station_wrf_5km_df)

    # Initialize legend handles
    cemaden_handle = plt.Line2D([], [], color='k', marker='o', linestyle='None',
                                markersize=10, label='CEMADEN', alpha=0.5)
    mpas_handle = plt.Line2D([], [], color='red', marker='o', linestyle='None',
                            markersize=10, label='MPAS', alpha=0.5)
    wrf_1km_handle = plt.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=10, label='WRF_1km', alpha=0.5)
    wrf_5km_handle = plt.Line2D([], [], color='green', marker='o', linestyle='None',
                            markersize=10, label='WRF_5km', alpha=0.5)

    # Add the custom legend handles
    ax.legend(handles=[cemaden_handle, mpas_handle, wrf_1km_handle, wrf_5km_handle], loc='upper left')

    # Save figure
    plt.savefig('../figures/stations_map.png')

    # Save model grid points closest to stations
    station_mpas_df.to_csv('../model_processed_data/stations_MPAS.csv', index=False)
    station_wrf_1km_df.to_csv('../model_processed_data/stations_WRF-1km.csv', index=False)
    station_wrf_5km_df.to_csv('../model_processed_data/stations_WRF-5km.csv', index=False)

if __name__ == "__main__":
    main()
