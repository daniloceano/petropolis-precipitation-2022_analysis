from glob import glob
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
import os

# Constants
CEMADEN_STATIONS_PATH = '../dados_CEMADEN/*csv'
MPAS_DATASET_PATH = "../dados_MPAS/*/latlon.nc"
WRF_1KM_DATASET_PATH = "../dados_WRF/PET1km_*.nc"
WRF_5KM_DATASET_PATH = "../dados_WRF/PET5km_*.nc"
MODEL_STATIONS_PATTERN = "../model_processed_data/stations*.csv"

def find_nearest(lat, lon, dataset):
    """
    Function to find the nearest point in the dataset for a given station's latitude and longitude.
    Handles both MPAS datasets (using 'latitude' and 'longitude') and WRF datasets (using 'lat' and 'lon').
    """
    # Determine which coordinate names are used in the dataset
    if 'latitude' in dataset and 'longitude' in dataset:
        dataset_lat = dataset['latitude'].values
        dataset_lon = dataset['longitude'].values
    elif 'lat' in dataset and 'lon' in dataset:
        dataset_lat = dataset['lat'].values
        dataset_lon = dataset['lon'].values
    else:
        raise ValueError("Latitude and longitude coordinates not found in dataset")

    # Compute the absolute difference between the station's coordinates and dataset's coordinates
    abs_diff_lat = np.abs(dataset_lat - lat)
    abs_diff_lon = np.abs(dataset_lon - lon)

    # Assuming a 2D grid, compute the 2D index of the minimum difference
    min_diff_idx = np.argmin(abs_diff_lat + abs_diff_lon)
    lat_idx, lon_idx = np.unravel_index(min_diff_idx, (dataset_lat.shape[0], dataset_lon.shape[0]))
    return lat_idx, lon_idx

def process_mpas_precipitation(ds):
    """
    Function to process MPAS precipitation data
    """
    if 'rainc' in ds.variables and 'rainnc' in ds.variables:
        # Summing rainc and rainnc if both are present
        return ds['rainc'] + ds['rainnc']
    elif 'rainc' in ds.variables:
        # Using rainc if only rainc is present
        return ds['rainc']
    elif 'rainnc' in ds.variables:
        # Using rainnc if only rainnc is present
        return ds['rainnc']
    else:
        raise ValueError("No precipitation data found in dataset")
    
def extract_mpas_experiment_name(mpas_dataset_path):
    parts = mpas_dataset_path.split('/')
    filename = parts[-2]  # Get the second last part of the path
    experiment_parts = filename.split('.')[3:5]  # Extract relevant parts of the filename
    microp = experiment_parts[0].split('_')[-1]
    cumulus = experiment_parts[1].split('_')[-1]
    experiment_name = 'MPAS_' + '_'.join([microp, cumulus])
    return experiment_name

# Function to process WRF precipitation data
def process_wrf_precipitation(ds):
    if 'rainnc' in ds.variables:
        return ds['rainnc']
    else:
        raise ValueError("No precipitation data found in WRF dataset")

# Function to extract WRF experiment name (modify as per your WRF dataset naming convention)
    
def extract_wrf_experiment_name(wrf_dataset_path):
    # Extract experiment name from WRF dataset path
    parts = wrf_dataset_path.split('/')
    filename = parts[-1]  
    experiment_parts = filename.split('_')[1]
    microp = experiment_parts.split('-')[0].lower()  # Convert to lowercase
    cumulus = experiment_parts.split('-')[1].lower()  # Convert to lowercase
    grid_spacing = '5km' if '5km' in filename else '1km'
    experiment_name = 'WRF' + grid_spacing + '_' + '_'.join([microp, cumulus])
    return experiment_name

# Use glob to find all files that match the pattern
model_stations_files = glob(MODEL_STATIONS_PATTERN)

# Iterate over the files and read them into the appropriate DataFrame
for file in model_stations_files:
    if 'MPAS' in file:
        mpas_stations = pd.read_csv(file)
    elif 'WRF-1km' in file:
        wrf_1km_stations = pd.read_csv(file)
    elif 'WRF-5km' in file:
        wrf_5km_stations = pd.read_csv(file)

# Filtering the dataframe to remove duplicates based on model latitudes and longitudes
# Keeping the first occurrence of each
mpas_stations = mpas_stations.drop_duplicates(subset=['Nearest Model Latitude', 'Nearest Model Longitude'], keep='first')
wrf_1km_stations = wrf_1km_stations.drop_duplicates(subset=['Nearest Model Latitude', 'Nearest Model Longitude'], keep='first')
wrf_5km_stations = wrf_5km_stations.drop_duplicates(subset=['Nearest Model Latitude', 'Nearest Model Longitude'], keep='first')

# Combining station names from all dataframes into a single list
combined_station_names = mpas_stations['Station Name'].tolist() + wrf_1km_stations['Station Name'].tolist() + wrf_5km_stations['Station Name'].tolist()

# Removing duplicates by converting the list to a set, then back to a list
unique_station_names = list(set(combined_station_names))

# Dictionary to store the precipitation time-series for each station in each experiment
precipitation_data = {}

# Iterate over each dataset (both MPAS and WRF)
dataset_path_list = glob(MPAS_DATASET_PATH) + glob(WRF_1KM_DATASET_PATH) + glob(WRF_5KM_DATASET_PATH)
for dataset_path in dataset_path_list:
    # Open the dataset
    ds = xr.open_dataset(dataset_path)
    # Change longitude from 0-360 to -180-180 for WRF datasets
    if 'WRF' in dataset_path:
        ds['lon'] = ((ds['lon'] + 180) % 360) - 180

    # Check if the dataset is MPAS or WRF and process accordingly
    if 'MPAS' in dataset_path:
        precip = process_mpas_precipitation(ds)
        experiment_name = extract_mpas_experiment_name(dataset_path)
        model_stations = mpas_stations
    else:  # Assuming it's a WRF dataset
        precip = process_wrf_precipitation(ds)
        experiment_name = extract_wrf_experiment_name(dataset_path)
        model_stations = wrf_1km_stations if '1km' in dataset_path else wrf_5km_stations

    # Iterate over each station
    for station_name in unique_station_names:
        # Check if the station is in the mpas_stations DataFrame
        if station_name in model_stations['Station Name'].values:
            station_lat = model_stations.loc[model_stations['Station Name'] == station_name, 'Nearest Model Latitude'].values[0]
            station_lon = model_stations.loc[model_stations['Station Name'] == station_name, 'Nearest Model Longitude'].values[0]

            # Extract the time-series for this station
            if 'WRF' in dataset_path:
                station_precip = precip.sel(lat=station_lat, lon=station_lon, method='nearest')
            else:
                station_precip = precip.sel(latitude=station_lat, longitude=station_lon)

            # Store the time-series in the dictionary with experiment name as key
            if station_name not in precipitation_data:
                precipitation_data[station_name] = {}
            precipitation_data[station_name][experiment_name] = station_precip
        else:
            # If the station is not found, continue to the next station
            print(f"Station {station_name} not found in MPAS dataset.")

    # Close the dataset
    ds.close()

print("Finished processing MPAS and WRF datasets.")

# Create a new directory for the processed precipitation data
output_directory = "../model_precipitation_processed"
os.makedirs(output_directory, exist_ok=True)

# Iterate over the precipitation data and save each station's data to a CSV
for station_name, experiments in precipitation_data.items():

    # Convert each experiment's time series into a 1-dimensional array
    for experiment_name, time_series in experiments.items():
        # Create a DataFrame with a proper time index
        if 'MPAS' in experiment_name:
            # Define the start and end times for the time range
            start, end = '2022-02-15_01:00:00', '2022-02-16_00:00:00'
            date_format = '%Y-%m-%d_%H:%M:%S'
            start = datetime.strptime(start, date_format)
            end = datetime.strptime(end, date_format)
            time = pd.date_range(start, end, freq='1H')
            df_station = pd.DataFrame(time_series.values, index=time, columns=[experiment_name])
        else:
            # For non-MPAS data, use the Time attribute from the dataset as the index
            df_station = pd.DataFrame(time_series, index=time_series.time, columns=[experiment_name])

        # Construct the file path for the CSV
        station_name = station_name.replace(" ", "").replace("/", "").replace("-", "")
        output_file_path = os.path.join(output_directory, f"{station_name}_{experiment_name}_precipitation.csv")

        # Save the DataFrame to a CSV file
        df_station.to_csv(output_file_path)

print("Precipitation data saved to CSV files.")