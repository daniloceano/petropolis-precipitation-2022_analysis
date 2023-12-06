import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

exported_precipitation_directory = "../model_precipitation_processed"
cemaden_directory = "../dados_CEMADEN"

def load_exported_precipitation_data(directory):
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    station_data_mpas = {}
    station_data_wrf = {}

    for file in all_files:
        # Extract station name from filename
        station_name = os.path.basename(file).split('_')[0]

        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if 'MPAS' in file:
            if station_name not in station_data_mpas:
                station_data_mpas[station_name] = []
            station_data_mpas[station_name].append(df)
        elif 'WRF' in file:
            if station_name not in station_data_wrf:
                station_data_wrf[station_name] = []
            station_data_wrf[station_name].append(df)

    # Concatenate dataframes for each station
    for station in station_data_mpas:
        station_data_mpas[station] = pd.concat(station_data_mpas[station], axis=1)
    for station in station_data_wrf:
        station_data_wrf[station] = pd.concat(station_data_wrf[station], axis=1)

    return station_data_mpas, station_data_wrf

def load_cemaden_precipitation_data(directory):
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    df_list = []
    for file in all_files:
        df = pd.read_csv(file, parse_dates=True, decimal=',')
        df.columns = ['City', 'ID', 'State', 'Station', 'Latitude', 'Longitude', 'Date_Time', 'Precipitation']
        df['Precipitation'] = pd.to_numeric(df['Precipitation'])

        # Pivot table to make stations as columns
        df_pivot = df.pivot_table(index='Date_Time', columns='Station', values='Precipitation', aggfunc='first')
        df_list.append(df_pivot)

    # Combine data from all files
    combined_df = pd.concat(df_list, axis=0)
    combined_df.index = pd.to_datetime(combined_df.index)
    return combined_df.resample('H').sum().dropna()  # Resample to hourly data and drop NA

def plot_precipitation_data_for_each_station(exported_mpas_data, exported_wrf_data, cemaden_data):
    # Define styles for each dataset
    mpas_style = {'linestyle': '--', 'color': 'blue'}
    wrf_style = {'linestyle': '-.', 'color': 'green'}
    cemaden_style = {'linestyle': '-', 'color': 'black', 'marker': 'o'}

    # Iterate over each station
    for station in cemaden_data.columns:
        plt.figure(figsize=(15, 8))
        title = f'Precipitation Data Comparison at {station}'

        # Format station name to match CEMADEN data columns
        formatted_station_name = station.replace(" ", "").replace("/", "").replace("-", "")

        if 'Dr.' in formatted_station_name:
            print(formatted_station_name)

        # Plot MPAS data for the station
        if formatted_station_name in exported_mpas_data:
            for column in exported_mpas_data[formatted_station_name].columns:
                plt.plot(exported_mpas_data[formatted_station_name].index,
                         exported_mpas_data[formatted_station_name][column],
                         label=f'MPAS {column}', **mpas_style)

        # Plot WRF data for the station
        if station in exported_wrf_data:
            for column in exported_wrf_data[formatted_station_name].columns:
                plt.plot(exported_wrf_data[formatted_station_name].index,
                         exported_wrf_data[formatted_station_name][column],
                         label=f'WRF {column}', **wrf_style)

        # Plot CEMADEN data for the station
        plt.plot(cemaden_data.index, cemaden_data[station].cumsum(), label='CEMADEN', **cemaden_style)

        plt.ylim(0, cemaden_data[station].cumsum().iloc[-1]+5)

        plt.xlabel('Time')
        plt.ylabel('Precipitation (mm)')
        plt.title(title)
        plt.legend()
        plt.savefig(f"../figures/accprec_timeseries/{formatted_station_name}_precipitation_comparison.png")

# Load data
mpas_data, wrf_data = load_exported_precipitation_data(exported_precipitation_directory)
cemaden_data = load_cemaden_precipitation_data(cemaden_directory)

# Plot data for each station
plot_precipitation_data_for_each_station(mpas_data, wrf_data, cemaden_data)
