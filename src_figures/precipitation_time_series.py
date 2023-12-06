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
    # Define styles for different model options
    model_styles = {
        'thompson': '-',
        'wsm6': '--',
        'off': 'x',
        'tiedtke': 'd',
        'ntiedtke': '^'
    }

    # Number of stations and subplot grid dimensions
    n_stations = len(cemaden_data.columns)
    n_cols = 4
    n_rows = n_stations // n_cols + n_stations % n_cols

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axs = axs.flatten()  # Flatten the array of axes

    # Custom legend entries
    legend_elements = [plt.Line2D([0], [0], color='blue', lw=2, label='MPAS'),
                       plt.Line2D([0], [0], color='green', lw=2, label='WRF'),
                       plt.Line2D([0], [0], color='black', marker='o', linestyle='-', label='CEMADEN')]

    for i, station in enumerate(cemaden_data.columns):
        ax = axs[i]
        formatted_station_name = station.replace(" ", "").replace("/", "").replace("-", "")

        # Plot MPAS and WRF data
        for data_dict, color in [(exported_mpas_data, 'blue'), (exported_wrf_data, 'green')]:
            if formatted_station_name in data_dict:
                for column in data_dict[formatted_station_name].columns:
                    microp = column.split('_')[-2].lower()
                    cumulus = column.split('_')[-1].lower()  # Extract model option from column name
                    ax.plot(data_dict[formatted_station_name].index, data_dict[formatted_station_name][column], 
                            color=color, linestyle=model_styles[microp], marker=model_styles[cumulus])

        # Plot CEMADEN data
        ax.plot(cemaden_data.index, cemaden_data[station].cumsum(), color='black', linestyle='-', marker='o')

        ax.set_title(station)
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        # Only add y-axis label to the first column
        if i % n_cols == 0:
            ax.set_ylabel('Precipitation (mm)')

    # Add custom legend outside of the last subplot
    axs[-1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout
    plt.tight_layout()
    plt.savefig("../figures/precipitation_timeseries_comparison.png")

def plot_precipitation_data_with_subplots(exported_mpas_data, exported_wrf_data, cemaden_data):
    # Define styles for each dataset
    mpas_style = {'linestyle': '--', 'color': 'blue', 'alpha': 0.3}
    wrf_style = {'linestyle': '-.', 'color': 'green', 'alpha': 0.3}
    cemaden_style = {'linestyle': '-', 'color': 'black', 'marker': 'o'}

    # Define the number of rows and columns for subplots
    n_rows = 2
    n_cols = len(cemaden_data.columns) // 2 + len(cemaden_data.columns) % 2

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axs = axs.flatten()  # Flatten to 1D array for easy iteration

    for i, station in enumerate(cemaden_data.columns):
        ax = axs[i]
        formatted_station_name = station.replace(" ", "").replace("/", "").replace("-", "")

        # MPAS shaded area
        if formatted_station_name in exported_mpas_data:
            mpas_max = exported_mpas_data[formatted_station_name].max(axis=1)
            mpas_min = exported_mpas_data[formatted_station_name].min(axis=1)
            ax.fill_between(mpas_max.index, mpas_min, mpas_max, color='blue', alpha=0.3, label='MPAS')

        # WRF shaded area
        if formatted_station_name in exported_wrf_data:
            wrf_max = exported_wrf_data[formatted_station_name].max(axis=1)
            wrf_min = exported_wrf_data[formatted_station_name].min(axis=1)
            ax.fill_between(wrf_max.index, wrf_min, wrf_max, color='green', alpha=0.3, label='WRF')

        # CEMADEN data plot
        ax.plot(cemaden_data.index, cemaden_data[station].cumsum(), 'k-o', label='CEMADEN', **cemaden_style)

        ax.set_ylim(0, cemaden_data[station].cumsum().iloc[-1]+5)
        ax.set_title(station)

    for i, ax in enumerate(axs):
        # Rotate x-axis ticks
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        # Only add y-axis label to the first column
        if i % n_cols == 0:
            ax.set_ylabel('Precipitation (mm)')

        # Only add legend to the first subplot
        if i == 0:
            ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"../figures/precipitation_timeseries_ensemble.png")

# Load data
mpas_data, wrf_data = load_exported_precipitation_data(exported_precipitation_directory)
cemaden_data = load_cemaden_precipitation_data(cemaden_directory)

# Plot data for each station
plot_precipitation_data_for_each_station(mpas_data, wrf_data, cemaden_data)

plot_precipitation_data_with_subplots(mpas_data, wrf_data, cemaden_data)

