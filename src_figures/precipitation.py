# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    precipitation.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/02/08 09:52:10 by Danilo            #+#    #+#              #
#    Updated: 2023/07/18 10:02:05 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import f90nml
import datetime
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors
from shapely.geometry import Polygon

prec_levels = [0.1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
cmap_precipitation = colors.ListedColormap(['#D3E6F1','#2980B9', '#A9DFBF','#196F3D',
    '#F9E79F', '#F39C12', '#f37012', '#E74C3C', '#943126', '#E6B0AA', '#7a548e'], N=len(prec_levels)-1)

bias_colors = ['#943126', '#E74C3C', '#f37012', '#F39C12', '#F9E79F',
               '#F8F2DB', 'white', '#D8EDF0',
                '#9eb3c2', '#1c7293', '#065a82', '#1b3b6f', '#21295c']
# cmap_bias = colors.ListedColormap(bias_colors, N=len(prec_levels)-1)
cmap_bias = matplotlib.colors.LinearSegmentedColormap.from_list("", bias_colors)
# bias_levels = [-300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250]


def get_times_nml(namelist, model_data):
    """
    Calculates the times of the model data.

    Parameters:
        namelist (dict): The namelist containing the configuration start time and run duration.
        model_data (pd.DataFrame): The model data.

    Returns:
        pd.DatetimeIndex: The times of the model data.
    """
    start_date_str = namelist['nhyd_model']['config_start_time']
    run_duration_str = namelist['nhyd_model']['config_run_duration']
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d_%H:%M:%S')
    run_duration = datetime.datetime.strptime(run_duration_str, '%d_%H:%M:%S')
    finish_date = start_date + datetime.timedelta(days=run_duration.day, hours=run_duration.hour)
    times = pd.date_range(start_date, finish_date, periods=len(model_data.Time) + 1)[1:]
    return times

def get_experiment_parameters(experiments):
    """
    A function that takes a list of experiments and returns the times, first day, and last day.

    Parameters:
    - experiments: a list of experiments

    Returns:
    - times: a list of times
    - first_day: a string representing the first day
    - last_day: a string representing the last day
    """

    # Dummy for getting model times
    dummy = experiments[0]
    model_output = f'{dummy}/latlon.nc'
    namelist_path = f'{dummy}/namelist.atmosphere'

    # open data and namelist
    model_data = xr.open_dataset(model_output).chunk({"Time": -1})
    namelist = f90nml.read(glob.glob(namelist_path)[0])
    times = get_times_nml(namelist,model_data)
    first_day = datetime.datetime.strftime(times[0], '%Y-%m-%d')
    last_day = datetime.datetime.strftime(times[-2], '%Y-%m-%d')

    parameters = {
        'times': times,
        'first_day': first_day,
        'last_day': last_day,
        'max_latitude': float(model_data.latitude[0]),
        'max_longitude': float(model_data.longitude[-1]),
        'min_latitude': float(model_data.latitude[-1]),
        'min_longitude': float(model_data.longitude[0]),
    }

    return parameters 

def get_exp_name(experiment):
    """
    Returns the name of the experiment based on the given experiment path.

    Parameters:
        experiment (str): The path of the experiment.

    Returns:
        str: The name of the experiment.
    """
    expname = os.path.basename(experiment)

    microp_options = ["thompson", "kessler", "wsm6", "off"]
    microp = next((option for option in microp_options if option in expname), None)
    if microp is None:
        raise ValueError("Microp option not found in experiment name.")

    cumulus_options = ["ntiedtke", "tiedtke", "freitas", "fritsch", "off"]
    cumulus = next((option for option in cumulus_options if option in expname), None)
    if cumulus is None:
        raise ValueError("Cumulus option not found in experiment name.")

    pbl_options = ["ysu", "mynn"]
    pbl = next((option for option in pbl_options if option in expname), None)

    return f"{microp}_{cumulus}_{pbl}" if pbl else f"{microp}_{cumulus}"

def get_model_accprec(model_data):
    """
    Returns the accumulated precipitation from the given model data.
    
    Parameters:
        model_data (dict): A dictionary containing the model data.
        
    Returns:
        float: The accumulated precipitation.
    """
    if ('rainnc' in model_data.variables
        ) and ('rainc' in model_data.variables):
        acc_prec = model_data['rainnc']+model_data['rainc']
    # Get only micrphysics precipitation
    elif ('rainnc' in model_data.variables
        ) and ('rainc' not in model_data.variables):
        acc_prec = model_data['rainnc']
    # Get convective precipitation
    elif ('rainnc' not in model_data.variables
        ) and ('rainc' in model_data.variables):
        acc_prec = model_data['rainc'] 
    elif ('rainnc' not in model_data.variables
        ) and ('rainc' not in model_data.variables):
        acc_prec = model_data.uReconstructMeridional[0]*0
    return acc_prec[-1]

def process_experiment_data(data, experiment, experiment_name, times): 
    """
    Processes experiment data and adds it to the given data dictionary.

    Parameters:
    - data: The dictionary to which the processed data will be added.
    - experiment: The path to the experiment.
    - experiment_name: The name of the experiment.
    - imerg_accprec: The IMERG accumulated precipitation data.
    - times: The times associated with the model data.

    Returns:
    - The updated data dictionary.
    """ 
    model_data = xr.open_dataset(f'{experiment}/latlon.nc').chunk({"Time": -1})
    model_data = model_data.assign_coords({"Time":times})

    acc_prec = get_model_accprec(model_data)
    acc_prec = acc_prec.where(acc_prec >= 0, 0)
    
    print(f'limits for {experiment_name}: {float(acc_prec.min())} - {float(acc_prec.max())}')

    data[experiment_name] = acc_prec
    
    return data
    
def configure_gridlines(ax, col, row):
    """
    Configure gridlines for the map.

    Parameters:
        ax (AxesSubplot): The axes on which to configure the gridlines.
        col (int): The column index of the map.
        row (int): The row index of the map.

    Returns:
        None
    """
    # Configure gridlines for the map
    gl = ax.gridlines(
        draw_labels=True,
        zorder=2,
        linestyle='dashed',
        alpha=0.8,
        color='#383838'
    )
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838'}
    gl.right_labels = None
    gl.top_labels = None
    gl.bottom_labels = None if row != 1 else gl.bottom_labels
    gl.left_labels = None if col != 0 else gl.left_labels

def plot_precipitation_panels(data, experiments, figures_directory, zoom=False):
    """
    Plot precipitation panels for the given benchmarks.

    Parameters:
    - data (dict): Dictionary of precipitation data for each experiment.
    - experiments (list): List of experiment names.
    - figures_directory (str): Directory to save the figures.
    - zoom (bool): Whether to plot the zoomed-in version. Default is False.

    Returns:
    None
    """
    print('\nPlotting maps...')
    plt.close('all')

    ncol, nrow, imax, figsize = 3, 2, 6, (12, 10)
    print('Figure will have ncols:', ncol, 'rows:', nrow, 'n:', imax)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)
    datacrs = ccrs.PlateCarree()

    domain_coords = {
        'zoom': {
            'lon': [-43, -42],
            'lat': [-23, -22]
        },
        'full': {
            'lon': [-45, -40],
            'lat': [-25, -20]
        }
    }

    i = 0
    for col in range(ncol):
        for row in range(nrow):
            
            if i == imax:
                break

            experiment = experiments[i]
            experiment = get_exp_name(experiment)
            print(experiment)

            if 'off_' in experiment:
                continue
            
            prec = data[experiment]
            prec = prec.sortby('latitude', ascending=True).sortby('longitude', ascending=True)

            # Slice data for the domain being plotted
            if zoom:
                prec_domain = prec.sel(latitude=slice(*domain_coords['zoom']['lat']),
                                   longitude=slice(*domain_coords['zoom']['lon'])) 
            else:
                prec_domain = prec.sel(latitude=slice(*domain_coords['full']['lat']),
                                   longitude=slice(*domain_coords['full']['lon'])) 
                
            max_prec = float(np.amax(prec_domain.compute()))
                            
            ax = fig.add_subplot(gs[row, col], projection=datacrs, frameon=True)

            if zoom == False:
                ax.set_extent(domain_coords['full']['lon'] + domain_coords['full']['lat'], crs=datacrs)
                ax.text(-45, -19.8, f'{experiment}: {max_prec:.2f}', fontdict={'size': 14})
                
                # Plot polygon around smaller domain
                if col == 2 and row == 1:
                    polygon = Polygon([(-43, -23), (-43, -22), (-42, -22), (-42, -23)])
                    ax.add_geometries([polygon], ccrs.PlateCarree(), facecolor='none',
                                       edgecolor='red', linewidth=1, zorder=101)
            else:
                ax.set_extent(domain_coords['zoom']['lon'] + domain_coords['zoom']['lat'], crs=datacrs)
                ax.text(-43, -21.95, f'{experiment}: {max_prec:.2f}', fontdict={'size': 14})

            cf = ax.contourf(prec_domain.longitude, prec_domain.latitude, prec_domain, extend='max',
                             cmap=cmap_precipitation, levels=prec_levels)
            
            configure_gridlines(ax, col, row)
            ax.coastlines(zorder=1)            
            i += 1

    cb_axes = fig.add_axes([0.85, 0.18, 0.04, 0.6])
    fig.subplots_adjust(wspace=0.1, hspace=0, right=0.8)
    fig.colorbar(cf, cax=cb_axes, orientation="vertical")

    os.makedirs(figures_directory, exist_ok=True)
    if zoom == False:
        fname = f"{figures_directory}/acc_prec.png"
    else:
        fname = f"{figures_directory}/acc_prec_zoom.png"
    
    fig.savefig(fname, dpi=500)
    print(fname, 'saved')

def main(benchmarks_directory, figures_directory):

    ## Inputs ##
    experiments = glob.glob(benchmarks_directory+'/run*')
    experiments = sorted(experiments)
    
    ## Start the code ##
    parameters = get_experiment_parameters(experiments)

    print('\nOpening all data and putting it into a dictionary...')

    max_precipitation = float('-inf')

    data = {}

    for experiment in experiments:
        experiment_name = get_exp_name(experiment)
        print(experiment_name)

        if 'off_' in experiment_name: continue
        
        data = process_experiment_data(data, experiment, experiment_name, parameters["times"])

        acc_prec = data[experiment_name]

        experiment_max = np.max(acc_prec).compute().item()

        max_precipitation = max(max_precipitation, experiment_max)

    ## Make plots
    plot_precipitation_panels(data, experiments, figures_directory)
    plot_precipitation_panels(data, experiments, figures_directory, zoom=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("benchmarks_directory", nargs="?",
                        default='/p1-nemo/danilocs/mpas/MPAS-BR/benchmarks/Petropolis_2022/petropolis_250-4km.physics-test/',
                        help="Path to MPAS benchmarks")
    parser.add_argument("output_directory", nargs="?",default="../figures",
                        help="Name to use for this set of experiments")
    args = parser.parse_args()

    main(args.benchmarks_directory, args.output_directory)
