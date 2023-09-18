import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm 
import matplotlib.colors as colors

def convert_longitude(lon):
    """
    Convert longitude from 0-360 degree format to -180 to 180 degree format.

    Parameters:
        lon (float or numpy.ndarray): Longitude value(s) in 0-360 degree format.

    Returns:
        float or numpy.ndarray: Longitude value(s) in -180 to 180 degree format.
    """
    return (lon + 180) % 360 - 180

petropolis_contour = [
    [-43.272908, -22.487562],
    [-43.253338, -22.483438],
    [-43.239949, -22.473921],
    [-43.235829, -22.480265],
    [-43.226216, -22.494857],
    [-43.233769, -22.507545],
    [-43.225872, -22.516108],
    [-43.226216, -22.527525],
    [-43.237202, -22.523720],
    [-43.233769, -22.535770],
    [-43.241322, -22.548454],
    [-43.260205, -22.558442],
    [-43.256772, -22.571440],
    [-43.239605, -22.566051],
    [-43.227246, -22.561929],
    [-43.228619, -22.548613],
    [-43.225872, -22.544173],
    [-43.210423, -22.552734],
    [-43.185360, -22.553369],
    [-43.177464, -22.538149],
    [-43.163044, -22.531172],
    [-43.149998, -22.538149],
    [-43.157208, -22.528635],
    [-43.152058, -22.518804],
    [-43.141758, -22.511509],
    [-43.132145, -22.516584],
    [-43.125279, -22.516901],
    [-43.125966, -22.508972],
    [-43.133175, -22.509606],
    [-43.139698, -22.505483],
    [-43.133862, -22.502946],
    [-43.126996, -22.495016],
    [-43.140728, -22.488989],
    [-43.149312, -22.482010],
    [-43.152058, -22.476934],
    [-43.143132, -22.468051],
    [-43.136609, -22.475665],
    [-43.132145, -22.467734],
    [-43.142445, -22.458216],
    [-43.139698, -22.452187],
    [-43.132145, -22.454408],
    [-43.130429, -22.462023],
    [-43.105710, -22.470589],
    [-43.091805, -22.462023],
    [-43.094723, -22.446159],
    [-43.107083, -22.439336],
    [-43.114979, -22.440130],
    [-43.119271, -22.447428],
    [-43.125966, -22.444413],
    [-43.130944, -22.446476],
    [-43.138669, -22.442668],
    [-43.133347, -22.436639],
    [-43.129055, -22.437115],
    [-43.111718, -22.433148],
    [-43.108885, -22.412519],
    [-43.135149, -22.411091],
    [-43.140471, -22.413789],
    [-43.133433, -22.395855],
    [-43.179610, -22.405854],
    [-43.208277, -22.405854],
    [-43.214629, -22.414265],
    [-43.230937, -22.417756],
    [-43.258231, -22.421406],
    [-43.258059, -22.429657],
    [-43.261321, -22.436322],
    [-43.260291, -22.448697],
    [-43.264754, -22.459009],
    [-43.260291, -22.464561],
    [-43.261149, -22.471065],
    [-43.267157, -22.476776],
    [-43.272908, -22.487562]
]

file = '/p1-nemo/danilocs/mpas/MPAS-BRv7.2/benchmarks/Petropolis_2022/petropolis_250-4km.physics-test/run.petropolis_250-4km.physics-test.microp_mp_thompson.cu_cu_ntiedtke/latlon.nc'
ds = xr.open_dataset(file)

# make a colormap that has land and ocean clearly delineated and of the
# same length (256 + 256)
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map',
    all_colors)

topography = ds.zgrid[0]
topography['longitude'] = convert_longitude(topography['longitude'].values)

minlon, maxlon = -45, -40
minlat, maxlat = -23.30, -20.30

fig = plt.figure(figsize=(13, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

norm = TwoSlopeNorm(vmin=-10, vcenter=1, vmax=2500)

# Plot the topography data
ax.set_extent([minlon, maxlon, minlat, maxlat])
im = ax.pcolormesh(topography['longitude'], topography['latitude'],
                    topography, cmap=terrain_map, rasterized=True, norm=norm,
                      transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.05, aspect=20, extend='both')
cbar.set_label('Elevation (m)', fontsize=12)

petropolis_contour = np.array(petropolis_contour)  # Convertendo para uma matriz numpy
x_coords = petropolis_contour[:, 0]
y_coords = petropolis_contour[:, 1]

ax.plot(x_coords, y_coords, c='k', lw='1')


# Configure gridlines using your function
ax.gridlines(draw_labels=True)
ax.coastlines()

# Set the title and labels
ax.set_title('Topography Map', fontsize=16)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Show the plot
plt.savefig('../figures/topography_MPAS.png', dpi=500)
