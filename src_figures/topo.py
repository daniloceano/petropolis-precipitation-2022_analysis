import pygmt
import numpy as np


minlon, maxlon = -43.35, -43.00
minlat, maxlat = -22.60, -22.35
# Coordenadas que definem o contorno da cidade de Petrópolis
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

petropolis_contour = np.array(petropolis_contour)  # Convertendo para uma matriz numpy
x_coords = petropolis_contour[:, 0]
y_coords = petropolis_contour[:, 1]

inset_projection = "M3.5c"
region = [minlon, maxlon, minlat, maxlat]

fig = pygmt.Figure()
fig.basemap(frame=True, projection="M20c", region=[minlon, maxlon, minlat, maxlat])
pygmt.makecpt(cmap='geo', series='-6000/4000/100', continuous=True)
fig.grdimage(grid="@earth_relief_03s", shading=True)

fig.coast(projection="M20c", region=[minlon, maxlon, minlat, maxlat], frame="a",
          borders=["1/0.5p,black", "2/1p,black", "3/0.5p,blue"])

# Adicionar a linha pontilhada do contorno da cidade de Petrópolis
#fig.plot(x=x_coords, y=y_coords, pen="2p,black,4_2:0")
fig.plot(x=x_coords, y=y_coords, pen="2p,black")
fig.colorbar(frame=["a2000", "x+l Elevação (m)", "y+lm"])
fig.savefig("Topografia_Petropolis.png", crop=True, dpi=300)
fig.show()















#https://github.com/andrebelem/3D-Antarctic-maps

import pygmt

#RJ
#minlon, maxlon = -43.70, -42.55
#minlat, maxlat = -22.80, -22.15

minlon, maxlon = -43.35, -43.00
minlat, maxlat = -22.60, -22.35

#MG
#minlon, maxlon = -51.5, -39.5
#minlat, maxlat = -23.5, -14.0

#SP
#minlon, maxlon = -53.5, -43.5
#minlat, maxlat = -25.5, -19.5
#inset_region = [-80.,-35.,-35.5,10.]
inset_projection = "M3.5c"
region=[minlon, maxlon, minlat, maxlat]
#minlon, maxlon = 120., 122.1
#minlat, maxlat = 21.8, 25.6

# Load sample earth relief data
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=[minlon, maxlon, minlat, maxlat])

frame = ["xa1f0.25","ya1f0.25", "z2000+lMetros", "wSEnZ"]

pygmt.makecpt(cmap='geo', series=f'0/4000/100', continuous=True)


fig = pygmt.Figure()
fig.grdview(
    grid=grid,
    region=[minlon, maxlon, minlat, maxlat, 0, 4000],
    perspective=[150, 30],
    frame=frame,
    projection="M15c",
    zsize="4c",
    surftype="i",
    plane="0+gazure",
    shading=0,
    contourpen="1p",
    cmap="geo",  # Aplicar a paleta personalizada aqui
)

'''
fig.grdview(
    grid=grid,
    region=[minlon, maxlon, minlat, maxlat, -6000, 4000],
    perspective=[150, 30],
    frame=frame,
    projection="M15c",
    zsize="4c",
    surftype="i",
    plane="-6000+gazure",
    shading=0,
    # Set the contour pen thickness to "1p"
    contourpen="1p",
)
'''
#fig.basemap(
   # perspective=True,
    #rose="jTL+w3c+l+o-2c/-1c", #map directional rose at the top left corner
   # )


'''
with fig.inset(position="jBR+w3.5c+o-4.5c/8.2c", margin=0, box="+pgreen"):
    fig.coast(
        region=inset_region,
        shorelines="thin",
        projection=inset_projection,
        land="lightyellow",
        water="lightblue",
        frame="a",
        borders=["1/0.5p,black", "2/1p,black", "3/0.5p,blue"]
    )
    rectangle = [[region[0], region[2], region[1], region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="1p,blue")
'''
fig.colorbar(perspective=True, frame=["a2000", "x+lElevação (m)", "y+lm"])
fig.savefig("Topografia_Petropolis_3D.png", crop=True, dpi=300)
# Show the plot
fig.show()










#RJ
#minlon, maxlon = -43.50, -42.50
#minlat, maxlat = -22.70, -22.10
minlon, maxlon = -43.35, -43.00
minlat, maxlat = -22.60, -22.35
#MG
#minlon, maxlon = -51.5, -39.5
#minlat, maxlat = -23.5, -14.0

#SP
#minlon, maxlon = -53.5, -43.5
#minlat, maxlat = -25.5, -19.5

#inset_region = [-80.,-35.,-35.5,10.]
inset_projection = "M3.5c"
region=[minlon, maxlon, minlat, maxlat]

fig = pygmt.Figure()
fig.basemap(frame=True, projection="M20c", region=[minlon, maxlon, minlat, maxlat])
pygmt.makecpt(
        cmap='geo',
        series=f'-6000/4000/100',
        continuous=True
    )

fig.grdimage(grid="@earth_relief_03s", shading=True)
# Call the coast method for the plot
fig.coast(
    # Set the projection to Mercator, and plot size to 10 cm
    projection="M20c",
    # Set the region of the plot
    region=[minlon, maxlon, minlat, maxlat],
    # Set the frame of the plot
    frame="a",
    # Draw national borders with a 1-point black line
    borders=["1/0.5p,black", "2/1p,black", "3/0.5p,blue"],
)
'''

with fig.inset(position="jBR+w3.5c+o15.5c/17c", margin=0, box="+pgreen"):
    fig.coast(
        region=inset_region,
        shorelines="thin",
        projection=inset_projection,
        land="lightyellow",
        water="lightblue",
        frame="a",
        borders=["1/0.5p,black", "2/1p,red", "3/0.5p,blue"]
    )
    rectangle = [[region[0], region[2], region[1], region[3]]]
    fig.plot(data=rectangle, style="r+s", pen="1p,blue")
'''
fig.colorbar(frame=["a2000", "x+l Elevação (m)", "y+lm"])
fig.savefig("Topografia_Petropolis_2D.png", crop=True, dpi=300)
fig.show()


















