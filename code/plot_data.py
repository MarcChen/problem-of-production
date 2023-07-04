import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box
import pyproj
import numpy as np
import matplotlib.patches as patches
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_europe(shapefile_path, data, save_path=None):
    # Read the Shapefile data
    world = gpd.read_file(shapefile_path)

    # List of European countries
    europe_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus',
                        'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 
                        'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 
                        'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 
                        'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 
                        'Sweden', 'United Kingdom']

    # Filter for Europe
    europe = world[world['ADMIN'].isin(europe_countries)]

    # Add the data to the GeoDataFrame
    europe = europe.set_index('ADMIN').join(data)

    # Plot the map
    europe.plot(column='your_data_column', legend=True, legend_kwds={'label': "Your data label", 'orientation': "horizontal"})
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



"""data_dir = "../data/map/"
path_rg = data_dir + "NUTS_RG_01M_2021_3035_LEVL_0.json"
gdf_rg = gpd.read_file(path_rg)
path_bn = data_dir + "NUTS_BN_01M_2021_3035_LEVL_0.json"
gdf_bn = gpd.read_file(path_bn)
path_lb = data_dir + "NUTS_LB_2021_3035_LEVL_0.json"
gdf_lb = gpd.read_file(path_lb)

shapefile_path = '../data/map/europeNUTS.geojson'  # replace with the actual path to your Shapefile
save_path = '../data/map/figure.png'  # replace with the actual path where you want to save the figure
"""
"""
min_longitude = -12.102
max_longitude = 44.587
min_latitude = 33.700
max_latitude = 71.590

# Load GeoJSON data
gdf = gpd.read_file(shapefile_path)

# Define bounding box of main landmass (minx, miny, maxx, maxy)
bounds_geo = [min_longitude, min_latitude, max_longitude, max_latitude]

# Transform bounds to EPSG:3857
proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
bounds = [proj.transform(bounds_geo[0], bounds_geo[1]), proj.transform(bounds_geo[2], bounds_geo[3])]

# Flatten the bounds list
bounds = [coord for pair in bounds for coord in pair]

# Project to a Cartesian coordinate system (e.g., a suitable UTM zone)
# Here, EPSG:3857 is used as an example, which is suitable for worldwide data
gdf = gdf.to_crs(epsg=3857)

# Filter out regions whose centroids are outside of bounding box
gdf = gdf[gdf.geometry.centroid.x.between(bounds[0], bounds[2]) & gdf.geometry.centroid.y.between(bounds[1], bounds[3])]"""

""" # Plot the GeoDataFrame
ax = gdf_rg.plot(figsize=(20,15), color="gray")
gdf_bn.plot(figsize=(20,15), ax=ax, color="red")
gdf_lb.plot(figsize=(20,15), ax=ax, color="yellow")
# Save the plot as a PNG image
plt.savefig('europe_map.png') """


data_dir = "../data/map/"
path_rg = data_dir + "NUTS_RG_01M_2021_3035_LEVL_0.json"
path_bn = data_dir + "NUTS_BN_01M_2021_3035_LEVL_0.json"

gdf_rg = gpd.read_file(path_rg)
gdf_bn = gpd.read_file(path_bn)

# Define your geographical boundary box
minx, miny, maxx, maxy = 2.0e6, 0, 8.0e6, 6.0e6  # Replace these values with the ones from your plot axes

# Create a box from your boundary
boundary = gpd.GeoSeries(box(minx, miny, maxx, maxy), crs=gdf_rg.crs)

# Check for any invalid geometries and fix them : ps warning message but this solution make it work fine
gdf_rg['geometry'] = gdf_rg.geometry.apply(lambda x: x.buffer(0) if not x.is_valid else x)
gdf_bn['geometry'] = gdf_bn.geometry.apply(lambda x: x.buffer(0) if not x.is_valid else x)

# Replace 'within' with 'intersection'
gdf_rg.geometry = gdf_rg.geometry.intersection(boundary.geometry.iloc[0])
gdf_bn.geometry = gdf_bn.geometry.intersection(boundary.geometry.iloc[0])

# Plot the GeoDataFrame
ax = gdf_rg.plot(figsize=(20,15), color ="lightgray")
gdf_bn.plot(figsize=(20,15), ax=ax, color="red")

# Plot the centroids
centroids = gdf_rg.geometry.centroid
centroids.plot(ax=ax, color="purple")

###

# Create a divider for the existing axes instance
divider = make_axes_locatable(ax)

# Append axes to the right of ax, with 5% width of ax
cax = divider.append_axes("right", size="5%", pad=0.05)

# Generate a colormap
#cmap = plt.cm.coolwarm 
cmap = plt.get_cmap('inferno')


# Normalize data to 0-1
max_value = 1100  # Replace this with the actual max value
norm = plt.Normalize(0, max_value)

# Create a colorbar in the appended axes
# Tick locations can be set with the "ticks" keyword
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')

# Set colorbar label
cb.set_label('Data Scale')

# Add artificial data to GeoDataFrame
gdf_rg['data1'] = np.random.randint(100, 1000, gdf_rg.shape[0])
gdf_rg['data2'] = np.random.randint(100, 1000, gdf_rg.shape[0])

# Normalize the data to 0-1
gdf_rg['data1_norm'] = gdf_rg['data1'] / max_value
gdf_rg['data2_norm'] = gdf_rg['data2'] / max_value

# Define function to draw bar
def draw_bar(ax, x, y, width, height, color, alpha):
    ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='none'))

# Width of the bars
width = 5e4  # adjust as needed



# Iterate over GeoDataFrame
""" for idx, row in gdf_rg.iterrows():
    # Get centroid for current country
    centroid = centroids[idx]

    # Determine bar heights based on your data
    bar1_height = row['data1'] * 100  # adjust as needed
    bar2_height = row['data2'] * 100  # adjust as needed

    # Draw bars
    draw_bar(ax, centroid.x - width, centroid.y, width, bar1_height, color='blue')
    draw_bar(ax, centroid.x, centroid.y, width, bar2_height, color='red') """

for i, row in gdf_rg.iterrows():
    centroid = row.geometry.centroid
    bar1_height = row['data1'] * 300  # adjust as needed
    bar2_height = row['data2'] * 300  # adjust as needed
    max_1 = row['data1'] * 500 
    max_2 = row['data1'] * 500

    # Get the colors from the colormap
    color1 = cmap(row['data1_norm'])
    color2 = cmap(row['data2_norm'])

    """ # First draw the 'background' bars
    draw_bar(ax, centroid.x - width, centroid.y, width, max_1, color='green', alpha = 1)
    draw_bar(ax, centroid.x , centroid.y, width, max_2, color='green', alpha = 1) """

    # Then draw the actual data bars
    draw_bar(ax, centroid.x - width - width/14, centroid.y, width, bar1_height, color=color1, alpha=0.7)
    draw_bar(ax, centroid.x + width/14, centroid.y, width, bar2_height, color=color2, alpha=0.7)

    font_size = 6
    # Add label for percentage
    ax.text(centroid.x - width / 2, centroid.y - 0.05, f'{row["data1"] * 100:.2f}%',
            color='blue', ha='center', va='top', rotation='vertical', fontsize=font_size)
    ax.text(centroid.x + width / 2, centroid.y - 0.05, f'{row["data2"] * 100:.2f}%',
            color='red', ha='center', va='top', rotation='vertical', fontsize=font_size)

    
# Add legend rectangle
legend_x = 0.90
legend_y = 0.94
legend_width = 0.09
legend_height = 0.1

# Add legend header
header_height = 0.05
ax.add_patch(patches.Rectangle((legend_x, legend_y), legend_width, header_height,
                               linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.4,
                               transform=ax.transAxes, zorder=10))
ax.text(legend_x + legend_width / 2, legend_y + header_height / 2, 'Legend', color='black',
        ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

ax.add_patch(patches.Rectangle((legend_x, legend_y - legend_height), legend_width, legend_height,
                               linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.4,
                               transform=ax.transAxes, zorder=10))

# Add custom legend bars and text
spacing = 0.01
bar_width = legend_width * 0.35
bar_height = legend_height * 0.65
text_offset = 0.005  # Offset to position the text below the bars
x1 = legend_x + (legend_width - (2 * bar_width + spacing)) / 2  # X-coordinate for the first bar
x2 = x1 + bar_width + spacing * 0.6  # X-coordinate for the second bar

ax.add_patch(patches.Rectangle((x1, legend_y - legend_height + 2 * spacing),
                               bar_width, bar_height, facecolor='blue', alpha=0.7, transform=ax.transAxes))
ax.add_patch(patches.Rectangle((x2, legend_y - legend_height + 2 * spacing),
                               bar_width, bar_height, facecolor='red', alpha=0.7, transform=ax.transAxes))

ax.text(x1 + bar_width / 2, legend_y - legend_height + 2 * spacing - text_offset,
        'Wind', color='darkblue', ha='center', va='top', transform=ax.transAxes, alpha=1.0)
ax.text(x2 + bar_width / 2, legend_y - legend_height + 2 * spacing - text_offset,
        'PV', color='darkred', ha='center', va='top', transform=ax.transAxes, alpha=1.0)

# Plot point between the rectangles
dot_offset = 0.016
point_x = (x1+x2)/2 + dot_offset 
point_y = legend_y - legend_height + 2 * spacing + 0.1 * dot_offset

ax.plot(point_x, point_y, 'o', markersize=10, color='purple', transform=ax.transAxes)

# Save the plot as a PNG image
plt.savefig('europe_map_test.png', dpi=350)
