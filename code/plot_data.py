import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib.ticker as ticker


data_dir="../data/map/"
path_rg="NUTS_RG_01M_2021_3035_LEVL_0.json"
path_bn="NUTS_BN_01M_2021_3035_LEVL_0.json"

def save_or_show_plot(save_path, dpi):

    """
    Saves the plot as a PNG image or shows it.

    Parameters:
        save_path (str or None): The file path to save the plot as a PNG image. If None, the plot is displayed instead.
        dpi (int): The resolution of the saved image in dots per inch. Default is 350.

    Returns:
        None
    """
        
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=dpi)
    plt.close()

def draw_map_plot(data_dir, path_rg, path_bn, c_max, c, n, countries, dpi = 350, save_path= None, title = None):

    """
    Function to draw a map plot.

    Parameters:
    - data_dir: Directory path of the data files (string)
    - path_rg: Path to the file containing regional data (string)
    - path_bn: Path to the file containing boundary data (string)
    - c_max: Maximum capacity values for wind and PV (1D numpy array of shape (2*n,))
    - c: Capacity values for wind and PV (1D numpy array of shape (2*n,))
    - n: Number of country (int)
    - countries: List of country codes (list)
    - dpi: DPI (dots per inch) for the plot (int, optional)
    - save_path: Path to save the plot (string, optional)
    - title: Title for the plot (string, optional)
    """

    path_rg = data_dir + path_rg
    path_bn = data_dir + path_bn

    gdf_rg = gpd.read_file(path_rg)
    gdf_bn = gpd.read_file(path_bn)

    # Filter GeoDataFrames by countries
    gdf_rg = gdf_rg[gdf_rg['CNTR_CODE'].isin(countries)]

    # Define your geographical boundary box
    minx, miny, maxx, maxy = 2.0e6, 0, 8.0e6, 6.0e6  # Replace these values with the ones from your plot axes

    # Create a box from your boundary
    boundary = gpd.GeoSeries(box(minx, miny, maxx, maxy), crs=gdf_rg.crs)

    # Check for any invalid geometries and fix them
    gdf_rg['geometry'] = gdf_rg.geometry.apply(lambda x: x.buffer(0) if not x.is_valid else x)
    gdf_bn['geometry'] = gdf_bn.geometry.apply(lambda x: x.buffer(0) if not x.is_valid else x)

    # Replace 'within' with 'intersection'
    gdf_rg.geometry = gdf_rg.geometry.intersection(boundary.geometry.iloc[0])
    gdf_bn.geometry = gdf_bn.geometry.intersection(boundary.geometry.iloc[0])

    # Plot the GeoDataFrame
    ax = gdf_rg.plot(figsize=(20, 15), color="lightgray")
    gdf_bn.plot(figsize=(20, 15), ax=ax, color="black")

    # Plot the centroids
    centroids = gdf_rg.geometry.centroid
    centroids.plot(ax=ax, color="purple")

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)

    # Append axes to the right of ax, with 5% width of ax
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Generate a colormap
    cmap = plt.get_cmap('inferno')

    # Normalize data to 0-1
    max_value = np.max(c)
    norm = plt.Normalize(0, max_value)

    # Create a colorbar in the appended axes
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')

    # Set colorbar label
    cb.set_label('Data Scale (MWh)')

    # Sort the GeoDataFrame based on CNTR_CODE column
    gdf_rg = gdf_rg.sort_values('CNTR_CODE')

    gdf_rg['wind_capacities_allocated'] = c[:n]
    gdf_rg['pv_capacities_allocated'] = c[n:]

    gdf_rg['wind_max_capacities'] = c_max[:n]
    gdf_rg['pv_max_capacities'] = c_max[n:]

    # Normalize the data to 0-1
    gdf_rg['wind_allocation_ratio'] = gdf_rg['wind_capacities_allocated'] / c_max[:n].reshape(-1)
    gdf_rg['pv_allocation_ratio'] = gdf_rg['pv_capacities_allocated'] / c_max[n:].reshape(-1)

    print(gdf_rg)

    # Define function to draw bar
    def draw_bar(ax, x, y, width, height, color, alpha):
        ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='none'))

    # Width of the bars
    width = 5e4  # adjust as needed
    for index, (i, row) in enumerate(gdf_rg.iterrows()):
        centroid = row.geometry.centroid
        bar1_height = row['wind_capacities_allocated'] * 8  # adjust as needed
        bar2_height = row['pv_capacities_allocated'] * 8  # adjust as needed

        # Get the colors from the colormap
        color1 = cmap(norm(row['wind_capacities_allocated']))
        color2 = cmap(norm(row['pv_capacities_allocated']))

        # Then draw the actual data bars
        draw_bar(ax, centroid.x - width - width / 14, centroid.y, width, bar1_height, color=color1, alpha=0.7)
        draw_bar(ax, centroid.x + width / 14, centroid.y, width, bar2_height, color=color2, alpha=0.7)

        # Add label for percentage (wind_capacities_allocated)
        font_size = 6
        """ index =  countries.index(row["CNTR_CODE"]) """
        percentage1 = float((row["wind_capacities_allocated"] / c_max[:n][index]) * 100)
        ax.text(centroid.x - width / 2, centroid.y - 0.05, f'{percentage1:.0f}%',
                color='blue', ha='center', va='top', rotation='vertical', fontsize=font_size)

        # Add label for percentage (pv_capacities_allocated)
        percentage2 = float((row["pv_capacities_allocated"] / c_max[n:][index]) * 100)
        ax.text(centroid.x + width / 2, centroid.y - 0.05, f'{percentage2:.0f}%',
                color='red', ha='center', va='top', rotation='vertical', fontsize=font_size)

    # Add legend rectangle
    legend_x = 0.90
    legend_y = 0.94
    legend_width = 0.09
    legend_height = 0.13

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
    x1 = legend_x + (legend_width - (2 * bar_width + spacing)) / 2 + 0.2 * spacing # X-coordinate for the first bar
    x2 = x1 + bar_width + spacing * 0.6 + 0.2 * spacing  # X-coordinate for the second bar
    y = legend_y - legend_height + 3.5 * spacing

    # Add Wind bar and text
    ax.add_patch(patches.Rectangle((x1, y),
                                bar_width, bar_height, facecolor='blue', alpha=0.7, transform=ax.transAxes))
    ax.text(x1 + bar_width / 2, y - text_offset,
            'Wind', color='darkblue', ha='center', va='top', transform=ax.transAxes, alpha=1.0)

    # Add PV bar and text
    ax.add_patch(patches.Rectangle((x2, y),
                                bar_width, bar_height, facecolor='red', alpha=0.7, transform=ax.transAxes))
    ax.text(x2 + bar_width / 2, y - text_offset,
            'PV', color='darkred', ha='center', va='top', transform=ax.transAxes, alpha=1.0)

    # Add Allocation text
    allocation_x = legend_x + legend_width / 2  # X-coordinate for the allocation text
    allocation_y = legend_y - legend_height + 1.5 * spacing   # Y-coordinate for the allocation text
    ax.text(allocation_x, allocation_y,
            'allocation', color='black', ha='center', va='top', transform=ax.transAxes)

    # Plot point between the rectangles
    dot_offset = 0.016
    point_x = (x1 + x2) / 2 + dot_offset
    point_y = y + 0.1 * dot_offset

    ax.plot(point_x, point_y, 'o', markersize=10, color='purple', transform=ax.transAxes)

    # Set the title of the plot if provided
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold', ha='center')
        
    save_or_show_plot(save_path,dpi)

def plot_heatmap(data, x_labels, y_labels, cmap='coolwarm', save_path=None, title=None, dpi=350):
    """
    Function to plot a heatmap using Matplotlib.

    Parameters:
    - data: 2D numpy array to plot
    - x_labels: labels for the x axis (list or array-like)
    - y_labels: labels for the y axis (list or array-like)
    - cmap: colormap to use for the heatmap (string, optional)
    - save_path: path to save the plot (string, optional)
    - dpi: DPI (dots per inch) for the plot (int, optional)
    - title: title for the plot (string, optional)
    """

    fig, ax = plt.subplots()

    # Create a heatmap
    cax = ax.imshow(data, cmap=cmap)
    
    """# Find the intervals at which to place the x and y labels
    x_interval = data.shape[1] / len(x_labels)
    y_interval = data.shape[0] / len(y_labels)

    # Set labels for the x and y axes at the calculated intervals
    ax.set_xticks(np.arange(x_interval / 2, data.shape[1], x_interval))
    ax.set_yticks(np.arange(y_interval / 2, data.shape[0], y_interval))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)"""

    # Define the interval for displaying x and y labels
    x_label_interval = 2
    y_label_interval = 2

    # Set labels for the x and y axes at the defined intervals
    ax.set_xticks(np.arange(0, data.shape[1], x_label_interval))
    ax.set_yticks(np.arange(0, data.shape[0], y_label_interval))
    ax.set_xticklabels(x_labels[::x_label_interval])
    ax.set_yticklabels(y_labels[::y_label_interval])

    # Rotate the x labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create a colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Color scale')

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    # Save the plot if a save path is provided
    save_or_show_plot(save_path,dpi)

def plot_text_heatmap(data, x_labels, y_labels, cmap='coolwarm', save_path=None, title=None, dpi=350):
    """
    Function to plot a heatmap using Seaborn.

    Parameters:
    - data: 2D numpy array to plot
    - x_labels: labels for the x axis (list or array-like)
    - y_labels: labels for the y axis (list or array-like)
    - cmap: colormap to use for the heatmap (string, optional)
    - save_path: path to save the plot (string, optional)
    - dpi: DPI (dots per inch) for the plot (int, optional)
    - title: title for the plot (string, optional)
    """

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(data,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.1f',
                     annot_kws={'size': 10},
                     yticklabels=y_labels,
                     xticklabels=x_labels)

    plt.title(title if title is not None else 'Heatmap')
    plt.tight_layout()

    # Create a legend for the two sets of labels
    legend_labels = ['Wind', 'PV']
    legend_colors = ['blue', 'red']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    plt.legend(legend_handles, legend_labels)

    # Set color for x-axis labels
    half = len(x_labels) // 2
    for i, xtick in enumerate(hm.get_xticklabels()):
        xtick.set_color('blue' if i < half else 'red')

    # Set color for y-axis labels
    for i, ytick in enumerate(hm.get_yticklabels()):
        ytick.set_color('blue' if i < half else 'red')

    save_or_show_plot(save_path,dpi)

def plot_country_data(time, country_data, country_names, plot_title, yaxis_label, save_path=None, dpi = 350):
    
    """
    Function to plot country-specific data over time.

    Parameters:
    - time: List of time values (list)
    - country_data: 2D numpy array of country-specific data (shape: (m, n))
    - country_names: List of country names (list)
    - plot_title: Title for the plot (string)
    - yaxis_label: Label for the y-axis (string)
    - dpi: DPI (dots per inch) for the plot (int, optional)
    - save_path: Path to save the plot (string, optional)
    """

    # Create a larger figure with adjusted width
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Convert dates to 'YYYY' format only if the format is YYYY-MM-DD
    years = [date[:4] for date in time]

    # Define a list of marker shapes and colors
    markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x']

    # Plot each country's data with a unique color and shape
    for i, country in enumerate(country_data.T):
        marker = markers[i % len(markers)]  # Assign a marker from the list
        plt.plot(years, country, marker=marker, label=country_names[i])

    # Customize the plot
    plt.xlabel('Year')
    plt.ylabel(yaxis_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid()

    # Rotate the x-axis tick labels vertically
    plt.xticks(rotation='vertical')

    # Increase spacing between the years on the x-axis
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # Set the interval between labels to 5 years

    # Format the y-axis values with spaced thousands separator
    formatter = ticker.StrMethodFormatter('{x:,.2f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    # Adjust the legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    save_or_show_plot(save_path,dpi)

    # Close the plot
    plt.close()

def plot_histograms(country_data, country_names, xlabel, type_of_data, save_path=None):   

    """
    Function to plot histograms with KDE curves for each column of country-specific data.

    Parameters:
    - country_data: 2D numpy array of country-specific data (shape: (m, n))
    - country_names: List of country names (list)
    - xlabel: Label for the x-axis (string)
    - type_of_data: Type of data (string)
    - save_path: Path to save the plots (string, optional)
    """

    # Set a nicer style using Seaborn
    sns.set(style='ticks')
    
    # Plot histograms with KDE curves for each column
    for i, column in enumerate(country_names):        
        # Create a combined histogram and KDE plot
        plt.figure()
        sns.histplot(country_data[:, i], kde=True, color='blue', edgecolor='white')
        
        # Set plot title and labels
        plt.title(f'Histogram of {column}', fontsize=14)
        plt.xlabel(f'{xlabel} {column}', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add a grid
        plt.grid(axis='y', alpha=0.5)
        
        # Show or save the plot
        if save_path is not None:
            plt.savefig(f'{save_path}/{column}_histogram_{type_of_data}.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_histograms_column(country_data, country_names, xlabel, type_of_data, save_path=None):

    """
    Function to plot histograms with KDE curves for each column of country-specific data.

    Parameters:
    - country_data: 2D numpy array of country-specific data (shape: (m, n))
    - country_names: List of country names (list)
    - xlabel: Label for the x-axis (string)
    - type_of_data: Type of data (string)
    - save_path: Path to save the plot (string, optional)
    """

    # Set a nicer style using Seaborn
    sns.set(style='ticks')

    # Create a figure with subplots for each country's histogram
    fig, axes = plt.subplots(len(country_names), 1, figsize=(8, 5 * len(country_names)), sharex=True)

    # Plot histograms with KDE curves for each country
    for i, (data, country) in enumerate(zip(country_data.T, country_names)):
        # Select the subplot for the current country
        ax = axes[i]

        # Create a histogram plot with KDE
        sns.histplot(data, bins='auto', kde=True, color='blue', edgecolor='white', ax=ax)

        # Set plot title and labels
        ax.set_title(f'Histogram of {country}', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=12)

        # Add a grid
        ax.grid(axis='y', alpha=0.5)

    # Set the x-axis label for the last subplot
    axes[-1].set_xlabel(xlabel, fontsize=12)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show or save the plot
    if save_path is not None:
        # Iterate over all subplots and show x-axis labels
        for ax in axes:
            ax.tick_params(labelbottom=True)
        plt.savefig(f'{save_path}/{type_of_data}_histograms.png', bbox_inches='tight')
        plt.close()
    else:
        
        plt.show()
