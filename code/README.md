# global_problem.py

This script is the primary entry point for our project. Here's how it works and what you can configure.

## Overview

The script primarily does the following:

1. **Data Processing:** It loads various data files such as wind data, PV data, demand data, and max capacities for wind and PV. Then, it optionally aggregates, sorts, and filters the data.

2. **Importing the Data:** It reads the processed data files and computes the mean and covariance of the data.

3. **Configuration of Parameters:** Set parameters such as epsilon and c_bar (maximum total capacity for all technologies).

4. **Calculations and Plots:** It uses the solver functions to calculate the optimal capacities, then generates various plots for visual inspection.

## Configurable Parameters

You can adjust several parameters in the script:

1. `k`: The number of countries to delete. Range should be between 1 and 24. If k=0, no country will be deleted.

2. `epsilon`: This is a parameter for the solver function, representing the risk aversion of the decision-maker.

3. `c_bar`: This is the total capacity for all technologies (wind and PV). This can be set as a fixed number, or calculated based on the loaded data.

In the code, you can adjust these parameters directly.

## Optional Functionality

Various parts of the script are commented out and can be enabled if desired:

1. **Data Processing:** You can uncomment lines to perform data processing steps like aggregating data annually and sorting CSV columns.

2. **Plotting:** The script has various plot functions commented out. If you want to visualize your data or results, you can uncomment these sections. The available plots are:
    - `draw_map_plot`: This function is used to **visualize the result of the optimization**, which represents the allocation of wind and PV capacities for each country.
    - `draw_map_plot_comparaison`: This function is used to **visualize the difference between a benchmark scenario and other scenario** which is plotted on the same map.
    - `plot_country_data`: Displays the original wind and PV capacity data.
    - `plot_text_heatmap`: Shows the correlation heatmap of wind and PV capacity data with a colormap.
    - `plot_text_heatmap`: Displays the correlation heatmap of wind and PV capacity data with a color map and the value for each cell.
    - `plot_histograms`: Displays the histogram for a sample of data with the KDE (kernel density estimate) for **each country**.
    - `plot_histograms_column`: Same plot as `plot_histograms`, but it displays **every country** on the same plot.

3. **Testing Values:** If you want to test the functionality with random data, you can uncomment the section under "TESTING VALUES".

4. **Different Solver Functions:** You can choose between different solver functions: `cvxpy_solver`, `scipy_solver`, and `scipy_solver_penalty`. Currently, `scipy_solver` is used, but you can comment out and uncomment lines as needed to switch between them.

## How to Run

To run the script, simply execute it using a Python interpreter. Make sure the data files are in the correct locations, and the necessary libraries are installed.

## Available Plots

1. **plot_country_data:** Displays the original wind and PV capacity data.

2. **plot_text_heatmap:** Shows the correlation heatmap of wind and PV capacity data with a colormap.

3. **plot_text_heatmap:** Displays the correlation heatmap of wind and PV capacity data with a color map and the value for each cell.

4. **plot_histograms:** Displays the histogram for a sample of data with the KDE (kernel density estimate) for **each country**.

5. **plot_histograms_column:** Same plot as `plot_histograms`, but it displays **every country** verticaly on the same plot.

6. **draw_map_plot:** Visualizes the result of the optimization, which represents the allocation of wind and PV capacities for each country.

7. **draw_map_plot_comparaison:** Visualizes the difference between a benchmark scenario and other scenarios, which are plotted on the same map.

Please note that some plots might require uncommenting specific sections in the code. Feel free to utilize these plots to visualize and interpret your data or results effectively.
