import pandas as pd

# Provide the paths for input and output files

wind = '../data/wind_data_annual.csv'
pv = '../data/pv_data_annual.csv'
demand = '../data/demand_data_annual.csv'

def aggregate_data_daily(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert the 'time' column to a datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Set the 'time' column as the DataFrame's index
    df.set_index('time', inplace=True)

    # Resample the data on a daily basis and sum the values
    df_daily = df.resample('D').first()

    # Write the aggregated data to a new CSV file
    df_daily.to_csv(output_file)


def aggregate_data_annual(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert the 'time' column to a datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Set the 'time' column as the DataFrame's index
    df.set_index('time', inplace=True)

    # Resample the data on a yearly basis and sum the values
    df_yearly = df.resample('Y').first()

    # Write the aggregated data to a new CSV file
    df_yearly.to_csv(output_file)

def sort_csv_column(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Get column headers starting from the second column
    columns = list(df.columns)[1:]

    # Sort the column headers in alphabetical order
    sorted_columns = sorted(columns)

    # Reorganize the DataFrame with sorted columns
    sorted_df = df[['time'] + sorted_columns]

    # Modify the file names with "_matching" before the extension
    file_matching = input_file.replace(".csv", "_sorted.csv")

    # Save the sorted DataFrame to a new CSV file
    sorted_df.to_csv(file_matching, index=False)

def keep_matching_columns(file1, file2, file3):
    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    # Get the common columns present in all three DataFrames
    common_columns = sorted(set(df1.columns) & set(df2.columns) & set(df3.columns))
    
    # Ensure "time" column is the first column
    common_columns.remove("time")
    common_columns = ["time"] + common_columns

    # Save each DataFrame with only the common columns as a separate file
    df1_common = df1[list(common_columns)]
    df2_common = df2[list(common_columns)]
    df3_common = df3[list(common_columns)]
    
    # Modify the file names with "_matching" before the extension
    file1_matching = file1.replace(".csv", "_matching.csv")
    file2_matching = file2.replace(".csv", "_matching.csv")
    file3_matching = file3.replace(".csv", "_matching.csv")

    # Writing the data 
    df1_common.to_csv(file1_matching, index=False)
    df2_common.to_csv(file2_matching, index=False)
    df3_common.to_csv(file3_matching, index=False)

def delete_last_columns(input_file, n):
    # Read the input CSV file using pandas
    df = pd.read_csv(input_file)

    # Remove the last n columns
    modified_df = df.iloc[:, :-n]

    # Write the modified DataFrame to a new CSV file
    output_file = input_file.rstrip('.csv') + '_modified.csv'
    modified_df.to_csv(output_file, index=False)

# Call the function to aggregate the data and create the output CSV

#input_csv = '../data/demand_data_hourly.csv'
#output_csv = '../data/demand_data_annual.csv'
#aggregate_data_annual(input_csv, output_csv)


#sort_csv_column(wind)
#sort_csv_column(pv)
#sort_csv_column(demand)

#keep_matching_columns(wind,pv,demand)


k = 15 # n - k localization 

delete_last_columns('../data/demand_data_annual_matching.csv',k)
delete_last_columns('../data/pv_data_annual_matching.csv',k)
delete_last_columns('../data/wind_data_annual_matching.csv',k)
