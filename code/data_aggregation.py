import pandas as pd

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


# Provide the paths for input and output files
input_csv = '../data/demand_data_hourly.csv'
output_csv = '../data/demand_data_annual.csv'

# Call the function to aggregate the data and create the output CSV
aggregate_data_annual(input_csv, output_csv)
