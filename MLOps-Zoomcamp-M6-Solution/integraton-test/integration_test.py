## pycode/predict_batch_S3.py
#!/usr/bin/env python
# coding: utf-8

import os
import sys
## Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the save_data function
from pycode.predict_batch_S3 import save_to_s3, prepare_data, read_data

import logging
import pandas as pd
from datetime import datetime

## Set Docker environment variable
os.environ['DOCKER_ENV'] = '1'

## Setting Up Environment Variables
os.environ['AWS_ACCESS_KEY_ID'] = "test"
os.environ['AWS_SECRET_ACCESS_KEY'] = "test"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ['S3_ENDPOINT_URL'] = f"http://{'host.docker.internal' if os.getenv('DOCKER_ENV') else 'localhost'}:4566"  # Localstack S3 endpoint
# os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/yellow_tripdata_{year:04d}-{month:02d}.parquet"
# os.environ['OUTPUT_FILE_PATTERN'] = "s3://nyc-duration/out/yellow_tripdata_{year:04d}-{month:02d}.parquet"


def dt(hour, minute, second=0):
    """
    Helper function to create a datetime object for the given hour, minute, and second.
    All datetime objects are set to January 1, 2023.
    
    Args:
    hour (int): Hour part of the datetime.
    minute (int): Minute part of the datetime.
    second (int, optional): Second part of the datetime. Defaults to 0.
    
    Returns:
    datetime: Datetime object with the specified time.
    """
    return datetime(2023, 1, 1, hour, minute, second)

def create_test_data():
    ## Create a DataFrame (df) as described in Q3
    ## Pretend it's data for January 2023
    ## Sample data representing pickup and dropoff locations and times
    data = [
        (None, None, dt(1, 1), dt(1, 10)),        # None values for location IDs
        (1, 1, dt(1, 2), dt(1, 10)),             # Valid trip with duration of 8 minutes
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),    # Very short trip (less than a minute)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),        # Trip longer than an hour (should be excluded)
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    # Prepare data
    categorical = ['PULocationID', 'DOLocationID']
    df = prepare_data(df, categorical)
    
    ## Assuming `output_file` and `options` are defined appropriately for localstack S3
    input_file = 's3://nyc-duration/in/yellow_tripdata_2023-01.parquet'  # Use INPUT_FILE_PATTERN here
    
    ## Save dataframe to S3
    save_to_s3(df, input_file)
    print(f"File saved to {input_file}")
    return None

if __name__ == "__main__":
    create_test_data()

    # Run the batch script
    os.system("python pycode/predict_batch_S3.py 2023 01")

    # Define output path and read the result
    output_file = "s3://nyc-duration/out/yellow_tripdata_2023-01.parquet"
    df_output = read_data(output_file)

    # Calculate the sum of predicted durations
    sum_predicted_durations = df_output['predicted_duration'].sum()
    print(f"Sum of predicted durations: {sum_predicted_durations}")

    # Verify the result
    assert abs(sum_predicted_durations - 36.28) < 1e-2, "Test failed: The sum of predicted durations is incorrect."
