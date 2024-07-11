## tests/test_batch.py
#!/usr/bin/env python
# coding: utf-8

import os
import sys
## Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest
import logging
import pandas as pd
from datetime import datetime
from pycode.predict_batch import prepare_data


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

def test_prepare_data():
    """
    Test the prepare_data function to ensure it correctly processes and transforms the input DataFrame.
    
    The test creates a sample DataFrame with test data, processes it using the prepare_data function, 
    and then compares the actual output with the expected output.
    """
    # Sample data representing pickup and dropoff locations and times
    data = [
        (None, None, dt(1, 1), dt(1, 10)),        # None values for location IDs
        (1, 1, dt(1, 2), dt(1, 10)),             # Valid trip with duration of 8 minutes
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),    # Very short trip (less than a minute)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),        # Trip longer than an hour (should be excluded)
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    # Define the categorical columns
    categorical = ['PULocationID', 'DOLocationID']
    
    # Process the DataFrame using the prepare_data function
    actual_df = prepare_data(df, categorical)
    
    # Define the expected output data after processing
    expected_data = [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': dt(1, 1), 'tpep_dropoff_datetime': dt(1, 10), 'duration': 9.0},
        {'PULocationID': '1', 'DOLocationID': '1', 'tpep_pickup_datetime': dt(1, 2), 'tpep_dropoff_datetime': dt(1, 10), 'duration': 8.0}
    ]
    expected_df = pd.DataFrame(expected_data)
    
    # Print the actual and expected DataFrames
    print("Actual DataFrame:")
    print(actual_df)
    
    print("\nExpected DataFrame:")
    print(expected_df)
    
    # Use Pandas testing utility to compare the actual and expected DataFrames
    pd.testing.assert_frame_equal(actual_df, expected_df)

if __name__ == "__main__":
    # pytest.main()    
    test_prepare_data()
