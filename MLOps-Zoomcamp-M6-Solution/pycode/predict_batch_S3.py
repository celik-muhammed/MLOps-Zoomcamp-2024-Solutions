## pycode/predict_batch_S3.py
#!/usr/bin/env python
# coding: utf-8

# from typing import Any
import os
import sys
# import s3fs
import pickle
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

## Set Docker environment variable
os.environ['DOCKER_ENV'] = '1'

## Setting Up Environment Variables
os.environ['AWS_ACCESS_KEY_ID'] = "test"
os.environ['AWS_SECRET_ACCESS_KEY'] = "test"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ['S3_ENDPOINT_URL'] = f"http://{'host.docker.internal' if os.getenv('DOCKER_ENV') else 'localhost'}:4566"  # Localstack S3 endpoint
os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/yellow_tripdata_{year:04d}-{month:02d}.parquet"
os.environ['OUTPUT_FILE_PATTERN'] = "s3://nyc-duration/out/yellow_tripdata_{year:04d}-{month:02d}.parquet"


def load_pickle(file_path: str) -> tuple:
    """
    Load a pre-trained model from the specified path.
    """
    with open(file_path, 'rb') as f_in:
        return pickle.load(f_in)

def get_input_path(year, month):
    ## Get input pattern from environment variable or use default
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    ## Get output pattern from environment variable or use default
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def read_data(data_path):
    """
    Read data from a specified file path.
    """
    ## Check if S3 endpoint URL is set
    if 'S3_ENDPOINT_URL' in os.environ:
        options = {
            ## S3 storage options
            'storage_options': {
                'client_kwargs': {
                    'endpoint_url': os.getenv('S3_ENDPOINT_URL')  # Localstack endpoint
                }
            }
        }
        ## Reading from Localstack S3 with Pandas
        df = pd.read_parquet(data_path, **options)
    else:
        df = pd.read_parquet(data_path)
    
    return df
    
def prepare_data(df: pd.DataFrame, categorical) -> pd.DataFrame:
    """
    Preprocess the data (e.g., handle missing values, scale features).
    """
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])

    df["duration"] = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    # df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # df[categorical] = df[categorical].astype(str)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def preprocess_data(df: pd.DataFrame, year, month) -> pd.DataFrame:
    """
    Preprocess the data (e.g., handle missing values, scale features).
    """
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def make_prediction(df: pd.DataFrame, dv, model, categorical) -> np.ndarray:
    """
    Use the loaded model to make predictions on the preprocessed data.
    """
    dicts  = df[categorical].to_dict(orient='records')
    X_val  = dv.transform(dicts)
    
    y_pred = model.predict(X_val)
    return y_pred

def save_to_prediction(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Save the data with predictions.
    """
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    return df_result

def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the data to a Parquet file at the specified output path.
    """
    ## Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    
    ## Save dataframe to Parquet file
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return None

def save_to_s3(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a dataframe to a specified path using given storage options.
    """    
    ## s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': os.getenv('S3_ENDPOINT_URL')})
    ## Check if S3 endpoint URL is set
    if 'S3_ENDPOINT_URL' in os.environ:
        options = {
            ## S3 storage options
            'storage_options': {
                'client_kwargs': {
                    'endpoint_url': os.getenv('S3_ENDPOINT_URL')  # Localstack endpoint
                }
            }
        }
    ## Save dataframe to Parquet file
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False,
        **options
    )
    return None

def run_prediction_pipeline(year, month) -> None:
    """
    Run the entire prediction pipeline: load model, read data, preprocess data, 
    make predictions, and save results.
    """
    ## Define categorical columns
    categorical = ['PULocationID', 'DOLocationID']
    
    ## Parameters
    # data_path   = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    # output_path = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    data_path   = get_input_path(year, month)
    output_path = get_output_path(year, month)
    logging.info(f"input_file : {data_path}")
    logging.info(f"output_file: {output_path}")
    
    # Check if the file exists
    model_path = os.getenv('MODEL_PATH', 'model/model.bin')
    if not os.path.exists(model_path):
        model_path = 'model.bin'
    
    # 1. Load model
    dv, lr = load_pickle(model_path)
    # 2. Read data
    df = read_data(data_path)
    logging.info(f"shape : {df.shape}")
    
    # 3. Prepare data
    df = prepare_data(df, categorical)
    # 4. Preprocess data
    df = preprocess_data(df, year, month)        
    # 5. Make prediction
    y_pred = make_prediction(df, dv, lr, categorical)
    # Print Prediction
    print('predicted mean duration:', y_pred.mean().round(2))
    
    # 6. Save prediction to df
    df = save_to_prediction(df, y_pred)
    # 7. Save results to Parquet
    # save_to_parquet(df, output_path)
    save_to_s3(df, output_path)

    return None


if __name__ == '__main__':
    ## Parameters
    year  = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3
    print(year, month)
    
    ## Runs the entire prediction pipeline
    run_prediction_pipeline(year, month)
