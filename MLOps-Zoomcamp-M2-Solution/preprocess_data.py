# Source: https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2023/02-experiment-tracking/homework/preprocess_data.py
import os
import click
import pickle
import urllib.request
from glob import glob

# import pathlib
# import argparse
# import requests
# from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def download_data(raw_data_path: str, services: list, years: list, months: list) -> None:
    """Fetches data from the NYC Taxi dataset and saves it locally"""
    ## Specify the directory to save the files
    raw_data_path = './data'
    os.makedirs(raw_data_path, exist_ok=True)

    ## Download data use the green taxi trips
    URLs = []  # ['https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv']
    for service in services:
        for year in years:
            for month in months:
                ## Define URLs (Uniform Resource Locators) for the data files
                url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{service}_tripdata_{year}-{month:02d}.parquet"
                URLs.append(url)

    for url in URLs:    
        ## Extract filename from the URLs
        filename = os.path.basename(url)
        filepath = f"{raw_data_path}/{filename}"       

        ## Download via `urllib.request`
        urllib.request.urlretrieve(url, filepath)
        print(f"File downloaded to: {filepath}")
    return None


def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(filename)

    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['lpep_pickup_datetime']  = pd.to_datetime(df['lpep_pickup_datetime'])

    df["duration"] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df["duration"] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical     = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer = None, fit_dv: bool = False) -> tuple:
    """Add features to the model"""
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ["PU_DO"]
    numerical   = ['trip_distance']
    dicts       = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        ## Return sparse matrix
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    try:
        ## Extract the target
        target = 'duration'
        y = df[target].values
    except:
        pass
        
    ## Convert X the sparse matrix  to pandas DataFrame, but too slow
    # X = pd.DataFrame(X.toarray(), columns=dv.get_feature_names_out())
    # X = pd.DataFrame.sparse.from_spmatrix(X, columns=dv.get_feature_names_out())
    return (X, y), dv


def dump_pickle(obj, filename: str, dest_path: str) -> None:
    file_path = os.path.join(dest_path, filename)
       
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    with open(file_path, "wb") as f_out:
        return pickle.dump(obj, f_out)
                
                
@click.command()
@click.option(
    "--raw_data_path",
    default="./data",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    default="./output",
    help="Location where the resulting model files will be saved"
)
@click.option(
    "--services",
    default="green",
    help="Colors where the raw NYC taxi trip data was saved (space-separated)"
)
@click.option(
    "--years",
    default="2023",
    help="Years where the raw NYC taxi trip data was saved (space-separated)"
)
@click.option(
    "--months",
    default="1 2 3",
    help="Months where the raw NYC taxi trip data was saved (space-separated)"
)
def run_data_prep(raw_data_path: str, dest_path: str, services: str, years: str, months: str) -> None:
    """The main preprocess pipeline"""
    ## Parameters
    services = services.split()
    years    = [int(year) for year in years.split()]
    months   = [int(month) for month in months.split()]

    ## Download data  
    download_data(raw_data_path, services, years, months)
    # print(sorted(glob(f'{raw_data_path}/*')))
    
    ## Read parquet files
    df_train = read_data(
        os.path.join(raw_data_path, f"{services[0]}_tripdata_{years[0]}-{months[0]:0>2}.parquet")
    )
    df_val = read_data(
        os.path.join(raw_data_path, f"{services[0]}_tripdata_{years[0]}-{months[1]:0>2}.parquet")
    )
    df_test = read_data(
        os.path.join(raw_data_path, f"{services[0]}_tripdata_{years[0]}-{months[2]:0>2}.parquet")
    )
    # print(df_train.shape, df_val.shape, df_test.shape, )  

    ## Fit the DictVectorizer and preprocess data
    (X_train, y_train), dv = preprocess(df_train, fit_dv=True)
    (X_val, y_val)    , _  = preprocess(df_val, dv, fit_dv=False)
    (X_test, y_test)  , _  = preprocess(df_test, dv, fit_dv=False)    
    # print((X_train.shape, y_train.shape))
    # print((X_val.shape, y_val.shape))
    # print((X_test.shape, y_test.shape))

    ## Save DictVectorizer and datasets
    dump_pickle(dv, "dv.pkl", dest_path)
    dump_pickle((X_train, y_train), "train.pkl", dest_path)
    dump_pickle((X_val, y_val), "val.pkl", dest_path)
    dump_pickle((X_test, y_test), "test.pkl", dest_path)
    return None


if __name__ == '__main__':
    run_data_prep()
