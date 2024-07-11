
import os
import click
import pickle
import scipy
import numpy as np
# from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

import mlflow
# import warnings
# Ignore all warnings
# warnings.filterwarnings("ignore")
# Filter the specific warning message, MLflow autologging encountered a warning
# warnings.filterwarnings("ignore", category=UserWarning, module="setuptools")
# warnings.filterwarnings("ignore", category=UserWarning, message="Setuptools is replacing distutils.")


def load_pickle(
    data_path: str, filename: str
) -> tuple([scipy.sparse._csr.csr_matrix, np.ndarray]):
    ## Create data_path folder unless it already exists
    file_path = os.path.join(data_path, filename)
    with open(file_path, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    default="./model",
    help="Location where the resulting files will be saved"
)
def run_train(data_path: str, dest_path: str) -> None:
    """The main training pipeline""" 
    ## Load train and test Data
    X_train, y_train = load_pickle(data_path, "train.pkl")
    X_val, y_val     = load_pickle(data_path, "val.pkl")
    # print(type(X_train), type(y_train))

    ## MLflow settings
    ## Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    ## Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    ## Build or Connect mlflow experiment
    EXPERIMENT_NAME = "random-forest-train"
    mlflow.set_experiment(EXPERIMENT_NAME)
            
    ## before your training code to enable automatic logging of sklearn metrics, params, and models
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        # autolog_run = mlflow.last_active_run()

        ## Optional: Set some information about Model
        mlflow.set_tag("developer", "muce")
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
        mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
        mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')
        
        ## Set Model params information
        params = {"max_depth": 10, "random_state": 0}
        mlflow.log_params(params)
        
        ## Build Model
        rf     = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)        

        ## Set Model Evaluation Metric
        y_val_pred = rf.predict(X_val)
        val_rmse   = root_mean_squared_error(y_val, y_val_pred)
        mlflow.log_metric("val_rmse", val_rmse)
        # print("rmse", rmse)
                        
        ## Log Model two options
        ## Option1: Just only model in log
        mlflow.sklearn.log_model(sk_model= rf, artifact_path= "models_mlflow")
                
        ## Option 2: save Model, and Optional: Preprocessor or Pipeline in log
        ## Create dest_path folder unless it already exists
        # pathlib.Path(dest_path).mkdir(exist_ok=True)
        os.makedirs(dest_path, exist_ok=True)
        pickle_path = os.path.join(dest_path, "rf_model.pkl")
        with open(pickle_path, 'wb') as f_out:
            pickle.dump(rf, f_out)
            
        ## whole proccess like pickle, saved Model, Optional: Preprocessor or Pipeline
        mlflow.log_artifact(local_path= pickle_path, artifact_path="models_pickle")        
        
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


if __name__ == '__main__':
    run_train()
