
import os
import click
import pickle
import scipy
import numpy as np
# from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


def load_pickle(
    data_path: str, filename: str
) -> tuple([scipy.sparse._csr.csr_matrix, np.ndarray]):
    ## Create data_path folder unless it already exists
    file_path = os.path.join(data_path, filename)
    with open(file_path, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params, experiment_name): 
    """The main training pipeline"""    
    # Load train, val and test Data
    X_train, y_train = load_pickle(data_path, "train.pkl")
    X_val, y_val     = load_pickle(data_path, "val.pkl")
    X_test, y_test   = load_pickle(data_path, "test.pkl")
    # print(type(X_train), type(y_train))
    
    # MLflow settings
    # Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Build or Connect mlflow experiment
    EXPERIMENT_NAME = experiment_name
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # before your training code to enable automatic logging of sklearn metrics, params, and models
    mlflow.sklearn.autolog()

    with mlflow.start_run(nested=True):
        # Optional: Set some information about Model
        mlflow.set_tag("developer", "muce")
        mlflow.set_tag("algorithm", "Machine Learning")
        mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
        mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
        mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')  

        # Set Model params information
        RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
        for param in RF_PARAMS:
            params[param] = int(params[param])
            
        # Log the model params to the tracking server
        mlflow.log_params(params)

        # Build Model
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Set Model Evaluation Metric
        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        # print("test_rmse", test_rmse)
        
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the model
        # Option1: Just only model in log
        mlflow.sklearn.log_model(sk_model = rf, artifact_path = "model_mlflow")
        
        # print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int) -> None:
    """The main optimization pipeline"""
    # Parameters
    HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
    EXPERIMENT_NAME     = "random-forest-best-models"
    client = MlflowClient("sqlite:///mlflow.db")

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.val_rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params, experiment_name=EXPERIMENT_NAME)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id     = best_run.info.run_id
    model_uri  = f"runs:/{run_id}/model"
    model_name = "rf-best-model"
    mlflow.register_model(model_uri, name=model_name)

    print("Test RMSE of the best model: {:.4f}".format(best_run.data.metrics["test_rmse"]))
    return None


if __name__ == '__main__':
    run_register_model()
