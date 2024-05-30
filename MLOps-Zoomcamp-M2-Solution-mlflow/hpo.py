
import os
import pickle
import click
import scipy
import numpy as np
# from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from hyperopt.pyll import scope
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import mlflow


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
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int) -> None:
    """The main optimization pipeline"""
    ## Load train and test Data
    X_train, y_train = load_pickle(data_path, "train.pkl")
    X_val, y_val     = load_pickle(data_path, "val.pkl") 
    # print(type(X_train), type(y_train))
    
    ## MLflow settings
    ## Build or Connect Database Offline
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Connect Database Online
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Build or Connect mlflow experiment
    HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)
    
    # before your training code to disable automatic logging of sklearn metrics, params, and models
    mlflow.sklearn.autolog(disable=True)

    # Optional: Set some information about Model
    mlflow.set_tag("developer", "muce")
    mlflow.set_tag("algorithm", "Machine Learning")
    mlflow.set_tag("train-data-path", f'{data_path}/train.pkl')
    mlflow.set_tag("valid-data-path", f'{data_path}/val.pkl')
    mlflow.set_tag("test-data-path",  f'{data_path}/test.pkl')
    
        
    def objective(params):
        with mlflow.start_run(nested=True):
            # Log the model params to the tracking server
            mlflow.log_params(params)
            
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            # Log the validation RMSE to the tracking server
            y_val_pred = rf.predict(X_val)
            val_rmse   = root_mean_squared_error(y_val, y_val_pred)
            mlflow.log_metric("val_rmse", val_rmse)
        return {'loss': val_rmse, 'status': STATUS_OK}
    
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


if __name__ == '__main__':
    run_optimization()
