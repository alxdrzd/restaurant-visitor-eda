import json
from pathlib import Path

from catboost import CatBoostRegressor
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from restaurant_visitor_eda.config import MODELS_DIR, PROCESSED_DATA_DIR
from restaurant_visitor_eda.features import (
    binary_features,
    categorical_features,
    numeric_features,
)

app = typer.Typer()


def load_best_params(params_path: Path) -> dict:
    with open(params_path, "r") as f:
        params = json.load(f)

    params.update({"loss_function": "RMSE", "eval_metric": "RMSE", "random_seed": 42})
    return params


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    params_path: Path = MODELS_DIR / "best_catboost_params.json",
    model_path: Path = MODELS_DIR / "catboost_optuna_final.cb",
) -> None:
    logger.info(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path, parse_dates=["visit_date"])

    features = categorical_features + numeric_features + binary_features
    X_full = df_train[features]
    y_full = np.log1p(df_train["visitors"].values)

    logger.info(f"Loading best parameters from {params_path}...")
    best_params = load_best_params(params_path)

    mlflow.set_tracking_uri("sqlite:///mlflow_tracking.db")
    mlflow.set_experiment("CatBoost_Production")

    with mlflow.start_run(run_name="final_model_training"):
        logger.info("Training final model on full dataset...")
        model = CatBoostRegressor(**best_params, cat_features=categorical_features)

        model.fit(X_full, y_full, verbose=100)

        logger.info(f"Saving model to {model_path}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))

        mlflow.log_params(best_params)
        mlflow.catboost.log_model(model, "production_model")

    logger.success("Model successfully trained and saved!")


if __name__ == "__main__":
    app()
