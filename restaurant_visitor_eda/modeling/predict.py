from pathlib import Path

from catboost import CatBoostRegressor
from loguru import logger
import numpy as np
import pandas as pd
import typer

from restaurant_visitor_eda.config import MODELS_DIR, PROCESSED_DATA_DIR, SUMB_DIR
from restaurant_visitor_eda.features import (
    binary_features,
    categorical_features,
    numeric_features,
)

app = typer.Typer()


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "catboost_optuna_final.cb",
    output_path: Path = SUMB_DIR / "submission_catboost_pipeline.csv",
) -> None:
    logger.info(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path, parse_dates=["visit_date"])

    features = categorical_features + numeric_features + binary_features
    X_test = df_test[features]

    logger.info(f"Loading trained CatBoost model from {model_path}...")
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    logger.info("Generating predictions...")
    preds_log = model.predict(X_test)
    preds_real = np.expm1(preds_log)
    preds_clipped = np.clip(preds_real, 1.0, None)

    logger.info(f"Creating submission file {output_path}...")
    submission = pd.DataFrame(
        {
            "id": df_test["air_store_id"] + "_" + df_test["visit_date"].dt.strftime("%Y-%m-%d"),
            "visitors": preds_clipped,
        }
    )

    submission.to_csv(output_path, index=False)
    logger.success(f"Predictions saved successfully! Shape: {submission.shape}")


if __name__ == "__main__":
    app()
