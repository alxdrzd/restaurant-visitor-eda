from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from restaurant_visitor_eda.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def build_date_features(df_date: pd.DataFrame) -> pd.DataFrame:
    df_date = df_date.copy()
    df_date["calendar_date"] = pd.to_datetime(df_date["calendar_date"])
    df_date = df_date.sort_values("calendar_date")

    df_date["month"] = df_date["calendar_date"].dt.month

    df_date["day_of_week_num"] = df_date["calendar_date"].dt.dayofweek

    df_date["is_weekend"] = df_date["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df_date["is_off_day"] = ((df_date["holiday_flg"] == 1) | (df_date["is_weekend"] == 1)).astype(
        int
    )

    df_date["next_is_off_day"] = df_date["is_off_day"].shift(-1).fillna(0).astype(int)

    conditions = [
        (df_date["is_off_day"] == 1) & (df_date["next_is_off_day"] == 1),
        (df_date["is_off_day"] == 1) & (df_date["next_is_off_day"] == 0),
        (df_date["is_off_day"] == 0) & (df_date["next_is_off_day"] == 1),
        (df_date["is_off_day"] == 0) & (df_date["next_is_off_day"] == 0),
    ]
    choices = ["Off-day & Off-day", "Off-day & Workday", "Workday & Off-day", "Workday & Workday"]
    df_date["day_pattern"] = np.select(conditions, choices, default="Unknown")

    cols_to_keep = [
        "calendar_date",
        "day_of_week",
        "day_of_week_num",
        "month",
        "holiday_flg",
        "is_off_day",
        "day_pattern",
    ]
    return df_date[cols_to_keep]


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    logger.info("1. Reading raw data from data/raw/ ...")
    df_visit = pd.read_csv(raw_dir / "air_visit_data.csv")
    df_store = pd.read_csv(raw_dir / "air_store_info.csv")
    df_date = pd.read_csv(raw_dir / "date_info.csv")

    logger.info("2. Dates formating...")
    df_visit["visit_date"] = pd.to_datetime(df_visit["visit_date"])

    logger.info("3. Feautres generation (day_pattern, month, etc)...")
    df_date_processed = build_date_features(df_date)

    logger.info("4. Merging tables...")
    df_merged = pd.merge(df_visit, df_store, on="air_store_id", how="left")

    df_merged = pd.merge(
        df_merged, df_date_processed, left_on="visit_date", right_on="calendar_date", how="left"
    )
    df_merged.drop(columns=["calendar_date"], inplace=True)

    df_merged[["prefecture", "district", "block"]] = (
        df_merged["air_area_name"].str.strip().str.split(" ", n=2, expand=True)
    )

    logger.info("5. Sorting by date...")
    df_merged = df_merged.sort_values("visit_date").reset_index(drop=True)

    output_file = processed_dir / "train_baseline.csv"
    logger.info(f"6. Saving processed data in {output_file} ...")

    processed_dir.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_file, index=False)

    logger.success(f"The final DataFrame shape is: {df_merged.shape}")


if __name__ == "__main__":
    app()
