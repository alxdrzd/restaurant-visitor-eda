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


def get_daily_reservations(raw_dir: Path) -> pd.DataFrame:
    air_res = pd.read_csv(raw_dir / "air_reserve.csv")
    hpg_res = pd.read_csv(raw_dir / "hpg_reserve.csv")
    relation = pd.read_csv(raw_dir / "store_id_relation.csv")

    hpg_res = pd.merge(hpg_res, relation, on="hpg_store_id", how="inner")
    hpg_res = hpg_res.drop(columns=["hpg_store_id"])

    all_res = pd.concat([air_res, hpg_res], ignore_index=True)
    all_res["visit_date"] = pd.to_datetime(all_res["visit_datetime"]).dt.normalize()

    daily_res = (
        all_res.groupby(["air_store_id", "visit_date"])["reserve_visitors"].sum().reset_index()
    )
    return daily_res


def process_base_dataframe(
    df_base: pd.DataFrame,
    df_store: pd.DataFrame,
    df_date_processed: pd.DataFrame,
    df_reservations: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.merge(df_base, df_store, on="air_store_id", how="left")

    df = pd.merge(
        df, df_date_processed, left_on="visit_date", right_on="calendar_date", how="left"
    )
    df.drop(columns=["calendar_date"], inplace=True)

    df[["prefecture", "district", "block"]] = (
        df["air_area_name"].str.strip().str.split(" ", n=2, expand=True)
    )

    df = pd.merge(df, df_reservations, on=["air_store_id", "visit_date"], how="left")
    df["reserve_visitors"] = df["reserve_visitors"].fillna(0)

    return df.sort_values("visit_date").reset_index(drop=True)


@app.command()
def main(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    logger.info("Reading raw data...")
    df_visit = pd.read_csv(raw_dir / "air_visit_data.csv")
    df_store = pd.read_csv(raw_dir / "air_store_info.csv")
    df_date = pd.read_csv(raw_dir / "date_info.csv")
    df_sub = pd.read_csv(raw_dir / "sample_submission.csv")

    logger.info("Formatting dates & Parsing Test set...")
    df_visit["visit_date"] = pd.to_datetime(df_visit["visit_date"])

    df_sub["air_store_id"] = df_sub["id"].str.rsplit("_", n=1).str[0]
    df_sub["visit_date"] = pd.to_datetime(df_sub["id"].str.rsplit("_", n=1).str[1])
    df_test_base = df_sub[["air_store_id", "visit_date", "visitors"]].copy()

    logger.info("Generating date info and extracting reservations...")
    df_date_processed = build_date_features(df_date)
    df_reservations = get_daily_reservations(raw_dir)

    logger.info("Merging tables for TRAIN and TEST...")
    df_train = process_base_dataframe(df_visit, df_store, df_date_processed, df_reservations)
    df_test = process_base_dataframe(df_test_base, df_store, df_date_processed, df_reservations)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(processed_dir / "train_baseline.csv", index=False)
    df_test.to_csv(processed_dir / "test_baseline.csv", index=False)

    logger.success(f"Saved: Train {df_train.shape} | Test {df_test.shape}")


if __name__ == "__main__":
    app()
