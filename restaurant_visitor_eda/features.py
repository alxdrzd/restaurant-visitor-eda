from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from restaurant_visitor_eda.config import PROCESSED_DATA_DIR

app = typer.Typer()


def build_features_advanced(df_old: pd.DataFrame) -> pd.DataFrame:
    df = df_old.copy()

    df["doy_sin"] = np.sin(2 * np.pi * df["visit_date"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["visit_date"].dt.dayofyear / 365.25)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)

    gw_2016 = (df["visit_date"] >= "2016-04-29") & (df["visit_date"] <= "2016-05-05")
    gw_2017 = (df["visit_date"] >= "2017-04-29") & (df["visit_date"] <= "2017-05-05")
    df["is_gw"] = (gw_2016 | gw_2017).astype(np.int8)

    return df


def add_visitor_aggregations(df_old: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    df = df_old.copy()

    store_dow_stats = (
        df_train.groupby(["air_store_id", "day_of_week"])["visitors"]
        .agg(
            median_visitors_dow="median",
            mean_visitors_dow="mean",
            min_visitors_dow="min",
            max_visitors_dow="max",
        )
        .reset_index()
    )

    store_stats = (
        df_train.groupby("air_store_id")["visitors"]
        .agg(median_visitors_total="median")
        .reset_index()
    )

    df = pd.merge(df, store_dow_stats, on=["air_store_id", "day_of_week"], how="left")
    df = pd.merge(df, store_stats, on=["air_store_id"], how="left")

    for col in [
        "median_visitors_dow",
        "mean_visitors_dow",
        "min_visitors_dow",
        "max_visitors_dow",
    ]:
        df[col] = df[col].fillna(df["median_visitors_total"])

    gw_2016_train = df_train[
        (df_train["visit_date"] >= "2016-04-29") & (df_train["visit_date"] <= "2016-05-05")
    ]

    stores_with_gw_history = set(gw_2016_train["air_store_id"].unique())
    df["has_gw_history"] = df["air_store_id"].isin(stores_with_gw_history).astype(np.int8)

    gw_genre_geo_stats = (
        gw_2016_train.groupby(["air_genre_name", "prefecture"])["visitors"]
        .agg(gw_genre_geo_median="median")
        .reset_index()
    )

    df = pd.merge(df, gw_genre_geo_stats, on=["air_genre_name", "prefecture"], how="left")

    gw_genre_global = gw_2016_train.groupby("air_genre_name")["visitors"].median().to_dict()
    df["gw_genre_geo_median"] = df["gw_genre_geo_median"].fillna(
        df["air_genre_name"].map(gw_genre_global)
    )
    df["gw_genre_geo_median"] = df["gw_genre_geo_median"].fillna(
        gw_2016_train["visitors"].median()
    )

    reserve_dow = (
        df_train.groupby(["air_store_id", "day_of_week"])["reserve_visitors"]
        .agg(median_reserve_visitors_dow="median")
        .reset_index()
    )

    df = pd.merge(df, reserve_dow, on=["air_store_id", "day_of_week"], how="left")
    df["median_reserve_visitors_dow"] = df["median_reserve_visitors_dow"].fillna(0)

    df["walk_in_ratio"] = df["median_visitors_dow"] / (df["median_reserve_visitors_dow"] + 1)

    return df


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_baseline.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_baseline.csv",
    output_train: Path = PROCESSED_DATA_DIR / "train_features.csv",
    output_test: Path = PROCESSED_DATA_DIR / "test_features.csv",
) -> None:
    logger.info("Loading baseline datasets...")
    df_train = pd.read_csv(train_path, parse_dates=["visit_date"])
    df_test = pd.read_csv(test_path, parse_dates=["visit_date"])

    logger.info("Applying Calendar Base Features...")
    df_train = build_features_advanced(df_train)
    df_test = build_features_advanced(df_test)

    logger.info("Calculating Leak-Safe Aggregations...")
    df_train_feat = add_visitor_aggregations(df_train, df_train)
    df_test_feat = add_visitor_aggregations(df_test, df_train)

    logger.info("Saving engineered features...")
    df_train_feat.to_csv(output_train, index=False)
    df_test_feat.to_csv(output_test, index=False)

    logger.success(f"Features ready! Train: {df_train_feat.shape} | Test: {df_test_feat.shape}")


if __name__ == "__main__":
    app()
