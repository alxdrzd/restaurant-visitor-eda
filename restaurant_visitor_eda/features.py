from datetime import timedelta
from pathlib import Path

from catboost import CatBoostRegressor
from loguru import logger
import numpy as np
import pandas as pd
import typer

from restaurant_visitor_eda.config import PROCESSED_DATA_DIR

app = typer.Typer()


def build_calendar_rolling_features(
    df_train_param: pd.DataFrame, df_test_param: pd.DataFrame, windows: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_train_param.copy()
    df_test = df_test_param.copy()

    min_date, max_date = df_train["visit_date"].min(), df_train["visit_date"].max()

    stores = df_train["air_store_id"].unique()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    full_grid = (
        pd.MultiIndex.from_product([stores, all_dates], names=["air_store_id", "visit_date"])
        .to_frame()
        .reset_index(drop=True)
    )

    df_full = pd.merge(
        full_grid,
        df_train[["air_store_id", "visit_date", "visitors"]],
        on=["air_store_id", "visit_date"],
        how="left",
    )

    df_full = df_full.sort_values(["air_store_id", "visit_date"]).reset_index(drop=True)

    df_full["visitors_shifted_1D"] = df_full.groupby("air_store_id")["visitors"].shift(1)

    df_full = df_full.set_index("visit_date")

    for w in windows:
        col_name = f"store_roll_mean_{w}"
        df_full[col_name] = (
            df_full.groupby("air_store_id")["visitors_shifted_1D"]
            .rolling(f"{w}D", min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df_full = df_full.reset_index()

    last_values = (
        df_full[df_full["visit_date"] == max_date]
        .groupby("air_store_id")[[f"store_roll_mean_{w}" for w in windows]]
        .first()
        .reset_index()
    )

    rolling_cols = ["air_store_id", "visit_date"] + [f"store_roll_mean_{w}" for w in windows]
    df_train = pd.merge(
        df_train, df_full[rolling_cols], on=["air_store_id", "visit_date"], how="left"
    )

    df_test = pd.merge(df_test, last_values, on="air_store_id", how="left")

    return df_train, df_test


def build_base_calendar_features(df_old: pd.DataFrame) -> pd.DataFrame:
    df = df_old.copy()

    days_in_year = 365 + 1 * df_old["visit_date"].dt.is_leap_year

    df["doy_sin"] = np.sin(2 * np.pi * df["visit_date"].dt.dayofyear / days_in_year)
    df["doy_cos"] = np.cos(2 * np.pi * df["visit_date"].dt.dayofyear / days_in_year)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)

    gw_2016 = (df["visit_date"] >= "2016-04-29") & (df["visit_date"] <= "2016-05-05")
    gw_2017 = (df["visit_date"] >= "2017-04-29") & (df["visit_date"] <= "2017-05-05")
    df["is_gw"] = (gw_2016 | gw_2017).astype(np.int8)

    return df


def build_time_series_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train, df_test = build_calendar_rolling_features(df_train, df_test, windows=[14, 28])

    df_train = df_train.sort_values(["air_store_id", "visit_date"]).reset_index(drop=True)

    df_train["visitors_shifted"] = df_train.groupby("air_store_id")["visitors"].shift(1)
    df_train["visitors_dow_shifted"] = df_train.groupby(["air_store_id", "day_of_week"])[
        "visitors"
    ].shift(1)

    df_train["store_mean_cum"] = (
        df_train.groupby("air_store_id")["visitors_shifted"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_train["store_dow_mean_cum"] = (
        df_train.groupby(["air_store_id", "day_of_week"])["visitors_dow_shifted"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    daily_genre_geo = (
        df_train.groupby(["air_genre_name", "prefecture", "visit_date"])["visitors"]
        .mean()
        .reset_index()
    )
    daily_genre_geo = daily_genre_geo.sort_values(
        ["air_genre_name", "prefecture", "visit_date"]
    ).reset_index(drop=True)
    daily_genre_geo["genre_geo_shifted"] = daily_genre_geo.groupby(
        ["air_genre_name", "prefecture"]
    )["visitors"].shift(1)
    daily_genre_geo["genre_geo_mean_cum"] = (
        daily_genre_geo.groupby(["air_genre_name", "prefecture"])["genre_geo_shifted"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    daily_genre = (
        df_train.groupby(["air_genre_name", "visit_date"])["visitors"].mean().reset_index()
    )
    daily_genre = daily_genre.sort_values(["air_genre_name", "visit_date"]).reset_index(drop=True)
    daily_genre["genre_shifted"] = daily_genre.groupby(["air_genre_name"])["visitors"].shift(1)
    daily_genre["genre_mean_cum"] = (
        daily_genre.groupby(["air_genre_name"])["genre_shifted"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    daily_global = df_train.groupby("visit_date")["visitors"].mean().reset_index()
    daily_global["global_mean_cum"] = daily_global["visitors"].shift(1).expanding().mean()

    df_train = pd.merge(
        df_train,
        daily_genre_geo[["air_genre_name", "prefecture", "visit_date", "genre_geo_mean_cum"]],
        on=["air_genre_name", "prefecture", "visit_date"],
        how="left",
    )
    df_train = pd.merge(
        df_train,
        daily_genre[["air_genre_name", "visit_date", "genre_mean_cum"]],
        on=["air_genre_name", "visit_date"],
        how="left",
    )
    df_train = pd.merge(
        df_train, daily_global[["visit_date", "global_mean_cum"]], on="visit_date", how="left"
    )

    store_stats = (
        df_train.groupby("air_store_id")["visitors"]
        .mean()
        .reset_index()
        .rename(columns={"visitors": "store_mean_cum"})
    )
    store_dow_stats = (
        df_train.groupby(["air_store_id", "day_of_week"])["visitors"]
        .mean()
        .reset_index()
        .rename(columns={"visitors": "store_dow_mean_cum"})
    )

    genre_geo_stats = (
        df_train.groupby(["air_genre_name", "prefecture"])["visitors"]
        .mean()
        .reset_index()
        .rename(columns={"visitors": "genre_geo_mean_cum"})
    )
    genre_stats = (
        df_train.groupby("air_genre_name")["visitors"]
        .mean()
        .reset_index()
        .rename(columns={"visitors": "genre_mean_cum"})
    )

    df_test = pd.merge(df_test, store_stats, on="air_store_id", how="left")
    df_test = pd.merge(df_test, store_dow_stats, on=["air_store_id", "day_of_week"], how="left")
    df_test = pd.merge(df_test, genre_geo_stats, on=["air_genre_name", "prefecture"], how="left")
    df_test = pd.merge(df_test, genre_stats, on="air_genre_name", how="left")

    final_global_mean = df_train["visitors"].mean()

    df_train["genre_geo_mean_cum"] = (
        df_train["genre_geo_mean_cum"]
        .fillna(df_train["genre_mean_cum"])
        .fillna(df_train["global_mean_cum"])
    )

    df_test["genre_geo_mean_cum"] = (
        df_test["genre_geo_mean_cum"].fillna(df_test["genre_mean_cum"]).fillna(final_global_mean)
    )

    for d in [df_train, df_test]:
        d["store_mean_cum"] = d["store_mean_cum"].fillna(d["genre_geo_mean_cum"])
        d["store_dow_mean_cum"] = d["store_dow_mean_cum"].fillna(d["store_mean_cum"])
        d["store_roll_mean_14"] = d["store_roll_mean_14"].fillna(d["store_mean_cum"])
        d["store_roll_mean_28"] = d["store_roll_mean_28"].fillna(d["store_mean_cum"])

        d["reserve_visitors"] = d["reserve_visitors"].fillna(0)
        d["walk_in_ratio"] = d["store_dow_mean_cum"] / (d["reserve_visitors"] + 1)

        if "visitors_shifted" in d.columns:
            d.drop(
                columns=[
                    "visitors_shifted",
                    "visitors_dow_shifted",
                    "genre_mean_cum",
                    "global_mean_cum",
                ],
                inplace=True,
                errors="ignore",
            )
        else:
            d.drop(columns=["genre_mean_cum", "global_mean_cum"], inplace=True, errors="ignore")

    return df_train, df_test


def get_custom_cv_splits(df: pd.DataFrame, n_splits: int = 3, val_days: int = 39) -> list:
    splits = []
    max_date = df["visit_date"].max()

    for i in range(n_splits):
        val_end = max_date - timedelta(days=i * val_days)
        val_start = val_end - timedelta(days=val_days - 1)

        train_mask = df["visit_date"] < val_start
        val_mask = (df["visit_date"] >= val_start) & (df["visit_date"] <= val_end)

        train_idx = df.index[train_mask].tolist()
        val_idx = df.index[val_mask].tolist()

        splits.append((train_idx, val_idx))
        print(
            f"Fold {i + 1}: Train ends {val_start - timedelta(days=1):%Y-%m-%d}"
            + f"| Val: {val_start:%Y-%m-%d} to {val_end:%Y-%m-%d}"
        )

    return splits[::-1]


def compute_calendar_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    res_df = df.copy()

    for lag in lags:
        shifted = df[["air_store_id", "visit_date", "visitors"]].copy()
        shifted["visit_date"] = shifted["visit_date"] + pd.to_timedelta(lag, unit="D")
        shifted = shifted.rename(columns={"visitors": f"lag_{lag}"})

        res_df = pd.merge(res_df, shifted, on=["air_store_id", "visit_date"], how="left")

    return res_df


def predict_recursive(
    model: CatBoostRegressor,
    df_all: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    features: list[str],
) -> pd.DataFrame:
    df_all = df_all.copy()
    test_dates = pd.date_range(start=start_date, end=end_date)

    for current_date in test_dates:
        for lag in [1, 7, 14]:
            prev_date = current_date - pd.to_timedelta(lag, unit="D")

            prev_visitors = df_all.loc[
                df_all["visit_date"] == prev_date, ["air_store_id", "visitors"]
            ]

            current_mask = df_all["visit_date"] == current_date
            if current_mask.any():
                mapping = prev_visitors.set_index("air_store_id")["visitors"]
                df_all.loc[current_mask, f"lag_{lag}"] = df_all.loc[
                    current_mask, "air_store_id"
                ].map(mapping)

                df_all.loc[current_mask, f"lag_{lag}"] = (
                    df_all.loc[current_mask, f"lag_{lag}"]
                    .fillna(df_all.loc[current_mask, "store_dow_mean_cum"])
                    .fillna(df_all.loc[current_mask, "store_mean_cum"])
                )

        current_mask = df_all["visit_date"] == current_date
        if current_mask.any():
            X_current = df_all.loc[current_mask, features]

            log_preds = model.predict(X_current)

            preds = np.expm1(log_preds)

            preds = np.clip(preds, 0, None)
            df_all.loc[current_mask, "visitors"] = preds

    return df_all


categorical_features = [
    "air_store_id",
    "air_genre_name",
    "day_of_week",
    "month",
    "day_pattern",
    "prefecture",
    "district",
    "block",
]

numeric_features = [
    "latitude",
    "longitude",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "store_mean_cum",
    "store_dow_mean_cum",
    "store_roll_mean_14",
    # "store_roll_mean_28",
    "genre_geo_mean_cum",
    "reserve_visitors",
    # "walk_in_ratio",
]

binary_features = ["is_gw", "is_off_day", "holiday_flg"]


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

    logger.info("Applying Base Calendar Features...")
    df_train = build_base_calendar_features(df_train)
    df_test = build_base_calendar_features(df_test)

    logger.info(
        "Calculating Advanced Time-Series Lags & Cumulative Stats (Safe against Leakage)..."
    )
    df_train_feat, df_test_feat = build_time_series_features(df_train, df_test)

    logger.info("Handling categorical NaNs and converting to string...")
    for col in categorical_features:
        df_train_feat[col] = df_train_feat[col].fillna("Unknown").astype(str)
        df_test_feat[col] = df_test_feat[col].fillna("Unknown").astype(str)

    logger.info("Saving engineered features...")
    df_train_feat.to_csv(output_train, index=False)
    df_test_feat.to_csv(output_test, index=False)

    logger.success(f"Features ready! Train: {df_train_feat.shape} | Test: {df_test_feat.shape}")


if __name__ == "__main__":
    app()
