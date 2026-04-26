from pathlib import Path

from loguru import logger
import pandas as pd
from scipy import stats
from tqdm import tqdm
import typer

from restaurant_visitor_eda.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def count_unique_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"nan percentage": df.isna().mean() * 100, "unique": df.nunique()}
    ).sort_values(by="nan percentage", ascending=False)


def get_stats(df: pd.DataFrame):
    stats = df.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
    stats.loc["median"] = df.median()
    stats.loc["skewness"] = df.skew()
    stats.loc["kurtosis"] = df.kurtosis()
    return stats.round(3)


def calculate_holiday_significance(df):
    holiday_visitors = df[df["holiday_flg"] == 1]["visitors"]
    workday_visitors = df[df["holiday_flg"] == 0]["visitors"]

    stat, p_value = stats.mannwhitneyu(holiday_visitors, workday_visitors, alternative="two-sided")

    print(f"Statistic: {stat:.2f}")
    print(f"P-value: {p_value:.10f}")

    alpha = 0.05
    if p_value < alpha:
        print("Result: The difference is significant.")
    else:
        print("Результат: The difference is NOT significant.")


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
