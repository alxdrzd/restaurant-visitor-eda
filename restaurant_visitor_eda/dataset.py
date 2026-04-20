from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
from pathlib import Path

def count_unique_and_nans(df: pd.DataFrame) -> pd.DataFrame:
    """counts the nans in each column of the DataFrame"""
    missing = pd.DataFrame(df.isna().mean() * 100)
    missing.columns = ['nan percentage']
    missing['unique'] = df.nunique()
    missing = missing.sort_values(ascending=False, by='nan percentage')
    return missing.round(2)


from restaurant_visitor_eda.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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
