from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from restaurant_visitor_eda.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_target_distribution(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes.flatten()

    sns.set_style("whitegrid")

    visitors_log = np.log1p(data['visitors'])

    sns.histplot(data['visitors'], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title('Visitors Distribution', fontsize=14)
    ax[0].set_xlabel('Visitors')

    data['visitors_log'] = np.log1p(data['visitors'])
    sns.histplot(data['visitors_log'], kde=True, ax=ax[1], color='salmon')
    ax[1].set_title('Log-transformed Visitors (log1p)', fontsize=14)
    ax[1].set_xlabel('Log(Visitors + 1)')

    sns.boxplot(x=data['visitors'], ax=ax[2], color='skyblue')
    ax[2].set_title('Boxplot: Visitors', fontsize=12)

    sns.boxplot(x=visitors_log, ax=ax[3], color='salmon')
    ax[3].set_title('Boxplot: Log(Visitors)', fontsize=12)

    
    sns.despine()
    plt.tight_layout()
    plt.show()


def build_barplot_for_air_genres(data):
    df_air_genre_name = data.air_genre_name.value_counts()
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_air_genre_name, orient='h', palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('air restaurants')
    sns.despine()
    plt.show()

def build_barplot_for_hpg_genres(data):
    df_hpg_genre_name = data.hpg_genre_name.value_counts()
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_hpg_genre_name, orient='h', palette = 'viridis')
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('hpg restaurants')
    sns.despine()
    plt.show()

def plot_median_visitors_per_genre(df: pd.DataFrame):
    genre_stats = (
        df.groupby('air_genre_name')['visitors']
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=genre_stats, 
        x='visitors', 
        y='air_genre_name', 
        palette='viridis'
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('Median Visitors by Restaurant Genre', fontsize=16, pad=20)
    plt.xlabel('Median Daily Visitors', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    
    sns.despine(left=True, bottom=True) 
    plt.tight_layout()
    plt.show()




def plot_visitors_boxplot_air(df: pd.DataFrame):
    sorted_idx = (
        df.groupby('air_genre_name')['visitors']
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(data=df, x='visitors', y='air_genre_name', order=sorted_idx, palette='viridis', hue='air_genre_name', legend=False)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2) # Добавим сетку для логарифмов
    plt.title('Distribution of Visitors by Genre (Log Scale)')
    plt.xlabel('Visitors (log scale)')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
