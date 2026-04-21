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


def build_barplot_for_air_prefectures(data):
    df_air_genre_prefecture = data.prefecture.value_counts()
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_air_genre_prefecture, orient='h', palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('prefectures')
    sns.despine()
    plt.show()


def build_barplot_for_air_districts(data):
    df_air_genre_district = data.district.value_counts()
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_air_genre_district, orient='h', palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('districts')
    sns.despine()
    plt.show()


def build_barplot_for_air_blocks(data):
    df_air_genre_block = data.block.value_counts()
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_air_genre_block, orient='h', legend=False, palette='viridis')
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('blocks')
    sns.despine()
    plt.show()



def plot_median_visitors_per_prefecture(df: pd.DataFrame):
    genre_stats = (
        df.groupby('prefecture')['visitors']
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=genre_stats, 
        x='visitors', 
        y='prefecture', 
        palette='viridis'
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('Median Visitors by prefecture', fontsize=16, pad=20)
    plt.xlabel('Median Daily Visitors', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    
    sns.despine(left=True, bottom=True) 
    plt.tight_layout()
    plt.show()





def plot_median_visitors_per_district(df: pd.DataFrame):
    genre_stats = (
        df.groupby('district')['visitors']
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=genre_stats.iloc[0:25], 
        x='visitors', 
        y='district', 
        palette='viridis'
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('Median Visitors by district', fontsize=16, pad=20)
    plt.xlabel('Median Daily Visitors', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    
    sns.despine(left=True, bottom=True) 
    plt.tight_layout()
    plt.show()



def plot_median_visitors_per_block(df: pd.DataFrame):
    genre_stats = (
        df.groupby('block')['visitors']
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=genre_stats.iloc[0:25], 
        x='visitors', 
        y='block', 
        palette='viridis'
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)
    plt.title('Median Visitors by block', fontsize=16, pad=20)
    plt.xlabel('Median Daily Visitors', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    
    sns.despine(left=True, bottom=True) 
    plt.tight_layout()
    plt.show()


def plot_genre_prefecture_heatmap(df):
    pivot = pd.crosstab(df['air_genre_name'], df['prefecture'])
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Number of restaurants'})
    plt.title('Concentration of Genres by Prefecture', fontsize=16)
    plt.xlabel('Prefecture')
    plt.ylabel('Genre')
    plt.show()

def plot_visitors_heatmap(df, agg_func='median'):
    pivot = df.pivot_table(
        values='visitors', 
        index='air_genre_name', 
        columns='prefecture', 
        aggfunc=agg_func
    )
    
    pivot = pivot.fillna(0)

    plt.figure(figsize=(14, 9))
    
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", 
                cbar_kws={'label': f'{agg_func.capitalize()} Visitors'})
    
    plt.title(f'{agg_func.capitalize()} Daily Visitors by Genre and Prefecture', fontsize=16)
    plt.xlabel('Prefecture')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()


def plot_visitors_boxplot_air_by_day(df: pd.DataFrame):
    sorted_idx = (
        df.groupby('day_of_week')['visitors']
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(data=df, x='visitors', y='day_of_week', order=sorted_idx, palette='viridis', hue='day_of_week', legend=False)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2) 
    plt.title('Distribution of Visitors by days of week (Log Scale)')
    plt.xlabel('Visitors (log scale)')
    plt.ylabel('Day of week')
    plt.tight_layout()
    plt.show()


def plot_visitors_boxplot_air_by_holiday(df: pd.DataFrame):
    sorted_idx = (
        df.groupby('holiday_flg')['visitors']
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
    data=df, 
    x='visitors', 
    y=df['holiday_flg'].astype(str), 
    order=sorted_idx.astype(str),    
    palette='viridis', 
    hue=df['holiday_flg'].astype(str), 
    legend=False
    )

    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2) 
    plt.title('Distribution of Visitors by holiday flag (Log Scale)')
    plt.xlabel('Visitors (log scale)')
    plt.ylabel('holiday')
    plt.tight_layout()
    plt.show()


def plot_visitors_boxplot_air_by_holiday_and_day(df: pd.DataFrame):
    sorted_idx = (
        df.groupby('day_of_week')['visitors']
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
    data=df, 
    x='visitors', 
    y=df['day_of_week'].astype(str), 
    order=sorted_idx.astype(str),    
    palette='viridis', 
    hue=df['holiday_flg'].astype(str), 
    legend=False
    )

    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2) 
    plt.title('Distribution of Visitors by holiday flag (Log Scale)')
    plt.xlabel('Visitors (log scale)')
    plt.ylabel('holiday')
    plt.tight_layout()
    plt.show()

def plot_visitors_boxplot_air_by_holiday_and_day_and_eve(df: pd.DataFrame):
    sorted_idx = (
        df.groupby('day_pattern')['visitors']
        .median()
        .sort_values(ascending=False)
        .index
    )

    sns.boxplot(
    data=df, 
    x='visitors', 
    y='day_pattern', 
    order=sorted_idx,
    palette='viridis'
)

    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2) 
    plt.title('Distribution of Visitors by holiday flag (Log Scale)')
    plt.xlabel('Visitors (log scale)')
    plt.ylabel('holiday')
    plt.tight_layout()
    plt.show()

def plot_visitors_over_year(df: pd.DataFrame):
    median_ = df.groupby('visit_datetime')['visitors'].median()
    mean_ = df.groupby('visit_datetime')['visitors'].mean()
    sum_ = df.groupby('visit_datetime')['visitors'].sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_style("whitegrid")

    sns.lineplot(ax=axes[0], data=median_)
    axes[0].set_title('Median Visitors')
    axes[0].tick_params(axis='x', rotation=45)

    sns.lineplot(ax=axes[1], data=mean_)
    axes[1].set_title('Mean Visitors')
    axes[1].tick_params(axis='x', rotation=45)

    sns.lineplot(ax=axes[2], data=sum_)
    axes[2].set_title('Total Visitors (Sum)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def plot_visitors_with_rolling(df: pd.DataFrame, window=7):
    median_ = df.groupby('visit_datetime')['visitors'].median()
    mean_ = df.groupby('visit_datetime')['visitors'].mean()
    sum_ = df.groupby('visit_datetime')['visitors'].sum()

    stats = [
        ("Median", median_),
        ("Mean", mean_),
        ("Sum", sum_)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_style("whitegrid")

    for i, (title, data) in enumerate(stats):
        sns.lineplot(ax=axes[i], data=data, alpha=0.3, label='Daily')
        
        rolling_data = data.rolling(window=window, center=True).mean()
        sns.lineplot(ax=axes[i], data=rolling_data, color='red', linewidth=2, label=f'{window}-Day Rolling')
        
        axes[i].set_title(f'{title} Visitors')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_number_of_open_restaurants(df: pd.DataFrame):
    data = df.groupby('visit_datetime')['air_store_id'].nunique()
    sns.lineplot(data=data)
    plt.grid(True, which="both", ls="-", alpha=0.2) 
    plt.title('Number of opened restaurants')
    plt.xlabel('Date')
    plt.ylabel('Number')
    plt.tick_params(axis='x', rotation=45)
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
