from pathlib import Path
from typing import Any, Optional

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from tqdm import tqdm
import typer

from restaurant_visitor_eda.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_target_distribution(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes.flatten()

    sns.set_style("whitegrid")

    visitors_log = np.log1p(data["visitors"])

    sns.histplot(data["visitors"], kde=True, ax=ax[0], color="skyblue")
    ax[0].set_title("Visitors Distribution", fontsize=14)
    ax[0].set_xlabel("Visitors")

    data["visitors_log"] = np.log1p(data["visitors"])
    sns.histplot(data["visitors_log"], kde=True, ax=ax[1], color="salmon")
    ax[1].set_title("Log-transformed Visitors (log1p)", fontsize=14)
    ax[1].set_xlabel("Log(Visitors + 1)")

    sns.boxplot(x=data["visitors"], ax=ax[2], color="skyblue")
    ax[2].set_title("Boxplot: Visitors", fontsize=12)

    sns.boxplot(x=visitors_log, ax=ax[3], color="salmon")
    ax[3].set_title("Boxplot: Log(Visitors)", fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


def build_visitors_boxplot(
    df: pd.DataFrame, y_col: str, hue_col: Optional[str] = None, title: str = ""
):
    order = (
        df.groupby(y_col, observed=True)["visitors"].median().sort_values(ascending=False).index
    )

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x="visitors",
        y=y_col,
        order=order,
        hue=hue_col if hue_col else y_col,
        palette="viridis",
        legend=False if not hue_col else True,
    )
    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(title or f"Visitors Distribution by {y_col} (Log Scale)")
    plt.xlabel("Visitors (log scale)")
    plt.tight_layout()
    plt.show()


def plot_visitors_boxplot_air_by_holiday(df: pd.DataFrame):
    sorted_idx = df.groupby("holiday_flg")["visitors"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df,
        x="visitors",
        y=df["holiday_flg"].astype(str),
        order=sorted_idx.astype(str),
        palette="viridis",
        hue=df["holiday_flg"].astype(str),
        legend=True,
    )

    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title("Distribution of Visitors by holiday flag (Log Scale)")
    plt.xlabel("Visitors (log scale)")
    plt.ylabel("holiday")
    plt.tight_layout()
    plt.show()


def plot_visitors_over_year(df: pd.DataFrame):
    median_ = df.groupby("visit_date")["visitors"].median()
    mean_ = df.groupby("visit_date")["visitors"].mean()
    sum_ = df.groupby("visit_date")["visitors"].sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_style("whitegrid")

    sns.lineplot(ax=axes[0], data=median_)
    axes[0].set_title("Median Visitors")
    axes[0].tick_params(axis="x", rotation=45)

    sns.lineplot(ax=axes[1], data=mean_)
    axes[1].set_title("Mean Visitors")
    axes[1].tick_params(axis="x", rotation=45)

    sns.lineplot(ax=axes[2], data=sum_)
    axes[2].set_title("Total Visitors (Sum)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_visitors_with_rolling(df: pd.DataFrame, window=7):
    median_ = df.groupby("visit_date")["visitors"].median()
    mean_ = df.groupby("visit_date")["visitors"].mean()
    sum_ = df.groupby("visit_date")["visitors"].sum()

    stats = [("Median", median_), ("Mean", mean_), ("Sum", sum_)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_style("whitegrid")

    for i, (title, data) in enumerate(stats):
        sns.lineplot(ax=axes[i], data=data, alpha=0.3, label="Daily")

        rolling_data = data.rolling(window=window, center=True).mean()
        sns.lineplot(
            ax=axes[i], data=rolling_data, color="red", linewidth=2, label=f"{window}-Day Rolling"
        )

        axes[i].set_title(f"{title} Visitors")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_number_of_open_restaurants(df: pd.DataFrame):
    data = df.groupby("visit_date")["air_store_id"].nunique()
    sns.lineplot(data=data)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title("Number of opened restaurants")
    plt.xlabel("Date")
    plt.ylabel("Number")
    plt.tick_params(axis="x", rotation=45)
    plt.show()


def plot_hpg_coverage(df: pd.DataFrame):
    plt.figure(figsize=(5, 5))

    status_counts = df["hpg_store_id"].isna().value_counts()
    status_counts.index = (
        ["Missing HPG ID", "Has HPG ID"]
        if status_counts.index[0]
        else ["Has HPG ID", "Missing HPG ID"]
    )

    colors = sns.color_palette("pastel")[0:2]

    plt.pie(
        status_counts,
        labels=status_counts.index.to_list(),
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        explode=(0.05, 0),
    )

    plt.title("HPG Store ID Mapping Coverage", fontsize=15, pad=20)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def build_categorical_count_plot(data: pd.DataFrame, column: str, title: str, top_n: int = 0):

    counts = data[column].value_counts()
    if top_n > 0:
        counts = counts.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=True
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel(column.replace("_", " ").title(), fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.show()


def build_target_by_category_plot(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "visitors",
    agg_func: str = "median",
    title: str = "",
    top_n: int = 25,
):
    stats = (
        df.groupby(group_col)[target_col].agg(agg_func).sort_values(ascending=False).reset_index()
    )

    if top_n > 0:
        stats = stats.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=stats, x=target_col, y=group_col, hue=group_col, palette="viridis", legend=True
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    plt.title(title or f"{agg_func.title()} {target_col} by {group_col}", fontsize=16, pad=20)
    plt.xlabel(f"{agg_func.title()} {target_col}", fontsize=12)
    plt.ylabel(group_col.replace("_", " ").title(), fontsize=12)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    df: pd.DataFrame, index: str, columns: str, values: str = "visitors", agg_func: Any = "count"
):
    """Universal heatmap for cross-categorical analysis."""
    if agg_func == "count":
        pivot = pd.crosstab(df[index], df[columns])
        label = "Count"
    else:
        pivot = df.pivot_table(
            values=values, index=index, columns=columns, aggfunc=agg_func
        ).fillna(0)
        label = f"{agg_func.capitalize()} {values}"

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f" if agg_func != "count" else "d",
        cmap="YlOrRd",
        cbar_kws={"label": label},
    )
    plt.title(f"{label} by {index} and {columns}")
    plt.tight_layout()
    plt.show()


def train_test_overlap(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    test_ids = df_test["air_store_id"].unique()
    train_ids = df_train["air_store_id"].unique()

    overlap_data = pd.DataFrame(
        {
            "Category": ["Total Train IDs", "Total Test IDs", "Common IDs", "Only in Test"],
            "Count": [
                len(train_ids),
                len(test_ids),
                len(set(train_ids) & set(test_ids)),
                len(set(test_ids) - set(train_ids)),
            ],
        }
    )

    fig_overlap = px.bar(
        overlap_data,
        x="Category",
        y="Count",
        text="Count",
        title="Train vs Test: Restaurant IDs Overlap",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_overlap.update_traces(textposition="outside")
    fig_overlap.show()


def time_gap(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    train_min, train_max = df_train["visit_date"].min(), df_train["visit_date"].max()
    test_min, test_max = df_test["visit_date"].min(), df_test["visit_date"].max()

    fig_gap = go.Figure()

    fig_gap.add_trace(
        go.Scatter(
            x=[train_min, train_max],
            y=["Timeline", "Timeline"],
            mode="lines+markers+text",
            name="Train Period",
            line=dict(color="royalblue", width=20),
            text=[f"Start: {train_min.date()}", f"End: {train_max.date()}"],
            textposition="top center",
        )
    )

    fig_gap.add_trace(
        go.Scatter(
            x=[test_min, test_max],
            y=["Timeline", "Timeline"],
            mode="lines+markers+text",
            name="Test Period",
            line=dict(color="orange", width=20),
            text=[f"Start: {test_min.date()}", f"End: {test_max.date()}"],
            textposition="bottom center",
        )
    )

    fig_gap.update_layout(
        title="Timeline of Train and Test Data",
        yaxis={"showticklabels": False},
        height=300,
        showlegend=True,
    )
    fig_gap.show()

    print(f"Gap duration: {(test_min - train_max).days} day(s)")


def data_recency(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    train_max = df_train["visit_date"].max()

    test_ids = df_test["air_store_id"].unique()

    last_dates = (
        df_train[df_train["air_store_id"].isin(test_ids)]
        .groupby("air_store_id")["visit_date"]
        .max()
        .reset_index()
    )

    last_dates["days_missing_before_test"] = (train_max - last_dates["visit_date"]).dt.days

    fig_recency = px.histogram(
        last_dates,
        x="days_missing_before_test",
        title="Data Recency: How many days of data are missing right before the gap?",
        labels={"days_missing_before_test": "Days of silence before April 22"},
        color_discrete_sequence=["indianred"],
    )
    fig_recency.show()


def plot_reservation_lead_time(
    df_air_reserve: pd.DataFrame, df_hpg_reserve: pd.DataFrame, max_days: int = 60
) -> None:
    air = df_air_reserve[["visit_datetime", "reserve_datetime"]].copy()
    air["System"] = "Air"

    hpg = df_hpg_reserve[["visit_datetime", "reserve_datetime"]].copy()
    hpg["System"] = "HPG"

    df_res = pd.concat([air, hpg], ignore_index=True)

    df_res["visit_datetime"] = pd.to_datetime(df_res["visit_datetime"])
    df_res["reserve_datetime"] = pd.to_datetime(df_res["reserve_datetime"])

    df_res["lead_time_days"] = (df_res["visit_datetime"] - df_res["reserve_datetime"]).dt.days
    df_plot = df_res[df_res["lead_time_days"] <= max_days]

    fig = px.histogram(
        df_plot,
        x="lead_time_days",
        color="System",
        nbins=max_days + 1,
        barmode="overlay",
        title="Reservation Lead Time: Days Between Booking and Visit",
        labels={"lead_time_days": "Days to Visit", "System": "Booking System"},
        color_discrete_map={"AirREGI": "#1f77b4", "HPG": "#ff7f0e"},
    )

    fig.update_layout(
        xaxis_title="Days from Reservation to Visit (0 = Same Day)",
        yaxis_title="Number of Reservations",
        bargap=0.1,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_traces(opacity=0.75)
    fig.show()


def plot_golden_week_traffic(df_final: pd.DataFrame, year: int = 2016) -> None:

    daily_avg_all = df_final.groupby("visit_date")["visitors"].mean()
    overall_baseline = daily_avg_all.median()

    start_date = pd.to_datetime(f"{year}-04-20")
    end_date = pd.to_datetime(f"{year}-05-15")
    mask = (df_final["visit_date"] >= start_date) & (df_final["visit_date"] <= end_date)
    df_gw = df_final[mask].copy()

    daily_visitors = df_gw.groupby(["visit_date", "holiday_flg"])["visitors"].mean().reset_index()
    daily_visitors["Is Holiday"] = daily_visitors["holiday_flg"].map({1: "Yes", 0: "No"})

    daily_visitors["visitors_rounded"] = daily_visitors["visitors"].round(1)

    fig = px.bar(
        daily_visitors,
        x="visit_date",
        y="visitors",
        color="Is Holiday",
        color_discrete_map={"Yes": "#d62728", "No": "#1f77b4"},
        title=f"Golden Week Anomaly ({year}): Average Visitors per Restaurant",
        text="visitors_rounded",
    )

    fig.add_hline(
        y=overall_baseline,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"Yearly Baseline (Avg/Rest): {overall_baseline:.1f}",
        annotation_position="top left",
        annotation_font_size=12,
        annotation_font_color="#2ca02c",
    )

    gw_start = f"{year}-04-29"
    gw_end = f"{year}-05-05"

    fig.add_vline(
        x=pd.to_datetime(gw_start).timestamp() * 1000,
        line_width=2,
        line_dash="dot",
        line_color="orange",
        annotation_text="Start (Apr 29)",
        annotation_position="top left",
    )

    fig.add_vline(
        x=pd.to_datetime(gw_end).timestamp() * 1000,
        line_width=2,
        line_dash="dot",
        line_color="orange",
        annotation_text="End (May 5)",
        annotation_position="top right",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Visit Date",
        yaxis_title="Average Visitors per Restaurant",
        template="plotly_white",
        xaxis_tickangle=-45,
    )

    fig.update_yaxes(range=[0, daily_visitors["visitors"].max() * 1.15])

    fig.show()


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
