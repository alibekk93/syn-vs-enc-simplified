# src/visualization.py

import json
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FIGURES_DIR = Path("results/figures")

# ===========================================================
# 1. LOADING METRICS
# ===========================================================

def load_metrics(base_dir="results/metrics"):
    records = []

    for path in Path(base_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)

        row = {
            "mode": data.get("mode"),
            "model": data.get("model"),
            "dataset": data.get("dataset"),
            "split": data.get("split"),
            "n_bits": data.get("n_bits"),
            **data.get("metrics", {})
        }

        records.append(row)

    return pd.DataFrame(records)


# ===========================================================
# 2. LOADING RESOURCE PROFILES
# ===========================================================

def parse_filename_metadata(filename):
    """
    Extract mode / model / dataset / n_bits from filename.
    Example:
        fhe_4__logistic_regression__heart_disease.json
        standard__rf__diabetes.json
    """

    name = Path(filename).stem

    parts = name.split("__")

    meta = {
        "raw_name": name,
        "mode": None,
        "model": None,
        "dataset": None,
        "n_bits": None
    }

    # FHE pattern
    fhe_match = re.match(r"fhe_(\d+)", parts[0])
    if fhe_match:
        meta["mode"] = "fhe"
        meta["n_bits"] = int(fhe_match.group(1))
        parts[0] = "fhe"

    # Standard / synthetic
    else:
        meta["mode"] = parts[0]

    # Assign remaining
    if len(parts) >= 3:
        meta["model"] = parts[1]
        meta["dataset"] = parts[2]

    return meta


def load_resource_profiles(base_dir="results/resource_profiles"):
    records = []

    for path in Path(base_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)

        meta = parse_filename_metadata(path.name)

        row = {
            **meta,
            "train_time": sum(data.get("training_time", {}).values()),
            "inf_time_total": data.get("inference_time", {}).get("total"),
            "inf_time_per_sample": data.get("inference_time", {}).get("per_sample"),
            "mem_train_avg": data.get("memory", {}).get("training", {}).get("average_mb"),
            "mem_train_peak": data.get("memory", {}).get("training", {}).get("peak_mb"),
            "mem_inf_avg": data.get("memory", {}).get("inference", {}).get("average_mb"),
            "mem_inf_peak": data.get("memory", {}).get("inference", {}).get("peak_mb"),
            "model_size_mb": data.get("storage", {}).get("model_size_mb"),
            "data_size_mb": data.get("storage", {}).get("data_size_mb"),
            "circuit_complexity": data.get("fhe", {}).get("circuit_complexity"),
        }

        records.append(row)

    return pd.DataFrame(records)


# ===========================================================
# TRANSFORM / CLEAN
# ===========================================================

def preprocess_metrics(df):
    df = df.copy()

    df["mode_pretty"] = df["mode"].replace({
        "standard": "Real",
        "gaussian_copula": "Synthetic",
        "fhe": "FHE"
    })

    return df


def merge_metrics_profiles(metrics_df, profiles_df):
    return pd.merge(
        metrics_df,
        profiles_df,
        on=["mode", "model", "dataset", "n_bits"],
        how="left"
    )


# ===========================================================
# CONFIG UTIL
# ===========================================================

from src.utils import load_config

def get_metrics_from_config(cfg_path="config/models.yaml"):
    cfg = load_config(cfg_path)
    return cfg.get("metrics", [])


# ===========================================================
# PLOTS — PERFORMANCE
# ===========================================================

def format_metric_name(metric):
    return metric.replace("_", " ").title()


def plot_performance(df, metric, save_dir="results/figures"):
    plt.figure()

    sns.barplot(
        data=df[df["split"] == "test"],
        x="model",
        y=metric,
        hue="mode_pretty",
        errorbar=None   # to be updated when bootstrap is available
    )

    title = f"{format_metric_name(metric)} Comparison"
    plt.title(title)

    plt.xticks(rotation=20)
    plt.tight_layout()

    save_path = Path(save_dir) / f"{metric}_comparison.svg"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, format="svg")
    plt.close()


def plot_performance_grid(df, metric, save_dir=FIGURES_DIR):

    for dataset in df["dataset"].unique():

        subset = df[(df["dataset"] == dataset) & (df["split"] == "test")]

        if subset.empty or metric not in subset.columns:
            continue

        plt.figure(figsize=(8, 5))

        ax = sns.barplot(
            data=subset,
            x="model",
            y=metric,
            hue="mode_pretty",
            errorbar=None
        )

        plt.title(f"{format_metric_name(metric)} — {dataset}")
        plt.xticks(rotation=20)

        # shared legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Mode")

        plt.tight_layout()

        filename = f"{metric}_comparison__{dataset}.svg"
        save_path = Path(save_dir) / filename

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="svg")
        plt.close()


# ===========================================================
# PARETO TRADEOFF PLOTS
# ===========================================================

RESOURCE_MAP = {
    "training_time": ("train_time", "Training Time (s)"),
    "inference_time": ("inf_time_per_sample", "Inference Time per Sample (s)"),
    "memory": ("mem_inf_peak", "Peak Inference Memory (MB)"),
}

def compute_pareto_front(df, x_col, y_col):
    """
    Returns subset of df that lies on the Pareto frontier.
    
    We assume:
        - Minimize x (resource)
        - Maximize y (metric)
    """
    # Sort by resource (ascending), then metric (descending)
    df_sorted = df.sort_values(by=[x_col, y_col], ascending=[True, False])
    pareto_points = []
    best_so_far = -float("inf")
    for _, row in df_sorted.iterrows():
        if row[y_col] > best_so_far:
            pareto_points.append(row)
            best_so_far = row[y_col]
    pareto_df = pd.DataFrame(pareto_points)
    return pareto_df

def plot_tradeoff(df, metric, resource_key, save_dir=FIGURES_DIR):

    col, label = RESOURCE_MAP[resource_key]

    for dataset in df["dataset"].unique():

        subset = df[df["dataset"] == dataset]
        subset = subset.dropna(subset=[col, metric])

        if subset.empty:
            continue

        plt.figure(figsize=(7, 5))

        # -------------------------------------------------------
        # BASE PLOT
        # -------------------------------------------------------
        sns.scatterplot(
            data=subset,
            x=col,
            y=metric,
            hue="mode_pretty",
            style="model",
            s=100,
            alpha=0.8
        )

        plt.xscale("log")

        # -------------------------------------------------------
        # PARETO FRONTIER
        # -------------------------------------------------------
        pareto_df = compute_pareto_front(subset, col, metric)

        if not pareto_df.empty:

            # sort for line plotting
            pareto_df = pareto_df.sort_values(by=col)

            # highlight Pareto points
            plt.scatter(
                pareto_df[col],
                pareto_df[metric],
                color="black",
                s=140,
                facecolors="none",
                linewidths=1.8,
                label="Pareto Optimal"
            )

            # connect points
            plt.plot(
                pareto_df[col],
                pareto_df[metric],
                color="black",
                linestyle="--",
                linewidth=1.2
            )

        # -------------------------------------------------------
        # FINAL STYLING
        # -------------------------------------------------------
        plt.title(f"{format_metric_name(metric)} vs {label} — {dataset}")

        plt.xlabel(label)
        plt.ylabel(format_metric_name(metric))

        plt.tight_layout()

        filename = f"{metric}_vs_{resource_key}__{dataset}.svg"
        save_path = Path(save_dir) / filename

        plt.savefig(save_path, format="svg")
        plt.close()


# ===========================================================
# FHE ABLATION
# ===========================================================

def plot_fhe_ablation(df, metric, save_dir=FIGURES_DIR):
    fhe_df = df[df["mode"] == "fhe"]
    if fhe_df.empty or "n_bits" not in fhe_df.columns:
        return
    
    for dataset in fhe_df["dataset"].unique():
        subset = fhe_df[fhe_df["dataset"] == dataset]
        if subset.empty:
            continue

        plt.figure()

        sns.lineplot(
            data=subset,
            x="n_bits",
            y=metric,
            hue="model",
            marker="o"
        )

        plt.title(f"FHE (n_bits) vs {format_metric_name(metric)} — {dataset}")
        plt.tight_layout()

        filename = f"fhe_ablation_{metric}__{dataset}.svg"
        save_path = Path(save_dir) / filename

        plt.savefig(save_path, format="svg")
        plt.close()


# ===========================================================
# MAIN ENTRYPOINT
# ===========================================================

def generate_all_figures():

    metrics_df = load_metrics()
    profiles_df = load_resource_profiles()

    metrics_df = preprocess_metrics(metrics_df)
    full_df = merge_metrics_profiles(metrics_df, profiles_df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics dynamically from config
    metrics = get_metrics_from_config()

    # -------------------------------------------------------
    # BAR PLOTS (GRID PER DATASET)
    # -------------------------------------------------------
    for metric in metrics:
        if metric in full_df.columns:
            plot_performance_grid(full_df, metric)

    # -------------------------------------------------------
    # TRADEOFF PLOTS (ALL COMBINATIONS)
    # -------------------------------------------------------
    for metric in metrics:
        if metric not in full_df.columns:
            continue

        for resource in RESOURCE_MAP.keys():
            plot_tradeoff(full_df, metric, resource)

    # -------------------------------------------------------
    # FHE ABLATION
    # -------------------------------------------------------
    for metric in metrics:
        if metric in full_df.columns:
            plot_fhe_ablation(full_df, metric)

    # -------------------------------------------------------
    # SAVE MERGED TABLE
    # -------------------------------------------------------
    full_df.to_csv("results/summary_visualization.csv", index=False)