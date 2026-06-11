# src/visualization.py

import json
import re
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _load_viz_config(path="config/visualization.yaml"):
    return load_config(path)


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
# BOOTSTRAP DATA LOADING
# ===========================================================

def load_bootstrap(path="results/bootstrap/aggregated.json"):
    """Flatten aggregated bootstrap JSON into a merged DataFrame (test split only for metrics)."""
    with open(path) as f:
        data = json.load(f)

    key_cols = ["mode", "n_bits", "model", "dataset", "seed"]

    metric_records = []
    for raw_mode, models in data.get("metrics", {}).items():
        meta = parse_filename_metadata(raw_mode)
        for model_name, datasets_map in models.items():
            for dataset_name, entries in datasets_map.items():
                for entry in entries:
                    if entry.get("split") != "test":
                        continue
                    metric_records.append({
                        "mode": meta["mode"],
                        "n_bits": meta["n_bits"],
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": entry["seed"],
                        **entry.get("metrics", {})
                    })

    resource_records = []
    for raw_mode, models in data.get("resource_profiles", {}).items():
        meta = parse_filename_metadata(raw_mode)
        for model_name, datasets_map in models.items():
            for dataset_name, entries in datasets_map.items():
                for entry in entries:
                    resource_records.append({
                        "mode": meta["mode"],
                        "n_bits": meta["n_bits"],
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": entry["seed"],
                        "train_time": sum(entry.get("training_time", {}).values()),
                        "inf_time_total": entry.get("inference_time", {}).get("total"),
                        "inf_time_per_sample": entry.get("inference_time", {}).get("per_sample"),
                        "mem_train_avg": entry.get("memory", {}).get("training", {}).get("average_mb"),
                        "mem_train_peak": entry.get("memory", {}).get("training", {}).get("peak_mb"),
                        "mem_inf_avg": entry.get("memory", {}).get("inference", {}).get("average_mb"),
                        "mem_inf_peak": entry.get("memory", {}).get("inference", {}).get("peak_mb"),
                        "model_size_mb": entry.get("storage", {}).get("model_size_mb"),
                        "data_size_mb": entry.get("storage", {}).get("data_size_mb"),
                        "circuit_complexity": entry.get("fhe", {}).get("circuit_complexity"),
                    })

    metrics_df = pd.DataFrame(metric_records)
    profiles_df = pd.DataFrame(resource_records)

    if metrics_df.empty:
        return profiles_df
    if profiles_df.empty:
        return metrics_df

    return pd.merge(metrics_df, profiles_df, on=key_cols, how="outer")


def _mode_label(mode, n_bits):
    if mode == "fhe" and pd.notna(n_bits):
        return f"FHE ({int(n_bits)}-bit)"
    return {"standard": "Real", "gaussian_copula": "Synthetic"}.get(mode, mode)


def _mode_label_sort_key(label):
    if label == "Real":
        return (0, 0)
    if label == "Synthetic":
        return (1, 0)
    m = re.search(r"(\d+)", label)
    return (2, int(m.group(1)) if m else 0)


# ===========================================================
# BOXPLOTS (BOOTSTRAP)
# ===========================================================

def plot_boxplot(dataset, model, metric, palette=None, save_dir=None,
                 bootstrap_path="results/bootstrap/aggregated.json",
                 viz_cfg_path="config/visualization.yaml"):
    """
    Boxplot of bootstrap distributions for one dataset / model / metric combination.

    x-axis  : modes (Real, Synthetic, FHE N-bit) — one box per mode, colored by mode
    y-axis  : metric value
    seeds   : aggregated into the box distribution
    """
    cfg = _load_viz_config(viz_cfg_path)
    box_cfg = cfg["boxplot"]
    font_cfg = cfg["fonts"]
    fig_cfg = cfg["figures"]

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    if palette is None:
        palette = cfg["colors"]["palette"]
    if save_dir is None:
        save_dir = Path(fig_cfg["dir"])

    df = load_bootstrap(bootstrap_path)
    df["mode_label"] = df.apply(lambda r: _mode_label(r["mode"], r["n_bits"]), axis=1)

    subset = (
        df[(df["dataset"] == dataset) & (df["model"] == model)]
        .dropna(subset=[metric])
    )
    if subset.empty:
        return None, None

    order = sorted(subset["mode_label"].unique(), key=_mode_label_sort_key)

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.2), 5))

    show_means = box_cfg["showmeans"]
    meanprops = {
        "marker": "D",
        "markerfacecolor": box_cfg["mean_marker_color"],
        "markeredgecolor": box_cfg["mean_marker_color"],
        "markersize": box_cfg["mean_marker_size"],
    } if show_means else {}

    sns.boxplot(
        data=subset,
        x="mode_label",
        y=metric,
        hue="mode_label",
        hue_order=order,
        order=order,
        palette=palette,
        linewidth=box_cfg["linewidth"],
        notch=box_cfg["notch"],
        showmeans=show_means,
        meanprops=meanprops,
        legend=False,
        ax=ax,
    )

    for patch in ax.patches:
        patch.set_alpha(box_cfg["alpha"])

    sns.stripplot(
        data=subset,
        x="mode_label",
        y=metric,
        hue="mode_label",
        hue_order=order,
        order=order,
        palette=palette,
        dodge=False,
        alpha=box_cfg["strip_alpha"],
        size=box_cfg["strip_size"],
        legend=False,
        ax=ax,
    )

    if box_cfg.get("despine", True):
        sns.despine(ax=ax)

    grid_alpha = box_cfg.get("grid_alpha", 0.0)
    if grid_alpha > 0:
        ax.grid(axis="y", alpha=grid_alpha, linewidth=0.5)

    model_pretty = model.replace("_", " ").title()
    ax.set_title(
        f"{format_metric_name(metric)} — {dataset} / {model_pretty}",
        fontsize=font_cfg["title_size"],
        fontweight=font_cfg["title_weight"],
    )
    ax.set_xlabel("Mode", fontsize=font_cfg["label_size"])
    ax.set_ylabel(format_metric_name(metric), fontsize=font_cfg["label_size"])
    ax.tick_params(labelsize=font_cfg["tick_size"])
    plt.xticks(rotation=15)

    plt.tight_layout()

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"boxplot_{metric}__{dataset}__{model}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt)
    plt.close()

    return fig, ax


# ===========================================================
# BOXPLOT GRID (BOOTSTRAP) — models × datasets, one fig per metric
# ===========================================================

def plot_boxplot_grid(metric, palette=None, save_dir=None,
                     bootstrap_path="results/bootstrap/aggregated.json",
                     viz_cfg_path="config/visualization.yaml"):
    """
    Multi-panel boxplot for one metric.
    Rows = models, columns = datasets. Y-axis range shared across all panels.
    X-axis ticks = modes (Real, Synthetic, FHE N-bit).
    """
    cfg = _load_viz_config(viz_cfg_path)
    box_cfg = cfg["boxplot"]
    font_cfg = cfg["fonts"]
    fig_cfg = cfg["figures"]

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    if palette is None:
        palette = cfg["colors"]["palette"]
    if save_dir is None:
        save_dir = Path(fig_cfg["dir"])

    df = load_bootstrap(bootstrap_path)
    df["mode_label"] = df.apply(lambda r: _mode_label(r["mode"], r["n_bits"]), axis=1)
    df = df.dropna(subset=[metric])

    if df.empty:
        return None

    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    global_order = sorted(df["mode_label"].unique(), key=_mode_label_sort_key)

    n_rows = len(models)
    n_cols = len(datasets)

    show_means = box_cfg["showmeans"]
    meanprops = {
        "marker": "D",
        "markerfacecolor": box_cfg["mean_marker_color"],
        "markeredgecolor": box_cfg["mean_marker_color"],
        "markersize": box_cfg["mean_marker_size"],
    } if show_means else {}

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.5),
        squeeze=False,
    )

    for r_idx, model in enumerate(models):
        for c_idx, dataset in enumerate(datasets):
            ax = axes[r_idx][c_idx]
            subset = df[(df["model"] == model) & (df["dataset"] == dataset)]

            if subset.empty:
                ax.set_visible(False)
                continue

            local_order = [m for m in global_order if m in subset["mode_label"].values]

            sns.boxplot(
                data=subset,
                x="mode_label",
                y=metric,
                hue="mode_label",
                hue_order=local_order,
                order=local_order,
                palette=palette,
                linewidth=box_cfg["linewidth"],
                notch=box_cfg["notch"],
                showmeans=show_means,
                meanprops=meanprops,
                legend=False,
                ax=ax,
            )

            for patch in ax.patches:
                patch.set_alpha(box_cfg["alpha"])

            sns.stripplot(
                data=subset,
                x="mode_label",
                y=metric,
                hue="mode_label",
                hue_order=local_order,
                order=local_order,
                palette=palette,
                dodge=False,
                alpha=box_cfg["strip_alpha"],
                size=box_cfg["strip_size"],
                legend=False,
                ax=ax,
            )

            if box_cfg.get("despine", True):
                sns.despine(ax=ax)

            grid_alpha = box_cfg.get("grid_alpha", 0.0)
            if grid_alpha > 0:
                ax.grid(axis="y", alpha=grid_alpha, linewidth=0.5)

            if r_idx == 0:
                ax.set_title(
                    dataset.replace("_", " ").title(),
                    fontsize=font_cfg["label_size"],
                )
            else:
                ax.set_title("")

            if c_idx == 0:
                ax.set_ylabel(
                    model.replace("_", " ").title(),
                    fontsize=font_cfg["label_size"],
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            ax.tick_params(axis="y", labelsize=font_cfg["tick_size"])
            ax.tick_params(
                axis="x",
                labelsize=font_cfg["tick_size"] - 1,
                labelrotation=40,
            )
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

    # shared y-axis range across all visible panels
    y_values = df[metric].dropna()
    y_min, y_max = y_values.min(), y_values.max()
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
    for ax_row in axes:
        for ax in ax_row:
            if ax.get_visible():
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle(
        format_metric_name(metric),
        fontsize=font_cfg["title_size"],
        fontweight=font_cfg["title_weight"],
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"boxplot_grid_{metric}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt)
    plt.close()

    return fig


# ===========================================================
# MAIN ENTRYPOINT
# ===========================================================

def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_viz_config()
    metrics = cfg.get("metrics", [])

    df = load_bootstrap()
    available_cols = set(df.columns)

    for metric in metrics:
        if metric in available_cols:
            plot_boxplot_grid(metric)