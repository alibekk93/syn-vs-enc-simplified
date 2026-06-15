# src/visualization.py

import colorsys
import json
import re
from functools import lru_cache
from pathlib import Path
import matplotlib.colors as mcolors
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
        return f"FHE {int(n_bits)}-bit"
    labels = {"standard": "Real", "gaussian_copula": "Gaussian Copula", "ctgan": "CTGAN"}
    return labels.get(mode, mode)


def _raw_mode_key(mode, n_bits):
    """Reconstruct the raw aggregated.json key from parsed (mode, n_bits) fields."""
    if mode == "fhe" and pd.notna(n_bits):
        return f"fhe_{int(n_bits)}"
    return mode


def _sort_raw_keys(raw_keys):
    """
    Order: standard → non-FHE synthetic (alphabetical) → fhe_N (ascending N) → other.
    Adding a new synthetic mode requires only a config entry — it slots in alphabetically.
    """
    _SYNTHETIC_KEYS = {"gaussian_copula", "ctgan"}

    def _key(k):
        if k == "standard":
            return (0, 0, k)
        if k in _SYNTHETIC_KEYS or (k not in {"standard"} and not k.startswith("fhe_")):
            # synthetic family or unknown non-FHE — alphabetical within slot 1
            return (1, 0, k)
        if k.startswith("fhe_"):
            try:
                return (2, int(k.split("_")[1]), k)
            except ValueError:
                return (2, 0, k)
        return (3, 0, k)

    return sorted(raw_keys, key=_key)


def _lightness_ramp(base_color, n, lightness_range):
    """
    Return n hex colors along an HLS lightness ramp for the given base_color.
    lightness_range = [l_max, l_min]  (first item = lightest, last = darkest)
    """
    l_max, l_min = lightness_range
    r, g, b = mcolors.to_rgb(base_color)
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        lightness = l_max - t * (l_max - l_min)
        r2, g2, b2 = colorsys.hls_to_rgb(h, max(0.1, min(0.9, lightness)), s)
        colors.append(mcolors.to_hex((r2, g2, b2)))
    return colors


def _build_mode_display(cfg, raw_keys):
    """
    Build label, color, and group mappings from visualization config.

    Returns
    -------
    label_map      : dict  raw_key -> display label
    color_map      : dict  display_label -> hex color  (for seaborn palette=)
    label_group_map: dict  display_label -> group name
    """
    modes_cfg = cfg.get("modes", {})
    groups_cfg = cfg.get("groups", {})

    # ── determine group membership for every raw key ────────────────────────
    def _group_of(key):
        if key.startswith("fhe_"):
            return "fhe"
        return modes_cfg.get(key, {}).get("group", "other")

    # ── gather keys per group (preserving sort order) ────────────────────────
    sorted_keys = _sort_raw_keys(raw_keys)
    groups_present = {}  # group -> [raw_keys in order]
    for k in sorted_keys:
        g = _group_of(k)
        groups_present.setdefault(g, []).append(k)

    # ── precompute colors for multi-member groups ─────────────────────────────
    group_colors = {}  # raw_key -> hex color

    for group, members in groups_present.items():
        gcfg = groups_cfg.get(group, {})
        base = gcfg.get("base_color", "#999999")
        l_range = gcfg.get("lightness_range")

        if len(members) == 1 or not l_range:
            for k in members:
                group_colors[k] = base
        else:
            ramp = _lightness_ramp(base, len(members), l_range)
            for k, color in zip(members, ramp):
                group_colors[k] = color

    # ── build output maps ─────────────────────────────────────────────────────
    label_map = {}
    color_map = {}
    label_group_map = {}

    fhe_cfg = modes_cfg.get("fhe", {})
    fhe_prefix = fhe_cfg.get("label_prefix", "FHE")

    for key in raw_keys:
        group = _group_of(key)

        if key.startswith("fhe_"):
            try:
                n = int(key.split("_")[1])
            except ValueError:
                n = 0
            label = f"{fhe_prefix} {n}-bit"
        elif key in modes_cfg:
            label = modes_cfg[key].get("label", key)
        else:
            label = key

        color = group_colors.get(key, "#999999")
        label_map[key] = label
        color_map[label] = color
        label_group_map[label] = group

    return label_map, color_map, label_group_map


# ===========================================================
# BOXPLOTS (BOOTSTRAP)
# ===========================================================

def _add_group_separators(ax, order, label_group_map, cfg, show_labels=True):
    """
    Draw vertical separator lines between mode groups and optionally annotate
    with group name labels just above the axes top.

    order           : list of display labels in x-axis order
    label_group_map : dict  display_label -> group name
    """
    sep_cfg = cfg.get("separators", {})
    if not sep_cfg.get("enabled", True):
        return

    groups_in_order = [label_group_map.get(lbl, "other") for lbl in order]
    groups_cfg = cfg.get("groups", {})

    separator_xpos = []
    group_spans = []   # (group_name, start_idx, end_idx)
    current_group = groups_in_order[0]
    current_start = 0

    for i in range(1, len(groups_in_order)):
        if groups_in_order[i] != current_group:
            separator_xpos.append(i - 0.5)
            group_spans.append((current_group, current_start, i - 1))
            current_group = groups_in_order[i]
            current_start = i
    group_spans.append((current_group, current_start, len(groups_in_order) - 1))

    for xpos in separator_xpos:
        ax.axvline(
            x=xpos,
            color=sep_cfg.get("color", "#cccccc"),
            linewidth=sep_cfg.get("linewidth", 0.8),
            linestyle=sep_cfg.get("linestyle", "--"),
            zorder=0,
        )

    if show_labels and sep_cfg.get("show_group_labels", True):
        xform = ax.get_xaxis_transform()
        for group_name, start_idx, end_idx in group_spans:
            center_x = (start_idx + end_idx) / 2.0
            label = groups_cfg.get(group_name, {}).get("label", group_name)
            ax.text(
                center_x, 1.01,
                label,
                transform=xform,
                ha="center", va="bottom",
                fontsize=sep_cfg.get("group_label_fontsize", 7),
                color=sep_cfg.get("group_label_color", "#aaaaaa"),
                fontstyle="italic",
                clip_on=False,
            )


def _draw_boxplot_panel(ax, subset, metric, order, color_map, box_cfg):
    """
    Render the core boxplot + stripplot layer onto ax.
    Caller is responsible for separators, styling, and labels.
    subset must already have a 'mode_label' column.
    """
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
        palette=color_map,
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
        palette=color_map,
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


def plot_boxplot(dataset, model, metric, df=None, cfg=None, save_dir=None,
                 bootstrap_path="results/bootstrap/aggregated.json",
                 viz_cfg_path="config/visualization.yaml"):
    """
    Boxplot of bootstrap distributions for one dataset / model / metric combination.

    x-axis  : modes (Real · Gaussian Copula · CTGAN · FHE N-bit), grouped and colored
    y-axis  : metric value across bootstrap seeds

    Pass pre-loaded df and cfg to avoid repeated I/O when called in a loop.
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)
    box_cfg = cfg["boxplot"]
    font_cfg = cfg["fonts"]
    fig_cfg = cfg["figures"]

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    if save_dir is None:
        save_dir = Path(fig_cfg["dir"])

    if df is None:
        df = load_bootstrap(bootstrap_path)

    df = df.copy()
    df["mode_key"] = df.apply(lambda r: _raw_mode_key(r["mode"], r["n_bits"]), axis=1)

    subset = df[(df["dataset"] == dataset) & (df["model"] == model)].dropna(subset=[metric])
    if subset.empty:
        return None, None

    label_map, color_map, label_group_map = _build_mode_display(cfg, subset["mode_key"].unique())
    subset = subset.copy()
    subset["mode_label"] = subset["mode_key"].map(label_map)

    sorted_keys = _sort_raw_keys(subset["mode_key"].unique())
    order = [label_map[k] for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 1.4), 5))

    _draw_boxplot_panel(ax, subset, metric, order, color_map, box_cfg)
    _add_group_separators(ax, order, label_group_map, cfg, show_labels=True)

    ax.set_title("")
    ax.set_xlabel("", fontsize=font_cfg["label_size"])
    ax.set_ylabel(format_metric_name(metric), fontsize=font_cfg["label_size"])
    ax.tick_params(labelsize=font_cfg["tick_size"])
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"boxplot_{metric}__{dataset}__{model}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()

    return fig, ax


# ===========================================================
# MAIN ENTRYPOINT
# ===========================================================

def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_viz_config()
    metrics = cfg.get("metrics", [])

    df = load_bootstrap()
    df = df.dropna(how="all")
    available_cols = set(df.columns)

    datasets = df["dataset"].dropna().unique()
    models = df["model"].dropna().unique()

    for metric in metrics:
        if metric not in available_cols:
            continue
        for dataset in datasets:
            for model in models:
                plot_boxplot(dataset, model, metric, df=df, cfg=cfg)