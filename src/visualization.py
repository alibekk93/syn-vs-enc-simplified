# src/visualization.py

import colorsys
import json
import re
from functools import lru_cache
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FIGURES_DIR = Path("results/figures")
BOOTSTRAP_PATH = "results/internal_validation_bootstrap/aggregated.json"


# ===========================================================
# DATA LOADING
# ===========================================================

def parse_filename_metadata(filename):
    """
    Extract mode / model / dataset / n_bits / synth_scale from a filename or raw mode key.

    Examples:
        fhe_4__logistic_regression__heart_disease.json  → mode=fhe, n_bits=4
        ctgan_100__rf__diabetes.json                    → mode=ctgan, synth_scale=100
        standard__rf__diabetes.json                     → mode=standard
        ctgan                                           → mode=ctgan  (bare JSON mode key)
    """
    name = Path(filename).stem
    parts = name.split("__")

    meta = {
        "raw_name": name,
        "mode": None,
        "model": None,
        "dataset": None,
        "n_bits": None,
        "synth_scale": None,
    }

    # FHE: fhe_N
    fhe_match = re.match(r"fhe_(\d+)", parts[0])
    if fhe_match:
        meta["mode"] = "fhe"
        meta["n_bits"] = int(fhe_match.group(1))
        parts[0] = "fhe"
    else:
        # Synthetic mode with synth_scale suffix: e.g. ctgan_100, gaussian_copula_150
        synth_match = re.match(r"^(.+)_(\d+)$", parts[0])
        if synth_match:
            meta["mode"] = synth_match.group(1)
            meta["synth_scale"] = int(synth_match.group(2))
        else:
            meta["mode"] = parts[0]

    if len(parts) >= 3:
        meta["model"] = parts[1]
        meta["dataset"] = parts[2]

    return meta


def load_internal_validation_bootstrap(path=BOOTSTRAP_PATH):
    """Flatten aggregated bootstrap JSON into a merged DataFrame (test split only for metrics)."""
    with open(path) as f:
        data = json.load(f)

    key_cols = ["mode", "n_bits", "synth_scale", "model", "dataset", "seed"]

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
                        "synth_scale": meta["synth_scale"],
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
                        "synth_scale": meta["synth_scale"],
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": entry["seed"],
                        "train_time": sum(entry.get("training_time", {}).values()),
                        "synth_fit_time": entry.get("training_time", {}).get("synthesis_fit"),
                        "fhe_fit_time": entry.get("training_time", {}).get("training_fit"),
                        "fhe_compile_time": entry.get("training_time", {}).get("training_compile"),
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


def _metrics_dir_from_config(models_cfg_path="config/models.yaml"):
    cfg = load_config(models_cfg_path)
    return cfg.get("output", {}).get("results_dir", "results/metrics")


def _profiles_dir_from_config(resource_cfg_path="config/resource_profiling.yaml"):
    cfg = load_config(resource_cfg_path)
    return cfg.get("logging", {}).get("output_dir", "results/resource_profiles")


def load_simple_bootstrap(
    metrics_dir=None,
    profiles_dir=None,
):
    """
    Load simple bootstrap results from per-file JSONs.

    Metrics files store each metric as a list of per-iteration values (one list
    per bootstrap iteration). Resource profiles are single scalar measurements.
    Returns a DataFrame with one row per bootstrap iteration per
    (mode, model, dataset) combination, with resource columns joined as constants.
    """
    if metrics_dir is None:
        metrics_dir = _metrics_dir_from_config()
    if profiles_dir is None:
        profiles_dir = _profiles_dir_from_config()

    key_cols = ["mode", "n_bits", "synth_scale", "model", "dataset"]

    metric_records = []
    for path in Path(metrics_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        meta = parse_filename_metadata(path.name)
        metrics = data.get("metrics", {})
        n = max((len(v) for v in metrics.values() if isinstance(v, list)), default=0)
        for i in range(n):
            metric_records.append({
                "mode": meta["mode"],
                "n_bits": meta["n_bits"],
                "synth_scale": meta["synth_scale"],
                "model": meta["model"],
                "dataset": meta["dataset"],
                "bootstrap_iter": i,
                **{k: v[i] for k, v in metrics.items() if isinstance(v, list) and i < len(v)},
            })

    profile_records = []
    for path in Path(profiles_dir).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        meta = parse_filename_metadata(path.name)
        profile_records.append({
            "mode": meta["mode"],
            "n_bits": meta["n_bits"],
            "synth_scale": meta["synth_scale"],
            "model": meta["model"],
            "dataset": meta["dataset"],
            "train_time": sum(data.get("training_time", {}).values()),
            "synth_fit_time": data.get("training_time", {}).get("synthesis_fit"),
            "fhe_fit_time": data.get("training_time", {}).get("training_fit"),
            "fhe_compile_time": data.get("training_time", {}).get("training_compile"),
            "inf_time_total": data.get("inference_time", {}).get("total"),
            "inf_time_per_sample": data.get("inference_time", {}).get("per_sample"),
            "mem_train_avg": data.get("memory", {}).get("training", {}).get("average_mb"),
            "mem_train_peak": data.get("memory", {}).get("training", {}).get("peak_mb"),
            "mem_inf_avg": data.get("memory", {}).get("inference", {}).get("average_mb"),
            "mem_inf_peak": data.get("memory", {}).get("inference", {}).get("peak_mb"),
            "model_size_mb": data.get("storage", {}).get("model_size_mb"),
            "data_size_mb": data.get("storage", {}).get("data_size_mb"),
            "circuit_complexity": data.get("fhe", {}).get("circuit_complexity"),
        })

    metrics_df = pd.DataFrame(metric_records)
    profiles_df = pd.DataFrame(profile_records)

    if metrics_df.empty:
        return profiles_df
    if profiles_df.empty:
        return metrics_df

    return pd.merge(metrics_df, profiles_df, on=key_cols, how="left")


# ===========================================================
# CONFIG UTIL
# ===========================================================

from src.utils import load_config


@lru_cache(maxsize=None)
def _load_viz_config(path="config/visualization.yaml"):
    return load_config(path)


# ===========================================================
# DISPLAY HELPERS
# ===========================================================

def format_metric_name(metric):
    return metric.replace("_", " ").title()


def _raw_mode_key(mode, n_bits):
    """Reconstruct the display key from parsed (mode, n_bits) fields."""
    if mode == "fhe" and pd.notna(n_bits):
        return f"fhe_{int(n_bits)}"
    return mode


def _sort_raw_keys(raw_keys):
    """
    Order: standard → synthetic modes (alphabetical) → fhe_N (ascending N).
    Any key that is not "standard" and does not start with "fhe_" is treated as
    synthetic, so new synthesizers slot in automatically without code changes.
    """
    def _key(k):
        if k == "standard":
            return (0, 0, k)
        if k.startswith("fhe_"):
            try:
                return (2, int(k.split("_")[1]), k)
            except ValueError:
                return (2, 0, k)
        return (1, 0, k)

    return sorted(raw_keys, key=_key)


def _lightness_ramp(base_color, n, lightness_range):
    """Return n hex colors along an HLS lightness ramp for the given base_color."""
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

    def _group_of(key):
        if key.startswith("fhe_"):
            return "fhe"
        return modes_cfg.get(key, {}).get("group", "other")

    sorted_keys = _sort_raw_keys(raw_keys)
    groups_present = {}
    for k in sorted_keys:
        g = _group_of(k)
        groups_present.setdefault(g, []).append(k)

    group_colors = {}

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
# VIOLIN PLOTS (BOOTSTRAP DISTRIBUTIONS)
# ===========================================================

def _add_group_separators(ax, order, label_group_map, cfg, show_labels=True):
    """
    Draw vertical separator lines between mode groups and optionally annotate
    with group name labels just above the axes top.
    """
    sep_cfg = cfg.get("separators", {})
    if not sep_cfg.get("enabled", True):
        return

    groups_in_order = [label_group_map.get(lbl, "other") for lbl in order]
    groups_cfg = cfg.get("groups", {})

    separator_xpos = []
    group_spans = []
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


def _draw_violinplot_panel(ax, subset, metric, order, color_map, violin_cfg):
    """Render violinplot + stripplot onto ax. subset must have a 'mode_label' column."""
    sns.violinplot(
        data=subset,
        x="mode_label",
        y=metric,
        hue="mode_label",
        hue_order=order,
        order=order,
        palette=color_map,
        linewidth=violin_cfg["linewidth"],
        inner=violin_cfg["inner"],
        cut=violin_cfg["cut"],
        legend=False,
        ax=ax,
    )

    for collection in ax.collections:
        collection.set_alpha(violin_cfg["alpha"])

    sns.stripplot(
        data=subset,
        x="mode_label",
        y=metric,
        hue="mode_label",
        hue_order=order,
        order=order,
        palette=color_map,
        dodge=False,
        alpha=violin_cfg["strip_alpha"],
        size=violin_cfg["strip_size"],
        legend=False,
        ax=ax,
    )

    if violin_cfg.get("despine", True):
        sns.despine(ax=ax)

    grid_alpha = violin_cfg.get("grid_alpha", 0.0)
    if grid_alpha > 0:
        ax.grid(axis="y", alpha=grid_alpha, linewidth=0.5)


def plot_violinplot(dataset, model, metric, df=None, cfg=None, save_dir=None,
                    bootstrap_path=BOOTSTRAP_PATH,
                    viz_cfg_path="config/visualization.yaml"):
    """
    Violin plot of bootstrap distributions for one dataset / model / metric combination.

    x-axis: modes (Real · Gaussian Copula · CTGAN · FHE N-bit), grouped and colored
    y-axis: metric value across bootstrap seeds

    Pass pre-loaded df and cfg to avoid repeated I/O when called in a loop.
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)
    violin_cfg = cfg["violinplot"]
    font_cfg = cfg["fonts"]
    fig_cfg = cfg["figures"]

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    if save_dir is None:
        save_dir = Path(fig_cfg["dir"])

    if df is None:
        df = load_internal_validation_bootstrap(bootstrap_path)

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

    # KDE-based violins must be computed in the scale they're meant to represent —
    # rescaling the axis after the fact (ax.set_yscale("log")) stretches a density
    # estimated on raw values, distorting the shape. So for log-scale metrics we
    # log-transform the data first and relabel the (linear) axis to show real units.
    log_scale = metric in cfg.get("log_scale_metrics", [])
    plot_col = metric
    if log_scale:
        plot_col = f"__log10_{metric}"
        subset[plot_col] = np.log10(subset[metric].clip(lower=1e-12))

    _draw_violinplot_panel(ax, subset, plot_col, order, color_map, violin_cfg)
    _add_group_separators(ax, order, label_group_map, cfg, show_labels=True)

    ylabel = format_metric_name(metric)
    if log_scale:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{10 ** val:g}"))
        ylabel += " (log scale)"

    ax.set_title("")
    ax.set_xlabel("", fontsize=font_cfg["label_size"])
    ax.set_ylabel(ylabel, fontsize=font_cfg["label_size"])
    ax.tick_params(labelsize=font_cfg["tick_size"])
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"violinplot_{metric}__{dataset}__{model}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()

    return fig, ax


# ===========================================================
# FHE COST DECOMPOSITION
# ===========================================================

_FHE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def plot_fhe_training_breakdown(df, save_dir=FIGURES_DIR, cfg=None,
                                viz_cfg_path="config/visualization.yaml"):
    """
    Horizontal stacked bar: fit time (grey) + compile time (FHE blue gradient by n_bits).
    One figure per dataset.  Filename: fhe_training_breakdown__{dataset}.{fmt}
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    font_cfg = cfg["fonts"]
    fig_cfg  = cfg["figures"]
    fhe_gcfg = cfg.get("groups", {}).get("fhe", {})

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    fhe_df = df[df["mode"] == "fhe"].dropna(
        subset=["fhe_fit_time", "fhe_compile_time", "n_bits"]
    )
    if fhe_df.empty:
        return

    for dataset in fhe_df["dataset"].dropna().unique():
        subset = fhe_df[fhe_df["dataset"] == dataset]

        agg = (
            subset.groupby(["model", "n_bits"])[["fhe_fit_time", "fhe_compile_time"]]
            .mean()
            .reset_index()
            .sort_values(["model", "n_bits"])
            .reset_index(drop=True)
        )

        y_labels = [
            f"{r.model.replace('_', ' ').title()}  n={int(r.n_bits)}"
            for _, r in agg.iterrows()
        ]
        y_pos = list(range(len(agg)))

        n_bits_sorted  = sorted(agg["n_bits"].unique())
        base_color     = fhe_gcfg.get("base_color", "#1f77b4")
        l_range        = fhe_gcfg.get("lightness_range", [0.72, 0.32])
        compile_colors = _lightness_ramp(base_color, len(n_bits_sorted), l_range)
        nbits_color    = dict(zip(n_bits_sorted, compile_colors))

        fig, ax = plt.subplots(figsize=(8, max(4, len(agg) * 0.45)))

        ax.barh(y_pos, agg["fhe_fit_time"], color="#cccccc", height=0.6, label="Fit")

        for nb in n_bits_sorted:
            nb_rows  = agg[agg["n_bits"] == nb]
            y_subset = nb_rows.index.tolist()
            ax.barh(
                y_subset,
                nb_rows["fhe_compile_time"].values,
                left=nb_rows["fhe_fit_time"].values,
                color=nbits_color[nb],
                height=0.6,
                label=f"Compile  n={int(nb)}",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=font_cfg["tick_size"])
        ax.set_xlabel("Time (s)", fontsize=font_cfg["label_size"])
        ax.set_title("")
        ax.legend(
            fontsize=max(font_cfg["tick_size"] - 2, 8),
            loc="lower right",
            frameon=False,
            ncol=2,
        )
        sns.despine(ax=ax)
        plt.tight_layout()

        fmt       = fig_cfg["format"]
        save_path = Path(save_dir) / f"fhe_training_breakdown__{dataset}.{fmt}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()


def plot_fhe_complexity_cost(df, save_dir=FIGURES_DIR, cfg=None,
                             viz_cfg_path="config/visualization.yaml"):
    """
    Two-panel scatter: circuit complexity vs compile time (left) and vs inference
    time per sample (right).  Color encodes n_bits; marker shape encodes model.
    One figure per dataset.  Filename: fhe_complexity_cost__{dataset}.{fmt}
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    font_cfg = cfg["fonts"]
    fig_cfg  = cfg["figures"]
    fhe_gcfg = cfg.get("groups", {}).get("fhe", {})

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    needed   = ["circuit_complexity", "fhe_compile_time", "inf_time_per_sample", "n_bits"]
    agg_cols = ["circuit_complexity", "fhe_compile_time", "inf_time_per_sample"]
    fhe_df = df[df["mode"] == "fhe"].dropna(subset=needed)
    if fhe_df.empty:
        return

    for dataset in fhe_df["dataset"].dropna().unique():
        subset = fhe_df[fhe_df["dataset"] == dataset]

        agg = (
            subset.groupby(["model", "n_bits"])[agg_cols]
            .mean()
            .reset_index()
        )

        models_sorted = sorted(agg["model"].unique())
        n_bits_sorted = sorted(agg["n_bits"].unique())
        model_markers = {
            m: _FHE_MARKERS[i % len(_FHE_MARKERS)]
            for i, m in enumerate(models_sorted)
        }

        base_color   = fhe_gcfg.get("base_color", "#1f77b4")
        l_range      = fhe_gcfg.get("lightness_range", [0.72, 0.32])
        point_colors = _lightness_ramp(base_color, len(n_bits_sorted), l_range)
        nbits_color  = dict(zip(n_bits_sorted, point_colors))

        panels = [
            ("fhe_compile_time",    "Compile Time (s)"),
            ("inf_time_per_sample", "Inference Time per Sample (s)"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for (y_col, y_label), ax in zip(panels, axes):
            for model in models_sorted:
                m_data = agg[agg["model"] == model].sort_values("circuit_complexity")
                ax.plot(
                    m_data["circuit_complexity"],
                    m_data[y_col],
                    color="#cccccc",
                    linewidth=0.8,
                    zorder=1,
                )

            for model in models_sorted:
                for nb in n_bits_sorted:
                    pt = agg[(agg["model"] == model) & (agg["n_bits"] == nb)]
                    if pt.empty:
                        continue
                    ax.scatter(
                        pt["circuit_complexity"].values,
                        pt[y_col].values,
                        color=nbits_color[nb],
                        marker=model_markers[model],
                        s=55,
                        zorder=2,
                        edgecolors="none",
                    )

            ax.set_xlabel("Circuit Complexity", fontsize=font_cfg["label_size"])
            ax.set_ylabel(y_label, fontsize=font_cfg["label_size"])
            ax.set_title("")
            ax.tick_params(labelsize=font_cfg["tick_size"])
            sns.despine(ax=ax)

        legend_fs       = max(font_cfg["tick_size"] - 2, 8)
        legend_title_fs = max(font_cfg["tick_size"] - 1, 9)

        color_handles = [
            mpatches.Patch(color=nbits_color[nb], label=f"n={int(nb)}")
            for nb in n_bits_sorted
        ]
        marker_handles = [
            Line2D(
                [0], [0],
                marker=model_markers[m],
                color="grey",
                linestyle="None",
                markersize=7,
                label=m.replace("_", " ").title(),
            )
            for m in models_sorted
        ]

        axes[0].legend(
            handles=color_handles,
            title="Precision",
            fontsize=legend_fs,
            title_fontsize=legend_title_fs,
            loc="upper left",
            frameon=False,
        )
        axes[1].legend(
            handles=marker_handles,
            title="Model",
            fontsize=legend_fs,
            title_fontsize=legend_title_fs,
            loc="upper left",
            frameon=False,
        )

        plt.tight_layout()

        fmt       = fig_cfg["format"]
        save_path = Path(save_dir) / f"fhe_complexity_cost__{dataset}.{fmt}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()


# ===========================================================
# MAIN ENTRYPOINT
# ===========================================================

def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_viz_config()
    metrics = cfg.get("metrics", [])

    df = load_simple_bootstrap()
    df = df.dropna(how="all")

    # For synth modes with synth_scale variants, keep only synth_scale=100
    df = df[df["synth_scale"].isna() | (df["synth_scale"] == 100)]

    available_cols = set(df.columns)

    datasets = df["dataset"].dropna().unique()
    models = df["model"].dropna().unique()

    for metric in metrics:
        if metric not in available_cols:
            continue
        for dataset in datasets:
            for model in models:
                plot_violinplot(dataset, model, metric, df=df, cfg=cfg)

    plot_fhe_training_breakdown(df, cfg=cfg)
    plot_fhe_complexity_cost(df, cfg=cfg)
