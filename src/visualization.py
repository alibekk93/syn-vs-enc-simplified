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

    def _load_json_first(p):
        """Load the first complete JSON object from a file, ignoring any trailing content."""
        with open(p) as f:
            content = f.read().strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            obj, _ = json.JSONDecoder().raw_decode(content)
            return obj

    metric_records = []
    for path in Path(metrics_dir).glob("*.json"):
        data = _load_json_first(path)
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
        data = _load_json_first(path)
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

# Distinct qualitative colours for per-model lines (complexity cost redesign).
# Chosen to avoid conflict with the project group colours:
#   blue (#1f77b4) = FHE, orange (#e07b2a) = Synthetic, green (#2ca02c) = Real
_MODEL_COLORS = ["#d62728", "#9467bd", "#17becf", "#e377c2", "#8c564b"]


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

        ax.barh(y_pos, agg["fhe_fit_time"], color="#cccccc", height=0.6)

        for nb in n_bits_sorted:
            nb_rows  = agg[agg["n_bits"] == nb]
            y_subset = nb_rows.index.tolist()
            ax.barh(
                y_subset,
                nb_rows["fhe_compile_time"].values,
                left=nb_rows["fhe_fit_time"].values,
                color=nbits_color[nb],
                height=0.6,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=font_cfg["tick_size"])
        ax.set_xlabel("Time (s)", fontsize=font_cfg["label_size"])
        ax.set_title("")
        # Simplified legend: y-axis labels already carry model+n_bits info.
        # Show only the two segment types; a caption note covers the shade gradient.
        import matplotlib.patches as _mp
        _mid_nb = n_bits_sorted[len(n_bits_sorted) // 2]
        ax.legend(
            handles=[
                _mp.Patch(color="#cccccc", label="Fit"),
                _mp.Patch(color=nbits_color[_mid_nb], label="Compile\n(light→dark: low→high bits)"),
            ],
            fontsize=max(font_cfg["tick_size"] - 2, 8),
            loc="lower right",
            frameon=False,
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
    Two-panel line plot: n_bits (x) vs compile time (left) and inference time per
    sample (right).  Each line is one model; markers distinguish models.
    One figure per dataset.  Filename: fhe_complexity_cost__{dataset}.{fmt}

    Redesigned from the original scatter (circuit_complexity vs time): circuit
    complexity values cluster in a very narrow range so that scatter offers no
    spatial separation.  n_bits is the actual control variable and gives a clean,
    readable x-axis.
    """
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    font_cfg = cfg["fonts"]
    fig_cfg  = cfg["figures"]

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    needed   = ["n_bits", "fhe_compile_time", "inf_time_per_sample"]
    agg_cols = ["fhe_compile_time", "inf_time_per_sample"]
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
        model_markers = {m: _FHE_MARKERS[i % len(_FHE_MARKERS)] for i, m in enumerate(models_sorted)}
        model_colors  = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)]  for i, m in enumerate(models_sorted)}

        panels = [
            ("fhe_compile_time",    "Compile Time (s)"),
            ("inf_time_per_sample", "Inference Time per Sample (s)"),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for (y_col, y_label), ax in zip(panels, axes):
            for model in models_sorted:
                m_data = agg[agg["model"] == model].sort_values("n_bits")
                if m_data.empty:
                    continue
                ax.plot(
                    m_data["n_bits"], m_data[y_col],
                    color=model_colors[model],
                    marker=model_markers[model],
                    linewidth=1.5, markersize=6, zorder=2,
                )

            ax.set_xlabel("Precision (bits)", fontsize=font_cfg["label_size"])
            ax.set_ylabel(y_label, fontsize=font_cfg["label_size"])
            ax.set_xticks(n_bits_sorted)
            ax.tick_params(labelsize=font_cfg["tick_size"])
            sns.despine(ax=ax)

        legend_fs = max(font_cfg["tick_size"] - 2, 8)
        model_handles = [
            Line2D([0], [0],
                   color=model_colors[m], marker=model_markers[m],
                   linewidth=1.5, markersize=6,
                   label=m.replace("_", " ").title())
            for m in models_sorted
        ]
        fig.legend(
            handles=model_handles,
            fontsize=legend_fs,
            ncol=len(model_handles),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        fmt       = fig_cfg["format"]
        save_path = Path(save_dir) / f"fhe_complexity_cost__{dataset}.{fmt}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close()


# ===========================================================
# SYNTH SCALE LINE PLOTS
# ===========================================================

def plot_synth_scale_lines(dataset, model, metric, df=None, cfg=None, save_dir=None,
                            viz_cfg_path="config/visualization.yaml"):
    """
    Line plot of metric vs synth_scale for each synthesizer method.

    Lines:   one per synth method (orange shades, IQR band)
    Dashed:  raw/standard performance (green horizontal reference, IQR band)
    Dotted:  FHE 8-bit performance (blue horizontal reference, IQR band)
    x-axis:  synth_scale (100, 150, 300)
    y-axis:  metric (median of bootstrap iterations)

    Pass a df that retains all synth_scale values (do not pre-filter to scale=100).
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    font_cfg = cfg["fonts"]
    fig_cfg = cfg["figures"]
    modes_cfg = cfg.get("modes", {})
    groups_cfg = cfg.get("groups", {})

    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams["font.family"] = font_cfg.get("family", "sans-serif")

    if save_dir is None:
        save_dir = Path(fig_cfg["dir"])

    if df is None:
        df = load_simple_bootstrap()

    if metric not in df.columns:
        return None, None

    subset = df[(df["dataset"] == dataset) & (df["model"] == model)].dropna(subset=[metric])
    if subset.empty:
        return None, None

    # Synthetic modes: rows with a synth_scale value (not standard, not fhe)
    synth_df = subset[subset["synth_scale"].notna()].copy()
    synth_methods = sorted(m for m in synth_df["mode"].unique() if m not in ("standard", "fhe"))

    std_df = subset[subset["mode"] == "standard"]
    fhe_df = subset[(subset["mode"] == "fhe") & (subset["n_bits"] == 8)]

    if not synth_methods and std_df.empty and fhe_df.empty:
        return None, None

    # Colors for synth methods via the synthetic group lightness ramp
    synth_gcfg = groups_cfg.get("synthetic", {})
    base_color = synth_gcfg.get("base_color", "#e07b2a")
    l_range = synth_gcfg.get("lightness_range", [0.65, 0.38])
    if len(synth_methods) > 1:
        method_colors = dict(zip(synth_methods, _lightness_ramp(base_color, len(synth_methods), l_range)))
    elif synth_methods:
        method_colors = {synth_methods[0]: base_color}
    else:
        method_colors = {}

    synth_scales = sorted(synth_df["synth_scale"].dropna().unique().astype(int)) if not synth_df.empty else []

    fig, ax = plt.subplots(figsize=(7, 5))
    label_fs = max(font_cfg["tick_size"] - 2, 7)
    line_labels = []  # (y_data_value, text, color) — collected for right-side annotation

    for method in synth_methods:
        m_df = synth_df[synth_df["mode"] == method]
        grp = m_df.groupby("synth_scale")[metric]
        agg = pd.DataFrame({
            "median": grp.median(),
            "q25": grp.quantile(0.25),
            "q75": grp.quantile(0.75),
        }).reset_index().sort_values("synth_scale")

        if agg.empty:
            continue

        color = method_colors[method]
        label = modes_cfg.get(method, {}).get("label", method)
        ax.plot(agg["synth_scale"], agg["median"], color=color, marker="o",
                linewidth=1.8, markersize=5, zorder=3)
        ax.fill_between(agg["synth_scale"], agg["q25"], agg["q75"],
                        color=color, alpha=0.15, zorder=1)
        line_labels.append((agg["median"].iloc[-1], label, color))

    real_gcfg = groups_cfg.get("real", {})
    real_color = real_gcfg.get("base_color", "#2ca02c")
    if not std_df.empty:
        std_med = std_df[metric].median()
        ax.axhline(std_med, color=real_color, linewidth=1.8, linestyle="--", zorder=5)
        ax.axhspan(std_df[metric].quantile(0.25), std_df[metric].quantile(0.75),
                   color=real_color, alpha=0.10, zorder=0)
        line_labels.append((std_med, "Real", real_color))

    fhe_gcfg = groups_cfg.get("fhe", {})
    fhe_color = fhe_gcfg.get("base_color", "#1f77b4")
    if not fhe_df.empty:
        fhe_med = fhe_df[metric].median()
        ax.axhline(fhe_med, color=fhe_color, linewidth=1.8, linestyle=":", zorder=5)
        ax.axhspan(fhe_df[metric].quantile(0.25), fhe_df[metric].quantile(0.75),
                   color=fhe_color, alpha=0.10, zorder=0)
        line_labels.append((fhe_med, "FHE 8-bit", fhe_color))

    ax.set_xlabel("Synth Scale (%)", fontsize=font_cfg["label_size"])
    ax.set_ylabel(format_metric_name(metric), fontsize=font_cfg["label_size"])
    ax.tick_params(labelsize=font_cfg["tick_size"])

    if synth_scales:
        ax.set_xticks(synth_scales)
        ax.set_xticklabels([f"{int(s)}%" for s in synth_scales])

    # Draw labels to the right of each line using axes-x / data-y transform
    yaxis_xform = ax.get_yaxis_transform()
    for y_val, text, color in line_labels:
        ax.text(1.02, y_val, text, transform=yaxis_xform,
                color=color, va="center", ha="left",
                fontsize=label_fs, clip_on=False)

    sns.despine(ax=ax)
    plt.tight_layout()

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"synth_scale_lines_{metric}__{dataset}__{model}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()

    return fig, ax


# ===========================================================
# IEEE PUBLICATION MULTIPANEL FIGURES
# ===========================================================
#
# IEEE two-column conference format:
#   - Full-width figure: 7.16 in
#   - Single-column figure: 3.5 in
#   - Minimum font size: 6 pt (8 pt recommended)
#   - pdf.fonttype=42 / ps.fonttype=42 keeps text editable in Illustrator/Inkscape

_IEEE_FULL_WIDTH_IN = 7.16

_DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "maternal_health_risk": "Maternal Health",
    "pregnancy_outcome": "Preg. Outcome",
}

_MODEL_ABBREV = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
    "mlp": "MLP",
}


def _fmt_dataset(name: str) -> str:
    return _DATASET_LABELS.get(name, name.replace("_", " ").title())


def _abbrev_model(name: str) -> str:
    return _MODEL_ABBREV.get(name, name.replace("_", " ").title())


def _add_panel_label(ax, idx: int, fontsize: int = 8):
    """Bold (a), (b), … label at the upper-left of an axes panel."""
    letter = chr(ord("a") + idx)
    ax.text(
        -0.14, 1.05, f"({letter})",
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="bottom", ha="left",
        clip_on=False,
    )


def plot_fhe_complexity_cost_multipanel(
    df, save_dir=FIGURES_DIR, cfg=None, viz_cfg_path="config/visualization.yaml"
):
    """
    IEEE double-column multipanel figure — FHE complexity cost.

    Layout: 2 rows × N_datasets columns.
      Row 0: precision (n_bits) vs compile time.
      Row 1: precision (n_bits) vs inference time per sample.
    Each line is one model.  n_bits on x-axis is the natural control variable;
    the original circuit-complexity x-axis had near-identical values for all
    points (narrow range), making the scatter unreadable.
    Saved as: fhe_complexity_cost_multipanel.{fmt}
    """
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    fig_cfg = cfg["figures"]

    plt.rcParams.update({
        "font.family": cfg["fonts"].get("family", "sans-serif"),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    needed   = ["n_bits", "fhe_compile_time", "inf_time_per_sample"]
    agg_cols = ["fhe_compile_time", "inf_time_per_sample"]
    fhe_df = df[df["mode"] == "fhe"].dropna(subset=needed)
    if fhe_df.empty:
        return

    full_agg = (
        fhe_df.groupby(["dataset", "model", "n_bits"])[agg_cols]
        .mean()
        .reset_index()
    )

    datasets      = sorted(full_agg["dataset"].dropna().unique())
    models_sorted = sorted(full_agg["model"].unique())
    n_bits_sorted = sorted(full_agg["n_bits"].unique())
    n_col         = len(datasets)

    model_markers = {m: _FHE_MARKERS[i % len(_FHE_MARKERS)] for i, m in enumerate(models_sorted)}
    model_colors  = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)]  for i, m in enumerate(models_sorted)}

    panels = [
        ("fhe_compile_time",    "Compile Time (s)"),
        ("inf_time_per_sample", "Inf. Time / Sample (s)"),
    ]

    tick_fs  = 7
    label_fs = 8

    fig, axes = plt.subplots(
        2, n_col,
        figsize=(_IEEE_FULL_WIDTH_IN, 3.6),
        sharex="col",  # same n_bits x-axis per column (same for all datasets, but keeps ticks tidy)
        sharey=False,  # independent y-scale per panel
    )
    if n_col == 1:
        axes = axes.reshape(2, 1)

    for col_idx, dataset in enumerate(datasets):
        d_agg = full_agg[full_agg["dataset"] == dataset]

        for row_idx, (y_col, y_label) in enumerate(panels):
            ax = axes[row_idx, col_idx]

            for model in models_sorted:
                m_data = d_agg[d_agg["model"] == model].sort_values("n_bits")
                if m_data.empty:
                    continue
                ax.plot(
                    m_data["n_bits"], m_data[y_col],
                    color=model_colors[model],
                    marker=model_markers[model],
                    linewidth=1.2, markersize=4, zorder=2,
                )

            ax.set_xticks(n_bits_sorted)
            ax.tick_params(labelsize=tick_fs, length=3, pad=2)
            sns.despine(ax=ax)

            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=label_fs, labelpad=3)

            if row_idx == len(panels) - 1:
                ax.set_xlabel("Precision (bits)", fontsize=label_fs, labelpad=3)

            if row_idx == 0:
                ax.set_title(_fmt_dataset(dataset), fontsize=label_fs, pad=4)

            _add_panel_label(ax, row_idx * n_col + col_idx, fontsize=label_fs)

    # Shared legend — only model entries needed (n_bits is now the x-axis)
    legend_fs = 7
    model_handles = [
        Line2D([0], [0],
               color=model_colors[m], marker=model_markers[m],
               linewidth=1.2, markersize=4, label=_abbrev_model(m))
        for m in models_sorted
    ]
    fig.legend(
        handles=model_handles,
        fontsize=legend_fs,
        ncol=len(model_handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    fmt       = fig_cfg["format"]
    save_path = Path(save_dir) / f"fhe_complexity_cost_multipanel.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()


def plot_fhe_training_breakdown_multipanel(
    df, save_dir=FIGURES_DIR, cfg=None, viz_cfg_path="config/visualization.yaml"
):
    """
    IEEE double-column multipanel figure — FHE training time breakdown.

    Layout: 1 row × N_datasets columns, horizontal stacked bars.
      Grey segment  = fit time.
      Coloured segment = compile time (blue gradient, shade = n_bits).
    Panels share the same y-axis (model × precision categories) so the
    category labels appear only once on the left.
    Saved as: fhe_training_breakdown_multipanel.{fmt}
    """
    from matplotlib.patches import Patch

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    fig_cfg = cfg["figures"]
    fhe_gcfg = cfg.get("groups", {}).get("fhe", {})

    plt.rcParams.update({
        "font.family": cfg["fonts"].get("family", "sans-serif"),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fhe_df = df[df["mode"] == "fhe"].dropna(
        subset=["fhe_fit_time", "fhe_compile_time", "n_bits"]
    )
    if fhe_df.empty:
        return

    full_agg = (
        fhe_df.groupby(["dataset", "model", "n_bits"])[["fhe_fit_time", "fhe_compile_time"]]
        .mean()
        .reset_index()
    )

    datasets      = sorted(full_agg["dataset"].dropna().unique())
    models_sorted = sorted(full_agg["model"].unique())
    n_bits_sorted = sorted(full_agg["n_bits"].unique())
    n_col         = len(datasets)

    # Canonical bar order: outer = model (alphabetical), inner = n_bits (ascending)
    y_categories = [
        (model, int(nb)) for model in models_sorted for nb in n_bits_sorted
    ]
    y_labels_left = [f"{_abbrev_model(m)} {n}b" for m, n in y_categories]
    n_bars = len(y_categories)
    y_pos  = list(range(n_bars))

    base_color = fhe_gcfg.get("base_color", "#1f77b4")
    l_range    = fhe_gcfg.get("lightness_range", [0.72, 0.32])
    nbits_color = dict(
        zip(n_bits_sorted, _lightness_ramp(base_color, len(n_bits_sorted), l_range))
    )

    tick_fs  = 7
    label_fs = 8
    bar_h    = 0.60

    fig_h = max(3.0, n_bars * 0.22 + 0.8)
    fig, axes = plt.subplots(
        1, n_col,
        figsize=(_IEEE_FULL_WIDTH_IN, fig_h),
        sharey=True,
    )
    if n_col == 1:
        axes = [axes]

    def _safe_val(lookup, cat, col):
        row = lookup.get(cat)
        if row is None:
            return 0.0
        v = row[col]
        return float(v) if pd.notna(v) else 0.0

    for col_idx, dataset in enumerate(datasets):
        ax    = axes[col_idx]
        d_agg = full_agg[full_agg["dataset"] == dataset]
        d_lut = {(r["model"], int(r["n_bits"])): r for _, r in d_agg.iterrows()}

        fit_vals     = [_safe_val(d_lut, cat, "fhe_fit_time")     for cat in y_categories]
        compile_vals = [_safe_val(d_lut, cat, "fhe_compile_time") for cat in y_categories]
        bar_colors   = [nbits_color[nb] for _, nb in y_categories]

        ax.barh(y_pos, fit_vals,     height=bar_h, color="#cccccc",   zorder=2)
        ax.barh(y_pos, compile_vals, height=bar_h, color=bar_colors,
                left=fit_vals, zorder=2)

        ax.set_title(_fmt_dataset(dataset), fontsize=label_fs, pad=4)
        ax.set_xlabel("Time (s)", fontsize=label_fs, labelpad=3)
        ax.tick_params(labelsize=tick_fs, length=3, pad=2)
        sns.despine(ax=ax)

        _add_panel_label(ax, col_idx, fontsize=label_fs)

    # y-tick labels only on leftmost panel (sharey hides the rest automatically)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(y_labels_left, fontsize=tick_fs)
    for ax in axes:
        ax.set_ylim(-0.5, n_bars - 0.5)

    # Thin horizontal separators between model groups
    n_per_model = len(n_bits_sorted)
    for ax in axes:
        for i in range(len(models_sorted) - 1):
            ax.axhline(
                (i + 1) * n_per_model - 0.5,
                color="#dddddd", linewidth=0.5, zorder=1,
            )

    # Simplified shared legend — y-axis labels already encode model + n_bits,
    # so individual per-precision colour patches are redundant.  Show only the
    # two segment types; caption can note that darker shade = higher bit-width.
    _mid_nb = n_bits_sorted[len(n_bits_sorted) // 2]
    fit_handle     = Patch(color="#cccccc", label="Fit")
    compile_handle = Patch(color=nbits_color[_mid_nb],
                           label="Compile  (light→dark: low→high bits)")
    fig.legend(
        handles=[fit_handle, compile_handle],
        fontsize=7,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fmt       = fig_cfg["format"]
    save_path = Path(save_dir) / f"fhe_training_breakdown_multipanel.{fmt}"
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

    df_all = load_simple_bootstrap()
    df_all = df_all.dropna(how="all")

    # For violin plots: keep only synth_scale=100 (ignore 150, 300 variants)
    df = df_all[df_all["synth_scale"].isna() | (df_all["synth_scale"] == 100)]

    available_cols = set(df_all.columns)

    datasets = df_all["dataset"].dropna().unique()
    models = df_all["model"].dropna().unique()

    for metric in metrics:
        if metric not in available_cols:
            continue
        for dataset in datasets:
            for model in models:
                plot_violinplot(dataset, model, metric, df=df, cfg=cfg)
                plot_synth_scale_lines(dataset, model, metric, df=df_all, cfg=cfg)

    plot_fhe_training_breakdown(df, cfg=cfg)
    plot_fhe_complexity_cost(df, cfg=cfg)
    plot_fhe_training_breakdown_multipanel(df, cfg=cfg)
    plot_fhe_complexity_cost_multipanel(df, cfg=cfg)
