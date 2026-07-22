# src/visualization.py

import colorsys
import json
from functools import lru_cache
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.utils import parse_filename_metadata

FIGURES_DIR = Path("results/figures")
BOOTSTRAP_PATH = "results/internal_validation_bootstrap/aggregated.json"


# ===========================================================
# DATA LOADING
# ===========================================================


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


def _mode_color(cfg, key):
    """
    Single flat colour for a mode_key (no lightness ramp).

    Explicit ``modes[key].color`` wins; otherwise the mode's group ``base_color``.
    Used where one colour per mode is needed (synth-scale lines, reference bands).
    FHE bit-width shading is handled separately by _build_mode_display.
    """
    modes_cfg = cfg.get("modes", {})
    groups_cfg = cfg.get("groups", {})
    explicit = modes_cfg.get(key, {}).get("color")
    if explicit:
        return explicit
    group = "fhe" if key.startswith("fhe_") else modes_cfg.get(key, {}).get("group", "other")
    return groups_cfg.get(group, {}).get("base_color", "#999999")


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

        # Ramp only when a group asks for it (FHE, by bit-width); an explicit
        # per-mode color always wins over the ramp/base (synth methods).
        ramp_map = {}
        if l_range and len(members) > 1:
            ramp_map = dict(zip(members, _lightness_ramp(base, len(members), l_range)))

        for k in members:
            explicit = modes_cfg.get(k, {}).get("color")
            if explicit:
                group_colors[k] = explicit
            elif k in ramp_map:
                group_colors[k] = ramp_map[k]
            else:
                group_colors[k] = base

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


def _draw_violinplot_panel(ax, subset, metric, order, color_map, violin_cfg, show_strip=True):
    """Render violinplot (+ optional stripplot) onto ax. subset must have a 'mode_label' column."""
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

    if show_strip:
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

# Fallback per-model line colours for the complexity-cost plots (config:
# `model_line_colors`). That figure shows no modes, so this palette is deliberately
# independent of the mode colours — one colour per model, plus a distinct marker.
_MODEL_COLORS = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00"]


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
    fit_color = cfg.get("fhe_breakdown", {}).get("fit_color", "#cccccc")

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

        ax.barh(y_pos, agg["fhe_fit_time"], color=fit_color, height=0.6)

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
                _mp.Patch(color=fit_color, label="Fit"),
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
        model_palette = cfg.get("model_line_colors", _MODEL_COLORS)
        model_markers = {m: _FHE_MARKERS[i % len(_FHE_MARKERS)] for i, m in enumerate(models_sorted)}
        model_colors  = {m: model_palette[i % len(model_palette)]  for i, m in enumerate(models_sorted)}

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

    # Distinct per-method colors from config (same colour a method has in the violins).
    method_colors = {m: _mode_color(cfg, m) for m in synth_methods}

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

    real_color = _mode_color(cfg, "standard")
    if not std_df.empty:
        std_med = std_df[metric].median()
        ax.axhline(std_med, color=real_color, linewidth=1.8, linestyle="--", zorder=5)
        ax.axhspan(std_df[metric].quantile(0.25), std_df[metric].quantile(0.75),
                   color=real_color, alpha=0.10, zorder=0)
        line_labels.append((std_med, "Real", real_color))

    fhe_color = _mode_color(cfg, "fhe_8")
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

# Short x-tick codes for dense mode axes (violin multipanel) — keep the axis readable
# at ~0.5in per slot without overlapping full names.
_MODE_SHORT = {
    "standard": "Real",
    "arf": "ARF",
    "bayesian_network": "BN",
    "ctgan": "CTGAN",
    "gaussian_copula": "GC",
    "nflow": "NF",
}


def _fmt_dataset(name: str) -> str:
    return _DATASET_LABELS.get(name, name.replace("_", " ").title())


def _abbrev_model(name: str) -> str:
    return _MODEL_ABBREV.get(name, name.replace("_", " ").title())


def _mode_short_code(mode_key: str) -> str:
    """Short x-tick code for a mode_key."""
    if mode_key.startswith("fhe_"):
        return "FHE-" + mode_key.split("_")[1]     # fhe_8 -> FHE-8
    if mode_key in _MODE_SHORT:
        return _MODE_SHORT[mode_key]
    return "".join(w[0] for w in mode_key.split("_")).upper()  # fallback initials


def _add_panel_label(ax, idx: int, fontsize: int = 8, x: float = -0.14, suffix: str = ""):
    """
    Bold (a), (b), … label at the upper-left of an axes panel.

    x       : horizontal offset in axes coords. Use a smaller magnitude for
              full-width single-column panels than for narrow grid panels.
    suffix  : optional text appended after the letter, e.g. a model tag → "(a) LR".
    """
    letter = chr(ord("a") + idx)
    text = f"({letter}) {suffix}" if suffix else f"({letter})"
    ax.text(
        x, 1.05, text,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="bottom", ha="left",
        clip_on=False,
    )


def _apply_publication_style(cfg):
    """
    Apply the shared seaborn style/context + IEEE-friendly rcParams used by every
    multipanel figure. Keeps each function self-contained (no reliance on global
    state left by a previously drawn figure) and consistent with the single-panel
    setup. fonttype 42 keeps text editable in Illustrator/Inkscape.
    """
    sns.set_style(cfg.get("style", "white"))
    sns.set_context(cfg.get("context", "paper"))
    plt.rcParams.update({
        "font.family": cfg["fonts"].get("family", "sans-serif"),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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

    _apply_publication_style(cfg)

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

    model_palette = cfg.get("model_line_colors", _MODEL_COLORS)
    model_markers = {m: _FHE_MARKERS[i % len(_FHE_MARKERS)] for i, m in enumerate(models_sorted)}
    model_colors  = {m: model_palette[i % len(model_palette)]  for i, m in enumerate(models_sorted)}

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
        squeeze=False,
    )

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
    breakdown_cfg = cfg.get("fhe_breakdown", {})
    fit_color     = breakdown_cfg.get("fit_color", "#cccccc")
    divider_color = breakdown_cfg.get("group_divider_color", "#dddddd")

    _apply_publication_style(cfg)

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
        squeeze=False,
    )

    def _safe_val(lookup, cat, col):
        row = lookup.get(cat)
        if row is None:
            return 0.0
        v = row[col]
        return float(v) if pd.notna(v) else 0.0

    for col_idx, dataset in enumerate(datasets):
        ax    = axes[0, col_idx]
        d_agg = full_agg[full_agg["dataset"] == dataset]
        d_lut = {(r["model"], int(r["n_bits"])): r for _, r in d_agg.iterrows()}

        fit_vals     = [_safe_val(d_lut, cat, "fhe_fit_time")     for cat in y_categories]
        compile_vals = [_safe_val(d_lut, cat, "fhe_compile_time") for cat in y_categories]
        bar_colors   = [nbits_color[nb] for _, nb in y_categories]

        ax.barh(y_pos, fit_vals,     height=bar_h, color=fit_color,   zorder=2)
        ax.barh(y_pos, compile_vals, height=bar_h, color=bar_colors,
                left=fit_vals, zorder=2)

        ax.set_title(_fmt_dataset(dataset), fontsize=label_fs, pad=4)
        ax.set_xlabel("Time (s)", fontsize=label_fs, labelpad=3)
        ax.tick_params(labelsize=tick_fs, length=3, pad=2)
        sns.despine(ax=ax)

        _add_panel_label(ax, col_idx, fontsize=label_fs)

    # y-tick labels only on leftmost panel (sharey hides the rest automatically)
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(y_labels_left, fontsize=tick_fs)
    for ax in axes.flat:
        ax.set_ylim(-0.5, n_bars - 0.5)

    # Thin horizontal separators between model groups
    n_per_model = len(n_bits_sorted)
    for ax in axes.flat:
        for i in range(len(models_sorted) - 1):
            ax.axhline(
                (i + 1) * n_per_model - 0.5,
                color=divider_color, linewidth=0.5, zorder=1,
            )

    # Simplified shared legend — y-axis labels already encode model + n_bits,
    # so individual per-precision colour patches are redundant.  Show only the
    # two segment types; caption can note that darker shade = higher bit-width.
    legend_fs = 7
    _mid_nb = n_bits_sorted[len(n_bits_sorted) // 2]
    fit_handle     = Patch(color=fit_color, label="Fit")
    compile_handle = Patch(color=nbits_color[_mid_nb],
                           label="Compile  (light→dark: low→high bits)")
    fig.legend(
        handles=[fit_handle, compile_handle],
        fontsize=legend_fs,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fmt       = fig_cfg["format"]
    save_path = Path(save_dir) / f"fhe_training_breakdown_multipanel.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()


def plot_synth_scale_lines_multipanel(
    df, metric="roc_auc", save_dir=FIGURES_DIR, cfg=None,
    viz_cfg_path="config/visualization.yaml",
):
    """
    IEEE double-column multipanel figure — synthesis-scale metric lines.

    Layout: N_models rows × N_datasets columns.  Each panel plots `metric`
    (default ROC-AUC) vs. synthesis scale, one line per synthesizer method
    (orange lightness ramp, IQR band), with Real (green dashed) and FHE 8-bit
    (blue dotted) horizontal reference bands.

    Multipanel counterpart of plot_synth_scale_lines.  Two differences: synth
    method colours are computed once from the global method set (stable shade in
    every panel), and the single-panel's right-side inline labels are replaced by
    a shared bottom legend (inline labels overlap badly in a dense grid).

    Pass a df that retains all synth_scale values (do not pre-filter to scale=100).
    Saved as: synth_scale_lines_{metric}_multipanel.{fmt}
    """
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    fig_cfg    = cfg["figures"]
    modes_cfg  = cfg.get("modes", {})

    _apply_publication_style(cfg)

    if metric not in df.columns:
        return

    data = df.dropna(subset=[metric])
    if data.empty:
        return

    datasets      = sorted(data["dataset"].dropna().unique())
    models_sorted = sorted(data["model"].dropna().unique())
    if not datasets or not models_sorted:
        return

    # Global synth-method set + distinct per-method colours from config (same colour
    # a method has in the violins, identical in every panel).
    synth_all     = data[data["synth_scale"].notna()]
    synth_methods = sorted(m for m in synth_all["mode"].unique() if m not in ("standard", "fhe"))

    method_colors = {m: _mode_color(cfg, m) for m in synth_methods}
    real_color = _mode_color(cfg, "standard")
    fhe_color  = _mode_color(cfg, "fhe_8")

    synth_scales = sorted(synth_all["synth_scale"].dropna().unique().astype(int))

    n_row    = len(models_sorted)
    n_col    = len(datasets)
    tick_fs  = 7
    label_fs = 8

    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(_IEEE_FULL_WIDTH_IN, 5.2),
        sharex=True,
        sharey=False,   # independent y-scale keeps small synth-scale trends visible
        squeeze=False,
    )

    for row_idx, model in enumerate(models_sorted):
        for col_idx, dataset in enumerate(datasets):
            ax     = axes[row_idx, col_idx]
            subset = data[(data["dataset"] == dataset) & (data["model"] == model)]

            synth_df = subset[subset["synth_scale"].notna()]
            for method in synth_methods:
                m_df = synth_df[synth_df["mode"] == method]
                if m_df.empty:
                    continue
                grp = m_df.groupby("synth_scale")[metric]
                agg = pd.DataFrame({
                    "median": grp.median(),
                    "q25": grp.quantile(0.25),
                    "q75": grp.quantile(0.75),
                }).reset_index().sort_values("synth_scale")
                if agg.empty:
                    continue
                color = method_colors[method]
                ax.plot(agg["synth_scale"], agg["median"], color=color, marker="o",
                        linewidth=1.3, markersize=4, zorder=3)
                ax.fill_between(agg["synth_scale"], agg["q25"], agg["q75"],
                                color=color, alpha=0.15, zorder=1)

            std_df = subset[subset["mode"] == "standard"]
            if not std_df.empty:
                ax.axhline(std_df[metric].median(), color=real_color,
                           linewidth=1.3, linestyle="--", zorder=5)
                ax.axhspan(std_df[metric].quantile(0.25), std_df[metric].quantile(0.75),
                           color=real_color, alpha=0.10, zorder=0)

            fhe_df = subset[(subset["mode"] == "fhe") & (subset["n_bits"] == 8)]
            if not fhe_df.empty:
                ax.axhline(fhe_df[metric].median(), color=fhe_color,
                           linewidth=1.3, linestyle=":", zorder=5)
                ax.axhspan(fhe_df[metric].quantile(0.25), fhe_df[metric].quantile(0.75),
                           color=fhe_color, alpha=0.10, zorder=0)

            if synth_scales:
                # Show at most 3 evenly-spaced tick labels (first, middle, last) so
                # the narrow IEEE-width panels don't crowd with "%" labels.
                if len(synth_scales) <= 3:
                    label_scales = set(synth_scales)
                else:
                    label_scales = {
                        synth_scales[0],
                        synth_scales[len(synth_scales) // 2],
                        synth_scales[-1],
                    }
                ax.set_xticks(synth_scales)
                ax.set_xticklabels(
                    [f"{s}%" if s in label_scales else "" for s in synth_scales]
                )
            ax.tick_params(labelsize=tick_fs, length=3, pad=2)
            sns.despine(ax=ax)

            if row_idx == 0:
                ax.set_title(_fmt_dataset(dataset), fontsize=label_fs, pad=4)
            if row_idx == n_row - 1:
                ax.set_xlabel("Synthesis Scale (%)", fontsize=label_fs, labelpad=3)
            if col_idx == 0:
                ylabel = "ROC-AUC" if metric == "roc_auc" else format_metric_name(metric)
                ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=3)
                # Model row label — far left, rotated, since (a)/(b) occupies the corner
                ax.text(-0.48, 0.5, _abbrev_model(model),
                        transform=ax.transAxes, rotation=90,
                        fontsize=label_fs, fontweight="bold",
                        va="center", ha="center")

            _add_panel_label(ax, row_idx * n_col + col_idx, fontsize=label_fs)

    # Shared bottom legend — long method labels wrap onto two rows (ncol=4)
    legend_fs = 7
    handles = [
        Line2D([0], [0], color=method_colors[m], marker="o",
               linewidth=1.3, markersize=4,
               label=modes_cfg.get(m, {}).get("label", m))
        for m in synth_methods
    ]
    handles.append(Line2D([0], [0], color=real_color, linewidth=1.3,
                          linestyle="--", label="Real"))
    handles.append(Line2D([0], [0], color=fhe_color, linewidth=1.3,
                          linestyle=":", label="FHE 8-bit"))
    fig.legend(
        handles=handles,
        fontsize=legend_fs,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    fmt       = fig_cfg["format"]
    save_path = Path(save_dir) / f"synth_scale_lines_{metric}_multipanel.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close()


def plot_violinplot_multipanel(
    df, metric="roc_auc", save_dir=FIGURES_DIR, cfg=None,
    viz_cfg_path="config/visualization.yaml",
):
    """
    IEEE full-width violin multipanel — one file per dataset.

    For each dataset a separate figure is written: a vertical column of violin panels
    (one per model) that share a single y-axis label. Each panel plots `metric`
    (default ROC-AUC) bootstrap distributions across all modes (Real, synthesizers,
    FHE bit-widths) on the x-axis, coloured by group with vertical group separators.
    Violins only (no strip dots).

    The mode axis carries short codes (Real, ARF, BN, CTGAN, GC, NF, FHE-2…FHE-12)
    since the full names overlap at IEEE width. Mode colours/order are computed once
    from the global mode set, so every dataset file is directly comparable.

    Canonical scale-100 view: synthesizer rows are filtered to synth_scale == 100 so
    each synthesizer contributes a single violin (matches how the single-panel violins
    are generated). Saved as: violinplot_{metric}_multipanel__{dataset}.{fmt}
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    violin_cfg = cfg["violinplot"]
    fig_cfg    = cfg["figures"]

    _apply_publication_style(cfg)

    if metric not in df.columns:
        return

    # Canonical scale-100 view: one violin per synthesizer (drop 150/300 variants).
    df = df[df["synth_scale"].isna() | (df["synth_scale"] == 100)].copy()
    df["mode_key"] = df.apply(lambda r: _raw_mode_key(r["mode"], r["n_bits"]), axis=1)

    data = df.dropna(subset=[metric])
    if data.empty:
        return

    datasets      = sorted(data["dataset"].dropna().unique())
    models_sorted = sorted(data["model"].dropna().unique())
    if not datasets or not models_sorted:
        return

    # Global mode display — identical colours/order in every file/panel.
    all_keys    = data["mode_key"].unique()
    sorted_keys = _sort_raw_keys(all_keys)
    label_map, color_map, label_group_map = _build_mode_display(cfg, all_keys)
    order = [label_map[k] for k in sorted_keys]                  # full names (legend)
    codes = [_mode_short_code(k) for k in sorted_keys]           # short x-tick codes
    data = data.copy()
    data["mode_label"] = data["mode_key"].map(label_map)

    n_model   = len(models_sorted)
    n_mode    = len(order)
    tick_fs   = 7
    label_fs  = 8
    fmt       = fig_cfg["format"]
    ylabel    = "ROC-AUC" if metric == "roc_auc" else format_metric_name(metric)

    for dataset in datasets:
        # Taller panels than the cramped first cut; extra top/bottom room for the
        # suptitle and the horizontal x-codes.
        fig_h = n_model * 1.9 + 0.9
        fig, axes = plt.subplots(
            n_model, 1,
            figsize=(_IEEE_FULL_WIDTH_IN, fig_h),
            gridspec_kw={"hspace": 0.18},
            squeeze=False,
        )
        # Reserve margins in absolute inches (predictable without a local render):
        # left = y-ticks + supylabel, top = suptitle, bottom = horizontal x-codes.
        fig.subplots_adjust(
            left=0.85 / _IEEE_FULL_WIDTH_IN,
            right=0.99,
            top=1 - 0.5 / fig_h,     # room for suptitle + top panel's (a) tag
            bottom=0.45 / fig_h,     # room for horizontal x-codes
        )

        for mi, model in enumerate(models_sorted):
            ax     = axes[mi, 0]
            subset = data[(data["dataset"] == dataset) & (data["model"] == model)]

            if not subset.empty:
                _draw_violinplot_panel(ax, subset, metric, order, color_map, violin_cfg,
                                       show_strip=False)
                _add_group_separators(ax, order, label_group_map, cfg, show_labels=False)

            ax.set_xlim(-0.5, n_mode - 0.5)
            ax.set_xlabel("")
            ax.set_ylabel("")          # shared y-label added once per figure below
            ax.tick_params(labelsize=tick_fs, length=3, pad=2)

            # Panel letter + model tag, e.g. "(a) LR".
            _add_panel_label(ax, mi, fontsize=label_fs, x=-0.045,
                             suffix=_abbrev_model(model))

            # Short mode codes (horizontal) only on the bottom panel.
            if mi == n_model - 1:
                ax.set_xticks(range(n_mode))
                ax.set_xticklabels(codes, rotation=0, ha="center", fontsize=tick_fs)
            else:
                ax.tick_params(labelbottom=False)

        fig.suptitle(_fmt_dataset(dataset), fontsize=label_fs + 1, fontweight="bold")
        fig.supylabel(ylabel, fontsize=label_fs)   # single shared y-axis label

        save_path = Path(save_dir) / f"violinplot_{metric}_multipanel__{dataset}.{fmt}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=fmt, bbox_inches="tight")
        plt.close(fig)


# ===========================================================
# ROC CURVES (FROM RAW PREDICTIONS)
# ===========================================================
#
# Unlike every figure above, these read the raw per-sample scores in
# results/predictions/*.json (y_true, y_proba) rather than pre-computed metric
# JSONs, so they can draw true ROC curves. The 95% true-positive-rate band is a
# vertical-averaging bootstrap that reuses the project's resample scheme
# (np.random.default_rng(seed).integers(0, N, N), see src/bootstrap_utils.py):
# it draws the same resamples that produced the reported ROC-AUC CIs, so the band
# is the curve-level analog of those intervals.

PREDICTIONS_DIR = Path("results/predictions")

_ROC_FPR_GRID = np.linspace(0.0, 1.0, 201)


def load_predictions(predictions_dir=PREDICTIONS_DIR, split="test"):
    """
    Load raw per-sample predictions into preds[dataset][model][raw_mode] = (y_true, y_proba).

    raw_mode is the filename's leading "__" segment kept verbatim ("standard",
    "arf_100", "fhe_8", ...), so synthesis scales and FHE bit-widths stay distinct.
    Files without probabilities (y_proba is null) are skipped — ROC needs scores.
    """
    preds: dict = {}
    for path in Path(predictions_dir).glob(f"*__{split}__predictions.json"):
        parts = path.stem.split("__")
        if len(parts) < 4 or parts[3] != split:
            continue
        raw_mode, model, dataset = parts[0], parts[1], parts[2]

        with open(path) as f:
            data = json.load(f)
        y_proba = data.get("y_proba")
        y_true = data.get("y_true")
        if y_proba is None or y_true is None:
            continue

        (preds.setdefault(dataset, {})
              .setdefault(model, {})[raw_mode]) = (
            np.asarray(y_true), np.asarray(y_proba, dtype=float)
        )
    return preds


def _bootstrap_settings(bootstrap_cfg_path="config/bootstrap.yaml"):
    """Return (n, seed) from bootstrap.yaml, defaulting to the project convention."""
    try:
        cfg = load_config(bootstrap_cfg_path)
        return int(cfg.get("n", 1000)), int(cfg.get("seed", 42))
    except Exception:
        return 1000, 42


def _roc_from_pred(y_true, y_proba):
    """(fpr, tpr, auc) for one prediction set. Returns None if AUC is undefined."""
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return fpr, tpr, auc


def _bootstrap_tpr_band(y_true, y_proba, fpr_grid=_ROC_FPR_GRID, n=1000, seed=42,
                        lo_pct=2.5, hi_pct=97.5):
    """
    Vertical-averaging bootstrap band: (tpr_lo, tpr_hi) on fpr_grid, or None.

    Reproduces the reported ROC-AUC resamples (same rng scheme + seed). Each
    resample's ROC is interpolated onto the shared FPR grid; resamples that lose a
    class (ROC undefined) are dropped. The band is the lo/hi percentile of TPR at
    each FPR grid point.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    n_samples = len(y_true)
    rng = np.random.default_rng(seed)

    tprs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, y_proba[idx])
        interp = np.interp(fpr_grid, fpr_b, tpr_b)
        interp[0] = 0.0
        tprs.append(interp)

    if len(tprs) < 2:
        return None
    tprs = np.vstack(tprs)
    return np.percentile(tprs, lo_pct, axis=0), np.percentile(tprs, hi_pct, axis=0)


def _fmt_auc(auc):
    """Compact AUC string, e.g. 0.938 -> '.94', 1.0 -> '1.00'."""
    s = f"{auc:.2f}"
    return s[1:] if s.startswith("0") else s


def _draw_roc_panel(ax, curves, fpr_grid, band_labels, annotate_labels,
                    n_boot, seed, auc_fontsize=6):
    """
    Render one ROC panel.

    curves : ordered list of dicts {label, short, color, linestyle, y_true, y_proba}.
    band_labels     : labels that get a shaded 95% TPR bootstrap band.
    annotate_labels : labels whose AUC is printed in the lower-right block.
    """
    # Chance diagonal first so curves sit on top.
    ax.plot([0, 1], [0, 1], color="#9a9a9a", linestyle=(0, (1, 1)),
            linewidth=0.8, zorder=1)

    annotations = []  # (short, auc, color) in curve order
    for c in curves:
        roc = _roc_from_pred(c["y_true"], c["y_proba"])
        if roc is None:
            continue
        fpr, tpr, auc = roc

        if c["label"] in band_labels:
            band = _bootstrap_tpr_band(c["y_true"], c["y_proba"], fpr_grid,
                                       n=n_boot, seed=seed)
            if band is not None:
                lo, hi = band
                ax.fill_between(fpr_grid, lo, hi, color=c["color"],
                                alpha=0.15, linewidth=0, zorder=2)

        ax.plot(fpr, tpr, color=c["color"], linestyle=c["linestyle"],
                linewidth=1.1, zorder=3, solid_capstyle="round")

        if c["label"] in annotate_labels:
            annotations.append((c["short"], auc, c["color"]))

    # AUC block in the lower-right whitespace (below the diagonal), stacked so the
    # first curve sits at the top.
    n_ann = len(annotations)
    for i, (short, auc, color) in enumerate(annotations):
        y = 0.05 + (n_ann - 1 - i) * 0.105
        ax.text(0.96, y, f"{short} {_fmt_auc(auc)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=auc_fontsize, color=color, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def _roc_grid_figure(preds, cfg, *, panel_curves, band_labels, annotate_labels,
                     legend_handles, save_name, save_dir=FIGURES_DIR, legend_ncol=None):
    """
    Shared 3-model x N-dataset ROC multipanel scaffold.

    panel_curves(preds, dataset, model, cfg) -> ordered list of curve dicts for a panel.
    legend_handles : list of Line2D for the shared bottom legend.
    legend_ncol    : columns in the shared legend (defaults to a single row, capped
                     at 6 so a long precision legend wraps instead of overflowing).
    Mirrors the layout/margins of plot_synth_scale_lines_multipanel.
    """
    import math
    _apply_publication_style(cfg)

    # Canonical dataset order (known first, any extras appended), models alphabetical.
    datasets = [d for d in _DATASET_LABELS if d in preds]
    datasets += sorted(d for d in preds if d not in datasets)
    models = sorted({m for d in datasets for m in preds.get(d, {})})
    if not datasets or not models:
        return

    n_boot, seed = _bootstrap_settings()
    n_row, n_col = len(models), len(datasets)
    tick_fs, label_fs = 7, 8

    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(_IEEE_FULL_WIDTH_IN, 5.2),
        squeeze=False,
    )

    for r, model in enumerate(models):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            curves = panel_curves(preds, dataset, model, cfg)
            if curves:
                _draw_roc_panel(ax, curves, _ROC_FPR_GRID, band_labels,
                                annotate_labels, n_boot, seed)

            ax.tick_params(labelsize=tick_fs, length=3, pad=2,
                           labelbottom=(r == n_row - 1), labelleft=(c == 0))
            sns.despine(ax=ax)

            if r == 0:
                ax.set_title(_fmt_dataset(dataset), fontsize=label_fs, pad=4)
            if r == n_row - 1:
                ax.set_xlabel("False Positive Rate", fontsize=label_fs, labelpad=3)
            if c == 0:
                ax.set_ylabel("True Positive Rate", fontsize=label_fs, labelpad=3)
                # Model row label — far left, rotated (the (a)/(b) tag takes the corner).
                ax.text(-0.42, 0.5, _abbrev_model(model),
                        transform=ax.transAxes, rotation=90,
                        fontsize=label_fs, fontweight="bold",
                        va="center", ha="center")

            _add_panel_label(ax, r * n_col + c, fontsize=label_fs)

    if legend_ncol is None:
        legend_ncol = min(len(legend_handles), 6)
    legend_rows = math.ceil(len(legend_handles) / legend_ncol)
    bottom_frac = 0.06 + 0.035 * legend_rows

    fig.legend(
        handles=legend_handles,
        fontsize=7,
        ncol=legend_ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    plt.tight_layout(rect=[0, bottom_frac, 1, 1])

    fmt = cfg["figures"]["format"]
    save_path = Path(save_dir) / f"{save_name}.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def plot_roc_primary_multipanel(preds, save_dir=FIGURES_DIR, cfg=None,
                                viz_cfg_path="config/visualization.yaml"):
    """
    Primary-RQ ROC figure: Real vs best synthetic generator (ARF, scale 100) vs
    FHE 8-bit, one panel per (model, dataset), each curve with a 95% TPR band.
    Saved as: roc_curves_primary_multipanel.{fmt}
    """
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    real_color = _mode_color(cfg, "standard")
    arf_color  = _mode_color(cfg, "arf")
    fhe_color  = _mode_color(cfg, "fhe_8")

    # (raw_mode, label, short, color, linestyle). Distinct line styles keep the
    # trio separable in greyscale on top of the colorblind-safe hues.
    spec = [
        ("standard", "Real",      "R", real_color, "-"),
        ("arf_100",  "ARF",       "A", arf_color,  (0, (5, 1.5))),
        ("fhe_8",    "FHE 8-bit", "F", fhe_color,  (0, (3, 1, 1, 1))),
    ]

    def panel_curves(preds, dataset, model, cfg):
        avail = preds.get(dataset, {}).get(model, {})
        curves = []
        for raw, label, short, color, ls in spec:
            if raw in avail:
                yt, yp = avail[raw]
                curves.append({"label": label, "short": short, "color": color,
                               "linestyle": ls, "y_true": yt, "y_proba": yp})
        return curves

    labels = {label for _, label, *_ in spec}
    legend_handles = [
        Line2D([0], [0], color=real_color, linestyle="-",
               linewidth=1.3, label="Real"),
        Line2D([0], [0], color=arf_color, linestyle=(0, (5, 1.5)),
               linewidth=1.3, label="ARF (best generator)"),
        Line2D([0], [0], color=fhe_color, linestyle=(0, (3, 1, 1, 1)),
               linewidth=1.3, label="FHE 8-bit"),
        Line2D([0], [0], color="#9a9a9a", linestyle=(0, (1, 1)),
               linewidth=1.0, label="Chance"),
    ]

    _roc_grid_figure(
        preds, cfg,
        panel_curves=panel_curves,
        band_labels=labels,
        annotate_labels=labels,
        legend_handles=legend_handles,
        save_name="roc_curves_primary_multipanel",
        save_dir=save_dir,
    )


def plot_roc_fhe_precision_multipanel(preds, save_dir=FIGURES_DIR, cfg=None,
                                      viz_cfg_path="config/visualization.yaml"):
    """
    RQ2 ROC figure: Real baseline plus FHE at every bit-width (2..12, shaded
    light->dark by precision), one panel per (model, dataset). Only the Real
    reference carries a 95% TPR band — an FHE curve entering that band is
    statistically indistinguishable from the baseline, while seven overlapping
    bands would be unreadable.
    Saved as: roc_curves_fhe_precision_multipanel.{fmt}
    """
    from matplotlib.lines import Line2D

    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    real_color = _mode_color(cfg, "standard")

    # FHE bit-widths present anywhere, ascending, shaded via the existing FHE ramp.
    fhe_bits = sorted({
        int(raw.split("_")[1])
        for d in preds.values() for m in d.values() for raw in m
        if raw.startswith("fhe_")
    })
    fhe_gcfg = cfg.get("groups", {}).get("fhe", {})
    base_color = fhe_gcfg.get("base_color", "#8c564b")
    l_range = fhe_gcfg.get("lightness_range", [0.72, 0.32])
    bit_colors = dict(zip(fhe_bits, _lightness_ramp(base_color, len(fhe_bits), l_range)))

    def panel_curves(preds, dataset, model, cfg):
        avail = preds.get(dataset, {}).get(model, {})
        curves = []
        if "standard" in avail:
            yt, yp = avail["standard"]
            curves.append({"label": "Real", "short": "R", "color": real_color,
                           "linestyle": "-", "y_true": yt, "y_proba": yp})
        for nb in fhe_bits:
            raw = f"fhe_{nb}"
            if raw in avail:
                yt, yp = avail[raw]
                curves.append({"label": f"FHE {nb}-bit", "short": f"{nb}b",
                               "color": bit_colors[nb], "linestyle": "-",
                               "y_true": yt, "y_proba": yp})
        return curves

    # Annotate only the anchors (Real + lowest/highest precision) to avoid a
    # seven-line AUC block in a narrow panel.
    annotate = {"Real"}
    if fhe_bits:
        annotate |= {f"FHE {fhe_bits[0]}-bit", f"FHE {fhe_bits[-1]}-bit"}

    legend_handles = [
        Line2D([0], [0], color=real_color, linestyle="-", linewidth=1.3, label="Real"),
    ]
    legend_handles += [
        Line2D([0], [0], color=bit_colors[nb], linestyle="-", linewidth=1.3,
               label=f"FHE {nb}-bit")
        for nb in fhe_bits
    ]
    legend_handles.append(
        Line2D([0], [0], color="#9a9a9a", linestyle=(0, (1, 1)),
               linewidth=1.0, label="Chance")
    )

    _roc_grid_figure(
        preds, cfg,
        panel_curves=panel_curves,
        band_labels={"Real"},
        annotate_labels=annotate,
        legend_handles=legend_handles,
        save_name="roc_curves_fhe_precision_multipanel",
        save_dir=save_dir,
        legend_ncol=4,
    )


# ===========================================================
# MAIN ENTRYPOINT
# ===========================================================

# ===========================================================
# RADAR OVERVIEW (PER-MODE UTILITY-VS-COST SIGNATURES)
# ===========================================================
#
# One polar panel per mode (Real / synthesizers at scale=100 / FHE at a chosen
# bit-width), each aggregating over all datasets x models. The 10 axes split into
# two contiguous half-circles so the grouping the reader must make is done for
# them by geometry + colour:
#   - Performance (top half)   : native [0,1] metrics, mapped to an absolute band
#                                [0.5, 1.0] -> [centre, rim]. Outward = higher.
#   - Resource cost (bottom)   : log10 + min-max across the shown modes, INVERTED,
#                                so the cheapest shown mode reaches the rim.
# Outward therefore means "better" on every axis, and the polygon's area is a
# fair visual summary. Exact values live in the manuscript tables; the radar's
# job is the shape/signature comparison across modes.
#
# Colour does exactly one job here: group identity (performance vs resource),
# carried redundantly by a wedge wash + coloured spokes (never by the axis text).
# The polygon is a neutral slate, identical on every panel — the panel title names
# the mode, so the polygon spends no colour and cannot collide with the group hues.

# (column, short code, group) ordered by increasing display angle so the 5
# performance axes fill the top semicircle (ROC-AUC at top centre) and the 4
# resource axes fill the bottom semicircle, with no axis on the horizontal
# group boundary.
_RADAR_AXES = [
    ("accuracy",            "Acc",     "perf"),      # 18
    ("f1",                  "F1",      "perf"),      # 54
    ("roc_auc",             "AUC",     "perf"),      # 90  (top centre)
    ("precision",           "Prec",    "perf"),      # 126
    ("recall",              "Rec",     "perf"),      # 162
    ("mem_train_peak",      "Tr mem",  "resource"),  # 202.5
    ("mem_inf_peak",        "Inf mem", "resource"),  # 247.5
    ("inf_time_per_sample", "Inf t",   "resource"),  # 292.5
    ("train_time",          "Train t", "resource"),  # 337.5
]

# All 9 axes evenly spaced (40 deg apart). ROC-AUC sits at the top (90 deg); the
# 5 performance axes run 10..170 deg (top) and the 4 resource axes 210..330 deg
# (bottom), so the two groups stay contiguous while the spacing stays uniform.
_RADAR_ANGLES_DEG = [10, 50, 90, 130, 170, 210, 250, 290, 330]

_RADAR_DEFAULTS = {
    "perf_color": "#0072b2",
    "resource_color": "#d55e00",
    "poly_color": "#2b2b2b",
    "baseline_color": "#8a8a8a",
    "grid_color": "#cfcfcf",
    "perf_band": (0.5, 1.0),
    "perf_norm": "minmax",     # "minmax" (across shown modes) | "absolute" (fixed band)
    "fhe_n_bits": 8,
    "primary_synth": ["arf", "bayesian_network", "ctgan", "gaussian_copula", "nflow"],
}


def _radar_cfg(cfg):
    """Radar settings with hardcoded fallbacks, overlaid by any `radar:` config."""
    d = dict(_RADAR_DEFAULTS)
    d.update(cfg.get("radar", {}) or {})
    return d


def _radar_select_modes(df, rcfg):
    """
    Ordered (raw_key, sub_df) for the primary modes present in ``df``:
    Real -> synthesizers at synth_scale=100 (alphabetical) -> FHE at the chosen
    bit-width. Mirrors the canonical _sort_raw_keys ordering. Modes with no rows
    are skipped so the figure degrades gracefully on a partial results set.
    """
    out = []
    real = df[df["mode"] == "standard"]
    if not real.empty:
        out.append(("standard", real))

    for m in sorted(rcfg["primary_synth"]):
        sub = df[(df["mode"] == m) & (df["synth_scale"] == 100)]
        if not sub.empty:
            out.append((m, sub))

    nb = rcfg["fhe_n_bits"]
    fhe = df[(df["mode"] == "fhe") & (df["n_bits"] == nb)]
    if not fhe.empty:
        out.append((f"fhe_{int(nb)}", fhe))

    return out


def _radar_aggregate(mode_subs):
    """
    For each mode, aggregate to one value per axis.

    Each (dataset, model) pair is one "cell"; within a cell, metric columns are
    averaged over bootstrap iterations and resource columns are constant. The mode
    value is the mean over cells (equal weight per cell). Per-cell arrays are kept
    for the IQR spread band.
    """
    cols = [c for c, _, _ in _RADAR_AXES]
    means, cells = {}, {}
    for key, sub in mode_subs:
        cell_means = sub.groupby(["dataset", "model"])[cols].mean()
        cells[key] = {c: cell_means[c].to_numpy(dtype=float) for c in cols}
        means[key] = {c: float(cell_means[c].mean()) for c in cols}
    return means, cells


def _radar_axis_transforms(means, rcfg):
    """
    Build one normaliser per axis (col -> callable mapping a value/array to [0,1]).

    Resource axes: log10 then min-max across the shown modes, inverted (lower cost
    -> larger radius). Performance axes follow ``perf_norm``:
      - "absolute" (default): mapped through the fixed ``perf_band`` (e.g. 0.5->1.0),
        so radius reads as true metric magnitude and is stable regardless of which
        modes are shown.
      - "minmax": per-axis min-max across the shown modes (higher -> larger radius),
        which maximises the visible contrast between modes at the cost of absolute
        meaning (best shown mode always reaches the rim).
    The across-mode min/max come from the mode means, so per-cell values (spread
    band) reuse the same scale and may clip at the ends, which is fine.
    """
    band_lo, band_hi = rcfg["perf_band"]
    perf_norm = str(rcfg.get("perf_norm", "absolute")).lower()

    def perf_absolute(v):
        v = np.asarray(v, dtype=float)
        return np.clip((v - band_lo) / (band_hi - band_lo), 0.0, 1.0)

    def _minmax_across_modes(col, invert, use_log):
        """A min-max normaliser over the shown modes' means for one column."""
        vals = np.array([means[k][col] for k in means], dtype=float)
        vals = vals[np.isfinite(vals) & ((vals > 0) if use_log else True)]
        if vals.size < 2:
            return lambda v: np.full(np.shape(v), 0.5)
        ref = np.log10(vals) if use_log else vals
        lo, hi = float(ref.min()), float(ref.max())

        def f(v, lo=lo, hi=hi, invert=invert, use_log=use_log):
            v = np.asarray(v, dtype=float)
            if use_log:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = np.log10(np.where(v > 0, v, np.nan))
            else:
                x = v
            if hi <= lo:
                return np.full(np.shape(v), 0.5)
            norm = (hi - x) / (hi - lo) if invert else (x - lo) / (hi - lo)
            return np.clip(norm, 0.0, 1.0)

        return f

    transforms = {}
    for col, _short, group in _RADAR_AXES:
        if group == "perf":
            if perf_norm == "minmax":
                transforms[col] = _minmax_across_modes(col, invert=False, use_log=False)
            else:
                transforms[col] = perf_absolute
        else:
            transforms[col] = _minmax_across_modes(col, invert=True, use_log=True)
    return transforms


def _draw_radar_panel(ax, values, baseline, spread, rcfg, angles):
    """
    Draw one mode's radar into a polar axes.

    values   : normalised [0,1] per axis (len == len(angles))
    baseline : normalised Real reference per axis, or None (Real's own panel)
    spread   : (lo_list, hi_list) normalised IQR per axis, or None
    """
    perf_c = rcfg["perf_color"]
    res_c = rcfg["resource_color"]
    poly_c = rcfg["poly_color"]
    base_c = rcfg["baseline_color"]
    grid_c = rcfg["grid_color"]

    ax.grid(False)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("none")

    # group wedge washes spanning each group's actual angular extent (padded by
    # half a step on each side), so the wash follows the group even though the
    # axes are spaced evenly around the whole circle. Drawn as explicit wedges
    # (centre -> rim -> arc -> rim -> centre) for version-stable rendering.
    half_step = np.pi / len(angles)   # = (2*pi / n) / 2

    def _wash(a0, a1, color):
        t = np.linspace(a0, a1, 160)
        ax.fill(np.concatenate([[a0], t, [a1]]),
                np.concatenate([[0.0], np.ones_like(t), [0.0]]),
                color=color, alpha=0.06, linewidth=0, zorder=0)

    perf_ang = [a for a, (_c, _s, g) in zip(angles, _RADAR_AXES) if g == "perf"]
    res_ang = [a for a, (_c, _s, g) in zip(angles, _RADAR_AXES) if g == "resource"]
    if perf_ang:
        _wash(perf_ang[0] - half_step, perf_ang[-1] + half_step, perf_c)
    if res_ang:
        _wash(res_ang[0] - half_step, res_ang[-1] + half_step, res_c)

    # concentric hairline gridlines
    circ = np.linspace(0, 2 * np.pi, 240)
    for r in (0.25, 0.5, 0.75, 1.0):
        ax.plot(circ, np.full_like(circ, r), color=grid_c, lw=0.6, zorder=1)

    # radial spokes coloured by group
    for ang, (_col, _short, group) in zip(angles, _RADAR_AXES):
        spoke_c = perf_c if group == "perf" else res_c
        ax.plot([ang, ang], [0, 1], color=spoke_c, lw=0.7, alpha=0.45, zorder=1)

    # axis short-codes, anchored outward from the rim with angle-based alignment so
    # a label never straddles the ring, crowds the panel title, or spills onto a
    # neighbour. The top label sits a little farther out to clear the title.
    for ang, (_col, short, _group) in zip(angles, _RADAR_AXES):
        cx, sy = np.cos(ang), np.sin(ang)
        ha = "left" if cx > 0.25 else "right" if cx < -0.25 else "center"
        va = "bottom" if sy > 0.25 else "top" if sy < -0.25 else "center"
        r = 1.20 if sy > 0.85 else 1.15
        ax.text(ang, r, short, ha=ha, va=va,
                fontsize=6, color="#52514e", zorder=6)

    theta = np.concatenate([angles, angles[:1]])

    # IQR spread whiskers (radial p25->p75 at each vertex)
    if spread is not None:
        lo, hi = spread
        for ang, l, h in zip(angles, lo, hi):
            if np.isfinite(l) and np.isfinite(h) and h > l:
                ax.plot([ang, ang], [l, h], color=poly_c, lw=1.0,
                        alpha=0.28, zorder=3, solid_capstyle="round")

    # Real baseline reference (dashed, no fill), drawn in the standard-mode colour
    if baseline is not None:
        b = np.concatenate([baseline, baseline[:1]])
        ax.plot(theta, b, color=base_c, lw=1.5, linestyle=(0, (4, 2)),
                zorder=3, solid_capstyle="round")

    # mode polygon (neutral slate; fill only if every vertex is finite)
    v = np.concatenate([values, values[:1]])
    if np.all(np.isfinite(values)):
        ax.fill(theta, v, color=poly_c, alpha=0.14, linewidth=0, zorder=4)
    ax.plot(theta, v, color=poly_c, lw=2.0, zorder=5,
            solid_capstyle="round", solid_joinstyle="round")
    ax.plot(angles, values, "o", color=poly_c, markersize=3.5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=6)


def plot_radar_overview_multipanel(
    df, save_dir=FIGURES_DIR, cfg=None,
    viz_cfg_path="config/visualization.yaml", show_spread=True,
):
    """
    IEEE full-width multipanel radar — one panel per mode, aggregated over all
    datasets x models, with a performance half and an inverted resource-cost half.

    Saved as: radar_overview_multipanel.{fmt}
    """
    if cfg is None:
        cfg = _load_viz_config(viz_cfg_path)

    rcfg = _radar_cfg(cfg)
    fig_cfg = cfg["figures"]
    _apply_publication_style(cfg)

    cols = [c for c, _, _ in _RADAR_AXES]
    if any(c not in df.columns for c in cols) or "dataset" not in df.columns:
        return

    mode_subs = _radar_select_modes(df, rcfg)
    if len(mode_subs) < 2:
        return

    means, cells = _radar_aggregate(mode_subs)
    transforms = _radar_axis_transforms(means, rcfg)

    norm_mean, norm_spread = {}, {}
    for key in means:
        norm_mean[key] = [float(transforms[c](means[key][c])) for c in cols]
        lo, hi = [], []
        for c in cols:
            arr = np.asarray(transforms[c](cells[key][c]), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                lo.append(float(np.percentile(arr, 25)))
                hi.append(float(np.percentile(arr, 75)))
            else:
                lo.append(np.nan)
                hi.append(np.nan)
        norm_spread[key] = (lo, hi)

    raw_keys = [k for k, _ in mode_subs]
    label_map, _color_map, _group_map = _build_mode_display(cfg, raw_keys)
    baseline_vec = norm_mean.get("standard")
    # Real (standard) is the dashed reference on every panel, not its own panel,
    # drawn in the standard-mode colour (green) for continuity with the other figures.
    rcfg["baseline_color"] = _mode_color(cfg, "standard")
    panel_keys = [k for k in raw_keys if k != "standard"]
    angles = np.deg2rad(_RADAR_ANGLES_DEG)

    nrows, ncols = 2, 3
    total_cells = nrows * ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(_IEEE_FULL_WIDTH_IN, 5.0),
        subplot_kw=dict(polar=True),
        squeeze=False,
    )
    flat = axes.flatten()
    label_fs = 8

    n_panels = min(len(panel_keys), total_cells)
    for idx in range(n_panels):
        key = panel_keys[idx]
        ax = flat[idx]
        spread = norm_spread[key] if show_spread else None
        _draw_radar_panel(ax, norm_mean[key], baseline_vec, spread, rcfg, angles)
        ax.set_title(label_map.get(key, key), fontsize=label_fs, pad=20)
        _add_panel_label(ax, idx, fontsize=label_fs, x=-0.05)

    for j in range(n_panels, total_cells):
        flat[j].set_visible(False)

    # single legend entry: the Real (standard) dashed baseline. Everything else is
    # explained in the manuscript caption (added separately).
    if baseline_vec is not None:
        from matplotlib.lines import Line2D
        real_handle = Line2D([0], [0], color=rcfg["baseline_color"], lw=1.5,
                             linestyle=(0, (4, 2)), label=label_map.get("standard", "Real"))
        fig.legend(handles=[real_handle], loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.09,
                        wspace=0.55, hspace=0.72)

    fmt = fig_cfg["format"]
    save_path = Path(save_dir) / f"radar_overview_multipanel.{fmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def _render_multipanel_figures(df_all, cfg):
    """
    Render the IEEE multipanel figures.

    FHE rows carry no synth_scale, so the two FHE panels are unaffected by scale
    filtering; the synth-scale panel needs every scale (100/150/300); and the violin
    panel applies its own canonical scale-100 filter internally. Passing the full
    df_all to all of them therefore keeps behaviour identical while avoiding a
    separate pre-filtered frame.

    The ROC-curve multipanels are also multipanel figures but draw from the raw
    per-sample scores in results/predictions rather than the bootstrap dataframe, so
    they load their own data instead of taking df_all.
    """
    plot_fhe_training_breakdown_multipanel(df_all, cfg=cfg)
    plot_fhe_complexity_cost_multipanel(df_all, cfg=cfg)
    plot_synth_scale_lines_multipanel(df_all, cfg=cfg)
    plot_violinplot_multipanel(df_all, metric="roc_auc", cfg=cfg)
    plot_radar_overview_multipanel(df_all, cfg=cfg)

    preds = load_predictions()
    if preds:
        plot_roc_primary_multipanel(preds, cfg=cfg)
        plot_roc_fhe_precision_multipanel(preds, cfg=cfg)


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
    _render_multipanel_figures(df_all, cfg)


def generate_multipanel_figures():
    """
    Regenerate only the IEEE multipanel figures.

    Skips the per-(dataset, model, metric) violin and synth-scale single-panel
    plots, so it is fast when iterating on multipanel layout/style alone.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_viz_config()

    df_all = load_simple_bootstrap()
    df_all = df_all.dropna(how="all")

    _render_multipanel_figures(df_all, cfg)
