"""Figures and tables for interactive Bayesian concept learning (BCL).

Run from the project root (needs matplotlib: see requirements-analysis.txt):

    python -m bcl.analysis --config config_bcl.yaml

Outputs written
---------------
figures/bcl/fig1_concept_learning_curves.{pdf,png}
    10-panel small-multiples grid (2x5) showing trial 1..N proportion correct (with
    shaded 95% bootstrap CI ribbon and vertical dashed guide lines at quarter or
    repetition boundaries) for the model vs. the ideal Bayesian learner on each concept.
figures/bcl/fig2_complexity_hierarchy.{pdf,png}
    Learning curves grouped by Boolean complexity (single, negation, conjunction,
    disjunction, compound) across trials 1..N and block quarters Q1..Q4 (when single-pass)
    or repetitions 1..4 (when multi-pass), illustrating the Shepard/Feldman/Goodman
    complexity ordering.
figures/bcl/fig3_model_vs_bayes.{pdf,png}
    Scatter comparison of model vs. Bayesian ideal accuracy per concept (with 95% CIs)
    and per (concept, trial) across all N trials, with diagonal and Pearson r / RMSE.
tables/bcl/learning_curves.csv
    Every trial 1..N per concept, complexity group, and overall ("all").
tables/bcl/concept_summary.csv
    Per-concept summary metrics (overall accuracy, stage 1..4 accuracy, total errors,
    trials-to-criterion, curve Pearson r & RMSE, calibration TVD, parse failure rate).
tables/bcl/concept_summary.md
    Publication-ready Markdown summary table of per-concept accuracy by block quarter
    (Q1..Q4) or repetition (Rep 1..4), overall accuracy with 95% CI, Bayesian ideal
    accuracy, and complexity hierarchy.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")    # draw to files, never to a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bcl.score import (  # noqa: E402
    COMPLEXITY_GROUPS,
    _score_params,
    complexity_groups_in,
    concepts_in,
    load_config_or_fallback,
    models_in,
    quarter_trial_ranges,
    read_rows,
    score_bundle,
    stage_column_labels,
)

WIDTH = 3.4          # single-column width in inches
FULL_WIDTH = 7.0     # double-column width in inches

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COMPLEXITY_COLORS = {
    "single": "#2a78d6",
    "negation": "#1baf7a",
    "conjunction": "#eda100",
    "disjunction": "#eb6834",
    "compound": "#6250d6",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 150,
}

CONCEPT_SUMMARY_CSV_COLUMNS = (
    "model",
    "concept",
    "complexity_group",
    "n_blocks",
    "n_trials",
    "accuracy",
    "low",
    "high",
    "bayes_accuracy",
    "rep1_accuracy",
    "rep2_accuracy",
    "rep3_accuracy",
    "rep4_accuracy",
    "bayes_rep1",
    "bayes_rep2",
    "bayes_rep3",
    "bayes_rep4",
    "total_errors",
    "errors_low",
    "errors_high",
    "bayes_total_errors",
    "trials_to_criterion",
    "ttc_low",
    "ttc_high",
    "bayes_trials_to_criterion",
    "curve_r",
    "curve_rmse",
    "calibration_tvd",
    "parse_failure_rate",
)


def color_of(models):
    """Fixed color per model, matching ``hh.analysis``."""
    if len(models) > len(SERIES):
        raise ValueError(f"the figures have colors for {len(SERIES)} models, and the results hold {len(models)}")
    return dict(zip(models, SERIES))


def complexity_color(group, index=0):
    """Fixed color for each Boolean complexity group."""
    if group in COMPLEXITY_COLORS:
        return COMPLEXITY_COLORS[group]
    return SERIES[index % len(SERIES)]


def save(figure, folder, name):
    """Write a figure as both PDF and PNG. Returns the two written paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.pdf", folder / f"{name}.png"]
    for path in paths:
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return paths


def _curve_layout(curve_rows):
    """Infer ``(max_trial, single_pass, boundaries, xticks)`` dynamically from ``curve_rows``."""
    max_trial = max((int(r["trial"]) for r in curve_rows), default=16)
    reps = sorted({int(r["repetition"]) for r in curve_rows if r.get("repetition") is not None})
    single_pass = (not reps or max(reps) == 1) and max_trial >= 4
    if single_pass:
        boundaries = [hi for _, hi in quarter_trial_ranges(max_trial)[:-1] if 1 <= hi < max_trial]
    elif len(reps) > 1:
        boundaries = [
            max(int(r["trial"]) for r in curve_rows if int(r.get("repetition", 1)) == rep)
            for rep in reps[:-1]
        ]
        boundaries = [b for b in boundaries if 1 <= b < max_trial]
    else:
        boundaries = []
    xticks = sorted({t for t in (1, *boundaries, max_trial) if 1 <= t <= max_trial})
    return max_trial, single_pass, boundaries, xticks


# ---------------------------------------------------------------------------
# Figure 1: 10-panel small-multiples concept learning curves (2x5)
# ---------------------------------------------------------------------------

def figure_concept_learning_curves(curves, models=None, concepts=None):
    """Small-multiples grid (2x5 for 10 concepts) of trial 1..N proportion correct vs. Bayesian ideal."""
    concept_curves = [r for r in curves if r["scope_type"] == "concept"]
    if models is None:
        models = list(dict.fromkeys(r["model"] for r in concept_curves))
    if concepts is None:
        concepts = list(dict.fromkeys(r["scope"] for r in concept_curves))
    colors = color_of(models)

    n_concepts = max(1, len(concepts))
    ncols = 5 if n_concepts == 10 else min(5, n_concepts)
    nrows = -(-n_concepts // ncols)
    fig_width = FULL_WIDTH if ncols >= 3 else WIDTH * ncols * 0.65
    fig_height = 0.45 + 1.35 * nrows

    by_mc = defaultdict(list)
    group_of = {}
    for r in concept_curves:
        by_mc[r["model"], r["scope"]].append(r)
        group_of.setdefault(r["scope"], r["complexity_group"])

    max_trial, single_pass, boundaries, xticks = _curve_layout(concept_curves)

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            nrows, ncols, figsize=(fig_width, fig_height),
            sharex=True, sharey=True, squeeze=False,
        )
        for ax in axes.flat[len(concepts):]:
            ax.set_visible(False)

        for idx, (ax, concept) in enumerate(zip(axes.flat, concepts)):
            for boundary in boundaries:
                ax.axvline(boundary, color=MUTED, linestyle="--", linewidth=0.55, alpha=0.55, zorder=1)
            ax.axhline(0.5, color=GRID, linestyle=":", linewidth=0.6, zorder=1)

            # Plot Bayesian ideal learner once (from the first model that has this concept)
            bayes_drawn = False
            for model in models:
                pts = sorted(by_mc.get((model, concept), []), key=lambda r: r["trial"])
                if not pts:
                    continue
                trials = [r["trial"] for r in pts]
                if not bayes_drawn:
                    bayes = [r["bayes_correct"] for r in pts]
                    ax.plot(
                        trials, bayes, color=INK, linestyle="--", linewidth=1.0,
                        label="Bayesian ideal" if idx == 0 else None, zorder=3,
                    )
                    bayes_drawn = True

                acc = [r["correct"] for r in pts]
                ax.plot(
                    trials, acc, color=colors[model], linewidth=1.15,
                    label=model if idx == 0 else None, zorder=4,
                )
                if all(r["low"] is not None and r["high"] is not None for r in pts):
                    lows = [r["low"] for r in pts]
                    highs = [r["high"] for r in pts]
                    ax.fill_between(trials, lows, highs, color=colors[model], alpha=0.18, linewidth=0, zorder=2)

            ax.set_title(concept, loc="left", fontsize=7.5, pad=2.5)
            cgroup = group_of.get(concept, "")
            if cgroup:
                ax.text(
                    0.98, 0.06, cgroup, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color=MUTED,
                )
            ax.set_xlim(1, max(2, max_trial))
            ax.set_ylim(0.0, 1.03)
            ax.set_xticks(xticks)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, axis="y", color=GRID, linewidth=0.45)
            ax.set_axisbelow(True)

        for col in range(ncols):
            visible = [ax for ax in axes[:, col] if ax.get_visible()]
            if visible:
                visible[-1].tick_params(labelbottom=True)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles, labels, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, 1.02), ncols=len(handles),
                handlelength=1.6, columnspacing=1.4,
            )
        if single_pass:
            sup_xlabel = f"Trial ({max_trial} unique objects, 1 pass \u2014 no repetitions)"
        elif boundaries:
            sup_xlabel = f"Trial (dashed lines mark repetitions at {', '.join(str(b) for b in boundaries)})"
        else:
            sup_xlabel = "Trial"
        figure.supxlabel(sup_xlabel, fontsize=8, y=0.01)
        figure.supylabel("Proportion correct", fontsize=8, x=0.01)
        figure.tight_layout(pad=0.45, rect=(0.02, 0.03, 1.0, 0.94))
    return figure


# ---------------------------------------------------------------------------
# Figure 2: Boolean complexity hierarchy (trials 1..N and quarters/repetitions 1..4)
# ---------------------------------------------------------------------------

def figure_complexity_hierarchy(curves, rep_curves, models=None):
    """Learning curves grouped by Boolean complexity across trials 1..N and stages 1..4."""
    group_curves = [r for r in curves if r["scope_type"] == "complexity_group"]
    group_reps = [r for r in rep_curves if r["scope_type"] == "complexity_group"]
    if models is None:
        models = list(dict.fromkeys(r["model"] for r in group_curves))

    present_groups = list(dict.fromkeys(r["scope"] for r in group_curves))
    ordered_groups = sorted(
        present_groups,
        key=lambda g: (COMPLEXITY_GROUPS.index(g) if g in COMPLEXITY_GROUPS else len(COMPLEXITY_GROUPS),
                       present_groups.index(g)),
    )

    by_mg_trial = defaultdict(list)
    for r in group_curves:
        by_mg_trial[r["model"], r["scope"]].append(r)
    by_mg_rep = defaultdict(list)
    for r in group_reps:
        by_mg_rep[r["model"], r["scope"]].append(r)

    nrows = max(1, len(models))

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            nrows, 2, figsize=(FULL_WIDTH, 0.45 + 2.35 * nrows),
            gridspec_kw={"width_ratios": [1.45, 1.0]}, squeeze=False,
        )

        for row_idx, model in enumerate(models):
            ax_trial, ax_rep = axes[row_idx, 0], axes[row_idx, 1]
            model_group_curves = [r for r in group_curves if r["model"] == model]
            m_max_trial, m_single_pass, m_boundaries, m_xticks = _curve_layout(
                model_group_curves or group_curves
            )

            for boundary in m_boundaries:
                ax_trial.axvline(boundary, color=MUTED, linestyle="--", linewidth=0.55, alpha=0.55, zorder=1)
            ax_trial.axhline(0.5, color=GRID, linestyle=":", linewidth=0.6, zorder=1)
            ax_rep.axhline(0.5, color=GRID, linestyle=":", linewidth=0.6, zorder=1)

            for g_idx, group in enumerate(ordered_groups):
                col = complexity_color(group, g_idx)

                # Panel A: trial-by-trial curve (1..N)
                t_pts = sorted(by_mg_trial.get((model, group), []), key=lambda r: r["trial"])
                if t_pts:
                    xs = [r["trial"] for r in t_pts]
                    ys = [r["correct"] for r in t_pts]
                    ax_trial.plot(
                        xs, ys, color=col, linewidth=1.25,
                        label=group if row_idx == 0 else None, zorder=3,
                    )
                    if all(r["low"] is not None and r["high"] is not None for r in t_pts):
                        ax_trial.fill_between(
                            xs, [r["low"] for r in t_pts], [r["high"] for r in t_pts],
                            color=col, alpha=0.13, linewidth=0, zorder=2,
                        )

                # Panel B: stage-by-stage curve (1..4), model vs. Bayesian ideal
                r_pts = sorted(by_mg_rep.get((model, group), []), key=lambda r: r["repetition"])
                if r_pts:
                    r_xs = [r["repetition"] for r in r_pts]
                    r_ys = [r["correct"] for r in r_pts]
                    r_bayes = [r["bayes_correct"] for r in r_pts]
                    ax_rep.plot(
                        r_xs, r_bayes, color=col, linestyle="--", linewidth=0.95,
                        alpha=0.65, zorder=2,
                    )
                    ax_rep.plot(
                        r_xs, r_ys, color=col, marker="o", markersize=3.8,
                        linewidth=1.35, zorder=4,
                    )
                    for r in r_pts:
                        if r["low"] is not None and r["high"] is not None:
                            ax_rep.plot(
                                [r["repetition"], r["repetition"]], [r["low"], r["high"]],
                                color=col, linewidth=0.9, alpha=0.85, zorder=3,
                            )

            title_prefix = f"{model} \u2014 " if len(models) > 1 else ""
            ax_trial.set_title(f"{title_prefix}By trial (1\u2013{m_max_trial})", loc="left")
            ax_trial.set_xlim(1, max(2, m_max_trial))
            ax_trial.set_ylim(0.0, 1.03)
            ax_trial.set_xticks(m_xticks)
            ax_trial.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax_trial.set_ylabel("Proportion correct")
            if row_idx == nrows - 1:
                ax_trial.set_xlabel("Trial")
            ax_trial.grid(True, axis="y", color=GRID, linewidth=0.45)
            ax_trial.set_axisbelow(True)

            ax_rep.set_xlim(0.75, 4.25)
            ax_rep.set_ylim(0.0, 1.03)
            ax_rep.set_xticks([1, 2, 3, 4])
            if m_single_pass:
                ax_rep.set_title(f"{title_prefix}By block quarter (model solid, Bayes dashed)", loc="left")
                ax_rep.set_xticklabels(stage_column_labels(m_max_trial, True))
                if row_idx == nrows - 1:
                    ax_rep.set_xlabel("Trial quarter (no stimulus repetitions)")
            else:
                ax_rep.set_title(f"{title_prefix}By repetition (model solid, Bayes dashed)", loc="left")
                ax_rep.set_xticklabels([
                    lbl.replace(" (", "\n(") for lbl in stage_column_labels(m_max_trial, False)
                ])
                if row_idx == nrows - 1:
                    ax_rep.set_xlabel("Stimulus block repetition")
            ax_rep.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax_rep.grid(True, axis="y", color=GRID, linewidth=0.45)
            ax_rep.set_axisbelow(True)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        style_handles = [
            Line2D([0], [0], color=INK, linewidth=1.3, label="Model"),
            Line2D([0], [0], color=INK, linestyle="--", linewidth=1.0, alpha=0.7, label="Bayesian ideal"),
        ]
        if handles:
            figure.legend(
                handles + style_handles, labels + ["Model", "Bayesian ideal"],
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                ncols=len(handles) + 2, handlelength=1.5, columnspacing=1.1,
            )
        figure.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.92))
    return figure


# ---------------------------------------------------------------------------
# Figure 3: Model vs. Bayesian ideal scatter (per concept & per concept x trial)
# ---------------------------------------------------------------------------

def _spread_positions(ys, low_bound, high_bound, min_gap):
    """Nudge sorted 1D label positions so adjacent labels stay at least ``min_gap`` apart."""
    if not ys:
        return []
    n = len(ys)
    span = high_bound - low_bound
    gap = min(min_gap, span / max(1, n))
    indexed = sorted(enumerate(ys), key=lambda item: item[1])
    placed = [max(low_bound, min(high_bound, val)) for _, val in indexed]
    for i in range(1, n):
        if placed[i] - placed[i - 1] < gap:
            placed[i] = placed[i - 1] + gap
    if placed[-1] > high_bound:
        placed[-1] = high_bound
        for i in range(n - 2, -1, -1):
            if placed[i + 1] - placed[i] < gap:
                placed[i] = max(low_bound, placed[i + 1] - gap)
    out = [0.0] * n
    for (orig_idx, _), val in zip(indexed, placed):
        out[orig_idx] = val
    return out


def figure_model_vs_bayes(curves, concept_summaries, model_summaries, models=None):
    """Scatter comparison of model vs. Bayesian ideal accuracy per concept and per trial."""
    if models is None:
        models = [m["model"] for m in model_summaries]
    m_lookup = {m["model"]: m for m in model_summaries}
    concept_curves = [r for r in curves if r["scope_type"] == "concept"]
    overall_curves = [r for r in curves if r["scope_type"] == "all"]
    nrows = max(1, len(models))

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            nrows, 2, figsize=(FULL_WIDTH, 0.45 + 2.55 * nrows), squeeze=False,
        )

        seen_groups = []
        for row_idx, model in enumerate(models):
            ax_c, ax_t = axes[row_idx, 0], axes[row_idx, 1]
            c_rows = [r for r in concept_summaries if r["model"] == model]
            ct_rows = [r for r in concept_curves if r["model"] == model]
            all_rows = sorted([r for r in overall_curves if r["model"] == model], key=lambda r: r["trial"])
            m_sum = m_lookup.get(model, {})

            c_coords = (
                [r["accuracy"] for r in c_rows]
                + [r["bayes_accuracy"] for r in c_rows]
                + [r["low"] for r in c_rows if r["low"] is not None]
            )
            ct_coords = [r["correct"] for r in ct_rows] + [r["bayes_correct"] for r in ct_rows]
            lo_c = 0.0 if (not c_coords or min(c_coords) < 0.52) else 0.5
            lo_t = 0.0 if (not ct_coords or min(ct_coords) < 0.32) else 0.3

            for ax, lo in ((ax_c, lo_c), (ax_t, lo_t)):
                ax.plot([lo, 1.0], [lo, 1.0], color=MUTED, linestyle="--", linewidth=0.85, zorder=1)
                diag_xy = lo + 0.22 * (1.0 - lo)
                ax.annotate(
                    "model = ideal", xy=(diag_xy, diag_xy),
                    xytext=(4, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=6.5, color=MUTED,
                )
                ax.set_xlim(lo, 1.03)
                ax.set_ylim(lo, 1.03)
                if lo == 0.5:
                    ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                elif lo == 0.3:
                    ticks = [0.3, 0.5, 0.7, 0.85, 1.0]
                else:
                    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.grid(True, color=GRID, linewidth=0.45)
                ax.set_axisbelow(True)

            # Left panel: 10 concept mean accuracies with 95% bootstrap CI bars & non-overlapping labels
            mid_x = lo_c + 0.55 * (1.0 - lo_c)
            left_idxs = [i for i, r in enumerate(c_rows) if r["bayes_accuracy"] > mid_x]
            right_idxs = [i for i, r in enumerate(c_rows) if r["bayes_accuracy"] <= mid_x]
            label_y = {}
            min_gap = 0.052 * (1.0 - lo_c)
            for group_idxs in (left_idxs, right_idxs):
                spread = _spread_positions(
                    [c_rows[i]["accuracy"] for i in group_idxs],
                    lo_c + 0.04 * (1.0 - lo_c), 0.99, min_gap,
                )
                for i, sy in zip(group_idxs, spread):
                    label_y[i] = sy

            for idx_c, r in enumerate(c_rows):
                cgroup = r["complexity_group"]
                if cgroup not in seen_groups:
                    seen_groups.append(cgroup)
                col = complexity_color(cgroup, seen_groups.index(cgroup))
                bx, my = r["bayes_accuracy"], r["accuracy"]
                if r["low"] is not None and r["high"] is not None:
                    ax_c.plot([bx, bx], [r["low"], r["high"]], color=col, linewidth=0.9, alpha=0.8, zorder=2)
                ax_c.scatter(
                    [bx], [my], s=24, color=col, edgecolors="white", linewidths=0.5, zorder=3, clip_on=False,
                )
                place_left = bx > mid_x
                tx = bx - 0.045 * (1.0 - lo_c) if place_left else bx + 0.045 * (1.0 - lo_c)
                ty = label_y[idx_c]
                ax_c.annotate(
                    r["concept"], xy=(bx, my), xytext=(tx, ty),
                    ha="right" if place_left else "left", va="center",
                    fontsize=6, color=INK, zorder=4,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.45, "alpha": 0.55, "shrinkA": 0, "shrinkB": 2},
                )

            c_r = m_sum.get("concept_r")
            c_rmse = m_sum.get("concept_rmse")
            c_stat = f"r = {c_r:.3f}, RMSE = {c_rmse:.3f}" if c_r is not None else f"RMSE = {c_rmse:.3f}"
            prefix = f"{model} — " if len(models) > 1 else ""
            ax_c.set_title(f"{prefix}Per concept ({len(c_rows)} concepts)", loc="left")
            ax_c.text(
                0.04, 0.94, c_stat, transform=ax_c.transAxes,
                ha="left", va="top", fontsize=7, color=INK,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
            )
            ax_c.set_ylabel("Model proportion correct")
            if row_idx == nrows - 1:
                ax_c.set_xlabel("Bayesian ideal proportion correct")

            # Right panel: all (concept, trial) points + N overall trial means
            for r in ct_rows:
                cgroup = r["complexity_group"]
                if cgroup not in seen_groups:
                    seen_groups.append(cgroup)
                col = complexity_color(cgroup, seen_groups.index(cgroup))
                ax_t.scatter(
                    [r["bayes_correct"]], [r["correct"]], s=12, color=col,
                    alpha=0.45, edgecolors="none", zorder=2, clip_on=False,
                )
            if all_rows:
                max_t = int(all_rows[-1]["trial"])
                ax_t.scatter(
                    [r["bayes_correct"] for r in all_rows],
                    [r["correct"] for r in all_rows],
                    s=20, color=INK, alpha=0.85, edgecolors="white", linewidths=0.4,
                    zorder=3, label=f"Trial mean (1\u2013{max_t})", clip_on=False,
                )

            ct_r = m_sum.get("concept_trial_r")
            ct_rmse = m_sum.get("concept_trial_rmse")
            ct_stat = (f"r = {ct_r:.3f}, RMSE = {ct_rmse:.3f}"
                       if ct_r is not None else f"RMSE = {ct_rmse:.3f}")
            ax_t.set_title(f"{prefix}Per concept × trial ({len(ct_rows)} points)", loc="left")
            ax_t.text(
                0.04, 0.94, ct_stat, transform=ax_t.transAxes,
                ha="left", va="top", fontsize=7, color=INK,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
            )
            if row_idx == nrows - 1:
                ax_t.set_xlabel("Bayesian ideal proportion correct")

        ordered_seen = sorted(
            seen_groups,
            key=lambda g: (COMPLEXITY_GROUPS.index(g) if g in COMPLEXITY_GROUPS else len(COMPLEXITY_GROUPS),
                           seen_groups.index(g)),
        )
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=complexity_color(g, idx), markeredgecolor="white", label=g)
            for idx, g in enumerate(ordered_seen)
        ]
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                   markerfacecolor=INK, markeredgecolor="white", label="Overall trial mean")
        )
        figure.legend(
            handles=legend_handles, frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 1.02), ncols=len(legend_handles),
            handletextpad=0.3, columnspacing=1.0,
        )
        figure.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.92))
    return figure


# ---------------------------------------------------------------------------
# CSV & Markdown tables
# ---------------------------------------------------------------------------

def write_csv(path, lines, fieldnames=None):
    """Write a sequence of dicts to ``path`` as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames) if fieldnames is not None else list(lines[0].keys())
    with open(path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)
    return path


def _md_pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _md_pct_ci(mean, low, high):
    if mean is None:
        return "-"
    if low is None or high is None:
        return _md_pct(mean)
    return f"{100 * mean:.1f}% [{100 * low:.1f}%, {100 * high:.1f}%]"


def _md_num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def concept_summary_markdown(concept_summaries, complexity_summaries, model_summaries, interval=0.95):
    """Clean Markdown report of per-concept accuracy by stage 1..4, overall accuracy, and Bayesian ideal."""
    level = f"{100 * interval:.0f}%"
    models = [m["model"] for m in model_summaries]
    multi_model = len(models) > 1
    lines = []

    for m_sum in model_summaries:
        model = m_sum["model"]
        c_rows = [r for r in concept_summaries if r["model"] == model]
        g_rows = [r for r in complexity_summaries if r["model"] == model]
        n_trials = int(m_sum["n_trials"])
        reps_val = m_sum.get("repetitions")
        if reps_val is None and c_rows:
            reps_val = c_rows[0].get("repetitions")
        if reps_val is None:
            reps_val = 1 if n_trials == 16 else 4
        single_pass = (int(reps_val) == 1 and n_trials >= 4)
        s1, s2, s3, s4 = stage_column_labels(n_trials, single_pass)

        if multi_model:
            lines.extend([f"## {model}", ""])

        lines.extend([
            f"| Concept | Complexity | Blocks | {s1} | {s2} | {s3} | {s4} | Overall [{level} CI] | Bayesian ideal | Errors | Trials to criterion | Curve r |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ])
        for r in c_rows:
            lines.append(
                f"| {r['concept']} | {r['complexity_group']} | {r['n_blocks']} | "
                f"{_md_pct(r['rep1_accuracy'])} | {_md_pct(r['rep2_accuracy'])} | "
                f"{_md_pct(r['rep3_accuracy'])} | {_md_pct(r['rep4_accuracy'])} | "
                f"{_md_pct_ci(r['accuracy'], r['low'], r['high'])} | "
                f"{_md_pct(r['bayes_accuracy'])} | {_md_num(r['total_errors'], 1)} | "
                f"{_md_num(r['trials_to_criterion'], 1)} | {_md_num(r['curve_r'], 3)} |"
            )

        lines.extend([
            "",
            f"| Boolean complexity | Blocks | {s1} | {s2} | {s3} | {s4} | Overall [{level} CI] | Bayesian ideal | Errors | Trials to criterion |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ])
        for r in g_rows:
            label = "**All concepts**" if r["scope_type"] == "all" else r["scope"]
            lines.append(
                f"| {label} | {r['n_blocks']} | "
                f"{_md_pct(r['rep1_accuracy'])} | {_md_pct(r['rep2_accuracy'])} | "
                f"{_md_pct(r['rep3_accuracy'])} | {_md_pct(r['rep4_accuracy'])} | "
                f"{_md_pct_ci(r['accuracy'], r['low'], r['high'])} | "
                f"{_md_pct(r['bayes_accuracy'])} | {_md_num(r['total_errors'], 1)} | "
                f"{_md_num(r['trials_to_criterion'], 1)} |"
            )

        lines.extend([
            "",
            f"Model `{model}` ({m_sum['n_blocks']} blocks across {m_sum['n_concepts']} concepts, "
            f"{n_trials} trials/block). Brackets show {level} bootstrap intervals from resampling whole blocks.",
            "",
            f"- Concept mean accuracy alignment vs. Bayesian ideal: "
            f"r = {_md_num(m_sum['concept_r'], 3)}, RMSE = {_md_num(m_sum['concept_rmse'], 3)}.",
            f"- Overall {n_trials}-trial learning curve alignment vs. Bayesian ideal: "
            f"r = {_md_num(m_sum['overall_curve_r'], 3)}, RMSE = {_md_num(m_sum['overall_curve_rmse'], 3)}.",
            f"- Concept \u00d7 trial ({m_sum['n_concepts'] * n_trials} points) alignment vs. Bayesian ideal: "
            f"r = {_md_num(m_sum['concept_trial_r'], 3)}, RMSE = {_md_num(m_sum['concept_trial_rmse'], 3)}.",
        ])
        if m_sum.get("calibration_tvd") is not None:
            lines.append(
                f"- Mean `p_yes` calibration TVD (`|p_yes - bayes_p_yes|`): "
                f"{_md_num(m_sum['calibration_tvd'], 3)}."
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

def _resolve_root_and_rows(config=None, rows=None, output_dir=None):
    """Resolve ``(config_obj_or_dict, rows_list, figures_dir, tables_dir)`` flexibly."""
    if isinstance(config, (str, Path)):
        config = load_config_or_fallback(config)

    if output_dir is not None:
        base = Path(output_dir)
        figures_dir = base / "figures" / "bcl"
        tables_dir = base / "tables" / "bcl"
    elif hasattr(config, "resolve"):
        figures_dir = config.resolve("figures/bcl")
        tables_dir = config.resolve("tables/bcl")
    else:
        base = Path(".")
        figures_dir = base / "figures" / "bcl"
        tables_dir = base / "tables" / "bcl"

    if rows is None:
        if hasattr(config, "resolve") and hasattr(config, "run"):
            results_path = config.resolve(config.run["results_file"])
        elif isinstance(config, Mapping):
            results_path = Path(output_dir or ".") / config.get("run", {}).get("results_file", "data/bcl/results.jsonl")
        else:
            results_path = Path("data/bcl/results.jsonl")
        rows = read_rows(results_path)
        if not rows:
            raise SystemExit(f"no rows in {results_path}; run python -m bcl.run first")
    elif not rows:
        raise SystemExit("no rows supplied to bcl.analysis.make_all")

    return config, rows, figures_dir, tables_dir


def make_all(config=None, *, rows=None, output_dir=None):
    """Regenerate every BCL figure and table into ``figures/bcl/`` and ``tables/bcl/``."""
    config, rows, figures_dir, tables_dir = _resolve_root_and_rows(config, rows, output_dir)
    _, _, interval, rule_order = _score_params(config)
    bundle = score_bundle(rows, config=config)

    curves = bundle["learning_curves"]
    rep_curves = bundle["repetition_curves"]
    c_summaries = bundle["concept_summary"]
    g_summaries = bundle["complexity_summary"]
    m_summaries = bundle["model_summary"]
    models = models_in(rows)
    concepts = concepts_in(rows, rule_order)

    written = []
    written += save(
        figure_concept_learning_curves(curves, models=models, concepts=concepts),
        figures_dir, "fig1_concept_learning_curves",
    )
    written += save(
        figure_complexity_hierarchy(curves, rep_curves, models=models),
        figures_dir, "fig2_complexity_hierarchy",
    )
    written += save(
        figure_model_vs_bayes(curves, c_summaries, m_summaries, models=models),
        figures_dir, "fig3_model_vs_bayes",
    )

    written.append(write_csv(tables_dir / "learning_curves.csv", curves))
    written.append(write_csv(tables_dir / "concept_summary.csv", c_summaries, fieldnames=CONCEPT_SUMMARY_CSV_COLUMNS))

    md_path = tables_dir / "concept_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(concept_summary_markdown(c_summaries, g_summaries, m_summaries, interval=interval))
    written.append(md_path)

    if hasattr(config, "save_copy"):
        written.append(config.save_copy(figures_dir / "figures"))
        written.append(config.save_copy(tables_dir / "tables"))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate BCL figures and tables from the results file.")
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    args = parser.parse_args(argv)

    config = load_config_or_fallback(args.config)
    for path in make_all(config):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
