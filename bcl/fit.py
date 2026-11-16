"""Fit cognitive parameters (alpha, beta, prior_temperature) of Rational Rules to BCL model data.

Run from the project root (needs numpy & matplotlib: see requirements-analysis.txt):

    python -m bcl.fit --config config_bcl.yaml

Model family
------------
Every model in the nested comparison shares the exact posterior over ``config.hypothesis_rules``
from ``rule_learner.py`` and maps each trial's pre-feedback posterior to a probability of
answering ``"yes"`` through up to three standard cognitive parameters:

- ``alpha`` in (0, 1): internal likelihood fidelity (LOTlib3 ``BinaryLikelihood`` ``alpha``;
  related to Goodman et al., 2008's outlier sensitivity ``b = log((1 + alpha) / (1 - alpha))``).
  Lower ``alpha`` means each labeled observation updates log-odds more gradually.
- ``beta`` in (0, inf]: Luce-choice / logistic softmax inverse response temperature over the
  posterior predictive log-odds ``logit(p_pred)``, where
  ``p_pred(x_t) = sum_h P(h | D_<t) * (alpha * h(x_t) + (1 - alpha) / 2)``.
  ``beta = inf`` is the deterministic MAP/argmax rule (``bayes_correct``); ``beta = 1`` is
  probability matching (Goodman et al., 2008).
- ``prior_temp`` in (0, inf): simplicity-prior temperature ``T_prior`` dividing log prior
  weights (``RuleSet.trajectory(..., prior_temperature=T_prior)``). Values above 1 flatten the
  simplicity bias toward single-feature rules.

Parameters are estimated by maximum likelihood on the LLM's trial-by-trial binary choices
(minimizing negative log-likelihood across all valid trials) and evaluated both in-sample
(NLL, AIC, BIC, curve r & RMSE) and out-of-sample via Leave-One-Concept-Out (LOCO)
cross-validation across the 10 target concepts.

Outputs written
---------------
figures/bcl/fig4_fitted_rational_rules.{pdf,png}
    10-panel small-multiples learning curves (top) and 160-point concept x trial calibration
    comparison (bottom) contrasting gemma4:26b against the normative ideal, the 1-parameter
    Goodman alpha fit, and the 3-parameter (alpha, beta, T_prior) fit.
tables/bcl/parameter_fits.csv
    In-sample and Leave-One-Concept-Out cross-validated metrics for all nested models.
tables/bcl/parameter_fits.md
    Publication-ready Markdown summary of parameter estimates, AIC/BIC, and LOCO generalization.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bcl.analysis import (  # noqa: E402
    FULL_WIDTH,
    GRID,
    INK,
    MUTED,
    STYLE,
    _curve_layout,
    complexity_color,
    save,
    write_csv,
)
from bcl.config import load_config  # noqa: E402
from bcl.score import (  # noqa: E402
    COMPLEXITY_GROUPS,
    concepts_in,
    learning_curves,
    models_in,
    pearson_r,
    read_rows,
    rmse,
)

MODEL_SPECS = (
    ("normative_argmax", "Normative ideal (argmax)", 0),
    ("prob_matching", "Prob-matching ideal (beta=1)", 0),
    ("fit_alpha", "1-param: likelihood alpha", 1),
    ("fit_beta", "1-param: softmax beta", 1),
    ("fit_alpha_beta", "2-param: alpha + beta", 2),
    ("fit_alpha_prior_temp", "2-param: alpha + prior_temp", 2),
    ("fit_alpha_beta_prior_temp", "3-param: alpha + beta + prior_temp", 3),
)

FIT_CSV_COLUMNS = (
    "llm_model",
    "fit_id",
    "fit_label",
    "k_params",
    "alpha",
    "beta",
    "prior_temp",
    "n_valid_trials",
    "nll",
    "mean_nll",
    "aic",
    "bic",
    "pred_accuracy",
    "llm_accuracy",
    "concept_r",
    "concept_rmse",
    "overall_curve_r",
    "overall_curve_rmse",
    "concept_trial_r",
    "concept_trial_rmse",
    "loco_nll",
    "loco_mean_nll",
    "loco_pred_accuracy",
    "loco_concept_r",
    "loco_concept_rmse",
    "loco_overall_curve_r",
    "loco_overall_curve_rmse",
    "loco_concept_trial_r",
    "loco_concept_trial_rmse",
)


def _nelder_mead(func, x0, *, step: float = 0.2, max_iter: int = 450, tol: float = 1e-7):
    """Dependency-free Nelder-Mead simplex minimizer for 1D-3D smooth objectives."""
    x0 = np.asarray(x0, dtype=np.float64)
    dim = len(x0)
    simplex = [x0.copy()]
    for i in range(dim):
        pt = x0.copy()
        pt[i] += step
        simplex.append(pt)
    vals = [float(func(p)) for p in simplex]

    for _ in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if max(abs(vals[i] - vals[0]) for i in range(1, dim + 1)) <= tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        vr = float(func(xr))
        if vals[0] <= vr < vals[-2]:
            simplex[-1], vals[-1] = xr, vr
            continue
        if vr < vals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            ve = float(func(xe))
            if ve < vr:
                simplex[-1], vals[-1] = xe, ve
            else:
                simplex[-1], vals[-1] = xr, vr
            continue
        xc = centroid + 0.5 * (simplex[-1] - centroid)
        vc = float(func(xc))
        if vc < vals[-1]:
            simplex[-1], vals[-1] = xc, vc
        else:
            for i in range(1, dim + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                vals[i] = float(func(simplex[i]))
    return simplex[0], vals[0]


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


class _HypothesisEvaluator:
    """Fast vectorized evaluator of exact Bayesian posteriors across all blocks for one LLM."""

    def __init__(self, config, model_rows):
        self.config = config
        self.rows = list(model_rows)
        stimuli = config.stimuli
        stim_key = lambda s: tuple(s[f] for f in config.features)
        stim_to_idx = {stim_key(s): i for i, s in enumerate(stimuli)}

        hyp_rules = config.hypothesis_rules
        self.extensions = np.array(
            [[1.0 if r(s) else 0.0 for s in stimuli] for r in hyp_rules],
            dtype=np.float64,
        )
        self.log_w0 = np.array([math.log(r.weight) for r in hyp_rules], dtype=np.float64)

        n_blocks = len(self.rows)
        n_trials = max(len(r["trials"]) for r in self.rows)
        self.n_trials = n_trials
        self.stim_idx = np.zeros((n_blocks, n_trials), dtype=np.int64)
        self.truth = np.zeros((n_blocks, n_trials), dtype=np.float64)
        self.correct = np.zeros((n_blocks, n_trials), dtype=np.float64)
        self.ans_yes = np.zeros((n_blocks, n_trials), dtype=np.float64)
        self.valid = np.zeros((n_blocks, n_trials), dtype=bool)
        self.concepts = np.array([r["concept"] for r in self.rows])
        self.concept_order = concepts_in(self.rows, list(config.programs.keys()))

        for b_idx, row in enumerate(self.rows):
            for t_idx, trial in enumerate(sorted(row["trials"], key=lambda x: int(x["trial"]))):
                self.stim_idx[b_idx, t_idx] = stim_to_idx[stim_key(trial["object"])]
                self.truth[b_idx, t_idx] = 1.0 if trial["truth"] else 0.0
                self.correct[b_idx, t_idx] = 1.0 if trial["correct"] else 0.0
                if trial.get("answer") is not None:
                    self.valid[b_idx, t_idx] = True
                    self.ans_yes[b_idx, t_idx] = 1.0 if trial["answer"] else 0.0

        # Shape (B, T, H): each hypothesis's truth value on each block's trial stimulus
        self.ext_bt = self.extensions.T[self.stim_idx]
        self.agree_bt = (self.ext_bt == self.truth[:, :, None]).astype(np.float64)

    def predict_q_yes(self, alpha: float, beta: float | None, prior_temp: float) -> np.ndarray:
        """Predicted P(respond 'yes') of shape ``(n_blocks, n_trials)``."""
        alpha = min(1.0 - 1e-9, max(1e-9, float(alpha)))
        prior_temp = max(1e-6, float(prior_temp))
        p_match = 0.5 * (1.0 + alpha)
        p_mismatch = 0.5 * (1.0 - alpha)
        delta_ll = self.agree_bt * (math.log(p_match) - math.log(p_mismatch)) + math.log(p_mismatch)

        cum_ll = np.zeros_like(delta_ll)
        cum_ll[:, 1:, :] = np.cumsum(delta_ll[:, :-1, :], axis=1)
        log_post = (self.log_w0[None, None, :] / prior_temp) + cum_ll
        log_post -= np.max(log_post, axis=2, keepdims=True)
        post = np.exp(log_post)
        post /= np.sum(post, axis=2, keepdims=True)

        rule_p_yes = np.sum(post * self.ext_bt, axis=2)
        p_pred = alpha * rule_p_yes + 0.5 * (1.0 - alpha)

        if beta is None or math.isinf(beta):
            return np.where(np.abs(p_pred - 0.5) <= 1e-9, 0.5, (p_pred > 0.5).astype(np.float64))
        if abs(beta - 1.0) <= 1e-12:
            return p_pred
        clipped = np.clip(p_pred, 1e-12, 1.0 - 1e-12)
        logits = np.clip(float(beta) * np.log(clipped / (1.0 - clipped)), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def nll(self, q_yes: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Negative log-likelihood of observed binary answers on ``mask`` blocks."""
        if mask is None:
            qc = np.clip(q_yes, 1e-12, 1.0 - 1e-12)
            ll = (self.ans_yes * np.log(qc) + (1.0 - self.ans_yes) * np.log(1.0 - qc))[self.valid]
            return -float(np.sum(ll))
        qc = np.clip(q_yes[mask], 1e-12, 1.0 - 1e-12)
        ans = self.ans_yes[mask]
        val = self.valid[mask]
        ll = (ans * np.log(qc) + (1.0 - ans) * np.log(1.0 - qc))[val]
        return -float(np.sum(ll))

    def fit_spec(self, fit_id: str, mask: np.ndarray | None = None) -> tuple[float, float | None, float]:
        """Return ``(alpha, beta, prior_temp)`` for ``fit_id`` fit on ``mask`` blocks."""
        base_alpha = float(self.config.alpha)
        base_temp = float(self.config.settings["prior"]["temperature"] or 1.0)

        if fit_id == "normative_argmax":
            return base_alpha, None, base_temp
        if fit_id == "prob_matching":
            return base_alpha, 1.0, base_temp

        if fit_id == "fit_alpha":
            best_p, _ = _nelder_mead(
                lambda z: self.nll(self.predict_q_yes(_sigmoid(z[0]), 1.0, base_temp), mask),
                [1.6],
            )
            return _sigmoid(best_p[0]), 1.0, base_temp

        if fit_id == "fit_beta":
            best_p, _ = _nelder_mead(
                lambda z: self.nll(self.predict_q_yes(base_alpha, math.exp(z[0]), base_temp), mask),
                [-0.8],
            )
            return base_alpha, math.exp(max(-15.0, min(15.0, best_p[0]))), base_temp

        if fit_id == "fit_alpha_beta":
            best = None
            for a_init in (0.5, 2.0, 3.5):
                for b_init in (-0.9, -0.2):
                    p, val = _nelder_mead(
                        lambda z: self.nll(
                            self.predict_q_yes(_sigmoid(z[0]), math.exp(max(-15.0, min(15.0, z[1]))), base_temp),
                            mask,
                        ),
                        [a_init, b_init],
                    )
                    if best is None or val < best[1]:
                        best = (p, val)
            return _sigmoid(best[0][0]), math.exp(max(-15.0, min(15.0, best[0][1]))), base_temp

        if fit_id == "fit_alpha_prior_temp":
            best = None
            for a_init in (1.0, 2.5):
                for t_init in (0.2, 0.9):
                    p, val = _nelder_mead(
                        lambda z: self.nll(
                            self.predict_q_yes(_sigmoid(z[0]), 1.0, math.exp(max(-10.0, min(10.0, z[1])))),
                            mask,
                        ),
                        [a_init, t_init],
                    )
                    if best is None or val < best[1]:
                        best = (p, val)
            return _sigmoid(best[0][0]), 1.0, math.exp(max(-10.0, min(10.0, best[0][1])))

        if fit_id == "fit_alpha_beta_prior_temp":
            best = None
            for a_init in (0.8, 2.5):
                for b_init in (-0.4, 0.3):
                    for t_init in (0.3, 0.9):
                        p, val = _nelder_mead(
                            lambda z: self.nll(
                                self.predict_q_yes(
                                    _sigmoid(z[0]),
                                    math.exp(max(-15.0, min(15.0, z[1]))),
                                    math.exp(max(-10.0, min(10.0, z[2]))),
                                ),
                                mask,
                            ),
                            [a_init, b_init, t_init],
                        )
                        if best is None or val < best[1]:
                            best = (p, val)
            p = best[0]
            return (
                _sigmoid(p[0]),
                math.exp(max(-15.0, min(15.0, p[1]))),
                math.exp(max(-10.0, min(10.0, p[2]))),
            )

        raise ValueError(f"unknown fit_id: {fit_id!r}")

    def summarize_predictions(self, q_yes: np.ndarray) -> dict:
        """Compute NLL and curve-alignment metrics for a ``(n_blocks, n_trials)`` prediction matrix."""
        n_valid = int(np.sum(self.valid))
        nll = self.nll(q_yes)
        pred_acc = np.where(self.truth == 1.0, q_yes, 1.0 - q_yes)

        m_overall_curve = np.mean(self.correct, axis=0).tolist()
        p_overall_curve = np.mean(pred_acc, axis=0).tolist()

        c_m, c_p = [], []
        ct_m, ct_p = [], []
        concept_curves = {}
        for cn in self.concept_order:
            cmask = self.concepts == cn
            mc = np.mean(self.correct[cmask], axis=0).tolist()
            pc = np.mean(pred_acc[cmask], axis=0).tolist()
            concept_curves[cn] = pc
            c_m.append(sum(mc) / len(mc))
            c_p.append(sum(pc) / len(pc))
            ct_m.extend(mc)
            ct_p.extend(pc)

        return {
            "n_valid_trials": n_valid,
            "nll": nll,
            "mean_nll": nll / n_valid if n_valid else None,
            "pred_accuracy": float(np.mean(pred_acc)),
            "llm_accuracy": float(np.mean(self.correct)),
            "concept_r": pearson_r(c_m, c_p),
            "concept_rmse": rmse(c_m, c_p),
            "overall_curve_r": pearson_r(m_overall_curve, p_overall_curve),
            "overall_curve_rmse": rmse(m_overall_curve, p_overall_curve),
            "concept_trial_r": pearson_r(ct_m, ct_p),
            "concept_trial_rmse": rmse(ct_m, ct_p),
            "concept_curves": concept_curves,
            "concept_means": dict(zip(self.concept_order, c_p)),
        }


def fit_all_models(config, rows=None) -> dict:
    """Fit all nested Rational Rules parameterizations and run Leave-One-Concept-Out CV."""
    if rows is None:
        results_path = config.resolve(config.run["results_file"])
        rows = read_rows(results_path)
    if not rows:
        raise ValueError("no results rows to fit")

    summary_rows = []
    curves_by_model = {}

    for llm in models_in(rows):
        m_rows = [r for r in rows if r["model"] == llm]
        ev = _HypothesisEvaluator(config, m_rows)
        curves_by_model[llm] = {"evaluator": ev, "fits": {}}

        for fit_id, fit_label, k_params in MODEL_SPECS:
            alpha, beta, prior_temp = ev.fit_spec(fit_id)
            q_in = ev.predict_q_yes(alpha, beta, prior_temp)
            in_stats = ev.summarize_predictions(q_in)
            n_valid = in_stats["n_valid_trials"]
            aic = 2.0 * k_params + 2.0 * in_stats["nll"]
            bic = k_params * math.log(max(1, n_valid)) + 2.0 * in_stats["nll"]

            if k_params == 0 or len(ev.concept_order) < 2:
                loco_stats = in_stats
                q_loco = q_in
            else:
                q_loco = np.zeros_like(q_in)
                for held_out in ev.concept_order:
                    train_mask = ev.concepts != held_out
                    test_mask = ev.concepts == held_out
                    a_cv, b_cv, t_cv = ev.fit_spec(fit_id, mask=train_mask)
                    q_loco[test_mask] = ev.predict_q_yes(a_cv, b_cv, t_cv)[test_mask]
                loco_stats = ev.summarize_predictions(q_loco)

            curves_by_model[llm]["fits"][fit_id] = {
                "alpha": alpha,
                "beta": beta,
                "prior_temp": prior_temp,
                "in_sample": in_stats,
                "loco": loco_stats,
            }
            summary_rows.append({
                "llm_model": llm,
                "fit_id": fit_id,
                "fit_label": fit_label,
                "k_params": k_params,
                "alpha": alpha,
                "beta": ("inf" if beta is None else beta),
                "prior_temp": prior_temp,
                "n_valid_trials": n_valid,
                "nll": in_stats["nll"],
                "mean_nll": in_stats["mean_nll"],
                "aic": aic,
                "bic": bic,
                "pred_accuracy": in_stats["pred_accuracy"],
                "llm_accuracy": in_stats["llm_accuracy"],
                "concept_r": in_stats["concept_r"],
                "concept_rmse": in_stats["concept_rmse"],
                "overall_curve_r": in_stats["overall_curve_r"],
                "overall_curve_rmse": in_stats["overall_curve_rmse"],
                "concept_trial_r": in_stats["concept_trial_r"],
                "concept_trial_rmse": in_stats["concept_trial_rmse"],
                "loco_nll": loco_stats["nll"],
                "loco_mean_nll": loco_stats["mean_nll"],
                "loco_pred_accuracy": loco_stats["pred_accuracy"],
                "loco_concept_r": loco_stats["concept_r"],
                "loco_concept_rmse": loco_stats["concept_rmse"],
                "loco_overall_curve_r": loco_stats["overall_curve_r"],
                "loco_overall_curve_rmse": loco_stats["overall_curve_rmse"],
                "loco_concept_trial_r": loco_stats["concept_trial_r"],
                "loco_concept_trial_rmse": loco_stats["concept_trial_rmse"],
            })

    return {"summary_rows": summary_rows, "by_model": curves_by_model}


def figure_fitted_rational_rules(config, rows, fit_bundle):
    """2x5 concept trajectories + bottom 2-panel norm vs. fitted scatter for the primary model."""
    llm = models_in(rows)[0]
    m_rows = [r for r in rows if r["model"] == llm]
    emp_curves = [r for r in learning_curves(m_rows, config=config) if r["scope_type"] == "concept"]
    by_concept_emp = {}
    group_of = {}
    for r in emp_curves:
        by_concept_emp.setdefault(r["scope"], []).append(r)
        group_of.setdefault(r["scope"], r["complexity_group"])

    ev = fit_bundle["by_model"][llm]["evaluator"]
    fits = fit_bundle["by_model"][llm]["fits"]
    concepts = ev.concept_order
    max_trial, _, boundaries, xticks = _curve_layout(emp_curves)

    full_fit = fits["fit_alpha_beta_prior_temp"]
    loco_curves = full_fit["loco"]["concept_curves"]

    llm_color = "#2a78d6"
    in_color = "#1baf7a"
    full_color = "#eb6834"

    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(FULL_WIDTH, 5.55))
        outer = GridSpec(2, 1, figure=figure, height_ratios=[1.35, 1.0], hspace=0.36)
        top_gs = outer[0].subgridspec(2, 5, wspace=0.18, hspace=0.34)
        bot_gs = outer[1].subgridspec(1, 2, wspace=0.25)

        trials = list(range(1, max_trial + 1))
        top_axes = [[figure.add_subplot(top_gs[r, c]) for c in range(5)] for r in range(2)]

        for idx, concept in enumerate(concepts[:10]):
            r_idx, c_idx = divmod(idx, 5)
            ax = top_axes[r_idx][c_idx]
            for boundary in boundaries:
                ax.axvline(boundary, color=MUTED, linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
            ax.axhline(0.5, color=GRID, linestyle=":", linewidth=0.6, zorder=1)

            pts = sorted(by_concept_emp[concept], key=lambda x: x["trial"])
            emp_y = [p["correct"] for p in pts]
            lows = [p["low"] for p in pts]
            highs = [p["high"] for p in pts]

            ax.plot(trials, full_fit["in_sample"]["concept_curves"][concept], color=in_color, linestyle="--", linewidth=1.05, zorder=3)
            ax.plot(trials, loco_curves[concept], color=full_color, linestyle="-", linewidth=1.2, zorder=4)
            ax.plot(trials, emp_y, color=llm_color, linewidth=1.15, zorder=5)
            if all(lo is not None and hi is not None for lo, hi in zip(lows, highs)):
                ax.fill_between(trials, lows, highs, color=llm_color, alpha=0.16, linewidth=0, zorder=1.5)

            r_fit = pearson_r(emp_y, loco_curves[concept])
            ax.set_title(concept, loc="left", fontsize=7.5, pad=2.0)
            if r_fit is not None:
                ax.text(0.97, 0.06, f"LOCO r={r_fit:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=INK)
            ax.set_xlim(1, max_trial)
            ax.set_ylim(0.0, 1.03)
            ax.set_xticks(xticks)
            ax.set_yticks([0.0, 0.5, 1.0])
            if r_idx == 0:
                ax.tick_params(labelbottom=False)
            if c_idx > 0:
                ax.tick_params(labelleft=False)
            if c_idx == 0:
                ax.set_ylabel("Prop. correct", fontsize=7.5)
            if r_idx == 1 and c_idx == 2:
                ax.set_xlabel(f"Trial (1\u2013{max_trial}, unique stimuli)", fontsize=7.5)
            ax.grid(True, axis="y", color=GRID, linewidth=0.45)
            ax.set_axisbelow(True)

        top_legend = [
            Line2D([0], [0], color=llm_color, lw=1.3, label=llm),
            Line2D([0], [0], color=full_color, lw=1.25, ls="-", label="3-param Rational Rules (out-of-sample LOCO)"),
            Line2D(
                [0], [0], color=in_color, lw=1.1, ls="--",
                label=(
                    f"3-param full fit (\u03b1={full_fit['alpha']:.2f}, "
                    f"\u03b2={full_fit['beta']:.2f}, T={full_fit['prior_temp']:.2f})"
                ),
            ),
        ]
        figure.legend(
            handles=top_legend, frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 0.995), ncols=3, handlelength=1.8, columnspacing=1.2, fontsize=6.7,
        )

        # Bottom panels: 160-point (concept x trial) 3-param calibration (in-sample vs out-of-sample LOCO)
        ax_norm = figure.add_subplot(bot_gs[0])
        ax_loco = figure.add_subplot(bot_gs[1])

        for ax, pred_map, title_str, r_val, rmse_val in (
            (
                ax_norm,
                full_fit["in_sample"]["concept_curves"],
                "3-param Rational Rules \u2014 full fit (160 concept \u00d7 trial pts)",
                full_fit["in_sample"]["concept_trial_r"],
                full_fit["in_sample"]["concept_trial_rmse"],
            ),
            (
                ax_loco,
                loco_curves,
                "3-param Rational Rules \u2014 out-of-sample LOCO CV (160 pts)",
                full_fit["loco"]["concept_trial_r"],
                full_fit["loco"]["concept_trial_rmse"],
            ),
        ):
            ax.plot([0.15, 1.0], [0.15, 1.0], color=MUTED, ls="--", lw=0.8, zorder=1)
            for idx_g, cgroup in enumerate(COMPLEXITY_GROUPS):
                col = complexity_color(cgroup, idx_g)
                xs_g, ys_g = [], []
                for concept in concepts:
                    if group_of.get(concept) != cgroup:
                        continue
                    pts = sorted(by_concept_emp[concept], key=lambda x: x["trial"])
                    xs_g.extend(pred_map[concept])
                    ys_g.extend(p["correct"] for p in pts)
                if xs_g:
                    ax.scatter(xs_g, ys_g, s=13, color=col, alpha=0.58, edgecolors="none", label=cgroup, zorder=2)

            # Overlay 16 overall trial means
            overall_emp = [
                sum(sorted(by_concept_emp[c], key=lambda x: x["trial"])[t]["correct"] for c in concepts) / len(concepts)
                for t in range(max_trial)
            ]
            overall_pred = [
                sum(pred_map[c][t] for c in concepts) / len(concepts)
                for t in range(max_trial)
            ]
            ax.scatter(
                overall_pred, overall_emp, s=20, color=INK, alpha=0.88,
                edgecolors="white", linewidths=0.4, zorder=3, label="Trial mean",
            )
            ax.set_xlim(0.15, 1.03)
            ax.set_ylim(0.15, 1.03)
            ax.set_xticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_title(title_str, loc="left", fontsize=7.5)
            ax.set_xlabel("Bayesian model predicted proportion correct", fontsize=7.5)
            ax.grid(True, color=GRID, linewidth=0.45)
            ax.set_axisbelow(True)
            ax.text(
                0.04, 0.94, f"r = {r_val:.3f}, RMSE = {rmse_val:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
            )
        ax_norm.set_ylabel(f"{llm} proportion correct", fontsize=7.5)
        handles_b, labels_b = ax_loco.get_legend_handles_labels()
        ax_loco.legend(
            handles_b, labels_b, frameon=False, loc="lower right",
            fontsize=6.3, markerscale=0.95, handletextpad=0.2, borderpad=0.1,
        )
        figure.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.99)
    return figure


def _fmt_param(val) -> str:
    if val == "inf" or val is None or (isinstance(val, float) and math.isinf(val)):
        return "\u221e"
    return f"{float(val):.3f}"


def parameter_fits_markdown(summary_rows) -> str:
    """Publication-ready Markdown table of nested parameter fits and LOCO cross-validation."""
    lines = []
    llms = list(dict.fromkeys(r["llm_model"] for r in summary_rows))
    for llm in llms:
        sub = [r for r in summary_rows if r["llm_model"] == llm]
        if len(llms) > 1:
            lines.extend([f"## `{llm}`", ""])
        lines.extend([
            "| Model | Params (`k`) | `alpha` | `beta` | `prior_temp` | Trial NLL | BIC | Pred. Acc | Curve RMSE (16) | 160-pt `r` (RMSE) | LOCO NLL | LOCO 160-pt `r` (RMSE) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ])
        for r in sub:
            lines.append(
                f"| {r['fit_label']} | {r['k_params']} | {_fmt_param(r['alpha'])} | "
                f"{_fmt_param(r['beta'])} | {_fmt_param(r['prior_temp'])} | "
                f"{r['nll']:.1f} | {r['bic']:.1f} | {100 * r['pred_accuracy']:.1f}% | "
                f"{r['overall_curve_rmse']:.3f} | {r['concept_trial_r']:.3f} ({r['concept_trial_rmse']:.3f}) | "
                f"{r['loco_nll']:.1f} | {r['loco_concept_trial_r']:.3f} ({r['loco_concept_trial_rmse']:.3f}) |"
            )
        llm_acc = sub[0]["llm_accuracy"]
        n_valid = sub[0]["n_valid_trials"]
        lines.extend([
            "",
            f"Empirical `{llm}` overall accuracy: **{100 * llm_acc:.1f}%** across {n_valid:,} valid binary responses "
            f"(coin-flip baseline NLL = {-n_valid * math.log(0.5):.1f}). "
            "LOCO columns fit parameters on 9 concepts and evaluate out-of-sample on all 100 blocks of the held-out 10th concept.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def make_all_fits(config=None, *, rows=None, output_dir=None):
    """Fit Rational Rules parameters and write Figure 4 + ``parameter_fits.{csv,md}``."""
    if isinstance(config, (str, Path)) or config is None:
        config = load_config(config or "config_bcl.yaml")
    if output_dir is not None:
        base = Path(output_dir)
        figures_dir = base / "figures" / "bcl"
        tables_dir = base / "tables" / "bcl"
    else:
        figures_dir = config.resolve("figures/bcl")
        tables_dir = config.resolve("tables/bcl")

    if rows is None:
        rows = read_rows(config.resolve(config.run["results_file"]))
    fit_bundle = fit_all_models(config, rows=rows)

    written = []
    written += save(
        figure_fitted_rational_rules(config, rows, fit_bundle),
        figures_dir, "fig4_fitted_rational_rules",
    )
    written.append(
        write_csv(tables_dir / "parameter_fits.csv", fit_bundle["summary_rows"], fieldnames=FIT_CSV_COLUMNS)
    )
    md_path = tables_dir / "parameter_fits.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(parameter_fits_markdown(fit_bundle["summary_rows"]))
    written.append(md_path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit Rational Rules cognitive parameters to BCL results.")
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for path in make_all_fits(config):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
