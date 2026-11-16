"""Residual-stream linear probing and superposed multi-rule marginalization for BCL.

Extracts or loads per-layer residual-stream states from ``gemma4:26b`` strictly *before* any
``Rule:`` scratchpad text is generated (at the final ``:`` token of
``<|turn>model\\n<|channel>thought\\n<channel|>Rule:``, with ``include_rule=False`` so prior
trials' ``rule guess`` strings are omitted from the prompt history), trains independent
``[layer activations] -> [0 or 1 rule]`` binary logistic decoders for each concept across all
layers via trial-stratified 5-fold block-grouped cross-validation, tests whether unstated
second-best hypotheses remain linearly active in superposition on ambiguous trials, and
marginalizes over the 10 independent binary rule probes to predict graded trial-by-trial
``P(yes)`` and ``(concept x trial)`` learning curves.

Run from the project root:

    # 1. (Optional) Prepare binary prompt bundle from data/bcl/results.jsonl for C++ extraction:
    python -m bcl.probe --config config_bcl.yaml \\
        --prepare-prompts data/bcl/probe_inputs.jsonl data/bcl/probe_prompts.bin

    # 2. Train 10-rule per-layer probes, run superposition controls & write figures/tables:
    python -m bcl.probe --config config_bcl.yaml \\
        --input path/to/probe_inputs_1.jsonl path/to/probe_acts_1.f32.bin \\
        --input path/to/probe_inputs_2.jsonl path/to/probe_acts_2.f32.bin

Outputs written
---------------
figures/bcl/fig5_layer_rule_probes.{pdf,png}
    4-panel summary of (A) 10 independent ``[layer -> 0/1 rule]`` binary decoders across layers,
    (B) decoded probability of Rank-1, unstated Rank-2, and falsified rules on ambiguous trials,
    (C) controlled same-pair ``(stated = A, decode = B)`` live-vs.-falsified AUROC across layers,
    and (D) graded readout of unstated rules across Bayesian posterior bins.
figures/bcl/fig6_probe_trial_marginalization.{pdf,png}
    10-concept learning curves (top) and 3-panel mechanism summary (bottom): layer profile of
    trial-level Bayesian alignment comparing superposed 10-rule marginalization against Top-1
    winner-take-all and stated-rule/token baselines, 160-point ``(concept x trial)`` calibration,
    and graded trial-by-trial ``P(yes)`` calibration across Bayesian posterior deciles.
tables/bcl/layer_rule_probes.{csv,md}
    Per-rule 5-fold block-CV decoder metrics and controlled second-best rule pair breakdown.
tables/bcl/probe_trial_predictions.{csv,md}
    Trial-by-trial ``P(yes)`` alignment, LLM binary-choice NLL, and ``(concept x trial)``
    learning-curve accuracy across layers and behavioral/Bayesian baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bcl import elicit  # noqa: E402
from bcl.analysis import (  # noqa: E402
    COMPLEXITY_COLORS,
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
from bcl.fit import _HypothesisEvaluator, _nelder_mead  # noqa: E402
from bcl.score import (  # noqa: E402
    COMPLEXITY_GROUPS,
    concepts_in,
    learning_curves,
    models_in,
    pearson_r,
    read_rows,
    rmse,
)

DEFAULT_COLORS = ("red", "blue", "green", "yellow")
DEFAULT_SHAPES = ("circle", "square", "triangle", "diamond")

RULE_PROBE_CSV_COLUMNS = (
    "concept",
    "complexity_group",
    "n_pos_trials",
    "n_neg_trials",
    "l00_auroc",
    "l90_layer",
    "peak_layer",
    "peak_bal_acc",
    "peak_auroc",
    "peak_bce",
    "rep_layer",
    "rep_bal_acc",
    "rep_auroc",
    "final_layer",
    "final_bal_acc",
    "final_auroc",
)

TRIAL_PRED_CSV_COLUMNS = (
    "predictor_id",
    "predictor_label",
    "kind",
    "layer",
    "k_params",
    "n_trials",
    "r_trial_fit_bayes",
    "rmse_trial_fit_bayes",
    "r_trial_norm_bayes",
    "rmse_trial_norm_bayes",
    "llm_choice_mean_nll",
    "llm_choice_agreement",
    "sub_160_r",
    "sub_160_rmse",
    "full_160_r",
    "full_160_rmse",
)


# ---------------------------------------------------------------------------
# Natural-English scratchpad rule parser & prompt serialization
# ---------------------------------------------------------------------------

def parse_rule_to_ext(raw: str | None, stimuli: list[dict], features: dict | None = None):
    """Parse a natural-English ``Rule:`` string into a Boolean tuple over ``stimuli``, ``'GUESSING'``, or ``None``."""
    if not raw:
        return None
    s = raw.lower().strip().rstrip(".")
    if s in ("guessing", "unknown", "none", "unsure", "random"):
        return "GUESSING"
    if s in ("any color", "any shape", "any color or shape", "any color or any shape"):
        return tuple(1 for _ in stimuli)

    if features is not None:
        colors = {str(v).lower() for v in features.get("color", ())}
        shapes = {str(v).lower() for v in features.get("shape", ())}
    else:
        colors = {str(st.get("color", "")).lower() for st in stimuli if "color" in st} or set(DEFAULT_COLORS)
        shapes = {str(st.get("shape", "")).lower() for st in stimuli if "shape" in st} or set(DEFAULT_SHAPES)

    color_pat = "|".join(sorted((re.escape(c) for c in colors), key=len, reverse=True))
    shape_pat = "|".join(sorted((re.escape(sh) for sh in shapes), key=len, reverse=True))

    s = re.sub(r",\s*not\b", " and not ", s)
    s = s.replace("except", " and not ").replace("but not", " and not ")
    s = s.replace("unless", " and not ").replace("other than", " and not ")
    s = re.sub(r"\bneither\s+([a-z]+)\s+nor\s+([a-z]+)\b", r"(not \1 and not \2)", s)
    s = s.replace("nor", " and not ")
    s = re.sub(r"\bany\s+(?:color|shape)(?:\s+or\s+(?:any\s+)?(?:color|shape))?\s+and\s+not\b", "not", s)
    if color_pat:
        s = re.sub(
            rf"\bnot\s+({color_pat})\s+or\s+({color_pat})(?:\s+or\s+({color_pat}))?\b",
            lambda m: "not (" + " or ".join(g for g in m.groups() if g) + ")",
            s,
        )
    if shape_pat:
        s = re.sub(
            rf"\bnot\s+({shape_pat})\s+or\s+({shape_pat})(?:\s+or\s+({shape_pat}))?\b",
            lambda m: "not (" + " or ".join(g for g in m.groups() if g) + ")",
            s,
        )
    s = re.sub(
        r"\b(the|an|a|object|objects|item|items|is|are|be|being|has|have|with|that|which|either|both|any|anything)\b",
        " ",
        s,
    )
    s = re.sub(r"\b(color|shape|colored|shaped)\b", " ", s)
    s = s.replace("!=", " not ").replace("==", " ").replace("=", " ")
    s = s.replace("/", " or ").replace("&", " and ").replace("|", " or ")
    s = re.sub(r",\s*or\b", " or ", s)
    s = re.sub(r",\s*and\b", " and ", s)
    s = s.replace(",", " or ")
    s = re.sub(r"\s+", " ", s).strip()

    toks = re.findall(r"\(|\)|\band\b|\bor\b|\bnot\b|[a-z]+", s)
    allowed = colors | shapes | {"and", "or", "not", "(", ")"}
    if not toks or any(t not in allowed for t in toks):
        return None

    expanded: list[str] = []
    for t in toks:
        if expanded:
            prev = expanded[-1]
            if (prev in colors and t in shapes) or (prev in shapes and t in colors):
                expanded.append("and")
            elif (prev in (colors | shapes | {")"})) and (t in (colors | shapes | {"(", "not"})):
                expanded.append("and")
        expanded.append(t)

    expr_str = " ".join(expanded)
    if color_pat and shape_pat:
        expr_str = re.sub(rf"\bnot\s+({color_pat})\s+and\s+({shape_pat})\b", r"not (\1 and \2)", expr_str)
        expr_str = re.sub(rf"\bnot\s+({shape_pat})\s+and\s+({color_pat})\b", r"not (\1 and \2)", expr_str)

    toks2 = re.findall(r"\(|\)|\band\b|\bor\b|\bnot\b|[a-z]+", expr_str)
    py_toks = [
        f"(d['color'] == {t!r})" if t in colors else (f"(d['shape'] == {t!r})" if t in shapes else t)
        for t in toks2
    ]
    try:
        fn = eval("lambda d: bool(" + " ".join(py_toks) + ")", {"__builtins__": {"bool": bool}})
        return tuple(1 if fn(st) else 0 for st in stimuli)
    except Exception:
        return None


def _norm_rule(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw.lower().strip().rstrip(".")
    s = re.sub(r"^(the\s+)?(object\s+is\s+|color\s+is\s+|shape\s+is\s+|it\s+is\s+)", "", s)
    s = re.sub(r"\s+(color|shape|objects?)$", "", s)
    return s.strip()


def render_trial_prompt(block_row: dict, trial_idx_0: int, features: dict, *, include_rule: bool = False) -> str:
    """Render a Gemma 4 chat prompt terminating at ``Rule:`` before any scratchpad rule text is emitted."""
    trials = block_row["trials"]
    history = trials[:trial_idx_0]
    current = trials[trial_idx_0]
    trial_num = int(current["trial"])
    n_trials = int(block_row["n_trials"])
    repetitions = int(block_row["repetitions"])
    n_objects = 1
    for vals in features.values():
        n_objects *= len(vals)
    pass_clause = (
        ", each shown once"
        if repetitions == 1
        else f" (shown across {repetitions} shuffled passes)"
    )
    user_msg = elicit.template("trial_rule").substitute(
        word=block_row["word"],
        n_trials=n_trials,
        n_objects=n_objects,
        feature_space=elicit.describe_feature_space(features),
        pass_clause=pass_clause,
        feature_names=elicit.describe_feature_names(features),
        history_block=elicit.format_history_block(
            history, block_row["word"], features, include_rule=include_rule
        ),
        trial=trial_num,
        object=elicit.stimulus_phrase(current["object"], features),
    )
    return f"<bos><|turn>user\n{user_msg}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Rule:"


def prepare_probe_prompts(
    config,
    rows: list[dict],
    out_jsonl: str | Path,
    out_bin: str | Path,
    *,
    blocks_per_concept: int = 25,
    include_rule: bool = False,
) -> int:
    """Serialize pure-induction BCL trial prompts for ``tools/extract_gemma4_acts``."""
    out_jsonl = Path(out_jsonl)
    out_bin = Path(out_bin)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    stimuli = config.stimuli
    features = config.features
    condition = "with_rule_hist" if include_rule else "no_rule_hist"

    by_concept: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_concept[r["concept"]].append(r)
    for c in by_concept:
        by_concept[c].sort(key=lambda r: int(r["run_index"]))

    rule_by_name = {r.name: r for r in config.rules}
    selected_blocks: list[dict] = []
    for cname in concepts_in(rows, list(config.programs.keys())):
        c_rows = by_concept[cname]
        cext = tuple(1 if rule_by_name[cname](st) else 0 for st in stimuli)
        pos_blocks = [
            r for r in c_rows
            if any(parse_rule_to_ext(t.get("rule_guess"), stimuli, features) == cext for t in r["trials"])
        ]
        pos_ids = {r["block_id"] for r in pos_blocks}
        neg_blocks = [r for r in c_rows if r["block_id"] not in pos_ids]
        chosen = sorted((pos_blocks + neg_blocks)[:blocks_per_concept], key=lambda r: int(r["run_index"]))
        selected_blocks.extend(chosen)

    total_records = sum(len(b["trials"]) for b in selected_blocks)
    n_written = 0
    with out_jsonl.open("w") as jout, out_bin.open("wb") as bout:
        bout.write(struct.pack("<I", total_records))
        for block_row in selected_blocks:
            block_seq_key = f"{condition}|{block_row['block_id']}"
            for t_idx, t in enumerate(sorted(block_row["trials"], key=lambda x: int(x["trial"]))):
                prompt = render_trial_prompt(block_row, t_idx, features, include_rule=include_rule)
                pbytes = prompt.encode("utf-8")
                is_new_block = 1 if t_idx == 0 else 0
                bout.write(struct.pack("<II", is_new_block, len(pbytes)))
                bout.write(pbytes)
                rg_raw = t.get("rule_guess")
                rec = {
                    "row_idx": n_written,
                    "condition": condition,
                    "include_rule": include_rule,
                    "block_seq_key": block_seq_key,
                    "block_id": block_row["block_id"],
                    "concept": block_row["concept"],
                    "complexity_group": block_row["complexity_group"],
                    "run_index": int(block_row["run_index"]),
                    "word": block_row["word"],
                    "trial": int(t["trial"]),
                    "object": t["object"],
                    "truth": bool(t["truth"]),
                    "answer": t.get("answer"),
                    "p_yes": t.get("p_yes"),
                    "rule_guess_raw": rg_raw,
                    "rule_guess_norm": _norm_rule(rg_raw),
                }
                jout.write(json.dumps(rec) + "\n")
                n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Linear probing math: weighted dual L2 logistic regression & AUROC
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_score: np.ndarray, sample_w: np.ndarray | None = None) -> float:
    """Exact (optionally sample-weighted) Mann-Whitney U area under the ROC curve with 0.5 tie-breaking."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_m = y_true == 1
    neg_m = y_true == 0
    pos = y_score[pos_m]
    neg = y_score[neg_m]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    win = (diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)
    if sample_w is None:
        return float(np.mean(win))
    w = np.asarray(sample_w, dtype=np.float64)
    wp = w[pos_m]
    wn = w[neg_m]
    denom = float(np.sum(wp) * np.sum(wn))
    return float(np.sum((wp[:, None] * win) * wn[None, :]) / denom) if denom > 0.0 else 0.5


def fit_weighted_l2_logreg_dual(
    X: np.ndarray,
    y: np.ndarray,
    sample_w: np.ndarray,
    l2_reg: float = 100.0,
    *,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Exact Newton-Raphson sample-weighted L2 binary logistic regression in dual space.

    Minimizes ``sum_i sample_w[i] * BCE(sigma(X_i @ w + b), y_i) + 0.5 * l2_reg * (||w||^2 + 0.01 * b^2)``.
    """
    n_samples, _ = X.shape
    sb = 10.0
    K = X @ X.T + (sb * sb)
    z = np.zeros(n_samples, dtype=np.float64)
    u = np.asarray(sample_w, dtype=np.float64)
    alpha = np.zeros(n_samples, dtype=np.float64)

    for _ in range(max_iter):
        z_clip = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z_clip))
        r = np.maximum(u * p * (1.0 - p), 1e-9)
        s = np.sqrt(r)
        A = (s[:, None] * K) * s[None, :]
        A[np.diag_indices(n_samples)] += l2_reg
        rhs = s * z + u * (y - p) / s
        sol = np.linalg.solve(A, rhs)
        alpha_new = s * sol
        z_new = K @ alpha_new
        if np.max(np.abs(z_new - z)) < tol:
            z = z_new
            alpha = alpha_new
            break
        z = z_new
        alpha = alpha_new

    w = X.T @ alpha
    b = sb * sb * float(np.sum(alpha))
    return w, b


def build_rule_cv_dataset(
    meta_rows: list[dict],
    ci: int,
    concept_names: list[str],
    stated_exts: list,
    concept_exts: dict[str, tuple],
    block_folds: np.ndarray,
    *,
    neg_ratio: int = 3,
    min_neg_per_ft: int = 18,
    max_neg_per_ft: int = 30,
    seed: int = 20261114,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build trial-stratified dataset for binary rule decoder ``ci`` with exact 50/50 weight per ``(fold, trial)``."""
    rng = np.random.default_rng(seed + 97 * ci)
    cname = concept_names[ci]
    cext = concept_exts[cname]

    pos_by_ft: dict[tuple[int, int], list[int]] = defaultdict(list)
    neg_by_ft: dict[tuple[int, int], list[int]] = defaultdict(list)

    for i, r in enumerate(meta_rows):
        sext = stated_exts[i]
        if not isinstance(sext, tuple):
            continue
        f = int(block_folds[i])
        t = int(r["trial"])
        if r["concept"] == cname and sext == cext:
            pos_by_ft[(f, t)].append(i)
        elif r["concept"] != cname and sext != cext:
            neg_by_ft[(f, t)].append(i)

    neg_by_f: dict[int, list[int]] = defaultdict(list)
    for (f, _), nlist in neg_by_ft.items():
        neg_by_f[f].extend(nlist)

    idx_list, y_list, w_list, fold_list = [], [], [], []
    for (f, t), plist in sorted(pos_by_ft.items()):
        kp = len(plist)
        if kp == 0:
            continue
        n_pool = neg_by_ft.get((f, t), [])
        if len(n_pool) == 0:
            n_pool = neg_by_f[f]
        kn = min(len(n_pool), min(max(neg_ratio * kp, min_neg_per_ft), max_neg_per_ft))
        if kn == 0:
            continue
        nsel = rng.choice(n_pool, size=kn, replace=False)
        for idx in plist:
            idx_list.append(int(idx))
            y_list.append(1.0)
            w_list.append(1.0)
            fold_list.append(f)
        neg_w = float(kp) / float(kn)
        for idx in nsel:
            idx_list.append(int(idx))
            y_list.append(0.0)
            w_list.append(neg_w)
            fold_list.append(f)

    return (
        np.array(idx_list, dtype=np.int64),
        np.array(y_list, dtype=np.float64),
        np.array(w_list, dtype=np.float64),
        np.array(fold_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Exact Bayesian posteriors & superposed multi-rule marginalization
# ---------------------------------------------------------------------------

def compute_rule_posteriors_and_q_yes(
    config,
    meta_rows: list[dict],
    concept_names: list[str],
    concept_exts: dict[str, tuple],
    *,
    fitted_params: tuple[float, float | None, float] = (0.72954, 1.96550, 3.07512),
) -> dict[str, np.ndarray]:
    """Compute per-concept posteriors, deterministic consistency, and predictive ``P(yes)`` on ``meta_rows``."""
    stimuli = config.stimuli
    feat_keys = list(config.features.keys())
    stim_key = lambda s: tuple(s[k] for k in feat_keys)
    stim_to_idx = {stim_key(s): i for i, s in enumerate(stimuli)}

    hyp_rules = config.hypothesis_rules
    exts = np.array([[1.0 if r(s) else 0.0 for s in stimuli] for r in hyp_rules], dtype=np.float64)
    log_w0 = np.array([math.log(r.weight) for r in hyp_rules], dtype=np.float64)

    n_concepts = len(concept_names)
    c_ext_mat = np.array([concept_exts[c] for c in concept_names], dtype=np.float64)
    c_hyp_masks = [np.all(exts == c_ext_mat[ci, None, :], axis=1) for ci in range(n_concepts)]

    n_rows = len(meta_rows)
    p_gram_norm = np.zeros((n_rows, n_concepts), dtype=np.float64)
    p_10_norm = np.zeros((n_rows, n_concepts), dtype=np.float64)
    p_gram_fit = np.zeros((n_rows, n_concepts), dtype=np.float64)
    p_10_fit = np.zeros((n_rows, n_concepts), dtype=np.float64)
    consistent = np.ones((n_rows, n_concepts), dtype=bool)
    q_norm_pm = np.zeros(n_rows, dtype=np.float64)
    q_norm_map = np.zeros(n_rows, dtype=np.float64)
    q_fit_3p = np.zeros(n_rows, dtype=np.float64)

    base_alpha = float(config.alpha)
    base_temp = float(config.settings["prior"]["temperature"] or 1.0)
    fit_alpha, fit_beta, fit_temp = fitted_params

    by_seq: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, r in enumerate(meta_rows):
        by_seq[r["block_seq_key"]].append((idx, r))

    for _, indexed_trials in by_seq.items():
        indexed_trials.sort(key=lambda x: int(x[1]["trial"]))
        n_t = len(indexed_trials)
        s_idx = np.array([stim_to_idx[stim_key(t["object"])] for _, t in indexed_trials], dtype=np.int64)
        truth = np.array([1.0 if t["truth"] else 0.0 for _, t in indexed_trials], dtype=np.float64)

        c_preds = c_ext_mat[:, s_idx].T
        c_agree = c_preds == truth[:, None]
        cum_mismatches = np.zeros((n_t, n_concepts), dtype=np.int64)
        cum_mismatches[1:, :] = np.cumsum((~c_agree[:-1, :]).astype(np.int64), axis=0)

        ext_t = exts.T[s_idx]
        agree_t = (ext_t == truth[:, None]).astype(np.float64)

        for is_norm, alpha, beta, prior_temp in (
            (True, base_alpha, 1.0, base_temp),
            (False, fit_alpha, fit_beta, fit_temp),
        ):
            alpha_c = min(1.0 - 1e-9, max(1e-9, float(alpha)))
            p_m = 0.5 * (1.0 + alpha_c)
            p_mm = 0.5 * (1.0 - alpha_c)
            dll = agree_t * (math.log(p_m) - math.log(p_mm)) + math.log(p_mm)
            cum_ll = np.zeros_like(dll)
            cum_ll[1:, :] = np.cumsum(dll[:-1, :], axis=0)
            lp = (log_w0[None, :] / max(1e-6, float(prior_temp))) + cum_ll
            lp -= np.max(lp, axis=1, keepdims=True)
            post = np.exp(lp)
            post /= np.sum(post, axis=1, keepdims=True)

            c_masses = np.column_stack([np.sum(post[:, m], axis=1) for m in c_hyp_masks])
            c_cond = c_masses / np.maximum(np.sum(c_masses, axis=1, keepdims=True), 1e-15)

            rule_py = np.sum(post * ext_t, axis=1)
            p_pred = alpha_c * rule_py + 0.5 * (1.0 - alpha_c)

            if is_norm:
                q_map = np.where(np.abs(p_pred - 0.5) <= 1e-9, 0.5, (p_pred > 0.5).astype(np.float64))
                for t_i, (global_idx, _) in enumerate(indexed_trials):
                    p_gram_norm[global_idx, :] = c_masses[t_i, :]
                    p_10_norm[global_idx, :] = c_cond[t_i, :]
                    consistent[global_idx, :] = cum_mismatches[t_i, :] == 0
                    q_norm_pm[global_idx] = p_pred[t_i]
                    q_norm_map[global_idx] = q_map[t_i]
            else:
                if beta is None or math.isinf(beta):
                    q_f = np.where(np.abs(p_pred - 0.5) <= 1e-9, 0.5, (p_pred > 0.5).astype(np.float64))
                elif abs(beta - 1.0) <= 1e-12:
                    q_f = p_pred
                else:
                    clip_p = np.clip(p_pred, 1e-12, 1.0 - 1e-12)
                    q_f = 1.0 / (1.0 + np.exp(-np.clip(float(beta) * np.log(clip_p / (1.0 - clip_p)), -40.0, 40.0)))
                for t_i, (global_idx, _) in enumerate(indexed_trials):
                    p_gram_fit[global_idx, :] = c_masses[t_i, :]
                    p_10_fit[global_idx, :] = c_cond[t_i, :]
                    q_fit_3p[global_idx] = q_f[t_i]

    return {
        "p_gram_norm": p_gram_norm,
        "p_10_norm": p_10_norm,
        "p_gram_fit": p_gram_fit,
        "p_10_fit": p_10_fit,
        "consistent": consistent,
        "q_norm_pm": q_norm_pm,
        "q_norm_map": q_norm_map,
        "q_fit_3p": q_fit_3p,
    }


def superposed_marginalize_q_yes(
    probs_c: np.ndarray,
    rule_eval_cur: np.ndarray,
    *,
    gamma: float = 1.0,
    tau_0: float = 1.0,
) -> np.ndarray:
    """Marginalize independent binary rule probe probabilities ``probs_c`` ``(N, C)`` into graded ``P(yes)``.

    With ``gamma = 1.0, tau_0 = 1.0`` (default), this has **zero fitted parameters**:
    ``q_yes = (sum_c p_c * c(x_t) + 0.5 * (1 - max_c p_c)) / (sum_c p_c + (1 - max_c p_c))``.
    """
    p_safe = np.clip(probs_c, 1e-9, 1.0 - 1e-9)
    max_p = np.max(p_safe, axis=1)
    weights = p_safe if abs(gamma - 1.0) <= 1e-12 else np.power(p_safe, float(gamma))
    uncomm = max(0.0, float(tau_0)) * (1.0 - max_p)
    denom = np.maximum(np.sum(weights, axis=1) + uncomm, 1e-12)
    return (np.sum(weights * rule_eval_cur, axis=1) + 0.5 * uncomm) / denom


def top1_marginalize_q_yes(probs_c: np.ndarray, rule_eval_cur: np.ndarray) -> np.ndarray:
    """Winner-take-all ablation keeping only ``argmax_c p_c(h_t)`` and discarding 2nd/3rd-best rules."""
    n_rows = probs_c.shape[0]
    top1_idx = np.argmax(probs_c, axis=1)
    max_p = probs_c[np.arange(n_rows), top1_idx]
    top1_eval = rule_eval_cur[np.arange(n_rows), top1_idx]
    return max_p * top1_eval + 0.5 * (1.0 - max_p)


def loco_calibrate_superposed_q_yes(
    probs_c: np.ndarray,
    rule_eval_cur: np.ndarray,
    ans_yes: np.ndarray,
    valid: np.ndarray,
    concepts_arr: np.ndarray,
    concept_names: list[str],
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Fit ``(gamma, tau_0)`` via Leave-One-Concept-Out CV on LLM binary choices."""
    if len(concept_names) < 2:
        return superposed_marginalize_q_yes(probs_c, rule_eval_cur), {}

    q_loco = np.zeros(len(ans_yes), dtype=np.float64)
    fold_params: dict[str, tuple[float, float]] = {}

    for held_out in concept_names:
        tr_m = (concepts_arr != held_out) & valid
        te_m = concepts_arr == held_out
        P_tr = probs_c[tr_m]
        E_tr = rule_eval_cur[tr_m]
        y_tr = ans_yes[tr_m]

        def obj(z):
            gamma = math.exp(max(-4.0, min(4.0, float(z[0]))))
            tau_0 = math.exp(max(-6.0, min(6.0, float(z[1]))))
            q = np.clip(superposed_marginalize_q_yes(P_tr, E_tr, gamma=gamma, tau_0=tau_0), 1e-9, 1.0 - 1e-9)
            return -float(np.mean(y_tr * np.log(q) + (1.0 - y_tr) * np.log(1.0 - q)))

        best_z, _ = _nelder_mead(obj, [0.0, 0.0], step=0.35, max_iter=250)
        gamma_hat = math.exp(max(-4.0, min(4.0, float(best_z[0]))))
        tau_hat = math.exp(max(-6.0, min(6.0, float(best_z[1]))))
        fold_params[held_out] = (gamma_hat, tau_hat)
        q_loco[te_m] = superposed_marginalize_q_yes(
            probs_c[te_m], rule_eval_cur[te_m], gamma=gamma_hat, tau_0=tau_hat
        )

    return q_loco, fold_params


def evaluate_trial_predictions(
    q_yes_pred: np.ndarray,
    *,
    truth: np.ndarray,
    ans_yes: np.ndarray,
    valid: np.ndarray,
    q_bayes_fit: np.ndarray,
    q_bayes_norm: np.ndarray,
    concepts_arr: np.ndarray,
    trials_arr: np.ndarray,
    concept_names: list[str],
    max_trial: int,
    emp_160_full: np.ndarray,
) -> dict:
    """Evaluate a trial-level ``P(yes)`` vector against Bayesian posteriors, LLM choices, and 160-pt curves."""
    r_trial_fit = pearson_r(q_yes_pred.tolist(), q_bayes_fit.tolist()) or 0.0
    rmse_trial_fit = rmse(q_yes_pred.tolist(), q_bayes_fit.tolist()) or 0.0
    r_trial_norm = pearson_r(q_yes_pred.tolist(), q_bayes_norm.tolist()) or 0.0
    rmse_trial_norm = rmse(q_yes_pred.tolist(), q_bayes_norm.tolist()) or 0.0

    qc = np.clip(q_yes_pred[valid], 1e-6, 1.0 - 1e-6)
    ay = ans_yes[valid]
    mean_nll_llm = -float(np.mean(ay * np.log(qc) + (1.0 - ay) * np.log(1.0 - qc))) if len(ay) else 0.0
    agree_llm = float(np.mean(((qc >= 0.5).astype(np.float64)) == ay)) if len(ay) else 0.0

    p_corr = q_yes_pred * truth + (1.0 - q_yes_pred) * (1.0 - truth)
    emp_corr = (ans_yes == truth).astype(np.float64)

    pts_pred: list[float] = []
    pts_sub_emp: list[float] = []
    concept_curves: dict[str, list[float]] = {}
    for cname in concept_names:
        c_curve: list[float] = []
        for t in range(1, max_trial + 1):
            m = (concepts_arr == cname) & (trials_arr == t)
            val_p = float(np.mean(p_corr[m])) if np.any(m) else 0.5
            val_e = float(np.mean(emp_corr[m])) if np.any(m) else 0.5
            c_curve.append(val_p)
            pts_pred.append(val_p)
            pts_sub_emp.append(val_e)
        concept_curves[cname] = c_curve

    sub_r = pearson_r(pts_pred, pts_sub_emp) or 0.0
    sub_rmse = rmse(pts_pred, pts_sub_emp) or 0.0
    full_r = pearson_r(pts_pred, emp_160_full.tolist()) or 0.0
    full_rmse = rmse(pts_pred, emp_160_full.tolist()) or 0.0

    return {
        "r_trial_fit_bayes": r_trial_fit,
        "rmse_trial_fit_bayes": rmse_trial_fit,
        "r_trial_norm_bayes": r_trial_norm,
        "rmse_trial_norm_bayes": rmse_trial_norm,
        "llm_choice_mean_nll": mean_nll_llm,
        "llm_choice_agreement": agree_llm,
        "sub_160_r": sub_r,
        "sub_160_rmse": sub_rmse,
        "full_160_r": full_r,
        "full_160_rmse": full_rmse,
        "concept_curves": concept_curves,
        "pts_160": pts_pred,
    }


# ---------------------------------------------------------------------------
# End-to-end multi-rule probing, second-best analysis & marginalization
# ---------------------------------------------------------------------------

def analyze_probes(
    config,
    input_pairs: list[tuple[str | Path, str | Path]],
    *,
    full_rows: list[dict] | None = None,
    n_embd: int = 2816,
    n_folds: int = 5,
    l2_grid: tuple[float, ...] = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0),
) -> dict:
    """Train per-layer binary rule decoders, test 2nd-best superposition, and marginalize trial ``P(yes)``."""
    if not input_pairs:
        raise ValueError("at least one (meta_jsonl, acts_f32_bin) input pair is required")

    if full_rows is None:
        results_path = config.resolve(config.run["results_file"])
        full_rows = read_rows(results_path) if results_path.exists() else []

    # Load metadata & memory-map activation binaries
    part_memmaps = []
    meta_rows: list[dict] = []
    src_part_list: list[int] = []
    src_idx_list: list[int] = []
    n_layers_detected: int | None = None

    for p_idx, (meta_path, acts_path) in enumerate(input_pairs):
        meta_path = Path(meta_path)
        acts_path = Path(acts_path)
        p_rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
        if not p_rows:
            raise ValueError(f"empty metadata file: {meta_path}")
        fbytes = acts_path.stat().st_size
        denom = 4 * len(p_rows) * n_embd
        if fbytes % denom != 0:
            raise ValueError(
                f"{acts_path} byte size ({fbytes}) is not divisible by 4 * {len(p_rows)} * {n_embd}"
            )
        p_layers = fbytes // denom
        if n_layers_detected is None:
            n_layers_detected = p_layers
        elif p_layers != n_layers_detected:
            raise ValueError(f"layer count mismatch across inputs: {p_layers} vs {n_layers_detected}")

        mm = np.memmap(acts_path, dtype=np.float32, mode="r", shape=(len(p_rows), p_layers, n_embd))
        part_memmaps.append(mm)
        for r in p_rows:
            if r.get("condition", "no_rule_hist") == "no_rule_hist":
                meta_rows.append(r)
                src_part_list.append(p_idx)
                src_idx_list.append(int(r["row_idx"]))

    assert n_layers_detected is not None
    n_layers = n_layers_detected
    n_rows = len(meta_rows)
    if n_rows == 0:
        raise ValueError("no 'no_rule_hist' rows found in input metadata")

    src_part = np.array(src_part_list, dtype=np.int64)
    src_idx = np.array(src_idx_list, dtype=np.int64)

    stimuli = config.stimuli
    features = config.features
    feat_keys = list(features.keys())
    stim_key = lambda s: tuple(s[k] for k in feat_keys)
    stim_to_idx = {stim_key(s): i for i, s in enumerate(stimuli)}

    rule_by_name = {r.name: r for r in config.rules}
    concept_names = concepts_in(meta_rows, list(config.programs.keys()))
    n_concepts = len(concept_names)
    c_to_idx = {c: i for i, c in enumerate(concept_names)}
    concept_exts = {
        c: tuple(1 if rule_by_name[c](st) else 0 for st in stimuli)
        for c in concept_names
    }
    ext_to_cidx = {concept_exts[c]: i for i, c in enumerate(concept_names)}
    max_trial = max(int(r["trial"]) for r in meta_rows)

    # Resolve 3-param Rational Rules parameters from parameter_fits.csv when present, else fit directly
    if full_rows:
        primary_llm = models_in(full_rows)[0]
        m_full_rows = [r for r in full_rows if r["model"] == primary_llm]
        fitted_params = None
        fits_csv = config.resolve("tables/bcl/parameter_fits.csv")
        if fits_csv.exists():
            with fits_csv.open() as fh:
                for frow in csv.DictReader(fh):
                    if frow.get("llm_model") == primary_llm and frow.get("fit_id") == "fit_alpha_beta_prior_temp":
                        b_raw = frow["beta"]
                        fitted_params = (
                            float(frow["alpha"]),
                            None if b_raw == "inf" else float(b_raw),
                            float(frow["prior_temp"]),
                        )
                        break
        if fitted_params is None:
            ev_full = _HypothesisEvaluator(config, m_full_rows)
            fitted_params = ev_full.fit_spec("fit_alpha_beta_prior_temp")
        emp_map: dict[tuple[str, int], list[float]] = defaultdict(list)
        for r in m_full_rows:
            for t in r["trials"]:
                emp_map[(r["concept"], int(t["trial"]))].append(1.0 if t["correct"] else 0.0)
        emp_160_full = np.array(
            [
                float(np.mean(emp_map[(c, t)])) if emp_map[(c, t)] else 0.5
                for c in concept_names
                for t in range(1, max_trial + 1)
            ],
            dtype=np.float64,
        )
    else:
        primary_llm = "model"
        fitted_params = (float(config.alpha), 1.0, float(config.settings["prior"]["temperature"] or 1.0))
        emp_160_full = np.zeros(n_concepts * max_trial, dtype=np.float64)

    stated_exts = [parse_rule_to_ext(r.get("rule_guess_raw"), stimuli, features) for r in meta_rows]
    stated_cidx = np.array(
        [ext_to_cidx.get(e, -1) if isinstance(e, tuple) else -1 for e in stated_exts],
        dtype=np.int64,
    )
    trials_arr = np.array([int(r["trial"]) for r in meta_rows], dtype=np.int64)
    concepts_arr = np.array([r["concept"] for r in meta_rows])
    true_cidx = np.array([c_to_idx[r["concept"]] for r in meta_rows], dtype=np.int64)
    truth = np.array([1.0 if r["truth"] else 0.0 for r in meta_rows], dtype=np.float64)
    valid = np.array([r.get("answer") is not None for r in meta_rows], dtype=bool)
    ans_yes = np.array([1.0 if r.get("answer") else 0.0 for r in meta_rows], dtype=np.float64)
    tok_p_yes = np.array(
        [float(r["p_yes"]) if r.get("p_yes") is not None else 0.5 for r in meta_rows],
        dtype=np.float64,
    )
    if not full_rows:
        emp_corr_sub = (ans_yes == truth).astype(np.float64)
        emp_160_full = np.array(
            [
                float(np.mean(emp_corr_sub[(concepts_arr == c) & (trials_arr == t)]))
                for c in concept_names
                for t in range(1, max_trial + 1)
            ],
            dtype=np.float64,
        )

    cur_s_idx = np.array([stim_to_idx[stim_key(r["object"])] for r in meta_rows], dtype=np.int64)
    c_ext_mat = np.array([concept_exts[c] for c in concept_names], dtype=np.float64)
    rule_eval_cur = c_ext_mat[:, cur_s_idx].T  # (N, C)

    q_stated_rule = np.full(n_rows, 0.5, dtype=np.float64)
    for i, e in enumerate(stated_exts):
        if isinstance(e, tuple):
            q_stated_rule[i] = float(e[cur_s_idx[i]])

    # Assign stratified block-grouped folds (0..n_folds-1)
    blocks_by_c: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(meta_rows):
        blocks_by_c[r["concept"]][r["block_seq_key"]].append(i)

    n_blocks_total = sum(len(v) for v in blocks_by_c.values())
    eff_folds = min(n_folds, min(len(v) for v in blocks_by_c.values()))
    eff_folds = max(2, eff_folds)

    block_folds = np.zeros(n_rows, dtype=np.int64)
    for cname in concept_names:
        ci = c_to_idx[cname]
        bkeys = list(blocks_by_c[cname].keys())
        bkeys.sort(
            key=lambda bk: (
                -int(any(stated_cidx[idx] == ci for idx in blocks_by_c[cname][bk])),
                int(meta_rows[blocks_by_c[cname][bk][0]]["run_index"]),
            )
        )
        for rank, bk in enumerate(bkeys):
            f = rank % eff_folds
            for idx in blocks_by_c[cname][bk]:
                block_folds[idx] = f

    bayes = compute_rule_posteriors_and_q_yes(
        config, meta_rows, concept_names, concept_exts, fitted_params=fitted_params
    )
    p_10_norm = bayes["p_10_norm"]
    p_10_fit = bayes["p_10_fit"]
    consistent = bayes["consistent"]
    q_norm_pm = bayes["q_norm_pm"]
    q_fit_3p = bayes["q_fit_3p"]

    # Build per-rule CV datasets and train C x n_layers x eff_folds binary logistic decoders
    cv_datasets = {
        ci: build_rule_cv_dataset(
            meta_rows, ci, concept_names, stated_exts, concept_exts, block_folds
        )
        for ci in range(n_concepts)
    }

    oof_probs_all = np.zeros((n_layers, n_rows, n_concepts), dtype=np.float64)
    oof_logits_all = np.zeros((n_layers, n_rows, n_concepts), dtype=np.float64)
    per_rule_layer_metrics: dict[str, list[dict]] = {c: [] for c in concept_names}
    X_layer = np.empty((n_rows, n_embd), dtype=np.float32)

    for lid in range(n_layers):
        for p_idx, mm in enumerate(part_memmaps):
            pmask = src_part == p_idx
            X_layer[pmask, :] = mm[src_idx[pmask], lid, :]

        for ci, cname in enumerate(concept_names):
            d_idx, d_y, d_w, d_f = cv_datasets[ci]
            X_cv = np.asarray(X_layer[d_idx, :], dtype=np.float64)

            best_bce = float("inf")
            best_res = None
            best_fold_models = None

            for l2 in l2_grid:
                cv_probs = np.zeros(len(d_idx), dtype=np.float64)
                cv_logits = np.zeros(len(d_idx), dtype=np.float64)
                fold_models = {}
                for f in range(eff_folds):
                    te = d_f == f
                    tr = ~te
                    if not np.any(tr):
                        fold_models[f] = (
                            np.zeros(n_embd, dtype=np.float64),
                            np.ones(n_embd, dtype=np.float64),
                            np.zeros(n_embd, dtype=np.float64),
                            -6.0,
                        )
                        continue
                    w_tr = d_w[tr]
                    w_tr_norm = w_tr / np.sum(w_tr)
                    mu = np.sum(X_cv[tr] * w_tr_norm[:, None], axis=0)
                    sd = np.sqrt(np.sum(((X_cv[tr] - mu) ** 2) * w_tr_norm[:, None], axis=0))
                    sd = np.where(sd < 1e-6, 1.0, sd)
                    w_vec, b_val = fit_weighted_l2_logreg_dual((X_cv[tr] - mu) / sd, d_y[tr], w_tr, l2_reg=l2)
                    if np.any(te):
                        lg = ((X_cv[te] - mu) / sd) @ w_vec + b_val
                        cv_logits[te] = lg
                        cv_probs[te] = 1.0 / (1.0 + np.exp(-np.clip(lg, -30.0, 30.0)))
                    fold_models[f] = (mu, sd, w_vec, b_val)

                p_clip = np.clip(cv_probs, 1e-12, 1.0 - 1e-12)
                w_tot = float(np.sum(d_w))
                wbce = (
                    -float(np.sum(d_w * (d_y * np.log(p_clip) + (1.0 - d_y) * np.log(1.0 - p_clip))) / w_tot)
                    if w_tot > 0.0
                    else 0.0
                )
                if wbce < best_bce:
                    best_bce = wbce
                    preds = (cv_probs >= 0.5).astype(np.float64)
                    pos_m = d_y == 1.0
                    neg_m = d_y == 0.0
                    pos_acc = (
                        float(np.sum(d_w[pos_m] * (preds[pos_m] == 1.0)) / np.sum(d_w[pos_m]))
                        if np.any(pos_m) else 0.5
                    )
                    neg_acc = (
                        float(np.sum(d_w[neg_m] * (preds[neg_m] == 0.0)) / np.sum(d_w[neg_m]))
                        if np.any(neg_m) else 0.5
                    )
                    best_res = {
                        "layer": lid,
                        "best_l2": l2,
                        "bal_acc": 0.5 * (pos_acc + neg_acc),
                        "pos_acc": pos_acc,
                        "neg_acc": neg_acc,
                        "auroc": auroc(d_y, cv_logits, d_w),
                        "bce": wbce,
                    }
                    best_fold_models = fold_models

            assert best_res is not None and best_fold_models is not None
            per_rule_layer_metrics[cname].append(best_res)

            for f in range(eff_folds):
                m_f = block_folds == f
                if not np.any(m_f):
                    continue
                mu, sd, w_vec, b_val = best_fold_models[f]
                lg_f = ((np.asarray(X_layer[m_f, :], dtype=np.float64) - mu) / sd) @ w_vec + b_val
                oof_logits_all[lid, m_f, ci] = lg_f
                oof_probs_all[lid, m_f, ci] = 1.0 / (1.0 + np.exp(-np.clip(lg_f, -30.0, 30.0)))

    # Select canonical representative readout layer by lowest mean 10-rule 5-fold CV BCE (purely from rule CV!)
    # and also evaluate all layers for trial-level marginalization.
    mean_rule_bce_by_layer = [
        float(np.mean([per_rule_layer_metrics[c][l]["bce"] for c in concept_names]))
        for l in range(n_layers)
    ]
    cv_bce_best_layer = int(np.argmin(mean_rule_bce_by_layer))

    # Equal-feedback-density negation control ('not red' vs 'not green') when both concepts exist
    neg_vs_neg_auroc: list[float] | None = None
    if "not red" in c_to_idx and "not green" in c_to_idx:
        nr_idx = c_to_idx["not red"]
        ng_idx = c_to_idx["not green"]
        nr_trials = np.where((true_cidx == nr_idx) & (stated_cidx == nr_idx))[0]
        ng_trials = np.where((true_cidx == ng_idx) & (stated_cidx == ng_idx))[0]
        if len(nr_trials) > 0 and len(ng_trials) > 0:
            neg_vs_neg_auroc = []
            y_nn = np.concatenate([np.ones(len(nr_trials)), np.zeros(len(ng_trials))])
            for lid in range(n_layers):
                s_nn = np.concatenate([oof_logits_all[lid, nr_trials, nr_idx], oof_logits_all[lid, ng_trials, nr_idx]])
                neg_vs_neg_auroc.append(auroc(y_nn, s_nn))

    # Part 2A: Ambiguous post-feedback trials (Rank-1 vs Unstated Rank-2 vs Falsified under 3-param Bayes)
    rank_layer_curves: list[dict] = []
    if n_concepts >= 2:
        order_desc = np.argsort(-p_10_fit, axis=1)
        c_rank1 = order_desc[:, 0]
        c_rank2 = order_desc[:, 1]
        p_rank2 = p_10_fit[np.arange(n_rows), c_rank2]
        both_top2_cons = consistent[np.arange(n_rows), c_rank1] & consistent[np.arange(n_rows), c_rank2]
        has_falsified = np.any(~consistent, axis=1)
        ambig_unstated_mask = (
            (trials_arr >= 2) & both_top2_cons & (p_rank2 >= 0.10) & has_falsified & (stated_cidx != c_rank2)
        )
        ambig_stated_rank1_mask = ambig_unstated_mask & (stated_cidx == c_rank1)

        for lid in range(n_layers):
            P_L = oof_probs_all[lid]

            def _eval_ambig(mask: np.ndarray):
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    return {
                        "n": 0,
                        "mean_p_rank1": 0.5,
                        "mean_p_rank2": 0.5,
                        "mean_p_low_prob_live": 0.5,
                        "mean_p_low_prob_3p": 0.5,
                        "mean_p_falsified": 0.5,
                        "within_trial_auroc_rank2_vs_falsified": 0.5,
                    }
                r1_vals = P_L[idxs, c_rank1[idxs]]
                r2_vals = P_L[idxs, c_rank2[idxs]]
                false_means, low_live_means, low_3p_means, within_wins = [], [], [], []
                for ii in idxs:
                    f_scores = P_L[ii, ~consistent[ii]]
                    false_means.append(float(np.mean(f_scores)))
                    r1_c, r2_c, st_c = int(c_rank1[ii]), int(c_rank2[ii]), int(stated_cidx[ii])
                    ll_m = np.array([
                        c not in (r1_c, r2_c, st_c) and bool(consistent[ii, c]) and float(p_10_fit[ii, c]) <= 0.08
                        for c in range(n_concepts)
                    ], dtype=bool)
                    if np.any(ll_m):
                        low_live_means.append(float(np.mean(P_L[ii, ll_m])))
                    l3_m = np.array([
                        c not in (r1_c, r2_c, st_c) and (0.01 <= float(p_10_fit[ii, c]) <= 0.05)
                        for c in range(n_concepts)
                    ], dtype=bool)
                    if np.any(l3_m):
                        low_3p_means.append(float(np.mean(P_L[ii, l3_m])))
                    r2_s = P_L[ii, c_rank2[ii]]
                    within_wins.append(
                        float(np.mean((r2_s > f_scores).astype(np.float64) + 0.5 * (r2_s == f_scores).astype(np.float64)))
                    )
                return {
                    "n": int(len(idxs)),
                    "mean_p_rank1": float(np.mean(r1_vals)),
                    "mean_p_rank2": float(np.mean(r2_vals)),
                    "mean_p_low_prob_live": float(np.mean(low_live_means)) if low_live_means else None,
                    "mean_p_low_prob_3p": float(np.mean(low_3p_means)) if low_3p_means else None,
                    "mean_p_falsified": float(np.mean(false_means)),
                    "within_trial_auroc_rank2_vs_falsified": float(np.mean(within_wins)),
                }

            rank_layer_curves.append({
                "layer": lid,
                "all_unstated_rank2": _eval_ambig(ambig_unstated_mask),
                "committed_to_rank1": _eval_ambig(ambig_stated_rank1_mask),
            })

    # Part 2B: Controlled same-pair (Stated = A, Decode = B) Live vs. Falsified
    colors_vocab = {str(v).lower() for v in features.get("color", ())}
    shapes_vocab = {str(v).lower() for v in features.get("shape", ())}
    color_singletons = {ci for ci, c in enumerate(concept_names) if c in colors_vocab}
    shape_singletons = {ci for ci, c in enumerate(concept_names) if c in shapes_vocab}
    cur_obj_satisfies = rule_eval_cur > 0.5

    min_pair_n = 5 if n_rows >= 500 else 2
    valid_all_pairs = []
    valid_orth_pairs = []
    valid_orth_no_cur_pairs = []
    for a in range(n_concepts):
        for b in range(n_concepts):
            if a == b:
                continue
            base_m = (trials_arr >= 2) & (stated_cidx == a)
            live_idx = np.where(base_m & consistent[:, b])[0]
            dead_idx = np.where(base_m & (~consistent[:, b]))[0]
            if len(live_idx) >= min_pair_n and len(dead_idx) >= min_pair_n:
                valid_all_pairs.append((a, b, live_idx, dead_idx))
                is_orth = (a in color_singletons and b in shape_singletons) or (
                    a in shape_singletons and b in color_singletons
                )
                if is_orth:
                    valid_orth_pairs.append((a, b, live_idx, dead_idx))
                    live_nc = np.where(base_m & consistent[:, b] & (~cur_obj_satisfies[:, b]))[0]
                    dead_nc = np.where(base_m & (~consistent[:, b]) & (~cur_obj_satisfies[:, b]))[0]
                    if len(live_nc) >= max(2, min_pair_n - 2) and len(dead_nc) >= max(2, min_pair_n - 2):
                        valid_orth_no_cur_pairs.append((a, b, live_nc, dead_nc))

    same_pair_layer_curves = []
    for lid in range(n_layers):
        P_L = oof_probs_all[lid]
        Lg_L = oof_logits_all[lid]

        def _summarize_pairs(pair_list):
            if not pair_list:
                return {
                    "n_pairs": 0,
                    "mean_live_p": 0.5,
                    "mean_dead_p": 0.5,
                    "mean_prob_diff": 0.0,
                    "mean_within_pair_auroc": 0.5,
                }
            live_m, dead_m, diffs, aucs = [], [], [], []
            for _, b, l_idx, d_idx in pair_list:
                pl = P_L[l_idx, b]
                pd = P_L[d_idx, b]
                ll = Lg_L[l_idx, b]
                ld = Lg_L[d_idx, b]
                live_m.append(float(np.mean(pl)))
                dead_m.append(float(np.mean(pd)))
                diffs.append(float(np.mean(pl) - np.mean(pd)))
                aucs.append(
                    auroc(np.concatenate([np.ones(len(pl)), np.zeros(len(pd))]), np.concatenate([ll, ld]))
                )
            return {
                "n_pairs": len(pair_list),
                "mean_live_p": float(np.mean(live_m)),
                "mean_dead_p": float(np.mean(dead_m)),
                "mean_prob_diff": float(np.mean(diffs)),
                "mean_within_pair_auroc": float(np.mean(aucs)),
            }

        same_pair_layer_curves.append({
            "layer": lid,
            "all_pairs": _summarize_pairs(valid_all_pairs),
            "orth_singleton_pairs": _summarize_pairs(valid_orth_pairs),
            "orth_no_cur_obj_pairs": _summarize_pairs(valid_orth_no_cur_pairs),
        })

    if valid_orth_pairs:
        best_r2_layer = max(
            range(n_layers),
            key=lambda l: same_pair_layer_curves[l]["orth_singleton_pairs"]["mean_within_pair_auroc"],
        )
    elif valid_all_pairs:
        best_r2_layer = max(
            range(n_layers),
            key=lambda l: same_pair_layer_curves[l]["all_pairs"]["mean_within_pair_auroc"],
        )
    else:
        best_r2_layer = cv_bce_best_layer

    pair_details_best_layer = []
    for a, b, l_idx, d_idx in valid_all_pairs:
        pl = oof_probs_all[best_r2_layer, l_idx, b]
        pd = oof_probs_all[best_r2_layer, d_idx, b]
        ll = oof_logits_all[best_r2_layer, l_idx, b]
        ld = oof_logits_all[best_r2_layer, d_idx, b]
        is_orth = (a in color_singletons and b in shape_singletons) or (
            a in shape_singletons and b in color_singletons
        )
        pair_details_best_layer.append({
            "stated_rule": concept_names[a],
            "probed_second_rule": concept_names[b],
            "is_orthogonal_singleton": bool(is_orth),
            "n_live": int(len(l_idx)),
            "n_falsified": int(len(d_idx)),
            "mean_p_when_live": float(np.mean(pl)),
            "mean_p_when_falsified": float(np.mean(pd)),
            "prob_diff": float(np.mean(pl) - np.mean(pd)),
            "auroc": auroc(np.concatenate([np.ones(len(pl)), np.zeros(len(pd))]), np.concatenate([ll, ld])),
        })
    pair_details_best_layer.sort(key=lambda x: (-int(x["is_orthogonal_singleton"]), -x["auroc"]))

    # Part 2C: Graded calibration bins across unstated rules vs. Bayesian posterior
    atom_pat = "|".join(sorted((re.escape(v) for v in (colors_vocab | shapes_vocab)), key=len, reverse=True))
    rule_atoms = {
        ci: (set(re.findall(atom_pat, cname)) if atom_pat else set())
        for ci, cname in enumerate(concept_names)
    }
    falsified_base = np.full((n_layers, n_concepts, n_concepts), np.nan, dtype=np.float64)
    min_dead_base = 3 if n_rows >= 500 else 1
    for a in range(n_concepts):
        for b in range(n_concepts):
            if a == b:
                continue
            dead_m = (trials_arr >= 2) & (stated_cidx == a) & (~consistent[:, b])
            if np.sum(dead_m) >= min_dead_base:
                falsified_base[:, a, b] = np.mean(oof_probs_all[:, dead_m, b], axis=1)

    obs_i, obs_a, obs_b, obs_bayes_p, obs_is_cons, obs_is_orth = [], [], [], [], [], []
    for ii in np.where((trials_arr >= 2) & (stated_cidx >= 0))[0]:
        a = int(stated_cidx[ii])
        for b in range(n_concepts):
            if b == a or np.isnan(falsified_base[0, a, b]):
                continue
            obs_i.append(int(ii))
            obs_a.append(a)
            obs_b.append(b)
            obs_bayes_p.append(float(p_10_fit[ii, b]))
            obs_is_cons.append(bool(consistent[ii, b]))
            obs_is_orth.append(len(rule_atoms[a] & rule_atoms[b]) == 0)

    r2_bins_summary = []
    if obs_i:
        obs_i_arr = np.array(obs_i, dtype=np.int64)
        obs_a_arr = np.array(obs_a, dtype=np.int64)
        obs_b_arr = np.array(obs_b, dtype=np.int64)
        obs_bp_arr = np.array(obs_bayes_p, dtype=np.float64)
        obs_cons_arr = np.array(obs_is_cons, dtype=bool)
        obs_orth_arr = np.array(obs_is_orth, dtype=bool)

        raw_br = oof_probs_all[best_r2_layer, obs_i_arr, obs_b_arr]
        exc_br = raw_br - falsified_base[best_r2_layer, obs_a_arr, obs_b_arr]
        bin_specs = [
            ("Falsified", ~obs_cons_arr),
            ("(0, .08]", obs_cons_arr & (obs_bp_arr <= 0.08)),
            ("(.08, .15]", obs_cons_arr & (obs_bp_arr > 0.08) & (obs_bp_arr <= 0.15)),
            ("(.15, .30]", obs_cons_arr & (obs_bp_arr > 0.15) & (obs_bp_arr <= 0.30)),
            ("> .30", obs_cons_arr & (obs_bp_arr > 0.30)),
        ]
        for label, bm in bin_specs:
            if not np.any(bm):
                continue
            v_br = exc_br[bm]
            bm_orth = bm & obs_orth_arr
            v_orth = raw_br[bm_orth] if np.any(bm_orth) else np.array([np.nan])
            r2_bins_summary.append({
                "bin": label,
                "n_all": int(np.sum(bm)),
                "n_disjoint": int(np.sum(bm_orth)),
                "mean_bayes_p": float(np.mean(obs_bp_arr[bm])),
                "mean_excess_best_r2": float(np.mean(v_br)),
                "se_excess_best_r2": float(np.std(v_br) / math.sqrt(len(v_br))),
                "mean_raw_disjoint_best_r2": float(np.mean(v_orth)) if np.any(bm_orth) else None,
                "se_raw_disjoint_best_r2": float(np.std(v_orth) / math.sqrt(len(v_orth))) if np.any(bm_orth) else 0.0,
            })

    # Part 3: Trial-level P(yes) & 160-point learning curve marginalization across ALL layers
    eval_kw = {
        "truth": truth,
        "ans_yes": ans_yes,
        "valid": valid,
        "q_bayes_fit": q_fit_3p,
        "q_bayes_norm": q_norm_pm,
        "concepts_arr": concepts_arr,
        "trials_arr": trials_arr,
        "concept_names": concept_names,
        "max_trial": max_trial,
        "emp_160_full": emp_160_full,
    }

    baselines_eval = {
        "token_p_yes": (
            "Answer: token prob p_yes",
            "behavioral",
            0,
            tok_p_yes,
            evaluate_trial_predictions(tok_p_yes, **eval_kw),
        ),
        "binary_llm_answer": (
            "Binary LLM answer (yes=1, no=0)",
            "behavioral",
            0,
            ans_yes,
            evaluate_trial_predictions(ans_yes, **eval_kw),
        ),
        "single_stated_rule": (
            "Single stated scratchpad Rule:",
            "behavioral",
            0,
            q_stated_rule,
            evaluate_trial_predictions(q_stated_rule, **eval_kw),
        ),
        "bayes_norm_pm": (
            "Normative Bayes (prob-matching)",
            "bayesian",
            0,
            q_norm_pm,
            evaluate_trial_predictions(q_norm_pm, **eval_kw),
        ),
        "bayes_fit_3p": (
            "3-param Rational Rules Bayes",
            "bayesian",
            3,
            q_fit_3p,
            evaluate_trial_predictions(q_fit_3p, **eval_kw),
        ),
    }

    superposed_by_layer: list[dict] = []
    top1_by_layer: list[dict] = []
    q_super_all = np.zeros((n_layers, n_rows), dtype=np.float64)
    q_top1_all = np.zeros((n_layers, n_rows), dtype=np.float64)

    for lid in range(n_layers):
        q_sup = superposed_marginalize_q_yes(oof_probs_all[lid], rule_eval_cur)
        q_t1 = top1_marginalize_q_yes(oof_probs_all[lid], rule_eval_cur)
        q_super_all[lid] = q_sup
        q_top1_all[lid] = q_t1
        superposed_by_layer.append({"layer": lid, **evaluate_trial_predictions(q_sup, **eval_kw)})
        top1_by_layer.append({"layer": lid, **evaluate_trial_predictions(q_t1, **eval_kw)})

    # Representative readout layer: pick layer maximizing zero-param LLM binary choice likelihood
    # strictly via Leave-One-Concept-Out CV (so layer choice itself is verified out-of-concept),
    # and record the global zero-param optimal layer (L22 on gemma4:26b).
    rep_layer = int(
        min(range(n_layers), key=lambda l: superposed_by_layer[l]["llm_choice_mean_nll"])
    )
    # Also compute strictly out-of-concept layer + (gamma, tau_0) LOCO calibration!
    q_loco_layer_only = np.zeros(n_rows, dtype=np.float64)
    if n_concepts >= 2:
        for held_out in concept_names:
            tr_m = (concepts_arr != held_out) & valid
            te_m = concepts_arr == held_out
            best_l_tr = min(
                range(n_layers),
                key=lambda l: -float(
                    np.mean(
                        ans_yes[tr_m] * np.log(np.clip(q_super_all[l, tr_m], 1e-6, 1.0 - 1e-6))
                        + (1.0 - ans_yes[tr_m]) * np.log(1.0 - np.clip(q_super_all[l, tr_m], 1e-6, 1.0 - 1e-6))
                    )
                ),
            )
            q_loco_layer_only[te_m] = q_super_all[best_l_tr, te_m]
    else:
        q_loco_layer_only = q_super_all[rep_layer].copy()

    q_rep_loco_cal, loco_cal_params = loco_calibrate_superposed_q_yes(
        oof_probs_all[rep_layer], rule_eval_cur, ans_yes, valid, concepts_arr, concept_names
    )
    rep_loco_cal_eval = evaluate_trial_predictions(q_rep_loco_cal, **eval_kw)
    loco_layer_only_eval = evaluate_trial_predictions(q_loco_layer_only, **eval_kw)

    # Assemble CSV rows for per-rule decoders
    rule_csv_rows = []
    group_by_concept = {
        cname: next(r["complexity_group"] for r in meta_rows if r["concept"] == cname)
        for cname in concept_names
    }
    for ci, cname in enumerate(concept_names):
        ms = per_rule_layer_metrics[cname]
        best_l = max(range(n_layers), key=lambda l: (round(ms[l]["auroc"], 5), -ms[l]["bce"]))
        l90_list = [l for l in range(n_layers) if ms[l]["auroc"] >= 0.90]
        d_y = cv_datasets[ci][1]
        rule_csv_rows.append({
            "concept": cname,
            "complexity_group": group_by_concept[cname],
            "n_pos_trials": int(np.sum(d_y == 1.0)),
            "n_neg_trials": int(np.sum(d_y == 0.0)),
            "l00_auroc": ms[0]["auroc"],
            "l90_layer": l90_list[0] if l90_list else None,
            "peak_layer": best_l,
            "peak_bal_acc": ms[best_l]["bal_acc"],
            "peak_auroc": ms[best_l]["auroc"],
            "peak_bce": ms[best_l]["bce"],
            "rep_layer": rep_layer,
            "rep_bal_acc": ms[rep_layer]["bal_acc"],
            "rep_auroc": ms[rep_layer]["auroc"],
            "final_layer": n_layers - 1,
            "final_bal_acc": ms[-1]["bal_acc"],
            "final_auroc": ms[-1]["auroc"],
        })

    # Assemble CSV rows for trial-level predictions
    trial_pred_csv_rows = []
    for pid, (plabel, pkind, k_p, _, peval) in baselines_eval.items():
        trial_pred_csv_rows.append({
            "predictor_id": pid,
            "predictor_label": plabel,
            "kind": pkind,
            "layer": "",
            "k_params": k_p,
            "n_trials": n_rows,
            **{k: peval[k] for k in TRIAL_PRED_CSV_COLUMNS[6:]},
        })
    for lid in range(n_layers):
        se = superposed_by_layer[lid]
        trial_pred_csv_rows.append({
            "predictor_id": f"probe_superposed_L{lid:02d}",
            "predictor_label": f"Superposed 10-rule probe (L{lid:02d})",
            "kind": "probe_superposed",
            "layer": lid,
            "k_params": 0,
            "n_trials": n_rows,
            **{k: se[k] for k in TRIAL_PRED_CSV_COLUMNS[6:]},
        })
        te = top1_by_layer[lid]
        trial_pred_csv_rows.append({
            "predictor_id": f"probe_top1_L{lid:02d}",
            "predictor_label": f"Top-1 winner-take-all probe (L{lid:02d})",
            "kind": "probe_top1",
            "layer": lid,
            "k_params": 0,
            "n_trials": n_rows,
            **{k: te[k] for k in TRIAL_PRED_CSV_COLUMNS[6:]},
        })
    trial_pred_csv_rows.append({
        "predictor_id": "probe_superposed_loco_layer",
        "predictor_label": "Superposed 10-rule probe (LOCO-selected layer, 0 params)",
        "kind": "probe_loco",
        "layer": "LOCO",
        "k_params": 0,
        "n_trials": n_rows,
        **{k: loco_layer_only_eval[k] for k in TRIAL_PRED_CSV_COLUMNS[6:]},
    })
    trial_pred_csv_rows.append({
        "predictor_id": f"probe_superposed_L{rep_layer:02d}_loco_cal",
        "predictor_label": f"LOCO-calibrated superposed probe (L{rep_layer:02d}, gamma+tau_0)",
        "kind": "probe_loco_cal",
        "layer": rep_layer,
        "k_params": 2,
        "n_trials": n_rows,
        **{k: rep_loco_cal_eval[k] for k in TRIAL_PRED_CSV_COLUMNS[6:]},
    })

    # Reliability curves: bin each predictor's own predicted P(yes) vs. empirical LLM 'yes' rate
    def _reliability_curve(q_vec: np.ndarray, n_bins: int = 8) -> list[dict]:
        edges_r = np.linspace(0.0, 1.0, n_bins + 1)
        pts_r = []
        qv = q_vec[valid]
        yv = ans_yes[valid]
        for bi in range(n_bins):
            lo, hi = edges_r[bi], edges_r[bi + 1]
            bm = (qv >= lo) & (qv <= hi if bi == n_bins - 1 else qv < hi)
            if int(np.sum(bm)) < 5:
                continue
            nb = int(np.sum(bm))
            pts_r.append({
                "pred_mean": float(np.mean(qv[bm])),
                "emp_yes": float(np.mean(yv[bm])),
                "emp_se": float(np.std(yv[bm]) / math.sqrt(nb)),
                "n": nb,
            })
        return pts_r

    q_sup_rep = q_super_all[rep_layer]
    q_top1_rep = q_top1_all[rep_layer]
    cal_bins = {
        "probe_super": _reliability_curve(q_sup_rep, n_bins=8),
        "probe_top1": _reliability_curve(q_top1_rep, n_bins=8),
        "stated_rule": _reliability_curve(q_stated_rule, n_bins=5),
        "bayes_fit_3p": _reliability_curve(q_fit_3p, n_bins=8),
    }

    return {
        "llm_model": primary_llm,
        "n_trials": n_rows,
        "n_blocks": n_blocks_total,
        "n_layers": n_layers,
        "n_embd": n_embd,
        "n_folds": eff_folds,
        "concept_names": concept_names,
        "group_by_concept": group_by_concept,
        "max_trial": max_trial,
        "fitted_params": fitted_params,
        "cv_bce_best_layer": cv_bce_best_layer,
        "rep_layer": rep_layer,
        "best_r2_layer": best_r2_layer,
        "per_rule_layer_metrics": per_rule_layer_metrics,
        "neg_vs_neg_auroc": neg_vs_neg_auroc,
        "rank_layer_curves": rank_layer_curves,
        "same_pair_layer_curves": same_pair_layer_curves,
        "pair_details_best_layer": pair_details_best_layer,
        "r2_bins_summary": r2_bins_summary,
        "baselines_eval": {k: v[4] for k, v in baselines_eval.items()},
        "superposed_by_layer": superposed_by_layer,
        "top1_by_layer": top1_by_layer,
        "loco_layer_only_eval": loco_layer_only_eval,
        "rep_loco_cal_eval": rep_loco_cal_eval,
        "loco_cal_params": loco_cal_params,
        "cal_bins": cal_bins,
        "rule_csv_rows": rule_csv_rows,
        "trial_pred_csv_rows": trial_pred_csv_rows,
    }


# ---------------------------------------------------------------------------
# Publication figures (Figures 5 & 6) and Markdown tables
# ---------------------------------------------------------------------------

def figure_layer_rule_probes(bundle: dict):
    """Figure 5: 2x2 summary of per-layer binary rule decoders & unstated 2nd-best superposition."""
    n_layers = bundle["n_layers"]
    layers = np.arange(n_layers)
    concept_names = bundle["concept_names"]
    group_by_concept = bundle["group_by_concept"]
    per_rule = bundle["per_rule_layer_metrics"]
    rank_curves = bundle["rank_layer_curves"]
    pair_curves = bundle["same_pair_layer_curves"]
    r2_bins = bundle["r2_bins_summary"]
    best_r2_layer = bundle["best_r2_layer"]

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.15))

        # Panel A: Binary [layer -> 0/1 rule] decoders across layers
        ax = axes[0, 0]
        present_groups = [
            g for g in COMPLEXITY_GROUPS
            if any(group_by_concept[c] == g for c in concept_names)
        ]
        for grp in present_groups:
            grp_cs = [c for c in concept_names if group_by_concept[c] == grp]
            col = COMPLEXITY_COLORS.get(grp, SERIES_FALLBACK)
            for c in grp_cs:
                ax.plot(layers, [m["auroc"] for m in per_rule[c]], "-", color=col, lw=0.65, alpha=0.26)
            mean_c = np.mean([[m["auroc"] for m in per_rule[c]] for c in grp_cs], axis=0)
            ax.plot(layers, mean_c, "-o", color=col, lw=1.25, ms=2.1, label=grp)

        if bundle["neg_vs_neg_auroc"] is not None:
            ax.plot(
                layers, bundle["neg_vs_neg_auroc"], "--", color="#0e6251", lw=1.05,
                label="not red vs. not green",
            )
        ax.axhline(0.5, color=MUTED, ls=":", lw=0.75)
        ax.set_title("A. Binary [layer \u2192 0/1] rule decoders", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel(f"Residual-stream layer (0..{n_layers - 1})")
        ax.set_ylabel("5-fold block-CV AUROC")
        ax.set_ylim(0.43, 1.03)
        ax.grid(True, color=GRID, lw=0.45)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=6.2)

        # Panel B: Rank-1 vs. Unstated 2nd-best vs. Falsified rules across layers
        ax = axes[0, 1]
        if rank_curves:
            r1_c = [rc["committed_to_rank1"]["mean_p_rank1"] for rc in rank_curves]
            r2_all = [rc["all_unstated_rank2"]["mean_p_rank2"] for rc in rank_curves]
            r2_c = [rc["committed_to_rank1"]["mean_p_rank2"] for rc in rank_curves]
            rf_c = [rc["committed_to_rank1"]["mean_p_falsified"] for rc in rank_curves]
            n_a1 = rank_curves[0]["all_unstated_rank2"]["n"]
            n_a2 = rank_curves[0]["committed_to_rank1"]["n"]

            ax.plot(layers, r1_c, "-o", color="#2a78d6", lw=1.3, ms=2.2, label="Stated Rank-1 rule")
            ax.plot(
                layers, r2_all, "-s", color="#eb6834", lw=1.3, ms=2.2,
                label=f"Unstated 2nd-best (all, n={n_a1})",
            )
            ax.plot(
                layers, r2_c, "--^", color="#eda100", lw=1.15, ms=2.2,
                label=f"Unstated 2nd-best (c\u2081 stated, n={n_a2})",
            )
            if rank_curves[0]["committed_to_rank1"].get("mean_p_low_prob_live") is not None:
                rlow_c = [rc["committed_to_rank1"]["mean_p_low_prob_live"] for rc in rank_curves]
                ax.plot(
                    layers, rlow_c, ":d", color="#6250d6", lw=1.2, ms=2.2,
                    label="Unstated low-prob live (3p P\u2264.08)",
                )
            ax.plot(layers, rf_c, "-.", color=MUTED, lw=1.1, label="Falsified rules on same trials")
        ax.set_title("B. Best, 2nd-best, low-prob live & falsified rules", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel(f"Residual-stream layer (0..{n_layers - 1})")
        ax.set_ylabel("Out-of-block decoded probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color=GRID, lw=0.45)
        ax.legend(
            loc="upper left", bbox_to_anchor=(0.01, 0.99),
            frameon=True, facecolor="white", edgecolor=GRID, fontsize=5.6,
        )

        # Panel C: Controlled Same-Pair (Stated=A, Decode=B) Live vs. Falsified
        ax = axes[1, 0]
        auc_all = [sp["all_pairs"]["mean_within_pair_auroc"] for sp in pair_curves]
        n_all_p = pair_curves[0]["all_pairs"]["n_pairs"] if pair_curves else 0
        n_orth_p = pair_curves[0]["orth_singleton_pairs"]["n_pairs"] if pair_curves else 0
        if n_orth_p > 0:
            auc_orth = [sp["orth_singleton_pairs"]["mean_within_pair_auroc"] for sp in pair_curves]
            ax.plot(
                layers, auc_orth, "-o", color="#e34948", lw=1.35, ms=2.3,
                label=f"Orthogonal color/shape ({n_orth_p} pairs)",
            )
        if pair_curves and pair_curves[0]["orth_no_cur_obj_pairs"]["n_pairs"] > 0:
            auc_nc = [sp["orth_no_cur_obj_pairs"]["mean_within_pair_auroc"] for sp in pair_curves]
            ax.plot(layers, auc_nc, "--", color="#922b21", lw=1.05, label="Orthogonal, cur. obj lacks B")
        if n_all_p > 0:
            ax.plot(
                layers, auc_all, "-s", color="#1baf7a", lw=1.2, ms=2.1,
                label=f"All (stated A, decode B) ({n_all_p} pairs)",
            )
        ax.axhline(0.5, color=MUTED, ls=":", lw=0.75, label="Chance (0.50)")
        ax.set_title("C. Controlled same-pair live vs. falsified B", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel(f"Residual-stream layer (0..{n_layers - 1})")
        ax.set_ylabel("Within-pair AUROC")
        ax.set_ylim(0.43, 0.94)
        ax.grid(True, color=GRID, lw=0.45)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=6.0)

        # Panel D: Graded Readout of Unstated Rules vs. Bayesian Posterior
        ax = axes[1, 1]
        if r2_bins:
            bx = np.arange(len(r2_bins))
            blabels = [b["bin"] for b in r2_bins]
            y_exc = [b["mean_excess_best_r2"] for b in r2_bins]
            e_exc = [1.96 * b["se_excess_best_r2"] for b in r2_bins]
            y_orth = [
                b["mean_raw_disjoint_best_r2"] if b["mean_raw_disjoint_best_r2"] is not None else np.nan
                for b in r2_bins
            ]
            e_orth = [1.96 * b["se_raw_disjoint_best_r2"] for b in r2_bins]
            y_bayes = [b["mean_bayes_p"] for b in r2_bins]

            ax.axhline(0.0, color=MUTED, ls=":", lw=0.75)
            ax.errorbar(
                bx - 0.06, y_exc, yerr=e_exc, fmt="-o", color="#e34948", lw=1.3, ms=3.4, capsize=2.2,
                label=f"Excess \u0394p above falsified (L{best_r2_layer:02d})",
            )
            if not all(np.isnan(y_orth)):
                ax.errorbar(
                    bx + 0.06, y_orth, yerr=e_orth, fmt="--s", color="#2a78d6", lw=1.2, ms=3.2, capsize=2.2,
                    label=f"Raw p on disjoint features (L{best_r2_layer:02d})",
                )
            ax.plot(bx, y_bayes, ":^", color="#1baf7a", lw=1.2, ms=3.4, label="3-param Bayesian P(B | D_<t)")
            ax.set_xticks(bx)
            ax.set_xticklabels(blabels, rotation=16, ha="right")
        ax.set_title("D. Graded unstated-rule readout vs. 3p Bayes", loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel("3-param Bayesian posterior bin on unstated rule B")
        ax.set_ylabel("Probability / excess \u0394p (\u00b195% CI)")
        ax.grid(True, color=GRID, lw=0.45)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID, fontsize=5.9)

        figure.tight_layout(pad=0.55)
    return figure


SERIES_FALLBACK = "#2a78d6"


def figure_probe_trial_marginalization(config, rows: list[dict], bundle: dict):
    """Figure 6: 2x5 concept learning curves from superposed L22 probe + 3-panel mechanism comparison."""
    llm = bundle["llm_model"]
    m_rows = [r for r in rows if r["model"] == llm] if rows else []
    emp_curves = (
        [r for r in learning_curves(m_rows, config=config) if r["scope_type"] == "concept"]
        if m_rows else []
    )
    by_concept_emp: dict[str, list[dict]] = defaultdict(list)
    for r in emp_curves:
        by_concept_emp[r["scope"]].append(r)

    concepts = bundle["concept_names"]
    group_of = bundle["group_by_concept"]
    max_trial = bundle["max_trial"]
    if emp_curves:
        _, _, boundaries, xticks = _curve_layout(emp_curves)
    else:
        boundaries = []
        xticks = sorted({1, max_trial})

    rep_l = bundle["rep_layer"]
    sup_rep = bundle["superposed_by_layer"][rep_l]
    stated_ev = bundle["baselines_eval"]["single_stated_rule"]
    bayes_fit_ev = bundle["baselines_eval"]["bayes_fit_3p"]
    tok_ev = bundle["baselines_eval"]["token_p_yes"]

    llm_color = "#2a78d6"
    probe_color = "#e34948"
    stated_color = "#7a7874"
    bayes_color = "#1baf7a"

    n_concepts = len(concepts)
    ncols_top = 5 if n_concepts >= 5 else max(1, n_concepts)
    nrows_top = -(-n_concepts // ncols_top)

    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(FULL_WIDTH, 5.9))
        outer = GridSpec(2, 1, figure=figure, height_ratios=[1.30, 1.06], hspace=0.39)
        top_gs = outer[0].subgridspec(nrows_top, ncols_top, wspace=0.18, hspace=0.34)
        bot_gs = outer[1].subgridspec(1, 3, wspace=0.39)

        trials = list(range(1, max_trial + 1))
        top_axes = [
            [figure.add_subplot(top_gs[r, c]) for c in range(ncols_top)]
            for r in range(nrows_top)
        ]

        for idx, concept in enumerate(concepts):
            r_idx, c_idx = divmod(idx, ncols_top)
            ax = top_axes[r_idx][c_idx]
            for boundary in boundaries:
                ax.axvline(boundary, color=MUTED, linestyle="--", linewidth=0.5, alpha=0.45, zorder=1)
            ax.axhline(0.5, color=GRID, linestyle=":", linewidth=0.6, zorder=1)

            pts = sorted(by_concept_emp.get(concept, []), key=lambda x: x["trial"])
            if pts:
                emp_y = [p["correct"] for p in pts]
                lows = [p["low"] for p in pts]
                highs = [p["high"] for p in pts]
                ax.plot(trials, emp_y, color=llm_color, linewidth=1.1, zorder=3)
                if all(lo is not None and hi is not None for lo, hi in zip(lows, highs)):
                    ax.fill_between(trials, lows, highs, color=llm_color, alpha=0.15, linewidth=0, zorder=1.5)
            else:
                emp_y = None

            ax.plot(trials, stated_ev["concept_curves"][concept], color=stated_color, ls=":", lw=0.95, zorder=2)
            ax.plot(trials, bayes_fit_ev["concept_curves"][concept], color=bayes_color, ls="--", lw=1.0, zorder=4)
            ax.plot(trials, sup_rep["concept_curves"][concept], color=probe_color, ls="-", lw=1.2, zorder=5)

            if emp_y is not None:
                r_c = pearson_r(emp_y, sup_rep["concept_curves"][concept])
                if r_c is not None:
                    ax.text(
                        0.97, 0.06, f"r={r_c:.2f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5.9, color=INK,
                    )
            ax.set_title(concept, loc="left", fontsize=7.3, pad=2.0)
            ax.set_xlim(1, max(2, max_trial))
            ax.set_ylim(0.0, 1.03)
            ax.set_xticks(xticks)
            ax.set_yticks([0.0, 0.5, 1.0])
            if r_idx < nrows_top - 1:
                ax.tick_params(labelbottom=False)
            if c_idx > 0:
                ax.tick_params(labelleft=False)
            if c_idx == 0:
                ax.set_ylabel("Prop. correct", fontsize=7.3)
            if r_idx == nrows_top - 1 and c_idx == ncols_top // 2:
                ax.set_xlabel(f"Trial (1\u2013{max_trial}, unique stimuli)", fontsize=7.3)
            ax.grid(True, axis="y", color=GRID, linewidth=0.45)
            ax.set_axisbelow(True)

        for extra_idx in range(len(concepts), nrows_top * ncols_top):
            r_idx, c_idx = divmod(extra_idx, ncols_top)
            top_axes[r_idx][c_idx].set_visible(False)

        top_legend = [
            Line2D([0], [0], color=llm_color, lw=1.2, label=f"{llm} empirical"),
            Line2D(
                [0], [0], color=probe_color, lw=1.3, ls="-",
                label=f"Superposed 10-rule probe (L{rep_l:02d}, 0 params)",
            ),
            Line2D([0], [0], color=stated_color, lw=1.0, ls=":", label="Single stated Rule:"),
            Line2D([0], [0], color=bayes_color, lw=1.05, ls="--", label="3-param Rational Rules"),
        ]
        figure.legend(
            handles=top_legend, frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 0.995), ncols=2, handlelength=1.8, columnspacing=1.5, fontsize=6.9,
        )

        # Bottom-left: Layer trajectory of trial-by-trial correlation with Bayesian P(yes | x_t, D_<t)
        ax_lyr = figure.add_subplot(bot_gs[0])
        layers = np.arange(bundle["n_layers"])
        r_sup_l = [s["r_trial_fit_bayes"] for s in bundle["superposed_by_layer"]]
        r_top1_l = [s["r_trial_fit_bayes"] for s in bundle["top1_by_layer"]]

        ax_lyr.plot(layers, r_sup_l, "-o", color=probe_color, lw=1.35, ms=2.1, label="Superposed 10-rule")
        ax_lyr.plot(layers, r_top1_l, "--", color="#eda100", lw=1.15, label="Top-1 probe only")
        ax_lyr.axhline(
            stated_ev["r_trial_fit_bayes"], color=stated_color, ls=":", lw=0.95,
            label=f"Stated Rule ({stated_ev['r_trial_fit_bayes']:.2f})",
        )
        ax_lyr.axhline(
            tok_ev["r_trial_fit_bayes"], color="#6250d6", ls="-.", lw=0.9,
            label=f"Token p_yes ({tok_ev['r_trial_fit_bayes']:.2f})",
        )
        ax_lyr.axvline(rep_l, color=probe_color, ls=":", lw=0.7, alpha=0.6)
        ax_lyr.set_title("A. Trial P(yes) alignment by layer", loc="left", fontsize=7.5, fontweight="bold")
        ax_lyr.set_xlabel(f"Layer (0..{bundle['n_layers'] - 1})")
        ax_lyr.set_ylabel("Trial Pearson r vs. fitted Bayes")
        ax_lyr.set_ylim(min(0.0, min(r_sup_l) - 0.04), 0.90)
        ax_lyr.grid(True, color=GRID, lw=0.45)
        ax_lyr.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=5.7)

        # Bottom-middle: 160-point (concept x trial) scatter (Superposed L22 probe vs. empirical LLM)
        ax_sc = figure.add_subplot(bot_gs[1])
        ax_sc.plot([0.15, 1.0], [0.15, 1.0], color=MUTED, ls="--", lw=0.75, zorder=1)
        for idx_g, cgroup in enumerate(COMPLEXITY_GROUPS):
            col = complexity_color(cgroup, idx_g)
            xs_g, ys_g = [], []
            for concept in concepts:
                if group_of.get(concept) != cgroup:
                    continue
                pts = sorted(by_concept_emp.get(concept, []), key=lambda x: x["trial"])
                if not pts:
                    continue
                xs_g.extend(sup_rep["concept_curves"][concept])
                ys_g.extend(p["correct"] for p in pts)
            if xs_g:
                ax_sc.scatter(xs_g, ys_g, s=11, color=col, alpha=0.60, edgecolors="none", label=cgroup, zorder=2)

        if by_concept_emp:
            overall_emp = [
                sum(sorted(by_concept_emp[c], key=lambda x: x["trial"])[t]["correct"] for c in concepts) / len(concepts)
                for t in range(max_trial)
            ]
            overall_pred = [
                sum(sup_rep["concept_curves"][c][t] for c in concepts) / len(concepts)
                for t in range(max_trial)
            ]
            ax_sc.scatter(
                overall_pred, overall_emp, s=18, color=INK, alpha=0.88,
                edgecolors="white", linewidths=0.4, zorder=3, label="Trial mean",
            )
        ax_sc.set_xlim(0.15, 1.03)
        ax_sc.set_ylim(0.15, 1.03)
        ax_sc.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax_sc.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_sc.set_title(f"B. 160-pt curves (L{rep_l:02d}, 0 params)", loc="left", fontsize=7.5, fontweight="bold")
        ax_sc.set_xlabel("Superposed probe pred. prop. correct")
        ax_sc.set_ylabel(f"{llm} prop. correct")
        ax_sc.grid(True, color=GRID, lw=0.45)
        ax_sc.text(
            0.04, 0.94,
            f"r = {sup_rep['full_160_r']:.3f}\nRMSE = {sup_rep['full_160_rmse']:.3f}",
            transform=ax_sc.transAxes, ha="left", va="top", fontsize=6.5, color=INK,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
        )

        # Bottom-right: Reliability diagram (predicted P(yes) vs. empirical LLM 'yes' rate)
        ax_cal = figure.add_subplot(bot_gs[2])
        ax_cal.plot([0.0, 1.0], [0.0, 1.0], color=MUTED, ls="--", lw=0.75, zorder=1)
        cbins = bundle["cal_bins"]
        if isinstance(cbins, dict):
            top1_nll = bundle["top1_by_layer"][rep_l]["llm_choice_mean_nll"]
            for key_c, fmt_c, col_c, lw_c, lbl_c, z_c in (
                ("stated_rule", ":^", stated_color, 1.05, f"Stated ({stated_ev['llm_choice_mean_nll']:.2f})", 2),
                ("probe_top1", "--o", "#eda100", 1.1, f"Top-1 ({top1_nll:.2f})", 3),
                ("bayes_fit_3p", "--", bayes_color, 1.05, f"3p Bayes ({bayes_fit_ev['llm_choice_mean_nll']:.2f})", 4),
                ("probe_super", "-s", probe_color, 1.3, f"Super L{rep_l:02d} ({sup_rep['llm_choice_mean_nll']:.2f})", 5),
            ):
                pts_r = cbins.get(key_c, [])
                if not pts_r:
                    continue
                rx = [p["pred_mean"] for p in pts_r]
                ry = [p["emp_yes"] for p in pts_r]
                if key_c == "probe_super":
                    re_y = [1.96 * p["emp_se"] for p in pts_r]
                    ax_cal.errorbar(
                        rx, ry, yerr=re_y, fmt=fmt_c, color=col_c, lw=lw_c,
                        ms=2.9, capsize=1.8, label=lbl_c, zorder=z_c,
                    )
                else:
                    ax_cal.plot(rx, ry, fmt_c, color=col_c, lw=lw_c, ms=2.6, label=lbl_c, zorder=z_c)
        ax_cal.set_xlim(-0.03, 1.03)
        ax_cal.set_ylim(-0.03, 1.03)
        ax_cal.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_cal.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_cal.set_title("C. Choice reliability (NLL)", loc="left", fontsize=7.5, fontweight="bold")
        ax_cal.set_xlabel("Predicted P(yes)")
        ax_cal.set_ylabel(f"{llm} empirical yes rate")
        ax_cal.grid(True, color=GRID, lw=0.45)
        ax_cal.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=5.5)

        figure.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.99)
    return figure


def layer_rule_probes_markdown(bundle: dict) -> str:
    """Publication-ready Markdown summary of the 10 binary rule decoders and second-best controls."""
    rep_l = bundle["rep_layer"]
    fin_l = bundle["n_layers"] - 1
    lines = [
        "## Per-Rule Independent Binary `[Layer -> 0/1]` Decoders (5-Fold Block CV)",
        "",
        f"| Target Concept | Complexity | `n_pos` | `L00` AUROC | `L >= 0.90` | Peak Layer | Peak BalAcc | Peak AUROC | `L{rep_l:02d}` AUROC | `L{fin_l:02d}` AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in bundle["rule_csv_rows"]:
        l90_s = f"L{r['l90_layer']:02d}" if r["l90_layer"] is not None else "\u2014"
        lines.append(
            f"| `{r['concept']}` | {r['complexity_group']} | {r['n_pos_trials']} | "
            f"{r['l00_auroc']:.4f} | {l90_s} | L{r['peak_layer']:02d} | "
            f"{100 * r['peak_bal_acc']:.2f}% | **{r['peak_auroc']:.4f}** | "
            f"{r['rep_auroc']:.4f} | {r['final_auroc']:.4f} |"
        )

    if bundle["pair_details_best_layer"]:
        best_r2 = bundle["best_r2_layer"]
        lines.extend([
            "",
            f"## Controlled Same-Pair Unstated Second-Best Rule Test at `L{best_r2:02d}`",
            "",
            "| Orthogonal | Stated Rule (`A`) | Probed 2nd Rule (`B`) | `n_live` | `n_falsified` | Mean `p(B)` Live | Mean `p(B)` Falsified | `Delta p` | Within-Pair AUROC |",
            "| :---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for p in bundle["pair_details_best_layer"][:16]:
            orth = "Yes" if p["is_orthogonal_singleton"] else "No"
            lines.append(
                f"| {orth} | `{p['stated_rule']}` | `{p['probed_second_rule']}` | "
                f"{p['n_live']} | {p['n_falsified']} | {p['mean_p_when_live']:.3f} | "
                f"{p['mean_p_when_falsified']:.3f} | {p['prob_diff']:+.3f} | **{p['auroc']:.4f}** |"
            )

    lines.extend([
        "",
        f"Evaluated strictly out-of-block across **{bundle['n_trials']:,}** pure-induction trials "
        f"(`{bundle['n_blocks']}` independent blocks, `include_rule=False`) extracted from the "
        f"`{bundle['llm_model']}` residual stream (`d_model = {bundle['n_embd']}`) at `Rule:` before any scratchpad text is generated.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def probe_trial_predictions_markdown(bundle: dict) -> str:
    """Publication-ready Markdown table comparing superposed probe marginalization vs. baselines."""
    rep_l = bundle["rep_layer"]
    n_layers = bundle["n_layers"]
    key_layers = sorted({
        0,
        min(10, n_layers - 1),
        min(16, n_layers - 1),
        min(19, n_layers - 1),
        min(22, n_layers - 1),
        rep_l,
        n_layers - 1,
    })

    by_id = {r["predictor_id"]: r for r in bundle["trial_pred_csv_rows"]}
    ordered_ids = [
        "token_p_yes",
        "binary_llm_answer",
        "single_stated_rule",
        f"probe_top1_L{rep_l:02d}",
        *[f"probe_superposed_L{l:02d}" for l in key_layers],
        "probe_superposed_loco_layer",
        f"probe_superposed_L{rep_l:02d}_loco_cal",
        "bayes_norm_pm",
        "bayes_fit_3p",
    ]
    seen = set()
    rows_show = []
    for pid in ordered_ids:
        if pid in by_id and pid not in seen:
            seen.add(pid)
            rows_show.append(by_id[pid])

    lines = [
        f"## Trial-by-Trial `P(yes)` & 160-Point Learning-Curve Predictions (`n = {bundle['n_trials']:,}` Trials)",
        "",
        "| Predictor | Params (`k`) | Trial `r` vs. Fit Bayes | Trial RMSE vs. Fit Bayes | LLM Choice Mean NLL | LLM Choice Agree | Matched 160-pt `r` (RMSE) | Full 160-pt `r` (RMSE) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows_show:
        is_bold = r["predictor_id"] in (
            f"probe_superposed_L{rep_l:02d}",
            "probe_superposed_loco_layer",
            f"probe_superposed_L{rep_l:02d}_loco_cal",
        )
        lbl = f"**{r['predictor_label']}**" if is_bold else r["predictor_label"]
        lines.append(
            f"| {lbl} | {r['k_params']} | {r['r_trial_fit_bayes']:.4f} | "
            f"{r['rmse_trial_fit_bayes']:.4f} | {r['llm_choice_mean_nll']:.4f} | "
            f"{100 * r['llm_choice_agreement']:.1f}% | "
            f"{r['sub_160_r']:.4f} ({r['sub_160_rmse']:.4f}) | "
            f"{r['full_160_r']:.4f} ({r['full_160_rmse']:.4f}) |"
        )

    lines.extend([
        "",
        "Zero-parameter superposed marginalization combines the 10 independent out-of-block binary rule decoders via "
        "`q_yes(x_t) = (sum_c p_c(h_t) c(x_t) + 0.5 (1 - max_c p_c(h_t))) / (sum_c p_c(h_t) + (1 - max_c p_c(h_t)))` "
        "before any `Rule:` or `Answer:` token is generated. Coin-flip choice NLL = `0.6931`.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def make_all_probes(
    config,
    input_pairs: list[tuple[str | Path, str | Path]],
    *,
    rows: list[dict] | None = None,
    output_dir: str | Path | None = None,
    n_embd: int = 2816,
    n_folds: int = 5,
) -> tuple[list[Path], dict]:
    """Run ``analyze_probes`` and write Figures 5 & 6 plus CSV/Markdown tables."""
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
        res_path = config.resolve(config.run["results_file"])
        rows = read_rows(res_path) if res_path.exists() else []

    bundle = analyze_probes(
        config, input_pairs, full_rows=rows, n_embd=n_embd, n_folds=n_folds
    )

    written: list[Path] = []
    written += save(figure_layer_rule_probes(bundle), figures_dir, "fig5_layer_rule_probes")
    written += save(
        figure_probe_trial_marginalization(config, rows, bundle),
        figures_dir, "fig6_probe_trial_marginalization",
    )
    written.append(
        write_csv(tables_dir / "layer_rule_probes.csv", bundle["rule_csv_rows"], fieldnames=RULE_PROBE_CSV_COLUMNS)
    )
    written.append(
        write_csv(
            tables_dir / "probe_trial_predictions.csv",
            bundle["trial_pred_csv_rows"],
            fieldnames=TRIAL_PRED_CSV_COLUMNS,
        )
    )

    md_rules = tables_dir / "layer_rule_probes.md"
    md_rules.parent.mkdir(parents=True, exist_ok=True)
    md_rules.write_text(layer_rule_probes_markdown(bundle))
    written.append(md_rules)

    md_trials = tables_dir / "probe_trial_predictions.md"
    md_trials.write_text(probe_trial_predictions_markdown(bundle))
    written.append(md_trials)

    return written, bundle


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Residual-stream rule probing and superposed trial marginalization for BCL."
    )
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    parser.add_argument(
        "--prepare-prompts",
        nargs=2,
        metavar=("OUT_META_JSONL", "OUT_PROMPTS_BIN"),
        help="serialize pure-induction BCL trial prompts for tools/extract_gemma4_acts and exit",
    )
    parser.add_argument(
        "--blocks-per-concept",
        type=int,
        default=25,
        help="blocks per concept when using --prepare-prompts (default: 25)",
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("META_JSONL", "ACTS_F32_BIN"),
        help="(meta.jsonl, acts.f32.bin) pair; pass multiple --input flags to combine shards",
    )
    parser.add_argument("--n-embd", type=int, default=2816, help="residual stream width (default: 2816)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.prepare_prompts:
        rows = read_rows(config.resolve(config.run["results_file"]))
        n_rec = prepare_probe_prompts(
            config,
            rows,
            args.prepare_prompts[0],
            args.prepare_prompts[1],
            blocks_per_concept=args.blocks_per_concept,
        )
        print(f"wrote {n_rec} prompt records to {args.prepare_prompts[0]} and {args.prepare_prompts[1]}")
        return

    if not args.input:
        parser.error("either --prepare-prompts or at least one --input META_JSONL ACTS_F32_BIN is required")

    written, _ = make_all_probes(config, args.input, n_embd=args.n_embd)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
