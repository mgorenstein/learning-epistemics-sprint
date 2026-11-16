"""Scores and learning-curve aggregation for interactive Bayesian concept learning.

Run from the project root:

    python -m bcl.score --config config_bcl.yaml

Each results row in ``data/bcl/results.jsonl`` holds one ``N``-trial block (either a
single shuffled pass over all 16 unique stimuli with ``repetitions == 1``, or multiple
shuffled repetitions such as 4 passes over 9 stimuli = 36 trials) for one model and one
target concept.

What is computed
----------------
learning_curves      Trial-by-trial proportion correct (trials 1..N) across blocks for
                     each (model, concept, trial), (model, complexity_group, trial), and
                     (model, "all", trial), with 95% bootstrap confidence intervals from
                     resampling whole blocks with replacement, alongside mean ideal
                     Bayesian accuracy (``bayes_correct``) and ``p_yes`` calibration TVD
                     (``|p_yes - bayes_p_yes|``) when ``p_yes`` is available.
concept_summary      Per-concept summary across the ``N`` trials: overall accuracy, accuracy
                     in each block quarter (``Q1..Q4`` when ``repetitions == 1``) or each
                     stimulus repetition (``Rep 1..4`` when ``repetitions > 1``), total
                     errors (``N - sum(correct)``), trials-to-criterion (first trial 1..N
                     from which the model makes 0 errors for the rest of the block, or N+1
                     if never), and Pearson ``r`` + RMSE against the ideal Bayesian
                     learner's ``N``-point concept learning curve.
complexity_summary   The same summary grouped by Boolean complexity (``single``, ``negation``,
                     ``conjunction``, ``disjunction``, ``compound``) and across ``all`` blocks.
model_summary        Model-level accuracy, errors, trials-to-criterion, calibration TVD,
                     parse-failure rate, and Pearson ``r`` + RMSE against the ideal Bayesian
                     learner across the 10 concept means, the ``N``-point overall curve, and
                     all concept-by-trial points.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import runpy
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

COMPLEXITY_GROUPS = ("single", "negation", "conjunction", "disjunction", "compound")
ALL_SCOPE = "all"

DEFAULT_SEED = 20261113
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_INTERVAL = 0.95
TIE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Config and JSONL helpers (decoupled with clean fallback)
# ---------------------------------------------------------------------------

class _StandaloneConfig:
    """Lightweight YAML config reader used when ``bcl.config`` is not imported."""

    def __init__(self, path):
        import yaml

        self.path = Path(path)
        self.text = self.path.read_text()
        self.settings = yaml.safe_load(self.text) or {}
        self.seed = self.settings.get("seed", DEFAULT_SEED)
        self.alpha = self.settings.get("alpha", 0.99)
        self.blocks = self.settings.get("blocks", {})
        self.run = self.settings.get("run", {"results_file": "data/bcl/results.jsonl"})
        self.programs = {}
        rules_file = self.settings.get("rules_file")
        if rules_file:
            resolved = self.resolve(rules_file)
            if resolved.exists():
                namespace = runpy.run_path(str(resolved))
                if isinstance(namespace.get("RULES"), dict):
                    self.programs = dict(namespace["RULES"])

    def resolve(self, relative_path):
        return self.path.parent / relative_path

    def save_copy(self, output_path):
        copy = Path(output_path).with_suffix(".config.yaml")
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(self.text)
        return copy


def load_config_or_fallback(config_or_path="config_bcl.yaml"):
    """Load ``config_bcl.yaml`` via ``bcl.config.load_config`` if present, else YAML fallback."""
    if hasattr(config_or_path, "settings") and hasattr(config_or_path, "resolve"):
        return config_or_path
    path = Path(config_or_path)
    try:
        from bcl.config import load_config as _bcl_load_config
    except ImportError:
        return _StandaloneConfig(path)
    return _bcl_load_config(path)


def _score_params(config=None, settings=None):
    """Extract ``(seed, bootstrap_samples, interval, rule_order)`` from config or settings dict."""
    source = settings if settings is not None else config
    if source is None:
        return DEFAULT_SEED, DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_INTERVAL, []
    if isinstance(source, (str, Path)):
        source = load_config_or_fallback(source)
    if hasattr(source, "settings"):
        raw = source.settings
        seed = getattr(source, "seed", raw.get("seed", DEFAULT_SEED))
        programs = getattr(source, "programs", None) or {}
        rule_order = list(programs.keys())
    elif isinstance(source, Mapping):
        raw = source
        seed = raw.get("seed", DEFAULT_SEED)
        rule_order = list(raw.get("programs", {}).keys()) if isinstance(raw.get("programs"), Mapping) else []
    else:
        raise TypeError(f"expected Config, settings dict, path, or None; got {type(source).__name__}")

    scores = raw.get("scores", raw) if isinstance(raw, Mapping) else {}
    samples = int(scores.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES))
    interval = float(scores.get("interval", DEFAULT_INTERVAL))
    return int(seed), samples, interval, rule_order


def read_rows(path):
    """Read block rows from a JSONL results file, or return ``[]`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as lines:
        return [json.loads(line) for line in lines if line.strip()]


def infer_complexity_group(program):
    """AST fallback for classifying a Boolean rule when ``complexity_group`` is omitted."""
    if not program:
        return "compound"
    tree = ast.parse(program.strip(), mode="eval").body
    if isinstance(tree, ast.Lambda):
        tree = tree.body
    has_and = any(isinstance(node, ast.And) for node in ast.walk(tree))
    has_or = any(isinstance(node, ast.Or) for node in ast.walk(tree))
    has_not = any(isinstance(node, (ast.Not, ast.NotEq)) for node in ast.walk(tree))
    has_eq = any(isinstance(node, ast.Compare) and all(isinstance(op, ast.Eq) for op in node.ops)
                 for node in ast.walk(tree))
    if not has_and and not has_or and not has_not and has_eq:
        return "single"
    if not has_and and not has_or and has_not:
        return "negation"
    if has_and and not has_or and not has_not:
        return "conjunction"
    if has_or and not has_and and not has_not:
        return "disjunction"
    return "compound"


def row_complexity_group(row):
    """The row's ``complexity_group``, falling back to AST inspection only if missing."""
    if row.get("complexity_group"):
        return row["complexity_group"]
    return infer_complexity_group(row.get("program", ""))


# ---------------------------------------------------------------------------
# Ordering helpers
# ---------------------------------------------------------------------------

def models_in(rows):
    """Model labels in the order they first appear in ``rows``."""
    return list(dict.fromkeys(row["model"] for row in rows))


def _complexity_rank(group):
    try:
        return COMPLEXITY_GROUPS.index(group)
    except ValueError:
        return len(COMPLEXITY_GROUPS)


def complexity_groups_in(rows):
    """Complexity groups present in ``rows``, ordered by the Shepard/Feldman/Goodman hierarchy."""
    present = list(dict.fromkeys(row_complexity_group(row) for row in rows))
    return sorted(present, key=lambda group: (_complexity_rank(group), present.index(group)))


def concepts_in(rows, rule_order=()):
    """Concepts present in ``rows``, ordered by complexity group and then config/appearance order."""
    seen = []
    group_of = {}
    for row in rows:
        concept = row["concept"]
        if concept not in group_of:
            seen.append(concept)
            group_of[concept] = row_complexity_group(row)
    order_index = {name: idx for idx, name in enumerate(rule_order)}
    fallback_offset = len(order_index)
    for idx, name in enumerate(seen):
        order_index.setdefault(name, fallback_offset + idx)
    return sorted(seen, key=lambda name: (_complexity_rank(group_of[name]), order_index[name]))


# ---------------------------------------------------------------------------
# Block-level and curve-level metrics
# ---------------------------------------------------------------------------

def ordered_trials(row):
    """A block's trial dicts sorted by 1-indexed ``trial``."""
    return sorted(row["trials"], key=lambda trial: trial["trial"])


def trial_repetition(trial_dict, stim_per_rep=9):
    """1-indexed repetition number stored on ``trial_dict`` (or inferred from ``stim_per_rep``)."""
    if trial_dict.get("repetition") is not None:
        return int(trial_dict["repetition"])
    return (int(trial_dict["trial"]) - 1) // stim_per_rep + 1


def trial_quarter(trial_number, n_trials):
    """1-indexed block quarter ``1..4`` partitioning ``1..n_trials`` into four equal quarters."""
    return min(4, (int(trial_number) - 1) * 4 // int(n_trials) + 1)


def quarter_trial_ranges(n_trials):
    """Inclusive ``(start_trial, end_trial)`` pairs for the 4 quarters of ``1..n_trials``."""
    n = max(4, int(n_trials))
    by_q = defaultdict(list)
    for t in range(1, n + 1):
        by_q[trial_quarter(t, n)].append(t)
    return [(by_q[q][0], by_q[q][-1]) for q in range(1, 5)]


def stage_column_labels(n_trials, single_pass, *, short_rep=False):
    """Column/tick labels for the 4 block stages (``Q1..Q4`` when single-pass, else ``Rep 1..4``)."""
    if single_pass:
        return [
            f"Q{q} ({lo}\u2013{hi})"
            for q, (lo, hi) in enumerate(quarter_trial_ranges(n_trials), start=1)
        ]
    if short_rep:
        return [f"rep {r}" for r in range(1, 5)]
    n = max(4, int(n_trials))
    stim = max(1, n // 4)
    return [
        f"Rep {r} ({(r - 1) * stim + 1}\u2013{r * stim if r < 4 else max(r * stim, n)})"
        for r in range(1, 5)
    ]


def _block_n_trials(row):
    trials = row.get("trials") or ()
    if row.get("n_trials") is not None:
        return int(row["n_trials"])
    if not trials:
        return 0
    return max(len(trials), max(int(t["trial"]) for t in trials))


def is_single_pass_block(row):
    """True when ``row`` has ``repetitions == 1`` (or max ``trial['repetition'] == 1``) and ``n_trials >= 4``."""
    n_trials = _block_n_trials(row)
    if n_trials < 4:
        return False
    if row.get("repetitions") is not None:
        return int(row["repetitions"]) == 1
    reps = [int(t["repetition"]) for t in (row.get("trials") or ()) if t.get("repetition") is not None]
    return bool(reps) and max(reps) == 1


def trial_stage_bin(trial_dict, row=None):
    """1..4 stage bin: equal block quarter when ``row`` is single-pass, else ``trial_repetition``."""
    if row is not None and is_single_pass_block(row):
        return trial_quarter(trial_dict["trial"], _block_n_trials(row))
    return trial_repetition(trial_dict)


def trial_calibration_tvd(trial_dict):
    """Binary total variation distance ``|p_yes - bayes_p_yes|``, or ``None`` if ``p_yes`` is absent."""
    p_yes = trial_dict.get("p_yes")
    if p_yes is None:
        return None
    return abs(float(p_yes) - float(trial_dict["bayes_p_yes"]))


def block_accuracy(row):
    """Overall proportion of trials answered accurately in one block."""
    trials = row["trials"]
    if not trials:
        return 0.0
    return sum(1.0 for t in trials if t["correct"]) / len(trials)


def block_bayes_accuracy(row):
    """Mean ``bayes_correct`` across all trials of one block."""
    trials = row["trials"]
    if not trials:
        return 0.0
    return sum(float(t["bayes_correct"]) for t in trials) / len(trials)


def block_total_errors(row):
    """Total errors across the block (``N - sum(correct)``)."""
    trials = row["trials"]
    return len(trials) - sum(1 for t in trials if t["correct"])


def block_bayes_errors(row):
    """Expected/ideal error count ``sum(1 - bayes_correct)`` across the block."""
    return sum(1.0 - float(t["bayes_correct"]) for t in row["trials"])


def trials_to_criterion(trials):
    """First trial (1..N) from which the model makes 0 errors for the rest of the block, or N+1 if never.

    Specifically, if the last error occurs at trial ``L`` (with ``L = 0`` when all trials are
    accurate), criterion is reached at trial ``L + 1``:
    - 0 errors across trials 1..N -> ``1``
    - last error on trial ``k`` (1 <= k < N) -> ``k + 1``
    - error on trial ``N`` -> ``N + 1`` (never reached criterion before the block ended).
    """
    ordered = sorted(trials, key=lambda t: t["trial"])
    if not ordered:
        return 1
    n_trials = int(ordered[-1]["trial"])
    last_error = 0
    for t in ordered:
        if not t["correct"]:
            last_error = int(t["trial"])
    return last_error + 1 if last_error < n_trials else n_trials + 1


def bayes_trials_to_criterion(trials):
    """Trials-to-criterion for the ideal Bayesian learner (counting ties 0.5 and 0.0 as non-criterion)."""
    ordered = sorted(trials, key=lambda t: t["trial"])
    if not ordered:
        return 1
    n_trials = int(ordered[-1]["trial"])
    last_error = 0
    for t in ordered:
        if float(t["bayes_correct"]) < 1.0 - TIE_TOLERANCE:
            last_error = int(t["trial"])
    return last_error + 1 if last_error < n_trials else n_trials + 1


def block_repetition_accuracies(row, repetitions=4):
    """Per-stage accuracy ``{1: ..., 2: ..., 3: ..., 4: ...}`` (quarters if single-pass, else repetitions)."""
    by_rep = defaultdict(list)
    for t in row["trials"]:
        by_rep[trial_stage_bin(t, row)].append(1.0 if t["correct"] else 0.0)
    return {
        rep: (sum(by_rep[rep]) / len(by_rep[rep]) if by_rep[rep] else None)
        for rep in range(1, repetitions + 1)
    }


def block_bayes_repetition_accuracies(row, repetitions=4):
    """Per-stage ideal Bayesian accuracy ``{1: ..., 2: ..., 3: ..., 4: ...}`` for one block."""
    by_rep = defaultdict(list)
    for t in row["trials"]:
        by_rep[trial_stage_bin(t, row)].append(float(t["bayes_correct"]))
    return {
        rep: (sum(by_rep[rep]) / len(by_rep[rep]) if by_rep[rep] else None)
        for rep in range(1, repetitions + 1)
    }


def block_parse_failures(row):
    """Number of trials with unreadable answers in one block."""
    if row.get("parse_failures") is not None:
        return int(row["parse_failures"])
    return sum(1 for t in row["trials"] if t.get("answer") is None)


def pearson_r(xs, ys):
    """Pearson correlation ``r`` between two equal-length series, or ``None`` if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    ss_y = sum((y - mean_y) ** 2 for y in ys)
    if ss_x <= 1e-15 or ss_y <= 1e-15:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    r = cov / math.sqrt(ss_x * ss_y)
    return max(-1.0, min(1.0, r))


def rmse(xs, ys):
    """Root-mean-square error between two equal-length series, or ``None`` if empty."""
    if len(xs) != len(ys) or not xs:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


# ---------------------------------------------------------------------------
# Whole-block bootstrap intervals
# ---------------------------------------------------------------------------

def _block_key(row, fallback_index=0):
    return row.get("block_id") or row.get("key") or f"block-{fallback_index}"


def mean_with_interval(values, samples, level, seed):
    """Mean of ``values`` with a bootstrap interval from resampling whole blocks with replacement.

    ``values`` is a sequence of ``(block_id, number)`` pairs. Returns ``(mean, low, high)``.
    With fewer than two distinct blocks, ``low`` and ``high`` are ``None``.
    """
    if not values:
        return None, None, None
    by_block = defaultdict(lambda: [0.0, 0])
    for block_id, value in values:
        by_block[block_id][0] += float(value)
        by_block[block_id][1] += 1
    totals = list(by_block.values())
    mean = sum(total for total, _ in totals) / sum(count for _, count in totals)
    if len(totals) < 2:
        return mean, None, None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        drawn = rng.choices(totals, k=len(totals))
        means.append(sum(total for total, _ in drawn) / sum(count for _, count in drawn))
    means.sort()
    low = means[int((1 - level) / 2 * samples)]
    high = means[min(samples - 1, int((1 + level) / 2 * samples))]
    return mean, low, high


def _bootstrap_curves_over_blocks(blocks, trial_numbers, samples, level, seed):
    """Resample whole blocks once across all ``trial_numbers`` simultaneously.

    Produces exact parity with ``mean_with_interval`` per trial while keeping all trials
    of each resampled block together in every bootstrap replicate.
    """
    block_ids = [_block_key(row, idx) for idx, row in enumerate(blocks)]
    trial_maps = [{int(t["trial"]): (1.0 if t["correct"] else 0.0) for t in row["trials"]} for row in blocks]
    complete_unique = (
        len(set(block_ids)) == len(blocks)
        and all(all(num in tmap for num in trial_numbers) for tmap in trial_maps)
    )
    if not complete_unique:
        result = {}
        for num in trial_numbers:
            pairs = [(bid, tmap[num]) for bid, tmap in zip(block_ids, trial_maps) if num in tmap]
            result[num] = mean_with_interval(pairs, samples, level, seed)
        return result

    n_blocks = len(blocks)
    vectors = [tuple(tmap[num] for num in trial_numbers) for tmap in trial_maps]
    means = [sum(vec[pos] for vec in vectors) / n_blocks for pos in range(len(trial_numbers))]
    if n_blocks < 2:
        return {num: (means[pos], None, None) for pos, num in enumerate(trial_numbers)}

    rng = random.Random(seed)
    replicates = [[] for _ in trial_numbers]
    for _ in range(samples):
        drawn = rng.choices(vectors, k=n_blocks)
        for pos, col in enumerate(zip(*drawn)):
            replicates[pos].append(sum(col) / n_blocks)

    low_idx = int((1 - level) / 2 * samples)
    high_idx = min(samples - 1, int((1 + level) / 2 * samples))
    result = {}
    for pos, num in enumerate(trial_numbers):
        col = replicates[pos]
        col.sort()
        result[num] = (means[pos], col[low_idx], col[high_idx])
    return result


# ---------------------------------------------------------------------------
# Aggregations: learning curves, concept summaries, complexity & model summaries
# ---------------------------------------------------------------------------

def _group_blocks_for_scopes(model_rows, rule_order):
    """Yield ``(scope_type, scope, complexity_group, blocks)`` in canonical order."""
    by_concept = defaultdict(list)
    concept_group = {}
    by_group = defaultdict(list)
    for row in model_rows:
        concept = row["concept"]
        cgroup = row_complexity_group(row)
        by_concept[concept].append(row)
        concept_group.setdefault(concept, cgroup)
        by_group[cgroup].append(row)

    for concept in concepts_in(model_rows, rule_order):
        yield "concept", concept, concept_group[concept], by_concept[concept]
    for cgroup in complexity_groups_in(model_rows):
        yield "complexity_group", cgroup, cgroup, by_group[cgroup]
    yield "all", ALL_SCOPE, ALL_SCOPE, list(model_rows)


def learning_curves(rows, config=None, *, settings=None):
    """Trial-by-trial learning curves (trials 1..N) by concept, complexity_group, and all.

    Returns a list of dicts with keys:
    ``model``, ``scope_type``, ``scope``, ``complexity_group``, ``trial``, ``repetition``,
    ``n_blocks``, ``correct``, ``low``, ``high``, ``bayes_correct``, ``calibration_tvd``, ``n_p_yes``.
    """
    seed, samples, interval, rule_order = _score_params(config, settings)
    records = []
    for model in models_in(rows):
        model_rows = [row for row in rows if row["model"] == model]
        for scope_type, scope, cgroup, blocks in _group_blocks_for_scopes(model_rows, rule_order):
            trial_numbers = sorted({int(t["trial"]) for row in blocks for t in row["trials"]})
            if not trial_numbers:
                continue
            ci_by_trial = _bootstrap_curves_over_blocks(blocks, trial_numbers, samples, interval, seed)
            by_trial = defaultdict(list)
            for row in blocks:
                for t in row["trials"]:
                    by_trial[int(t["trial"])].append(t)

            for num in trial_numbers:
                entries = by_trial[num]
                mean_acc, low, high = ci_by_trial[num]
                mean_bayes = sum(float(t["bayes_correct"]) for t in entries) / len(entries)
                tvds = [val for val in (trial_calibration_tvd(t) for t in entries) if val is not None]
                mean_tvd = (sum(tvds) / len(tvds)) if tvds else None
                rep = trial_repetition(entries[0])
                records.append({
                    "model": model,
                    "scope_type": scope_type,
                    "scope": scope,
                    "complexity_group": cgroup,
                    "trial": num,
                    "repetition": rep,
                    "n_blocks": len(entries),
                    "correct": mean_acc,
                    "low": low,
                    "high": high,
                    "bayes_correct": mean_bayes,
                    "calibration_tvd": mean_tvd,
                    "n_p_yes": len(tvds),
                })
    return records


def repetition_curves(rows, config=None, *, settings=None):
    """Accuracy by stage (quarters 1..4 if single-pass, else repetitions 1..4) with whole-block bootstrap CIs."""
    seed, samples, interval, rule_order = _score_params(config, settings)
    records = []
    for model in models_in(rows):
        model_rows = [row for row in rows if row["model"] == model]
        for scope_type, scope, cgroup, blocks in _group_blocks_for_scopes(model_rows, rule_order):
            reps = sorted({trial_stage_bin(t, row) for row in blocks for t in row["trials"]})
            for rep in reps:
                acc_pairs = []
                bayes_vals = []
                tvd_vals = []
                for idx, row in enumerate(blocks):
                    bid = _block_key(row, idx)
                    rep_trials = [t for t in row["trials"] if trial_stage_bin(t, row) == rep]
                    if not rep_trials:
                        continue
                    for t in rep_trials:
                        acc_pairs.append((bid, 1.0 if t["correct"] else 0.0))
                        bayes_vals.append(float(t["bayes_correct"]))
                        tvd = trial_calibration_tvd(t)
                        if tvd is not None:
                            tvd_vals.append(tvd)
                if not acc_pairs:
                    continue
                mean_acc, low, high = mean_with_interval(acc_pairs, samples, interval, seed)
                records.append({
                    "model": model,
                    "scope_type": scope_type,
                    "scope": scope,
                    "complexity_group": cgroup,
                    "repetition": rep,
                    "n_blocks": len({bid for bid, _ in acc_pairs}),
                    "correct": mean_acc,
                    "low": low,
                    "high": high,
                    "bayes_correct": sum(bayes_vals) / len(bayes_vals),
                    "calibration_tvd": (sum(tvd_vals) / len(tvd_vals)) if tvd_vals else None,
                })
    return records


def _summarize_block_group(model, scope_type, name, cgroup, blocks, curve_rows, samples, interval, seed):
    """Compute overall, per-stage, error, TTC, calibration, and curve-fit summary metrics."""
    n_blocks = len(blocks)
    n_trials = max((len(row["trials"]) for row in blocks), default=0)
    single_pass = bool(blocks) and all(is_single_pass_block(row) for row in blocks)
    repetitions = (
        1
        if single_pass
        else max(
            (
                int(row.get("repetitions") or max((trial_repetition(t) for t in row["trials"]), default=1))
                for row in blocks
            ),
            default=1,
        )
    )

    acc_pairs = []
    err_pairs = []
    ttc_pairs = []
    tvd_pairs = []
    bayes_accs = []
    bayes_errs = []
    bayes_ttcs = []
    rep_pairs = defaultdict(list)
    bayes_rep_vals = defaultdict(list)
    total_trials_count = 0
    total_parse_failures = 0

    for idx, row in enumerate(blocks):
        bid = _block_key(row, idx)
        trials = row["trials"]
        total_trials_count += len(trials)
        total_parse_failures += block_parse_failures(row)
        for t in trials:
            val = 1.0 if t["correct"] else 0.0
            stage = trial_stage_bin(t, row)
            acc_pairs.append((bid, val))
            rep_pairs[stage].append((bid, val))
            bayes_rep_vals[stage].append(float(t["bayes_correct"]))
            tvd = trial_calibration_tvd(t)
            if tvd is not None:
                tvd_pairs.append((bid, tvd))
        err_pairs.append((bid, float(block_total_errors(row))))
        ttc_pairs.append((bid, float(trials_to_criterion(trials))))
        bayes_accs.append(block_bayes_accuracy(row))
        bayes_errs.append(block_bayes_errors(row))
        bayes_ttcs.append(float(bayes_trials_to_criterion(trials)))

    acc_mean, acc_low, acc_high = mean_with_interval(acc_pairs, samples, interval, seed)
    err_mean, err_low, err_high = mean_with_interval(err_pairs, samples, interval, seed)
    ttc_mean, ttc_low, ttc_high = mean_with_interval(ttc_pairs, samples, interval, seed)
    tvd_mean, tvd_low, tvd_high = mean_with_interval(tvd_pairs, samples, interval, seed) if tvd_pairs else (None, None, None)

    rep_means = {}
    bayes_reps = {}
    for rep in range(1, 5):
        if rep_pairs[rep]:
            rep_means[rep], _, _ = mean_with_interval(rep_pairs[rep], samples, interval, seed)
            bayes_reps[rep] = sum(bayes_rep_vals[rep]) / len(bayes_rep_vals[rep])
        else:
            rep_means[rep] = None
            bayes_reps[rep] = None

    ordered_curve = sorted(curve_rows, key=lambda r: r["trial"])
    xs = [r["correct"] for r in ordered_curve]
    ys = [r["bayes_correct"] for r in ordered_curve]

    record = {
        "model": model,
        "scope_type": scope_type,
        "complexity_group": cgroup,
        "n_blocks": n_blocks,
        "n_trials": n_trials,
        "repetitions": repetitions,
        "accuracy": acc_mean,
        "low": acc_low,
        "high": acc_high,
        "bayes_accuracy": sum(bayes_accs) / len(bayes_accs) if bayes_accs else None,
        "rep1_accuracy": rep_means[1],
        "rep2_accuracy": rep_means[2],
        "rep3_accuracy": rep_means[3],
        "rep4_accuracy": rep_means[4],
        "bayes_rep1": bayes_reps[1],
        "bayes_rep2": bayes_reps[2],
        "bayes_rep3": bayes_reps[3],
        "bayes_rep4": bayes_reps[4],
        "total_errors": err_mean,
        "errors_low": err_low,
        "errors_high": err_high,
        "bayes_total_errors": sum(bayes_errs) / len(bayes_errs) if bayes_errs else None,
        "trials_to_criterion": ttc_mean,
        "ttc_low": ttc_low,
        "ttc_high": ttc_high,
        "bayes_trials_to_criterion": sum(bayes_ttcs) / len(bayes_ttcs) if bayes_ttcs else None,
        "curve_r": pearson_r(xs, ys),
        "curve_rmse": rmse(xs, ys),
        "calibration_tvd": tvd_mean,
        "calibration_tvd_low": tvd_low,
        "calibration_tvd_high": tvd_high,
        "parse_failure_rate": (total_parse_failures / total_trials_count) if total_trials_count else 0.0,
    }
    if scope_type == "concept":
        record = {"model": model, "concept": name, **{k: v for k, v in record.items() if k not in ("model", "scope_type")}}
    else:
        record = {"model": model, "scope_type": scope_type, "scope": name,
                  **{k: v for k, v in record.items() if k not in ("model", "scope_type")}}
    return record


def concept_summary(rows, config=None, *, settings=None, curves=None):
    """Per-concept summary metrics across all trials in a block for every ``(model, concept)``."""
    seed, samples, interval, rule_order = _score_params(config, settings)
    if curves is None:
        curves = learning_curves(rows, config=config, settings=settings)
    curve_lookup = defaultdict(list)
    for r in curves:
        if r["scope_type"] == "concept":
            curve_lookup[r["model"], r["scope"]].append(r)

    summaries = []
    for model in models_in(rows):
        model_rows = [row for row in rows if row["model"] == model]
        for scope_type, scope, cgroup, blocks in _group_blocks_for_scopes(model_rows, rule_order):
            if scope_type != "concept":
                continue
            summaries.append(_summarize_block_group(
                model, "concept", scope, cgroup, blocks,
                curve_lookup[model, scope], samples, interval, seed,
            ))
    return summaries


def complexity_summary(rows, config=None, *, settings=None, curves=None):
    """Summary metrics grouped by Boolean complexity hierarchy and across ``all`` blocks."""
    seed, samples, interval, rule_order = _score_params(config, settings)
    if curves is None:
        curves = learning_curves(rows, config=config, settings=settings)
    curve_lookup = defaultdict(list)
    for r in curves:
        if r["scope_type"] in ("complexity_group", "all"):
            curve_lookup[r["model"], r["scope_type"], r["scope"]].append(r)

    summaries = []
    for model in models_in(rows):
        model_rows = [row for row in rows if row["model"] == model]
        for scope_type, scope, cgroup, blocks in _group_blocks_for_scopes(model_rows, rule_order):
            if scope_type == "concept":
                continue
            summaries.append(_summarize_block_group(
                model, scope_type, scope, cgroup, blocks,
                curve_lookup[model, scope_type, scope], samples, interval, seed,
            ))
    return summaries


def model_summary(rows, config=None, *, settings=None, curves=None, concepts=None, complexities=None):
    """Model-level alignment and accuracy summary comparing each model to the ideal Bayesian learner."""
    if curves is None:
        curves = learning_curves(rows, config=config, settings=settings)
    if concepts is None:
        concepts = concept_summary(rows, config=config, settings=settings, curves=curves)
    if complexities is None:
        complexities = complexity_summary(rows, config=config, settings=settings, curves=curves)

    results = []
    for model in models_in(rows):
        overall = next((r for r in complexities if r["model"] == model and r["scope_type"] == "all"), None)
        if overall is None:
            continue
        c_rows = [r for r in concepts if r["model"] == model]
        ct_rows = [r for r in curves if r["model"] == model and r["scope_type"] == "concept"]

        c_model_acc = [r["accuracy"] for r in c_rows]
        c_bayes_acc = [r["bayes_accuracy"] for r in c_rows]
        c_model_err = [r["total_errors"] for r in c_rows]
        c_bayes_err = [r["bayes_total_errors"] for r in c_rows]
        ct_model = [r["correct"] for r in ct_rows]
        ct_bayes = [r["bayes_correct"] for r in ct_rows]

        results.append({
            "model": model,
            "n_concepts": len(c_rows),
            "n_blocks": overall["n_blocks"],
            "n_trials": overall["n_trials"],
            "repetitions": overall.get("repetitions", 1 if overall["n_trials"] == 16 else 4),
            "accuracy": overall["accuracy"],
            "low": overall["low"],
            "high": overall["high"],
            "bayes_accuracy": overall["bayes_accuracy"],
            "total_errors": overall["total_errors"],
            "errors_low": overall["errors_low"],
            "errors_high": overall["errors_high"],
            "bayes_total_errors": overall["bayes_total_errors"],
            "trials_to_criterion": overall["trials_to_criterion"],
            "ttc_low": overall["ttc_low"],
            "ttc_high": overall["ttc_high"],
            "bayes_trials_to_criterion": overall["bayes_trials_to_criterion"],
            "concept_r": pearson_r(c_model_acc, c_bayes_acc),
            "concept_rmse": rmse(c_model_acc, c_bayes_acc),
            "concept_errors_r": pearson_r(c_model_err, c_bayes_err),
            "concept_errors_rmse": rmse(c_model_err, c_bayes_err),
            "overall_curve_r": overall["curve_r"],
            "overall_curve_rmse": overall["curve_rmse"],
            "concept_trial_r": pearson_r(ct_model, ct_bayes),
            "concept_trial_rmse": rmse(ct_model, ct_bayes),
            "calibration_tvd": overall["calibration_tvd"],
            "calibration_tvd_low": overall["calibration_tvd_low"],
            "calibration_tvd_high": overall["calibration_tvd_high"],
            "parse_failure_rate": overall["parse_failure_rate"],
        })
    return results


# ---------------------------------------------------------------------------
# Plain-text CLI report
# ---------------------------------------------------------------------------

def _fmt_pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _fmt_pct_ci(mean, low, high):
    if mean is None:
        return "-"
    if low is None or high is None:
        return _fmt_pct(mean)
    return f"{100 * mean:.1f}% [{100 * low:.1f}%, {100 * high:.1f}%]"


def _fmt_num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def report(rows, config=None, *, settings=None):
    """Plain-text summary report of per-concept, per-complexity, and model-vs-Bayes metrics."""
    if not rows:
        return "No results yet."
    _, _, interval, _ = _score_params(config, settings)
    level = f"{100 * interval:.0f}%"
    curves = learning_curves(rows, config=config, settings=settings)
    concepts = concept_summary(rows, config=config, settings=settings, curves=curves)
    complexities = complexity_summary(rows, config=config, settings=settings, curves=curves)
    models = model_summary(
        rows, config=config, settings=settings,
        curves=curves, concepts=concepts, complexities=complexities,
    )

    lines = []
    for m_sum in models:
        model = m_sum["model"]
        n_trials = m_sum["n_trials"]
        single_pass = (m_sum.get("repetitions") == 1 and n_trials >= 4)
        s1, s2, s3, s4 = stage_column_labels(n_trials, single_pass, short_rep=True)
        col_w = max(7, max(len(lbl) for lbl in (s1, s2, s3, s4)))
        c_rows = [r for r in concepts if r["model"] == model]
        g_rows = [r for r in complexities if r["model"] == model]
        lines.append(
            f"=== {model} ===  ({m_sum['n_blocks']} blocks across {m_sum['n_concepts']} concepts, "
            f"{n_trials} trials/block; {level} bootstrap CI over blocks)"
        )
        lines.append("")
        lines.append("Per-concept learning summary:")
        header = (
            f"  {'concept':<20} {'group':<12} {'blocks':>6} {s1:>{col_w}} {s2:>{col_w}} "
            f"{s3:>{col_w}} {s4:>{col_w}} {'overall [' + level + ' CI]':<24} {'bayes':>7} "
            f"{'errors':>7} {'ttc':>6} {'curve r':>8} {'rmse':>6}"
        )
        lines.append(header)
        for r in c_rows:
            lines.append(
                f"  {r['concept']:<20} {r['complexity_group']:<12} {r['n_blocks']:>6} "
                f"{_fmt_pct(r['rep1_accuracy']):>{col_w}} {_fmt_pct(r['rep2_accuracy']):>{col_w}} "
                f"{_fmt_pct(r['rep3_accuracy']):>{col_w}} {_fmt_pct(r['rep4_accuracy']):>{col_w}} "
                f"{_fmt_pct_ci(r['accuracy'], r['low'], r['high']):<24} "
                f"{_fmt_pct(r['bayes_accuracy']):>7} {_fmt_num(r['total_errors'], 1):>7} "
                f"{_fmt_num(r['trials_to_criterion'], 1):>6} "
                f"{_fmt_num(r['curve_r'], 3):>8} {_fmt_num(r['curve_rmse'], 3):>6}"
            )
        lines.append("")
        lines.append("Boolean complexity hierarchy:")
        lines.append(
            f"  {'complexity':<20} {'blocks':>6} {s1:>{col_w}} {s2:>{col_w}} {s3:>{col_w}} {s4:>{col_w}} "
            f"{'overall [' + level + ' CI]':<24} {'bayes':>7} {'errors':>7} {'ttc':>6}"
        )
        for r in g_rows:
            label = "all concepts" if r["scope_type"] == "all" else r["scope"]
            lines.append(
                f"  {label:<20} {r['n_blocks']:>6} "
                f"{_fmt_pct(r['rep1_accuracy']):>{col_w}} {_fmt_pct(r['rep2_accuracy']):>{col_w}} "
                f"{_fmt_pct(r['rep3_accuracy']):>{col_w}} {_fmt_pct(r['rep4_accuracy']):>{col_w}} "
                f"{_fmt_pct_ci(r['accuracy'], r['low'], r['high']):<24} "
                f"{_fmt_pct(r['bayes_accuracy']):>7} {_fmt_num(r['total_errors'], 1):>7} "
                f"{_fmt_num(r['trials_to_criterion'], 1):>6}"
            )
        lines.append("")
        lines.append(
            f"Alignment with ideal Bayesian learner:"
            f"  concept means r = {_fmt_num(m_sum['concept_r'], 3)}, RMSE = {_fmt_num(m_sum['concept_rmse'], 3)};"
            f"  overall {n_trials}-trial curve r = {_fmt_num(m_sum['overall_curve_r'], 3)}, RMSE = {_fmt_num(m_sum['overall_curve_rmse'], 3)};"
            f"  concept x trial ({m_sum['n_concepts'] * n_trials} pts) r = {_fmt_num(m_sum['concept_trial_r'], 3)}, "
            f"RMSE = {_fmt_num(m_sum['concept_trial_rmse'], 3)}."
        )
        extra = [f"Parse failures: {100 * m_sum['parse_failure_rate']:.2f}% of trials."]
        if m_sum["calibration_tvd"] is not None:
            tvd_str = _fmt_num(m_sum["calibration_tvd"], 3)
            if m_sum["calibration_tvd_low"] is not None:
                tvd_str += f" [{_fmt_num(m_sum['calibration_tvd_low'], 3)}, {_fmt_num(m_sum['calibration_tvd_high'], 3)}]"
            extra.insert(0, f"Mean p_yes calibration TVD |p_yes - bayes_p_yes|: {tvd_str}.")
        lines.append("  " + "  ".join(extra))
        lines.append("")
    return "\n".join(lines).rstrip()


def score_bundle(rows, config=None, *, settings=None):
    """Compute and return all score tables in a single dict (used by ``--json`` and ``bcl.analysis``)."""
    curves = learning_curves(rows, config=config, settings=settings)
    rep_curves = repetition_curves(rows, config=config, settings=settings)
    concepts = concept_summary(rows, config=config, settings=settings, curves=curves)
    complexities = complexity_summary(rows, config=config, settings=settings, curves=curves)
    models = model_summary(
        rows, config=config, settings=settings,
        curves=curves, concepts=concepts, complexities=complexities,
    )
    return {
        "learning_curves": curves,
        "repetition_curves": rep_curves,
        "concept_summary": concepts,
        "complexity_summary": complexities,
        "model_summary": models,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print learning-curve and concept summary scores for BCL results.")
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    parser.add_argument("--json", action="store_true", help="print all computed score tables as JSON instead")
    args = parser.parse_args(argv)

    config = load_config_or_fallback(args.config)
    results_path = config.resolve(config.run["results_file"])
    rows = read_rows(results_path)
    if args.json:
        print(json.dumps(score_bundle(rows, config), indent=2))
    else:
        print(report(rows, config))


if __name__ == "__main__":
    main()
