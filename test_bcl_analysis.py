"""Unit and integration tests for bcl.score and bcl.analysis on synthetic 36-trial blocks.

Run with:  .venv/bin/python -m pytest test_bcl_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matplotlib")

import example_rules  # noqa: E402
from bcl import analysis, score  # noqa: E402
from rule_learner import BooleanGrammar, RuleSet, all_stimuli  # noqa: E402

ROOT = Path(__file__).parent

EXPECTED_GROUPS = {
    "red": "single",
    "blue": "single",
    "square": "single",
    "circle": "single",
    "not red": "negation",
    "not green": "negation",
    "red and square": "conjunction",
    "red or square": "disjunction",
    "blue or square": "disjunction",
    "red and not square": "compound",
}


def make_bcl_settings(results_file="data/bcl/results.jsonl", bootstrap_samples=250, seed=20261113):
    """Complete config_bcl.yaml settings dict matching the shared BCL config schema."""
    return {
        "seed": seed,
        "rules_file": str(ROOT / "example_rules.py"),
        "alpha": 0.99,
        "prior": {
            "kind": "pcfg",
            "hypothesis_space": "rules_file",
            "max_bool_nodes": None,
            "grammar_weights": None,
            "dirichlet": None,
            "temperature": 3.0,
        },
        "blocks": {
            "file": "data/bcl/blocks.jsonl",
            "blocks_per_concept": 4,
            "repetitions": 4,
            "nonce_words": ["widge", "daxy", "lanch", "sowdy", "blira"],
        },
        "models": [{"name": "gemma4:26b", "backend": "ollama", "think": False, "extra_body": None}],
        "run": {
            "mode": "running_log_rule",
            "results_file": results_file,
            "timeout_seconds": 300,
            "temperature": 0.0,
            "max_tokens": 96,
            "top_logprobs": 20,
        },
        "backends": {
            "ollama": {"base_url": "http://localhost:11434", "options": {"num_ctx": 8192}, "reuse_cache": True},
            "openai": {"base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY"},
        },
        "scores": {"bootstrap_samples": bootstrap_samples, "interval": 0.95},
    }


def make_block_row(
    concept="red",
    program="lambda t: t['color'] == 'red'",
    complexity_group="single",
    run_index=0,
    model="gemma4:26b",
    correct_flags=None,
    bayes_p_yes_seq=None,
    p_yes_seq=None,
    parse_fail_indices=(),
):
    """Build one contract-compliant 36-trial JSONL block row."""
    stimuli = all_stimuli(example_rules.FEATURES)
    rule_fn = eval(program)  # noqa: S307 - trusted test rule from example_rules.py
    slug = concept.replace(" ", "_")
    block_id = f"{slug}-{run_index:03d}"

    if correct_flags is None:
        correct_flags = [True] * 36
    trials = []
    parse_failures = 0
    for idx in range(36):
        trial_num = idx + 1
        rep = idx // 9 + 1
        stim = stimuli[idx % len(stimuli)]
        truth = bool(rule_fn(stim))
        bp = 0.8 if truth else 0.2
        if bayes_p_yes_seq is not None:
            bp = float(bayes_p_yes_seq[idx])
        if abs(bp - 0.5) <= 1e-9:
            b_corr = 0.5
        else:
            b_corr = 1.0 if ((bp > 0.5) == truth) else 0.0

        if trial_num in parse_fail_indices:
            ans = None
            is_correct = False
            p_yes = None
            raw = "???"
            parse_failures += 1
        else:
            is_correct = bool(correct_flags[idx])
            ans = truth if is_correct else (not truth)
            raw = "yes" if ans else "no"
            if p_yes_seq is not None:
                p_yes = p_yes_seq[idx]
            else:
                p_yes = 0.85 if ans else 0.15

        trials.append({
            "trial": trial_num,
            "repetition": rep,
            "object": dict(stim),
            "truth": truth,
            "answer": ans,
            "raw_response": raw,
            "p_yes": p_yes,
            "correct": is_correct,
            "bayes_p_yes": bp,
            "bayes_correct": b_corr,
            "uniform_p_yes": bp,
            "cached_tokens": 0 if trial_num == 1 else 64 * idx,
            "seconds": 0.05,
        })

    return {
        "key": f"{model}|{block_id}",
        "fingerprint": "abcdef012345",
        "time": "2026-11-13T15:50:00Z",
        "model": model,
        "model_name": model,
        "model_digest": "sha256:test",
        "backend": "ollama",
        "backend_version": "0.6.0",
        "think": False,
        "block_id": block_id,
        "concept": concept,
        "program": program,
        "complexity_group": complexity_group,
        "run_index": run_index,
        "word": "widge",
        "n_trials": 36,
        "repetitions": 4,
        "seconds": 1.8,
        "parse_failures": parse_failures,
        "trials": trials,
    }


@pytest.fixture(scope="module")
def realistic_rows():
    """40 blocks (4 blocks x 10 concepts) with exact RuleSet.trajectory Bayesian predictions."""
    grammar = BooleanGrammar(example_rules.FEATURES)
    weights = grammar.prior_weights(example_rules.RULES, kind="pcfg", temperature=3.0)
    pcfg_rules = RuleSet.from_programs(example_rules.RULES, weights=weights)
    uniform_rules = RuleSet.from_programs(example_rules.RULES)
    stimuli = all_stimuli(example_rules.FEATURES)
    alpha = 0.99

    # Complexity-dependent learning speed to emulate Shepard/Goodman hierarchy
    lag_by_group = {
        "single": 0.92,
        "negation": 0.85,
        "conjunction": 0.78,
        "disjunction": 0.68,
        "compound": 0.58,
    }

    rng = random.Random(42)
    rows = []
    for concept, program in example_rules.RULES.items():
        cgroup = EXPECTED_GROUPS[concept]
        rule_obj = pcfg_rules[concept]
        for run_idx in range(4):
            seq_stimuli = []
            for _ in range(4):
                seq_stimuli.extend(rng.sample(stimuli, len(stimuli)))
            labeled = [{**s, "label": bool(rule_obj(s))} for s in seq_stimuli]
            traj = pcfg_rules.trajectory(labeled, alpha=alpha)
            uni_traj = uniform_rules.trajectory(labeled, alpha=alpha)

            slug = concept.replace(" ", "_")
            block_id = f"{slug}-{run_idx:03d}"
            trials = []
            for idx, item in enumerate(labeled):
                trial_num = idx + 1
                rep = idx // 9 + 1
                stim = {"color": item["color"], "shape": item["shape"]}
                truth = item["label"]

                post_before = traj[idx]
                bp = sum(
                    post_before[name] * pcfg_rules.label_probability(pcfg_rules[name], stim, alpha=alpha)
                    for name in pcfg_rules.names
                )
                uni_before = uni_traj[idx]
                up = sum(
                    uni_before[name] * uniform_rules.label_probability(uniform_rules[name], stim, alpha=alpha)
                    for name in uniform_rules.names
                )
                if abs(bp - 0.5) <= 1e-9:
                    b_corr = 0.5
                else:
                    b_corr = 1.0 if ((bp > 0.5) == truth) else 0.0

                # Model tracks Bayesian posterior partially, improving over trials
                skill = min(0.99, lag_by_group[cgroup] * (0.45 + 0.55 * (idx / 35.0)))
                model_p_correct = skill * (bp if truth else (1.0 - bp)) + (1.0 - skill) * 0.5
                p_yes = model_p_correct if truth else (1.0 - model_p_correct)
                ans = truth if (rng.random() < model_p_correct) else (not truth)
                trials.append({
                    "trial": trial_num,
                    "repetition": rep,
                    "object": stim,
                    "truth": truth,
                    "answer": ans,
                    "raw_response": "yes" if ans else "no",
                    "p_yes": p_yes,
                    "correct": ans is truth,
                    "bayes_p_yes": bp,
                    "bayes_correct": b_corr,
                    "uniform_p_yes": up,
                    "cached_tokens": 0 if trial_num == 1 else 50 * idx,
                    "seconds": 0.04,
                })

            rows.append({
                "key": f"gemma4:26b|{block_id}",
                "fingerprint": "123456abcdef",
                "time": "2026-11-13T15:55:00Z",
                "model": "gemma4:26b",
                "model_name": "gemma4:26b",
                "model_digest": "sha256:gemma",
                "backend": "ollama",
                "backend_version": "0.6.0",
                "think": False,
                "block_id": block_id,
                "concept": concept,
                "program": program,
                "complexity_group": cgroup,
                "run_index": run_idx,
                "word": "widge",
                "n_trials": 36,
                "repetitions": 4,
                "seconds": 1.44,
                "parse_failures": 0,
                "trials": trials,
            })
    return rows


# ---------------------------------------------------------------------------
# Unit tests: bcl.score metrics & bootstrap
# ---------------------------------------------------------------------------

def test_ast_complexity_fallback_matches_all_ten_rules():
    for name, program in example_rules.RULES.items():
        assert score.infer_complexity_group(program) == EXPECTED_GROUPS[name]


def test_trials_to_criterion_boundary_cases():
    # 0 errors anywhere -> trial 1
    assert score.trials_to_criterion([{"trial": t, "correct": True} for t in range(1, 37)]) == 1
    # Error only on trial 1 -> trial 2
    assert score.trials_to_criterion([{"trial": t, "correct": t > 1} for t in range(1, 37)]) == 2
    # Errors up to trial 18, then 0 errors for trials 19..36 -> trial 19
    assert score.trials_to_criterion([{"trial": t, "correct": t not in (3, 11, 18)} for t in range(1, 37)]) == 19
    # Last error on trial 35, trial 36 accurate -> trial 36
    assert score.trials_to_criterion([{"trial": t, "correct": t != 35} for t in range(1, 37)]) == 36
    # Error on trial 36 -> 37 (never reaches error-free rest of block)
    assert score.trials_to_criterion([{"trial": t, "correct": t != 36} for t in range(1, 37)]) == 37
    # All 36 trials wrong -> 37
    assert score.trials_to_criterion([{"trial": t, "correct": False} for t in range(1, 37)]) == 37
    # Unsorted input trials still evaluate in trial order
    shuffled = [{"trial": t, "correct": t > 10} for t in range(36, 0, -1)]
    assert score.trials_to_criterion(shuffled) == 11


def test_block_errors_repetitions_and_parse_failures():
    # Rep 1 (1..9): 5/9, Rep 2 (10..18): 7/9, Rep 3 (19..27): 8/9, Rep 4 (28..36): 9/9
    flags = (
        [True] * 5 + [False] * 4
        + [True] * 7 + [False] * 2
        + [True] * 8 + [False] * 1
        + [True] * 9
    )
    row = make_block_row(correct_flags=flags, parse_fail_indices=(9, 18))
    # Trials 9 and 18 were already False in flags; total accurate = 5 + 7 + 8 + 9 = 29
    assert score.block_total_errors(row) == 36 - 29 == 7
    assert score.block_accuracy(row) == pytest.approx(29 / 36)
    reps = score.block_repetition_accuracies(row)
    assert reps[1] == pytest.approx(5 / 9)
    assert reps[2] == pytest.approx(7 / 9)
    assert reps[3] == pytest.approx(8 / 9)
    assert reps[4] == pytest.approx(9 / 9)
    bayes_reps = score.block_bayes_repetition_accuracies(row)
    assert all(v == pytest.approx(1.0) for v in bayes_reps.values())
    # Last error in flags is at trial 27 (end of rep 3) -> TTC = 28
    assert score.trials_to_criterion(row["trials"]) == 28
    assert score.block_parse_failures(row) == 2


def test_calibration_tvd_handles_present_and_missing_p_yes():
    b1 = make_block_row(run_index=0, bayes_p_yes_seq=[0.8] * 36, p_yes_seq=[0.6] * 36)
    b2 = make_block_row(run_index=1, bayes_p_yes_seq=[0.8] * 36, p_yes_seq=[None] * 36)
    curves = score.learning_curves([b1, b2], settings={"seed": 1, "scores": {"bootstrap_samples": 100, "interval": 0.95}})
    t1 = next(r for r in curves if r["scope_type"] == "concept" and r["trial"] == 1)
    assert t1["calibration_tvd"] == pytest.approx(0.2)
    assert t1["n_p_yes"] == 1

    # When no block has p_yes, calibration_tvd is None
    only_none = score.learning_curves([b2])
    assert all(r["calibration_tvd"] is None and r["n_p_yes"] == 0 for r in only_none)


def test_pearson_r_and_rmse_edge_cases():
    assert score.pearson_r([0.1, 0.2, 0.3], [0.2, 0.4, 0.6]) == pytest.approx(1.0)
    assert score.pearson_r([0.1, 0.2, 0.3], [0.6, 0.4, 0.2]) == pytest.approx(-1.0)
    assert score.pearson_r([0.5, 0.5, 0.5], [0.1, 0.5, 0.9]) is None
    assert score.pearson_r([0.5], [0.5]) is None
    assert score.rmse([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0)
    assert score.rmse([0.0, 1.0], [0.3, 0.6]) == pytest.approx(math.sqrt((0.09 + 0.16) / 2))


def test_bootstrap_resamples_whole_blocks_and_matches_per_trial():
    # 4 blocks: 2 perfect blocks and 2 all-wrong blocks
    blocks = [
        make_block_row(run_index=0, correct_flags=[True] * 36),
        make_block_row(run_index=1, correct_flags=[False] * 36),
        make_block_row(run_index=2, correct_flags=[True] * 36),
        make_block_row(run_index=3, correct_flags=[False] * 36),
    ]
    cfg = {"seed": 7, "scores": {"bootstrap_samples": 300, "interval": 0.95}}
    curves = [r for r in score.learning_curves(blocks, settings=cfg) if r["scope_type"] == "concept"]
    assert len(curves) == 36
    for r in curves:
        assert r["correct"] == pytest.approx(0.5)
        assert {r["low"], r["high"]} <= {0.0, 0.25, 0.5, 0.75, 1.0}
        assert r["low"] <= r["correct"] <= r["high"]
        # Exact parity with scalar mean_with_interval at every trial
        pairs = [(b["block_id"], 1.0 if b["trials"][r["trial"] - 1]["correct"] else 0.0) for b in blocks]
        m, lo, hi = score.mean_with_interval(pairs, 300, 0.95, 7)
        assert (r["correct"], r["low"], r["high"]) == (m, lo, hi)

    # Single block has no bootstrap interval (low=None, high=None)
    single_curve = score.learning_curves([blocks[0]], settings=cfg)
    assert all(r["low"] is None and r["high"] is None for r in single_curve)


def test_learning_curves_and_summaries_cover_all_scopes(realistic_rows):
    cfg = {"seed": 20261113, "scores": {"bootstrap_samples": 200, "interval": 0.95}}
    curves = score.learning_curves(realistic_rows, settings=cfg)
    # 10 concepts + 5 complexity groups + 1 "all" = 16 scopes x 36 trials = 576 rows
    assert len(curves) == 16 * 36

    by_type = {st: {r["scope"] for r in curves if r["scope_type"] == st}
               for st in ("concept", "complexity_group", "all")}
    assert by_type["concept"] == set(example_rules.RULES)
    assert by_type["complexity_group"] == set(score.COMPLEXITY_GROUPS)
    assert by_type["all"] == {"all"}

    c_sum = score.concept_summary(realistic_rows, settings=cfg, curves=curves)
    assert len(c_sum) == 10
    for rec in c_sum:
        assert 0.0 <= rec["accuracy"] <= 1.0
        assert rec["low"] <= rec["accuracy"] <= rec["high"]
        assert rec["total_errors"] == pytest.approx(36.0 * (1.0 - rec["accuracy"]))
        assert 1.0 <= rec["trials_to_criterion"] <= 37.0
        assert rec["curve_rmse"] >= 0.0

    m_sum = score.model_summary(realistic_rows, settings=cfg, curves=curves, concepts=c_sum)
    assert len(m_sum) == 1
    assert m_sum[0]["n_blocks"] == 40
    assert m_sum[0]["n_concepts"] == 10
    assert m_sum[0]["concept_r"] is not None and -1.0 <= m_sum[0]["concept_r"] <= 1.0
    assert m_sum[0]["concept_trial_r"] is not None and -1.0 <= m_sum[0]["concept_trial_r"] <= 1.0


def test_score_cli_report_and_json(tmp_path, realistic_rows, capsys):
    cfg_path = tmp_path / "config_bcl.yaml"
    cfg_path.write_text(yaml.safe_dump(make_bcl_settings(bootstrap_samples=150), sort_keys=False))

    # Empty results -> "No results yet."
    score.main(["--config", str(cfg_path)])
    assert capsys.readouterr().out.strip() == "No results yet."

    # Write results.jsonl and run plain-text report + --json mode
    res_path = tmp_path / "data" / "bcl" / "results.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text("".join(json.dumps(row) + "\n" for row in realistic_rows))

    score.main(["--config", str(cfg_path)])
    text = capsys.readouterr().out
    assert "=== gemma4:26b ===" in text
    assert "Per-concept learning summary:" in text
    assert "Boolean complexity hierarchy:" in text
    assert "Alignment with ideal Bayesian learner:" in text
    assert "red and not square" in text

    score.main(["--config", str(cfg_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "learning_curves", "repetition_curves", "concept_summary", "complexity_summary", "model_summary",
    }
    assert len(payload["learning_curves"]) == 576
    assert len(payload["concept_summary"]) == 10


# ---------------------------------------------------------------------------
# Integration tests: bcl.analysis figures & tables
# ---------------------------------------------------------------------------

def test_analysis_cli_writes_all_figures_and_tables(tmp_path, realistic_rows, capsys):
    cfg_path = tmp_path / "config_bcl.yaml"
    cfg_text = yaml.safe_dump(make_bcl_settings(bootstrap_samples=150), sort_keys=False)
    cfg_path.write_text(cfg_text)

    res_path = tmp_path / "data" / "bcl" / "results.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text("".join(json.dumps(row) + "\n" for row in realistic_rows))

    analysis.main(["--config", str(cfg_path)])
    printed = capsys.readouterr().out

    expected_files = [
        "figures/bcl/fig1_concept_learning_curves.pdf",
        "figures/bcl/fig1_concept_learning_curves.png",
        "figures/bcl/fig2_complexity_hierarchy.pdf",
        "figures/bcl/fig2_complexity_hierarchy.png",
        "figures/bcl/fig3_model_vs_bayes.pdf",
        "figures/bcl/fig3_model_vs_bayes.png",
        "tables/bcl/learning_curves.csv",
        "tables/bcl/concept_summary.csv",
        "tables/bcl/concept_summary.md",
        "figures/bcl/figures.config.yaml",
        "tables/bcl/tables.config.yaml",
    ]
    for rel in expected_files:
        path = tmp_path / rel
        assert path.exists() and path.stat().st_size > 200, f"missing or tiny output: {rel}"
        assert str(path) in printed

    for fig_name in ("fig1_concept_learning_curves", "fig2_complexity_hierarchy", "fig3_model_vs_bayes"):
        assert (tmp_path / f"figures/bcl/{fig_name}.pdf").read_bytes().startswith(b"%PDF")
        assert (tmp_path / f"figures/bcl/{fig_name}.png").read_bytes().startswith(b"\x89PNG")

    assert (tmp_path / "tables/bcl/tables.config.yaml").read_text() == cfg_text

    # Verify CSV contents match bcl.score calculations
    bundle = score.score_bundle(realistic_rows, config=cfg_path)
    with open(tmp_path / "tables/bcl/learning_curves.csv", newline="") as f:
        lc_csv = list(csv.DictReader(f))
    assert len(lc_csv) == 576
    assert {r["scope_type"] for r in lc_csv} == {"concept", "complexity_group", "all"}

    with open(tmp_path / "tables/bcl/concept_summary.csv", newline="") as f:
        cs_csv = list(csv.DictReader(f))
    assert len(cs_csv) == 10
    expected_by_concept = {r["concept"]: r for r in bundle["concept_summary"]}
    for line in cs_csv:
        ref = expected_by_concept[line["concept"]]
        assert float(line["accuracy"]) == pytest.approx(ref["accuracy"])
        assert float(line["bayes_accuracy"]) == pytest.approx(ref["bayes_accuracy"])
        assert float(line["total_errors"]) == pytest.approx(ref["total_errors"])
        assert float(line["trials_to_criterion"]) == pytest.approx(ref["trials_to_criterion"])

    md = (tmp_path / "tables/bcl/concept_summary.md").read_text()
    assert md.startswith("| Concept | Complexity | Blocks | Rep 1 (1–9) | Rep 2 (10–18) |")
    for concept in example_rules.RULES:
        assert f"| {concept} |" in md
    assert "| **All concepts** |" in md
    assert "95% bootstrap intervals" in md


def test_figure_panels_and_visual_elements(realistic_rows):
    bundle = score.score_bundle(realistic_rows, settings={"seed": 1, "scores": {"bootstrap_samples": 100, "interval": 0.95}})

    fig1 = analysis.figure_concept_learning_curves(bundle["learning_curves"])
    visible1 = [ax for ax in fig1.axes if ax.get_visible()]
    assert len(visible1) == 10
    assert {ax.get_title(loc="left") for ax in visible1} == set(example_rules.RULES)
    # Shaded 95% CI ribbon (PolyCollection) present on every panel
    assert all(len(ax.collections) >= 1 for ax in visible1)
    analysis.plt.close(fig1)

    fig2 = analysis.figure_complexity_hierarchy(bundle["learning_curves"], bundle["repetition_curves"])
    visible2 = [ax for ax in fig2.axes if ax.get_visible()]
    assert len(visible2) == 2
    analysis.plt.close(fig2)

    fig3 = analysis.figure_model_vs_bayes(
        bundle["learning_curves"], bundle["concept_summary"], bundle["model_summary"],
    )
    visible3 = [ax for ax in fig3.axes if ax.get_visible()]
    assert len(visible3) == 2
    analysis.plt.close(fig3)


def test_analysis_supports_direct_rows_multi_model_and_empty_error(tmp_path, realistic_rows):
    # Empty results file raises SystemExit
    cfg_path = tmp_path / "empty_config.yaml"
    cfg_path.write_text(yaml.safe_dump(make_bcl_settings(), sort_keys=False))
    with pytest.raises(SystemExit, match="no rows"):
        analysis.make_all(cfg_path)

    # Multi-model direct rows call without config file on disk
    second_model_rows = [
        {**row, "model": "second-model", "key": f"second-model|{row['block_id']}"}
        for row in realistic_rows[:20]
    ]
    out_dir = tmp_path / "direct_out"
    written = analysis.make_all(
        {"seed": 1, "scores": {"bootstrap_samples": 80, "interval": 0.95}},
        rows=realistic_rows[:20] + second_model_rows,
        output_dir=out_dir,
    )
    assert len(written) == 9    # 6 figure files + 3 table files
    md = (out_dir / "tables/bcl/concept_summary.md").read_text()
    assert "## gemma4:26b" in md and "## second-model" in md


# ---------------------------------------------------------------------------
# Single-pass 16-stimulus (repetitions == 1) quartile scoring & plotting
# ---------------------------------------------------------------------------

def make_single_pass_16_row(
    concept="red",
    program="lambda t: t['color'] == 'red'",
    complexity_group="single",
    run_index=0,
    model="gemma4:26b",
    correct_flags=None,
    include_repetitions_field=True,
):
    """Build one 16-trial single-pass block row over the 4x4 bcl_rules stimulus space."""
    import bcl_rules

    stimuli = all_stimuli(bcl_rules.FEATURES)
    assert len(stimuli) == 16
    rule_fn = eval(program)  # noqa: S307 - trusted test rule from bcl_rules.py
    slug = concept.replace(" ", "_")
    block_id = f"{slug}-{run_index:03d}"

    if correct_flags is None:
        correct_flags = [True] * 16
    trials = []
    for idx, stim in enumerate(stimuli):
        trial_num = idx + 1
        truth = bool(rule_fn(stim))
        # Monotonically strengthening Bayesian ideal across the 16 unique trials
        conf = 0.55 + 0.40 * (idx / 15.0)
        bp = conf if truth else (1.0 - conf)
        b_corr = 1.0 if ((bp > 0.5) == truth) else 0.0
        is_correct = bool(correct_flags[idx])
        ans = truth if is_correct else (not truth)
        trials.append({
            "trial": trial_num,
            "repetition": 1,
            "object": dict(stim),
            "truth": truth,
            "answer": ans,
            "raw_response": "yes" if ans else "no",
            "p_yes": 0.85 if ans else 0.15,
            "correct": is_correct,
            "bayes_p_yes": bp,
            "bayes_correct": b_corr,
            "uniform_p_yes": bp,
            "cached_tokens": 0 if trial_num == 1 else 64 * idx,
            "seconds": 0.04,
        })

    row = {
        "key": f"{model}|{block_id}",
        "fingerprint": "16obj0001pass",
        "time": "2026-11-13T18:10:00Z",
        "model": model,
        "model_name": model,
        "model_digest": "sha256:test16",
        "backend": "ollama",
        "backend_version": "0.6.0",
        "think": False,
        "block_id": block_id,
        "concept": concept,
        "program": program,
        "complexity_group": complexity_group,
        "run_index": run_index,
        "word": "widge",
        "n_trials": 16,
        "repetitions": 1,
        "seconds": 0.64,
        "parse_failures": 0,
        "trials": trials,
    }
    if not include_repetitions_field:
        row.pop("repetitions")
    return row


def test_single_pass_16_trial_quartile_scoring_and_plotting(tmp_path):
    import bcl_rules

    # Exact quarter accuracies: Q1 (1-4)=1/4, Q2 (5-8)=2/4, Q3 (9-12)=3/4, Q4 (13-16)=4/4
    q_flags = (
        [True, False, False, False]
        + [True, True, False, False]
        + [True, True, True, False]
        + [True, True, True, True]
    )
    row_explicit = make_single_pass_16_row(correct_flags=q_flags, include_repetitions_field=True)
    row_inferred = make_single_pass_16_row(correct_flags=q_flags, include_repetitions_field=False)

    for r_test in (row_explicit, row_inferred):
        assert score.is_single_pass_block(r_test) is True
        q_acc = score.block_repetition_accuracies(r_test)
        assert q_acc == {
            1: pytest.approx(0.25),
            2: pytest.approx(0.50),
            3: pytest.approx(0.75),
            4: pytest.approx(1.00),
        }
        assert score.block_total_errors(r_test) == 6
        # Last error is on trial 12 -> criterion reached at trial 13
        assert score.trials_to_criterion(r_test["trials"]) == 13

    # Error on trial 16 -> TTC = 17
    err_end = make_single_pass_16_row(correct_flags=[True] * 15 + [False])
    assert score.trials_to_criterion(err_end["trials"]) == 17

    # Build 30 blocks (3 blocks x 10 concepts) with realistic variation for end-to-end report & plots
    rng = random.Random(16)
    rows_16 = []
    for concept, program in bcl_rules.RULES.items():
        cgroup = EXPECTED_GROUPS[concept]
        for run_idx in range(3):
            flags = [rng.random() < (0.45 + 0.50 * (i / 15.0)) for i in range(16)]
            rows_16.append(make_single_pass_16_row(
                concept=concept, program=program, complexity_group=cgroup,
                run_index=run_idx, correct_flags=flags,
            ))

    cfg_16 = make_bcl_settings(bootstrap_samples=120)
    cfg_16["rules_file"] = str(ROOT / "bcl_rules.py")
    cfg_16["blocks"]["repetitions"] = 1
    cfg_path = tmp_path / "config_bcl_16.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_16, sort_keys=False))

    res_path = tmp_path / "data" / "bcl" / "results.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text("".join(json.dumps(r) + "\n" for r in rows_16))

    bundle = score.score_bundle(rows_16, config=cfg_path)
    # 16 scopes x 16 trials = 256 learning_curve rows; 16 scopes x 4 quarters = 64 repetition_curve rows
    assert len(bundle["learning_curves"]) == 16 * 16
    assert len(bundle["repetition_curves"]) == 16 * 4
    assert {r["repetition"] for r in bundle["repetition_curves"]} == {1, 2, 3, 4}

    txt = score.report(rows_16, config=cfg_path)
    for q_lbl in ("Q1 (1\u20134)", "Q2 (5\u20138)", "Q3 (9\u201312)", "Q4 (13\u201316)"):
        assert q_lbl in txt
    assert "16 trials/block" in txt
    assert "overall 16-trial curve r =" in txt
    assert "concept x trial (160 pts)" in txt
    assert "36-trial" not in txt and "1\u201336" not in txt and "36 trials" not in txt
    assert "rep 1" not in txt.lower()

    written = analysis.make_all(cfg_path)
    assert len(written) == 11

    md = (tmp_path / "tables/bcl/concept_summary.md").read_text()
    assert "| Concept | Complexity | Blocks | Q1 (1\u20134) | Q2 (5\u20138) | Q3 (9\u201312) | Q4 (13\u201316) |" in md
    assert "| Boolean complexity | Blocks | Q1 (1\u20134) | Q2 (5\u20138) | Q3 (9\u201312) | Q4 (13\u201316) |" in md
    assert "Overall 16-trial learning curve alignment" in md
    assert "(160 points)" in md
    assert "36-trial" not in md and "1\u201336" not in md and "Rep 1" not in md

    fig1 = analysis.figure_concept_learning_curves(bundle["learning_curves"])
    visible1 = [ax for ax in fig1.axes if ax.get_visible()]
    assert [int(x) for x in visible1[0].get_xticks()] == [1, 4, 8, 12, 16]
    assert fig1._supxlabel.get_text() == "Trial (16 unique objects, 1 pass \u2014 no repetitions)"
    analysis.plt.close(fig1)

    fig2 = analysis.figure_complexity_hierarchy(bundle["learning_curves"], bundle["repetition_curves"])
    ax_trial, ax_q = fig2.axes[0], fig2.axes[1]
    assert ax_trial.get_title(loc="left") == "By trial (1\u201316)"
    assert [int(x) for x in ax_trial.get_xticks()] == [1, 4, 8, 12, 16]
    assert ax_q.get_title(loc="left") == "By block quarter (model solid, Bayes dashed)"
    assert ax_q.get_xlabel() == "Trial quarter (no stimulus repetitions)"
    assert [t.get_text() for t in ax_q.get_xticklabels()] == [
        "Q1 (1\u20134)", "Q2 (5\u20138)", "Q3 (9\u201312)", "Q4 (13\u201316)",
    ]
    analysis.plt.close(fig2)

    fig3 = analysis.figure_model_vs_bayes(
        bundle["learning_curves"], bundle["concept_summary"], bundle["model_summary"],
    )
    assert fig3.axes[1].get_title(loc="left") == "Per concept \u00d7 trial (160 points)"
    analysis.plt.close(fig3)

