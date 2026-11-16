"""Tests for ``bcl.fit`` cognitive parameter fitting and LOCO cross-validation."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from bcl import blocks as bcl_blocks  # noqa: E402
from bcl.config import load_config  # noqa: E402
from bcl.fit import _HypothesisEvaluator, fit_all_models, make_all_fits  # noqa: E402


def _make_synthetic_results(tmp_path):
    cfg_path = tmp_path / "config_bcl.yaml"
    rules_src = (
        "FEATURES = {'color': ['red', 'blue'], 'shape': ['square', 'circle']}\n"
        "RULES = {\n"
        "    'red': \"lambda t: t['color'] == 'red'\",\n"
        "    'not red': \"lambda t: t['color'] != 'red'\",\n"
        "    'red and square': \"lambda t: t['color'] == 'red' and t['shape'] == 'square'\",\n"
        "}\n"
    )
    (tmp_path / "tiny_rules.py").write_text(rules_src)
    cfg_path.write_text(
        """
seed: 42
rules_file: tiny_rules.py
alpha: 0.99
prior:
  kind: rational_rules
  hypothesis_space: grammar
  max_bool_nodes: 4
  grammar_weights: null
  dirichlet: 2.0
  temperature: null
blocks:
  file: data/bcl/blocks.jsonl
  blocks_per_concept: 4
  repetitions: 1
  nonce_words: [lanch, sowdy]
models:
  - name: mock-llm
    backend: ollama
    think: false
    extra_body: null
run:
  mode: running_log_rule
  results_file: data/bcl/results.jsonl
  timeout_seconds: 30
  temperature: 0.0
  max_tokens: 64
  top_logprobs: 5
backends:
  ollama:
    base_url: http://localhost:11434
    options: {num_ctx: 2048}
    reuse_cache: true
  openai:
    base_url_env: OPENAI_BASE_URL
    api_key_env: OPENAI_API_KEY
scores:
  bootstrap_samples: 50
  interval: 0.95
""".lstrip()
    )
    cfg = load_config(cfg_path)
    gen_blocks = bcl_blocks.generate(cfg)

    rows = []
    for blk in gen_blocks:
        trials_out = []
        for t in blk["trials"]:
            # Deterministic imperfect learner: answers right on trials >= 2, flips trial 1
            ans = (not t["truth"]) if t["trial"] == 1 else bool(t["truth"])
            trials_out.append({
                **t,
                "answer": ans,
                "correct": ans == t["truth"],
                "p_yes": 0.75 if ans else 0.25,
            })
        rows.append({
            **blk,
            "model": "mock-llm",
            "parse_failures": 0,
            "trials": trials_out,
        })
    return cfg, rows


def test_evaluator_exact_parity_with_ruleset_trajectory(tmp_path):
    cfg, rows = _make_synthetic_results(tmp_path)
    ev = _HypothesisEvaluator(cfg, rows)

    for alpha in (0.99, 0.75):
        for prior_temp in (1.0, 2.5):
            q_pred = ev.predict_q_yes(alpha, beta=1.0, prior_temp=prior_temp)
            for b_idx, row in enumerate(rows):
                labeled = [{**t["object"], "label": bool(t["truth"])} for t in row["trials"]]
                traj = cfg.hypothesis_rules.trajectory(
                    labeled, alpha=alpha, prior_temperature=prior_temp
                )
                for t_idx, t in enumerate(row["trials"]):
                    expected = bcl_blocks.predictive_p_yes(
                        traj[t_idx], cfg.hypothesis_rules, t["object"], alpha
                    )
                    assert q_pred[b_idx, t_idx] == pytest.approx(expected, abs=1e-12)


def test_nested_mle_monotonicity_and_outputs(tmp_path):
    cfg, rows = _make_synthetic_results(tmp_path)
    bundle = fit_all_models(cfg, rows=rows)
    by_id = {r["fit_id"]: r for r in bundle["summary_rows"]}

    assert by_id["fit_alpha"]["nll"] <= by_id["prob_matching"]["nll"] + 1e-5
    assert by_id["fit_beta"]["nll"] <= by_id["prob_matching"]["nll"] + 1e-5
    assert by_id["fit_alpha_beta"]["nll"] <= by_id["fit_alpha"]["nll"] + 1e-4
    assert by_id["fit_alpha_beta"]["nll"] <= by_id["fit_beta"]["nll"] + 1e-4
    assert by_id["fit_alpha_beta_prior_temp"]["nll"] <= by_id["fit_alpha_prior_temp"]["nll"] + 1e-4

    written = make_all_fits(cfg, rows=rows, output_dir=tmp_path)
    names = {p.name for p in written}
    assert "fig4_fitted_rational_rules.png" in names
    assert "fig4_fitted_rational_rules.pdf" in names
    assert "parameter_fits.csv" in names
    assert "parameter_fits.md" in names
    for p in written:
        assert p.exists() and p.stat().st_size > 0
