"""Tests for ``bcl.probe`` residual-stream rule probing and superposed trial marginalization."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from bcl import blocks as bcl_blocks  # noqa: E402
from bcl.config import load_config  # noqa: E402
from bcl.probe import (  # noqa: E402
    auroc,
    compute_rule_posteriors_and_q_yes,
    fit_weighted_l2_logreg_dual,
    make_all_probes,
    parse_rule_to_ext,
    prepare_probe_prompts,
    superposed_marginalize_q_yes,
    top1_marginalize_q_yes,
)


def _make_synthetic_probe_env(tmp_path):
    cfg_path = tmp_path / "config_bcl.yaml"
    rules_src = (
        "FEATURES = {'color': ['red', 'blue'], 'shape': ['square', 'circle']}\n"
        "RULES = {\n"
        "    'red': \"lambda t: t['color'] == 'red'\",\n"
        "    'blue': \"lambda t: t['color'] == 'blue'\",\n"
        "    'square': \"lambda t: t['shape'] == 'square'\",\n"
        "    'red and square': \"lambda t: t['color'] == 'red' and t['shape'] == 'square'\",\n"
        "}\n"
    )
    (tmp_path / "tiny_rules.py").write_text(rules_src)
    cfg_path.write_text(
        """
seed: 17
rules_file: tiny_rules.py
alpha: 0.99
prior:
  kind: rational_rules
  hypothesis_space: grammar
  max_bool_nodes: 3
  grammar_weights: null
  dirichlet: 2.0
  temperature: null
blocks:
  file: data/bcl/blocks.jsonl
  blocks_per_concept: 10
  repetitions: 2
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
  bootstrap_samples: 30
  interval: 0.95
""".lstrip()
    )
    cfg = load_config(cfg_path)
    gen_blocks = bcl_blocks.generate(cfg)
    rule_by_name = {r.name: r for r in cfg.rules}
    concept_names = list(cfg.programs.keys())
    concept_exts = {
        c: tuple(1 if rule_by_name[c](st) else 0 for st in cfg.stimuli)
        for c in concept_names
    }

    rng = np.random.default_rng(12345)
    rows = []
    for blk in gen_blocks:
        labeled = [{**t["object"], "label": bool(t["truth"])} for t in blk["trials"]]
        traj = cfg.hypothesis_rules.trajectory(labeled, alpha=0.999)
        trials_out = []
        for t_idx, t in enumerate(blk["trials"]):
            post = traj[t_idx]
            # Pick MAP target rule among the 4 concepts (or 'guessing' on trial 1)
            if int(t["trial"]) == 1:
                rg = "guessing"
                p_pred = 0.5
            else:
                c_masses = []
                for cn in concept_names:
                    cext = concept_exts[cn]
                    mass = sum(
                        post[r_h.name] for r_h in cfg.hypothesis_rules
                        if tuple(1 if r_h(s) else 0 for s in cfg.stimuli) == cext
                    )
                    c_masses.append((mass, cn))
                c_masses.sort(reverse=True)
                rg = c_masses[0][1]
                p_pred = bcl_blocks.predictive_p_yes(post, cfg.hypothesis_rules, t["object"], 0.90)
            ans = bool(rng.random() < p_pred)
            trials_out.append({
                **t,
                "rule_guess": rg,
                "answer": ans,
                "correct": ans == bool(t["truth"]),
                "p_yes": 0.98 if ans else 0.02,
            })
        rows.append({
            **blk,
            "model": "mock-llm",
            "parse_failures": 0,
            "trials": trials_out,
        })
    return cfg, rows, concept_names, concept_exts


def test_parse_rule_to_ext_covers_english_idioms():
    stimuli = [
        {"color": "red", "shape": "square"},
        {"color": "red", "shape": "circle"},
        {"color": "blue", "shape": "square"},
        {"color": "blue", "shape": "circle"},
    ]
    feats = {"color": ["red", "blue"], "shape": ["square", "circle"]}

    assert parse_rule_to_ext("guessing", stimuli, feats) == "GUESSING"
    assert parse_rule_to_ext("The object is red.", stimuli, feats) == (1, 1, 0, 0)
    assert parse_rule_to_ext("not blue", stimuli, feats) == (1, 1, 0, 0)
    assert parse_rule_to_ext("either blue or a square", stimuli, feats) == (1, 0, 1, 1)
    assert parse_rule_to_ext("red, not square", stimuli, feats) == (0, 1, 0, 0)
    assert parse_rule_to_ext("red except square", stimuli, feats) == (0, 1, 0, 0)
    assert parse_rule_to_ext("unrecognized size big", stimuli, feats) is None


def test_prepare_probe_prompts_binary_framing_and_no_rule_leakage(tmp_path):
    cfg, rows, _, _ = _make_synthetic_probe_env(tmp_path)
    meta_path = tmp_path / "probe_meta.jsonl"
    bin_path = tmp_path / "probe_prompts.bin"

    n_written = prepare_probe_prompts(cfg, rows, meta_path, bin_path, blocks_per_concept=4, include_rule=False)
    meta_rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    assert len(meta_rows) == n_written
    assert n_written == 4 * 4 * 8  # 4 concepts * 4 blocks * 8 trials

    raw = bin_path.read_bytes()
    (n_rec_hdr,) = struct.unpack_from("<I", raw, 0)
    assert n_rec_hdr == n_written
    offset = 4
    for idx, mrow in enumerate(meta_rows):
        is_new, blen = struct.unpack_from("<II", raw, offset)
        offset += 8
        prompt_str = raw[offset:offset + blen].decode("utf-8")
        offset += blen
        assert is_new == (1 if mrow["trial"] == 1 else 0)
        assert prompt_str.endswith("<|turn>model\n<|channel>thought\n<channel|>Rule:")
        assert "rule guess" not in prompt_str.lower()
    assert offset == len(raw)


def test_weighted_dual_logreg_and_superposed_probe_pipeline(tmp_path):
    # 1. Check weighted dual logistic regression recovers known linear separator
    rng = np.random.default_rng(99)
    X_lin = rng.normal(size=(80, 16))
    true_w = np.zeros(16)
    true_w[0] = 3.0
    true_w[1] = -2.5
    y_lin = (X_lin @ true_w + 0.4 > 0).astype(np.float64)
    w_hat, b_hat = fit_weighted_l2_logreg_dual(X_lin, y_lin, np.ones(80), l2_reg=1.0)
    assert auroc(y_lin, X_lin @ w_hat + b_hat) > 0.99

    # 2. End-to-end synthetic residual stream with planted superposed rule posteriors
    cfg, rows, concept_names, concept_exts = _make_synthetic_probe_env(tmp_path)
    meta_path = tmp_path / "probe_meta.jsonl"
    bin_path = tmp_path / "probe_prompts.bin"
    acts_path = tmp_path / "probe_acts.f32.bin"
    prepare_probe_prompts(cfg, rows, meta_path, bin_path, blocks_per_concept=10, include_rule=False)
    meta_rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]

    bayes = compute_rule_posteriors_and_q_yes(
        cfg, meta_rows, concept_names, concept_exts, fitted_params=(0.90, 1.0, 1.0)
    )
    p_cond = bayes["p_10_fit"]  # (N, 4)

    n_rows = len(meta_rows)
    n_layers = 5
    n_embd = 24
    # Orthonormal concept directions in R^24
    dirs = np.eye(len(concept_names), n_embd, dtype=np.float64) * 5.0
    acts = np.zeros((n_rows, n_layers, n_embd), dtype=np.float32)

    # Layer 0: pure noise (must stay near chance = 0.50 AUROC with zero trial-number leakage)
    acts[:, 0, :] = rng.normal(scale=1.0, size=(n_rows, n_embd)).astype(np.float32)
    for lid, strength in ((1, 0.25), (2, 0.70), (3, 1.40), (4, 0.90)):
        # Superpose ALL consistent candidate directions weighted by p_cond!
        signal = (p_cond @ dirs) * strength
        noise = rng.normal(scale=0.35, size=(n_rows, n_embd))
        acts[:, lid, :] = (signal + noise).astype(np.float32)
    acts.tofile(acts_path)

    written, bundle = make_all_probes(
        cfg,
        [(meta_path, acts_path)],
        rows=rows,
        output_dir=tmp_path,
        n_embd=n_embd,
        n_folds=5,
    )

    names = {p.name for p in written}
    assert {
        "fig5_layer_rule_probes.png",
        "fig5_layer_rule_probes.pdf",
        "fig6_probe_trial_marginalization.png",
        "fig6_probe_trial_marginalization.pdf",
        "layer_rule_probes.csv",
        "layer_rule_probes.md",
        "probe_trial_predictions.csv",
        "probe_trial_predictions.md",
    } <= names
    for p in written:
        assert p.exists() and p.stat().st_size > 0

    # Verify Layer 0 stays near chance and Layer 3 achieves strong out-of-block decoding
    l0_mean_auc = float(np.mean([bundle["per_rule_layer_metrics"][c][0]["auroc"] for c in concept_names]))
    l3_mean_auc = float(np.mean([bundle["per_rule_layer_metrics"][c][3]["auroc"] for c in concept_names]))
    assert 0.35 <= l0_mean_auc <= 0.65
    assert l3_mean_auc > 0.92

    # Verify superposed marginalization beats Top-1 winner-take-all & binary token p_yes on trial RMSE
    sup_l3 = bundle["superposed_by_layer"][3]
    top1_l3 = bundle["top1_by_layer"][3]
    tok_ev = bundle["baselines_eval"]["token_p_yes"]
    assert sup_l3["rmse_trial_fit_bayes"] < top1_l3["rmse_trial_fit_bayes"]
    assert sup_l3["rmse_trial_fit_bayes"] < tok_ev["rmse_trial_fit_bayes"]

    # Direct unit check on marginalization boundary conditions
    p_test = np.array([[0.8, 0.4, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    e_test = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    q_sup = superposed_marginalize_q_yes(p_test, e_test)
    q_t1 = top1_marginalize_q_yes(p_test, e_test)
    assert q_sup[1] == pytest.approx(0.5, abs=1e-6)
    assert q_sup[0] < q_t1[0]  # 2nd active rule voting 0 pulls P(yes) below Top-1 winner-take-all!
