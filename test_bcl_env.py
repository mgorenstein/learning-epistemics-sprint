"""Tests for bcl.config, bcl.blocks, and bcl.elicit.

Run with:  .venv/bin/python -m pytest test_bcl_env.py
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from bcl import blocks, elicit
from bcl.config import COMPLEXITY_GROUPS, complexity_group, load_config, model_label
from example_rules import FEATURES, RULES
from rule_learner import BooleanGrammar, Rule, RuleSet

ROOT = Path(__file__).parent


@pytest.fixture
def bcl_settings():
    """Small, self-contained valid config settings for fast unit tests."""
    return {
        "seed": 42,
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
            "blocks_per_concept": 3,
            "repetitions": 4,
            "nonce_words": ["lanch", "sowdy", "widge", "pluic"],
        },
        "models": [
            {"name": "test-model", "backend": "ollama", "think": False, "extra_body": None},
        ],
        "run": {
            "mode": "multiturn_yesno",
            "results_file": "data/bcl/results.jsonl",
            "timeout_seconds": 30,
            "temperature": 0.0,
            "max_tokens": 4,
            "top_logprobs": 20,
        },
        "backends": {
            "ollama": {
                "base_url": "http://localhost:11434",
                "options": {"num_ctx": 8192},
                "reuse_cache": True,
            },
            "openai": {
                "base_url_env": "OPENAI_BASE_URL",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
        "scores": {"bootstrap_samples": 100, "interval": 0.95},
    }


@pytest.fixture
def make_bcl_config(tmp_path):
    """Write a settings dict into a temporary directory and load it via bcl.config.load_config."""
    counter = 0

    def _make(settings_dict, folder=None):
        nonlocal counter
        counter += 1
        target_dir = folder or (tmp_path / f"cfg_{counter}")
        target_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = target_dir / "config_bcl.yaml"
        cfg_path.write_text(yaml.safe_dump(settings_dict, sort_keys=False))
        return load_config(cfg_path)

    return _make


# ---------------------------------------------------------------------------
# Config & complexity classification
# ---------------------------------------------------------------------------

def test_project_config_bcl_yaml_loads_and_matches_spec():
    config = load_config(ROOT / "config_bcl.yaml")
    hh_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    assert config.seed == 20261113
    assert config.settings["rules_file"] == "bcl_rules.py"
    assert len(config.stimuli) == 16
    assert config.alpha == 0.99
    assert config.settings["prior"] == {
        "kind": "rational_rules",
        "hypothesis_space": "grammar",
        "max_bool_nodes": 6,
        "grammar_weights": None,
        "dirichlet": 2.0,
        "temperature": None,
    }
    assert config.n_grammar_derivations == 27184
    assert len(config.hypothesis_rules) == 830
    assert sum(config.hypothesis_rules.prior().values()) == pytest.approx(1.0)
    assert list(config.uniform_rules.prior().values()) == pytest.approx([1.0 / 830] * 830)
    assert config.blocks["file"] == "data/bcl/blocks.jsonl"
    assert config.blocks["blocks_per_concept"] == 100
    assert config.blocks["repetitions"] == 1
    assert len(config.blocks["nonce_words"]) == 40
    assert config.blocks["nonce_words"] == hh_cfg["puzzles"]["nonce_words"]
    assert config.settings["models"] == [
        {"name": "gemma4:26b", "backend": "ollama", "think": False, "extra_body": None}
    ]
    assert config.run == {
        "mode": "running_log_rule",
        "results_file": "data/bcl/results.jsonl",
        "timeout_seconds": 300,
        "temperature": 0.0,
        "max_tokens": 96,
        "top_logprobs": 20,
    }
    assert config.settings["backends"]["ollama"] == {
        "base_url": "http://localhost:11434",
        "options": {"num_ctx": 8192},
        "reuse_cache": True,
    }
    assert config.scores == {"bootstrap_samples": 2000, "interval": 0.95}
    assert sum(config.rules.prior().values()) == pytest.approx(1.0)
    # Symmetric grammar prior gives identical prior mass to 'not red' and 'not green', and to all 4 singles
    tp = config.target_concept_priors
    assert tp["not red"] == pytest.approx(tp["not green"])
    assert tp["red"] == pytest.approx(tp["blue"]) == pytest.approx(tp["square"]) == pytest.approx(tp["circle"])
    assert tp["red"] > tp["not red"] > tp["red and square"] > tp["red and not square"]


def test_complexity_group_classifies_all_ten_rules_and_ast_variants():
    expected = {
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
    for name, group in expected.items():
        assert complexity_group(name) == group
        assert complexity_group(name, RULES) == group
        assert complexity_group(RULES[name]) == group
        assert complexity_group(Rule(name, RULES[name])) == group

    assert set(expected.values()) == set(COMPLEXITY_GROUPS)

    # Additional AST forms
    assert complexity_group("lambda t: not (t['color'] == 'red')") == "negation"
    assert complexity_group("lambda t: not (t['color'] != 'red')") == "compound"
    assert complexity_group("lambda t: t['color'] == 'red' or not (t['shape'] == 'square')") == "compound"
    assert complexity_group(
        "lambda t: (t['color'] == 'red' and t['shape'] == 'square') or t['color'] == 'blue'"
    ) == "compound"
    assert complexity_group(
        "lambda t: not (t['color'] == 'red' and t['shape'] == 'square')"
    ) == "compound"

    with pytest.raises(KeyError):
        complexity_group("nonexistent rule")
    with pytest.raises(ValueError):
        complexity_group("lambda a, b: a == b")
    with pytest.raises(ValueError):
        complexity_group("lambda t: len(t) == 2")


def test_config_priors_and_save_copy(bcl_settings, make_bcl_config, tmp_path):
    s_uni = copy.deepcopy(bcl_settings)
    s_uni["prior"].update(kind="uniform", temperature=None)
    cfg_uni = make_bcl_config(s_uni)
    assert list(cfg_uni.rules.prior().values()) == pytest.approx([0.1] * 10)

    s_rr = copy.deepcopy(bcl_settings)
    s_rr["prior"].update(kind="rational_rules", dirichlet=2.0, temperature=None)
    cfg_rr = make_bcl_config(s_rr)
    grammar = BooleanGrammar(FEATURES)
    expected_rr = RuleSet.from_programs(
        RULES, weights=grammar.prior_weights(RULES, kind="rational_rules", dirichlet=2.0)
    ).prior()
    assert cfg_rr.rules.prior() == pytest.approx(expected_rr)

    cfg_path = tmp_path / "commented.yaml"
    cfg_path.write_text(yaml.safe_dump(bcl_settings, sort_keys=False) + "# keep this comment\n")
    loaded = load_config(cfg_path)
    copied = loaded.save_copy(tmp_path / "out.jsonl")
    assert copied == tmp_path / "out.config.yaml"
    assert copied.read_text().endswith("# keep this comment\n")


@pytest.mark.parametrize("section, key, bad_value, match", [
    (None, "alpha", 1.2, "alpha must be"),
    (None, "seed", True, "seed must be"),
    ("prior", "kind", "magic", "prior.kind must be"),
    ("prior", "hypothesis_space", "mcmc", "prior.hypothesis_space must be"),
    ("prior", "max_bool_nodes", 4, "prior.max_bool_nodes has no effect"),
    ("prior", "dirichlet", 1.0, "prior.dirichlet has no effect"),
    ("blocks", "blocks_per_concept", 0, "blocks.blocks_per_concept must be"),
    ("blocks", "repetitions", 0, "blocks.repetitions must be"),
    ("blocks", "nonce_words", ["dup", "dup"], "blocks.nonce_words must be"),
    ("run", "mode", "unknown_mode", "run.mode must be"),
    ("run", "max_tokens", 0, "run.max_tokens must be"),
    ("run", "top_logprobs", 25, "run.top_logprobs must be"),
    ("run", "temperature", -0.1, "run.temperature must be"),
    ("backends.ollama", "reuse_cache", "yes", "backends.ollama.reuse_cache must be"),
    ("scores", "bootstrap_samples", 0, "scores.bootstrap_samples must be"),
    ("scores", "interval", 1.0, "scores.interval must be"),
])
def test_config_rejects_invalid_values(bcl_settings, make_bcl_config, section, key, bad_value, match):
    s = copy.deepcopy(bcl_settings)
    if section is None:
        s[key] = bad_value
    elif "." in section:
        outer, inner = section.split(".")
        s[outer][inner][key] = bad_value
    else:
        s[section][key] = bad_value
    with pytest.raises(ValueError, match=match):
        make_bcl_config(s)


def test_config_rejects_unknown_keys_and_thinking_with_logprobs(bcl_settings, make_bcl_config, tmp_path):
    s = copy.deepcopy(bcl_settings)
    s["blocks"]["extra_key"] = 123
    with pytest.raises(ValueError, match="unknown settings"):
        make_bcl_config(s)

    s2 = copy.deepcopy(bcl_settings)
    s2["models"][0]["think"] = True
    with pytest.raises(ValueError, match="first-token logprobs needs thinking off"):
        make_bcl_config(s2)
    s2["run"]["top_logprobs"] = 0
    assert model_label(make_bcl_config(s2).settings["models"][0]) == "test-model (thinking)"

    # Indistinguishable rules are rejected
    clash_file = tmp_path / "clash_rules.py"
    clash_rules = {
        "not green": RULES["not green"],
        "red or blue": "lambda t: t['color'] == 'red' or t['color'] == 'blue'",
    }
    clash_file.write_text(f"FEATURES = {FEATURES!r}\nRULES = {clash_rules!r}\n")
    s3 = copy.deepcopy(bcl_settings)
    s3["rules_file"] = str(clash_file)
    with pytest.raises(ValueError, match="cannot be distinguished"):
        make_bcl_config(s3)


# ---------------------------------------------------------------------------
# Block generation & pre-feedback Bayesian trajectory math
# ---------------------------------------------------------------------------

def test_generated_blocks_structure_shuffles_and_determinism(bcl_settings, make_bcl_config):
    config = make_bcl_config(bcl_settings)
    first = blocks.generate(config)
    second = blocks.generate(config)
    assert first == second
    assert len(first) == 10 * config.blocks["blocks_per_concept"]

    bcl_settings_alt = copy.deepcopy(bcl_settings)
    bcl_settings_alt["seed"] = 99
    different = blocks.generate(make_bcl_config(bcl_settings_alt))
    assert different != first

    all_objects = [dict(s) for s in config.stimuli]
    assert len(all_objects) == 9

    for block in first:
        slug = block["concept"].replace(" ", "_")
        assert block["block_id"] == f"{slug}-{block['run_index']:03d}"
        assert block["program"] == RULES[block["concept"]]
        assert block["complexity_group"] == complexity_group(block["concept"])
        assert block["word"] in config.blocks["nonce_words"]
        assert block["n_trials"] == 36
        assert block["repetitions"] == 4
        assert len(block["trials"]) == 36

        rule = config.rules[block["concept"]]
        rep_orders = []
        for rep in range(1, 5):
            rep_trials = [t for t in block["trials"] if t["repetition"] == rep]
            assert [t["trial"] for t in rep_trials] == list(range((rep - 1) * 9 + 1, rep * 9 + 1))
            rep_objs = [t["object"] for t in rep_trials]
            # Every repetition contains all 9 stimuli once
            assert sorted(rep_objs, key=lambda d: (d["color"], d["shape"])) == sorted(
                all_objects, key=lambda d: (d["color"], d["shape"])
            )
            rep_orders.append(rep_objs)

        # Across 4 shuffles of 9! = 362,880 permutations, repetitions are shuffled independently
        assert len({ json.dumps(order) for order in rep_orders }) > 1

        for t in block["trials"]:
            assert t["truth"] is bool(rule(t["object"]))


def test_bayes_p_yes_uses_only_strictly_prior_trials_never_leaking_current_label(
    bcl_settings, make_bcl_config
):
    config = make_bcl_config(bcl_settings)
    generated = blocks.generate(config)

    # Independent step-by-step posterior check on generated blocks
    for block in generated[:5]:
        seen_trials = []
        for t in block["trials"]:
            post_before = config.rules.posterior(seen_trials, alpha=config.alpha)
            uni_before = config.uniform_rules.posterior(seen_trials, alpha=config.alpha)
            expected_bayes = sum(
                post_before[r] * config.rules.label_probability(config.rules[r], t["object"], alpha=config.alpha)
                for r in config.rules.names
            )
            expected_uni = sum(
                uni_before[r]
                * config.uniform_rules.label_probability(config.uniform_rules[r], t["object"], alpha=config.alpha)
                for r in config.uniform_rules.names
            )
            assert t["bayes_p_yes"] == pytest.approx(expected_bayes, abs=1e-12)
            assert t["uniform_p_yes"] == pytest.approx(expected_uni, abs=1e-12)
            assert t["bayes_correct"] == blocks.bayes_correct(expected_bayes, t["truth"])
            seen_trials.append({**t["object"], "label": t["truth"]})

    # Direct non-leakage check at trial 1: for the same stimulus, opposite concepts ("red" vs "not red")
    # must have the exact same trial-1 bayes_p_yes because trial 1's label has not been observed yet!
    prior = config.rules.prior()
    for stimulus in config.stimuli:
        from_prior = blocks.predictive_p_yes(prior, config.rules, stimulus, config.alpha)
        for block in generated:
            if block["trials"][0]["object"] == stimulus:
                assert block["trials"][0]["bayes_p_yes"] == pytest.approx(from_prior, abs=1e-12)

    # Check bayes_correct tie handling
    assert blocks.bayes_correct(0.5, True) == 0.5
    assert blocks.bayes_correct(0.5 + 1e-12, False) == 0.5
    assert blocks.bayes_correct(0.8, True) == 1.0
    assert blocks.bayes_correct(0.8, False) == 0.0
    assert blocks.bayes_correct(0.2, False) == 1.0


def test_blocks_cli_writes_files_and_respects_force(bcl_settings, make_bcl_config, capsys):
    config = make_bcl_config(bcl_settings)
    blocks.main(["--config", str(config.path)])
    out = capsys.readouterr().out
    assert "30 blocks" in out

    jsonl_path = config.resolve(config.blocks["file"])
    cfg_copy = jsonl_path.with_suffix(".config.yaml")
    meta_path = jsonl_path.with_suffix(".meta.json")
    assert jsonl_path.exists() and cfg_copy.exists() and meta_path.exists()
    assert len(blocks.read_blocks(jsonl_path)) == 30
    meta = json.loads(meta_path.read_text())
    assert meta["n_blocks"] == 30 and meta["n_trials"] == 36

    # Modifying seed without --force raises SystemExit; with --force overwrites cleanly
    updated = copy.deepcopy(bcl_settings)
    updated["seed"] = 777
    config.path.write_text(yaml.safe_dump(updated, sort_keys=False))
    with pytest.raises(SystemExit, match="Rerun with --force"):
        blocks.main(["--config", str(config.path)])
    blocks.main(["--config", str(config.path), "--force"])
    assert json.loads(meta_path.read_text())["seed"] == 777


# ---------------------------------------------------------------------------
# Prompts, answer parsing, and logprob extraction
# ---------------------------------------------------------------------------

def test_prompts_hide_candidate_programs_and_reflect_exact_feedback(bcl_settings, make_bcl_config):
    config = make_bcl_config(bcl_settings)
    sample_blocks = blocks.generate(config)
    block = sample_blocks[0]
    word = block["word"]

    msg1 = elicit.start_message(block, config)
    assert word in msg1
    assert "36 trials" in msg1
    assert "Trial 1 of 36:" in msg1
    assert repr(block["trials"][0]["object"]) in msg1
    assert "Answer with only yes or no." in msg1
    # Must NEVER show Python programs or lambda rules
    assert "lambda" not in msg1
    for program in RULES.values():
        assert program not in msg1

    # Check all feedback combinations on trial 2
    cases = [
        (True, True, f"Correct. That object is a {word}."),
        (False, True, f"Incorrect. That object is a {word}."),
        (None, True, f"Incorrect. That object is a {word}."),
        (False, False, f"Correct. That object is not a {word}."),
        (True, False, f"Incorrect. That object is not a {word}."),
        (None, False, f"Incorrect. That object is not a {word}."),
    ]
    for prev_ans, prev_truth, expected_line in cases:
        msg2 = elicit.step_message(block, 2, prev_ans, prev_truth, config)
        assert msg2.startswith(expected_line + "\n\nTrial 2 of 36: ")
        assert repr(block["trials"][1]["object"]) in msg2
        assert msg2.endswith(f"Is this object a {word}? Answer with only yes or no.")

    with pytest.raises(ValueError, match="trial_number"):
        elicit.step_message(block, 1, True, True, config)
    with pytest.raises(ValueError, match="trial_number"):
        elicit.step_message(block, 37, True, True, config)


@pytest.mark.parametrize("raw, expected", [
    ("yes", True),
    ("no", False),
    ("Yes", True),
    ("NO", False),
    ("  yes. \n", True),
    ("No!", False),
    ("Answer: yes", True),
    ("Answer: No.", False),
    ("Yes, that object is", True),
    ("No, it is not", False),
])
def test_parse_answer_valid(raw, expected):
    parsed, err = elicit.parse_answer(raw)
    assert parsed is expected
    assert err is None


@pytest.mark.parametrize("raw", ["", "   ", "maybe", "true", "false", "yes or no", "no or yes", "red"])
def test_parse_answer_invalid(raw):
    parsed, err = elicit.parse_answer(raw)
    assert parsed is None
    assert err is not None


def test_extract_p_yes_from_top_logprobs():
    logprobs = [
        {"token": " ", "logprob": -0.01, "top_logprobs": [{"token": " ", "logprob": -0.01}]},
        {
            "token": "Yes",
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": "Yes", "logprob": math.log(0.50)},
                {"token": "yes", "logprob": math.log(0.25)},
                {"token": " no", "logprob": math.log(0.15)},
                {"token": "No", "logprob": math.log(0.10)},
                {"token": "not", "logprob": math.log(0.05)},       # must not count as "no"
                {"token": "yesterday", "logprob": math.log(0.05)}, # must not count as "yes"
            ],
        },
    ]
    assert elicit.extract_p_yes(logprobs) == pytest.approx(0.75 / (0.75 + 0.25))
    assert elicit.extract_p_yes(logprobs, thinking="hmm") is None
    assert elicit.extract_p_yes(None) is None
    assert elicit.extract_p_yes([]) is None

    no_yes_no = [{"token": "The", "logprob": -0.1, "top_logprobs": [{"token": "The", "logprob": -0.1}]}]
    assert elicit.extract_p_yes(no_yes_no) is None


# ---------------------------------------------------------------------------
# 16-stimulus single-pass (zero exemplar repetition) design: bcl_rules.py
# ---------------------------------------------------------------------------

def test_bcl_rules_sixteen_unique_stimuli_and_zero_repetition_blocks(bcl_settings, make_bcl_config):
    import bcl_rules
    from rule_learner import all_stimuli

    assert bcl_rules.FEATURES == {
        "color": ["red", "blue", "green", "yellow"],
        "shape": ["circle", "square", "triangle", "diamond"],
    }
    assert bcl_rules.RULES == RULES

    stimuli_16 = all_stimuli(bcl_rules.FEATURES)
    assert len(stimuli_16) == 16
    assert len({(s["color"], s["shape"]) for s in stimuli_16}) == 16
    RuleSet.from_programs(bcl_rules.RULES).check_distinct(stimuli_16)

    s16 = copy.deepcopy(bcl_settings)
    s16["rules_file"] = str(ROOT / "bcl_rules.py")
    s16["blocks"]["repetitions"] = 1
    config_16 = make_bcl_config(s16)

    generated = blocks.generate(config_16)
    assert len(generated) == 10 * s16["blocks"]["blocks_per_concept"]

    expected_sorted = sorted(stimuli_16, key=lambda d: (d["color"], d["shape"]))
    for block in generated:
        assert block["n_trials"] == 16
        assert block["repetitions"] == 1
        assert len(block["trials"]) == 16
        assert [t["trial"] for t in block["trials"]] == list(range(1, 17))
        assert all(t["repetition"] == 1 for t in block["trials"])

        objs = [t["object"] for t in block["trials"]]
        # Zero duplicates within any block; every 16-stimulus object appears once
        assert len({(o["color"], o["shape"]) for o in objs}) == 16
        assert sorted(objs, key=lambda d: (d["color"], d["shape"])) == expected_sorted

        rule = config_16.rules[block["concept"]]
        seen_trials = []
        for t in block["trials"]:
            assert t["truth"] is bool(rule(t["object"]))
            post_before = config_16.rules.posterior(seen_trials, alpha=config_16.alpha)
            expected_bp = blocks.predictive_p_yes(post_before, config_16.rules, t["object"], config_16.alpha)
            assert t["bayes_p_yes"] == pytest.approx(expected_bp, abs=1e-12)
            seen_trials.append({**t["object"], "label": t["truth"]})

    # Verify prompt wording on 16-trial single-pass blocks
    sample = generated[0]
    msg1 = elicit.start_message(sample, config_16)
    assert "16 trials" in msg1
    assert "(16 possible objects in total)" in msg1
    assert "'yellow'" in msg1 and "'diamond'" in msg1
    assert "Trial 1 of 16:" in msg1
    assert "lambda" not in msg1

    msg16 = elicit.step_message(sample, 16, True, sample["trials"][14]["truth"], config_16)
    assert f"\n\nTrial 16 of 16: {repr(sample['trials'][15]['object'])}\n" in msg16
    with pytest.raises(ValueError, match="trial_number"):
        elicit.step_message(sample, 17, True, True, config_16)

    summary_text = blocks.summarize(generated, config_16)
    assert "16 trials/block" in summary_text
    assert "Q1 (1\u20134)" in summary_text and "Q4 (13\u201316)" in summary_text


def test_running_log_rule_prompts_parser_and_answer_line_logprobs(bcl_settings, make_bcl_config):
    s16 = copy.deepcopy(bcl_settings)
    s16["rules_file"] = str(ROOT / "bcl_rules.py")
    s16["blocks"]["repetitions"] = 1
    s16["run"]["mode"] = "running_log_rule"
    s16["run"]["max_tokens"] = 96
    config_16 = make_bcl_config(s16)
    block = blocks.generate(config_16)[0]
    word = block["word"]

    msg1 = elicit.trial_message(block, 1, [], config_16)
    assert f'made-up word "{word}"' in msg1
    assert "16 unique objects in total, each shown once" in msg1
    assert "a color (red, blue, green, yellow) and a shape (circle, square, triangle, diamond)" in msg1
    assert "Trials so far: (none yet — make an initial yes or no guess)" in msg1
    assert "Trial 1 of 16: " in msg1
    assert 'Rule: <short rule over color and/or shape, or "guessing" if unsure>' in msg1
    assert "Answer: <yes or no — you must guess either yes or no, never unknown>" in msg1
    assert "lambda" not in msg1

    t1 = block["trials"][0]
    t2 = block["trials"][1]
    p1 = elicit.stimulus_phrase(t1["object"], config_16.features)
    p2 = elicit.stimulus_phrase(t2["object"], config_16.features)
    history = [
        {"trial": 1, "object": t1["object"], "truth": True, "answer": False, "rule_guess": "guessing"},
        {"trial": 2, "object": t2["object"], "truth": False, "answer": False, "rule_guess": "red or square"},
    ]
    msg3 = elicit.trial_message(block, 3, history, config_16)
    assert "Trials and feedback so far:" in msg3
    assert f'  1. {p1} — rule guess: "guessing", answer: no -> Incorrect ({p1} IS a {word})' in msg3
    assert f'  2. {p2} — rule guess: "red or square", answer: no -> Correct ({p2} is NOT a {word})' in msg3
    assert "Trial 3 of 16: " in msg3

    msg3_yesno = elicit.trial_message(block, 3, history, config_16, mode="running_log_yesno")
    assert "rule guess:" not in msg3_yesno
    assert f"  1. {p1} — your guess: no -> Incorrect ({p1} IS a {word})" in msg3_yesno
    assert "Answer with only yes or no." in msg3_yesno

    with pytest.raises(ValueError, match="history entries"):
        elicit.trial_message(block, 2, [], config_16)
    with pytest.raises(ValueError, match="trial_number"):
        elicit.trial_message(block, 17, history, config_16)

    # Parser tests for Rule + Answer format
    assert elicit.parse_rule_and_answer("Rule: color is red, blue, or yellow\nAnswer: yes") == (
        True, "color is red, blue, or yellow", None
    )
    assert elicit.parse_rule_and_answer("**Rule:** square\n**Answer:** No.") == (
        False, "square", None
    )
    ans_unk, r_unk, err_unk = elicit.parse_rule_and_answer("Rule: unknown\nAnswer: unknown")
    assert ans_unk is None and r_unk == "unknown" and err_unk is not None
    ans_amb, r_amb, err_amb = elicit.parse_rule_and_answer("Rule: red\nAnswer: yes or no")
    assert ans_amb is None and r_amb == "red" and err_amb is not None

    # Logprobs: "no" inside Rule: line must NEVER shadow "yes" on the Answer: line
    rule_answer_logprobs = [
        {"token": "Rule", "logprob": -0.01, "top_logprobs": [{"token": "Rule", "logprob": -0.01}]},
        {"token": ":", "logprob": -0.01, "top_logprobs": [{"token": ":", "logprob": -0.01}]},
        {"token": " no", "logprob": -0.1, "top_logprobs": [{"token": " no", "logprob": -0.1}]},
        {"token": " green", "logprob": -0.1, "top_logprobs": [{"token": " green", "logprob": -0.1}]},
        {"token": "\n", "logprob": -0.01, "top_logprobs": [{"token": "\n", "logprob": -0.01}]},
        {"token": "Answer", "logprob": -0.01, "top_logprobs": [{"token": "Answer", "logprob": -0.01}]},
        {"token": ":", "logprob": -0.01, "top_logprobs": [{"token": ":", "logprob": -0.01}]},
        {
            "token": " yes",
            "logprob": math.log(0.8),
            "top_logprobs": [
                {"token": " yes", "logprob": math.log(0.8)},
                {"token": " no", "logprob": math.log(0.2)},
            ],
        },
    ]
    assert elicit.extract_p_yes(rule_answer_logprobs) == pytest.approx(0.8)


def test_grammar_hypothesis_space_exact_parity_with_boolean_grammar_rational_rules_and_pcfg(
    bcl_settings, make_bcl_config
):
    from bcl.config import _rr_dirichlet_log_term, enumerate_grammar_derivations
    from rule_learner import all_stimuli

    # 1. Verify fast count-based log_prior matches BooleanGrammar.log_prior to 1e-12 on every tree
    small_stimuli = all_stimuli(FEATURES)
    derivs = enumerate_grammar_derivations(FEATURES, small_stimuli, max_bool_nodes=4)
    assert len(derivs) == 312  # 6 + 6 + 78 + 222 = 312 across 3 colors x 3 shapes
    grammar = BooleanGrammar(FEATURES, weights={"pred": 5.0, "and": 1.0, "or": 2.0, "not": 1.5})

    tot_w = sum(grammar.weights.values())
    bool_lps = tuple(math.log(grammar.weights[p] / tot_w) for p in grammar.BOOL_PRODUCTIONS)
    lp_pred = -math.log(len(FEATURES))
    lp_vals = tuple(-math.log(len(FEATURES[f])) for f in FEATURES)

    for expr, _, b_c, p_c, v_c in derivs:
        src = f"lambda t: {expr}"
        fast_rr = (
            _rr_dirichlet_log_term(b_c, 2.0)
            + _rr_dirichlet_log_term(p_c, 2.0)
            + sum(_rr_dirichlet_log_term(vc, 2.0) for vc in v_c)
        )
        assert fast_rr == pytest.approx(grammar.log_prior(src, kind="rational_rules", dirichlet=2.0), abs=1e-12)

        fast_pcfg = (
            sum(c * w for c, w in zip(b_c, bool_lps))
            + sum(p_c) * lp_pred
            + sum(sum(vc) * lpv for vc, lpv in zip(v_c, lp_vals))
        )
        assert fast_pcfg == pytest.approx(grammar.log_prior(src, kind="pcfg"), abs=1e-12)

    # 2. Verify extension-collapsed hypothesis_rules gives IDENTICAL bayes_p_yes to un-collapsed full tree RuleSet
    s_rr = copy.deepcopy(bcl_settings)
    s_rr["blocks"]["repetitions"] = 1
    s_rr["prior"] = {
        "kind": "rational_rules",
        "hypothesis_space": "grammar",
        "max_bool_nodes": 4,
        "grammar_weights": None,
        "dirichlet": 2.0,
        "temperature": None,
    }
    cfg_rr = make_bcl_config(s_rr)

    uncollapsed_programs = {f"tree_{i}": f"lambda t: {expr}" for i, (expr, *_) in enumerate(derivs)}
    uncollapsed_weights = BooleanGrammar(FEATURES).prior_weights(
        uncollapsed_programs, kind="rational_rules", dirichlet=2.0
    )
    full_tree_ruleset = RuleSet.from_programs(uncollapsed_programs, weights=uncollapsed_weights)

    generated = blocks.generate(cfg_rr)
    for blk in generated[:4]:
        seen = []
        for t in blk["trials"]:
            post = full_tree_ruleset.posterior(seen, alpha=cfg_rr.alpha)
            expected_p_yes = blocks.predictive_p_yes(post, full_tree_ruleset, t["object"], cfg_rr.alpha)
            assert t["bayes_p_yes"] == pytest.approx(expected_p_yes, abs=1e-12)
            seen.append({**t["object"], "label": t["truth"]})

    # 3. Too-small max_bool_nodes (e.g. 1 cannot express 'red and not square') raises a clear ValueError
    s_tiny = copy.deepcopy(s_rr)
    s_tiny["prior"]["max_bool_nodes"] = 1
    with pytest.raises(ValueError, match="not expressible within prior.max_bool_nodes=1"):
        make_bcl_config(s_tiny)

