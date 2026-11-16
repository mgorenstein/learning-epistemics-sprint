"""Tests for bcl.run with MockBackend and mocked Ollama KV-cache verification.

Run with:  .venv/bin/python -m pytest test_bcl_run.py
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from bcl import blocks, elicit, run
from bcl.config import load_config
from example_rules import RULES
from hh.backends import BackendError, MockBackend, OllamaBackend, Reply

ROOT = Path(__file__).parent

EXPECTED_TOP_KEYS = (
    "key",
    "fingerprint",
    "time",
    "model",
    "model_name",
    "model_digest",
    "backend",
    "backend_version",
    "think",
    "mode",
    "block_id",
    "concept",
    "program",
    "complexity_group",
    "run_index",
    "word",
    "n_trials",
    "repetitions",
    "seconds",
    "parse_failures",
    "trials",
)

EXPECTED_TRIAL_KEYS = (
    "trial",
    "repetition",
    "object",
    "truth",
    "rule_guess",
    "answer",
    "raw_response",
    "p_yes",
    "correct",
    "bayes_p_yes",
    "bayes_correct",
    "uniform_p_yes",
    "cached_tokens",
    "seconds",
)


def yes_no_reply(say_yes: bool, p_yes: float = 0.8, cached_tokens: int | None = 0, seconds: float = 0.05):
    """Construct a Reply with realistic yes/no top_logprobs."""
    text = "yes" if say_yes else "no"
    p_no = 1.0 - p_yes
    chosen_p = p_yes if say_yes else p_no
    alts = [
        {"token": "yes", "logprob": math.log(p_yes)},
        {"token": "no", "logprob": math.log(p_no)},
    ]
    return Reply(
        text=text,
        logprobs=[{"token": text, "logprob": math.log(chosen_p), "top_logprobs": alts}],
        cached_tokens=cached_tokens,
        seconds=seconds,
    )


@pytest.fixture
def env_config(tmp_path):
    """Write a small 3-blocks/concept config and generate its 30 blocks."""
    settings = {
        "seed": 123,
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
            "nonce_words": ["lanch", "sowdy", "widge", "pluic", "rosth"],
        },
        "models": [
            {"name": "mock-gemma", "backend": "ollama", "think": False, "extra_body": None},
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
    cfg_path = tmp_path / "config_bcl.yaml"
    cfg_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    config = load_config(cfg_path)
    blocks.write_blocks(blocks.generate(config), config)
    return config


def test_balanced_order_runs_round_robin_by_run_index_then_concept(env_config):
    all_blocks = blocks.read_blocks(env_config.resolve(env_config.blocks["file"]))
    ordered = run.balanced_order(all_blocks, blocks_per_concept=2)
    assert len(ordered) == 20
    # First 10 are run_index 0 across all 10 concepts in RULES order; next 10 are run_index 1
    assert [b["run_index"] for b in ordered[:10]] == [0] * 10
    assert [b["concept"] for b in ordered[:10]] == list(RULES)
    assert [b["run_index"] for b in ordered[10:20]] == [1] * 10
    assert [b["concept"] for b in ordered[10:20]] == list(RULES)

    with pytest.raises(ValueError, match="1 or more"):
        run.balanced_order(all_blocks, blocks_per_concept=0)


def test_run_writes_exact_contract_and_accumulates_36_turn_history(env_config):
    def alternating(model, messages, **kwargs):
        turn = (len(messages) + 1) // 2
        say_yes = (turn % 2 == 1)
        cached = 0 if turn == 1 else (turn - 1) * 40
        return yes_no_reply(say_yes=say_yes, p_yes=0.8 if say_yes else 0.2, cached_tokens=cached)

    mock = MockBackend(alternating)
    written = run.run(env_config, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)
    assert written == 10
    assert len(mock.calls) == 10 * 36

    rows = run.read_rows(env_config.resolve(env_config.run["results_file"]))
    assert len(rows) == 10
    assert [row["concept"] for row in rows] == list(RULES)
    assert all(row["run_index"] == 0 for row in rows)

    # Verify exact schema on every block row and every trial dict
    block_lookup = {
        b["block_id"]: b for b in blocks.read_blocks(env_config.resolve(env_config.blocks["file"]))
    }
    for row in rows:
        assert tuple(row.keys()) == EXPECTED_TOP_KEYS
        src_block = block_lookup[row["block_id"]]
        assert row["key"] == f"mock-gemma|{row['block_id']}"
        assert row["fingerprint"] == run.fingerprint(env_config)
        assert row["model"] == "mock-gemma"
        assert row["model_name"] == "mock-gemma"
        assert row["model_digest"] == "mock-digest"
        assert row["backend"] == "mock"
        assert row["backend_version"] == "mock"
        assert row["think"] is False
        assert row["mode"] == "multiturn_yesno"
        assert row["n_trials"] == 36
        assert row["repetitions"] == 4
        assert row["parse_failures"] == 0
        assert row["seconds"] == pytest.approx(36 * 0.05)
        assert len(row["trials"]) == 36

        for idx, trial in enumerate(row["trials"], start=1):
            assert tuple(trial.keys()) == EXPECTED_TRIAL_KEYS
            assert trial["trial"] == idx
            assert trial["repetition"] == (idx - 1) // 9 + 1
            assert isinstance(trial["truth"], bool)
            assert trial["rule_guess"] is None
            expected_ans = (idx % 2 == 1)
            assert trial["answer"] is expected_ans
            assert trial["raw_response"] == ("yes" if expected_ans else "no")
            assert trial["p_yes"] == pytest.approx(0.8 if expected_ans else 0.2)
            assert trial["correct"] is (expected_ans is trial["truth"])
            assert trial["bayes_p_yes"] == pytest.approx(src_block["trials"][idx - 1]["bayes_p_yes"])
            assert trial["bayes_correct"] == pytest.approx(src_block["trials"][idx - 1]["bayes_correct"])
            assert trial["uniform_p_yes"] == pytest.approx(src_block["trials"][idx - 1]["uniform_p_yes"])
            assert trial["cached_tokens"] == (0 if idx == 1 else (idx - 1) * 40)

    # Verify the 36 calls of block 0 form one growing multi-turn prefix with exact feedback
    first_block = block_lookup[rows[0]["block_id"]]
    first_36_calls = mock.calls[:36]
    for idx, call in enumerate(first_36_calls, start=1):
        msgs = call["messages"]
        assert len(msgs) == 2 * idx - 1
        if idx == 1:
            assert msgs[0] == {"role": "user", "content": elicit.start_message(first_block, env_config)}
        else:
            prev_msgs = first_36_calls[idx - 2]["messages"]
            prev_trial = rows[0]["trials"][idx - 2]
            assert msgs[:-2] == prev_msgs
            assert msgs[-2] == {"role": "assistant", "content": prev_trial["raw_response"]}
            expected_step = elicit.step_message(
                first_block, idx, prev_trial["answer"], prev_trial["truth"], env_config
            )
            assert msgs[-1] == {"role": "user", "content": expected_step}


def test_ollama_backend_preserves_kv_cache_without_reset_prompts(env_config, monkeypatch):
    backends = run.make_backends(env_config)
    ollama = backends["ollama"]
    assert isinstance(ollama, OllamaBackend)
    assert ollama.fresh is False

    # Flip reuse_cache -> False and verify fresh becomes True
    no_cache_settings = copy.deepcopy(env_config.settings)
    no_cache_settings["backends"]["ollama"]["reuse_cache"] = False
    cfg_no_cache_path = env_config.path.parent / "config_nocache.yaml"
    cfg_no_cache_path.write_text(yaml.safe_dump(no_cache_settings, sort_keys=False))
    assert run.make_backends(load_config(cfg_no_cache_path))["ollama"].fresh is True

    # Intercept HTTP POSTs in hh.backends._post to verify zero RESET_PROMPT calls during a 36-turn block
    posted_bodies = []

    def fake_post(url, body, timeout, headers=None):
        posted_bodies.append(body)
        n_msgs = len(body["messages"])
        cached = 0 if n_msgs == 1 else n_msgs * 25
        return (
            {
                "message": {"content": "yes"},
                "done_reason": "stop",
                "prompt_eval_count": n_msgs * 30,
                "prompt_eval_cached_count": cached,
                "eval_count": 2,
                "logprobs": [
                    {
                        "token": "yes",
                        "logprob": math.log(0.9),
                        "top_logprobs": [
                            {"token": "yes", "logprob": math.log(0.9)},
                            {"token": "no", "logprob": math.log(0.1)},
                        ],
                    }
                ],
            },
            0.01,
        )

    monkeypatch.setattr("hh.backends._post", fake_post)
    one_block = blocks.read_blocks(env_config.resolve(env_config.blocks["file"]))[0]
    model = env_config.settings["models"][0]
    trial_rows, failures, _ = run.run_block(ollama, model, one_block, env_config)

    assert len(posted_bodies) == 36
    assert not any(
        b["messages"] == [{"role": "user", "content": OllamaBackend.RESET_PROMPT}]
        for b in posted_bodies
    )
    assert failures == 0
    assert trial_rows[0]["cached_tokens"] == 0
    assert all(t["cached_tokens"] > 0 for t in trial_rows[1:])


def test_parse_failures_and_resume_and_error_recovery(env_config, capsys):
    # 1. Parse failures on specific turns are counted and receive "Incorrect." feedback
    def flaky_text(model, messages, **kwargs):
        turn = (len(messages) + 1) // 2
        if turn in (3, 7):
            return Reply(text="unsure", logprobs=None, seconds=0.01)
        return yes_no_reply(say_yes=True, p_yes=0.75, seconds=0.01)

    one_block = blocks.read_blocks(env_config.resolve(env_config.blocks["file"]))[0]
    trial_rows, parse_failures, _ = run.run_block(
        MockBackend(flaky_text), env_config.settings["models"][0], one_block, env_config
    )
    assert parse_failures == 2
    assert trial_rows[2]["answer"] is None and trial_rows[2]["correct"] is False
    assert trial_rows[6]["answer"] is None and trial_rows[6]["correct"] is False

    # 2. Mid-block BackendError writes no partial row, logs failure, and succeeds on retry
    state = {"fail_first_block": True}

    def fail_once_mid_block(model, messages, **kwargs):
        if state["fail_first_block"] and len(messages) == 11:  # fail on trial 6 of very first block
            state["fail_first_block"] = False
            raise BackendError("simulated timeout on trial 6")
        return yes_no_reply(say_yes=True, p_yes=0.6, seconds=0.02)

    mock = MockBackend(fail_once_mid_block)
    written_1 = run.run(env_config, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)
    assert written_1 == 9
    results_path = env_config.resolve(env_config.run["results_file"])
    assert len(run.read_rows(results_path)) == 9
    assert "simulated timeout on trial 6" in results_path.with_suffix(".log").read_text()

    # Estimate CLI reports the 1 remaining block in the 1-per-concept pilot
    run.main(["--config", str(env_config.path), "--blocks-per-concept", "1", "--estimate"])
    est_out = capsys.readouterr().out
    assert "1 blocks (36 trials) to run; 9 done." in est_out
    assert "minutes" in est_out

    # Rerun completes only the 1 failed block
    calls_before = len(mock.calls)
    written_2 = run.run(env_config, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)
    assert written_2 == 1
    assert len(mock.calls) - calls_before == 36
    assert len(run.read_rows(results_path)) == 10

    # Rerunning again does nothing
    assert run.run(env_config, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None) == 0

    # Expanding pilot to blocks_per_concept=2 runs only the 10 new run_index=1 blocks
    written_3 = run.run(env_config, {"ollama": mock}, blocks_per_concept=2, log=lambda _: None)
    assert written_3 == 10
    assert len(run.read_rows(results_path)) == 20


def test_stale_fingerprint_stops_mixed_results_while_safe_changes_do_not(env_config):
    mock = MockBackend(lambda model, messages, **kw: yes_no_reply(True))
    run.run(env_config, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)
    original_fp = run.fingerprint(env_config)

    # Changing base_url or timeout_seconds leaves fingerprint unchanged
    safe = copy.deepcopy(env_config.settings)
    safe["backends"]["ollama"]["base_url"] = "http://localhost:11435"
    safe["run"]["timeout_seconds"] = 600
    env_config.path.write_text(yaml.safe_dump(safe, sort_keys=False))
    reloaded = load_config(env_config.path)
    assert run.fingerprint(reloaded) == original_fp

    # Changing temperature changes fingerprint and halts resume
    stale = copy.deepcopy(safe)
    stale["run"]["temperature"] = 0.7
    env_config.path.write_text(yaml.safe_dump(stale, sort_keys=False))
    with pytest.raises(SystemExit, match="different blocks, prompt templates or run settings"):
        run.run(load_config(env_config.path), {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)


def test_running_log_rule_mode_single_turn_prompts_rule_guesses_and_logprobs(tmp_path):
    settings = {
        "seed": 321,
        "rules_file": str(ROOT / "bcl_rules.py"),
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
            "blocks_per_concept": 2,
            "repetitions": 1,
            "nonce_words": ["lanch", "sowdy", "widge", "pluic", "rosth"],
        },
        "models": [{"name": "mock-gemma", "backend": "ollama", "think": False, "extra_body": None}],
        "run": {
            "mode": "running_log_rule",
            "results_file": "data/bcl/results.jsonl",
            "timeout_seconds": 30,
            "temperature": 0.0,
            "max_tokens": 96,
            "top_logprobs": 20,
        },
        "backends": {
            "ollama": {"base_url": "http://localhost:11434", "options": {"num_ctx": 8192}, "reuse_cache": True},
            "openai": {"base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY"},
        },
        "scores": {"bootstrap_samples": 100, "interval": 0.95},
    }
    cfg_path = tmp_path / "config_bcl_rule.yaml"
    cfg_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    cfg = load_config(cfg_path)
    blocks.write_blocks(blocks.generate(cfg), cfg)

    call_idx = {"n": 0}

    def rule_reply(model, messages, **kwargs):
        assert len(messages) == 1
        call_idx["n"] += 1
        trial_num = (call_idx["n"] - 1) % 16 + 1
        say_yes = (trial_num % 2 == 0)
        rule_txt = "guessing" if trial_num == 1 else f"hypothesis_{trial_num}"
        ans_txt = "yes" if say_yes else "no"
        p_yes = 0.85 if say_yes else 0.15
        p_chosen = p_yes if say_yes else (1.0 - p_yes)
        return Reply(
            text=f"Rule: {rule_txt}\nAnswer: {ans_txt}",
            logprobs=[
                {"token": "Rule", "logprob": -0.01, "top_logprobs": [{"token": "Rule", "logprob": -0.01}]},
                {"token": ":", "logprob": -0.01, "top_logprobs": [{"token": ":", "logprob": -0.01}]},
                {"token": f" {rule_txt}", "logprob": -0.1, "top_logprobs": [{"token": f" {rule_txt}", "logprob": -0.1}]},
                {"token": "\nAnswer:", "logprob": -0.01, "top_logprobs": [{"token": "\nAnswer:", "logprob": -0.01}]},
                {
                    "token": f" {ans_txt}",
                    "logprob": math.log(p_chosen),
                    "top_logprobs": [
                        {"token": " yes", "logprob": math.log(p_yes)},
                        {"token": " no", "logprob": math.log(1.0 - p_yes)},
                    ],
                },
            ],
            seconds=0.02,
        )

    mock = MockBackend(rule_reply)
    written = run.run(cfg, {"ollama": mock}, blocks_per_concept=1, log=lambda _: None)
    assert written == 10
    assert len(mock.calls) == 10 * 16

    rows = run.read_rows(cfg.resolve(cfg.run["results_file"]))
    assert len(rows) == 10
    first_row = rows[0]
    assert first_row["mode"] == "running_log_rule"
    assert first_row["n_trials"] == 16
    assert first_row["parse_failures"] == 0
    assert first_row["trials"][0]["rule_guess"] == "guessing"
    assert first_row["trials"][0]["answer"] is False
    assert first_row["trials"][0]["p_yes"] == pytest.approx(0.15)
    assert first_row["trials"][1]["rule_guess"] == "hypothesis_2"
    assert first_row["trials"][1]["answer"] is True
    assert first_row["trials"][1]["p_yes"] == pytest.approx(0.85)

    # Trial 16 prompt of block 0 carries all 15 prior rule guesses and answers
    t16_prompt = mock.calls[15]["messages"][0]["content"]
    assert "Trial 16 of 16:" in t16_prompt
    assert '1. ' in t16_prompt and 'rule guess: "guessing", answer: no' in t16_prompt
    assert '15. ' in t16_prompt and 'rule guess: "hypothesis_15", answer: no' in t16_prompt

    # Switching mode changes fingerprint
    alt = copy.deepcopy(settings)
    alt["run"]["mode"] = "running_log_yesno"
    cfg_alt_path = tmp_path / "config_bcl_yesno.yaml"
    cfg_alt_path.write_text(yaml.safe_dump(alt, sort_keys=False))
    assert run.fingerprint(load_config(cfg_alt_path)) != run.fingerprint(cfg)
