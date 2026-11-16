"""Shared test fixtures: a small, complete set of settings and a way to load it.

Tests do not read config.yaml (apart from one check that it loads), so changing the
experiment's settings cannot break them. They use example_rules.py as their rule set.
"""

import copy
from pathlib import Path

import pytest
import yaml

from hh.config import load_config

ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def default_settings():
    """A complete, small set of settings. Do not change it; ``settings`` gives a private copy."""
    return {
        "seed": 1,
        "rules_file": str(ROOT / "example_rules.py"),
        "alpha": 0.99,
        "prior": {"kind": "pcfg", "grammar_weights": None, "dirichlet": None, "temperature": 3.0},
        "puzzles": {
            "file": "data/puzzles/puzzles.jsonl",
            "n_confident": 10,
            "n_contested": 10,
            "lengths": [6, 8],
            "query_steps": 3,
            "confident": {"leader_min": 0.95},
            "contested": {"runner_up_min": 0.20, "runner_up_max": 0.40, "lead_min": 0.05, "others_max": 0.10},
            "max_attempts": 20000,
            "nonce_words": ["wudsy", "daxy", "fepsy"],
        },
        "elicitation": ["choice"],
        "followups": ["neutral", "leading", "pushback", "evidence"],
        "models": [{"name": "test-model", "backend": "ollama", "think": False, "extra_body": None}],
        "run": {
            "results_file": "data/results/results.jsonl",
            "timeout_seconds": 30,
            "choice": {"max_tokens": 8, "top_logprobs": 20, "samples": 5},
            "stated": {"temperature": 0.0, "max_tokens": 400, "structured_output": True},
        },
        "backends": {
            "ollama": {"base_url": "http://localhost:11434", "options": {"num_ctx": 8192}, "fresh": True},
            "openai": {"base_url_env": "OPENAI_BASE_URL", "api_key_env": "OPENAI_API_KEY"},
        },
        "scores": {"bootstrap_samples": 200, "interval": 0.95},
    }


@pytest.fixture
def settings(default_settings):
    """A copy of the default settings that a test may change freely."""
    return copy.deepcopy(default_settings)


@pytest.fixture(scope="session")
def load_settings(tmp_path_factory):
    """A function that writes a settings dict to a config file and loads it.

    The file goes in ``folder``, or in a fresh temporary folder. Outputs named in the
    settings land in the same folder, because config paths are relative to the config file.
    """
    def load(settings, folder=None):
        folder = folder or tmp_path_factory.mktemp("config")
        path = folder / "config.yaml"
        path.write_text(yaml.safe_dump(settings, sort_keys=False))
        return load_config(path)
    return load
