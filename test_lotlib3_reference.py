"""rule_learner must reproduce numbers computed by LOTlib3 itself.

reference/lotlib3_reference.json was produced by tools/make_lotlib3_reference.py,
which runs LOTlib3's SimpleBoolean_noTF grammar, its PCFG and Rational Rules
priors and its BinaryLikelihood on 120 sampled hypotheses and 10 data sets,
including the data of LOTlib3's own Rational Rules example. LOTlib3 does not
need to be installed to run these tests. Do not edit the JSON by hand;
regenerate it with the script.

Run with:  python -m pytest test_lotlib3_reference.py
"""

import json
import math
from pathlib import Path

import pytest

from rule_learner import BooleanGrammar, Rule, RuleSet

REFERENCE = json.loads((Path(__file__).parent / "reference" / "lotlib3_reference.json").read_text())
HYPOTHESES = REFERENCE["hypotheses"]
DATASETS = REFERENCE["datasets"]
GRAMMAR = BooleanGrammar.lotlib3_simple_boolean(REFERENCE["features"])
RR_ALPHA = REFERENCE["source"]["rational_rules_alpha"]
TOLERANCE = 1e-9

ids = [h["lotlib3"] for h in HYPOTHESES]


def test_reference_file_is_the_expected_shape():
    assert REFERENCE["source"]["project"] == "LOTlib3"
    assert len(HYPOTHESES) >= 100 and len(DATASETS) >= 10
    assert DATASETS[0]["name"].startswith("LOTlib3 Rational Rules example")


@pytest.mark.parametrize("hyp", HYPOTHESES, ids=ids)
def test_rules_evaluate_like_lotlib3(hyp):
    rule = Rule("h", hyp["python"])
    assert [rule(obj) for obj in REFERENCE["objects"]] == hyp["truth_table"]


@pytest.mark.parametrize("hyp", HYPOTHESES, ids=ids)
def test_pcfg_prior_matches_lotlib3(hyp):
    assert GRAMMAR.log_prior(hyp["python"], kind="pcfg") == pytest.approx(
        hyp["pcfg_log_prior"], abs=TOLERANCE)


@pytest.mark.parametrize("hyp", HYPOTHESES, ids=ids)
def test_rational_rules_prior_matches_lotlib3(hyp):
    assert GRAMMAR.log_prior(hyp["python"], kind="rational_rules", dirichlet=RR_ALPHA) == pytest.approx(
        hyp["rational_rules_log_prior"], abs=TOLERANCE)


@pytest.mark.parametrize("hyp", HYPOTHESES, ids=ids)
def test_likelihood_matches_lotlib3(hyp):
    rule = Rule("h", hyp["python"])
    rules = RuleSet([rule])
    for dataset, expected in zip(DATASETS, hyp["log_likelihood"]):
        got = rules.log_likelihood(rule, dataset["trials"], alpha=dataset["alpha"])
        assert got == pytest.approx(expected, abs=TOLERANCE), dataset["name"]


@pytest.mark.parametrize("case", REFERENCE["posterior_cases"],
                         ids=[f"{c['prior']}-data{c['dataset']}-{i}" for i, c in enumerate(REFERENCE["posterior_cases"])])
def test_posterior_matches_lotlib3(case):
    members = [HYPOTHESES[i] for i in case["hypotheses"]]
    programs = {f"h{i}": h["python"] for i, h in enumerate(members)}
    weights = GRAMMAR.prior_weights(programs, kind=case["prior"], dirichlet=RR_ALPHA)
    rules = RuleSet.from_programs(programs, weights=weights)
    dataset = DATASETS[case["dataset"]]
    posterior = rules.posterior(dataset["trials"], alpha=dataset["alpha"])
    assert list(posterior.values()) == pytest.approx(case["posterior"], abs=TOLERANCE)
    assert math.isclose(sum(posterior.values()), 1.0)


def test_lotlib3_rational_rules_example_picks_red_square():
    # The example's data label only the red square as True. Among the sampled
    # hypotheses that fit all four labels, "red and square" style rules must
    # beat everything that misclassifies an object.
    dataset = DATASETS[0]
    programs = {h["lotlib3"]: h["python"] for h in HYPOTHESES}
    weights = GRAMMAR.prior_weights(programs, kind="rational_rules", dirichlet=RR_ALPHA)
    rules = RuleSet.from_programs(programs, weights=weights)
    posterior = rules.posterior(dataset["trials"], alpha=dataset["alpha"])
    best = max(posterior, key=posterior.get)
    assert all(rules[best](t) == t["label"] for t in dataset["trials"])
