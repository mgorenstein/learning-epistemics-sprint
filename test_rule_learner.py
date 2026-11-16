"""Tests for rule_learner. Run with:  python -m pytest test_rule_learner.py"""

import math
import random

import pytest

from example_rules import FEATURES, RULES
from rule_learner import BooleanGrammar, Rule, RuleSet, all_stimuli, compile_program, ranked

STIMULI = all_stimuli(FEATURES)


def labeled(stimulus, label):
    return {**stimulus, "label": label}


@pytest.fixture
def rules():
    return RuleSet.from_programs(RULES)


def test_prior_is_normalized_weights():
    rs = RuleSet([Rule("a", lambda t: True, weight=3), Rule("b", lambda t: False, weight=1)])
    assert rs.prior() == {"a": 0.75, "b": 0.25}
    assert rs.posterior([]) == rs.prior()


def test_hand_worked_two_rules():
    # One red circle labeled True, alpha = 0.8, uniform prior.
    # "red" agrees:     0.8 + 0.2/2 = 0.9
    # "square" disagrees:      0.2/2 = 0.1
    # posterior(red) = 0.9 / (0.9 + 0.1) = 0.9
    rs = RuleSet.from_programs({"red": RULES["red"], "square": RULES["square"]})
    post = rs.posterior([{"color": "red", "shape": "circle", "label": True}], alpha=0.8)
    assert post["red"] == pytest.approx(0.9)
    assert post["square"] == pytest.approx(0.1)


def test_posterior_odds_follow_mismatch_count(rules):
    # Under a uniform prior, a rule that disagrees with m labels has odds
    # (q / p) ** m against a rule that agrees with all of them, where
    # p = alpha + (1 - alpha) / 2 and q = (1 - alpha) / 2.
    alpha = 0.7
    p, q = alpha + (1 - alpha) / 2, (1 - alpha) / 2
    truth = rules["red or square"]
    trials = [labeled(s, truth(s)) for s in STIMULI]
    post = rules.posterior(trials, alpha=alpha)
    for rule in rules:
        m = sum(rule(t) != t["label"] for t in trials)
        assert post[rule.name] / post["red or square"] == pytest.approx((q / p) ** m)


def test_sums_to_one_and_order_does_not_matter(rules):
    rng = random.Random(0)
    trials = [labeled(rng.choice(STIMULI), rng.random() < 0.5) for _ in range(25)]
    post = rules.posterior(trials, alpha=0.85)
    assert sum(post.values()) == pytest.approx(1.0)
    shuffled = trials[:]
    rng.shuffle(shuffled)
    again = rules.posterior(shuffled, alpha=0.85)
    for name in rules.names:
        assert again[name] == pytest.approx(post[name])


def test_trajectory_matches_posterior_of_prefixes(rules):
    rng = random.Random(1)
    truth = rules["blue or square"]
    trials = [labeled(s, truth(s)) for s in (rng.choice(STIMULI) for _ in range(8))]
    steps = rules.trajectory(trials, alpha=0.9)
    assert len(steps) == len(trials) + 1
    assert steps[0] == rules.prior()
    for k, step in enumerate(steps):
        prefix = rules.posterior(trials[:k], alpha=0.9)
        for name in rules.names:
            assert step[name] == pytest.approx(prefix[name])


def test_noiseless_labels_eliminate_inconsistent_rules(rules):
    truth = rules["red and square"]
    trials = [labeled(s, truth(s)) for s in STIMULI]
    post = rules.posterior(trials, alpha=1.0)
    assert post["red and square"] == pytest.approx(1.0)
    assert all(p == 0.0 for name, p in post.items() if name != "red and square")


def test_alpha_zero_leaves_the_prior(rules):
    trials = [labeled(s, True) for s in STIMULI[:4]]
    post = rules.posterior(trials, alpha=0.0)
    for name, p in rules.prior().items():
        assert post[name] == pytest.approx(p)


def test_no_rule_can_explain_noiseless_data():
    rs = RuleSet.from_programs({"red": RULES["red"], "blue": RULES["blue"]})
    with pytest.raises(ValueError, match="no rule"):
        rs.posterior([{"color": "green", "shape": "circle", "label": True}], alpha=1.0)


def test_prior_weights_carry_through():
    rs = RuleSet.from_programs(
        {"red": RULES["red"], "not green": RULES["not green"]},
        weights={"red": 1, "not green": 3},
    )
    # Both rules agree on a red object, so the data cannot move the prior.
    post = rs.posterior([{"color": "red", "shape": "circle", "label": True}], alpha=0.9)
    assert post["not green"] == pytest.approx(0.75)


def test_source_callable_and_def_forms_agree():
    as_lambda = Rule("r", "lambda t: t['color'] == 'red' or t['shape'] == 'square'")
    as_callable = Rule("r", lambda t: t["color"] == "red" or t["shape"] == "square")
    as_def = Rule(
        "r",
        """
        def rule(t):
            if t['shape'] == 'square':
                return True
            return t['color'] == 'red'
        """,
    )
    for s in STIMULI:
        assert as_lambda(s) == as_callable(s) == as_def(s)
    assert as_callable.source is None and as_lambda.source.startswith("lambda")


def test_primitives_are_available_to_source():
    prims = {"is_color": lambda t, c: t["color"] == c, "or_": lambda a, b: a or b}
    rule = Rule("r", "lambda t: or_(is_color(t, 'red'), is_color(t, 'blue'))", primitives=prims)
    assert [rule(s) for s in STIMULI] == [s["color"] != "green" for s in STIMULI]


def test_bad_inputs_are_reported():
    with pytest.raises(ValueError):
        compile_program("42")
    with pytest.raises(ValueError, match="unique"):
        RuleSet([Rule("a", lambda t: True), Rule("a", lambda t: False)])
    rs = RuleSet([Rule("needs size", lambda t: t["size"] > 1)])
    with pytest.raises(RuntimeError, match="needs size"):
        rs.posterior([{"color": "red", "label": True}])
    with pytest.raises(KeyError, match="label"):
        RuleSet([Rule("a", lambda t: True)]).posterior([{"color": "red"}])
    with pytest.raises(ValueError, match="alpha"):
        RuleSet([Rule("a", lambda t: True)]).posterior([], alpha=1.5)


def test_distinctness_check(rules):
    rules.check_distinct(STIMULI)
    clash = RuleSet.from_programs({"not green": RULES["not green"],
                                   "red or blue": "lambda t: t['color'] in ('red', 'blue')"})
    assert clash.equivalent_rules(STIMULI) == [["not green", "red or blue"]]
    with pytest.raises(ValueError, match="distinguished"):
        clash.check_distinct(STIMULI)


def test_ranked_orders_by_probability():
    assert ranked({"a": 0.2, "b": 0.7, "c": 0.1}) == [("b", 0.7), ("a", 0.2), ("c", 0.1)]


def test_temperature_flattens_the_likelihood(rules):
    truth = rules["red"]
    trials = [labeled(s, truth(s)) for s in STIMULI]
    sharp = rules.posterior(trials, alpha=0.8)["red"]
    flat = rules.posterior(trials, alpha=0.8, likelihood_temperature=5.0)["red"]
    assert flat < sharp
    assert math.isclose(sum(rules.posterior(trials, alpha=0.8, likelihood_temperature=5.0).values()), 1.0)


# ---- simplicity prior -------------------------------------------------------

@pytest.fixture
def grammar():
    return BooleanGrammar(FEATURES)


def test_pcfg_prior_hand_worked(grammar):
    # BOOL has 4 productions, PRED has 2 features, each feature has 3 values.
    # "red":           1/4 * 1/2 * 1/3                 = 1/24
    # "not red":       1/4 * (1/24)                    = 1/96
    # "red or square": 1/4 * (1/24) * (1/24)           = 1/2304
    assert math.exp(grammar.log_prior(RULES["red"])) == pytest.approx(1 / 24)
    assert math.exp(grammar.log_prior(RULES["not red"])) == pytest.approx(1 / 96)
    assert math.exp(grammar.log_prior(RULES["red or square"])) == pytest.approx(1 / 2304)
    assert math.exp(grammar.log_prior(RULES["red and not square"])) == pytest.approx(1 / 2304 / 4)


def test_rational_rules_prior_hand_worked(grammar):
    # For "red" with dirichlet = 1 the Dirichlet-multinomial terms are
    # BOOL 1/4, PRED 1/2, color 1/3, shape 1, so it equals the PCFG value.
    assert math.exp(grammar.log_prior(RULES["red"], kind="rational_rules")) == pytest.approx(1 / 24)
    # Reusing a production is penalized less than under the PCFG prior ...
    reuse = "lambda t: t['color'] == 'red' or t['color'] == 'red'"
    mixed = "lambda t: t['color'] == 'red' or t['shape'] == 'square'"
    rr = lambda src: grammar.log_prior(src, kind="rational_rules")
    assert rr(reuse) > rr(mixed)
    # ... while the PCFG prior treats the two alike.
    assert grammar.log_prior(reuse) == pytest.approx(grammar.log_prior(mixed))


def test_equivalent_spellings_have_the_same_derivation(grammar):
    assert grammar.derivation("lambda t: t['color'] != 'red'") == grammar.derivation(
        "lambda t: not t['color'] == 'red'")
    assert grammar.derivation("lambda x: 'red' == x['color']") == grammar.derivation(RULES["red"])
    three = grammar.derivation("lambda t: t['color'] == 'red' or t['color'] == 'blue' or t['shape'] == 'square'")
    assert three.count(("BOOL", "or")) == 2 and three.count(("BOOL", "pred")) == 3


def test_simpler_rules_get_more_prior(grammar):
    rules = RuleSet.from_programs(RULES, weights=grammar.prior_weights(RULES))
    prior = rules.prior()
    assert sum(prior.values()) == pytest.approx(1.0)
    assert prior["red"] > prior["not red"] > prior["red or square"] > prior["red and not square"]
    flat = RuleSet.from_programs(RULES, weights=grammar.prior_weights(RULES, temperature=4)).prior()
    assert flat["red"] < prior["red"] and flat["red or square"] > prior["red or square"]


def test_production_weights_shift_the_prior():
    cheap_or = BooleanGrammar(FEATURES, weights={"or": 5})
    plain = BooleanGrammar(FEATURES)
    gap = lambda g: g.log_prior(RULES["red or square"]) - g.log_prior(RULES["red and square"])
    assert gap(plain) == pytest.approx(0.0)
    assert gap(cheap_or) == pytest.approx(math.log(5))


def test_unreadable_rules_are_reported(grammar):
    for bad in ("lambda t: t['color'] in ('red', 'blue')",
                "lambda t: t['size'] == 'big'",
                "lambda t: t['color'] == 'purple'",
                "lambda t, u: True",
                "42"):
        with pytest.raises(ValueError):
            grammar.log_prior(bad)
