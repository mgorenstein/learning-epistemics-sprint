"""Checks of rule_learner against sources other than LOTlib3.

1. Exact arithmetic: Bayes' rule redone with fractions, no floating point.
2. A textbook base-rate problem with a known answer.
3. The likelihood of Goodman, Tenenbaum, Feldman & Griffiths (2008),
   "A rational analysis of rule-based concept learning": exp(-b * mislabeled).
4. pgmpy: the same model as a Bayesian network, solved by variable elimination.
5. SciPy: the Rational Rules prior as a Dirichlet-multinomial probability.
6. Simulation: when data really come from the model, the posterior is calibrated.

Checks 4 and 5 are skipped when pgmpy or SciPy is not installed.

Run with:  python -m pytest test_independent_checks.py
"""

import math
import random
from fractions import Fraction

import pytest

from rule_learner import BooleanGrammar, RuleSet, all_stimuli

FEATURES = {"color": ["red", "blue", "green"], "shape": ["circle", "square", "triangle"]}
STIMULI = all_stimuli(FEATURES)
PROGRAMS = {
    "red": "lambda t: t['color'] == 'red'",
    "blue": "lambda t: t['color'] == 'blue'",
    "square": "lambda t: t['shape'] == 'square'",
    "circle": "lambda t: t['shape'] == 'circle'",
    "red and square": "lambda t: t['color'] == 'red' and t['shape'] == 'square'",
    "red or square": "lambda t: t['color'] == 'red' or t['shape'] == 'square'",
    "blue or square": "lambda t: t['color'] == 'blue' or t['shape'] == 'square'",
    "not red": "lambda t: t['color'] != 'red'",
    "red and not square": "lambda t: t['color'] == 'red' and t['shape'] != 'square'",
    "not green": "lambda t: t['color'] != 'green'",
}
WEIGHTS = {name: i + 1 for i, name in enumerate(PROGRAMS)}  # deliberately unequal: 1..10


def random_trials(rng, n):
    return [{**rng.choice(STIMULI), "label": rng.random() < 0.5} for _ in range(n)]


# 1 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [Fraction(3, 5), Fraction(4, 5), Fraction(19, 20), Fraction(1)])
def test_matches_exact_fraction_arithmetic(alpha):
    rules = RuleSet.from_programs(PROGRAMS, weights=WEIGHTS)
    rng = random.Random(11)
    for _ in range(40):
        trials = random_trials(rng, rng.randint(0, 14))
        agree, differ = alpha + (1 - alpha) / 2, (1 - alpha) / 2
        scores = []
        for rule in rules:
            score = Fraction(WEIGHTS[rule.name])
            for trial in trials:
                score *= agree if rule(trial) == trial["label"] else differ
            scores.append(score)
        total = sum(scores)
        if total == 0:  # alpha = 1 and no rule fits: the learner must say so
            with pytest.raises(ValueError):
                rules.posterior(trials, alpha=float(alpha))
            continue
        posterior = rules.posterior(trials, alpha=float(alpha))
        for rule, score in zip(rules, scores):
            assert posterior[rule.name] == pytest.approx(float(score / total), abs=1e-12)


# 2 ---------------------------------------------------------------------------------

def test_textbook_base_rate_problem():
    # 1% of people have a condition. A test is positive for 90% of those who
    # have it and for 5% of those who do not. P(condition | positive) is
    # 0.01 * 0.9 / (0.01 * 0.9 + 0.99 * 0.05) = 2/13.
    # alpha = 0.85 with baserate = 1/3 gives exactly those two test rates.
    rules = RuleSet.from_programs(
        {"has it": lambda t: True, "does not": lambda t: False},
        weights={"has it": 1, "does not": 99},
    )
    posterior = rules.posterior([{"label": True}], alpha=0.85, baserate=1 / 3)
    assert posterior["has it"] == pytest.approx(2 / 13)


# 3 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.99])
def test_equals_rational_rules_outlier_likelihood(alpha):
    # Goodman et al. (2008) score a rule by exp(-b * Q), where Q is the number
    # of examples it mislabels. That is this module's likelihood with
    # b = log(agree / differ).
    b = math.log((alpha + (1 - alpha) / 2) / ((1 - alpha) / 2))
    rules = RuleSet.from_programs(PROGRAMS, weights=WEIGHTS)
    rng = random.Random(5)
    for _ in range(25):
        trials = random_trials(rng, rng.randint(1, 12))
        scores = {
            rule.name: WEIGHTS[rule.name] * math.exp(-b * sum(rule(t) != t["label"] for t in trials))
            for rule in rules
        }
        total = sum(scores.values())
        posterior = rules.posterior(trials, alpha=alpha)
        for name in rules.names:
            assert posterior[name] == pytest.approx(scores[name] / total)


# 4 ---------------------------------------------------------------------------------

def test_matches_pgmpy_variable_elimination():
    pytest.importorskip("pgmpy")
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as Network
    except ImportError:  # older pgmpy
        from pgmpy.models import BayesianNetwork as Network

    rules = RuleSet.from_programs(PROGRAMS, weights=WEIGHTS)
    prior = rules.prior()
    rng = random.Random(23)
    for alpha in (0.7, 0.9):
        for _ in range(5):
            trials = random_trials(rng, rng.randint(1, 7))
            labels = [f"label{i}" for i in range(len(trials))]
            network = Network([("rule", label) for label in labels])
            network.add_cpds(TabularCPD("rule", len(rules), [[prior[name]] for name in rules.names]))
            for label, trial in zip(labels, trials):
                p_true = [alpha * rule(trial) + (1 - alpha) / 2 for rule in rules]
                network.add_cpds(TabularCPD(
                    label, 2, [[1 - p for p in p_true], p_true], evidence=["rule"], evidence_card=[len(rules)]))
            assert network.check_model()
            evidence = {label: int(trial["label"]) for label, trial in zip(labels, trials)}
            result = VariableElimination(network).query(["rule"], evidence=evidence, show_progress=False)
            posterior = rules.posterior(trials, alpha=alpha)
            assert list(posterior.values()) == pytest.approx(list(result.values), abs=1e-9)


# 5 ---------------------------------------------------------------------------------

def test_rational_rules_prior_is_a_dirichlet_multinomial():
    stats = pytest.importorskip("scipy.stats")
    if not hasattr(stats, "dirichlet_multinomial"):
        pytest.skip("needs SciPy 1.11 or later")
    grammar = BooleanGrammar(FEATURES)
    options = {"BOOL": list(grammar.BOOL_PRODUCTIONS), "PRED": list(FEATURES), **FEATURES}
    for dirichlet in (0.5, 1.0, 2.0):
        for source in PROGRAMS.values():
            steps = grammar.derivation(source)
            expected = 0.0
            for nonterminal, productions in options.items():
                counts = [steps.count((nonterminal, p)) for p in productions]
                n = sum(counts)
                if n == 0:
                    continue
                # probability of this exact sequence of choices = pmf of the counts
                # divided by the number of orderings with those counts
                orderings = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
                expected += stats.dirichlet_multinomial.logpmf(counts, [dirichlet] * len(counts), n) - orderings
            assert grammar.log_prior(source, kind="rational_rules", dirichlet=dirichlet) == pytest.approx(expected)


# 6 ---------------------------------------------------------------------------------

def test_posterior_is_calibrated_on_data_from_the_model():
    # Draw a rule from the prior and labels from the likelihood, many times.
    # Among all (data set, rule) pairs given posterior probability near p, the
    # rule should be the true one about p of the time.
    alpha, n_sets, n_bins = 0.7, 4000, 5
    rules = RuleSet.from_programs(PROGRAMS, weights=WEIGHTS)
    rng = random.Random(2026)
    names, weights = rules.names, [WEIGHTS[n] for n in rules.names]
    predicted, hits, counts = [0.0] * n_bins, [0] * n_bins, [0] * n_bins
    for _ in range(n_sets):
        truth = rng.choices(names, weights=weights)[0]
        trials = []
        for _ in range(rng.randint(1, 10)):
            stimulus = rng.choice(STIMULI)
            label = rules[truth](stimulus) if rng.random() < alpha else rng.random() < 0.5
            trials.append({**stimulus, "label": label})
        for name, p in rules.posterior(trials, alpha=alpha).items():
            b = min(int(p * n_bins), n_bins - 1)
            predicted[b] += p
            hits[b] += name == truth
            counts[b] += 1
    for b in range(n_bins):
        assert counts[b] > 200, "every probability bin should be well populated"
        assert hits[b] / counts[b] == pytest.approx(predicted[b] / counts[b], abs=0.03)
