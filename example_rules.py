"""Example: the placeholder ten-rule set, scored against a short sequence.

Run with:  python example_rules.py
Replace RULES with the rules from your own experiment.
"""

from rule_learner import BooleanGrammar, RuleSet, all_stimuli, ranked

FEATURES = {
    "color": ["red", "blue", "green"],
    "shape": ["circle", "square", "triangle"],
}

# name -> program. Each program takes a trial dict and returns True or False.
RULES = {
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

if __name__ == "__main__":
    trials = [
        {"color": "red", "shape": "circle", "label": True},
        {"color": "green", "shape": "square", "label": True},
        {"color": "green", "shape": "triangle", "label": False},
        {"color": "blue", "shape": "circle", "label": False},
    ]

    # Two priors over the same rules: uniform, and a simplicity prior from the grammar.
    grammar = BooleanGrammar(FEATURES)
    rule_sets = {
        "uniform prior": RuleSet.from_programs(RULES),
        "simplicity prior": RuleSet.from_programs(RULES, weights=grammar.prior_weights(RULES)),
    }

    for label, rules in rule_sets.items():
        rules.check_distinct(all_stimuli(FEATURES))  # no two rules agree on all 9 objects
        print(label)
        for step, posterior in enumerate(rules.trajectory(trials, alpha=0.8)):
            (leader, p1), (runner_up, p2) = ranked(posterior)[:2]
            seen = "prior" if step == 0 else f"after trial {step}"
            print(f"{seen:>16}:  {leader} {p1:.3f}   |   {runner_up} {p2:.3f}")
        print()
