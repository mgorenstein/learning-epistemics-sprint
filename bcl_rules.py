"""Sixteen-object stimulus space and ten Boolean rules for Bayesian concept learning.

Run with:  python bcl_rules.py
Expands FEATURES to 4 colors x 4 shapes = 16 distinct objects so each block can present
every stimulus once (repetitions: 1, 16 trials/block) with zero exemplar repetition.
"""

from rule_learner import BooleanGrammar, RuleSet, all_stimuli, ranked

FEATURES = {
    "color": ["red", "blue", "green", "yellow"],
    "shape": ["circle", "square", "triangle", "diamond"],
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
    stimuli = all_stimuli(FEATURES)
    grammar = BooleanGrammar(FEATURES)
    rules = RuleSet.from_programs(RULES, weights=grammar.prior_weights(RULES, temperature=3.0))
    rules.check_distinct(stimuli)
    print(f"{len(stimuli)} unique stimuli, {len(rules)} distinct rules")
    for name, prob in ranked(rules.prior()):
        print(f"  {name:<20}: prior {prob:.3f}")
