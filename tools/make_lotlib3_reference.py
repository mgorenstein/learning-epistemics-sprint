"""Regenerate reference/lotlib3_reference.json by running LOTlib3 itself.

Usage:
    python tools/make_lotlib3_reference.py /path/to/parent/of/LOTlib3  [output.json]

The argument is the directory that *contains* the LOTlib3 package, for example
the folder you ran ``git clone https://github.com/thelogicalgrammar/LOTlib3`` in.

Nothing here is copied from LOTlib3. The script imports it, builds hypotheses
from its ``SimpleBoolean_noTF`` default grammar with the predicates and data of
its Rational Rules example (``LOTlib3/Examples/RationalRules/Model.py``), and
records the numbers LOTlib3 computes:

* each hypothesis's truth value on every object,
* its PCFG prior (``LOTHypothesis.compute_prior``),
* its Rational Rules prior (``RationaRulesPrior.compute_prior``, rrAlpha = 2),
* its log likelihood on several data sets (``BinaryLikelihood``),
* the posterior over sets of ten hypotheses, from LOTlib3's prior + likelihood.

``test_lotlib3_reference.py`` checks that ``rule_learner`` reproduces them.
"""

import json
import math
import random
import subprocess
import sys
from pathlib import Path

lotlib_parent = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "reference" / "lotlib3_reference.json"
sys.path.insert(0, str(lotlib_parent))

import numpy  # noqa: E402  (LOTlib3 depends on it)

from LOTlib3.DataAndObjects import FunctionData, Obj, make_all_objects  # noqa: E402
from LOTlib3.DefaultGrammars import SimpleBoolean_noTF as grammar  # noqa: E402
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood  # noqa: E402
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis  # noqa: E402
from LOTlib3.Hypotheses.Priors.RationalRules import RationaRulesPrior  # noqa: E402
from LOTlib3.Miscellaneous import q  # noqa: E402

SEED = 20261112
RR_ALPHA = 2.0  # as in LOTlib3's Rational Rules example
FEATURES = {  # the predicates of LOTlib3's Rational Rules example
    "color": ["red", "blue", "green", "mauve"],
    "shape": ["square", "circle", "triangle", "diamond"],
}

grammar.add_rule("PREDICATE", "is_color_", ["x", "COLOR"], 1.0)
grammar.add_rule("PREDICATE", "is_shape_", ["x", "SHAPE"], 1.0)
for color in FEATURES["color"]:
    grammar.add_rule("COLOR", q(color), None, 1.0)
for shape in FEATURES["shape"]:
    grammar.add_rule("SHAPE", q(shape), None, 1.0)


class PCFGHypothesis(BinaryLikelihood, LOTHypothesis):
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, **kwargs)


class RRHypothesis(RationaRulesPrior, BinaryLikelihood, LOTHypothesis):
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, **kwargs)
        self.rrAlpha = RR_ALPHA


def to_python(node):
    """A LOTlib3 tree as the Python expression rule_learner uses for the same rule."""
    name = node.name
    if name == "":  # START -> BOOL and BOOL -> PREDICATE pass straight through
        return to_python(node.args[0])
    if name in ("and_", "or_"):
        return f"({to_python(node.args[0])} {name[:-1]} {to_python(node.args[1])})"
    if name == "not_":
        return f"(not {to_python(node.args[0])})"
    feature = {"is_color_": "color", "is_shape_": "shape"}[name]
    value = node.args[1].name.strip("'\"")
    return f"t['{feature}'] == '{value}'"


def datum(obj, label, alpha):
    return FunctionData(input=[obj], output=label, alpha=alpha)


def as_trial(d):
    return {"color": d.input[0].color, "shape": d.input[0].shape, "label": bool(d.output)}


random.seed(SEED)
numpy.random.seed(SEED)
objects = make_all_objects(color=FEATURES["color"], shape=FEATURES["shape"])

# --- data sets -------------------------------------------------------------------
datasets = [{
    "name": "LOTlib3 Rational Rules example, make_data()",
    "alpha": 0.999,
    "data": [datum(Obj(shape="square", color="red"), True, 0.999),
             datum(Obj(shape="square", color="blue"), False, 0.999),
             datum(Obj(shape="triangle", color="blue"), False, 0.999),
             datum(Obj(shape="triangle", color="red"), False, 0.999)],
}]
for alpha in (0.6, 0.8, 0.95):
    for length in (3, 8, 15):
        datasets.append({
            "name": f"random labels, alpha={alpha}, n={length}",
            "alpha": alpha,
            "data": [datum(random.choice(objects), random.random() < 0.5, alpha) for _ in range(length)],
        })

# --- hypotheses ------------------------------------------------------------------
hypotheses, seen = [], set()
while len(hypotheses) < 120:
    h = PCFGHypothesis()  # LOTlib3 samples a tree from the grammar
    text = str(h)
    pcfg_prior = h.compute_prior()
    if text in seen or not math.isfinite(pcfg_prior):  # skip repeats and trees over maxnodes
        continue
    seen.add(text)
    rr = RRHypothesis(value=h.value)
    hypotheses.append({
        "lotlib3": text,
        "python": "lambda t: " + to_python(h.value),
        "truth_table": [bool(h(o)) for o in objects],
        "pcfg_log_prior": pcfg_prior,
        "rational_rules_log_prior": rr.compute_prior(),
        "log_likelihood": [h.compute_likelihood(ds["data"]) for ds in datasets],
    })

# --- posteriors over sets of ten hypotheses ---------------------------------------
posterior_cases = []
for _ in range(12):
    members = random.sample(range(len(hypotheses)), 10)
    for prior_kind in ("pcfg_log_prior", "rational_rules_log_prior"):
        for ds_index in random.sample(range(len(datasets)), 3):
            scores = [hypotheses[i][prior_kind] + hypotheses[i]["log_likelihood"][ds_index] for i in members]
            top = max(scores)
            z = sum(math.exp(s - top) for s in scores)
            posterior_cases.append({
                "hypotheses": members,
                "prior": prior_kind.replace("_log_prior", ""),
                "dataset": ds_index,
                "posterior": [math.exp(s - top) / z for s in scores],
            })

try:
    commit = subprocess.run(["git", "-C", str(lotlib_parent / "LOTlib3"), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
except OSError:
    commit = "unknown"

reference = {
    "source": {
        "project": "LOTlib3",
        "url": "https://github.com/thelogicalgrammar/LOTlib3",
        "commit": commit,
        "grammar": "LOTlib3.DefaultGrammars.SimpleBoolean_noTF (predicate weight 5.0) with the "
                   "is_color_/is_shape_ predicates of Examples/RationalRules/Model.py",
        "rational_rules_alpha": RR_ALPHA,
        "seed": SEED,
        "generated_by": "tools/make_lotlib3_reference.py",
    },
    "features": FEATURES,
    "objects": [{"color": o.color, "shape": o.shape} for o in objects],
    "datasets": [{"name": ds["name"], "alpha": ds["alpha"], "trials": [as_trial(d) for d in ds["data"]]}
                 for ds in datasets],
    "hypotheses": hypotheses,
    "posterior_cases": posterior_cases,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(reference, indent=1) + "\n")
print(f"wrote {output}: {len(hypotheses)} hypotheses, {len(datasets)} data sets, {len(posterior_cases)} posteriors")
