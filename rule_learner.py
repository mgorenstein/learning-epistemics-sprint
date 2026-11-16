"""Rules as Python programs, and exact Bayesian posteriors over a set of them.

This is a small, dependency-free replacement for the part of LOTlib3 that scores
a *fixed* set of hypotheses against labeled data. It does not generate rules
from a grammar or run MCMC; you hand it the rules.

The three ideas
---------------
Trial
    A plain dict describing one stimulus, plus the label that was observed for
    it. For example ``{"color": "red", "shape": "square", "label": True}``.
    Rules read whatever keys they need; the learner reads only the label.

Rule
    A named Python program that takes a trial and returns True or False:
    "is this object an example of the word?". The program can be a function, a
    lambda, or source code in a string.

RuleSet
    An ordered collection of rules with prior weights. It answers one question:
    given these trials, how probable is each rule?

The model
---------
Prior. Each rule has a non-negative weight. The prior is the weights divided by
their sum, so equal weights give a uniform prior. For a simplicity prior, where
shorter rules are more probable, compute the weights with
:class:`BooleanGrammar` (see "Simplicity prior" below).

Likelihood. This is LOTlib3's ``BinaryLikelihood``. With probability ``alpha``
the observed label is the rule's own answer; otherwise it is a coin flip that
comes up True with probability ``baserate``::

    P(label is True | rule, trial) = alpha * rule(trial) + (1 - alpha) * baserate

With the default ``baserate=0.5`` this is exactly LOTlib3's
``alpha * (h(x) == y) + (1 - alpha) / 2``: a label that agrees with the rule has
probability ``alpha + (1 - alpha) / 2`` and one that disagrees has
``(1 - alpha) / 2``.

Posterior. Bayes' rule over the rule set, computed in log space::

    P(rule | trials)  is proportional to  P(rule) * product of P(label_i | rule, trial_i)

Simplicity prior
----------------
LOTlib3 scores a rule by the probability of generating it from a grammar: every
choice made while writing the rule down costs probability, so longer rules are
less probable. :class:`BooleanGrammar` does the same for rules written as
Python expressions over a trial's features::

    grammar = BooleanGrammar({"color": [...], "shape": [...]})
    rules = RuleSet.from_programs(RULES, weights=grammar.prior_weights(RULES))

What is deliberately not here
-----------------------------
The size principle. Objects are assumed to be chosen independently of the rule
and then labeled (weak sampling), so how many objects a rule covers does not
enter the likelihood: a rule is rewarded only for agreeing with the labels. If
your examples are instead drawn from the rule's positive set, the likelihood
needs an extra factor of ``1 / (number of objects the rule covers)`` per
example, which this module does not apply.

Quick start
-----------
>>> rules = RuleSet([
...     Rule("red", "lambda t: t['color'] == 'red'"),
...     Rule("square", "lambda t: t['shape'] == 'square'"),
...     Rule("red or square", "lambda t: t['color'] == 'red' or t['shape'] == 'square'"),
... ])
>>> trials = [
...     {"color": "red", "shape": "circle", "label": True},
...     {"color": "blue", "shape": "square", "label": True},
... ]
>>> post = rules.posterior(trials, alpha=0.9)
>>> max(post, key=post.get)
'red or square'
"""

from __future__ import annotations

import ast
import math
import textwrap
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Trial = Mapping[str, Any]
Program = Callable[[Trial], Any]

__all__ = [
    "Trial",
    "Rule",
    "RuleSet",
    "BooleanGrammar",
    "compile_program",
    "all_stimuli",
    "ranked",
]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def compile_program(source: str, primitives: Optional[Mapping[str, Any]] = None) -> Program:
    """Turn Python source into a function of one trial.

    Two forms are accepted.

    An expression that evaluates to a function, usually a lambda::

        "lambda t: t['color'] == 'red'"

    Or a single ``def``, for rules that need more than one line::

        '''
        def rule(t):
            if t['shape'] == 'square':
                return True
            return t['color'] == 'red'
        '''

    ``primitives`` is an optional dict of names the source may use, in the way
    LOTlib3 rules call primitives such as ``and_`` or ``is_color_``. For
    example ``{"is_color": lambda t, c: t["color"] == c}`` lets a rule be
    written as ``"lambda t: is_color(t, 'red')"``.

    The source is executed with ``eval``/``exec``. Only compile source you
    wrote or trust.
    """
    namespace: Dict[str, Any] = dict(primitives or {})
    src = textwrap.dedent(source).strip()
    if not src:
        raise ValueError("rule source is empty")

    if src.startswith("def "):
        before = set(namespace)
        exec(src, namespace)  # noqa: S102 - trusted, user-written rule source
        defined = [
            value for key, value in namespace.items()
            if key not in before and key != "__builtins__" and callable(value)
        ]
        if len(defined) != 1:
            raise ValueError(
                f"a 'def' rule must define exactly one function; found {len(defined)}"
            )
        return defined[0]

    program = eval(src, namespace)  # noqa: S307 - trusted, user-written rule source
    if not callable(program):
        raise ValueError(
            "rule source must evaluate to a function of one trial, "
            f"for example \"lambda t: t['color'] == 'red'\"; got {type(program).__name__}"
        )
    return program


class Rule:
    """A named program that says whether a trial is an example of the word.

    Parameters
    ----------
    name
        A human-readable label, such as ``"red or square"``. Names must be
        unique within a RuleSet; posteriors are reported under this name.
    program
        Either a callable taking a trial dict, or Python source for one (see
        :func:`compile_program`). Whatever it returns is converted with
        ``bool``.
    weight
        Unnormalized prior weight. Only ratios between rules matter.
    primitives
        Names made available to ``program`` when it is given as source.

    Calling the rule applies the program: ``rule(trial) -> bool``.
    """

    def __init__(
        self,
        name: str,
        program: Union[Program, str],
        weight: float = 1.0,
        primitives: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a rule needs a non-empty name")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"rule {name!r}: weight must be a finite number >= 0, got {weight}")

        self.name = name
        self.weight = weight
        if isinstance(program, str):
            #: The source text, kept so rules can be saved, printed or shown in a prompt.
            self.source: Optional[str] = textwrap.dedent(program).strip()
            self.program: Program = compile_program(program, primitives)
        elif callable(program):
            self.source = None
            self.program = program
        else:
            raise TypeError(f"rule {name!r}: program must be a callable or source string")

    def __call__(self, trial: Trial) -> bool:
        try:
            return bool(self.program(trial))
        except Exception as err:  # make it obvious which rule and trial broke
            raise RuntimeError(f"rule {self.name!r} failed on trial {dict(trial)!r}: {err}") from err

    def __repr__(self) -> str:
        shown = self.source if self.source is not None else "<callable>"
        return f"Rule({self.name!r}, {shown!r}, weight={self.weight:g})"


# ---------------------------------------------------------------------------
# Rule sets and inference
# ---------------------------------------------------------------------------

class RuleSet:
    """An ordered set of rules with a prior, scored against trials.

    Parameters
    ----------
    rules
        The candidate rules. Order is preserved in every result.

    Every inference method takes the same keyword arguments:

    alpha
        Probability that a label comes from the rule and not from noise.
        1.0 means labels are never wrong; 0.0 means labels carry no information.
    baserate
        Probability that a noise label is True. 0.5 reproduces LOTlib3.
    label_key
        The key in each trial dict that holds the observed label.
    likelihood_temperature, prior_temperature
        LOTlib3's temperatures: the log likelihood and log prior are divided by
        these before combining. Leave both at 1.0 for ordinary Bayes.
    """

    def __init__(self, rules: Iterable[Rule]) -> None:
        self.rules: List[Rule] = list(rules)
        if not self.rules:
            raise ValueError("a RuleSet needs at least one rule")
        names = [rule.name for rule in self.rules]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"rule names must be unique; repeated: {duplicates}")
        if sum(rule.weight for rule in self.rules) <= 0:
            raise ValueError("at least one rule must have a positive prior weight")

    @classmethod
    def from_programs(
        cls,
        programs: Mapping[str, Union[Program, str]],
        weights: Optional[Mapping[str, float]] = None,
        primitives: Optional[Mapping[str, Any]] = None,
    ) -> "RuleSet":
        """Build a RuleSet from ``{name: program}``, with optional ``{name: weight}``."""
        weights = weights or {}
        return cls(
            Rule(name, program, weight=weights.get(name, 1.0), primitives=primitives)
            for name, program in programs.items()
        )

    # -- container conveniences ------------------------------------------------

    @property
    def names(self) -> List[str]:
        return [rule.name for rule in self.rules]

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __getitem__(self, name: str) -> Rule:
        for rule in self.rules:
            if rule.name == name:
                return rule
        raise KeyError(name)

    def __repr__(self) -> str:
        return f"RuleSet({self.names!r})"

    # -- the three ingredients ---------------------------------------------------

    def prior(self) -> Dict[str, float]:
        """Prior probability of each rule: its weight over the sum of weights."""
        total = sum(rule.weight for rule in self.rules)
        return {rule.name: rule.weight / total for rule in self.rules}

    def label_probability(
        self, rule: Rule, trial: Trial, *, alpha: float = 0.9, baserate: float = 0.5
    ) -> float:
        """P(a label of True is observed for this trial | this rule is the true one)."""
        _check_noise(alpha, baserate)
        return alpha * (1.0 if rule(trial) else 0.0) + (1.0 - alpha) * baserate

    def log_likelihood(
        self,
        rule: Rule,
        trials: Sequence[Trial],
        *,
        alpha: float = 0.9,
        baserate: float = 0.5,
        label_key: str = "label",
    ) -> float:
        """Log probability of all the observed labels if ``rule`` were true.

        Returns ``-inf`` when the rule makes some observed label impossible,
        which can only happen at ``alpha == 1`` or with a baserate of 0 or 1.
        """
        total = 0.0
        for trial in trials:
            p_true = self.label_probability(rule, trial, alpha=alpha, baserate=baserate)
            p_observed = p_true if _label(trial, label_key) else 1.0 - p_true
            if p_observed <= 0.0:
                return -math.inf
            total += math.log(p_observed)
        return total

    def posterior(
        self,
        trials: Sequence[Trial],
        *,
        alpha: float = 0.9,
        baserate: float = 0.5,
        label_key: str = "label",
        likelihood_temperature: float = 1.0,
        prior_temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Posterior probability of each rule given all of ``trials``.

        Returns ``{rule name: probability}`` in rule order; the values sum to 1.
        With no trials this is the prior.
        """
        return self.trajectory(
            trials,
            alpha=alpha,
            baserate=baserate,
            label_key=label_key,
            likelihood_temperature=likelihood_temperature,
            prior_temperature=prior_temperature,
        )[-1]

    def trajectory(
        self,
        trials: Sequence[Trial],
        *,
        alpha: float = 0.9,
        baserate: float = 0.5,
        label_key: str = "label",
        likelihood_temperature: float = 1.0,
        prior_temperature: float = 1.0,
    ) -> List[Dict[str, float]]:
        """The posterior after every trial.

        Returns a list of ``len(trials) + 1`` posteriors. Entry 0 is the prior,
        and entry ``k`` is the posterior after the first ``k`` trials, so
        ``trajectory(trials)[k] == posterior(trials[:k])``.
        """
        _check_noise(alpha, baserate)
        if likelihood_temperature <= 0 or prior_temperature <= 0:
            raise ValueError("temperatures must be positive")

        log_scores = [
            (math.log(rule.weight) / prior_temperature) if rule.weight > 0 else -math.inf
            for rule in self.rules
        ]
        steps = [_normalize(self.names, log_scores)]
        for index, trial in enumerate(trials):
            observed = _label(trial, label_key)
            for i, rule in enumerate(self.rules):
                p_true = alpha * (1.0 if rule(trial) else 0.0) + (1.0 - alpha) * baserate
                p_observed = p_true if observed else 1.0 - p_true
                log_scores[i] += (
                    math.log(p_observed) / likelihood_temperature if p_observed > 0 else -math.inf
                )
            if all(score == -math.inf for score in log_scores):
                raise ValueError(
                    f"after trial {index} no rule in the set can produce the observed labels; "
                    "use alpha < 1 to allow for label noise"
                )
            steps.append(_normalize(self.names, log_scores))
        return steps

    # -- what the rules say about stimuli ---------------------------------------

    def extension(self, rule: Union[Rule, str], stimuli: Sequence[Trial]) -> Tuple[bool, ...]:
        """The rule's answer for each stimulus, in order."""
        rule = self[rule] if isinstance(rule, str) else rule
        return tuple(rule(stimulus) for stimulus in stimuli)

    def equivalent_rules(self, stimuli: Sequence[Trial]) -> List[List[str]]:
        """Groups of rules that give identical answers on every stimulus.

        Rules in the same group can never be told apart by data drawn from
        ``stimuli``; the posterior will only ever split between them according
        to their prior weights. An empty list means every rule is distinct.
        """
        groups: Dict[Tuple[bool, ...], List[str]] = {}
        for rule in self.rules:
            groups.setdefault(self.extension(rule, stimuli), []).append(rule.name)
        return [names for names in groups.values() if len(names) > 1]

    def check_distinct(self, stimuli: Sequence[Trial]) -> None:
        """Raise ValueError if two rules agree on every stimulus."""
        clashes = self.equivalent_rules(stimuli)
        if clashes:
            raise ValueError(f"these rules cannot be distinguished on the given stimuli: {clashes}")


# ---------------------------------------------------------------------------
# Simplicity prior
# ---------------------------------------------------------------------------

class BooleanGrammar:
    """A grammar over Boolean rules, used to give simpler rules a higher prior.

    The grammar has one way to build each piece of a rule::

        BOOL  -> PRED | BOOL and BOOL | BOOL or BOOL | not BOOL
        PRED  -> t[feature] == VALUE          one production per feature
        VALUE -> each value of that feature   one nonterminal per feature

    A rule's *derivation* is the list of choices needed to write it. For
    ``t['color'] == 'red'`` with three colors and three shapes that is: choose
    PRED (1 of 4 BOOL productions), choose the color feature (1 of 2), choose
    red (1 of 3). Its probability is 1/4 * 1/2 * 1/3 = 1/24. Every extra
    connective or feature test multiplies in more choices, so longer rules get
    smaller priors.

    Parameters
    ----------
    features
        ``{feature name: list of values}``, the same dict you would pass to
        :func:`all_stimuli`.
    weights
        Relative weights of the four BOOL productions, as a dict with any of
        the keys ``"pred"``, ``"and"``, ``"or"``, ``"not"``. Missing keys
        default to 1.0, which matches a LOTlib3 grammar whose rules all have
        ``p=1.0``. Features and values are always chosen uniformly.

    Rules must be given as source strings: a one-argument ``lambda`` whose body
    uses only ``and``, ``or``, ``not``, ``==`` and ``!=`` on ``t[feature]`` and
    a constant. ``a != b`` is read as ``not (a == b)``, and ``a or b or c`` as
    ``a or (b or c)``. For rules written any other way, set ``Rule.weight``
    yourself.

    Two priors are available through ``kind``:

    ``"pcfg"``
        The product of the production probabilities in the derivation. This is
        what LOTlib3's ``grammar.log_probability`` computes, and what a plain
        ``LOTHypothesis`` uses as its prior.
    ``"rational_rules"``
        The Rational Rules prior of Goodman et al. (2008), as in LOTlib3's
        ``RR_prior``. The production probabilities are unknown and integrated
        out under a symmetric Dirichlet with parameter ``dirichlet``. Rules
        that reuse the same productions are penalized less. The BOOL
        ``weights`` are ignored.
    """

    BOOL_PRODUCTIONS = ("pred", "and", "or", "not")

    def __init__(
        self,
        features: Mapping[str, Sequence[Any]],
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not features or any(len(values) == 0 for values in features.values()):
            raise ValueError("features must map each feature name to a non-empty list of values")
        self.features: Dict[str, List[Any]] = {name: list(values) for name, values in features.items()}
        unknown = set(weights or {}) - set(self.BOOL_PRODUCTIONS)
        if unknown:
            raise ValueError(f"unknown production weights: {sorted(unknown)}")
        self.weights: Dict[str, float] = {
            name: float((weights or {}).get(name, 1.0)) for name in self.BOOL_PRODUCTIONS
        }
        if any(w <= 0 for w in self.weights.values()):
            raise ValueError("production weights must be positive")

    @classmethod
    def lotlib3_simple_boolean(cls, features: Mapping[str, Sequence[Any]]) -> "BooleanGrammar":
        """The weights of LOTlib3's ``SimpleBoolean_noTF`` default grammar.

        That grammar gives the predicate production a weight of 5.0
        (``DEFAULT_FEATURE_WEIGHT``) and ``and``, ``or`` and ``not`` a weight
        of 1.0 each, so a bare feature test is five times as likely as any one
        connective.
        """
        return cls(features, weights={"pred": 5.0})

    # -- reading a rule as a derivation ------------------------------------------

    def derivation(self, source: str) -> List[Tuple[str, Any]]:
        """The choices needed to write ``source``, as ``(nonterminal, production)`` pairs.

        >>> g = BooleanGrammar({"color": ["red", "blue"], "shape": ["square"]})
        >>> g.derivation("lambda t: t['color'] == 'red'")
        [('BOOL', 'pred'), ('PRED', 'color'), ('color', 'red')]
        """
        try:
            tree = ast.parse(textwrap.dedent(source).strip(), mode="eval").body
        except SyntaxError as err:
            raise ValueError(f"cannot parse rule source {source!r}: {err}") from err
        if not isinstance(tree, ast.Lambda) or len(tree.args.args) != 1:
            raise ValueError(
                "the simplicity prior needs a one-argument lambda, "
                f"for example \"lambda t: t['color'] == 'red'\"; got {source!r}"
            )
        steps: List[Tuple[str, Any]] = []
        self._walk(tree.body, tree.args.args[0].arg, steps, source)
        return steps

    def _walk(self, node: ast.AST, arg: str, steps: List[Tuple[str, Any]], source: str) -> None:
        if isinstance(node, ast.BoolOp):
            name = "and" if isinstance(node.op, ast.And) else "or"
            # a or b or c  ->  a or (b or c): k operands need k - 1 binary nodes
            for position, value in enumerate(node.values):
                if position < len(node.values) - 1:
                    steps.append(("BOOL", name))
                self._walk(value, arg, steps, source)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            steps.append(("BOOL", "not"))
            self._walk(node.operand, arg, steps, source)
            return
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            feature, value = self._feature_test(node.left, node.comparators[0], arg, source)
            if isinstance(node.ops[0], ast.NotEq):
                steps.append(("BOOL", "not"))
            steps.extend([("BOOL", "pred"), ("PRED", feature), (feature, value)])
            return
        raise ValueError(
            f"the simplicity prior cannot read this part of {source!r}: "
            f"{ast.unparse(node)!r}. Use only and, or, not, == and != on t[feature], "
            "or set the rule's weight yourself."
        )

    def _feature_test(self, left: ast.AST, right: ast.AST, arg: str, source: str) -> Tuple[str, Any]:
        for lookup, constant in ((left, right), (right, left)):
            if (
                isinstance(lookup, ast.Subscript)
                and isinstance(lookup.value, ast.Name)
                and lookup.value.id == arg
                and isinstance(lookup.slice, ast.Constant)
                and isinstance(constant, ast.Constant)
            ):
                feature, value = lookup.slice.value, constant.value
                if feature not in self.features:
                    raise ValueError(f"{source!r} tests unknown feature {feature!r}")
                if value not in self.features[feature]:
                    raise ValueError(f"{source!r} tests {feature!r} against unknown value {value!r}")
                return feature, value
        raise ValueError(f"{source!r}: comparisons must look like t['feature'] == 'value'")

    # -- scoring -------------------------------------------------------------------

    def log_prior(self, source: str, kind: str = "pcfg", dirichlet: float = 1.0) -> float:
        """Log prior of one rule. Not normalized over any particular rule set."""
        steps = self.derivation(source)
        if kind == "pcfg":
            total_weight = sum(self.weights.values())
            log_p = 0.0
            for nonterminal, production in steps:
                if nonterminal == "BOOL":
                    log_p += math.log(self.weights[production] / total_weight)
                elif nonterminal == "PRED":
                    log_p -= math.log(len(self.features))
                else:  # a value of the feature named by `nonterminal`
                    log_p -= math.log(len(self.features[nonterminal]))
            return log_p
        if kind == "rational_rules":
            if dirichlet <= 0:
                raise ValueError("dirichlet must be positive")
            options: Dict[str, List[Any]] = {"BOOL": list(self.BOOL_PRODUCTIONS), "PRED": list(self.features)}
            options.update(self.features)
            log_p = 0.0
            for nonterminal, productions in options.items():
                counts = [sum(1 for step in steps if step == (nonterminal, p)) for p in productions]
                log_p += _log_beta([c + dirichlet for c in counts]) - _log_beta([dirichlet] * len(counts))
            return log_p
        raise ValueError(f"kind must be 'pcfg' or 'rational_rules', got {kind!r}")

    def prior_weights(
        self,
        programs: Mapping[str, str],
        kind: str = "pcfg",
        dirichlet: float = 1.0,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Prior weights for ``{name: source}``, ready to pass as ``weights=``.

        ``temperature`` flattens the prior: weights are raised to the power
        ``1 / temperature``, which is what LOTlib3's ``prior_temperature``
        does. Values above 1 make long rules less improbable.

        The weights are scaled so the largest is 1; :class:`RuleSet` normalizes
        them over the rule set.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logs = {
            name: self.log_prior(source, kind=kind, dirichlet=dirichlet) / temperature
            for name, source in programs.items()
        }
        top = max(logs.values())
        return {name: math.exp(value - top) for name, value in logs.items()}


def _log_beta(values: Sequence[float]) -> float:
    """Log of the multivariate Beta function, the Dirichlet normalizer."""
    return sum(math.lgamma(v) for v in values) - math.lgamma(sum(values))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def all_stimuli(features: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Every combination of feature values, as unlabeled trial dicts.

    >>> all_stimuli({"color": ["red", "blue"], "shape": ["square"]})
    [{'color': 'red', 'shape': 'square'}, {'color': 'blue', 'shape': 'square'}]
    """
    keys = list(features)
    return [dict(zip(keys, values)) for values in product(*(features[key] for key in keys))]


def ranked(posterior: Mapping[str, float]) -> List[Tuple[str, float]]:
    """A posterior as ``(name, probability)`` pairs, most probable first.

    ``ranked(post)[0]`` is the leading rule and ``ranked(post)[1]`` the runner-up.
    Ties keep rule order.
    """
    return sorted(posterior.items(), key=lambda item: -item[1])


def _label(trial: Trial, label_key: str) -> bool:
    if label_key not in trial:
        raise KeyError(f"trial {dict(trial)!r} has no {label_key!r} key holding the observed label")
    return bool(trial[label_key])


def _check_noise(alpha: float, baserate: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if not 0.0 <= baserate <= 1.0:
        raise ValueError(f"baserate must be between 0 and 1, got {baserate}")


def _normalize(names: Sequence[str], log_scores: Sequence[float]) -> Dict[str, float]:
    """Exponentiate and normalize log scores without underflow (log-sum-exp)."""
    top = max(log_scores)
    if top == -math.inf:
        raise ValueError("every rule has zero probability")
    unnormalized = [math.exp(score - top) for score in log_scores]
    total = sum(unnormalized)
    return {name: value / total for name, value in zip(names, unnormalized)}
