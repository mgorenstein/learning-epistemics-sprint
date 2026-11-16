"""Read config_bcl.yaml, classify rule complexity, and build the rule sets.

Every setting of the interactive Bayesian concept-learning experiment lives in one YAML
file. ``load_config`` reads it and

1. checks that each section holds exactly the expected settings, so a misspelled or
   missing setting is an error rather than a silently ignored line;
2. loads FEATURES and RULES from the rules file;
3. builds the configured RuleSet alongside a uniform-prior RuleSet for comparison;
4. classifies every rule's Boolean complexity from its AST structure;
5. stops with an error if two rules give the same answer on every object.

Paths in the config are relative to the folder that holds the config file.
"""

from __future__ import annotations

import ast
import math
import runpy
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

from rule_learner import BooleanGrammar, Rule, RuleSet, all_stimuli

PRIOR_KINDS = ("uniform", "pcfg", "rational_rules")
HYPOTHESIS_SPACES = ("rules_file", "grammar")
MAX_GRAMMAR_BOOL_NODES = 7
BACKENDS = ("ollama", "openai")
RUN_MODES = ("running_log_rule", "running_log_yesno", "multiturn_yesno")
COMPLEXITY_GROUPS = ("single", "negation", "conjunction", "disjunction", "compound")

# The settings each section must hold: no more and no fewer.
SECTIONS = {
    "config": ("seed", "rules_file", "alpha", "prior", "blocks", "models", "run", "backends", "scores"),
    "prior": ("kind", "hypothesis_space", "max_bool_nodes", "grammar_weights", "dirichlet", "temperature"),
    "blocks": ("file", "blocks_per_concept", "repetitions", "nonce_words"),
    "models[]": ("name", "backend", "think", "extra_body"),
    "run": ("mode", "results_file", "timeout_seconds", "temperature", "max_tokens", "top_logprobs"),
    "backends": ("ollama", "openai"),
    "backends.ollama": ("base_url", "options", "reuse_cache"),
    "backends.openai": ("base_url_env", "api_key_env"),
    "scores": ("bootstrap_samples", "interval"),
}

# Prior settings that the given kind of prior ignores. They must be null.
UNUSED_PRIOR_SETTINGS = {
    "uniform": ("grammar_weights", "dirichlet", "temperature"),
    "pcfg": ("dirichlet",),
    "rational_rules": ("grammar_weights",),
}

_DEFAULT_RULES_CACHE: dict[str, str] | None = None


class Config:
    """The settings of one interactive concept-learning experiment and its rule sets."""

    def __init__(self, path):
        self.path = Path(path)
        self.text = self.path.read_text()
        self.settings = yaml.safe_load(self.text)
        _check_settings(self.settings)
        self.seed = self.settings["seed"]
        self.alpha = self.settings["alpha"]
        self.blocks = self.settings["blocks"]
        self.run = self.settings["run"]
        self.scores = self.settings["scores"]

        rules_file = self.resolve(self.settings["rules_file"])
        self.features, self.programs = _read_rules_file(rules_file)
        self.stimuli = all_stimuli(self.features)
        weights = _prior_weights(self.settings["prior"], self.features, self.programs)
        self.rules = RuleSet.from_programs(self.programs, weights=weights)
        _require(len(self.rules) >= 2, f"{rules_file} must define at least 2 rules")
        try:
            self.rules.check_distinct(self.stimuli)
        except ValueError as err:
            raise ValueError(
                f"config: {rules_file}: {err}. No examples could tell these rules apart; "
                "change or remove one rule in each group."
            ) from err
        self.complexity_groups = {
            name: complexity_group(name, self.programs) for name in self.rules.names
        }
        if self.settings["prior"]["hypothesis_space"] == "grammar":
            (
                self.hypothesis_rules,
                self.uniform_rules,
                self.n_grammar_derivations,
                self.target_concept_priors,
            ) = _enumerate_grammar_rulesets(
                self.settings["prior"], self.features, self.stimuli, self.rules
            )
        else:
            self.hypothesis_rules = self.rules
            self.uniform_rules = RuleSet.from_programs(self.programs)
            self.n_grammar_derivations = len(self.rules)
            self.target_concept_priors = self.rules.prior()

    def resolve(self, relative_path):
        """A path from the config, made relative to the folder that holds the config file."""
        return self.path.parent / relative_path

    def save_copy(self, output_path):
        """Save the config's text beside an output file, as ``<output name>.config.yaml``."""
        copy = Path(output_path).with_suffix(".config.yaml")
        copy.write_text(self.text)
        return copy


def load_config(path="config_bcl.yaml"):
    """Read, check and return the config at ``path``."""
    return Config(path)


def model_label(model):
    """The name a model entry goes by in results, tables and figures."""
    if model["think"] in (None, False):
        return model["name"]
    if model["think"] is True:
        return f"{model['name']} (thinking)"
    return f"{model['name']} (thinking: {model['think']})"


# ---------------------------------------------------------------------------
# Boolean complexity classification
# ---------------------------------------------------------------------------

def complexity_group(rule_name, programs=None):
    """Classify a Boolean rule into one of the five complexity groups by AST structure.

    Accepts a rule name (looked up in ``programs`` or ``example_rules.RULES``), a ``Rule``
    instance, or a Python ``lambda`` source string directly:

    - bare ``==`` -> ``"single"``
    - bare ``!=`` or ``not (<bare ==>)`` -> ``"negation"``
    - ``and`` without ``!=``/``not`` (and without ``or``) -> ``"conjunction"``
    - ``or`` without ``!=``/``not`` (and without ``and``) -> ``"disjunction"``
    - otherwise -> ``"compound"``
    """
    source = _resolve_rule_source(rule_name, programs)
    body, arg = _parse_lambda_body(source)
    _validate_boolean_ast(body, arg, source)

    if _is_atomic_eq(body):
        return "single"
    if _is_atomic_neq(body) or (
        isinstance(body, ast.UnaryOp)
        and isinstance(body.op, ast.Not)
        and _is_atomic_eq(body.operand)
    ):
        return "negation"

    has_and = False
    has_or = False
    has_neg = False
    for node in ast.walk(body):
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                has_and = True
            elif isinstance(node.op, ast.Or):
                has_or = True
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            has_neg = True
        elif _is_atomic_neq(node):
            has_neg = True

    if has_and and not has_or and not has_neg:
        return "conjunction"
    if has_or and not has_and and not has_neg:
        return "disjunction"
    return "compound"


def _resolve_rule_source(rule_or_source, programs=None):
    if isinstance(rule_or_source, Rule):
        if rule_or_source.source is None:
            raise ValueError(f"rule {rule_or_source.name!r} has no source string to classify")
        return rule_or_source.source
    if not isinstance(rule_or_source, str) or not rule_or_source.strip():
        raise ValueError(f"expected a rule name or lambda source string, got {rule_or_source!r}")
    if programs is not None and rule_or_source in programs:
        return programs[rule_or_source]
    stripped = textwrap.dedent(rule_or_source).strip()
    if stripped.startswith("lambda ") or stripped.startswith("lambda\t"):
        return stripped
    if programs is not None:
        raise KeyError(f"unknown rule {rule_or_source!r}")
    default_rules = _default_rules()
    if rule_or_source in default_rules:
        return default_rules[rule_or_source]
    raise KeyError(f"unknown rule {rule_or_source!r}")


def _default_rules():
    global _DEFAULT_RULES_CACHE
    if _DEFAULT_RULES_CACHE is None:
        default_path = Path(__file__).resolve().parent.parent / "example_rules.py"
        _, rules = _read_rules_file(default_path)
        _DEFAULT_RULES_CACHE = dict(rules)
    return _DEFAULT_RULES_CACHE


def _parse_lambda_body(source):
    src = textwrap.dedent(source).strip()
    try:
        tree = ast.parse(src, mode="eval").body
    except SyntaxError as err:
        raise ValueError(f"cannot parse rule source {source!r}: {err}") from err
    if not isinstance(tree, ast.Lambda) or len(tree.args.args) != 1:
        raise ValueError(
            "expected a one-argument lambda such as \"lambda t: t['color'] == 'red'\"; "
            f"got {source!r}"
        )
    return tree.body, tree.args.args[0].arg


def _validate_boolean_ast(node, arg, source):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        for value in node.values:
            _validate_boolean_ast(value, arg, source)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _validate_boolean_ast(node.operand, arg, source)
        return
    if _is_feature_compare(node, arg):
        return
    raise ValueError(
        f"cannot classify Boolean complexity of {source!r} at {ast.unparse(node)!r}: "
        "use only and, or, not, == and != on t[feature]"
    )


def _is_feature_compare(node, arg):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq))):
        return False
    left, right = node.left, node.comparators[0]
    for lookup, constant in ((left, right), (right, left)):
        if (
            isinstance(lookup, ast.Subscript)
            and isinstance(lookup.value, ast.Name)
            and lookup.value.id == arg
            and isinstance(lookup.slice, ast.Constant)
            and isinstance(constant, ast.Constant)
        ):
            return True
    return False


def _is_atomic_eq(node):
    return isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)


def _is_atomic_neq(node):
    return isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)


# ---------------------------------------------------------------------------
# Rules and prior
# ---------------------------------------------------------------------------

def _read_rules_file(path):
    """FEATURES and RULES from a Python file. The file is run, so it must be one you trust."""
    namespace = runpy.run_path(str(path))
    for name in ("FEATURES", "RULES"):
        _require(
            isinstance(namespace.get(name), dict) and namespace[name],
            f"{path} must define a non-empty dict named {name}",
        )
    return namespace["FEATURES"], namespace["RULES"]


def _prior_weights(prior, features, programs):
    """Unnormalized prior weight of each rule, or None for equal weights."""
    if prior["kind"] == "uniform":
        return None
    grammar = BooleanGrammar(features, weights=prior["grammar_weights"])
    return grammar.prior_weights(
        programs,
        kind=prior["kind"],
        dirichlet=1.0 if prior["dirichlet"] is None else prior["dirichlet"],
        temperature=1.0 if prior["temperature"] is None else prior["temperature"],
    )


def _rr_dirichlet_log_term(counts: tuple[int, ...], dirichlet: float) -> float:
    """Dirichlet-multinomial log probability for one nonterminal's production counts."""
    k = len(counts)
    return (
        sum(math.lgamma(c + dirichlet) for c in counts)
        - math.lgamma(sum(counts) + k * dirichlet)
        - (k * math.lgamma(dirichlet) - math.lgamma(k * dirichlet))
    )


def enumerate_grammar_derivations(features: dict, stimuli: list[dict], max_bool_nodes: int):
    """Enumerate all syntactic BooleanGrammar trees with ``1..max_bool_nodes`` BOOL steps.

    Returns ``list[(expr_str, mask_int, bool_counts, pred_counts, val_counts)]`` where
    ``mask_int`` encodes the tree's truth table across ``stimuli``.
    """
    feat_names = list(features.keys())
    n_feats = len(feat_names)
    full_mask = (1 << len(stimuli)) - 1
    by_size: dict[int, list[tuple[str, int, tuple[int, int, int, int], tuple[int, ...], tuple[tuple[int, ...], ...]]]] = (
        defaultdict(list)
    )

    for f_idx, feat in enumerate(feat_names):
        vals = list(features[feat])
        n_vals = len(vals)
        p_c = tuple(1 if i == f_idx else 0 for i in range(n_feats))
        for v_idx, val in enumerate(vals):
            expr = f"t[{feat!r}] == {val!r}"
            mask = 0
            for s_idx, stim in enumerate(stimuli):
                if stim[feat] == val:
                    mask |= 1 << s_idx
            v_c = tuple(
                (tuple(1 if j == v_idx else 0 for j in range(n_vals)) if i == f_idx else (0,) * len(features[feat_names[i]]))
                for i in range(n_feats)
            )
            by_size[1].append((expr, mask, (1, 0, 0, 0), p_c, v_c))

    for size in range(2, max_bool_nodes + 1):
        for sub_expr, sub_mask, (b0, b1, b2, b3), p_c, v_c in by_size[size - 1]:
            by_size[size].append((f"(not {sub_expr})", full_mask ^ sub_mask, (b0, b1, b2, b3 + 1), p_c, v_c))
        for left_sz in range(1, size - 1):
            right_sz = size - 1 - left_sz
            for l_expr, l_mask, lb, lp, lv in by_size[left_sz]:
                for r_expr, r_mask, rb, rp, rv in by_size[right_sz]:
                    p_sum = tuple(a + b for a, b in zip(lp, rp))
                    v_sum = tuple(tuple(a + b for a, b in zip(va, vb)) for va, vb in zip(lv, rv))
                    by_size[size].append((
                        f"({l_expr} and {r_expr})",
                        l_mask & r_mask,
                        (lb[0] + rb[0], lb[1] + rb[1] + 1, lb[2] + rb[2], lb[3] + rb[3]),
                        p_sum,
                        v_sum,
                    ))
                    by_size[size].append((
                        f"({l_expr} or {r_expr})",
                        l_mask | r_mask,
                        (lb[0] + rb[0], lb[1] + rb[1], lb[2] + rb[2] + 1, lb[3] + rb[3]),
                        p_sum,
                        v_sum,
                    ))

    derivations = []
    for sz in range(1, max_bool_nodes + 1):
        derivations.extend(by_size[sz])
    return derivations


def _enumerate_grammar_rulesets(prior: dict, features: dict, stimuli: list[dict], target_rules: RuleSet):
    """Build exact extension-collapsed RuleSets over all BooleanGrammar trees up to ``max_bool_nodes``."""
    max_nodes = int(prior["max_bool_nodes"])
    derivations = enumerate_grammar_derivations(features, stimuli, max_nodes)
    feat_keys = tuple(features.keys())
    stim_tuples = [tuple(s[k] for k in feat_keys) for s in stimuli]

    kind = prior["kind"]
    temp = 1.0 if prior["temperature"] is None else float(prior["temperature"])
    dirichlet = 1.0 if prior["dirichlet"] is None else float(prior["dirichlet"])

    if kind == "pcfg":
        grammar = BooleanGrammar(features, weights=prior["grammar_weights"])
        tot_w = sum(grammar.weights.values())
        bool_lps = tuple(math.log(grammar.weights[p] / tot_w) for p in grammar.BOOL_PRODUCTIONS)
        lp_pred_feat = -math.log(len(features))
        lp_vals = tuple(-math.log(len(features[f])) for f in feat_keys)

    ext_log_terms: dict[int, list[float]] = defaultdict(list)
    seen_masks_order: list[int] = []
    for _, mask, b_c, p_c, v_c in derivations:
        if mask not in ext_log_terms:
            seen_masks_order.append(mask)
        if kind == "uniform":
            ext_log_terms[mask].append(0.0)
        elif kind == "pcfg":
            lp = (
                b_c[0] * bool_lps[0]
                + b_c[1] * bool_lps[1]
                + b_c[2] * bool_lps[2]
                + b_c[3] * bool_lps[3]
                + sum(p_c) * lp_pred_feat
                + sum(sum(vc) * lpv for vc, lpv in zip(v_c, lp_vals))
            ) / temp
            ext_log_terms[mask].append(lp)
        elif kind == "rational_rules":
            lp = (
                _rr_dirichlet_log_term(b_c, dirichlet)
                + _rr_dirichlet_log_term(p_c, dirichlet)
                + sum(_rr_dirichlet_log_term(vc, dirichlet) for vc in v_c)
            ) / temp
            ext_log_terms[mask].append(lp)

    # Check every target concept in target_rules is expressible within max_bool_nodes
    target_mask_of: dict[str, int] = {}
    name_for_mask: dict[int, str] = {}
    for t_name in target_rules.names:
        ext = target_rules.extension(t_name, stimuli)
        t_mask = sum((1 << idx) for idx, val in enumerate(ext) if val)
        _require(
            t_mask in ext_log_terms,
            f"target rule {t_name!r} is not expressible within prior.max_bool_nodes={max_nodes}; "
            "increase prior.max_bool_nodes",
        )
        target_mask_of[t_name] = t_mask
        name_for_mask[t_mask] = t_name

    ordered_masks = [target_mask_of[name] for name in target_rules.names]
    target_mask_set = set(ordered_masks)
    extra_idx = 0
    for mask in seen_masks_order:
        if mask not in target_mask_set:
            extra_idx += 1
            name_for_mask[mask] = f"grammar_concept_{extra_idx:04d}"
            ordered_masks.append(mask)

    ext_log_prior: dict[int, float] = {}
    for mask in ordered_masks:
        if kind == "uniform":
            ext_log_prior[mask] = 0.0
        else:
            lps = ext_log_terms[mask]
            top_m = max(lps)
            ext_log_prior[mask] = top_m + math.log(sum(math.exp(x - top_m) for x in lps))

    top_ext = max(ext_log_prior.values())
    weighted_rules = []
    uniform_rules = []
    for mask in ordered_masks:
        r_name = name_for_mask[mask]
        yes_set = frozenset(stim_tuples[i] for i in range(len(stimuli)) if (mask >> i) & 1)
        fn = lambda trial, yes=yes_set, keys=feat_keys: tuple(trial[k] for k in keys) in yes  # noqa: E731
        weighted_rules.append(Rule(r_name, fn, weight=math.exp(ext_log_prior[mask] - top_ext)))
        uniform_rules.append(Rule(r_name, fn, weight=1.0))

    hyp_ruleset = RuleSet(weighted_rules)
    uni_ruleset = RuleSet(uniform_rules)
    hyp_prior = hyp_ruleset.prior()
    target_priors = {t_name: hyp_prior[t_name] for t_name in target_rules.names}
    return hyp_ruleset, uni_ruleset, len(derivations), target_priors


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise ValueError(f"config: {message}")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_keys(section, name):
    """Raise ValueError unless ``section`` holds exactly the settings listed in SECTIONS."""
    _require(isinstance(section, dict), f"{name} must be a section of settings")
    expected = SECTIONS[name]
    missing = [key for key in expected if key not in section]
    unknown = [key for key in section if key not in expected]
    _require(
        not missing and not unknown,
        f"in {name}, missing settings: {missing}; unknown settings: {unknown}",
    )


def _check_settings(settings):
    """Raise ValueError with a readable message if any setting is missing, unknown or invalid."""
    _check_keys(settings, "config")
    _check_keys(settings["prior"], "prior")
    _check_keys(settings["blocks"], "blocks")

    _require(_is_int(settings["seed"]), "seed must be a whole number")
    _require(isinstance(settings["rules_file"], str) and settings["rules_file"], "rules_file must be a path")
    _require(
        _is_number(settings["alpha"]) and 0 <= settings["alpha"] <= 1,
        f"alpha must be a number between 0 and 1, got {settings['alpha']!r}",
    )

    prior = settings["prior"]
    _require(
        prior["kind"] in PRIOR_KINDS,
        f"prior.kind must be one of {list(PRIOR_KINDS)}, got {prior['kind']!r}",
    )
    _require(
        prior["hypothesis_space"] in HYPOTHESIS_SPACES,
        f"prior.hypothesis_space must be one of {list(HYPOTHESIS_SPACES)}, got {prior['hypothesis_space']!r}",
    )
    if prior["hypothesis_space"] == "rules_file":
        _require(
            prior["max_bool_nodes"] is None,
            "prior.max_bool_nodes has no effect when prior.hypothesis_space is rules_file; set it to null",
        )
    else:
        _require(
            _is_int(prior["max_bool_nodes"]) and 1 <= prior["max_bool_nodes"] <= MAX_GRAMMAR_BOOL_NODES,
            f"prior.max_bool_nodes must be a whole number between 1 and {MAX_GRAMMAR_BOOL_NODES} when "
            "prior.hypothesis_space is grammar",
        )
    for name in UNUSED_PRIOR_SETTINGS[prior["kind"]]:
        _require(
            prior[name] is None,
            f"prior.{name} has no effect when prior.kind is {prior['kind']}; set it to null",
        )
    _require(
        prior["grammar_weights"] is None or isinstance(prior["grammar_weights"], dict),
        "prior.grammar_weights must be null or a mapping such as {pred: 5.0}",
    )
    for name in ("dirichlet", "temperature"):
        _require(
            prior[name] is None or (_is_number(prior[name]) and prior[name] > 0),
            f"prior.{name} must be null or a positive number",
        )

    blocks = settings["blocks"]
    _require(isinstance(blocks["file"], str) and blocks["file"], "blocks.file must be a path")
    _require(
        _is_int(blocks["blocks_per_concept"]) and blocks["blocks_per_concept"] >= 1,
        "blocks.blocks_per_concept must be a whole number, 1 or more",
    )
    _require(
        _is_int(blocks["repetitions"]) and blocks["repetitions"] >= 1,
        "blocks.repetitions must be a whole number, 1 or more",
    )
    words = blocks["nonce_words"]
    _require(
        isinstance(words, list) and words and len(set(words)) == len(words)
        and all(isinstance(word, str) and word for word in words),
        "blocks.nonce_words must be a list of distinct words",
    )

    _require(isinstance(settings["models"], list) and settings["models"], "models must be a non-empty list")
    for model in settings["models"]:
        _check_keys(model, "models[]")
        _require(isinstance(model["name"], str) and model["name"], "every model needs a name")
        _require(
            model["backend"] in BACKENDS,
            f"model {model['name']}: backend must be one of {list(BACKENDS)}, got {model['backend']!r}",
        )
        _require(
            model["think"] is None or isinstance(model["think"], (bool, str)),
            f"model {model['name']}: think must be true, false, null or the name of a thinking level",
        )
        _require(
            model["think"] is None or model["backend"] == "ollama",
            f"model {model['name']}: think is an Ollama setting; with the openai backend set it to null "
            "and put the provider's own field in extra_body",
        )
        _require(
            model["extra_body"] is None or isinstance(model["extra_body"], dict),
            f"model {model['name']}: extra_body must be null or a mapping",
        )
    labels = [model_label(model) for model in settings["models"]]
    _require(len(set(labels)) == len(labels), f"two models would share a name in the results: {labels}")

    _check_keys(settings["run"], "run")
    run = settings["run"]
    _require(
        run["mode"] in RUN_MODES,
        f"run.mode must be one of {list(RUN_MODES)}, got {run['mode']!r}",
    )
    _require(isinstance(run["results_file"], str) and run["results_file"], "run.results_file must be a path")
    _require(
        _is_number(run["timeout_seconds"]) and run["timeout_seconds"] > 0,
        "run.timeout_seconds must be a positive number",
    )
    _require(
        _is_number(run["temperature"]) and run["temperature"] >= 0,
        "run.temperature must be 0 or more",
    )
    _require(
        _is_int(run["max_tokens"]) and run["max_tokens"] >= 1,
        "run.max_tokens must be a whole number, 1 or more",
    )
    _require(
        _is_int(run["top_logprobs"]) and 0 <= run["top_logprobs"] <= 20,
        "run.top_logprobs must be a whole number from 0 to 20",
    )
    if run["top_logprobs"]:
        for model in settings["models"]:
            _require(
                model["backend"] != "ollama" or model["think"] is False,
                f"model {model['name']}: reading yes/no probabilities from first-token logprobs needs "
                "thinking off, because a thinking model's first tokens are its thinking. Set think to "
                "false, or set run.top_logprobs to 0",
            )

    _check_keys(settings["backends"], "backends")
    _check_keys(settings["backends"]["ollama"], "backends.ollama")
    _check_keys(settings["backends"]["openai"], "backends.openai")
    ollama = settings["backends"]["ollama"]
    _require(isinstance(ollama["base_url"], str) and ollama["base_url"], "backends.ollama.base_url must be a URL")
    _require(
        ollama["options"] is None or isinstance(ollama["options"], dict),
        "backends.ollama.options must be null or a mapping",
    )
    _require(isinstance(ollama["reuse_cache"], bool), "backends.ollama.reuse_cache must be true or false")
    for name in ("base_url_env", "api_key_env"):
        _require(
            isinstance(settings["backends"]["openai"][name], str) and settings["backends"]["openai"][name],
            f"backends.openai.{name} must be the name of an environment variable",
        )

    _check_keys(settings["scores"], "scores")
    scores = settings["scores"]
    _require(
        _is_int(scores["bootstrap_samples"]) and scores["bootstrap_samples"] >= 1,
        "scores.bootstrap_samples must be a whole number, 1 or more",
    )
    _require(
        _is_number(scores["interval"]) and 0 < scores["interval"] < 1,
        "scores.interval must be between 0 and 1",
    )
