"""Read config.yaml and build the rule set it describes.

Every setting of the experiment lives in one YAML file. ``load_config`` reads it and

1. checks that each section holds exactly the expected settings, so a misspelled
   setting is an error and not a silently ignored line;
2. loads FEATURES and RULES from the rules file;
3. builds the RuleSet with the configured prior;
4. stops with an error if two rules give the same answer on every object, because
   no sequence of examples could then tell them apart.

Paths in the config are relative to the folder that holds the config file.

    config = load_config("config.yaml")
    config.rules.prior()              # {rule name: prior probability}
    config.save_copy(output_path)     # put a copy of the YAML beside an output
"""

import runpy
from pathlib import Path

import yaml

from rule_learner import BooleanGrammar, RuleSet, all_stimuli

PRIOR_KINDS = ("uniform", "pcfg", "rational_rules")
ELICITATION_MODES = ("choice", "stated")
FOLLOWUPS = ("neutral", "leading", "pushback", "evidence")
BACKENDS = ("ollama", "openai")

# The settings each section must hold: no more and no fewer.
SECTIONS = {
    "config": ("seed", "rules_file", "alpha", "prior", "puzzles", "elicitation", "followups",
               "models", "run", "backends", "scores"),
    "prior": ("kind", "grammar_weights", "dirichlet", "temperature"),
    "puzzles": ("file", "n_confident", "n_contested", "lengths", "query_steps",
                "confident", "contested", "max_attempts", "nonce_words"),
    "puzzles.confident": ("leader_min",),
    "puzzles.contested": ("runner_up_min", "runner_up_max", "lead_min", "others_max"),
    "models[]": ("name", "backend", "think", "extra_body"),
    "run": ("results_file", "timeout_seconds", "choice", "stated"),
    "run.choice": ("max_tokens", "top_logprobs", "samples"),
    "run.stated": ("temperature", "max_tokens", "structured_output"),
    "backends": ("ollama", "openai"),
    "backends.ollama": ("base_url", "options", "fresh"),
    "backends.openai": ("base_url_env", "api_key_env"),
    "scores": ("bootstrap_samples", "interval"),
}

# Prior settings that the given kind of prior ignores. They must be null.
UNUSED_PRIOR_SETTINGS = {
    "uniform": ("grammar_weights", "dirichlet", "temperature"),
    "pcfg": ("dirichlet",),
    "rational_rules": ("grammar_weights",),
}


class Config:
    """The settings of one experiment and the rule set they describe.

    Attributes
    ----------
    path
        Where the YAML file is.
    text
        Its exact text as read, kept so a copy can be saved beside every output.
    settings
        The parsed YAML, a plain dict.
    seed, alpha, puzzles, run
        Shortcuts for ``settings["seed"]``, ``settings["alpha"]``, ``settings["puzzles"]`` and
        ``settings["run"]``.
    features
        ``{feature name: list of values}``, from the rules file.
    programs
        ``{rule name: source}``, from the rules file.
    stimuli
        Every object: all combinations of feature values.
    rules
        The RuleSet, carrying the configured prior.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.text = self.path.read_text()
        self.settings = yaml.safe_load(self.text)
        _check_settings(self.settings)
        self.seed = self.settings["seed"]
        self.alpha = self.settings["alpha"]
        self.puzzles = self.settings["puzzles"]
        self.run = self.settings["run"]

        rules_file = self.resolve(self.settings["rules_file"])
        self.features, self.programs = _read_rules_file(rules_file)
        self.stimuli = all_stimuli(self.features)
        weights = _prior_weights(self.settings["prior"], self.features, self.programs)
        self.rules = RuleSet.from_programs(self.programs, weights=weights)
        _require(2 <= len(self.rules) <= 26,
                 f"{rules_file} must define between 2 and 26 rules (options are lettered A to Z)")
        try:
            self.rules.check_distinct(self.stimuli)
        except ValueError as err:
            raise ValueError(
                f"config: {rules_file}: {err}. No examples could tell these rules apart; "
                "change or remove one rule in each group."
            ) from err

    def resolve(self, relative_path):
        """A path from the config, made relative to the folder that holds the config file."""
        return self.path.parent / relative_path

    def save_copy(self, output_path):
        """Save the config's text beside an output file, as ``<output name>.config.yaml``."""
        copy = Path(output_path).with_suffix(".config.yaml")
        copy.write_text(self.text)
        return copy


def load_config(path="config.yaml"):
    """Read, check and return the config at ``path``."""
    return Config(path)


def model_label(model):
    """The name a model entry goes by in results, tables and figures.

    It is the model's name, with the thinking setting added when thinking is on, so that
    the same model can be run with and without thinking.
    """
    if model["think"] in (None, False):
        return model["name"]
    return f"{model['name']} (thinking)" if model["think"] is True else f"{model['name']} (thinking: {model['think']})"


# ---------------------------------------------------------------------------
# Rules and prior
# ---------------------------------------------------------------------------

def _read_rules_file(path):
    """FEATURES and RULES from a Python file. The file is run, so it must be one you trust."""
    namespace = runpy.run_path(str(path))
    for name in ("FEATURES", "RULES"):
        _require(isinstance(namespace.get(name), dict) and namespace[name],
                 f"{path} must define a non-empty dict named {name}")
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
    _require(not missing and not unknown,
             f"in {name}, missing settings: {missing}; unknown settings: {unknown}")


def _check_names(values, allowed, name):
    _require(isinstance(values, list) and len(set(values)) == len(values)
             and all(value in allowed for value in values),
             f"{name} must be a list of distinct names from {list(allowed)}, got {values!r}")


def _check_settings(settings):
    """Raise ValueError with a readable message if any setting is missing, unknown or invalid."""
    _check_keys(settings, "config")
    _check_keys(settings["prior"], "prior")
    _check_keys(settings["puzzles"], "puzzles")
    _check_keys(settings["puzzles"]["confident"], "puzzles.confident")
    _check_keys(settings["puzzles"]["contested"], "puzzles.contested")

    _require(_is_int(settings["seed"]), "seed must be a whole number")
    _require(isinstance(settings["rules_file"], str), "rules_file must be a path")
    _require(_is_number(settings["alpha"]) and 0 <= settings["alpha"] <= 1,
             f"alpha must be a number between 0 and 1, got {settings['alpha']!r}")

    prior = settings["prior"]
    _require(prior["kind"] in PRIOR_KINDS,
             f"prior.kind must be one of {list(PRIOR_KINDS)}, got {prior['kind']!r}")
    for name in UNUSED_PRIOR_SETTINGS[prior["kind"]]:
        _require(prior[name] is None,
                 f"prior.{name} has no effect when prior.kind is {prior['kind']}; set it to null")
    _require(prior["grammar_weights"] is None or isinstance(prior["grammar_weights"], dict),
             "prior.grammar_weights must be null or a mapping such as {pred: 5.0}")
    for name in ("dirichlet", "temperature"):
        _require(prior[name] is None or (_is_number(prior[name]) and prior[name] > 0),
                 f"prior.{name} must be null or a positive number")

    puzzles = settings["puzzles"]
    lengths = puzzles["lengths"]
    _require(isinstance(puzzles["file"], str), "puzzles.file must be a path")
    _require(isinstance(lengths, list) and lengths and all(_is_int(n) and n >= 1 for n in lengths)
             and len(set(lengths)) == len(lengths),
             "puzzles.lengths must be a list of distinct whole numbers, 1 or more")
    for kind in ("confident", "contested"):
        count = puzzles[f"n_{kind}"]
        _require(_is_int(count) and count >= 0 and count % len(lengths) == 0,
                 f"puzzles.n_{kind} must be a multiple of the number of lengths ({len(lengths)}), "
                 "so that both kinds of puzzle have the same number at each length")
    _require(_is_int(puzzles["query_steps"]) and 1 <= puzzles["query_steps"] <= min(lengths),
             "puzzles.query_steps must be a whole number from 1 to the shortest length")
    _require(_is_int(puzzles["max_attempts"]) and puzzles["max_attempts"] >= 1,
             "puzzles.max_attempts must be a whole number, 1 or more")

    leader_min = puzzles["confident"]["leader_min"]
    contested = puzzles["contested"]
    low, high = contested["runner_up_min"], contested["runner_up_max"]
    lead_min, others_max = contested["lead_min"], contested["others_max"]
    _require(all(_is_number(value) and 0 <= value <= 1 for value in (leader_min, low, high, lead_min, others_max))
             and low <= high,
             "the puzzle thresholds must be probabilities, with runner_up_min <= runner_up_max")
    _require(leader_min + low > 1,
             "confident.leader_min + contested.runner_up_min must be more than 1, "
             "so that no sequence can be both confident and contested")
    _require(others_max < low,
             "contested.others_max must be less than contested.runner_up_min, "
             "so that the runner-up is the only strong alternative")

    words = puzzles["nonce_words"]
    _require(isinstance(words, list) and words and len(set(words)) == len(words)
             and all(isinstance(word, str) and word for word in words),
             "puzzles.nonce_words must be a list of distinct words")

    _check_names(settings["elicitation"], ELICITATION_MODES, "elicitation")
    _require(settings["elicitation"], "elicitation must name at least one mode")
    _check_names(settings["followups"], FOLLOWUPS, "followups")

    _require(isinstance(settings["models"], list), "models must be a list")
    for model in settings["models"]:
        _check_keys(model, "models[]")
        _require(isinstance(model["name"], str) and model["name"], "every model needs a name")
        _require(model["backend"] in BACKENDS,
                 f"model {model['name']}: backend must be one of {list(BACKENDS)}, got {model['backend']!r}")
        _require(model["think"] is None or isinstance(model["think"], (bool, str)),
                 f"model {model['name']}: think must be true, false, null or the name of a thinking level")
        _require(model["think"] is None or model["backend"] == "ollama",
                 f"model {model['name']}: think is an Ollama setting; with the openai backend set it to null "
                 "and put the provider's own field in extra_body")
        _require(model["extra_body"] is None or isinstance(model["extra_body"], dict),
                 f"model {model['name']}: extra_body must be null or a mapping")
    labels = [model_label(model) for model in settings["models"]]
    _require(len(set(labels)) == len(labels), f"two models would share a name in the results: {labels}")

    _check_keys(settings["run"], "run")
    run = settings["run"]
    _require(isinstance(run["results_file"], str), "run.results_file must be a path")
    _require(_is_number(run["timeout_seconds"]) and run["timeout_seconds"] > 0,
             "run.timeout_seconds must be a positive number")
    _check_keys(run["choice"], "run.choice")
    _check_keys(run["stated"], "run.stated")
    choice, stated = run["choice"], run["stated"]
    _require(_is_int(choice["top_logprobs"]) and 0 <= choice["top_logprobs"] <= 20,
             "run.choice.top_logprobs must be a whole number from 0 to 20")
    _require(_is_int(choice["samples"]) and choice["samples"] >= 1,
             "run.choice.samples must be a whole number, 1 or more")
    _require(_is_number(stated["temperature"]) and stated["temperature"] >= 0,
             "run.stated.temperature must be 0 or more")
    _require(isinstance(stated["structured_output"], bool), "run.stated.structured_output must be true or false")
    for mode in ("choice", "stated"):
        _require(_is_int(run[mode]["max_tokens"]) and run[mode]["max_tokens"] >= 1,
                 f"run.{mode}.max_tokens must be a whole number, 1 or more")
    if "choice" in settings["elicitation"] and choice["top_logprobs"]:
        for model in settings["models"]:
            _require(model["backend"] != "ollama" or model["think"] is False,
                     f"model {model['name']}: choice mode reads the first token of the answer, and with thinking on "
                     "(or left to the model's default) the first tokens are the model's thinking. Set think to "
                     "false, or set run.choice.top_logprobs to 0 to sample answers instead")

    _check_keys(settings["backends"], "backends")
    _check_keys(settings["backends"]["ollama"], "backends.ollama")
    _check_keys(settings["backends"]["openai"], "backends.openai")
    ollama = settings["backends"]["ollama"]
    _require(isinstance(ollama["base_url"], str), "backends.ollama.base_url must be a URL")
    _require(ollama["options"] is None or isinstance(ollama["options"], dict),
             "backends.ollama.options must be null or a mapping")
    _require(isinstance(ollama["fresh"], bool), "backends.ollama.fresh must be true or false")
    for name in ("base_url_env", "api_key_env"):
        _require(isinstance(settings["backends"]["openai"][name], str),
                 f"backends.openai.{name} must be the name of an environment variable")

    _check_keys(settings["scores"], "scores")
    scores = settings["scores"]
    _require(_is_int(scores["bootstrap_samples"]) and scores["bootstrap_samples"] >= 1,
             "scores.bootstrap_samples must be a whole number, 1 or more")
    _require(_is_number(scores["interval"]) and 0 < scores["interval"] < 1,
             "scores.interval must be between 0 and 1")
