"""Turn a concept-learning block into prompts and read rule guesses, yes/no answers, and logprobs.

All prompt wording lives in ``bcl/prompts/``:

trial_rule.txt   running-log prompt (trials 1..N) eliciting ``Rule: ...`` + ``Answer: yes|no``
trial_yesno.txt  running-log prompt (trials 1..N) eliciting only ``yes`` or ``no``
start.txt        multi-turn trial 1: introduces the task and object 1
step.txt         multi-turn trials 2..N: gives feedback on trial t-1 and presents object t

Unlike ``hh/``, the candidate Python programs are never shown to the model; the learner
discovers the concept solely through guesses and trial-by-trial feedback.
"""

from __future__ import annotations

import argparse
import math
import re
import string
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def template(name: str) -> string.Template:
    """The template in ``bcl/prompts/<name>.txt``."""
    return string.Template((PROMPTS / f"{name}.txt").read_text().rstrip("\n"))


def format_object(stimulus: dict, features: dict) -> str:
    """Deterministic Python dict representation of one stimulus in feature order."""
    ordered = {name: stimulus[name] for name in features}
    return repr(ordered)


def describe_features(features: dict) -> str:
    """Human-readable description of dict feature keys and allowed values."""
    parts = [
        f"{name!r} (one of {', '.join(repr(value) for value in values)})"
        for name, values in features.items()
    ]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_feature_space(features: dict) -> str:
    """Natural-English summary of feature dimensions and values (e.g. 'a color (...) and a shape (...)')."""
    parts = [
        f"a {name} ({', '.join(str(value) for value in values)})"
        for name, values in features.items()
    ]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_feature_names(features: dict) -> str:
    """Natural-English list of feature names joined with 'and/or' (e.g. 'color and/or shape')."""
    names = list(features.keys())
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and/or {names[1]}"
    return ", ".join(names[:-1]) + f", and/or {names[-1]}"


def _values_are_disjoint_strings(features: dict) -> bool:
    seen: set[str] = set()
    for values in features.values():
        for val in values:
            if not isinstance(val, str) or val in seen:
                return False
            seen.add(val)
    return True


def stimulus_phrase(stimulus: dict, features: dict) -> str:
    """Compact natural phrase for one stimulus (e.g. 'yellow triangle')."""
    if _values_are_disjoint_strings(features):
        return " ".join(str(stimulus[name]) for name in features)
    return ", ".join(f"{name}={stimulus[name]}" for name in features)


def format_history_block(
    history: list[dict],
    word: str,
    features: dict,
    *,
    include_rule: bool,
) -> str:
    """Render accumulated trial-and-feedback lines prior to the current trial."""
    if not history:
        return "Trials so far: (none yet — make an initial yes or no guess)"
    lines = ["Trials and feedback so far:"]
    for h in history:
        phrase = stimulus_phrase(h["object"], features)
        verdict = "Correct" if (h["answer"] is h["truth"]) else "Incorrect"
        fact = f"IS a {word}" if h["truth"] else f"is NOT a {word}"
        guess_str = (
            "yes" if h["answer"] is True else ("no" if h["answer"] is False else "unreadable")
        )
        rule_guess = (h.get("rule_guess") or "").strip()
        if include_rule and rule_guess:
            lines.append(
                f'  {h["trial"]}. {phrase} — rule guess: "{rule_guess}", '
                f"answer: {guess_str} -> {verdict} ({phrase} {fact})"
            )
        else:
            lines.append(
                f"  {h['trial']}. {phrase} — your guess: {guess_str} -> {verdict} ({phrase} {fact})"
            )
    return "\n".join(lines)


def trial_message(
    block: dict,
    trial_number: int,
    history: list[dict],
    config,
    *,
    mode: str | None = None,
) -> str:
    """Single-turn running-log user message for 1-indexed ``trial_number`` (1..n_trials)."""
    n_trials = int(block["n_trials"])
    if not 1 <= trial_number <= n_trials:
        raise ValueError(f"trial_number must be between 1 and {n_trials}, got {trial_number}")
    if len(history) != trial_number - 1:
        raise ValueError(
            f"trial {trial_number} expects {trial_number - 1} history entries, got {len(history)}"
        )
    active_mode = mode if mode is not None else config.run["mode"]
    if active_mode == "running_log_rule":
        tpl_name = "trial_rule"
        include_rule = True
    elif active_mode == "running_log_yesno":
        tpl_name = "trial_yesno"
        include_rule = False
    else:
        raise ValueError(f"trial_message requires a running_log_* mode, got {active_mode!r}")

    repetitions = int(block["repetitions"])
    pass_clause = (
        ", each shown once"
        if repetitions == 1
        else f" (shown across {repetitions} shuffled passes)"
    )
    current = block["trials"][trial_number - 1]
    return template(tpl_name).substitute(
        word=block["word"],
        n_trials=n_trials,
        n_objects=len(config.stimuli),
        feature_space=describe_feature_space(config.features),
        pass_clause=pass_clause,
        feature_names=describe_feature_names(config.features),
        history_block=format_history_block(
            history, block["word"], config.features, include_rule=include_rule
        ),
        trial=trial_number,
        object=stimulus_phrase(current["object"], config.features),
    )


def start_message(block: dict, config) -> str:
    """Multi-turn user message for trial 1 of ``block``."""
    first_trial = block["trials"][0]
    values = {
        "word": block["word"],
        "n_trials": block["n_trials"],
        "n_objects": len(config.stimuli),
        "feature_description": describe_features(config.features),
        "object": format_object(first_trial["object"], config.features),
    }
    for name, feature_values in config.features.items():
        values[f"{name}_values"] = ", ".join(repr(value) for value in feature_values)
    return template("start").substitute(values)


def step_message(
    block: dict,
    trial_number: int,
    prev_answer: bool | None,
    prev_truth: bool,
    config,
) -> str:
    """Multi-turn user message for 1-indexed ``trial_number`` (2..n_trials) with feedback on t-1."""
    n_trials = block["n_trials"]
    if not 2 <= trial_number <= n_trials:
        raise ValueError(f"trial_number must be between 2 and {n_trials}, got {trial_number}")
    current = block["trials"][trial_number - 1]
    feedback = "Correct." if (prev_answer is prev_truth) else "Incorrect."
    truth_verb = "is" if prev_truth else "is not"
    return template("step").substitute(
        feedback=feedback,
        truth_verb=truth_verb,
        word=block["word"],
        trial=trial_number,
        n_trials=n_trials,
        object=format_object(current["object"], config.features),
    )


# ---------------------------------------------------------------------------
# Answers, rule guesses, and token probabilities
# ---------------------------------------------------------------------------

def parse_answer(text: str) -> tuple[bool | None, str | None]:
    """Parse a yes/no response into ``(True | False | None, error | None)``."""
    stripped = (text or "").strip()
    if not stripped:
        return None, "the answer is empty"

    exact = re.fullmatch(r"\W*(?:answer\W+)?(yes|no)\W*", stripped, flags=re.IGNORECASE)
    if exact:
        return exact.group(1).lower() == "yes", None

    leading = re.match(r"^\W*(?:answer\W+)?(yes|no)\b", stripped, flags=re.IGNORECASE)
    if leading:
        found = {w.lower() for w in re.findall(r"\b(yes|no)\b", stripped, flags=re.IGNORECASE)}
        if len(found) == 1:
            return leading.group(1).lower() == "yes", None

    return None, f"not yes or no: {stripped[:80]!r}"


def parse_rule_and_answer(text: str) -> tuple[bool | None, str | None, str | None]:
    """Parse a two-line ``Rule: ... / Answer: yes|no`` reply into ``(answer, rule_guess, error)``."""
    raw = text or ""
    rule_guess: str | None = None
    m_rule = re.search(
        r"^\s*\*{0,2}Rule\*{0,2}\s*:\s*(.+?)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m_rule:
        cleaned = re.sub(r"\s+", " ", m_rule.group(1)).strip().strip("*").strip()
        if cleaned:
            rule_guess = cleaned

    m_ans = re.search(
        r"^\s*\*{0,2}Answer\*{0,2}\s*:\s*(.+?)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m_ans:
        ans, err = parse_answer(m_ans.group(1))
        return ans, rule_guess, err

    ans, err = parse_answer(raw)
    return ans, rule_guess, err


def answer_token(logprobs) -> dict | None:
    """The token-probability entry where the model's yes/no answer starts, or None.

    When the response contains an ``Answer:`` prefix (as in ``Rule: ...\\nAnswer: yes``),
    returns the first non-empty token following ``Answer:`` so words on the ``Rule:`` line
    can never shadow the decision token. Otherwise returns the first non-whitespace token.
    """
    entries = logprobs or []
    accum = ""
    after_answer_prefix = False
    for entry in entries:
        tok = entry["token"]
        if after_answer_prefix:
            if tok.strip().strip("*"):
                return entry
        else:
            accum += tok
            if re.search(r"(?:^|\n)\s*\*{0,2}answer\*{0,2}\s*:\s*$", accum, flags=re.IGNORECASE):
                after_answer_prefix = True

    for entry in entries:
        if entry["token"].strip():
            return entry
    return None


def _normalize_yes_no_token(token: str) -> str | None:
    cleaned = token.strip().strip("*").rstrip(".,!?;:").lower()
    if cleaned in ("yes", "no"):
        return cleaned
    return None


def yes_no_from_logprobs(entry: dict) -> tuple[float | None, float]:
    """Relative probability on ``yes`` vs ``no`` from an answer-token ``top_logprobs`` entry.

    Sums ``exp(logprob)`` over all surface variants of ``yes`` and ``no`` (e.g. ``"yes"``,
    ``"Yes"``, ``" yes"``) and divides ``yes_mass`` by ``yes_mass + no_mass``.
    Returns ``(p_yes, total_mass)``, or ``(None, 0.0)`` if neither appears.
    """
    yes_mass = 0.0
    no_mass = 0.0
    for alternative in entry.get("top_logprobs") or []:
        label = _normalize_yes_no_token(alternative["token"])
        if label == "yes":
            yes_mass += math.exp(alternative["logprob"])
        elif label == "no":
            no_mass += math.exp(alternative["logprob"])
    total = yes_mass + no_mass
    if total <= 0.0:
        return None, 0.0
    return yes_mass / total, total


def extract_p_yes(logprobs, thinking: str = "") -> float | None:
    """Extract normalized ``p_yes`` from backend ``logprobs``, or ``None`` when unavailable."""
    if thinking:
        return None
    entry = answer_token(logprobs)
    if entry is None:
        return None
    p_yes, _ = yes_no_from_logprobs(entry)
    return p_yes


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------

def sample_prompts(block: dict, config, steps: int = 3) -> str:
    """Preview the first ``steps`` user prompts of ``block`` assuming alternating guesses."""
    mode = config.run["mode"]
    parts = [
        f"##### Block {block['block_id']} (concept: {block['concept']!r}, "
        f"group: {block['complexity_group']}, word: {block['word']!r}, mode: {mode})",
    ]
    limit = min(steps, block["n_trials"])
    if mode == "multiturn_yesno":
        parts += [
            "",
            "----- trial 1 user message -----",
            start_message(block, config),
        ]
        for t in range(2, limit + 1):
            prev = block["trials"][t - 2]
            guess = (t % 2 == 0)
            parts += [
                "",
                f"----- trial {t} user message (after model answered {'yes' if guess else 'no'}, "
                f"truth={prev['truth']}) -----",
                step_message(block, t, guess, prev["truth"], config),
            ]
    else:
        history: list[dict] = []
        for t in range(1, limit + 1):
            parts += [
                "",
                f"----- trial {t} running-log prompt -----",
                trial_message(block, t, history, config, mode=mode),
            ]
            cur = block["trials"][t - 1]
            guess = (t % 2 == 1)
            history.append({
                "trial": t,
                "object": cur["object"],
                "truth": cur["truth"],
                "answer": guess,
                "rule_guess": "guessing" if t == 1 else "sample hypothesis",
            })
    return "\n".join(parts)


def main(argv=None):
    from bcl.blocks import read_blocks
    from bcl.config import load_config

    parser = argparse.ArgumentParser(description="Print sample prompts for one concept-learning block.")
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    parser.add_argument("--block", help="block id (default: the first block)")
    parser.add_argument("--steps", type=int, default=3, help="number of trials to preview (default: 3)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    blocks = read_blocks(config.resolve(config.blocks["file"]))
    if args.block:
        chosen = [b for b in blocks if b["block_id"] == args.block]
        if not chosen:
            raise SystemExit(f"no block with id {args.block!r}")
        block = chosen[0]
    else:
        block = blocks[0]
    print(sample_prompts(block, config, steps=args.steps))


if __name__ == "__main__":
    main()
