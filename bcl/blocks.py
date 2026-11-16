"""Generate seeded concept-learning blocks and ideal Bayesian predictions.

Run from the project root:

    python -m bcl.blocks                         write the block file named in config_bcl.yaml
    python -m bcl.blocks --config config_bcl.yaml --force

How a block is made
-------------------
For each concept in ``RULES`` and ``run_index`` in ``0 .. blocks_per_concept - 1``:

1. ``block_id = f"{slug}-{run_index:03d}"`` where ``slug = concept.replace(" ", "_")``.
2. Draw ``word`` uniformly from ``blocks.nonce_words``.
3. Concatenate ``repetitions`` independent ``rng.sample(stimuli, len(stimuli))`` shuffles
   of all stimuli to form ``repetitions * len(stimuli)`` trials (e.g. 1 shuffled pass over
   16 unique stimuli = 16 trials with zero repetitions, or 4 passes over 9 stimuli = 36 trials).
4. Label each stimulus deterministically with ``rule(stimulus)`` (labels are never flipped).
5. Compute ``RuleSet.trajectory(trials, alpha=config.alpha)`` under both the configured
   simplicity prior and a uniform-prior ``RuleSet``. At 1-indexed trial ``t`` (``1..N``),
   ``trajectory[t - 1]`` holds the posterior BEFORE seeing trial ``t``'s label, giving the
   exact pre-feedback predictive probability ``bayes_p_yes`` and ``uniform_p_yes``.
"""

from __future__ import annotations

import argparse
import json
import random

from bcl.config import COMPLEXITY_GROUPS, load_config

TIE_TOLERANCE = 1e-9


def bayes_correct(bayes_p_yes: float, truth: bool) -> float:
    """Ideal accuracy at one trial: 1.0 when on the right side of 0.5, 0.5 on ties, else 0.0."""
    if abs(bayes_p_yes - 0.5) <= TIE_TOLERANCE:
        return 0.5
    return 1.0 if (bayes_p_yes > 0.5) == truth else 0.0


def predictive_p_yes(posterior, rules, stimulus, alpha: float) -> float:
    """P(label is True | posterior before seeing this trial's label, stimulus)."""
    return sum(
        posterior[name] * rules.label_probability(rules[name], stimulus, alpha=alpha)
        for name in rules.names
    )


def make_block(concept: str, run_index: int, config, rng: random.Random) -> dict:
    """Build one randomized multi-repetition block for ``concept``."""
    slug = concept.replace(" ", "_")
    block_id = f"{slug}-{run_index:03d}"
    word = rng.choice(config.blocks["nonce_words"])

    stimuli = config.stimuli
    n_stimuli = len(stimuli)
    repetitions = config.blocks["repetitions"]
    sequence = []
    for _ in range(repetitions):
        sequence.extend(rng.sample(stimuli, n_stimuli))

    rule = config.rules[concept]
    labeled_trials = [{**stimulus, "label": bool(rule(stimulus))} for stimulus in sequence]
    trajectory = config.hypothesis_rules.trajectory(labeled_trials, alpha=config.alpha)
    uniform_trajectory = config.uniform_rules.trajectory(labeled_trials, alpha=config.alpha)

    trials = []
    for t, (stimulus, labeled) in enumerate(zip(sequence, labeled_trials), start=1):
        truth = labeled["label"]
        posterior_before = trajectory[t - 1]
        uniform_before = uniform_trajectory[t - 1]
        p_yes = predictive_p_yes(posterior_before, config.hypothesis_rules, stimulus, config.alpha)
        u_p_yes = predictive_p_yes(uniform_before, config.uniform_rules, stimulus, config.alpha)
        trials.append({
            "trial": t,
            "repetition": (t - 1) // n_stimuli + 1,
            "object": {feature: stimulus[feature] for feature in config.features},
            "truth": truth,
            "bayes_p_yes": p_yes,
            "bayes_correct": bayes_correct(p_yes, truth),
            "uniform_p_yes": u_p_yes,
        })

    return {
        "block_id": block_id,
        "concept": concept,
        "program": rule.source,
        "complexity_group": config.complexity_groups[concept],
        "run_index": run_index,
        "word": word,
        "n_trials": len(trials),
        "repetitions": repetitions,
        "trials": trials,
    }


def generate(config) -> list[dict]:
    """Generate every block for ``config`` deterministically from ``config.seed``."""
    rng = random.Random(config.seed)
    blocks = []
    for concept in config.rules.names:
        for run_index in range(config.blocks["blocks_per_concept"]):
            blocks.append(make_block(concept, run_index, config, rng))
    return blocks


def write_blocks(blocks, config, force: bool = False):
    """Write ``blocks.jsonl``, ``blocks.config.yaml`` and ``blocks.meta.json``. Returns the path."""
    path = config.resolve(config.blocks["file"])
    text = "".join(json.dumps(block) + "\n" for block in blocks)
    if path.exists() and path.read_text() != text and not force:
        raise FileExistsError(
            f"{path} exists and the new blocks differ from it. Results made from the old "
            "blocks would no longer match. Rerun with --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    config.save_copy(path)
    meta = {
        "block_file": path.name,
        "n_blocks": len(blocks),
        "blocks_per_concept": config.blocks["blocks_per_concept"],
        "repetitions": config.blocks["repetitions"],
        "n_stimuli": len(config.stimuli),
        "n_trials": config.blocks["repetitions"] * len(config.stimuli),
        "seed": config.seed,
        "alpha": config.alpha,
        "label_flip_probability": 0.0,
        "features": config.features,
        "rules": {rule.name: rule.source for rule in config.rules},
        "complexity_groups": config.complexity_groups,
        "hypothesis_space": config.settings["prior"]["hypothesis_space"],
        "n_grammar_derivations": config.n_grammar_derivations,
        "n_hypothesis_concepts": len(config.hypothesis_rules),
        "prior": config.target_concept_priors,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def read_blocks(path) -> list[dict]:
    """The blocks in a JSONL file, as a list of dicts."""
    with open(path) as lines:
        return [json.loads(line) for line in lines if line.strip()]


def summarize(blocks, config) -> str:
    """A plain-text summary of the generated blocks and ideal Bayesian learning curves."""
    prior = config.settings["prior"]
    prior_map = config.target_concept_priors
    repetitions = config.blocks["repetitions"]
    n_trials = config.blocks["repetitions"] * len(config.stimuli)
    single_pass = (repetitions == 1 and n_trials >= 4)
    if single_pass:
        from bcl.score import stage_column_labels, trial_quarter

        col_labels = stage_column_labels(n_trials, True)
        rep_headers = "  ".join(f"{lbl:>10}" for lbl in col_labels)
    else:
        rep_headers = "  ".join(f"rep {r:>1}" for r in range(1, repetitions + 1))

    space_desc = (
        f" over {len(config.hypothesis_rules)} grammar concepts ({config.n_grammar_derivations:,} derivations)"
        if prior["hypothesis_space"] == "grammar"
        else f" over {len(config.hypothesis_rules)} rules"
    )
    lines = [
        f"{len(blocks)} blocks ({config.blocks['blocks_per_concept']} per concept x {len(config.rules)} concepts, "
        f"{n_trials} trials/block), seed {config.seed}, alpha {config.alpha}, {prior['kind']} prior"
        + (f" (dirichlet {prior['dirichlet']})" if prior["dirichlet"] else "")
        + (f" at temperature {prior['temperature']}" if prior["temperature"] else "")
        + space_desc,
        "",
        f"  {'concept':<20} {'group':<12} {'prior':>6}  {'bayes':>6}  {rep_headers}",
    ]
    for concept in config.rules.names:
        members = [b for b in blocks if b["concept"] == concept]
        all_scores = [t["bayes_correct"] for b in members for t in b["trials"]]
        overall = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        rep_means = []
        if single_pass:
            for q in range(1, 5):
                q_scores = [
                    t["bayes_correct"]
                    for b in members
                    for t in b["trials"]
                    if trial_quarter(t["trial"], n_trials) == q
                ]
                rep_means.append(f"{sum(q_scores) / len(q_scores):>10.3f}")
        else:
            for r in range(1, repetitions + 1):
                r_scores = [t["bayes_correct"] for b in members for t in b["trials"] if t["repetition"] == r]
                rep_means.append(f"{sum(r_scores) / len(r_scores):>5.3f}")
        lines.append(
            f"  {concept:<20} {config.complexity_groups[concept]:<12} {prior_map[concept]:>6.3f}  "
            f"{overall:>6.3f}  " + "  ".join(rep_means)
        )

    lines += ["", "Ideal Bayesian accuracy by complexity group:"]
    for group in COMPLEXITY_GROUPS:
        members = [b for b in blocks if b["complexity_group"] == group]
        if not members:
            continue
        scores = [t["bayes_correct"] for b in members for t in b["trials"]]
        lines.append(f"  {group:<12} ({len(members):>4} blocks): {sum(scores) / len(scores):.3f}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the block file named in the config.")
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing block file whose contents would change",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    blocks = generate(config)
    try:
        path = write_blocks(blocks, config, force=args.force)
    except FileExistsError as err:
        raise SystemExit(str(err))
    print(summarize(blocks, config))
    print(
        f"\nWrote {path}, {path.with_suffix('.config.yaml').name} "
        f"and {path.with_suffix('.meta.json').name}"
    )


if __name__ == "__main__":
    main()
