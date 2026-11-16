"""Run interactive concept-learning blocks, saving one JSONL row per block.

Run from the project root:

    python -m bcl.run                             run all pending blocks in config_bcl.yaml
    python -m bcl.run --blocks_per_concept 10     balanced pilot: 10 blocks per concept (100 blocks)
    python -m bcl.run --estimate                  estimate remaining runtime from timings so far

Conversation and KV-cache reuse
-------------------------------
``config.run["mode"]`` selects how each ``n_trials`` block is presented:

- ``running_log_rule``: each trial is a single-turn user prompt carrying an explicit
  running log of prior stimuli, feedback, and rule guesses, eliciting two short lines
  (``Rule: <hypothesis>`` + ``Answer: <yes or no>``).
- ``running_log_yesno``: same explicit running-log single-turn prompt, eliciting only
  ``yes`` or ``no``.
- ``multiturn_yesno``: one growing ``n_trials``-turn chat conversation using ``start.txt``
  and ``step.txt``.

With ``backends.ollama.reuse_cache: true``, ``OllamaBackend`` is constructed with
``fresh=False`` so no cache-busting reset prompt is sent between trials of a block, allowing
Ollama to reuse shared prompt prefixes in its active KV cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from bcl import elicit
from bcl.blocks import bayes_correct, read_blocks
from bcl.config import load_config, model_label
from hh.backends import BackendError, OllamaBackend, OpenAIBackend

NOT_IN_FINGERPRINT = ("results_file", "timeout_seconds")


# ---------------------------------------------------------------------------
# Ordering, keys, and backends
# ---------------------------------------------------------------------------

def balanced_order(blocks: list[dict], blocks_per_concept: int | None = None) -> list[dict]:
    """Order blocks round-robin by ``run_index`` outer loop, ``concept`` inner loop.

    For example: ``000`` across all 10 concepts, then ``001`` across all 10 concepts, and so on.
    When ``blocks_per_concept`` is given, keeps only the first ``N`` blocks of each concept.
    """
    if blocks_per_concept is not None and blocks_per_concept < 1:
        raise ValueError(f"blocks_per_concept must be 1 or more, got {blocks_per_concept}")
    concept_order: dict[str, int] = {}
    for block in blocks:
        concept_order.setdefault(block["concept"], len(concept_order))
    ordered = [
        block
        for _, block in sorted(
            enumerate(blocks),
            key=lambda item: (item[1]["run_index"], concept_order[item[1]["concept"]], item[0]),
        )
    ]
    if blocks_per_concept is None:
        return ordered
    seen: Counter[str] = Counter()
    kept = []
    for block in ordered:
        concept = block["concept"]
        if seen[concept] < blocks_per_concept:
            kept.append(block)
            seen[concept] += 1
    return kept


def fingerprint(config) -> str:
    """A short hash of everything a results row depends on besides the model."""
    digest = hashlib.sha256()
    digest.update(config.resolve(config.blocks["file"]).read_bytes())
    for path in sorted(elicit.PROMPTS.glob("*.txt")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    settings = {name: value for name, value in config.run.items() if name not in NOT_IN_FINGERPRINT}
    ollama = config.settings["backends"]["ollama"]
    settings.update(
        seed=config.seed,
        ollama_options=ollama["options"],
        ollama_reuse_cache=ollama["reuse_cache"],
    )
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def make_key(label: str, block_id: str) -> str:
    """Unique resume key for one (model, block) pair."""
    return f"{label}|{block_id}"


def read_rows(path) -> list[dict]:
    """The rows of a results file, or an empty list if it does not exist yet."""
    if not path.exists():
        return []
    with open(path) as lines:
        return [json.loads(line) for line in lines if line.strip()]


def make_backends(config) -> dict:
    """Build the backends that ``config.settings['models']`` need, keyed by backend name."""
    settings = config.settings["backends"]
    timeout = config.run["timeout_seconds"]
    backends = {}
    for name in {model["backend"] for model in config.settings["models"]}:
        if name == "ollama":
            ollama = settings["ollama"]
            backends[name] = OllamaBackend(
                ollama["base_url"],
                ollama["options"],
                timeout,
                fresh=not ollama["reuse_cache"],
            )
        elif name == "openai":
            openai = settings["openai"]
            backends[name] = OpenAIBackend.from_environment(
                openai["base_url_env"],
                openai["api_key_env"],
                timeout,
            )
    return backends


# ---------------------------------------------------------------------------
# Running one 36-trial block and a batch
# ---------------------------------------------------------------------------

def run_block(backend, model: dict, block: dict, config) -> tuple[list[dict], int, float]:
    """Execute one sequential block and return ``(trial_rows, parse_failures, seconds)``."""
    mode = config.run["mode"]
    messages: list[dict] = []
    trial_rows: list[dict] = []
    parse_failures = 0
    total_seconds = 0.0
    top_logprobs = config.run["top_logprobs"]

    for t_info in block["trials"]:
        trial_num = int(t_info["trial"])
        if mode == "multiturn_yesno":
            if trial_num == 1:
                prompt = elicit.start_message(block, config)
            else:
                prev = trial_rows[-1]
                prompt = elicit.step_message(block, trial_num, prev["answer"], prev["truth"], config)
            messages.append({"role": "user", "content": prompt})
            call_messages = messages
        else:
            prompt = elicit.trial_message(block, trial_num, trial_rows, config, mode=mode)
            call_messages = [{"role": "user", "content": prompt}]

        reply = backend.chat(
            model["name"],
            call_messages,
            temperature=config.run["temperature"],
            seed=config.seed,
            max_tokens=config.run["max_tokens"],
            think=model["think"],
            top_logprobs=top_logprobs,
            extra_body=model["extra_body"],
        )
        if mode == "multiturn_yesno":
            messages.append({"role": "assistant", "content": reply.text})
            answer, _ = elicit.parse_answer(reply.text)
            rule_guess = None
        elif mode == "running_log_rule":
            answer, rule_guess, _ = elicit.parse_rule_and_answer(reply.text)
        else:
            answer, _ = elicit.parse_answer(reply.text)
            rule_guess = None

        if answer is None:
            parse_failures += 1
        p_yes = elicit.extract_p_yes(reply.logprobs, thinking=reply.thinking) if top_logprobs else None

        truth = bool(t_info["truth"])
        b_p_yes = float(t_info["bayes_p_yes"])
        b_correct = float(t_info.get("bayes_correct", bayes_correct(b_p_yes, truth)))
        u_p_yes = float(t_info["uniform_p_yes"])
        seconds = float(reply.seconds)

        trial_rows.append({
            "trial": trial_num,
            "repetition": int(t_info["repetition"]),
            "object": dict(t_info["object"]),
            "truth": truth,
            "rule_guess": rule_guess,
            "answer": answer,
            "raw_response": reply.text,
            "p_yes": p_yes,
            "correct": answer is truth,
            "bayes_p_yes": b_p_yes,
            "bayes_correct": b_correct,
            "uniform_p_yes": u_p_yes,
            "cached_tokens": reply.cached_tokens,
            "seconds": seconds,
        })
        total_seconds += seconds

    return trial_rows, parse_failures, total_seconds


def pending(config, blocks: list[dict], rows: dict) -> list[tuple[dict, dict]]:
    """Every ``(model, block)`` pair not yet present in ``rows``."""
    todo = []
    for model in config.settings["models"]:
        label = model_label(model)
        for block in blocks:
            if make_key(label, block["block_id"]) not in rows:
                todo.append((model, block))
    return todo


def run(config, backends: dict, blocks_per_concept: int | None = None, log=print) -> int:
    """Run every pending block and append one JSONL row per completed block. Returns count written."""
    path = config.resolve(config.run["results_file"])
    blocks = balanced_order(
        read_blocks(config.resolve(config.blocks["file"])),
        blocks_per_concept=blocks_per_concept,
    )
    mark = fingerprint(config)
    rows = {row["key"]: row for row in read_rows(path)}
    stale = sum(row["fingerprint"] != mark for row in rows.values())
    if stale:
        raise SystemExit(
            f"{path} holds {stale} rows made with different blocks, prompt templates or run settings. "
            "Move that file aside, or name another run.results_file, so the two are not mixed."
        )

    todo = pending(config, blocks, rows)
    log(f"{len(todo)} blocks to run, {len(rows)} already in {path}")
    if not todo:
        if rows:
            config.save_copy(path)
        return 0

    described = {}
    for model in {model_label(m): m for m, _ in todo}.values():
        try:
            described[model_label(model)] = backends[model["backend"]].describe(model["name"])
        except BackendError as err:
            raise SystemExit(f"cannot run {model_label(model)}: {err}")

    path.parent.mkdir(parents=True, exist_ok=True)
    config.save_copy(path)
    written = failed = 0

    with open(path, "a") as out, open(path.with_suffix(".log"), "a") as failures:
        for number, (model, block) in enumerate(todo, start=1):
            label = model_label(model)
            key = make_key(label, block["block_id"])
            backend = backends[model["backend"]]
            try:
                trial_rows, parse_failures, total_seconds = run_block(backend, model, block, config)
            except BackendError as err:
                failed += 1
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                failures.write(f"{stamp}\t{key}\t{err}\n")
                failures.flush()
                log(f"FAILED {key}: {err}")
                continue

            row = {
                "key": key,
                "fingerprint": mark,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": label,
                "model_name": model["name"],
                "model_digest": described[label]["digest"],
                "backend": backend.name,
                "backend_version": described[label]["version"],
                "think": model["think"],
                "mode": config.run["mode"],
                "block_id": block["block_id"],
                "concept": block["concept"],
                "program": block["program"],
                "complexity_group": block["complexity_group"],
                "run_index": int(block["run_index"]),
                "word": block["word"],
                "n_trials": int(block["n_trials"]),
                "repetitions": int(block["repetitions"]),
                "seconds": total_seconds,
                "parse_failures": parse_failures,
                "trials": trial_rows,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows[key] = row
            written += 1
            if number % 10 == 0 or number == len(todo):
                acc = sum(t["correct"] for t in trial_rows) / len(trial_rows)
                log(
                    f"  {number}/{len(todo)} blocks, last: {key} "
                    f"(acc {acc:.1%}, {total_seconds:.1f} s)"
                )

    log(
        f"wrote {written} blocks to {path}"
        + (f"; {failed} blocks failed, see {path.with_suffix('.log').name}" if failed else "")
    )
    return written


def estimate(config, blocks_per_concept: int | None = None) -> str:
    """Estimate remaining runtime from completed block timings in ``run.results_file``."""
    path = config.resolve(config.run["results_file"])
    blocks = balanced_order(
        read_blocks(config.resolve(config.blocks["file"])),
        blocks_per_concept=blocks_per_concept,
    )
    rows = {row["key"]: row for row in read_rows(path)}
    todo = pending(config, blocks, rows)
    seconds_by_model: dict[str, list[float]] = {}
    for row in rows.values():
        seconds_by_model.setdefault(row["model"], []).append(row["seconds"])

    total = 0.0
    unknown = 0
    for model, _ in todo:
        timings = seconds_by_model.get(model_label(model))
        if timings:
            total += sum(timings) / len(timings)
        else:
            unknown += 1

    n_trials = config.blocks["repetitions"] * len(config.stimuli)
    text = f"{len(todo)} blocks ({len(todo) * n_trials} trials) to run; {len(rows)} done."
    if len(todo) > unknown:
        text += (
            f" About {total / 60:.1f} minutes for the {len(todo) - unknown} blocks "
            "with timings so far."
        )
    if unknown:
        text += f" No timings yet for {unknown} of them (a model that has not been run)."
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the interactive concept-learning blocks not yet in the results file."
    )
    parser.add_argument("--config", default="config_bcl.yaml", help="config file (default: config_bcl.yaml)")
    parser.add_argument(
        "--blocks_per_concept",
        "--blocks-per-concept",
        dest="blocks_per_concept",
        type=int,
        metavar="N",
        help="run only the first N blocks per concept, balanced across all concepts",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="only estimate the time the remaining blocks would take",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.estimate:
        print(estimate(config, args.blocks_per_concept))
        return
    run(config, make_backends(config), args.blocks_per_concept)


if __name__ == "__main__":
    main()
