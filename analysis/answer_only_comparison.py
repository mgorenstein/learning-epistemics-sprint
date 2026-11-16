"""Section 3.1 of the paper: does the model need to write a rule before it answers?

    python analysis/answer_only_comparison.py

Run from the folder that holds config_bcl.yaml. Compares two runs of gemma4:26b on the same 1,000 blocks:

    data/bcl/results.jsonl               the main run: one prompt per trial, reply "Rule: ..." then "Answer: yes|no"
    data/bcl_answer_only/results.jsonl   an earlier run: a multi-turn chat in which the model replies only yes or no

The two runs differ in two ways, the rule line and the single-prompt format, so the comparison does not isolate either.
Thinking mode was off in both (see "think" in each results.config.yaml).

It also prints accuracy by repetition for a still earlier version of the task with nine objects shown four times each
(data/bcl_9obj_4rep/learning_curves.csv), the reason the final task shows every object once.

Written on 2026-11-15, during the sprint.
"""
import csv, json
import numpy as np

def load(path):
    return [json.loads(l) for l in open(path)]

def correct(rows):
    a = np.full((len(rows), 16), np.nan)
    for i, b in enumerate(rows):
        for t in b["trials"]:
            if t.get("answer") is not None:
                a[i, int(t["trial"]) - 1] = float(bool(t["answer"]) == bool(t["truth"]))
    return a

main, only = load("data/bcl/results.jsonl"), load("data/bcl_answer_only/results.jsonl")
objects = lambda b: [(t["object"]["color"], t["object"]["shape"]) for t in b["trials"]]
same = sum(x["block_id"] == y["block_id"] and objects(x) == objects(y) for x, y in zip(main, only))
print(f"blocks: {len(main)} and {len(only)}; identical block and object order in {same}")
print("thinking mode:", {str(b.get("think")) for b in main}, "in the main run,", {str(b.get("think")) for b in only}, "in the answer-only run")
m, a = correct(main), correct(only)
last4 = lambda x: np.nanmean(x[:, 12:])
print(f"with the rule line: {np.nanmean(m):.3f} correct over all trials, {last4(m):.3f} over trials 13 to 16")
print(f"answer only:        {np.nanmean(a):.3f} correct over all trials, {last4(a):.3f} over trials 13 to 16")
rng = np.random.default_rng(0)
boot = [last4(m[i]) - last4(a[i]) for i in (rng.integers(0, len(m), len(m)) for _ in range(4000))]     # the same blocks are drawn for both runs
print(f"difference over trials 13 to 16: {last4(m) - last4(a):+.3f}, 95% interval from resampling blocks [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")
yes = lambda rows: np.mean([bool(t["answer"]) for b in rows for t in b["trials"] if t.get("answer") is not None])
truth = np.mean([bool(t["truth"]) for b in main for t in b["trials"]])
print(f"share of 'yes' answers: {yes(main):.3f} with the rule line, {yes(only):.3f} answer only; share of objects that are members {truth:.3f}")

rows = [r for r in csv.DictReader(open("data/bcl_9obj_4rep/learning_curves.csv")) if r["scope_type"] == "all"]
by_rep = {}
for r in rows:
    by_rep.setdefault(int(r["repetition"]), []).append(float(r["correct"]))
print("\nearlier version, nine objects shown four times each (answer only): accuracy by repetition "
      + ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in sorted(by_rep.items())) + f"  ({rows[0]['n_blocks']} blocks per point)")
