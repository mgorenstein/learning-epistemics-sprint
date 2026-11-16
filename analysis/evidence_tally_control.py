"""Control for the decoders (section 4.4 of the paper, and the reference lines of Figure 2A): run the project's own
decoder pipeline, bcl.probe.analyze_probes, on a plain record of the evidence in each prompt in place of the
residual-stream activations. No model and no activations are needed.

Same trials (285 blocks chosen by the rule in prepare_probe_prompts: 60 'red', 25 for each other concept),
same labels, folds, weights and metrics. The only change is the input vector for each trial:

    tally           16 numbers: for each of the 8 feature values, how many earlier objects with that value
                    were labelled positive and how many negative
    object table    32 numbers: for each of the 16 objects, seen-positive and seen-negative flags
    label counts     2 numbers: how many earlier objects were positive, how many negative
    current object   8 numbers: one-hot colour and shape of the object being asked about (no history)
    constant         all zeros (what layer 0 is at a fixed final token)

If these inputs reproduce the probe results, the results show that the evidence is readable at the final
token, not that the model holds a distribution over rules.

Run from the folder that holds config_bcl.yaml (about 7 minutes):   python analysis/evidence_tally_control.py [output.json]
Prints the comparison and saves the numbers (default: evidence_tally_control.json beside this script),
which analysis/figures.py uses as reference lines in the decoder figure.
Written on 2026-11-14 and 2026-11-15, during the sprint.
"""
import collections, json, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.getcwd())
from bcl import probe as P
from bcl.config import load_config
from bcl.score import concepts_in

cfg = load_config("config_bcl.yaml")
stim, feats = cfg.stimuli, cfg.features
rows = [json.loads(l) for l in open("data/bcl/results.jsonl")]
key = lambda o: (o["color"], o["shape"]); sidx = {key(s): i for i, s in enumerate(stim)}
colors, shapes = feats["color"], feats["shape"]

# ---- trials: same selection rule as prepare_probe_prompts (blocks that ever state the exact rule come first)
N_PER = {"red": 60}
by_c = collections.defaultdict(list)
for r in rows: by_c[r["concept"]].append(r)
rule_by_name = {r.name: r for r in cfg.rules}
sel = []
for c in concepts_in(rows, list(cfg.programs.keys())):
    cr = sorted(by_c[c], key=lambda r: int(r["run_index"]))
    cext = tuple(1 if rule_by_name[c](s) else 0 for s in stim)
    pos = [r for r in cr if any(P.parse_rule_to_ext(t.get("rule_guess"), stim, feats) == cext for t in r["trials"])]
    pid = {r["block_id"] for r in pos}
    sel += sorted((pos + [r for r in cr if r["block_id"] not in pid])[:N_PER.get(c, 25)], key=lambda r: int(r["run_index"]))
meta = []
for b in sel:
    for t in sorted(b["trials"], key=lambda x: int(x["trial"])):
        meta.append({"row_idx": len(meta), "condition": "no_rule_hist", "include_rule": False, "block_seq_key": "no_rule_hist|" + b["block_id"],
                     "block_id": b["block_id"], "concept": b["concept"], "complexity_group": b["complexity_group"], "run_index": int(b["run_index"]),
                     "word": b["word"], "trial": int(t["trial"]), "object": t["object"], "truth": bool(t["truth"]), "answer": t.get("answer"),
                     "p_yes": t.get("p_yes"), "rule_guess_raw": t.get("rule_guess"), "rule_guess_norm": P._norm_rule(t.get("rule_guess"))})
n, D = len(meta), 64

def encode(kind):
    X = np.zeros((n, D), np.float32); i = 0
    for b in sel:
        tr = sorted(b["trials"], key=lambda x: int(x["trial"]))
        for j, t in enumerate(tr):
            hist, v = tr[:j], []
            if kind == "tally":
                for val in colors: v += [sum(h["object"]["color"] == val and h["truth"] for h in hist), sum(h["object"]["color"] == val and not h["truth"] for h in hist)]
                for val in shapes: v += [sum(h["object"]["shape"] == val and h["truth"] for h in hist), sum(h["object"]["shape"] == val and not h["truth"] for h in hist)]
            elif kind == "object table":
                pos, neg = [0] * 16, [0] * 16
                for h in hist: (pos if h["truth"] else neg)[sidx[key(h["object"])]] = 1
                v = pos + neg
            elif kind == "label counts":
                v = [sum(bool(h["truth"]) for h in hist), sum(not h["truth"] for h in hist)]
            elif kind == "current object":
                v = [float(t["object"]["color"] == c) for c in colors] + [float(t["object"]["shape"] == s) for s in shapes]
            X[i, :len(v)] = v; i += 1
    return X

KINDS = ["constant", "current object", "label counts", "tally", "object table"]
acts = np.stack([encode(k) for k in KINDS], axis=1)          # (trials, 'layers', D)
tmp = tempfile.mkdtemp()
mpath, apath = os.path.join(tmp, "meta.jsonl"), os.path.join(tmp, "acts.f32.bin")
with open(mpath, "w") as f:
    for m in meta: f.write(json.dumps(m) + "\n")
acts.astype(np.float32).tofile(apath)

def run(second_best_from):
    """second_best_from: 'fit' (fitted 3-parameter posterior, as the text says) or 'ideal' (parameter-free posterior,
    which is what reproduces the write-up's trial counts of 243 and 1,173)."""
    names = concepts_in(meta, list(cfg.programs.keys()))
    cexts = {c: tuple(1 if rule_by_name[c](s) else 0 for s in stim) for c in names}
    fitted = (0.72954, 1.96550, 3.07512)
    bay = P.compute_rule_posteriors_and_q_yes(cfg, meta, names, cexts, fitted_params=fitted)
    # bcl/probe.py reads p_10_fit inside analyze_probes without assigning it (NameError at line 1016 as of 2026-11-14).
    # Supplying it as a module global lets the unmodified file run; it is ignored once the missing line is added.
    P.p_10_fit = bay["p_10_fit"] if second_best_from == "fit" else bay["p_10_norm"]
    orig = P.compute_rule_posteriors_and_q_yes
    if second_best_from == "ideal":
        P.compute_rule_posteriors_and_q_yes = lambda *a, **k: {**orig(*a, **k), "p_10_fit": bay["p_10_norm"]}
    try:
        return P.analyze_probes(cfg, [(mpath, apath)], full_rows=rows, n_embd=D)
    finally:
        P.compute_rule_posteriors_and_q_yes = orig

bf, bi = run("fit"), run("ideal")
saved = json.load(open("data/bcl/multirule_probe_diagnostics.json")) if os.path.exists("data/bcl/multirule_probe_diagnostics.json") else None
def saved_line(kind):
    """The residual-stream numbers of the last saved probe run, at layer 21, for comparison."""
    if not saved: return "  (no saved residual-stream run in data/bcl/multirule_probe_diagnostics.json)"
    if kind == "auroc":
        pk = [max(m["auroc"] for m in saved["per_rule_layer_metrics"][c]) for c in saved["per_rule_layer_metrics"]]
        return f"  saved residual-stream run, best layer per rule: {min(pk):.3f} to {max(pk):.3f}"
    if kind == "rank":
        r = saved["rank_layer_curves"][21]["committed_to_rank1"]
        return (f"  saved residual-stream run, layer 21 (n={r['n']}): stated {r['mean_p_rank1']:.3f}   unstated second {r['mean_p_rank2']:.3f}   "
                f"falsified {r['mean_p_falsified']:.3f}   AUROC {r['within_trial_auroc_rank2_vs_falsified']:.3f}")
    sp = saved["same_pair_layer_curves"][19]
    return (f"  saved residual-stream run, layer 19: all {sp['all_pairs']['n_pairs']} pairs {sp['all_pairs']['mean_within_pair_auroc']:.3f}; colour/shape pairs "
            f"{sp['orth_singleton_pairs']['mean_within_pair_auroc']:.3f}; current object lacks B {sp['orth_no_cur_obj_pairs']['mean_within_pair_auroc']:.3f}")
f = lambda x: "  n/a" if x is None else f"{x:5.3f}"
print(f"{n} trials from {len(sel)} blocks\n")
print("Out-of-block AUROC of each rule decoder")
print(saved_line("auroc"))
print(f"  {'rule':20s}" + "".join(f"{k:>16s}" for k in KINDS))
for c in bf["concept_names"]:
    print(f"  {c:20s}" + "".join(f"{bf['per_rule_layer_metrics'][c][l]['auroc']:16.4f}" for l in range(len(KINDS))))
print("\nTrials where the model states the top-ranked rule and a second rule keeps weight")
print(saved_line("rank"))
for label, bundle in (("second-best from the parameter-free posterior", bi), ("second-best from the fitted posterior", bf)):
    print(f"  {label}")
    for l, k in enumerate(KINDS):
        rc = bundle["rank_layer_curves"][l]["committed_to_rank1"]
        print(f"    {k:16s} n={rc['n']:4d}  stated {f(rc['mean_p_rank1'])}   unstated second {f(rc['mean_p_rank2'])}   falsified {f(rc['mean_p_falsified'])}   AUROC {f(rc['within_trial_auroc_rank2_vs_falsified'])}")
print("\nSame stated rule A, decode rule B: B still consistent with the feedback versus B falsified (mean AUROC over pairs)")
print(saved_line("pairs"))
for l, k in enumerate(KINDS):
    sp = bf["same_pair_layer_curves"][l]
    print(f"    {k:16s} all {sp['all_pairs']['n_pairs']} pairs {sp['all_pairs']['mean_within_pair_auroc']:.3f}; colour/shape pairs {sp['orth_singleton_pairs']['mean_within_pair_auroc']:.3f}; current object lacks B {sp['orth_no_cur_obj_pairs']['mean_within_pair_auroc']:.3f}")
print("\nTrial-level P(yes) from combining the ten decoders, against the fitted model's P(yes)")
print("  residual stream, layer 22, from tables/bcl/probe_trial_predictions.csv of 2026-11-14: combined r 0.815, RMSE 0.197, choice NLL 0.515; top decoder only RMSE 0.254")
for l, k in enumerate(KINDS):
    s, t = bf["superposed_by_layer"][l], bf["top1_by_layer"][l]
    print(f"    {k:16s} combined r {s['r_trial_fit_bayes']:.3f}, RMSE {s['rmse_trial_fit_bayes']:.3f}, choice NLL {s['llm_choice_mean_nll']:.3f}, 160-point r {s['full_160_r']:.3f}; top decoder only RMSE {t['rmse_trial_fit_bayes']:.3f}")

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_tally_control.json")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
per_input = lambda fn: {k: fn(l) for l, k in enumerate(KINDS)}
with open(out, "w") as fh:
    json.dump({
        "n_trials": n, "n_blocks": len(sel), "inputs": KINDS,
        "auroc_by_rule": per_input(lambda l: {c: bf["per_rule_layer_metrics"][c][l]["auroc"] for c in bf["concept_names"]}),
        "ambiguous_trials_second_best_from_parameter_free_posterior": per_input(lambda l: bi["rank_layer_curves"][l]["committed_to_rank1"]),
        "ambiguous_trials_second_best_from_fitted_posterior": per_input(lambda l: bf["rank_layer_curves"][l]["committed_to_rank1"]),
        "same_stated_rule_pairs": per_input(lambda l: bf["same_pair_layer_curves"][l]),
        "p_yes_combined_decoders": per_input(lambda l: bf["superposed_by_layer"][l]),
        "p_yes_top_decoder_only": per_input(lambda l: bf["top1_by_layer"][l]),
    }, fh, indent=1, default=float)
print("\nsaved", out)
