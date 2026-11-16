"""Evidence-only baselines for the follow-up decoder tests of 2026-11-15 (claims 4-5).

The follow-up tests compare residual-stream decoders with a 16-number tally of feature counts. A fuller record of
the evidence is the object table: for each of the 16 objects, was it seen labelled yes, and was it seen labelled no
(32 flags). This script runs the project's own decoder pipeline (bcl.probe.analyze_probes) on both records, with no
model involved, and then applies the tests to the out-of-block decoder outputs:

  1. conjunction     'red and square' when a red object and a square object have each been labelled no:
                     still consistent (the two were different objects) versus the red square itself labelled no
  2. contradicted    the rule the model states although earlier feedback contradicts it
  3. ambiguity       trials where another of the ten rules is still consistent versus trials where none is
  4. counterexample  an unstated single-feature rule on the trial before and after its first falsifying example
  5. same stated rule  the model states rule A; is rule B decoded higher while B is still consistent than once it is
                     ruled out (the pipeline's own same-pair test)
  6. graded          decoded value of an unstated rule by its probability under the fitted learner
  7. early trials    'red and not square' on its own blocks, trial by trial, and unlikely rules on trials 1 to 4
  8. P(yes)          the ten decoders combined into one P(yes), against the top decoder alone

Whatever an evidence record reproduces cannot by itself show that the model holds rule probabilities.
The trial set is the 285 blocks chosen by the rule in bcl.probe.prepare_probe_prompts. Tests 1 to 4 and 7 are defined
below because the code behind the saved residual-stream numbers is not in the repo, so trial counts differ slightly.
Tests 5, 6 and 8 use the pipeline's own definitions.

Run from the folder that holds config_bcl.yaml (about 3 minutes):   python analysis/evidence_baselines_new_tests.py [output.json]
The paper's section 4.4 uses tests 1 and 8 (an unstated second rule; the ten decoders pooled).
Written on 2026-11-15, during the sprint.
"""
import collections, json, os, sys, tempfile, types
import numpy as np

sys.path.insert(0, os.getcwd())
from bcl import probe as P_orig
from bcl.config import load_config
from bcl.score import concepts_in

# Run the repo's probe module unchanged except for one added line that exposes the out-of-block decoder outputs
# and the per-trial arrays the pipeline's own tests use.
anchor = "    oof_logits_all = np.zeros((n_layers, n_rows, n_concepts), dtype=np.float64)\n"
src = open(P_orig.__file__).read()
assert src.count(anchor) == 1, "bcl/probe.py has changed; update the anchor line in this script"
P = types.ModuleType("bcl.probe_exposed"); P.__package__ = "bcl"; P.__file__ = P_orig.__file__
sys.modules[P.__name__] = P
expose = ("    globals()['EXPOSED'] = dict(prob=oof_probs_all, p10=p_10_fit, cons=consistent, stated=stated_cidx, trial=trials_arr,"
          " true=true_cidx)\n")
exec(compile(src.replace(anchor, anchor + expose), P_orig.__file__, "exec"), P.__dict__)

cfg = load_config("config_bcl.yaml")
stim, feats = cfg.stimuli, cfg.features
rows = [json.loads(l) for l in open("data/bcl/results.jsonl")]
key = lambda o: (o["color"], o["shape"]); sidx = {key(s): i for i, s in enumerate(stim)}
colors, shapes = feats["color"], feats["shape"]
rule_by_name = {r.name: r for r in cfg.rules}

# ---- trials: same selection rule as prepare_probe_prompts (blocks that ever state the exact rule come first)
by_c = collections.defaultdict(list)
for r in rows: by_c[r["concept"]].append(r)
sel = []
for c in concepts_in(rows, list(cfg.programs.keys())):
    cr = sorted(by_c[c], key=lambda r: int(r["run_index"]))
    cext = tuple(1 if rule_by_name[c](s) else 0 for s in stim)
    pos = [r for r in cr if any(P.parse_rule_to_ext(t.get("rule_guess"), stim, feats) == cext for t in r["trials"])]
    pid = {r["block_id"] for r in pos}
    sel += sorted((pos + [r for r in cr if r["block_id"] not in pid])[:60 if c == "red" else 25], key=lambda r: int(r["run_index"]))
meta, hist = [], []
for b in sel:
    tr = sorted(b["trials"], key=lambda x: int(x["trial"]))
    for j, t in enumerate(tr):
        meta.append({"row_idx": len(meta), "condition": "no_rule_hist", "include_rule": False, "block_seq_key": "no_rule_hist|" + b["block_id"],
                     "block_id": b["block_id"], "concept": b["concept"], "complexity_group": b["complexity_group"], "run_index": int(b["run_index"]),
                     "word": b["word"], "trial": int(t["trial"]), "object": t["object"], "truth": bool(t["truth"]), "answer": t.get("answer"),
                     "p_yes": t.get("p_yes"), "rule_guess_raw": t.get("rule_guess"), "rule_guess_norm": P._norm_rule(t.get("rule_guess"))})
        hist.append(tr[:j])
n, D = len(meta), 64

def encode(kind):
    X = np.zeros((n, D), np.float32)
    for i, h in enumerate(hist):
        if kind == "feature tally":
            v = []
            for val in colors: v += [sum(x["object"]["color"] == val and x["truth"] for x in h), sum(x["object"]["color"] == val and not x["truth"] for x in h)]
            for val in shapes: v += [sum(x["object"]["shape"] == val and x["truth"] for x in h), sum(x["object"]["shape"] == val and not x["truth"] for x in h)]
        else:
            pos, neg = [0] * 16, [0] * 16
            for x in h: (pos if x["truth"] else neg)[sidx[key(x["object"])]] = 1
            v = pos + neg
        X[i, :len(v)] = v
    return X

KINDS = ["feature tally", "object table"]
acts = np.stack([encode(k) for k in KINDS], axis=1)
tmp = tempfile.mkdtemp(); mpath, apath = os.path.join(tmp, "meta.jsonl"), os.path.join(tmp, "acts.f32.bin")
with open(mpath, "w") as f:
    for m in meta: f.write(json.dumps(m) + "\n")
acts.astype(np.float32).tofile(apath)
if not hasattr(P, "p_10_fit"):      # harmless once bcl/probe.py assigns it inside analyze_probes (fixed on 2026-11-15)
    names0 = concepts_in(meta, list(cfg.programs.keys()))
    P.p_10_fit = P.compute_rule_posteriors_and_q_yes(cfg, meta, names0, {c: tuple(1 if rule_by_name[c](s) else 0 for s in stim) for c in names0},
                                                     fitted_params=(0.72954, 1.96550, 3.07512))["p_10_fit"]
bundle = P.analyze_probes(cfg, [(mpath, apath)], full_rows=rows, n_embd=D)
E = P.EXPOSED
prob = E["prob"]                                     # (2 records, trials, 10 rules), each trial scored by decoders trained on other blocks
names = list(bundle["concept_names"]); ci = {c: i for i, c in enumerate(names)}
cexts = {c: tuple(1 if rule_by_name[c](s) else 0 for s in stim) for c in names}
ext2c = {v: ci[k] for k, v in cexts.items()}
cons = np.array([[all(cexts[c][sidx[key(x["object"])]] == int(bool(x["truth"])) for x in h) for c in names] for h in hist])
stated = np.array([ext2c.get(e, -1) if isinstance(e, tuple) else -1 for e in (P.parse_rule_to_ext(m["rule_guess_raw"], stim, feats) for m in meta)])
rows_i = np.arange(n)

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

out = {"n_trials": n, "n_blocks": len(sel), "records": KINDS, "auroc_by_rule": {k: {c: bundle["per_rule_layer_metrics"][c][l]["auroc"] for c in names} for l, k in enumerate(KINDS)}}
print(f"{n} trials from {len(sel)} blocks; decoders given only a record of the evidence (no model)\n")
print("Mean out-of-block AUROC over the ten rule decoders: " + ", ".join(f"{k} {np.mean(list(out['auroc_by_rule'][k].values())):.3f}" for k in KINDS))

# ---- 1. conjunction
RS = ci["red and square"]
red_no = np.array([sum(x["object"]["color"] == "red" and not x["truth"] for x in h) for h in hist])
sq_no = np.array([sum(x["object"]["shape"] == "square" and not x["truth"] for x in h) for h in hist])
rs_no = np.array([any(key(x["object"]) == ("red", "square") and not x["truth"] for x in h) for h in hist])
n_yes = np.array([sum(bool(x["truth"]) for x in h) for h in hist])
elig = (red_no >= 1) & (sq_no >= 1) & (stated != RS)
live = elig & cons[:, RS]
dead_sets = {"red square labelled no and nothing else rules the rule out": elig & rs_no & (n_yes == 0), "red square labelled no (any history)": elig & rs_no}
print("\n1. 'red and square' once a red object and a square object have each been labelled no (rule not stated)")
print("   saved residual-stream numbers (data/bcl/multirule_probe_diagnostics.json): layer 21 live 0.27, dead 0.05, AUROC 0.76 (149 live, 32 dead); their feature tally AUROC 0.65")
out["conjunction"] = {}
for l, k in enumerate(KINDS):
    out["conjunction"][k] = {"n_live": int(live.sum()), "mean_live": float(prob[l, live, RS].mean())}
    print(f"   {k:14s} still consistent, the two were different objects: {prob[l, live, RS].mean():.3f} (n={live.sum()})")
    for label, dead in dead_sets.items():
        a = auroc(prob[l, live, RS], prob[l, dead, RS])
        out["conjunction"][k][label] = {"n_dead": int(dead.sum()), "mean_dead": float(prob[l, dead, RS].mean()), "auroc": a}
        print(f"   {'':14s} {label}: {prob[l, dead, RS].mean():.3f} (n={dead.sum()}), AUROC {a:.3f}")

# ---- 2. contradicted choices
has = stated >= 0
st_cons = has & cons[rows_i, np.where(has, stated, 0)]
st_contra = has & ~cons[rows_i, np.where(has, stated, 0)]
unst_contra = ~cons & (np.arange(len(names))[None, :] != stated[:, None])
print("\n2. Decoded probability of the rule the model states, by whether earlier feedback contradicts it")
print("   saved residual-stream numbers: stated and consistent 0.90 (layer 21); contradicted yet still emitted 0.39 (layer 21) to 0.66 (layer 30), n=40;")
print("   contradicted, no longer emitted without the rule history 0.25, n=41; unstated contradicted rules 0.03")
out["contradicted"] = {}
for l, k in enumerate(KINDS):
    p_st = prob[l, rows_i, np.where(has, stated, 0)]
    out["contradicted"][k] = {"n_stated_consistent": int(st_cons.sum()), "mean_stated_consistent": float(p_st[st_cons].mean()), "n_stated_contradicted": int(st_contra.sum()),
                              "mean_stated_contradicted": float(p_st[st_contra].mean()), "n_unstated_contradicted": int(unst_contra.sum()), "mean_unstated_contradicted": float(prob[l][unst_contra].mean())}
    print(f"   {k:14s} stated and consistent {p_st[st_cons].mean():.3f} (n={st_cons.sum()})   stated though contradicted {p_st[st_contra].mean():.3f} (n={st_contra.sum()})   unstated contradicted {prob[l][unst_contra].mean():.3f} (n={unst_contra.sum()})")

# ---- 3. ambiguous versus unambiguous trials
other = np.arange(len(names))[None, :] != stated[:, None]
n_other_live = (cons & other).sum(1)
amb, unamb = st_cons & (n_other_live >= 1), st_cons & (n_other_live == 0)
print("\n3. Trials where the stated rule is consistent: another of the ten rules still consistent (ambiguous) or none (unambiguous)")
print("   saved residual-stream numbers, layer 21: ambiguous (n=599) stated 0.78, other consistent 0.22, ruled out 0.03; unambiguous (n=1,398) stated 0.93, all others 0.04")
out["ambiguity"] = {}
for l, k in enumerate(KINDS):
    p_st = prob[l, rows_i, np.where(has, stated, 0)]
    o = {"n_ambiguous": int(amb.sum()), "ambiguous_stated": float(p_st[amb].mean()), "ambiguous_other_consistent": float(prob[l][amb[:, None] & cons & other].mean()),
         "ambiguous_ruled_out": float(prob[l][amb[:, None] & ~cons & other].mean()), "n_unambiguous": int(unamb.sum()), "unambiguous_stated": float(p_st[unamb].mean()),
         "unambiguous_others": float(prob[l][unamb[:, None] & other].mean())}
    out["ambiguity"][k] = o
    print(f"   {k:14s} ambiguous (n={o['n_ambiguous']}) stated {o['ambiguous_stated']:.3f}, other consistent {o['ambiguous_other_consistent']:.3f}, ruled out {o['ambiguous_ruled_out']:.3f};"
          f" unambiguous (n={o['n_unambiguous']}) stated {o['unambiguous_stated']:.3f}, all others {o['unambiguous_others']:.3f}")

# ---- 4. first counterexample for an unstated single-feature rule
print("\n4. Unstated single-feature rule (red, blue, square, circle): trial before and trial after its first falsifying example")
print("   saved residual-stream numbers, layer 21: one earlier positive example 0.21 -> 0.09 (n=50); two 0.74 -> 0.15 (n=5); none 0.06 -> 0.01 (n=74)")
single = {"red": ("color", "red"), "blue": ("color", "blue"), "square": ("shape", "square"), "circle": ("shape", "circle")}
pairs = collections.defaultdict(list)
for i in range(1, n):
    if meta[i]["block_id"] != meta[i - 1]["block_id"]: continue
    for c, (f, v) in single.items():
        k_ = ci[c]
        if cons[i - 1, k_] and not cons[i, k_] and stated[i - 1] != k_ and stated[i] != k_:
            hits = sum(x["object"][f] == v and x["truth"] for x in hist[i - 1])
            pairs[min(hits, 2)].append((i - 1, i, k_))
out["counterexample"] = {}
for l, k in enumerate(KINDS):
    out["counterexample"][k] = {}
    parts = []
    for hits in (0, 1, 2):
        pr = pairs[hits]
        if not pr: continue
        before = float(np.mean([prob[l, a, c_] for a, _, c_ in pr])); after = float(np.mean([prob[l, b_, c_] for _, b_, c_ in pr]))
        out["counterexample"][k][str(hits)] = {"n": len(pr), "before": before, "after": after}
        parts.append(f"{hits}{'+' if hits == 2 else ''} earlier positive: {before:.3f} -> {after:.3f} (n={len(pr)})")
    print(f"   {k:14s} " + ";  ".join(parts))

# ---- 5. same stated rule A, decode rule B (the pipeline's own test, bundle["same_pair_layer_curves"])
print("\n5. The model states rule A; decoder for another rule B: B still consistent versus B ruled out (mean AUROC over pairs of rules)")
print("   saved residual-stream numbers: layer 19 (the layer chosen for this test) all 30 pairs 0.79, the 4 colour-shape pairs 0.84, 0.82 when the current object lacks B;")
print("   layer 21: 0.77, 0.77, 0.73")
out["same_stated_rule"] = {}
for l, k in enumerate(KINDS):
    sp = bundle["same_pair_layer_curves"][l]
    out["same_stated_rule"][k] = {g: {"n_pairs": sp[g]["n_pairs"], "auroc": sp[g]["mean_within_pair_auroc"], "mean_consistent": sp[g]["mean_live_p"], "mean_ruled_out": sp[g]["mean_dead_p"]}
                                  for g in ("all_pairs", "orth_singleton_pairs", "orth_no_cur_obj_pairs")}
    print(f"   {k:14s} all {sp['all_pairs']['n_pairs']} pairs {sp['all_pairs']['mean_within_pair_auroc']:.3f}, colour-shape pairs {sp['orth_singleton_pairs']['mean_within_pair_auroc']:.3f},"
          f" current object lacks B {sp['orth_no_cur_obj_pairs']['mean_within_pair_auroc']:.3f}")

# ---- 6. decoded value of an unstated rule by its probability under the fitted learner (definitions of bcl/probe.py, part 2C)
tr, st, cs, p10 = E["trial"], E["stated"], E["cons"], E["p10"]
assert (cs == cons).all() and (st == stated).all(), "this script's consistency or stated-rule arrays differ from the pipeline's"
C = len(names)
base = np.full((len(KINDS), C, C), np.nan)           # mean decoded value of B on trials where the model states A and B is ruled out
for a in range(C):
    for b in range(C):
        m = (tr >= 2) & (st == a) & ~cs[:, b]
        if a != b and m.sum() >= 3: base[:, a, b] = prob[:, m, b].mean(1)
oi, oa, ob = map(np.array, zip(*[(i, st[i], b) for i in np.where((tr >= 2) & (st >= 0))[0] for b in range(C) if b != st[i] and not np.isnan(base[0, st[i], b])]))
bp, oc = p10[oi, ob], cs[oi, ob]
bins = [("ruled out", ~oc), ("consistent, up to 0.08", oc & (bp <= 0.08)), ("0.08 to 0.15", oc & (bp > 0.08) & (bp <= 0.15)),
        ("0.15 to 0.30", oc & (bp > 0.15) & (bp <= 0.30)), ("over 0.30", oc & (bp > 0.30))]
print("\n6. Decoded value of an unstated rule, by its probability under the fitted learner (mean, and mean above the ruled-out level for the same pair of rules)")
print("   saved residual-stream numbers, layer 21: ruled out 0.04; consistent with probability up to 0.08: 0.16 (+0.02); 0.08 to 0.15: 0.18 (+0.04); 0.15 to 0.30: 0.26 (+0.14); over 0.30: 0.46 (+0.36)")
print("   (n = 18,664, 79, 408, 325 and 351)")
out["graded"] = {}
for l, k in enumerate(KINDS):
    raw = prob[l, oi, ob]; exc = raw - base[l, oa, ob]
    out["graded"][k] = {lab: {"n": int(m.sum()), "mean_fitted_probability": float(bp[m].mean()), "mean_decoded": float(raw[m].mean()), "mean_above_ruled_out": float(exc[m].mean())} for lab, m in bins}
    print(f"   {k:14s} " + ";  ".join(f"{lab}: {raw[m].mean():.3f} ({exc[m].mean():+.3f}, n={m.sum()})" for lab, m in bins))
    live = oc
    out["graded"][k]["r_decoded_vs_fitted_probability_consistent_rules"] = float(np.corrcoef(raw[live], bp[live])[0, 1])
print("   correlation of decoded value with fitted probability over consistent unstated rules: " + ", ".join(f"{k} {out['graded'][k]['r_decoded_vs_fitted_probability_consistent_rules']:.3f}" for k in KINDS))

# ---- 7. early trials
RNS = ci["red and not square"]
own = E["true"] == RNS
print("\n7. Early trials. (a) Decoder for 'red and not square' on the blocks where it is the true rule, by trial")
print("   saved residual-stream numbers, layer 21: trials 1 to 4: 0.000, 0.000, 0.001, 0.004; trial 6: 0.09; trial 10: 0.25; trial 12: 0.52; trial 16: 0.75")
print("   share of these blocks in which the model states that rule: " + ", ".join(f"trial {t}: {np.mean(st[own & (tr == t)] == RNS):.2f}" for t in (4, 8, 10, 12, 14, 16)))
out["early_trials"] = {"red_and_not_square_own_blocks": {}, "unlikely_consistent_rules": {}}
for l, k in enumerate(KINDS):
    curve = [float(prob[l, own & (tr == t), RNS].mean()) for t in range(1, 17)]
    out["early_trials"]["red_and_not_square_own_blocks"][k] = curve
    print(f"   {k:14s} trials 1 to 4: " + ", ".join(f"{v:.3f}" for v in curve[:4]) + f"; trial 6: {curve[5]:.2f}; trial 10: {curve[9]:.2f}; trial 12: {curve[11]:.2f}; trial 16: {curve[15]:.2f}")
order = np.argsort(-p10, axis=1)
unlikely = cs & (p10 <= 0.08)
unlikely[rows_i, order[:, 0]] = False; unlikely[rows_i, order[:, 1]] = False
unlikely[has, stated[has]] = False
print("   (b) Rules that are still consistent, have fitted probability up to 0.08, and are neither stated nor among the two most probable: trials 1 to 4")
print("   saved residual-stream numbers, layer 21: 0.006, 0.048, 0.051, 0.042 (n = 1,140, 642, 295, 127)")
for l, k in enumerate(KINDS):
    vals = [(float(prob[l][unlikely & (tr == t)[:, None]].mean()), int((unlikely & (tr == t)[:, None]).sum())) for t in (1, 2, 3, 4)]
    out["early_trials"]["unlikely_consistent_rules"][k] = [{"trial": t, "mean_decoded": v, "n": m} for t, (v, m) in zip((1, 2, 3, 4), vals)]
    print(f"   {k:14s} " + ", ".join(f"{v:.3f}" for v, _ in vals) + " (n = " + ", ".join(str(m) for _, m in vals) + ")")

# ---- 8. the ten decoders combined into one P(yes) (the pipeline's own functions)
print("\n8. P(yes) on each trial from the top decoder alone and from the ten decoders combined: r and RMSE against the fitted learner's P(yes); NLL of the model's answers")
print("   saved residual-stream numbers, layer 22: top decoder alone r 0.773, RMSE 0.254, NLL 0.696; combined r 0.815, RMSE 0.197, NLL 0.515")
print("   for reference: the rule the model states r 0.652, RMSE 0.319, NLL 0.658; the fitted learner itself NLL 0.450; chance NLL 0.693")
out["p_yes"] = {}
for l, k in enumerate(KINDS):
    s, t = bundle["superposed_by_layer"][l], bundle["top1_by_layer"][l]
    pick = lambda d: {"r_vs_fitted": d["r_trial_fit_bayes"], "rmse_vs_fitted": d["rmse_trial_fit_bayes"], "nll_model_answers": d["llm_choice_mean_nll"], "agreement_model_answers": d["llm_choice_agreement"]}
    out["p_yes"][k] = {"top_decoder_alone": pick(t), "ten_decoders_combined": pick(s)}
    print(f"   {k:14s} top decoder alone r {t['r_trial_fit_bayes']:.3f}, RMSE {t['rmse_trial_fit_bayes']:.3f}, NLL {t['llm_choice_mean_nll']:.3f};"
          f" combined r {s['r_trial_fit_bayes']:.3f}, RMSE {s['rmse_trial_fit_bayes']:.3f}, NLL {s['llm_choice_mean_nll']:.3f}")

dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_baselines_new_tests.json")
os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
json.dump(out, open(dest, "w"), indent=1, default=float)
print("\nsaved", dest)
