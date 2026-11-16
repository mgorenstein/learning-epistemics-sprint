"""Behavioural numbers of the paper (sections 3.1 to 3.3), recomputed from the model's saved answers.

    python analysis/behaviour_checks.py              # everything except the exemplar-model fit
    python analysis/behaviour_checks.py --exemplar   # also fit the exemplar (GCM) model of Table 1, a few seconds more

Run from the folder that holds config_bcl.yaml. Reads data/bcl/results.jsonl and data/bcl/pilot_multi_hypothesis.json.
The output is grouped under the claims of the review it was written for: claim 1 is section 3.1 (learning), claim 2 is
section 3.3 (the fitted Bayesian learner and the exemplar model), claim 3 is section 3.2 (one rule at a time, and the
160-trial experiment that asks for several rules).

Written on 2026-11-14 and 2026-11-15, during the sprint.
"""
import argparse, collections, json, math, os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from bcl.config import load_config
from bcl.fit import _HypothesisEvaluator, _nelder_mead
from bcl.probe import parse_rule_to_ext

ap = argparse.ArgumentParser(); ap.add_argument("--exemplar", action="store_true"); args = ap.parse_args()
cfg = load_config("config_bcl.yaml")
stim, feats = cfg.stimuli, cfg.features
rows = [json.loads(l) for l in open("data/bcl/results.jsonl")]
ev = _HypothesisEvaluator(cfg, rows)
V, A, T, C = ev.valid, ev.ans_yes, ev.truth, ev.correct
B, N = A.shape
SX = ev.stim_idx
E, LOGW = ev.extensions, ev.log_w0
H = E.shape[0]
ext2h = {tuple(int(v) for v in E[i]): i for i in range(H)}
r_ = lambda x, y: float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])
rmse_ = lambda x, y: float(np.sqrt(np.mean((np.ravel(x) - np.ravel(y)) ** 2)))
quarters = lambda a: [round(float(a[:, i:i + 4].mean()), 3) for i in (0, 4, 8, 12)]
FIT3 = (0.72954, 1.96550, 3.07512)          # alpha, beta, prior temperature from tables/bcl/parameter_fits.csv
IDEAL = (float(cfg.alpha), 1.0, 1.0)        # parameter-free ideal learner

def posterior(alpha, temp):
    """Posterior over the 830 rule extensions before each trial: (blocks, trials, rules)."""
    pm, pmm = 0.5 * (1 + alpha), 0.5 * (1 - alpha)
    dll = ev.agree_bt * (math.log(pm) - math.log(pmm)) + math.log(pmm)
    cum = np.zeros_like(dll); cum[:, 1:, :] = np.cumsum(dll[:, :-1, :], axis=1)
    lp = LOGW[None, None, :] / temp + cum; lp -= lp.max(2, keepdims=True)
    p = np.exp(lp); return p / p.sum(2, keepdims=True)

P_ideal = posterior(IDEAL[0], IDEAL[2])
consistent = np.ones((B, N, H), bool); consistent[:, 1:, :] = np.cumprod(ev.agree_bt[:, :-1, :], axis=1).astype(bool)

print("=" * 100, "\nCLAIM 1: task performance against reference learners")
q_ideal = ev.predict_q_yes(*IDEAL)
ideal_argmax = np.where(np.abs(q_ideal - 0.5) < 1e-9, 0.5, ((q_ideal > 0.5) == (T > 0.5)).astype(float))
cumpos = np.zeros((B, N)); cumpos[:, 1:] = np.cumsum(T[:, :-1], 1); cumn = np.tile(np.arange(N), (B, 1))
maj = np.where(cumpos * 2 > cumn, 1.0, np.where(cumpos * 2 < cumn, 0.0, 0.5))
maj_acc = np.where(T == 1, maj, 1 - maj)
print(f"  model            overall {C.mean():.3f}  quarters {quarters(C)}  trial 16 {C[:, 15].mean():.3f}")
print(f"  ideal learner    overall {ideal_argmax.mean():.3f}  quarters {quarters(ideal_argmax)}  (best guess from the exact posterior)")
print(f"  label frequency  overall {maj_acc.mean():.3f}  quarters {quarters(maj_acc)}  (answers with the majority label seen so far)")
print(f"  always 'no'      overall {1 - T.mean():.3f}")
exts = [[parse_rule_to_ext(t.get("rule_guess"), stim, feats) for t in r["trials"]] for r in rows]
print("  share of trials answered 'guessing', by trial:", [round(float(np.mean([exts[b][j] == "GUESSING" for b in range(B)])), 2) for j in range(N)])
print("  by concept (the share of positive objects differs, so accuracy across concept families is not comparable as it stands;")
print("  balanced accuracy = mean of accuracy on objects inside the category and on objects outside it, 0.5 for any constant answer):")
majority_known = []
for c in ev.concept_order:
    m = ev.concepts == c; pos = float(T[m].mean()); majority_known.append(max(pos, 1 - pos))
    vm, tm, cm = V[m], T[m], C[m]
    balanced = 0.5 * (cm[vm & (tm == 1)].mean() + cm[vm & (tm == 0)].mean())
    print(f"    {c:20s} positive objects {round(pos * 16):2d}/16   always the majority label {max(pos, 1 - pos):.3f}   model {C[m].mean():.3f}   model, trials 13-16 {C[m][:, 12:].mean():.3f}   balanced {balanced:.3f}")
print(f"  always the majority label of the block's rule (known in advance), mean over concepts: {np.mean(majority_known):.3f}")

print("=" * 100, "\nCLAIM 2: how much of the 160-point fit is the shared learning curve?")
emp = np.array([C[ev.concepts == c].mean(0) for c in ev.concept_order])
def curves(q): pa = np.where(T == 1.0, q, 1 - q); return np.array([pa[ev.concepts == c].mean(0) for c in ev.concept_order])
def nll(q): q = np.clip(q, 1e-12, 1 - 1e-12); return -float(np.sum((A * np.log(q) + (1 - A) * np.log(1 - q))[V]))
nv = int(V.sum())
blind = np.array([np.delete(emp, i, axis=0).mean(0) for i in range(len(emp))])
print(f"  concept-blind baseline (mean curve of the other nine concepts): r = {r_(blind, emp):.3f}, RMSE = {rmse_(blind, emp):.3f}")
for name, par in (("3-parameter fit (in sample)", FIT3), ("ideal learner, probability matching", IDEAL)):
    m = curves(ev.predict_q_yes(*par))
    dev = lambda x: x - x.mean(0, keepdims=True)
    print(f"  {name:38s}: r = {r_(m, emp):.3f}, RMSE = {rmse_(m, emp):.3f}; concept-specific part only r = {r_(dev(m), dev(emp)):.3f}; "
          f"per-concept mean accuracy r = {r_(m.mean(1), emp.mean(1)):.3f}; trial NLL = {nll(ev.predict_q_yes(*par)):.1f} ({nll(ev.predict_q_yes(*par)) / nv:.4f} nats/trial)")
print(f"  chance: trial NLL = {nv * math.log(2):.1f} (0.6931 nats/trial)")
if args.exemplar:
    col = np.array([feats["color"].index(s["color"]) for s in stim]); shp = np.array([feats["shape"].index(s["shape"]) for s in stim])
    dc = (col[SX][:, :, None] != col[SX][:, None, :]).astype(float); ds = (shp[SX][:, :, None] != shp[SX][:, None, :]).astype(float)
    past = np.tril(np.ones((N, N)), -1)[None]
    def q_gcm(z):
        c, gam, by, bn = (math.exp(max(-12.0, min(6.0, v))) for v in z)
        s = np.exp(-c * 0.5 * (dc + ds)) * past
        ey = by + np.einsum("bji,bi->bj", s, T); en = bn + np.einsum("bji,bi->bj", s, 1 - T)
        return 1 / (1 + np.exp(-np.clip(gam * (np.log(ey) - np.log(en)), -40, 40)))
    best = min((_nelder_mead(lambda z: nll(q_gcm(z)), x0, step=0.4, max_iter=900) for x0 in ([1.0, 0.5, -3.0, -3.0], [3.0, 1.5, -6.0, -6.0], [0.0, 0.0, -1.0, -1.0])), key=lambda t: t[1])
    print(f"  exemplar model (similarity to earlier positive and negative objects, 4 parameters): trial NLL = {best[1]:.1f} ({best[1] / nv:.4f} nats/trial)")

print("=" * 100, "\nCLAIM 3: is the behaviour specifically 'single-particle'?")
hid = [[ext2h.get(e, -1) if isinstance(e, tuple) else None for e in row] for row in exts]
ws = collections.Counter(); cons_stated = []; cons_new = []; follows = []
py_rule, py_guess = [], []
for b, r in enumerate(rows):
    for j, t in enumerate(r["trials"]):
        e = exts[b][j]
        if t.get("p_yes") is not None:
            (py_guess if e == "GUESSING" else py_rule if isinstance(e, tuple) else []).append(t["p_yes"])
        if isinstance(e, tuple):
            if t["answer"] is not None: follows.append(bool(e[SX[b, j]]) == bool(t["answer"]))
            if hid[b][j] not in (None, -1): cons_stated.append(bool(consistent[b, j, hid[b][j]]))
        if j < N - 1 and isinstance(e, tuple) and isinstance(exts[b][j + 1], tuple):
            win = e[SX[b, j]] == T[b, j]; same = e == exts[b][j + 1]
            ws[("win" if win else "lose", "stay" if same else "shift")] += 1
            if not same and hid[b][j + 1] not in (None, -1): cons_new.append(bool(consistent[b, j + 1, hid[b][j + 1]]))
rate = lambda a, b_: ws[a] / (ws[a] + ws[b_])
print(f"  model: stays after a confirmed prediction {rate(('win','stay'),('win','shift')):.3f} (n={ws[('win','stay')]+ws[('win','shift')]}); "
      f"shifts after a falsified one {rate(('lose','shift'),('lose','stay')):.3f} (n={ws[('lose','stay')]+ws[('lose','shift')]})")
rng = np.random.default_rng(0)
def simulate(kind):
    c = collections.Counter(); distinct = 0
    for b in range(B):
        seq = []; cur = None
        for j in range(N):
            p = P_ideal[b, j]
            if kind == "best rule of the ideal learner": cur = int(np.argmax(p))
            elif kind == "fresh posterior sample every trial": cur = int(rng.choice(H, p=p))
            elif cur is None or (j > 0 and E[cur, SX[b, j - 1]] != T[b, j - 1]): cur = int(rng.choice(H, p=p))   # keep until falsified
            seq.append(cur)
        for j in range(1, N - 1):
            win = E[seq[j], SX[b, j]] == T[b, j]
            c[("win" if win else "lose", "stay" if seq[j] == seq[j + 1] else "shift")] += 1
        distinct += len(set(seq))
    return c[("win", "stay")] / (c[("win", "stay")] + c[("win", "shift")]), c[("lose", "shift")] / max(1, c[("lose", "shift")] + c[("lose", "stay")]), distinct / B
for kind in ("best rule of the ideal learner", "one rule kept until falsified, then resampled", "fresh posterior sample every trial"):
    s, l, d = simulate(kind)
    print(f"  simulated, {kind:46s}: stays {s:.3f}, shifts {l:.3f}, distinct rules per block {d:.2f}")
print(f"  model: distinct stated rules per block {np.mean([len({e for e in row if isinstance(e, tuple)}) for row in exts]):.2f}")
print(f"  stated rule agrees with ALL earlier feedback: {np.mean(cons_stated):.3f} (n={len(cons_stated)}); a newly adopted rule does: {np.mean(cons_new):.3f} (n={len(cons_new)})")
print(f"  answer follows the stated rule: {np.mean(follows):.3f} (n={len(follows)})")
ext_ = lambda a: float(np.mean((np.asarray(a) <= 0.05) | (np.asarray(a) >= 0.95)))
print(f"  answer probability at 0/1 extremes: {ext_(py_rule):.3f} when a rule was written (n={len(py_rule)}); {ext_(py_guess):.3f} when the model wrote 'guessing' (n={len(py_guess)})")
q3 = ev.predict_q_yes(*FIT3)
best_only = np.take_along_axis(ev.ext_bt, P_ideal.argmax(2)[:, :, None], axis=2)[:, :, 0]
print(f"  an ideal learner that reports only its best rule scores r = {r_(best_only, q3):.3f}, RMSE = {rmse_(best_only, q3):.3f} against the fitted model's P(yes)")

print("-" * 100, "\n  multi-hypothesis pilot (data/bcl/pilot_multi_hypothesis.json)")
pilot = json.load(open("data/bcl/pilot_multi_hypothesis.json"))
bid = {r["block_id"]: i for i, r in enumerate(rows)}
exp_u = [np.mean([len(set(rng.choice(H, size=8, p=P_ideal[bid[p["block_id"]], p["trial"] - 1]))) for _ in range(200)]) for p in pilot]
print(f"  {len(pilot)} trials from {len({p['block_id'] for p in pilot})} blocks; distinct rules in 8 samples: model {np.mean([p['m1_unique_rules'] for p in pilot]):.2f}, "
      f"eight draws from the ideal learner's posterior {np.mean(exp_u):.2f}")
print(f"  two or more distinct rules in the 8 samples on {sum(p['m1_unique_rules'] >= 2 for p in pilot)} of {len(pilot)} trials; "
      f"asked for its top three rules, the model lists three on {sum(p['m2_n_rules'] == 3 for p in pilot)} of {len(pilot)}")
cons3 = collections.defaultdict(list)
for p in pilot:
    b, j = bid[p["block_id"]], p["trial"] - 1
    for rank, text, _ in p["m2_items"]:
        e = parse_rule_to_ext(text, stim, feats)
        if isinstance(e, tuple): cons3[rank].append(all(e[SX[b, i]] == T[b, i] for i in range(j)))
print("  share of the listed rules that agree with all earlier feedback: " + ", ".join(f"rule {k} {np.mean(v):.3f} (n={len(v)})" for k, v in sorted(cons3.items())))
pb = sorted({p["block_id"] for p in pilot}); pidx = {b: [i for i, p in enumerate(pilot) if p["block_id"] == b] for b in pb}
base_ok = np.array([bool(p["base_correct"]) for p in pilot]); top3_ok = np.array([bool(p["m2_correct"]) for p in pilot])
bs = [top3_ok[s_].mean() - base_ok[s_].mean() for s_ in (np.concatenate([pidx[b] for b in rng.choice(pb, len(pb))]) for _ in range(4000))]
print(f"  accuracy: usual format {base_ok.mean():.3f}, top-three format {top3_ok.mean():.3f}; difference {top3_ok.mean() - base_ok.mean():+.3f}, "
      f"95% interval from resampling the {len(pb)} blocks [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
