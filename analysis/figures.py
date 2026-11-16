"""The two figures of the paper, built from files in this folder.

    python analysis/figures.py [output_dir]      (default: figures)

Run from the folder that holds config_bcl.yaml.

  fig_task_and_learning.png   Figure 1. A-C: the task as run with people, cropped from figures/dissertation_task_figure.png
                              (panels A to C of a figure in M. A. Gorenstein's unpublished dissertation);
                              D: the model's accuracy by trial for each rule, with the fitted Bayesian learner
  fig_decoders.png            Figure 2. A: held-out AUROC of the ten rule decoders by layer; B: decoded value of the stated
                              rule, of an unstated second rule, and of ruled-out rules. The reference lines in panel A
                              come from analysis/evidence_tally_control.json (written by evidence_tally_control.py).

Needs matplotlib and Pillow. The residual-stream curves come from data/bcl/multirule_probe_diagnostics.json.
Written on 2026-11-15, during the sprint.
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.getcwd())
from bcl.config import load_config
from bcl.fit import _HypothesisEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
MODEL, IDEAL, BASE, INK, GRID, SECOND = "#2a78d6", "#1a9a6c", "#8a8a86", "#222222", "#e6e6e3", "#b8791f"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans"})
rng = np.random.default_rng(0)
cfg = load_config("config_bcl.yaml")
rows = [json.loads(l) for l in open("data/bcl/results.jsonl")]
ev = _HypothesisEvaluator(cfg, rows)
T, C = ev.truth, ev.correct
N = C.shape[1]; trials = np.arange(1, N + 1)
FIT3 = (0.72954, 1.96550, 3.07512)                        # alpha, beta, prior temperature from tables/bcl/parameter_fits.csv
qf = ev.predict_q_yes(*FIT3); pf = np.where(T == 1.0, qf, 1 - qf)

def boot(per_block, n=400):
    idx = rng.integers(0, per_block.shape[0], size=(n, per_block.shape[0]))
    m = np.nanmean(per_block[idx], axis=1)
    return np.nanpercentile(m, 2.5, axis=0), np.nanpercentile(m, 97.5, axis=0)

# ------------------------------------------------------------------ figure 1: the task, and learning by rule
src = next((p for p in (os.path.join(OUT, "dissertation_task_figure.png"), "figures/dissertation_task_figure.png") if os.path.exists(p)), None)
fig = plt.figure(figsize=(8.8, 5.5))
if src:
    page = Image.open(src).convert("RGB"); W, H = page.size
    crops = {"A": (0.030, 0.045, 0.330, 0.385), "B": (0.475, 0.045, 0.970, 0.385), "C": (0.030, 0.400, 0.420, 0.740)}   # left, top, right, bottom as fractions of the page
    total = sum(r - l for l, t, r, b in crops.values()); x0 = 0.02
    for k, (l, t, r, b) in crops.items():                 # the three crops share one height, so they are drawn at one scale
        w = 0.96 * (r - l) / total
        ax = fig.add_axes([x0, 0.645, w, 0.33]); ax.imshow(page.crop((int(l * W), int(t * H), int(r * W), int(b * H)))); ax.axis("off"); x0 += w
else:
    print("figures/dissertation_task_figure.png not found: panels A to C left out")
gs = fig.add_gridspec(2, 5, left=0.065, right=0.985, top=0.525, bottom=0.09, wspace=0.14, hspace=0.42)
for k, c in enumerate(ev.concept_order):
    ax = fig.add_subplot(gs[k // 5, k % 5]); m = ev.concepts == c
    lo, hi = boot(C[m]); ax.fill_between(trials, lo, hi, color=MODEL, alpha=0.18, lw=0)
    ax.axhline(0.5, color=BASE, lw=0.8, ls=":")
    ax.plot(trials, C[m].mean(0), color=MODEL, lw=1.6); ax.plot(trials, pf[m].mean(0), color=INK, lw=1.2, ls="--")
    ax.set_xlim(1, 16); ax.set_ylim(0.25, 1.02); ax.set_xticks([1, 8, 16]); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(c, fontsize=8.5, fontweight="normal", pad=3); ax.tick_params(labelsize=7.5)
    if k % 5: ax.set_yticklabels([])
    if k // 5 == 0: ax.set_xticklabels([])
    else: lab = ax.get_xticklabels(); lab[0].set_ha("left"); lab[-1].set_ha("right")
    if k == 0: ax.text(15.6, 0.30, "gemma4:26b", color=MODEL, fontsize=7.5, ha="right"); ax.text(15.6, 0.42, "Bayesian learner", color=INK, fontsize=7.5, ha="right")
fig.text(0.010, 0.31, "Proportion correct", rotation=90, va="center", fontsize=9); fig.text(0.525, 0.012, "Trial number", ha="center", fontsize=9)
fig.text(0.030, 0.585, "D.", fontsize=13); fig.text(0.075, 0.585, "Model performance, by rule", fontsize=11.5)
fig.savefig(os.path.join(OUT, "fig_task_and_learning.png")); plt.close(fig)
print("wrote", os.path.join(OUT, "fig_task_and_learning.png"))

# ------------------------------------------------------------------ figure 2: rule decoders by layer
diag_p = "data/bcl/multirule_probe_diagnostics.json"
ctrl_p = next((p for p in (os.path.join(OUT, "evidence_tally_control.json"), os.path.join(HERE, "evidence_tally_control.json")) if os.path.exists(p)), None)
d = json.load(open(diag_p)); ctl = json.load(open(ctrl_p)) if ctrl_p else None
au = np.array([[m["auroc"] for m in d["per_rule_layer_metrics"][c]] for c in d["per_rule_layer_metrics"]])    # rules x layers
L = np.arange(au.shape[1])
amb = [x["committed_to_rank1"] for x in d["rank_layer_curves"]]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
ax = axes[0]
for row in au: ax.plot(L, row, color=MODEL, lw=0.7, alpha=0.35)
ax.plot(L, au.mean(0), color=MODEL, lw=2.2)
ax.text(30, 0.855, "residual stream:\nten decoders, mean", color=MODEL, fontsize=8.5, va="center", ha="right")
if ctl:
    ref = {k: float(np.mean(list(ctl["auroc_by_rule"][k].values()))) for k in ("object table", "label counts")}
    ax.axhline(ref["object table"], color=IDEAL, lw=1.5, ls="--"); ax.axhline(ref["label counts"], color=BASE, lw=1.2, ls=":")
    ax.text(0.6, 0.985, "record of the feedback, no model", color=IDEAL, fontsize=8.5, va="top", ha="left")
    ax.text(30, ref["label counts"] - 0.012, "counts of yes and no only", color=BASE, fontsize=8.5, va="top", ha="right")
ax.set_xlim(0, 30); ax.set_ylim(0.4, 1.02); ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Layer"); ax.set_ylabel("Held-out AUROC"); ax.set_title("A. Which rule fits: from layer 17", fontsize=10)
ax = axes[1]
series = (("mean_p_rank1", "the rule the model states", MODEL), ("mean_p_rank2", "unstated second rule", SECOND), ("mean_p_falsified", "ruled-out rules", BASE))
for k, name, colr in series:
    y = np.array([a[k] for a in amb], float)
    ax.plot(L[1:], y[1:], color=colr, lw=2.0)             # layer 0 is the same token on every trial, so it carries nothing
for (k, name, colr), dy in zip(series, (0.055, 0.075, 0.045)): ax.text(30, float(amb[-1][k]) + dy, name, color=colr, fontsize=8.5, ha="right")
ax.set_xlim(0, 30); ax.set_ylim(0, 1.0); ax.set_xticks([0, 5, 10, 15, 20, 25, 30]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("Layer"); ax.set_ylabel("Decoded probability of the rule")
ax.set_title(f"B. A second rule is readable too (n={amb[-1]['n']})", fontsize=10)
fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.155, wspace=0.26)
fig.savefig(os.path.join(OUT, "fig_decoders.png")); plt.close(fig)
print("wrote", os.path.join(OUT, "fig_decoders.png"), "| layer 21:", {k: round(float(amb[21][k]), 3) for k, _, _ in series}, "| mean AUROC layers 16, 17, 19:", np.round(au.mean(0)[[16, 17, 19]], 3).tolist())
