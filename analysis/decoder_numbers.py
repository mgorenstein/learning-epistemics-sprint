"""Sections 4.2 and 4.3 of the paper: the decoder numbers, read from the saved outputs of bcl.probe.

    python analysis/decoder_numbers.py

Run from the folder that holds config_bcl.yaml. Reads

    data/bcl/multirule_probe_diagnostics.json    per-rule, per-layer decoder scores and the tests on unstated rules
    tables/bcl/probe_trial_predictions.csv       how well each predictor predicts the model's answers (Table 2)

Both were written by `python -m bcl.probe` from the residual-stream activations, which are too large to include.
The README says how to regenerate them. The next-token check at the end comes from a one-off analysis whose code
was not kept; only its saved result is here.

Written on 2026-11-16 to print the paper's numbers from the saved files.
"""
import csv, json
import numpy as np

d = json.load(open("data/bcl/multirule_probe_diagnostics.json"))
au = np.array([[m["auroc"] for m in d["per_rule_layer_metrics"][c]] for c in d["per_rule_layer_metrics"]]).mean(0)
print(f"decoding trials: {d['n_trials_pure_induction']}")
print("mean held-out AUROC of the ten rule decoders: layers 1 to 10 %.2f to %.2f; layers 11 to 15 %.2f to %.2f; layer 16 %.2f; layer 17 %.2f; layer 19 %.2f; layers 19 to 30 at least %.3f"
      % (au[1:11].min(), au[1:11].max(), au[11:16].min(), au[11:16].max(), au[16], au[17], au[19], au[19:].min()))
r = d["rank_layer_curves"][21]["committed_to_rank1"]
print(f"layer 21, trials where the model states the most probable rule while a second rule still fits (n={r['n']}): "
      f"stated rule {r['mean_p_rank1']:.2f}, second rule {r['mean_p_rank2']:.2f}, ruled-out rules {r['mean_p_falsified']:.2f}")
print("layer 21, decoded value of an unstated rule by its probability under the fitted Bayesian learner:")
for b in d["r2_bins_summary"]:
    print(f"    {b['bin']:32s} n={b['n_all']:6d}   mean decoded value {b['mean_raw_l21']:.2f}")
nt = d["next_token_orth_r2"]
print("next-token check, four pairs of a colour rule and a shape rule: probability that the rule line begins with the second rule's word while that rule still fits, "
      f"at most {max(x['mean_next_tok_p_when_live'] for x in nt):.4f} (n = {min(x['n_live'] for x in nt)} to {max(x['n_live'] for x in nt)} trials per pair)")

rows = {r["predictor_id"]: r for r in csv.DictReader(open("tables/bcl/probe_trial_predictions.csv"))}
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan", "None") else float("nan")
print("\nTable 2. Predicting the model's yes or no answer on the decoding trials")
print(f"    {'predictor':52s} nats/trial  agreement  RMSE against the fitted learner")
for pid, name in (("single_stated_rule", "the rule the model wrote"), ("probe_top1_L19", "top decoder alone, layer 19"), ("probe_superposed_L19", "ten decoders pooled, layer 19"),
                  ("bayes_norm_pm", "ideal learner, nothing fitted"), ("bayes_fit_3p", "Bayesian learner, three fitted parameters"),
                  ("probe_superposed_L19_loco_cal", "pooled, layer 19, two scaling parameters fitted on other rules"), ("probe_superposed_loco_layer", "pooled, layer chosen on other rules")):
    x = rows[pid]
    print(f"    {name:52s} {f(x, 'llm_choice_mean_nll'):.3f}       {100 * f(x, 'llm_choice_agreement'):.0f}%        {f(x, 'rmse_trial_fit_bayes'):.3f}")
pool = {int(k[-2:]): f(v, "llm_choice_mean_nll") for k, v in rows.items() if k.startswith("probe_superposed_L") and k[-2:].isdigit()}
best = min(pool, key=pool.get)
print(f"    layer where the pool predicts the answers best: {best} ({pool[best]:.3f} nats per trial); chance is {np.log(2):.3f}")
