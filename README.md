# Hidden Hypotheses

Code and data for the report *Hidden Hypotheses: Measuring a Language Model's Reports and Internal States Against an
Ideal Bayesian Learner*, written for the AI x Epistemics Research Sprint (Apart Research, November 2026).

A language model, `gemma4:26b`, learns Boolean rules such as "blue or square" by trial and error. On each trial it sees
an object, writes its current rule, answers yes or no, and is told the right answer. We compare its answers with a
Bayesian learner (Rational Rules; Goodman, Tenenbaum, Feldman & Griffiths, 2008), and train linear decoders on its
residual stream to read out which rules fit the feedback so far, including rules the model does not state.

## Preexisting work

The core of the Bayesian learner (`rule_learner.py`) and the task generator build on code from the author's PhD
research, and panels A to C of Figure 1 are from the author's dissertation.

## Install

Python 3.10 or later.

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Reproduce the paper's numbers and figures

Run from this folder. None of these steps needs the model; they read its saved answers in `data/`.

| What | Command |
| --- | --- |
| Behavioural results (sections 3.1 to 3.3) | `.venv/bin/python analysis/behaviour_checks.py --exemplar` |
| With and without the written rule (section 3.1) | `.venv/bin/python analysis/answer_only_comparison.py` |
| Fit of the Bayesian learner (Table 1); slow | `.venv/bin/python -m bcl.fit --config config_bcl.yaml` |
| Decoder results (section 4, Table 2) | `.venv/bin/python analysis/decoder_numbers.py` |
| Figures 1 and 2 | `.venv/bin/python analysis/figures.py` |
| Tests | `.venv/bin/python -m pytest` |

To run the model again, install [Ollama](https://ollama.com) with `gemma4:26b` and use `python -m bcl.blocks` and
`python -m bcl.run`; settings are in `config_bcl.yaml`. The residual-stream activations are too large to include;
`tools/extract_gemma4_acts.cpp` and `python -m bcl.probe` regenerate them and the decoders.

## Layout

| Path | What it is |
| --- | --- |
| `rule_learner.py` | Rules as Python programs and the Bayesian learner over them |
| `bcl/` | The task: blocks, prompts, runner, scoring, model fit, and decoders |
| `analysis/` | Scripts that print the paper's numbers and draw its figures |
| `data/`, `tables/` | The model's answers (16,000 trials), saved decoder results, and result tables |
| `figures/` | The paper's two figures |
| `tools/`, `reference/` | The activation extractor, and reference values the tests check against |
| `hh/`, `config.yaml` | A model backend and settings loader used by the runner and the tests |
