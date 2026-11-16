## Per-Rule Independent Binary `[Layer -> 0/1]` Decoders (5-Fold Block CV)

| Target Concept | Complexity | `n_pos` | `L00` AUROC | `L >= 0.90` | Peak Layer | Peak BalAcc | Peak AUROC | `L19` AUROC | `L30` AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `red` | single | 510 | 0.5000 | L16 | L22 | 94.49% | **0.9861** | 0.9835 | 0.9816 |
| `blue` | single | 218 | 0.5000 | L16 | L28 | 98.64% | **0.9973** | 0.9965 | 0.9973 |
| `square` | single | 188 | 0.5000 | L17 | L29 | 97.98% | **0.9968** | 0.9927 | 0.9960 |
| `circle` | single | 161 | 0.5000 | L16 | L28 | 99.52% | **1.0000** | 0.9996 | 0.9998 |
| `not red` | negation | 122 | 0.5000 | L03 | L29 | 99.68% | **1.0000** | 0.9995 | 0.9998 |
| `not green` | negation | 132 | 0.5000 | L12 | L22 | 99.45% | **0.9999** | 0.9964 | 0.9967 |
| `red and square` | conjunction | 126 | 0.5000 | L12 | L29 | 98.28% | **0.9993** | 0.9974 | 0.9993 |
| `red or square` | disjunction | 140 | 0.5000 | L17 | L29 | 92.96% | **0.9880** | 0.9856 | 0.9838 |
| `blue or square` | disjunction | 133 | 0.5000 | L17 | L30 | 90.66% | **0.9756** | 0.9534 | 0.9756 |
| `red and not square` | compound | 27 | 0.5000 | L17 | L24 | 95.78% | **0.9989** | 0.9822 | 0.9764 |

## Controlled Same-Pair Unstated Second-Best Rule Test at `L19`

| Orthogonal | Stated Rule (`A`) | Probed 2nd Rule (`B`) | `n_live` | `n_falsified` | Mean `p(B)` Live | Mean `p(B)` Falsified | `Delta p` | Within-Pair AUROC |
| :---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Yes | `square` | `red` | 6 | 286 | 0.288 | 0.024 | +0.264 | **0.9580** |
| Yes | `red` | `circle` | 18 | 637 | 0.128 | 0.010 | +0.118 | **0.8741** |
| Yes | `blue` | `circle` | 7 | 281 | 0.199 | 0.033 | +0.167 | **0.8226** |
| Yes | `blue` | `square` | 7 | 281 | 0.013 | 0.018 | -0.006 | **0.7184** |
| No | `red or square` | `red and square` | 50 | 156 | 0.538 | 0.016 | +0.523 | **0.9945** |
| No | `square` | `red and square` | 10 | 282 | 0.459 | 0.044 | +0.415 | **0.9730** |
| No | `red and square` | `blue or square` | 7 | 151 | 0.169 | 0.018 | +0.152 | **0.9697** |
| No | `blue or square` | `blue` | 15 | 134 | 0.729 | 0.088 | +0.641 | **0.9637** |
| No | `red` | `red and square` | 23 | 632 | 0.593 | 0.074 | +0.519 | **0.9594** |
| No | `square` | `not green` | 7 | 285 | 0.119 | 0.015 | +0.104 | **0.9504** |
| No | `blue` | `not red` | 20 | 268 | 0.151 | 0.007 | +0.144 | **0.9450** |
| No | `red` | `not green` | 22 | 633 | 0.285 | 0.010 | +0.276 | **0.9415** |
| No | `red and square` | `red` | 32 | 126 | 0.592 | 0.091 | +0.501 | **0.9256** |
| No | `blue` | `not green` | 33 | 255 | 0.190 | 0.027 | +0.163 | **0.9078** |
| No | `blue` | `red and square` | 6 | 282 | 0.102 | 0.031 | +0.071 | **0.9054** |
| No | `red or square` | `red` | 66 | 140 | 0.458 | 0.086 | +0.372 | **0.8956** |

Evaluated strictly out-of-block across **4,560** pure-induction trials (`285` independent blocks, `include_rule=False`) extracted from the `gemma4:26b` residual stream (`d_model = 2816`) at `Rule:` before any scratchpad text is generated.
