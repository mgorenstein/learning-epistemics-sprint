| Model | Params (`k`) | `alpha` | `beta` | `prior_temp` | Trial NLL | BIC | Pred. Acc | Curve RMSE (16) | 160-pt `r` (RMSE) | LOCO NLL | LOCO 160-pt `r` (RMSE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Normative ideal (argmax) | 0 | 0.990 | ∞ | 1.000 | 91707.4 | 183414.8 | 87.5% | 0.122 | 0.761 (0.158) | 91707.4 | 0.761 (0.158) |
| Prob-matching ideal (beta=1) | 0 | 0.990 | 1.000 | 1.000 | 9205.6 | 18411.2 | 83.7% | 0.088 | 0.859 (0.115) | 9205.6 | 0.859 (0.115) |
| 1-param: likelihood alpha | 1 | 0.833 | 1.000 | 1.000 | 7344.1 | 14697.8 | 75.2% | 0.027 | 0.852 (0.077) | 7370.4 | 0.842 (0.079) |
| 1-param: softmax beta | 1 | 0.990 | 0.455 | 1.000 | 7298.0 | 14605.7 | 75.9% | 0.028 | 0.869 (0.073) | 7311.3 | 0.864 (0.074) |
| 2-param: alpha + beta | 2 | 1.000 | 0.328 | 1.000 | 7239.8 | 14498.9 | 75.7% | 0.032 | 0.873 (0.073) | 7264.9 | 0.866 (0.075) |
| 2-param: alpha + prior_temp | 2 | 0.927 | 1.000 | 2.438 | 7122.8 | 14264.9 | 75.7% | 0.025 | 0.884 (0.069) | 7153.3 | 0.876 (0.071) |
| 3-param: alpha + beta + prior_temp | 3 | 0.730 | 1.965 | 3.075 | 7096.2 | 14221.4 | 75.4% | 0.025 | 0.890 (0.067) | 7134.4 | 0.882 (0.069) |

Empirical `gemma4:26b` overall accuracy: **75.7%** across 15,993 valid binary responses (coin-flip baseline NLL = 11085.5). LOCO columns fit parameters on 9 concepts and evaluate out-of-sample on all 100 blocks of the held-out 10th concept.
