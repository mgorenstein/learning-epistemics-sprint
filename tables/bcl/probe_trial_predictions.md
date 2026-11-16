## Trial-by-Trial `P(yes)` & 160-Point Learning-Curve Predictions (`n = 4,560` Trials)

| Predictor | Params (`k`) | Trial `r` vs. Fit Bayes | Trial RMSE vs. Fit Bayes | LLM Choice Mean NLL | LLM Choice Agree | Matched 160-pt `r` (RMSE) | Full 160-pt `r` (RMSE) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Answer: token prob p_yes | 0 | 0.6428 | 0.3458 | 0.0427 | 100.0% | 0.9916 (0.0213) | 0.9260 (0.0593) |
| Binary LLM answer (yes=1, no=0) | 0 | 0.6069 | 0.3813 | 0.0000 | 100.0% | 1.0000 (0.0000) | 0.9158 (0.0653) |
| Single stated scratchpad Rule: | 0 | 0.6520 | 0.3194 | 0.6575 | 82.9% | 0.8790 (0.0782) | 0.8473 (0.0847) |
| Top-1 winner-take-all probe (L19) | 0 | 0.7674 | 0.2423 | 0.6267 | 78.2% | 0.7974 (0.1083) | 0.8424 (0.0905) |
| Superposed 10-rule probe (L00) | 0 | 0.2494 | 0.3034 | 0.6465 | 62.3% | 0.1795 (0.2368) | 0.1666 (0.2343) |
| Superposed 10-rule probe (L10) | 0 | 0.3439 | 0.2953 | 0.6420 | 65.2% | 0.2222 (0.2195) | 0.2163 (0.2159) |
| Superposed 10-rule probe (L16) | 0 | 0.5800 | 0.2616 | 0.6253 | 72.7% | 0.3848 (0.1777) | 0.3930 (0.1712) |
| **Superposed 10-rule probe (L19)** | 0 | 0.7872 | 0.2063 | 0.5138 | 77.6% | 0.7984 (0.0977) | 0.8442 (0.0783) |
| Superposed 10-rule probe (L22) | 0 | 0.8152 | 0.1970 | 0.5153 | 78.2% | 0.7978 (0.1002) | 0.8462 (0.0802) |
| Superposed 10-rule probe (L30) | 0 | 0.7758 | 0.2193 | 0.5334 | 76.3% | 0.7575 (0.1070) | 0.8082 (0.0869) |
| **Superposed 10-rule probe (LOCO-selected layer, 0 params)** | 0 | 0.8030 | 0.2010 | 0.5268 | 77.9% | 0.7884 (0.1018) | 0.8340 (0.0814) |
| **LOCO-calibrated superposed probe (L19, gamma+tau_0)** | 2 | 0.7777 | 0.1952 | 0.4977 | 77.2% | 0.7901 (0.1039) | 0.8371 (0.0887) |
| Normative Bayes (prob-matching) | 0 | 0.9494 | 0.1419 | 0.5868 | 79.0% | 0.7706 (0.1416) | 0.8356 (0.1209) |
| 3-param Rational Rules Bayes | 3 | 1.0000 | 0.0000 | 0.4498 | 78.6% | 0.8075 (0.0960) | 0.8633 (0.0753) |

Zero-parameter superposed marginalization combines the 10 independent out-of-block binary rule decoders via `q_yes(x_t) = (sum_c p_c(h_t) c(x_t) + 0.5 (1 - max_c p_c(h_t))) / (sum_c p_c(h_t) + (1 - max_c p_c(h_t)))` before any `Rule:` or `Answer:` token is generated. Coin-flip choice NLL = `0.6931`.
