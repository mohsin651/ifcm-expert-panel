# Results summary — derm

Protocol: `derm_official_index_split`. New results, distinct from the historical report. Diagnosis AUROC uses 395 held-out cases; Derm four-class accuracy uses 355 cases.

A/B1/B2 diagnosis AUROC: 0.822 / 0.802 / 0.775. A–B1 paired difference +0.020 [+0.007, +0.035]; A–B2 +0.047 [+0.024, +0.072]. Intervals come from 5,000 case-level resamples.

A had 93 errors, including 37 operational perception errors. Attribution scores and paired intervals are in the saved CSV tables; they are exploratory and not clinical error explanations.

Prespecified B1 zero-finding-hesitancy ablation AUROC: 0.796; the B2 zero-edge-width limit exactly reproduces B1. See the ablation and edge-quality tables for denominators.

A paired interval crossing zero does not establish a difference. The three variants were not selected by these held-out scores.
