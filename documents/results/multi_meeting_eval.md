# Multi-meeting replication of the noise-robustness result

Evidence F1@5 per AMI meeting and noise condition, reliability gate off (β = 0, original LISTEN) vs the learned gate (β = 0.785). Produced by `scripts/aggregate_uncertainty_eval.py` from the per-meeting `*_uncertainty_eval.json` files in this folder.

| meeting | queries | clean β=0 | clean β=0.785 | 10 dB β=0 | 10 dB β=0.785 | 5 dB β=0 | 5 dB β=0.785 | 0 dB β=0 | 0 dB β=0.785 |
|---|---|---|---|---|---|---|---|---|---|
| ES2002a | 6 | 0.133 | **0.140** (+5%) | 0.084 | **0.124** (+48%) | 0.087 | **0.117** (+35%) | 0.099 | 0.053 (-46%) |
| ES2002b | 6 | 0.199 | 0.172 (-14%) | 0.036 | 0.032 (-11%) | 0.012 | 0.008 (-33%) | 0.010 | 0.004 (-59%) |
| ES2002c | 5 | 0.061 | 0.061 (+0%) | 0.130 | **0.136** (+5%) | 0.154 | 0.116 (-24%) | 0.099 | 0.025 (-75%) |
| ES2002d | 6 | 0.035 | 0.017 (-53%) | 0.029 | 0.002 (-93%) | 0.004 | 0.004 (+0%) | 0.002 | **0.004** (+100%) |
| ES2003a | 6 | 0.204 | 0.106 (-48%) | 0.135 | 0.070 (-48%) | 0.091 | **0.093** (+3%) | 0.061 | **0.074** (+22%) |
| ES2003b | 6 | 0.067 | 0.042 (-38%) | 0.006 | **0.028** (+333%) | 0.032 | **0.067** (+110%) | 0.000 | 0.000 (n/a) |
| ES2003c | 5 | 0.062 | 0.050 (-19%) | 0.040 | 0.040 (+0%) | 0.040 | 0.000 (-100%) | 0.090 | 0.012 (-87%) |
| **mean of meetings** | | 0.109 | 0.084 (-23%) | 0.066 | 0.062 (-6%) | 0.060 | 0.058 (-3%) | 0.051 | 0.025 (-52%) |
| **pooled over queries** | 40 | 0.111 | 0.085 (-23%) | 0.065 | 0.060 (-7%) | 0.058 | 0.058 (-0%) | 0.049 | 0.025 (-50%) |

Bold = gate beats the ungated model for that meeting and condition. Percentages are relative to β = 0.

## Pooled gain of the gate (β = 0.785 − β = 0), paired over queries

| condition | queries | F1 β=0 | F1 gate | mean gain | bootstrap 95 % CI | queries up / down / tied |
|---|---|---|---|---|---|---|
| clean | 40 | 0.111 | 0.085 | -0.026 | [-0.053, -0.002] | 1 / 10 / 29 |
| 10 dB | 40 | 0.065 | 0.060 | -0.005 | [-0.028, +0.019] | 5 / 6 / 29 |
| 5 dB | 40 | 0.058 | 0.058 | -0.000 | [-0.026, +0.025] | 5 / 6 / 29 |
| 0 dB | 40 | 0.049 | 0.025 | -0.024 | [-0.053, +0.002] | 5 / 8 / 27 |

Meetings where the gate helps / hurts / total: clean: 1 / 5 / 7; 10 dB: 3 / 3 / 7; 5 dB: 3 / 3 / 7; 0 dB: 2 / 4 / 7.

## Pooled evidence F1 for every β evaluated

| condition | queries | β=0.0 | β=0.5 | β=0.785 | β=1.0 |
|---|---|---|---|---|---|
| clean | 40 | **0.111** | 0.086 | 0.085 | 0.085 |
| 10 dB | 40 | 0.065 | **0.066** | 0.060 | 0.060 |
| 5 dB | 40 | **0.058** | 0.056 | 0.058 | 0.057 |
| 0 dB | 40 | **0.049** | 0.041 | 0.025 | 0.026 |

Bold = best β for that condition.

## Reliability signal: gold vs non-gold candidate segments

Mean node reliability of gold evidence segments vs the other retrieved candidates, per meeting (query means) and pooled.

| meeting | clean gold / non-gold | 10 dB gold / non-gold | 5 dB gold / non-gold | 0 dB gold / non-gold |
|---|---|---|---|---|
| ES2002a | 0.527 / 0.526 | 0.459 / 0.363 | 0.349 / 0.274 | 0.230 / 0.218 |
| ES2002b | 0.540 / 0.541 | 0.474 / 0.428 | 0.341 / 0.340 | 0.208 / 0.222 |
| ES2002c | 0.456 / 0.490 | 0.432 / 0.358 | 0.325 / 0.280 | 0.193 / 0.200 |
| ES2002d | 0.479 / 0.526 | 0.274 / 0.339 | 0.250 / 0.248 | 0.126 / 0.164 |
| ES2003a | 0.578 / 0.597 | 0.303 / 0.384 | 0.296 / 0.269 | 0.197 / 0.182 |
| ES2003b | 0.585 / 0.575 | 0.317 / 0.344 | 0.242 / 0.247 | 0.193 / 0.191 |
| ES2003c | 0.475 / 0.505 | 0.249 / 0.259 | 0.114 / 0.196 | 0.142 / 0.168 |
| **pooled** | 0.533 / 0.539 (gap -0.006) | 0.373 / 0.356 (gap +0.018) | 0.288 / 0.266 (gap +0.022) | 0.191 / 0.193 (gap -0.002) |

## Mean word confidence per condition

| meeting | clean | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| ES2002a | 0.636 | 0.428 | 0.352 | 0.256 |
| ES2002b | 0.613 | 0.356 | 0.279 | 0.192 |
| ES2002c | 0.794 | 0.593 | 0.488 | 0.345 |
| ES2002d | 0.667 | 0.431 | 0.323 | 0.228 |
| ES2003a | 0.682 | 0.494 | 0.421 | 0.339 |
| ES2003b | 0.677 | 0.441 | 0.347 | 0.262 |
| ES2003c | 0.789 | 0.503 | 0.399 | 0.350 |
