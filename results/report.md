| Model | Prompt | Docs | Macro accuracy % [95% CI] | Macro F1 % [95% CI] | Hallucination rate % [95% CI] | Parse ok / repaired / failed | Median latency s |
|---|---|---:|---:|---:|---:|---:|---:|
| *always `null` (baseline)* | – | 50 | 54.0 [50.5, 57.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | – | – |
| `ollama_chat/gemma4:31b-cloud` | v1 | 50 | 96.8 [94.8, 98.5] | 93.2 [87.4, 97.4] | 3.7 [1.4, 6.5] | 0 / 50 / 0 | 2.0 |
| `ollama_chat/gemma4:31b-cloud` | v2 | 50 | 97.2 [95.0, 99.0] | 94.2 [87.4, 98.8] | 1.9 [0.5, 3.7] | 0 / 50 / 0 | 1.8 |
| `ollama_chat/qwen2.5vl:3b` | v1 | 50 | 49.5 [43.8, 55.8] | 54.4 [49.7, 58.7] | 88.0 [79.7, 95.0] | 0 / 50 / 0 | 4.2 |
| `ollama_chat/qwen2.5vl:3b` | v2 | 50 | 86.0 [80.5, 91.0] | 78.9 [71.3, 85.2] | 23.1 [14.1, 32.6] | 31 / 19 / 0 | 4.3 |

No output used a key outside the schema.

Per-field accuracy % (exact match, nulls included):

| Field | always null | gemma4:31b-cloud v1 | gemma4:31b-cloud v2 | qwen2.5vl:3b v1 | qwen2.5vl:3b v2 |
|---|---:|---:|---:|---:|---:|
| total_price | 6.0 | 92.0 | 94.0 | 94.0 | 94.0 |
| subtotal_price | 38.0 | 96.0 | 96.0 | 62.0 | 62.0 |
| tax_price | 60.0 | 100.0 | 98.0 | 48.0 | 88.0 |
| service_price | 86.0 | 98.0 | 98.0 | 22.0 | 88.0 |
| discount_price | 90.0 | 92.0 | 96.0 | 22.0 | 82.0 |
| cash_price | 30.0 | 96.0 | 96.0 | 64.0 | 88.0 |
| change_price | 38.0 | 100.0 | 100.0 | 62.0 | 96.0 |
| credit_card_price | 84.0 | 100.0 | 100.0 | 22.0 | 90.0 |

Paired comparison v1 -> v2 for `ollama_chat/gemma4:31b-cloud` (50 docs): macro accuracy +0.5 pts [-0.8, +2.0], macro F1 +1.0 pts [-1.3, +4.0].

| Field | Δ accuracy pts | McNemar p | Holm-adjusted p |
|---|---:|---:|---:|
| total_price | +2.0 | 1.000 | 1.000 |
| subtotal_price | +0.0 | 1.000 | 1.000 |
| tax_price | -2.0 | 1.000 | 1.000 |
| service_price | +0.0 | 1.000 | 1.000 |
| discount_price | +4.0 | 0.500 | 1.000 |
| cash_price | +0.0 | 1.000 | 1.000 |
| change_price | +0.0 | 1.000 | 1.000 |
| credit_card_price | +0.0 | 1.000 | 1.000 |

Paired comparison v1 -> v2 for `ollama_chat/qwen2.5vl:3b` (50 docs): macro accuracy +36.5 pts [+30.3, +42.2], macro F1 +24.5 pts [+18.1, +30.4].

| Field | Δ accuracy pts | McNemar p | Holm-adjusted p |
|---|---:|---:|---:|
| total_price | +0.0 | 1.000 | 1.000 |
| subtotal_price | +0.0 | 1.000 | 1.000 |
| tax_price | +40.0 | <0.001 | <0.001 |
| service_price | +66.0 | <0.001 | <0.001 |
| discount_price | +60.0 | <0.001 | <0.001 |
| cash_price | +24.0 | <0.001 | 0.001 |
| change_price | +34.0 | <0.001 | <0.001 |
| credit_card_price | +68.0 | <0.001 | <0.001 |
