| Model | Prompt | Docs | Macro accuracy % [95% CI] | Macro F1 % [95% CI] | Hallucination rate % | Parse ok / repaired / failed | Median latency s |
|---|---|---:|---:|---:|---:|---:|---:|
| `ollama_chat/gemma4:31b-cloud` | v1 | 50 | 96.8 [94.8, 98.5] | 93.2 [87.4, 97.4] | 3.7 | 0 / 50 / 0 | 2.0 |
| `ollama_chat/gemma4:31b-cloud` | v2 | 50 | 97.2 [95.0, 99.0] | 94.2 [87.4, 98.8] | 1.9 | 0 / 50 / 0 | 1.8 |

Per-field accuracy % (exact match, nulls included):

| Field | gemma4:31b-cloud v1 | gemma4:31b-cloud v2 |
|---|---:|---:|
| total_price | 92.0 | 94.0 |
| subtotal_price | 96.0 | 96.0 |
| tax_price | 100.0 | 98.0 |
| service_price | 98.0 | 98.0 |
| discount_price | 92.0 | 96.0 |
| cash_price | 96.0 | 96.0 |
| change_price | 100.0 | 100.0 |
| credit_card_price | 100.0 | 100.0 |

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
