# llm-doc-extraction-eval

[![CI](https://github.com/ricca200xx/llm-doc-extraction-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/ricca200xx/llm-doc-extraction-eval/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small, strict evaluation harness for vision-language models that read receipt
images and return structured fields. It runs on the public
[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset, swaps
model backends through [LiteLLM](https://github.com/BerriAI/litellm), and reports
every number with a confidence interval and against a do-nothing baseline.

**TL;DR of the committed run** (50 receipts, 8 fields, 2 models, 2 prompts):

- 54% of the field cells have no value on the receipt, so a "model" that always
  answers `null` already scores **54.0%** macro accuracy. A local 3B model with a
  naive prompt scores **49.5%**: below doing nothing, because it invents values.
- A stricter prompt lifts that model to **86.0%** (paired +36.5 points, 95% CI
  +30.3 to +42.2) and cuts its hallucination rate from 88% to 23%.
- The same prompt has **no measurable effect** on a 31B model that was already
  at 97% (+0.5 points, 95% CI -0.8 to +2.0).

The point is the measurement, not the model:

- a **scorer that does not reward guessing**: a value invented for a field that
  is not on the receipt is a false positive, and a failed call earns nothing;
- a **JSON repair layer that never invents values**: output cut off by the token
  limit loses the half-written field instead of keeping a wrong one;
- **statistics that say when a change is real**: comparisons are paired by
  document and corrected for testing 8 fields at once.

## Contents

- [Results](#results)
- [Quickstart](#quickstart)
- [How it works](#how-it-works): dataset, schema, prompts, extraction, JSON
  repair, scoring, statistics
- [Usage](#usage): CLI, other models, where results go, extending the harness
- [Reproducibility](#reproducibility)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Data, license and citation](#data-license-and-citation)

## Results

Two models on the same 50 receipts (the first 50 of the CORD-v2 test split),
8 fields each: 400 field cells per run, 184 with a gold value and 216 where the
right answer is `null`.

- `gemma4:31b-cloud`: a 31B vision model served by Ollama Cloud.
- `qwen2.5vl:3b`: a 3B vision model running locally through Ollama on an Apple
  M4 Pro laptop.

Both go through the same LiteLLM call; only the model string differs.

| Model | Prompt | Docs | Macro accuracy % [95% CI] | Macro F1 % [95% CI] | Hallucination rate % [95% CI] | Parse ok / repaired / failed | Median latency s |
|---|---|---:|---:|---:|---:|---:|---:|
| *always `null` (baseline)* | – | 50 | 54.0 [50.5, 57.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | – | – |
| `ollama_chat/gemma4:31b-cloud` | v1 | 50 | 96.8 [94.8, 98.5] | 93.2 [87.4, 97.4] | 3.7 [1.4, 6.5] | 0 / 50 / 0 | 2.0 |
| `ollama_chat/gemma4:31b-cloud` | v2 | 50 | 97.2 [95.0, 99.0] | 94.2 [87.4, 98.8] | 1.9 [0.5, 3.7] | 0 / 50 / 0 | 1.8 |
| `ollama_chat/qwen2.5vl:3b` | v1 | 50 | 49.5 [43.8, 55.8] | 54.4 [49.7, 58.7] | 88.0 [79.7, 95.0] | 0 / 50 / 0 | 4.2 |
| `ollama_chat/qwen2.5vl:3b` | v2 | 50 | 86.0 [80.5, 91.0] | 78.9 [71.3, 85.2] | 23.1 [14.1, 32.6] | 31 / 19 / 0 | 4.3 |

How to read the columns:

- **Macro accuracy**: share of correct cells, a correct `null` included,
  averaged over the 8 fields.
- **Macro F1**: ignores correct `null`s, so every invented value costs F1.
- **Hallucination rate**: share of the 216 `null` cells where the model still
  returned a value.
- **Parse status**: `ok` means the raw output was valid JSON as is, `repaired`
  means the repair layer had to fix it, `failed` means nothing could be
  recovered.

![Per-field accuracy, prompt v1 vs v2, for both models](results/chart.png)

*Dots are exact-match accuracy per field; grey ticks mark the always-`null`
baseline for that field. On the macro row the horizontal lines are 95% bootstrap
confidence intervals over documents.*

### What the numbers say

- **Accuracy needs a baseline.** Because most fields are absent from most
  receipts, answering `null` everywhere scores 54.0%. Qwen 3B with prompt v1
  (49.5%) is below that line overall, and below it on 4 of the 8 fields
  (tax, service, discount, card).
- **The prompt matters a lot for the small model and has no measurable effect
  on the large one.** For Qwen 3B, v2 raises macro accuracy from 49.5% to 86.0%:
  a paired difference of +36.5 points (95% CI +30.3 to +42.2), with 6 of 8
  fields significant after Holm correction (adjusted p ≤ 0.001). For Gemma 31B
  the difference is +0.5 points (CI -0.8 to +2.0), and no field changes
  significantly (smallest raw McNemar p = 0.5). Reading the Gemma point
  estimates alone would suggest an improvement that the data does not support.
- **The small model's v1 errors are mostly invented values, not misreadings.**
  With v1, Qwen puts a value in 190 of the 216 cells that should be `null`: 146
  of them are `"0"` and 21 are a copy of the total. v2 cuts that to 50, while
  correct values go from 172 to 178 of 184. v2 changes four things at once
  (system message, field definitions, the `null` rule, the no-reuse rule), so
  which of them does the work is not tested here.
- **Some rules do not stick.** v2 forbids reusing one printed number for two
  fields, yet Qwen still fills `subtotal_price` on all 19 receipts where CORD has
  no subtotal, 15 of them with an exact copy of the total. That field stays at
  62% under both prompts, and `discount_price` stays below its baseline (82% vs
  90%).
- **For the large model, v2 trades hallucinations for misses.** Invented values
  drop from 8 to 4, but correct values drop from 179 to 177: the stricter
  "use null" rule also makes Gemma abstain on two fields it used to get right.
- **The repair layer is not optional.** Gemma wrapped all 100 answers in
  markdown code fences, Qwen 50 of 50 with v1 and 19 of 50 with v2. A plain
  `json.loads` fails on 169 of the 200 outputs; the repair layer recovered all
  of them. No output was truncated and no output used a key outside the schema.
- **Gemma's remaining errors are reading errors and label conventions**, not
  format problems: `377,859` read as `377,059`, and totals reported for receipts
  where CORD labels the same printed amount as a subtotal (see
  [Limitations](#limitations)).

Per-field accuracy (with the baseline column), per-field McNemar p-values and
Holm-adjusted p-values for both comparisons are in
[`results/report.md`](results/report.md). Every raw model answer is in
[`results/raw`](results/raw).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Running a local model also requires
[Ollama](https://ollama.com). Run every command from the repository root.

```bash
git clone https://github.com/ricca200xx/llm-doc-extraction-eval
cd llm-doc-extraction-eval
uv sync
uv run pytest
```

Rebuild the report from the committed model outputs. No model, API key or
dataset download is needed:

```bash
uv run docextract report
```

Run a model yourself. The first run downloads the CORD-v2 test split (~230 MB)
into `data/` and checks its SHA-256:

```bash
ollama pull qwen2.5vl:3b
uv run docextract run --model ollama_chat/qwen2.5vl:3b --prompt v1 --n 50
uv run docextract run --model ollama_chat/qwen2.5vl:3b --prompt v2 --n 50
uv run docextract report
```

## How it works

```
receipt image ──► prompt v1 / v2 ──► LiteLLM ──► raw text ──► JSON repair ──► Pydantic schema
                                    (any model)                                     │
                         gold labels (CORD gt_parse) ──► strict per-field scorer ◄──┘
                                                             │
                     macro accuracy, macro F1, hallucination rate, always-null baseline
                              bootstrap 95% CIs, paired McNemar + Holm
```

### Dataset

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) is a set of
photographed Indonesian receipts with structured labels (`gt_parse`). The loader
([`data.py`](src/docextract/data.py)) downloads only the test split (100
receipts), pinned to a fixed dataset revision. The file is written to a
temporary path, checked against a known SHA-256 and only then moved into
`data/`, so an interrupted or altered download is never used. Each image is
downscaled so its longest side is at most 1280 px and re-encoded as JPEG
(quality 90) before it is sent to the model.

### Schema

The target is a Pydantic model ([`schema.py`](src/docextract/schema.py)) with 8
optional string fields. `null` means "not printed on this receipt". Numbers
returned by the model are converted to strings, and unknown keys are ignored
(the report counts them, so a model that renames fields cannot hide behind it).

| Field | CORD label | Meaning (also given to the model in prompt v2) | Present in the 50 receipts |
|---|---|---|---:|
| `total_price` | `total.total_price` | Final amount due after tax, service and discount | 47 |
| `subtotal_price` | `sub_total.subtotal_price` | Sum of the items before tax, service and discount | 31 |
| `tax_price` | `sub_total.tax_price` | Tax amount (TAX, VAT, PB1, PPN) | 20 |
| `service_price` | `sub_total.service_price` | Service charge amount | 7 |
| `discount_price` | `sub_total.discount_price` | Discount on the whole bill | 5 |
| `cash_price` | `total.cashprice` | Amount paid in cash | 35 |
| `change_price` | `total.changeprice` | Change given back to the customer | 31 |
| `credit_card_price` | `total.creditcardprice` | Amount paid by card | 8 |

Most fields are absent from most receipts, which is what makes the set useful
for measuring hallucination: a model that fills every field looks fine on the
common ones and fails on the rest.

### Prompts

[`prompts.py`](src/docextract/prompts.py) defines two versions. v1 is the prompt
most people write first: one user message listing the keys. v2 changes four
things at once: it adds a system message, a definition for each field, `null` as
the explicit answer for anything not printed, and a ban on computing or reusing
numbers. v2 was written once, before any run, and has not been tuned on these
results. Print the exact messages with `uv run docextract prompt v1` (or `v2`).

<details>
<summary>Prompt v1 (one user message, with the image)</summary>

```text
Extract the following fields from this receipt and return them as JSON: total_price,
subtotal_price, tax_price, service_price, discount_price, cash_price, change_price,
credit_card_price.
```
</details>

<details>
<summary>Prompt v2 (system message + user message with the image)</summary>

```text
[system]
You read receipt images and return one JSON object, nothing else.

Rules:
- Use exactly the keys listed below. No extra keys, no markdown, no comments.
- Copy each amount as it is printed on the receipt.
- If a field is not printed on the receipt, its value is null.
- Never compute, estimate or infer a value from other numbers, and never reuse
  one printed number for two different fields unless the receipt prints it twice.

[user]
Fields:
- "total_price": Final amount due after tax, service and discount (TOTAL, GRAND TOTAL).
- "subtotal_price": Sum of the items before tax, service and discount (SUBTOTAL).
- "tax_price": Tax amount (TAX, VAT, PB1, PPN).
- "service_price": Service charge amount (SERVICE, SC).
- "discount_price": Discount on the whole bill (DISCOUNT, DISC).
- "cash_price": Amount paid in cash (CASH, TUNAI).
- "change_price": Change given back to the customer (CHANGE, KEMBALI).
- "credit_card_price": Amount paid by card (CARD, CREDIT, DEBIT, VISA, MASTER).

Return the JSON object for this receipt.
```
</details>

### Extraction

[`extract.py`](src/docextract/extract.py) makes one `litellm.completion` call per
receipt with `temperature=0`, `max_tokens=512`, 2 retries on transient errors
and a 300 s timeout. The image travels as a base64 data URL, so any LiteLLM
provider with vision support works without code changes. JSON mode is
deliberately not used: not every provider supports it, and the point is to
measure what models return when asked in plain words.

- Every answer is appended to a JSONL file as soon as it arrives.
- Running the same command again resumes: receipts already done are skipped,
  receipts whose call failed are retried.
- Each record stores a fingerprint of the exact prompt text and the token
  limit. If you edit a prompt or change `--max-tokens`, the run refuses to
  append to the old file instead of silently mixing two configurations.
- If 3 calls in a row fail (a wrong model name, a missing API key), the run
  stops with the provider's error message instead of recording 50 failures.

### JSON repair

[`repair.py`](src/docextract/repair.py) turns raw model text into a JSON object
or gives up. It never guesses a value.

| Raw output | Result | Status |
|---|---|---|
| valid JSON object | parsed as is | `ok` |
| object inside a markdown code fence, closed or not | fence removed | `repaired` |
| prose before or after the object, even with a stray `{` | prose dropped | `repaired` |
| trailing comma before `}` or `]` | comma removed | `repaired` |
| cut off after a finished string value | object closed, value kept | `repaired` |
| cut off inside a string, after a number, after a key or a colon | last member dropped, object closed | `repaired` |
| cut off inside a nested list or object | inner containers closed in the right order | `repaired` |
| no object at all, a JSON array, or nothing complete before the cut | nothing | `failed` |

A number at the cut point is dropped because it may itself be incomplete
(`60` of `60000`). The effect is that truncation shows up as a miss (FN) rather
than as a wrong value (FP + FN).

### Scoring

[`scoring.py`](src/docextract/scoring.py) compares prediction and gold field by
field. Each (document, field) pair lands in exactly one cell:

| gold \ prediction | `null` | same value | different value |
|---|---|---|---|
| `null` | correct abstention | — | **hallucination**: FP |
| value | miss: FN | TP | wrong value: FP + FN |

A document whose call failed or whose output could not be parsed has no
prediction. It gets no credit for abstaining: all its cells count as wrong for
accuracy, and its gold values count as FN. Otherwise a model that crashes would
score 54% for free.

Values are compared after number canonicalisation, and only that:

| Printed / returned | Canonical form | Rule |
|---|---|---|
| `Rp 60.000`, `60,000`, `60000` | `60000` | currency, spaces and thousands separators removed |
| `60,000.00`, `365000.0` | `60000`, `365000` | one or two trailing decimal digits, zeros dropped |
| `3.50`, `3,5` | `3.5` | non-zero decimals kept |
| `-60.000`, `(0)`, `[145,900]` | `60000`, `0`, `145900` | sign and brackets ignored |
| `60,000.00 IDR`, `12.50%` | `60000`, `12.5` | trailing text ignored |
| `""`, `"n/a"`, `"-"` | `null` | no digit means no value |

Three digits after the last separator are a thousands group (`60.000` is sixty
thousand), one or two are decimals. Any other difference, including a single
digit, is a wrong value.

Metrics are computed per field and then averaged over the 8 fields (macro):

- **Accuracy** = (TP + correct abstentions) / documents.
- **F1** = 2·TP / (2·TP + FP + FN). A field with no gold values and no
  predictions has no F1 and is left out of the macro average.
- **Hallucination rate** = hallucinations / `null` cells, pooled over fields.
- **Always-`null` baseline**: the same metrics for an empty prediction on every
  document that all runs share. Its accuracy is the share of `null` cells; its
  F1 is 0.

### Statistics

[`stats.py`](src/docextract/stats.py). All resampling is over documents, not
cells, because the 8 fields of one receipt are not independent: a blurry photo
hurts all of them at once.

- **95% confidence intervals** for accuracy, F1 and hallucination rate:
  percentile bootstrap, 2,000 resamples of the documents with replacement,
  fixed seed.
- **v1 vs v2, overall**: paired bootstrap. Each resample draws the same
  documents for both prompts and recomputes the macro difference, so
  document-to-document difficulty cancels out.
- **v1 vs v2, per field**: exact McNemar test on per-document correctness.
  With *b* documents right only under v1 and *c* right only under v2,
  p = min(1, 2 · Σ_{k ≤ min(b,c)} C(b+c, k) / 2^(b+c)).
- **Multiple testing**: Holm correction across the 8 per-field tests, so
  "significant" means significant after looking at 8 fields, not 1.

## Usage

### CLI

```text
docextract run --model MODEL --prompt {v1,v2} [--n N] [--start S] [--max-tokens T]
docextract prompt {v1,v2}
docextract report
```

| Command / flag | Default | What it does |
|---|---|---|
| `run --model` | required | Any LiteLLM model string, e.g. `ollama_chat/qwen2.5vl:3b` |
| `run --prompt` | required | Prompt version, `v1` or `v2` |
| `run --n` | `50` | Number of receipts, 1 to 100 |
| `run --start` | `0` | Index of the first receipt in the test split; `--start 50 --n 50` uses the second half |
| `run --max-tokens` | `512` | Output token limit per call, 16 to 8192 |
| `prompt` | | Print the exact messages a prompt version sends |
| `report` | | Score every file in `results/raw`, write `results/report.md`, `results/summary.json` and `results/chart.png`, and print the report |

Document ids are the receipt's index in the test split (`cord-test-000` to
`cord-test-099`), so runs over different ranges can still be compared on the
documents they share.

### Other models and providers

The model string is the only thing that changes. Set the provider's API key in
the environment and pass its LiteLLM name:

| Backend | Example `--model` | Needs |
|---|---|---|
| Ollama, local | `ollama_chat/qwen2.5vl:3b` | `ollama pull qwen2.5vl:3b` |
| Ollama Cloud | `ollama_chat/gemma4:31b-cloud` | a signed-in Ollama account (`ollama signin`) |
| OpenAI | `openai/gpt-4.1-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |

Any other [LiteLLM provider](https://docs.litellm.ai/docs/providers) with image
input works the same way.

### Where results go

| Path | Content |
|---|---|
| `results/raw/<model>__<prompt>.jsonl` | One record per model call, appended as it arrives |
| `results/report.md` | Summary table with the baseline, per-field accuracy, paired comparisons |
| `results/summary.json` | The same numbers, machine-readable |
| `results/chart.png` | The per-field chart shown above |

- Each (model, prompt) pair has its own file, so runs never overwrite each
  other. Delete or rename a file to start that run from scratch.
- `report` scores **every** file in `results/raw`, including the committed
  ones, and overwrites `report.md`, `summary.json` and `chart.png`. To see only
  your own runs, move the committed JSONL files somewhere else first.
- `report` compares v1 and v2 for every model that has both, on the documents
  the two runs share. It also checks that every run carries the same gold
  labels for the same document and stops if they disagree.

<details>
<summary>One JSONL record</summary>

```json
{
  "doc_id": "cord-test-001",
  "model": "ollama_chat/gemma4:31b-cloud",
  "prompt": "v2",
  "prompt_sha": "baf801054113",
  "max_tokens": 512,
  "gold": {"total_price": "91000", "subtotal_price": null, "...": "..."},
  "pred": {"total_price": "91000", "subtotal_price": null, "...": "..."},
  "parse_status": "repaired",
  "error": null,
  "latency_s": 1.89,
  "raw": "```json\n{\n  \"total_price\": \"91000\", ...",
  "finish_reason": "stop",
  "completion_tokens": 93
}
```

- `raw` is the model's exact answer; `pred` is what the scorer saw after repair
  and schema validation.
- If a document appears more than once (a retry after an error), the report
  uses the last record.
- The committed runs were made before `prompt_sha` and `max_tokens` were
  recorded (512 for all of them); records without these keys are accepted.
</details>

### Extending the harness

- **A new field**: add it to `Receipt` in [`schema.py`](src/docextract/schema.py)
  with a description, and map it to its CORD label in `CORD_KEYS`. Prompts,
  scorer and report pick it up automatically. Existing runs must be re-run:
  their stored gold and predictions do not contain the new field, and `report`
  will refuse to mix them.
- **A new prompt version**: add a branch to `build_messages` in
  [`prompts.py`](src/docextract/prompts.py) and add the name to `PROMPTS` in
  [`__main__.py`](src/docextract/__main__.py). The paired comparison in
  `report.py` and the chart are written for `v1` against `v2`.
- **A new dataset**: write a loader that returns `Doc(doc_id, image_b64, gold)`
  objects, like `load_docs` in [`data.py`](src/docextract/data.py). Everything
  after that point is dataset-independent.

## Reproducibility

- The dataset is pinned to a fixed Hugging Face revision and verified by
  SHA-256; the document subset is a fixed index range, not a random sample.
- Bootstrap resampling uses a fixed seed, so the report is deterministic given
  the raw outputs.
- The raw outputs of every run, including the gold labels, are committed. CI
  rebuilds `results/report.md` from them and fails if it differs from the
  committed file, so the numbers in this README cannot drift away from the code.
- Model inference itself is not guaranteed to be deterministic, even at
  temperature 0, especially for hosted models.

## Development

```bash
uv run pytest               # 62 unit tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

The tests pin the behaviour that the numbers depend on:

- every repair case in the table above;
- every scoring cell, the failed-document rule, canonicalisation edge cases and
  F1 when a field is undefined;
- the McNemar and Holm arithmetic and the bootstrap;
- the report: last record wins, partial lines are skipped, runs with different
  gold are rejected, the baseline, the paired comparison;
- the prompts and the loader's input checks.

GitHub Actions runs lint, format check, tests and the report reproducibility
check on every push, with read-only permissions and actions pinned to commit
SHAs. Dependabot keeps those pins up to date.

## Repository layout

```
src/docextract/
  __main__.py  CLI: run, prompt, report
  schema.py    Pydantic schema + mapping to CORD labels
  data.py      CORD-v2 loader (pinned revision, SHA-256 check, image downscaling)
  prompts.py   prompt v1 and v2, prompt fingerprint
  extract.py   LiteLLM calls, resumable JSONL output, fail-fast on repeated errors
  repair.py    JSON repair for fenced, chatty or truncated output
  scoring.py   canonicalisation, per-field outcomes, accuracy / F1 / hallucination rate
  stats.py     bootstrap CIs, paired bootstrap, exact McNemar, Holm
  report.py    baseline, tables, paired comparison, chart
tests/
  test_repair.py    repair layer and output parsing
  test_scoring.py   scorer, canonicalisation, statistics
  test_report.py    run loading, baseline, failed documents, paired comparison
  test_prompts.py   prompt structure, fingerprint, loader input checks
results/            raw outputs (JSONL), report.md, summary.json, chart.png
.github/            CI workflow, Dependabot
```

## Limitations

- **Small sample.** 50 receipts, the first 50 of the CORD-v2 test split. The
  confidence intervals are the honest summary: differences of one or two points
  between runs are inside them.
- **Low power per field.** An exact McNemar test needs at least 6 discordant
  documents, all in the same direction, to reach p < 0.05 before correction, and
  more after Holm. "Not significant" here means "not shown", not "no effect".
- **One run per configuration.** Temperature is 0, but hosted inference is not
  guaranteed to be deterministic, and run-to-run variance was not measured.
- **No prompt ablation.** v2 bundles four changes. The results say the bundle
  works for the small model; they do not say which change does it.
- **Strict scoring meets annotation conventions.** Some false positives are
  disagreements with how CORD labels a receipt, not invented numbers. Example:
  in `cord-test-009` the only printed amount is labelled `subtotal`, so a model
  that reports it as `total` is scored as hallucinating a total; in
  `cord-test-039` a card payment is labelled but no total is. Nothing was
  relabelled, to avoid tuning the benchmark to the models.
- **`"0"` for an unprinted field counts as a hallucination.** CORD labels a
  printed `0` (for example a change of 0) as a value, so "0" and "not printed"
  are different answers. A scorer that mapped `"0"` to `null` would rate Qwen
  with prompt v1 much higher; most of its v1 errors are zeros.
- **Number canonicalisation is a choice.** The sign is ignored on every field
  (`-60.000` and `60.000` match), because CORD writes discounts both ways; only
  the discount field needs this. One or two digits after the last separator are
  read as decimals, three as a thousands group.
- **Only scalar fields.** Line items (`menu`) are not scored. They need an
  alignment step between predicted and gold items, which this repo does not do.
- **One dataset, one language.** CORD receipts are Indonesian, photographed,
  mostly in one currency. Results say nothing about invoices, forms or IDs.
- **Hosted model versions can change.** `gemma4:31b-cloud` is served by Ollama
  Cloud; the same tag may point to different weights later. The local model is
  the default quantised `qwen2.5vl:3b` build from the Ollama library, run on an
  Apple M4 Pro (48 GB).
- **Failed calls.** In the committed runs every call eventually succeeded; two
  early attempts that failed are kept in the JSONL for the record and superseded
  by their retry.

## Data, license and citation

Code: MIT, see [LICENSE](LICENSE).

CORD-v2 is published by NAVER Clova under CC-BY-4.0. The JSONL files in
`results/raw` contain the gold labels of the 50 receipts used here, copied from
that dataset so the report can be rebuilt without downloading it. If you use the
data, cite the original paper:

```bibtex
@inproceedings{park2019cord,
  title     = {CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author    = {Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and
               Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk},
  booktitle = {Workshop on Document Intelligence at NeurIPS 2019},
  year      = {2019}
}
```
