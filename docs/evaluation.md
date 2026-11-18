# Evaluation

The interesting failure mode of this system is not a bad sentence — it is the
supervisor sending a message to an agent that cannot answer it. That decision is
measurable on its own, so it is evaluated on its own.

## What is measured

**Supervisor routing accuracy**: for a labelled message, did the router choose
the expected one of `music_catalog`, `invoice_info`, `both`, or `FINISH`?

Each case is a single structured-output call with `temperature=0`, using the
same prompt the graph uses. Nothing is mocked.

## What is not measured

Being explicit about this matters more than the number itself:

- **Answer quality.** Whether the chosen agent then produces a correct, grounded
  reply is covered by the end-to-end tests, not scored here.
- **Tool selection.** A specialist can pick the wrong tool for a correctly
  routed message; that is a separate evaluation.
- **Multi-turn behaviour.** Cases are single messages with no conversation
  history. The `FINISH` label in particular is a judgement about a turn in
  isolation; in a real thread it also depends on what was just answered.
- **Memory effects.** `loaded_memory` is held constant ("No preferences on file
  yet.") so the router is measured on intent, not on personalisation.

## The dataset

`evaluation/routing_cases.jsonl` — one JSON object per line:

```json
{"message": "How much was my most recent purchase, and what albums do you have by the Rolling Stones?", "expected": "both", "note": "the acceptance scenario"}
```

It is deliberately small (26 cases) and skewed towards the mistakes that matter:

| Group | Why it is there |
|---|---|
| Plain catalog requests | Baseline: albums, tracks, genres, song existence, recommendations. |
| Plain account requests | Baseline: invoices, purchase history, support rep. |
| Two-intent messages | The `both` route is the expensive one to get wrong. |
| Invoice-only traps | *"What did I pay for my last order, and who handled it?"* — two questions, one specialist. Should **not** become `both`. |
| Courtesy closers | `FINISH` is easy to over-apply; four variants keep it honest. |

`tests/test_evaluation_dataset.py` validates the file in CI: unique messages,
all four routes present, every case annotated, and every `both` case actually
containing two requests.

## Running it

```bash
export GROQ_API_KEY=...          # or put it in .env
python -m evaluation.routing_eval
```

The runner prints per-case results, per-route accuracy, and writes
`evaluation/results.md` with a table of every prediction. It exits non-zero when
accuracy falls below `--threshold` (default `0.85`), so it can gate a change to
the routing prompt.

## Baseline

First recorded run — 18 November 2026, `openai/gpt-oss-120b`, 26 cases:

| Route | Correct | Notes |
|---|---|---|
| `music_catalog` | 8 / 8 | |
| `invoice_info` | 7 / 8 | one purchase question routed to `both` |
| `both` | 5 / 6 | one two-intent message routed to `invoice_info` |
| `FINISH` | 4 / 4 | |
| **Overall** | **24 / 26 (92.3%)** | |

Both misses are specific rather than random, which is what makes them useful:

- *"What was the most expensive track I ever bought?"* → `both`. The word
  "track" pulls toward the catalog, but the question is answerable from invoice
  line items alone. A prompt tweak that ties "bought/paid" to the invoice
  specialist would fix it.
- *"Did I buy the album 'No Security', and what did it cost?"* → `invoice_info`.
  The model treated "did I buy X" as a purchase-history lookup, but the invoice
  tools return `TrackId`, not album titles, so the answer needs the catalog too.
  This one is worth watching: it is the case where a plausible-sounding route
  produces a confidently incomplete answer.

Neither failure is worth changing the prompt for on its own — 26 cases is a
regression guard, not a benchmark. The value is that the next prompt change can
be measured instead of argued about.

## The results file

Results are written to [`../evaluation/results.md`](../evaluation/results.md) on each run. The file is
committed so a reviewer can see the current baseline next to the prompt that
produced it; the run timestamp and model name are recorded with it.

Aggregate accuracy is a weak signal on its own — read the per-route breakdown.
A router that scores 90% by never choosing `both` is worse than one that scores
85% with balanced errors.
