# Music Store Customer Support Multi-Agent System

A production-quality customer support system for a digital music store,
built with **LangChain**, **LangGraph**, and **Jupyter Notebook**, backed by
the **Chinook** SQLite database (loaded entirely in-memory, no external DB
setup required).

A **Supervisor** agent routes each customer message to one or both of two
specialist sub-agents:

- **Music Catalog Sub-Agent** — albums, tracks, genres, recommendations.
- **Invoice Information Sub-Agent** — invoices, purchase history, support
  employee lookups.

The system verifies the customer's identity before answering anything
sensitive, remembers music preferences across sessions (long-term memory),
and keeps `customer_id` + conversation context for the life of a session
(short-term memory), including across human-in-the-loop pauses.

---

## 1. Architecture

```
                              +-------------------+
                 START -----> |  verify_customer  |----(interrupt: ask for
                              |  (HITL if needed) |     Customer ID/Phone/
                              +---------+----------+     Email, then resume)
                                        |
                                        v
                              +-------------------+
                              |    load_memory     |  (reads long-term prefs
                              |                     |   from InMemoryStore)
                              +---------+----------+
                                        |
                                        v
                              +-------------------+
                              |     supervisor      |  (structured routing:
                              |                     |   music_catalog /
                              +----+-----+-----+----+   invoice_info / both
                                   |     |     |         / FINISH)
                    music_catalog  |     |     |  FINISH
                                   v     |     +----------------> END
                        +-------------+  |
                        |music_catalog|  | invoice_info
                        |  sub-agent  |  v
                        +------+------+ +--------------+
                               |        | invoice_info  |
                     (route == |        |  sub-agent    |
                      "both")  |        +-------+-------+
                               v                |
                        +-------------+         |
                        | invoice_info|<--------+ (route == "both")
                        |  sub-agent  |
                        +------+------+
                               |
                     (route=="both")
                               v
                        +-------------+
                        | synthesize  |  (merge both replies)
                        +------+------+
                               |
                               v
                        +-------------+
                        | save_memory |  (extract & persist prefs
                        +------+------+   into InMemoryStore)
                               |
                               v
                              END
```

Both the Music Catalog and Invoice Information sub-agents are built with
LangGraph's `create_react_agent` prebuilt, bound to their own tool sets, and
can each pause the whole graph via `interrupt()` if a tool call is missing a
required argument (e.g. no artist name, no invoice ID) — the user is asked
for the missing detail and the tool call resumes with it.

## 2. Project structure

```
music_store_support/
├── requirements.txt
├── .env.example
├── README.md
├── music_store_customer_support.ipynb
├── src/
│   ├── __init__.py
│   ├── db.py                  # Chinook engine + SQLDatabase (in-memory)
│   ├── llm.py                 # shared ChatOpenAI construction
│   ├── state.py                # graph State schema
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── music_tools.py      # get_albums_by_artist, get_tracks_by_artist, ...
│   │   └── invoice_tools.py    # get_invoices_by_customer_sorted_by_date, ...
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── music_agent.py      # Music Catalog Sub-Agent (create_react_agent)
│   │   ├── invoice_agent.py    # Invoice Information Sub-Agent
│   │   └── supervisor.py       # structured-output routing decision
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── verify_customer.py  # identity parsing + HITL interrupt loop
│   │   ├── memory.py           # load_memory / save_memory (long-term store)
│   │   └── hitl.py             # shared interrupt() helpers
│   ├── graph.py                 # builds & compiles the StateGraph
│   └── run.py                   # interactive loop + demo runner
└── tests/
    ├── test_tools.py
    ├── test_verification.py
    └── test_e2e.py
```

## 3. Setup

**Requirements:** Python 3.10+, and internet access on first run (to
download the Chinook SQL script — no local database installation needed).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

> **Note:** No external database setup is required. The Chinook database is
> downloaded as a `.sql` script and loaded into an in-memory SQLite
> connection automatically the first time `src/db.py` is imported.
> Internet access is required for this download on first run.

## 4. Running

### Jupyter notebook (recommended for exploring the system)

```bash
jupyter lab music_store_customer_support.ipynb
# or
jupyter notebook music_store_customer_support.ipynb
```

Run all cells top-to-bottom. The notebook prints the Chinook table list,
walks through every component, and runs both required test cases with
their outputs.

### CLI demo (non-interactive, runs the two required test cases)

```bash
python -m src.run
```

### CLI interactive loop

```python
from src.run import run_conversation
run_conversation(thread_id="my-session-1")
```

Type your messages at the `You:` prompt. When the assistant needs more
information (identity verification or a missing detail like an artist
name), it will print a prompt and wait for your reply, then resume
automatically.

### Tests

```bash
pytest -q
```

- `test_tools.py` and `test_verification.py` only need internet access (to
  load Chinook) — no API key required.
- `test_e2e.py` drives the full LLM-backed graph and is automatically
  skipped if `OPENAI_API_KEY` is not set.

## 5. Agents & tools

### Music Catalog Sub-Agent (`src/agents/music_agent.py`)

- `get_albums_by_artist(artist)` — albums by an artist (partial match).
- `get_tracks_by_artist(artist)` — tracks by an artist, with a fallback to
  a broader word-based search if no exact/partial match is found.
- `get_songs_by_genre(genre)` — songs in a genre.
- `check_for_songs(song_title)` — partial, case-insensitive title search.

Personalizes answers using `state["loaded_memory"]` (the customer's known
preferences).

### Invoice Information Sub-Agent (`src/agents/invoice_agent.py`)

- `get_invoices_by_customer_sorted_by_date(customer_id)` — invoices, newest
  first.
- `get_invoices_sorted_by_unit_price(customer_id)` — invoice line items,
  most expensive first.
- `get_employee_by_invoice_and_customer(invoice_id, customer_id)` — the
  support rep tied to an invoice (`Customer.SupportRepId -> Employee`).

Always uses the verified `customer_id` from short-term state — never asks
the customer to repeat it.

### Supervisor (`src/agents/supervisor.py`)

Uses a Pydantic-typed structured output (`RouteDecision`) to choose exactly
one of `"music_catalog"`, `"invoice_info"`, `"both"`, or `"FINISH"` for the
latest message, informed by the customer's loaded long-term preferences.

All SQL in every tool uses SQLAlchemy bound parameters (`:name`-style) —
user input is never concatenated into SQL text.

## 6. Human-in-the-loop (HITL) behavior

Two independent interrupt scenarios are implemented with
`langgraph.types.interrupt()` / `Command(resume=...)`:

1. **Identity verification** (`verify_customer` node): if the latest
   message doesn't contain a resolvable Customer ID, phone, or email, the
   graph interrupts and asks: *"Please provide your Customer ID, Phone, or
   Email so I can verify your identity."* Phone numbers are normalized
   (spaces, parentheses, dashes, and a leading `+` are stripped) before
   matching against `Customer.Phone`. On resume, verification is retried;
   if it still fails, the graph interrupts again.

2. **Missing tool arguments** (music/invoice tools): if the LLM calls a
   tool with a blank required argument (e.g. no artist name given), the
   tool itself calls `interrupt()` to ask a targeted clarifying question
   (e.g. *"Which artist would you like albums for?"*), then uses the
   resumed reply as the argument value and completes the lookup.

Because the graph is compiled with `InMemorySaver()` as the checkpointer,
`state["customer_id"]` and the full message history survive across any
number of interrupt/resume cycles within the same `thread_id`.

## 7. Long-term memory

`langgraph.store.memory.InMemoryStore` holds a per-customer preferences
summary under the namespace `("customer", customer_id, "preferences")`.

- `load_memory` reads this summary into `state["loaded_memory"]` at the
  start of every turn, and it's injected into both sub-agents' prompts.
- `save_memory` runs at the end of every turn: a small structured-output
  LLM call scans the recent conversation for concrete artist/genre/song
  preferences, merges them with whatever was already stored (deduplicated,
  kept concise), and writes the result back.

## 8. Known limitations

- Phone-number matching uses normalization + a same-length or 9-digit
  suffix comparison; unusual international formats not present in Chinook
  may not resolve.
- The `synthesize` step for `"both"` routing does a single extra LLM call
  to merge sub-agent replies; if that call fails, the two sub-agent replies
  are still present in the message history and used as-is (synthesis is
  best-effort, not required for correctness).
- Long-term memory (`InMemoryStore`) and short-term memory
  (`InMemorySaver`) are both **process-local and non-persistent** — restart
  the Python process/kernel and all memory is lost. Swap in a persistent
  store/checkpointer (e.g. Postgres-backed) for production use.
- The Chinook data is fetched fresh from GitHub on every process start;
  without internet access on first run, the system cannot start.
