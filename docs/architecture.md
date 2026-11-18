# Architecture notes

This document covers the pieces that do not fit comfortably in the README: the
state contract, the exact sequence of a request, how interrupts survive a
pause, and where the design would need to change for production.

---

## State

`src/state.py` defines one `TypedDict` shared by the parent graph and both
sub-agents:

| Field | Written by | Purpose |
|---|---|---|
| `customer_id` | `verify_customer` | Verified Chinook `CustomerId`; empty until resolved. |
| `messages` | everyone (`add_messages`) | Full thread history. LangGraph merges by message id, so re-emitting history from a sub-agent does not duplicate it. |
| `loaded_memory` | `load_memory` | Preference summary injected into both specialist prompts. |
| `remaining_steps` | LangGraph | Managed recursion guard used by the ReAct agents. |
| `route` | `supervisor` | Last routing decision; read only by the conditional edges. |

Keeping `route` in state rather than on the graph object makes routing
inspectable from a checkpoint and unit-testable as plain functions
(`tests/test_graph_structure.py`).

---

## Lifecycle of one message

```
CLI / notebook
   │  graph.invoke({"messages": [HumanMessage(...)]}, config={"thread_id": T})
   ▼
verify_customer ── state has customer_id? ── yes ──► load_memory
   │ no
   ├── try explicit ID → bare number → email → phone
   └── unresolved ── interrupt({"type": "identity_verification", ...})

load_memory ── store.get(("customer", id, "preferences"), "summary")
   ▼
supervisor ── structured output: {route, reasoning}
   ├── music_catalog ─► music ReAct agent ─► (both?) invoice_info
   ├── invoice_info ──► invoice ReAct agent ─► (both?) synthesize
   └── FINISH ────────► END
   ▼
save_memory ── structured extraction ── store.put(...)
   ▼
END
```

### The `both` route

The route is a fixed pipeline, not a negotiation:

1. `music_catalog` answers the catalog half.
2. `invoice_info` answers the account half.
3. `synthesize` makes one final call that merges both replies, with an
   explicit instruction to preserve every amount and title verbatim.

If step 3 fails, the two replies are already in `messages`, so the customer
still receives both answers — just less tidily formatted.

### Interrupt mechanics

`langgraph.types.interrupt()` raises out of the node, and the graph is
checkpointed at that point. The caller sees `result["__interrupt__"][0].value`
and later resumes with `Command(resume=<reply>)`; the return value of
`interrupt()` is that reply, so the node continues from the same line.

Two independent producers use it:

| Producer | Trigger | Where |
|---|---|---|
| `ask_for_identity()` | No identity could be resolved | `src/nodes/verify_customer.py` |
| `require_nonblank()` | A tool was called with a blank required argument | `src/tools/_common.py` |

Because `customer_id` lives in checkpointed state, the invoice specialist can
run *after* a pause and still know who the customer is. `tests/test_hitl.py`
verifies the interrupt/resume contract in isolation, without an LLM.

---

## Memory

**Short-term** is the checkpointer (`InMemorySaver`), keyed by `thread_id`. It
carries the message history and the verified id.

**Long-term** is the store (`InMemoryStore`), keyed by
`("customer", customer_id, "preferences")` with one `"summary"` item. Writes
happen in `save_memory` through a structured-output extraction that is told the
previous summary, so updates merge instead of overwrite.

The two are deliberately separate: a new `thread_id` starts a new conversation
but keeps the customer's preferences, which is the behaviour a returning
customer expects.

---

## Failure behaviour

| Failure | Result | Where |
|---|---|---|
| Supervisor LLM call fails | Route `FINISH`, turn ends cleanly | `src/agents/supervisor.py` |
| Synthesis call fails | Both specialist answers kept as-is | `src/graph.py` |
| Preference extraction fails | No store write, turn unaffected | `src/nodes/memory.py` |
| Chinook download fails | `DatabaseLoadError` with the URL and original error | `src/db.py` |
| Missing API key | `ConfigurationError` naming the variable to set | `src/llm.py` |
| LLM calls a tool with a blank argument | Graph interrupts and asks the customer | `src/tools/_common.py` |

The pattern is consistent: degrade one capability, never lose the turn.

---

## Extension points

- **Swap provider** — `src/llm.py` is the only place that constructs a client.
- **Add a specialist** — add tools under `src/tools/`, an agent under
  `src/agents/`, a node and one conditional edge; extend the `Literal` in
  `RouteDecision`.
- **Persist memory** — replace `InMemorySaver`/`InMemoryStore` in
  `build_graph()`; no node changes are required.
- **Serve it** — `get_graph()` returns a process-wide compiled graph, which is
  what a FastAPI/`langgraph serve` deployment would own per worker.

---

## Dependency policy

`requirements.txt` pins major versions (`>=1.0,<2.0`) rather than exact ones,
so patch releases are picked up but a breaking major cannot arrive silently.

One deliberate trade-off is recorded in the code: the specialists use
`langgraph.prebuilt.create_react_agent`, which is deprecated in LangGraph v1 in
favour of `langchain.agents.create_agent`. The newer API replaces the per-turn
prompt callable with `dynamic_prompt` middleware and carries request context
outside graph state, which is a real refactor rather than a rename. It is
deferred until v2, and the upper bound in `requirements.txt` keeps the current
version working in the meantime.

---

## Project brief

The system was built to satisfy a course brief with these requirements, all of
which are implemented and covered by the test suite:

1. Load Chinook into an in-memory SQLite database.
2. Use a shared graph state (`customer_id`, `messages`, `loaded_memory`).
3. Provide a music catalog sub-agent with album, track, genre, and song tools.
4. Provide an invoice sub-agent with date-sorted, price-sorted, and
   support-employee tools.
5. Route messages with a supervisor, including requests that need both.
6. Verify customers by ID, email, or phone, with a human-in-the-loop fallback.
7. Maintain per-customer long-term music preferences.
8. Pause and resume for missing information.
9. Cover the two acceptance scenarios: a verified combined request with a
   memory follow-up, and an unverified request that triggers an interrupt.
