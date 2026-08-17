"""Invoice Information Sub-Agent.

A ReAct-style agent (LangGraph prebuilt) bound to the invoice tools. It has
access to the verified ``customer_id`` from short-term (graph) state, so it
never needs to ask the customer to repeat their identity.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from src.llm import get_llm
from src.state import State
from src.tools.invoice_tools import INVOICE_TOOLS

INVOICE_AGENT_SYSTEM_PROMPT = """You are the Invoice Information Sub-Agent \
for a digital music store's customer support system.

Your job is to answer questions about a customer's invoices/purchases, \
including totals, dates, most expensive items, and which support employee \
handled a given invoice. Use ONLY the provided tools -- never invent \
numbers or dates.

The verified customer for this session has CustomerId = {customer_id}. \
Always use this CustomerId when calling tools; do not ask the customer to \
repeat it.

Rules:
- Always call a tool to look up real invoice data before answering.
- If a tool needs an invoice ID that hasn't been specified (e.g. asking \
"who was my support rep for this invoice" without saying which one), call \
the tool with an empty value for that field -- the tool will ask the user \
for the missing detail and pause until they answer.
- Present amounts with a currency symbol and dates in a readable format.
- Be concise and factual.
"""


def _build_prompt(state: State):
    """Build the message list for the invoice agent, injecting customer_id."""
    customer_id = state.get("customer_id") or "UNKNOWN"
    system_text = INVOICE_AGENT_SYSTEM_PROMPT.format(customer_id=customer_id)
    return [SystemMessage(content=system_text)] + state["messages"]


def build_invoice_agent():
    """Construct the Invoice Information Sub-Agent as a compiled LangGraph agent."""
    llm = get_llm()
    return create_react_agent(
        llm,
        tools=INVOICE_TOOLS,
        prompt=_build_prompt,
        state_schema=State,
    )


invoice_agent = build_invoice_agent()
