"""Music Catalog Sub-Agent.

A ReAct-style agent (LangGraph's ``create_react_agent``) bound to the music
catalog tools. The system prompt is rebuilt per turn so the customer's
long-term preferences can be injected for personalization.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from src.llm import get_llm
from src.state import State
from src.tools.music_tools import MUSIC_TOOLS

MUSIC_AGENT_SYSTEM_PROMPT = """You are the Music Catalog Sub-Agent for a \
digital music store's customer support system.

Your job is to answer questions about albums, tracks, genres, and to give \
music recommendations, using ONLY the provided tools to look up real data \
from the Chinook catalog. Never invent album or track names.

Rules:
- Always call a tool to look up catalog information before answering.
- If the user's request is missing a required detail (e.g. no artist name, \
no genre, no song title given), call the relevant tool anyway with an \
empty value for that field -- the tool itself will ask the user for the \
missing detail and pause until they answer.
- When you have results, summarize them clearly and concisely for the \
customer (e.g. a bulleted list of album titles or song names).
- If the customer's long-term preferences are provided below, use them to \
personalize recommendations when relevant.

Customer's known music preferences:
{loaded_memory}
"""


def _build_prompt(state: State):
    """Build the message list for the music agent, injecting long-term memory."""
    loaded_memory = state.get("loaded_memory") or "No preferences on file yet."
    system_text = MUSIC_AGENT_SYSTEM_PROMPT.format(loaded_memory=loaded_memory)
    return [SystemMessage(content=system_text), *state["messages"]]


@lru_cache(maxsize=1)
def get_music_agent():
    """Return the compiled Music Catalog Sub-Agent (built on first use)."""
    return create_react_agent(
        get_llm(),
        tools=MUSIC_TOOLS,
        prompt=_build_prompt,
        state_schema=State,
    )
