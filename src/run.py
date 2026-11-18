"""Command-line entry points.

Two modes:

* ``python -m src.run``          -- interactive chat loop over stdin/stdout.
* ``python -m src.run --demo``   -- scripted walkthrough of the two required
  test scenarios, with interrupts resumed automatically.

Both modes share the same compiled graph, so a session keeps its verified
customer id and conversation history for as long as the process lives.
"""

from __future__ import annotations

import argparse
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.config import ConfigurationError
from src.graph import get_graph


def _print_new_ai_messages(result: dict, prior_message_count: int) -> None:
    """Print only the AI messages appended by the latest turn.

    ``graph.invoke()`` returns the whole accumulated thread, so slicing at the
    previous message count avoids re-printing earlier answers.
    """
    for message in result.get("messages", [])[prior_message_count:]:
        if (
            isinstance(message, AIMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            print(f"\nAssistant: {message.content}\n")


def _interrupt_question(result: dict) -> str:
    """Extract the question text from a paused graph result."""
    return result["__interrupt__"][0].value.get("question", "Please provide more info:")


def _handle_interrupts(result: dict, thread_id: str) -> dict:
    """Prompt for each pending interrupt and resume until the turn completes."""
    graph, _ = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    while "__interrupt__" in result:
        print(f"\n[Assistant needs more info] {_interrupt_question(result)}")
        result = graph.invoke(Command(resume=input("You: ")), config=config)
    return result


def run_conversation(thread_id: str = "session-1") -> None:
    """Run an interactive conversation loop.

    Args:
        thread_id: Checkpoint thread for the session. Reusing an id resumes
            the same short-term memory.
    """
    graph, _ = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    print("Music Store Customer Support (type 'quit' to exit)")
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye!")
            return

        prior_state = graph.get_state(config)
        prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        result = _handle_interrupts(result, thread_id)
        _print_new_ai_messages(result, prior_count)


def _invoke_and_resolve(thread_id: str, user_input: str, resume_replies: list[str]) -> dict:
    """Invoke the graph and answer interrupts from a scripted list of replies."""
    graph, _ = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    replies = iter(resume_replies)
    while "__interrupt__" in result:
        print(f"[interrupt] {_interrupt_question(result)}")
        try:
            reply = next(replies)
        except StopIteration:
            print("[demo] No scripted reply left; stopping.")
            break
        print(f"[demo user reply] {reply}")
        result = graph.invoke(Command(resume=reply), config=config)
    return result


def run_demo() -> None:
    """Run the two required test scenarios end to end and print the answers."""
    graph, _ = get_graph()

    print("\n=== Test case 1: verified customer, combined music + invoice query ===")
    config = {"configurable": {"thread_id": "demo-thread-1"}}
    result = _invoke_and_resolve(
        thread_id="demo-thread-1",
        user_input=(
            "My phone number is +55 (12) 3923-5555. How much was my most "
            "recent purchase? What albums do you have by the Rolling Stones?"
        ),
        resume_replies=[],
    )
    _print_new_ai_messages(result, 0)

    print("\n=== Test case 1 (continued): long-term memory recall ===")
    prior_count = len(result.get("messages", []))
    result = graph.invoke(
        {"messages": [HumanMessage(content="List some songs that match my preferences?")]},
        config=config,
    )
    if "__interrupt__" in result:
        print(f"[interrupt] {_interrupt_question(result)}")
    else:
        _print_new_ai_messages(result, prior_count)

    print("\n=== Test case 2: new session without credentials -> identity interrupt ===")
    result = _invoke_and_resolve(
        thread_id="demo-thread-2",
        user_input=(
            "How much was my most recent purchase? What albums do you have by the Rolling Stones?"
        ),
        resume_replies=["My customer id is 1"],
    )
    _print_new_ai_messages(result, 0)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="music-store-support",
        description="Multi-agent customer support demo for a digital music store.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the scripted test scenarios instead of an interactive session",
    )
    parser.add_argument(
        "--thread",
        default="session-1",
        help="checkpoint thread id for interactive mode (default: session-1)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.demo:
            run_demo()
        else:
            run_conversation(thread_id=args.thread)
    except ConfigurationError as exc:
        print(f"\nConfiguration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
