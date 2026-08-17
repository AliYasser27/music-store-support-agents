"""Interactive CLI loop and non-interactive demo runner for the graph."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.graph import graph


def _print_new_ai_messages(result: dict, prior_message_count: int) -> None:
    """Print only the AI messages added since ``prior_message_count``.

    ``graph.invoke()`` always returns the FULL accumulated state, including
    every past message in the thread -- not just what changed this turn. If
    we printed every AIMessage in ``result["messages"]`` on every call, old
    answers would visibly repeat after each new input. This only looks at
    messages appended after the turn started.
    """
    new_messages = result.get("messages", [])[prior_message_count:]
    for m in new_messages:
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            print(f"\nAssistant: {m.content}\n")


def _handle_interrupts(result: dict, thread_id: str) -> dict:
    """Print interrupt prompts and resume the graph with stdin input until done.

    Args:
        result: The most recent graph.invoke()/graph result (may contain an
            "__interrupt__" key when the graph has paused).
        thread_id: The thread id used for this conversation.

    Returns:
        The final result dict once no interrupt remains.
    """
    config = {"configurable": {"thread_id": thread_id}}
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        question = interrupt_obj.value.get("question", "Please provide more info:")
        print(f"\n[Assistant needs more info] {question}")
        user_reply = input("You: ")
        result = graph.invoke(Command(resume=user_reply), config=config)
    return result


def run_conversation(thread_id: str) -> None:
    """Run an interactive stdin/stdout conversation loop against the graph.

    Args:
        thread_id: A unique identifier for this conversation's checkpoint
            thread. Reusing the same thread_id resumes the same session
            (short-term memory, including customer_id, persists).
    """
    config = {"configurable": {"thread_id": thread_id}}
    print("Music Store Customer Support (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        prior_state = graph.get_state(config)
        prior_message_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        result = _handle_interrupts(result, thread_id)
        _print_new_ai_messages(result, prior_message_count)


def _invoke_and_resolve(thread_id: str, user_input: str, resume_replies: list[str]) -> dict:
    """Helper for the non-interactive demo: invoke and auto-resume interrupts.

    Args:
        thread_id: Checkpoint thread id.
        user_input: The initial human message text.
        resume_replies: Canned replies to feed to successive interrupts, in
            order, standing in for a real user during the scripted demo.

    Returns:
        The final graph result dict.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    reply_iter = iter(resume_replies)
    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        question = interrupt_obj.value.get("question", "Please provide more info:")
        print(f"[interrupt] {question}")
        try:
            reply = next(reply_iter)
        except StopIteration:
            print("[demo] No more scripted replies available; stopping.")
            break
        print(f"[demo user reply] {reply}")
        result = graph.invoke(Command(resume=reply), config=config)
    return result


def run_demo() -> None:
    """Run the two required test-case scenarios end-to-end and print results."""
    print("\n=== Test Case 1: verified customer, combined music+invoice query ===")
    config1 = {"configurable": {"thread_id": "demo-thread-1"}}
    prior_count = 0
    result = _invoke_and_resolve(
        thread_id="demo-thread-1",
        user_input=(
            "My phone number is +55 (12) 3923-5555. How much was my most "
            "recent purchase? What albums do you have by the Rolling Stones?"
        ),
        resume_replies=[],
    )
    _print_new_ai_messages(result, prior_count)

    print("\n=== Test Case 1 (continued): long-term memory recall ===")
    prior_count = len(result.get("messages", []))
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="List some songs that match my preferences?")]},
        config=config1,
    )
    if "__interrupt__" in result2:
        question = result2["__interrupt__"][0].value.get("question")
        print(f"[interrupt] {question} (no scripted reply provided; stopping)")
    else:
        _print_new_ai_messages(result2, prior_count)

    print("\n=== Test Case 2: new session, no credentials -> identity interrupt ===")
    config2 = {"configurable": {"thread_id": "demo-thread-2"}}
    result3 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "How much was my most recent purchase? What albums "
                        "do you have by the Rolling Stones?"
                    )
                )
            ]
        },
        config=config2,
    )
    if "__interrupt__" in result3:
        question = result3["__interrupt__"][0].value.get("question")
        print(f"[interrupt raised, as expected] {question}")
    else:
        print("[unexpected] No interrupt was raised for an unverified customer.")


if __name__ == "__main__":
    run_demo()
