"""
Interview agent — asks the user about their codebase before exploration.

Keeps the conversation short and token-efficient: asks a few targeted
questions, accepts "skip" gracefully, and produces a concise user_context.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from failsafe.state.discovery import DiscoveryState
from failsafe.tools.interaction import ask_user
from failsafe.llm import invoke_cached


# ---------------------------------------------------------------------------
# System prompt — deliberately concise to save tokens
# ---------------------------------------------------------------------------
INTERVIEW_SYSTEM_PROMPT = """\
You are a codebase interview agent. Your job is to gather context from the \
user about their project BEFORE the codebase is explored.

Rules:
- Ask at most 3 targeted questions, one at a time.
- Accept "[skipped]" answers gracefully — do not repeat or push.
- After gathering info (or after 3 questions), produce a FINAL SUMMARY of \
  what you learned. Prefix it with "USER_CONTEXT:".
- Be concise. Do not explain yourself or add filler.

Suggested questions (pick the most relevant):
1. What does this project do? (purpose, domain)
2. What is the tech stack? (languages, frameworks, databases)
3. Any conventions, patterns, or proprietary tech I should know about?
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------
def interview_node(state: DiscoveryState) -> dict:
    """Run the interview phase.

    This node does NOT use the LLM to drive conversation — it asks the user
    directly via the ask_user tool. This avoids burning tokens on multi-turn
    LLM chat for simple Q&A. The LLM is only used once at the end to
    synthesize the answers into a clean user_context string.
    """
    from failsafe.llm import get_llm

    llm = get_llm()

    questions = [
        "What does this project do? (purpose, domain — or press Enter to skip)",
        "What is the tech stack? (languages, frameworks, databases — or press Enter to skip)",
        "Any conventions, patterns, or proprietary tech I should know about? (or press Enter to skip)",
    ]

    answers: list[str] = []

    for question in questions:
        answer = ask_user.invoke({"question": question})
        if answer and answer != "[skipped]":
            answers.append(f"Q: {question}\nA: {answer}")

    # If user skipped everything, that's fine.
    if not answers:
        return {
            "user_context": "No context provided by the user.",
            "current_phase": "interview_done",
        }

    # Use the LLM once to synthesize a concise context summary.
    qa_block = "\n\n".join(answers)
    synthesis_prompt = (
        f"The user answered these questions about their codebase:\n\n"
        f"{qa_block}\n\n"
        f"Produce a concise summary (max 200 words) capturing the key info. "
        f"Start with 'USER_CONTEXT:'."
    )

    context = invoke_cached(llm, [
        SystemMessage(
            content="You summarize user answers into concise context."),
        HumanMessage(content=synthesis_prompt),
    ])

    # Strip the prefix if the model added it
    if context.startswith("USER_CONTEXT:"):
        context = context[len("USER_CONTEXT:"):].strip()

    return {
        "user_context": context,
        "current_phase": "interview_done",
    }
