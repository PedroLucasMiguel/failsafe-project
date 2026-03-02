"""
ReviewerAgent — LLM-powered agent that reviews code changes made by the CodingAgent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from failsafe.llm import invoke_cached


@dataclass
class ReviewResult:
    """Result of a code review."""
    is_approved: bool
    feedback: str
    suggestions: list[str]


REVIEWER_SYSTEM_PROMPT = """\
You are an expert senior software engineer and code reviewer.
Your task is to review the code changes made by a coding agent to ensure they meet high quality standards,
correctly address the user's task, and follow best practices.

Review the provided task and the modified files.
Provide constructive feedback. If there are issues, be specific about what needs to be fixed.
If the changes are good and address the task, approve them.

Your output must be a valid JSON object matching the following schema:
{
  "is_approved": boolean,
  "comments": "string",
  "suggestions": ["string", ...]
}
"""

class ReviewerAgent:
    """Agent that reviews code changes."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def review(self, task: str, files_modified: dict[str, str], codebase_path: Path) -> ReviewResult:
        """Review the changes made for a task.

        Args:
            task: The original task description.
            files_modified: A mapping of relative file paths to their descriptions.
            codebase_path: Root path of the codebase to read the actual file contents.

        Returns:
            ReviewResult with approval status and feedback.
        """
        # Prepare the context for the reviewer
        file_contents = []
        for rel_path in files_modified:
            abs_path = codebase_path / rel_path
            if abs_path.exists():
                try:
                    content = abs_path.read_text(encoding="utf-8", errors="replace")
                    file_contents.append(f"### File: {rel_path}\n```python\n{content}\n```")
                except Exception as e:
                    file_contents.append(f"### File: {rel_path}\n(Error reading file: {e})")

        context = "\n\n".join(file_contents)

        messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"## Task\n{task}\n\n"
                f"## Modified Files\n{context}\n\n"
                "Please review these changes."
            )),
        ]

        content = invoke_cached(self._llm, messages)
        
        # Parse the response
        import json
        try:
            # Try to find JSON in the response if it's not pure JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            
            data = json.loads(content)
            return ReviewResult(
                is_approved=data.get("is_approved", False),
                feedback=data.get("comments", ""),
                suggestions=data.get("suggestions", [])
            )
        except Exception as e:
            return ReviewResult(
                is_approved=False,
                feedback=f"Error parsing reviewer response: {e}\nRaw response: {content}",
                suggestions=[]
            )
