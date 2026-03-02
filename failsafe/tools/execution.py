"""
Execution tools — running shell commands and creating temporary scripts.

Commands are run with a timeout to prevent hangs.
Temporary scripts are created in the system temp directory and auto-cleaned.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMAND_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 8_000  # Keep command outputs token-efficient


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def run_command(command: str, cwd: str | None = None) -> str:
    """Execute a shell command and return its output.

    Args:
        command: The shell command to run.
        cwd: Working directory. Defaults to the current directory.

    Returns:
        Combined stdout + stderr output (possibly truncated), or an error.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            cwd=cwd,
            check=False,
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        output = output.strip()

        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n\n[truncated]"

        if not output:
            return f"[ok] Command completed with exit code {result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {COMMAND_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return f"[error] Failed to run command: {exc}"


@tool
def write_temp_script(content: str, extension: str = ".py") -> str:
    """Create a temporary script file and return its path.

    Useful for agents that need to create ad-hoc tools or analysis scripts.

    Args:
        content: The script source code.
        extension: File extension (e.g. '.py', '.sh', '.js').

    Returns:
        Absolute path to the created temporary script.
    """
    extension = extension if extension.startswith(".") else f".{extension}"

    try:
        fd, path = tempfile.mkstemp(suffix=extension, prefix="failsafe_")
        os.close(fd)
        Path(path).write_text(content, encoding="utf-8")
        return path
    except OSError as exc:
        return f"[error] Could not create temp script: {exc}"
