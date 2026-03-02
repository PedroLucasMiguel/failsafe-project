"""
Knowledge base models — Pydantic schemas for structured codebase understanding.

These models are what the Analyzer agent produces and what downstream agents
consume. They are JSON-serializable for persistence.

Schema overview:
  KnowledgeBase
  ├── project_name / summary / tech_stack / architecture_notes
  ├── subsystems[]         — logical groupings of related files
  ├── coding_patterns[]    — recurring conventions WITH real code snippets
  │   └── code_snippets[]  — concrete examples extracted from source files
  ├── proprietary_tech[]   — custom frameworks / internal tools
  └── file_analyses[]      — per-file: purpose, key_elements, dependencies
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileAnalysis(BaseModel):
    """Summary of a single source file."""

    path: str = Field(description="Relative path from the codebase root.")
    purpose: str = Field(description="One-line purpose of this file.")
    key_elements: list[str] = Field(
        default_factory=list,
        description="Important classes, functions, or exports.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Internal or external modules this file depends on.",
    )


class Subsystem(BaseModel):
    """A logical grouping of related files that form a cohesive unit."""

    name: str = Field(description="Short name for this subsystem.")
    description: str = Field(description="What this subsystem does.")
    files: list[str] = Field(
        default_factory=list,
        description="Relative paths of files belonging to this subsystem.",
    )


class CodeSnippet(BaseModel):
    """A concrete code example illustrating a pattern or convention.

    This is the key structure that lets coding agents understand *how* to
    write code consistently with the existing codebase — not just *where*
    a pattern is used, but an actual runnable/reusable example of it.
    """

    file: str = Field(description="Source file this snippet came from.")
    label: str = Field(
        description="What this snippet demonstrates (e.g. 'creating a new tool')."
    )
    language: str = Field(
        default="",
        description="Programming language (python, typescript, go, etc.).",
    )
    code: str = Field(
        description="The actual code snippet — a real excerpt from the codebase."
    )


class CodingPattern(BaseModel):
    """A recurring pattern or convention detected in the codebase.

    Patterns carry real code snippets so that coding agents can:
      1. Recognize when a task matches a known pattern.
      2. Generate new code that is consistent with the established style.
      3. Avoid reinventing things that already exist in the codebase.
    """

    name: str = Field(
        description="Pattern name (e.g. 'LangChain Tool definition').")
    description: str = Field(description="How and why this pattern is used.")
    when_to_use: str = Field(
        default="",
        description="Guidance on when a coding agent should apply this pattern.",
    )
    code_snippets: list[CodeSnippet] = Field(
        default_factory=list,
        description="Concrete code examples extracted from the real codebase.",
    )
    # Keep examples for backwards compatibility with file-path references
    examples: list[str] = Field(
        default_factory=list,
        description="File paths that illustrate this pattern (legacy, prefer code_snippets).",
    )


class ProprietaryTech(BaseModel):
    """Custom/proprietary technology or framework the codebase uses."""

    name: str = Field(description="Name of the technology or framework.")
    description: str = Field(description="What it does and how it is used.")
    related_files: list[str] = Field(
        default_factory=list,
        description="Files where this tech is defined or heavily used.",
    )


class KnowledgeBase(BaseModel):
    """Top-level container for all codebase knowledge."""

    project_name: str = Field(default="", description="Project name.")
    summary: str = Field(default="", description="High-level project summary.")
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Languages, frameworks, and major libraries.",
    )
    subsystems: list[Subsystem] = Field(default_factory=list)
    coding_patterns: list[CodingPattern] = Field(default_factory=list)
    proprietary_tech: list[ProprietaryTech] = Field(default_factory=list)
    file_analyses: list[FileAnalysis] = Field(default_factory=list)
    architecture_notes: str = Field(
        default="",
        description="Free-form notes on the overall architecture.",
    )
