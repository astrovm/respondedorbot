"""Structural types shared by provider SDK adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ToolFunctionLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> str: ...


class ToolCallLike(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def function(self) -> ToolFunctionLike: ...


class AssistantMessageLike(Protocol):
    @property
    def content(self) -> object: ...


@dataclass(frozen=True, slots=True)
class ToolFunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    function: ToolFunctionCall


@dataclass(frozen=True, slots=True)
class EmptyAssistantMessage:
    content: str = ""
