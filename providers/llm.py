from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from google import genai

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    def answer(
        self,
        prompt: str,
        tools: Sequence[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
    ) -> LLMResponse:
        ...


class GeminiLLMProvider:
    """Gemini Interactions API adapter with bounded, application-controlled tools."""

    def __init__(
        self,
        api_key: str | None = settings.gemini_api_key,
        model_name: str = settings.gemini_model,
        max_tool_rounds: int = settings.max_tool_rounds,
    ) -> None:
        self.model_name = model_name
        self.max_tool_rounds = max_tool_rounds
        self.client = genai.Client(api_key=api_key) if api_key else None

    @property
    def configured(self) -> bool:
        return self.client is not None

    def answer(
        self,
        prompt: str,
        tools: Sequence[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
    ) -> LLMResponse:
        if not self.client:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured; LLM generation is unavailable."
            )

        system_instruction = (
            "You are a careful AI research assistant. "
            "Use retrieved evidence when relevant, do not invent sources, "
            "and use tools only when necessary."
        )

        records: list[ToolCall] = []
        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            system_instruction=system_instruction,
            tools=list(tools),
        )

        for _ in range(self.max_tool_rounds + 1):
            function_calls = [
                step for step in interaction.steps if step.type == "function_call"
            ]

            if not function_calls:
                text = (interaction.output_text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty final response.")
                return LLMResponse(text=text, tool_calls=records)

            function_results = []

            for call in function_calls:
                arguments = dict(call.arguments or {})

                try:
                    result = tool_executor(call.name, arguments)
                except Exception as exc:
                    logger.exception("Tool '%s' failed.", call.name)
                    result = {
                        "ok": False,
                        "error": f"Tool execution failed: {type(exc).__name__}",
                    }

                records.append(
                    ToolCall(
                        name=call.name,
                        arguments=arguments,
                        result=result,
                    )
                )

                function_results.append(
                    {
                        "type": "function_result",
                        "name": call.name,
                        "call_id": call.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    result,
                                    ensure_ascii=False,
                                    default=str,
                                ),
                            }
                        ],
                    }
                )

            interaction = self.client.interactions.create(
                previous_interaction_id=interaction.id,
                input=function_results,
                tools=list(tools),
            )

        raise RuntimeError("Agent exceeded the configured tool-call round limit.")


class FakeLLMProvider:
    """Offline provider for tests."""

    def __init__(self, text: str = "Test response.") -> None:
        self.text = text

    def answer(
        self,
        prompt: str,
        tools: Sequence[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
    ) -> LLMResponse:
        return LLMResponse(text=self.text)
