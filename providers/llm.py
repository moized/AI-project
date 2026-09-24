from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from google import genai
from google.genai import types

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
    """Controlled Gemini function-calling adapter.

    Automatic function execution is disabled so the application remains the
    authority over which Python functions may execute.
    """

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

        config = types.GenerateContentConfig(
            tools=list(tools),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        records: list[ToolCall] = []

        for _ in range(self.max_tool_rounds + 1):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                raise RuntimeError("Gemini returned no usable candidate content.")

            function_calls = list(response.function_calls or [])
            if not function_calls:
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty final response.")
                return LLMResponse(text=text, tool_calls=records)

            contents.append(candidate.content)
            function_response_parts: list[types.Part] = []

            for call in function_calls:
                arguments = dict(call.args or {})

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

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                        id=getattr(call, "id", None),
                    )
                )

            contents.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
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
