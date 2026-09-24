from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo

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


class GeminiDailyQuotaExhausted(RuntimeError):
    """Raised when Gemini reports a model-specific daily quota exhaustion."""

    def __init__(self, model_name: str, original: Exception) -> None:
        self.model_name = model_name
        self.original = original
        super().__init__(
            f"Gemini daily quota exhausted for model '{model_name}'."
        )


class GeminiLLMProvider:
    """Gemini Interactions API adapter with retry and model failover."""

    RETRY_DELAYS_SECONDS = (2, 4, 8, 16, 32)
    RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

    def __init__(
        self,
        api_key: str | None = settings.gemini_api_key,
        model_name: str = settings.gemini_model,
        max_tool_rounds: int = settings.max_tool_rounds,
        fallback_models: Sequence[str] = settings.gemini_fallback_models,
    ) -> None:
        self.model_name = model_name
        self.models = tuple(
            dict.fromkeys(
                model.strip()
                for model in (model_name, *fallback_models)
                if model and model.strip()
            )
        )
        self.max_tool_rounds = max_tool_rounds
        self._daily_blocked_until: dict[str, datetime] = {}

        if api_key:
            retry_options = types.HttpRetryOptions(attempts=1)
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(retry_options=retry_options),
            )
        else:
            self.client = None

    @property
    def configured(self) -> bool:
        return self.client is not None

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_daily_quota_error(cls, exc: Exception) -> bool:
        if cls._status_code(exc) != 429:
            return False

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "requests per day",
                "per day",
                "daily quota",
                "daily limit",
                "rpd",
            )
        )

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        return cls._status_code(exc) in cls.RETRYABLE_STATUS_CODES

    @classmethod
    def _is_model_specific_error(cls, exc: Exception) -> bool:
        code = cls._status_code(exc)
        if code == 404:
            return True

        if code != 400:
            return False

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "model not found",
                "model is not supported",
                "not supported for this method",
                "unknown model",
            )
        )

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        message = str(exc).lower()

        patterns = (
            r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s",
            r"retry[-_ ]after\s*[:=]?\s*(\d+(?:\.\d+)?)\s*s?",
            r'"retrydelay"\s*:\s*"?(\d+(?:\.\d+)?)s"?',
        )

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return float(match.group(1))

        return None

    @staticmethod
    def _next_pacific_reset() -> datetime:
        pacific = ZoneInfo("America/Los_Angeles")
        now_pacific = datetime.now(timezone.utc).astimezone(pacific)
        next_date = now_pacific.date() + timedelta(days=1)
        reset = datetime.combine(
            next_date,
            datetime_time.min,
            tzinfo=pacific,
        )
        return reset.astimezone(timezone.utc)

    def _is_model_available(self, model: str) -> bool:
        blocked_until = self._daily_blocked_until.get(model)
        if blocked_until is None:
            return True

        if datetime.now(timezone.utc) >= blocked_until:
            self._daily_blocked_until.pop(model, None)
            return True

        return False

    def _block_daily_quota(self, model: str) -> None:
        reset_at = self._next_pacific_reset()
        self._daily_blocked_until[model] = reset_at
        logger.warning(
            "Gemini model=%s hit daily quota; skipping until %s.",
            model,
            reset_at.isoformat(),
        )

    def _create_interaction_with_retry(
        self,
        *,
        model: str,
        input: Any,
        system_instruction: str | None,
        previous_interaction_id: str | None,
        tools: Sequence[dict[str, Any]],
    ) -> Any:
        if not self.client:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured; LLM generation is unavailable."
            )

        request = {
            "model": model,
            "input": input,
            "tools": list(tools),
        }
        if system_instruction is not None:
            request["system_instruction"] = system_instruction
        if previous_interaction_id is not None:
            request["previous_interaction_id"] = previous_interaction_id

        for retry_index in range(len(self.RETRY_DELAYS_SECONDS) + 1):
            try:
                return self.client.interactions.create(**request)
            except Exception as exc:
                if self._is_daily_quota_error(exc):
                    raise GeminiDailyQuotaExhausted(model, exc) from exc

                if not self._is_retryable(exc):
                    raise

                if retry_index >= len(self.RETRY_DELAYS_SECONDS):
                    raise

                configured_delay = self.RETRY_DELAYS_SECONDS[retry_index]
                server_delay = self._retry_after_seconds(exc)
                delay = max(configured_delay, server_delay or 0.0)

                logger.warning(
                    "Gemini transient failure model=%s status=%s; "
                    "retry=%d/%d in %.1fs.",
                    model,
                    self._status_code(exc),
                    retry_index + 1,
                    len(self.RETRY_DELAYS_SECONDS),
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Gemini request retry loop exited unexpectedly.")

    @staticmethod
    def _resume_prompt(
        prompt: str,
        records: Sequence[ToolCall],
    ) -> str:
        if not records:
            return prompt

        results = "\n".join(
            (
                f"Tool {record.name} with arguments "
                f"{json.dumps(record.arguments, ensure_ascii=False, default=str)} "
                f"returned: "
                f"{json.dumps(record.result, ensure_ascii=False, default=str)}"
            )
            for record in records
        )

        return (
            f"{prompt}\n\n"
            "A previous model already executed the following application tools. "
            "Use these results as trusted application evidence, and do not repeat "
            "a tool call unless genuinely necessary:\n"
            f"{results}\n\n"
            "Continue the task and produce the final answer."
        )

    def _run_model(
        self,
        *,
        model: str,
        prompt: str,
        tools: Sequence[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], Any],
        records: list[ToolCall],
    ) -> LLMResponse:
        system_instruction = (
            "You are a careful AI research assistant. "
            "Use retrieved evidence when relevant, do not invent sources, "
            "and use tools only when necessary."
        )

        interaction = self._create_interaction_with_retry(
            model=model,
            input=prompt,
            system_instruction=system_instruction,
            previous_interaction_id=None,
            tools=tools,
        )

        for _ in range(self.max_tool_rounds + 1):
            function_calls = [
                step for step in interaction.steps if step.type == "function_call"
            ]

            if not function_calls:
                text = (interaction.output_text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty final response.")
                return LLMResponse(text=text, tool_calls=list(records))

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

            interaction = self._create_interaction_with_retry(
                model=model,
                input=function_results,
                system_instruction=None,
                previous_interaction_id=interaction.id,
                tools=tools,
            )

        raise RuntimeError("Agent exceeded the configured tool-call round limit.")

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

        records: list[ToolCall] = []
        last_exception: Exception | None = None
        current_prompt = prompt

        for model in self.models:
            if not self._is_model_available(model):
                continue

            try:
                logger.info("Trying Gemini model=%s.", model)
                return self._run_model(
                    model=model,
                    prompt=current_prompt,
                    tools=tools,
                    tool_executor=tool_executor,
                    records=records,
                )
            except GeminiDailyQuotaExhausted as exc:
                last_exception = exc
                self._block_daily_quota(model)
                current_prompt = self._resume_prompt(prompt, records)
                logger.warning(
                    "Failing over from model=%s after daily quota exhaustion.",
                    model,
                )
                continue
            except Exception as exc:
                last_exception = exc

                if self._is_retryable(exc) or self._is_model_specific_error(exc):
                    logger.warning(
                        "Failing over from model=%s after error status=%s.",
                        model,
                        self._status_code(exc),
                    )
                    current_prompt = self._resume_prompt(prompt, records)
                    continue

                raise

        if last_exception is not None:
            raise last_exception

        raise RuntimeError("No configured Gemini model is currently available.")


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
