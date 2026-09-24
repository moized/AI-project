from providers.llm import (
    GeminiDailyQuotaExhausted,
    GeminiLLMProvider,
    LLMResponse,
)


class FakeError(Exception):
    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)


class FakeInteractions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.interactions = FakeInteractions(responses)


def test_retry_delays_are_exponential(monkeypatch):
    provider = GeminiLLMProvider(api_key=None)
    provider.client = FakeClient(
        [
            FakeError(503, "temporary"),
            FakeError(503, "temporary"),
            FakeError(503, "temporary"),
            FakeError(503, "temporary"),
            FakeError(503, "temporary"),
            object(),
        ]
    )

    delays = []
    monkeypatch.setattr("providers.llm.time.sleep", delays.append)

    response = provider._create_interaction_with_retry(
        model="gemini-3.5-flash-lite",
        input="test",
        system_instruction=None,
        previous_interaction_id=None,
        tools=[],
    )

    assert response is not None
    assert delays == [2, 4, 8, 16, 32]
    assert provider.client.interactions.calls == 6


def test_retry_after_from_api_is_honored(monkeypatch):
    provider = GeminiLLMProvider(api_key=None)
    provider.client = FakeClient(
        [
            FakeError(
                429,
                "Rate limit exceeded. Please retry in 59s.",
            ),
            object(),
        ]
    )

    delays = []
    monkeypatch.setattr("providers.llm.time.sleep", delays.append)

    provider._create_interaction_with_retry(
        model="gemini-3.5-flash-lite",
        input="test",
        system_instruction=None,
        previous_interaction_id=None,
        tools=[],
    )

    assert delays == [59]


def test_daily_quota_does_not_retry_same_model(monkeypatch):
    provider = GeminiLLMProvider(api_key=None)
    provider.client = FakeClient(
        [
            FakeError(
                429,
                "Rate limit exceeded for model gemini-3.8-flash "
                "(limit: 20 requests per day on Free Tier).",
            )
        ]
    )

    delays = []
    monkeypatch.setattr("providers.llm.time.sleep", delays.append)

    try:
        provider._create_interaction_with_retry(
            model="gemini-3.8-flash",
            input="test",
            system_instruction=None,
            previous_interaction_id=None,
            tools=[],
        )
    except GeminiDailyQuotaExhausted as exc:
        assert exc.model_name == "gemini-3.8-flash"
    else:
        raise AssertionError("Expected daily quota exception.")

    assert delays == []
    assert provider.client.interactions.calls == 1


def test_model_failover_after_daily_quota():
    provider = GeminiLLMProvider(
        api_key=None,
        model_name="gemini-3.8-flash",
        fallback_models=("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
    )
    provider.client = object()

    calls = []

    def fake_run_model(*, model, prompt, tools, tool_executor, records):
        calls.append(model)
        if model == "gemini-3.8-flash":
            raise GeminiDailyQuotaExhausted(
                model,
                FakeError(
                    429,
                    "limit: 20 requests per day on Free Tier",
                ),
            )
        return LLMResponse(text=f"served by {model}", tool_calls=[])

    provider._run_model = fake_run_model

    result = provider.answer(
        prompt="test",
        tools=[],
        tool_executor=lambda *_args, **_kwargs: None,
    )

    assert result.text == "served by gemini-3.5-flash-lite"
    assert calls == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
